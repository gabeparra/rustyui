//! SDXL's UNet in its original (sgm) layout, shapes read from the checkpoint, forward pass as
//! ComfyUI's openaimodel.py: f16 throughout, timestep and size embeddings computed in f32.

use crate::model::Source;
use crate::nn::{attention, nearest_to, Conv2d, GroupNorm, LayerNorm, Linear};
use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};

pub(crate) struct ResBlock {
    in_norm: GroupNorm,
    in_conv: Conv2d,
    emb: Linear,
    out_norm: GroupNorm,
    out_conv: Conv2d,
    skip: Option<Conv2d>,
}

impl ResBlock {
    fn forward(&self, x: &Tensor, emb: &Tensor) -> Result<Tensor> {
        let h = self.in_conv.forward(&self.in_norm.forward_silu(x)?)?;
        let e = self.emb.forward(&crate::kernels::torch_silu(emb)?)?.to_dtype(h.dtype())?.unsqueeze(2)?.unsqueeze(3)?;
        let h = h.broadcast_add(&e)?;
        let h = self.out_conv.forward(&self.out_norm.forward_silu(&h)?)?;
        let skip = match &self.skip {
            Some(c) => c.forward(x)?,
            None => x.clone(),
        };
        Ok((skip + h)?)
    }
}

struct Attn {
    q: Linear,
    k: Linear,
    v: Linear,
    out: Linear,
    heads: usize,
}

impl Attn {
    fn forward(&self, x: &Tensor, ctx: &Tensor) -> Result<Tensor> {
        let a = attention(&self.q.forward(x)?, &self.k.forward(ctx)?, &self.v.forward(ctx)?, self.heads, None)?;
        Ok(self.out.forward(&a)?)
    }

    /// AttentionCouple's patch in place of this cross-attention: q from the block's input, then
    /// for each chunk of the batch its side's prompts, the chunk's q repeated once for each, each
    /// prompt's k and v (to_k and to_v of its own context) repeated over the chunk, one attention
    /// call; with several prompts each output times its prompt's mask at this resolution, summed
    /// over the prompts (in f32, rounded to f16 once, as torch's sum of f16 does); then to_out.
    /// `latent` is the UNet input's (h, w) (extra_options' original_shape)
    fn forward_couple(&self, x: &Tensor, cc: &CoupleCall, latent: (usize, usize)) -> Result<Tensor> {
        let q = self.q.forward(x)?;
        let (bt, n, inner) = q.dims3()?;
        let b = bt / cc.sides.len();
        let mut outs = vec![];
        for (i, &side) in cc.sides.iter().enumerate() {
            let s = &cc.couple.sides[side];
            let len = s.ctx.dim(0)?;
            let (k, v) = (self.k.forward(&s.ctx)?, self.v.forward(&s.ctx)?);
            let qi = q.narrow(0, i * b, b)?;
            let qt = if len > 1 { qi.repeat((len, 1, 1))? } else { qi };
            let each = |t: &Tensor| -> Result<Tensor> {
                let parts = (0..len).map(|j| t.narrow(0, j, 1)?.repeat((b, 1, 1))).collect::<candle_core::Result<Vec<_>>>()?;
                Ok(Tensor::cat(&parts, 0)?)
            };
            let a = attention(&qt, &each(&k)?, &each(&v)?, self.heads, None)?;
            outs.push(match s.masks_for(n, latent, x.device())? {
                // one prompt: its mask is ones and the sum is over one output
                None => a,
                Some(m) => {
                    let a = a.reshape((len, b, n, inner))?.broadcast_mul(&m.reshape((len, 1, n, 1))?)?;
                    let mut acc = a.get(0)?.to_dtype(DType::F32)?;
                    for j in 1..len {
                        acc = (acc + a.get(j)?.to_dtype(DType::F32)?)?;
                    }
                    acc.to_dtype(DType::F16)?
                }
            });
        }
        Ok(self.out.forward(&Tensor::cat(&outs, 0)?)?)
    }
}

/// ComfyUI-TiledAttentionCouple's AttentionCouple patch: it replaces the cross-attention of every
/// transformer block in input blocks 4, 5, 7 and 8 and output blocks 0 to 5. (Its middle block's
/// patches are keyed ("middle", 8, i), the input loop's last id, where ComfyUI looks up ("middle",
/// 0, i), so the middle block keeps its own cross-attention.) For each side, 0 the positive and 1
/// the negative: its prompts' contexts (n, T, 2048) f16 and, with more than one prompt, their masks
/// normalised to sum to one at the picture's size, (n, H, W) values of f16 held as f32
pub struct Couple {
    pub sides: [CoupleSide; 2],
}

pub struct CoupleSide {
    pub ctx: Tensor,
    pub masks: Option<(Vec<f32>, usize, usize)>,
    /// the masks at each attention's resolution, by its token count
    at: std::cell::RefCell<std::collections::HashMap<usize, Tensor>>,
}

impl CoupleSide {
    pub fn new(ctx: Tensor, masks: Option<(Vec<f32>, usize, usize)>) -> Self {
        CoupleSide { ctx, masks, at: Default::default() }
    }

    /// get_masks_from_q: the masks at the resolution whose token count matches (the latent's, a
    /// half, a quarter, else an eighth), F.interpolate's nearest on the card (source index
    /// floor(dst * (in / out)) in f32), as (n, tokens) f16
    fn masks_for(&self, tokens: usize, latent: (usize, usize), dev: &Device) -> Result<Option<Tensor>> {
        let Some((m, mh, mw)) = &self.masks else { return Ok(None) };
        if let Some(t) = self.at.borrow().get(&tokens) {
            return Ok(Some(t.clone()));
        }
        let (h, w) = latent;
        let ds = if h * w == tokens {
            1
        } else if (h / 2) * (w / 2) == tokens {
            2
        } else if (h / 4) * (w / 4) == tokens {
            4
        } else {
            8
        };
        let (oh, ow) = (h / ds, w / ds);
        anyhow::ensure!(oh * ow == tokens, "TiledAttentionCouple: {tokens} tokens at a {w}x{h} latent");
        let (sy, sx) = (*mh as f32 / oh as f32, *mw as f32 / ow as f32);
        let n = m.len() / (mh * mw);
        let mut v = Vec::with_capacity(n * tokens);
        for j in 0..n {
            for y in 0..oh {
                let iy = if oh == *mh { y } else { ((y as f32 * sy).floor() as usize).min(mh - 1) };
                for x in 0..ow {
                    let ix = if ow == *mw { x } else { ((x as f32 * sx).floor() as usize).min(mw - 1) };
                    v.push(m[(j * mh + iy) * mw + ix]);
                }
            }
        }
        let t = Tensor::from_vec(v, (n, tokens), dev)?.to_dtype(DType::F16)?;
        self.at.borrow_mut().insert(tokens, t.clone());
        Ok(Some(t))
    }
}

/// One UNet call's use of a Couple: the side of each chunk of its batch (cond_or_uncond)
pub struct CoupleCall<'a> {
    pub couple: &'a Couple,
    pub sides: Vec<usize>,
}

struct Block {
    norm1: LayerNorm,
    attn1: Attn,
    norm2: LayerNorm,
    attn2: Attn,
    norm3: LayerNorm,
    ff_in: Linear,
    ff_out: Linear,
}

impl Block {
    fn forward(&self, x: &Tensor, ctx: &Tensor, couple: Option<(&CoupleCall, (usize, usize))>) -> Result<Tensor> {
        let n = self.norm1.forward(x)?;
        let x = (self.attn1.forward(&n, &n)? + x)?;
        let n = self.norm2.forward(&x)?;
        let a2 = match couple {
            Some((cc, latent)) => self.attn2.forward_couple(&n, cc, latent)?,
            None => self.attn2.forward(&n, ctx)?,
        };
        let x = (a2 + &x)?;
        // GEGLU: value * gelu(gate), the projection's two halves
        let h = self.ff_in.forward(&self.norm3.forward(&x)?)?;
        let h = crate::profile::time("geglu", h.device(), || {
            if h.device().is_cuda() {
                crate::kernels::geglu(&h)
            } else {
                let half = h.dim(2)? / 2;
                h.narrow(2, 0, half)? * h.narrow(2, half, half)?.gelu_erf()?
            }
        })?;
        Ok((&x + self.ff_out.forward(&h)?)?)
    }
}

pub(crate) struct Transformer {
    norm: GroupNorm,
    proj_in: Linear,
    blocks: Vec<Block>,
    proj_out: Linear,
}

impl Transformer {
    fn forward(&self, x: &Tensor, ctx: &Tensor, couple: Option<(&CoupleCall, (usize, usize))>) -> Result<Tensor> {
        let (b, c, h, w) = x.dims4()?;
        let mut t = self.norm.forward(x)?.reshape((b, c, h * w))?.transpose(1, 2)?.contiguous()?;
        t = self.proj_in.forward(&t)?;
        for blk in &self.blocks {
            t = blk.forward(&t, ctx, couple)?;
        }
        t = self.proj_out.forward(&t)?;
        let t = t.transpose(1, 2)?.reshape((b, c, h, w))?;
        Ok((t + x)?)
    }
}

pub(crate) enum Layer {
    Conv(Conv2d),
    Res(ResBlock),
    Attn(Transformer),
    Down(Conv2d),
    Up(Conv2d),
}

/// A ControlNet's residuals for one UNet call (ComfyUI's control dict): one for each input
/// block's output, added to the skips as the output blocks take them, and one for the middle block
pub struct Control {
    pub output: Vec<Tensor>,
    pub middle: Tensor,
}

pub struct Unet {
    time1: Linear,
    time2: Linear,
    label1: Linear,
    label2: Linear,
    input: Vec<Vec<Layer>>,
    middle: Vec<Layer>,
    output: Vec<Vec<Layer>>,
    out_norm: GroupNorm,
    out_conv: Conv2d,
    pub model_channels: usize,
}

pub(crate) fn lin(s: &Source, p: &str) -> Result<Linear> {
    Ok(Linear { w: s.weight(&format!("{p}.weight"))?, b: s.weight_opt(&format!("{p}.bias"))? })
}

pub(crate) fn conv(s: &Source, p: &str, stride: usize) -> Result<Conv2d> {
    let w = s.weight(&format!("{p}.weight"))?;
    let k = w.dim(2)?;
    Ok(Conv2d { w, b: s.weight_opt(&format!("{p}.bias"))?, stride, padding: k / 2 })
}

fn gn(s: &Source, p: &str, eps: f64) -> Result<GroupNorm> {
    Ok(GroupNorm { w: s.weight(&format!("{p}.weight"))?, b: s.weight(&format!("{p}.bias"))?, groups: 32, eps })
}

pub(crate) fn ln(s: &Source, p: &str) -> Result<LayerNorm> {
    Ok(LayerNorm { w: s.weight(&format!("{p}.weight"))?, b: s.weight(&format!("{p}.bias"))?, eps: 1e-5 })
}

fn has(s: &Source, name: &str) -> bool {
    match (s.map)(name) {
        crate::model::Mapped::Key(k) | crate::model::Mapped::Rows(k, _, _) => s.file.has(&k),
    }
}

fn resblock(s: &Source, p: &str) -> Result<ResBlock> {
    Ok(ResBlock {
        in_norm: gn(s, &format!("{p}.in_layers.0"), 1e-5)?,
        in_conv: conv(s, &format!("{p}.in_layers.2"), 1)?,
        emb: lin(s, &format!("{p}.emb_layers.1"))?,
        out_norm: gn(s, &format!("{p}.out_layers.0"), 1e-5)?,
        out_conv: conv(s, &format!("{p}.out_layers.3"), 1)?,
        skip: if has(s, &format!("{p}.skip_connection.weight")) { Some(conv(s, &format!("{p}.skip_connection"), 1)?) } else { None },
    })
}

fn attn(s: &Source, p: &str, heads: usize) -> Result<Attn> {
    Ok(Attn {
        q: lin(s, &format!("{p}.to_q"))?,
        k: lin(s, &format!("{p}.to_k"))?,
        v: lin(s, &format!("{p}.to_v"))?,
        out: lin(s, &format!("{p}.to_out.0"))?,
        heads,
    })
}

fn transformer(s: &Source, p: &str) -> Result<Transformer> {
    let proj_in = lin(s, &format!("{p}.proj_in"))?;
    let channels = proj_in.w.dim(0)?;
    let heads = channels / 64; // SDXL: num_head_channels 64
    let mut blocks = vec![];
    let mut i = 0;
    while has(s, &format!("{p}.transformer_blocks.{i}.norm1.weight")) {
        let b = format!("{p}.transformer_blocks.{i}");
        blocks.push(Block {
            norm1: ln(s, &format!("{b}.norm1"))?,
            attn1: attn(s, &format!("{b}.attn1"), heads)?,
            norm2: ln(s, &format!("{b}.norm2"))?,
            attn2: attn(s, &format!("{b}.attn2"), heads)?,
            norm3: ln(s, &format!("{b}.norm3"))?,
            ff_in: lin(s, &format!("{b}.ff.net.0.proj"))?,
            ff_out: lin(s, &format!("{b}.ff.net.2"))?,
        });
        i += 1;
    }
    Ok(Transformer { norm: gn(s, &format!("{p}.norm"), 1e-6)?, proj_in, blocks, proj_out: lin(s, &format!("{p}.proj_out"))? })
}

pub(crate) fn layers_at(s: &Source, p: &str) -> Result<Vec<Layer>> {
    let mut out = vec![];
    for j in 0..3 {
        let q = format!("{p}.{j}");
        if has(s, &format!("{q}.in_layers.0.weight")) {
            out.push(Layer::Res(resblock(s, &q)?));
        } else if has(s, &format!("{q}.norm.weight")) && has(s, &format!("{q}.proj_in.weight")) {
            out.push(Layer::Attn(transformer(s, &q)?));
        } else if has(s, &format!("{q}.op.weight")) {
            out.push(Layer::Down(conv(s, &format!("{q}.op"), 2)?));
        } else if has(s, &format!("{q}.conv.weight")) {
            out.push(Layer::Up(conv(s, &format!("{q}.conv"), 1)?));
        } else if has(s, &format!("{q}.weight")) {
            out.push(Layer::Conv(conv(s, &q, 1)?));
        }
    }
    Ok(out)
}

/// Sinusoidal embedding, cos half first (timestep_embedding in ComfyUI), in f32.
pub fn timestep_embedding(t: &[f32], dim: usize) -> Vec<f32> {
    // as torch computes it on the CPU (the SDXL size embedding is a CPU tensor): -ln(max_period)
    // rounded to f32 times arange, over half, then exp, cos and sin each one MKL VML call over the
    // whole row (at::vml), where libm's would part by an ulp at some sizes (1864 and 1264 do)
    let half = dim / 2;
    let ln = -(10000f64.ln()) as f32;
    let scaled: Vec<f32> = (0..half).map(|i| ln * i as f32 / half as f32).collect();
    let freqs = crate::torchcpu::exp_f32(&scaled);
    let mut out = Vec::with_capacity(t.len() * dim);
    for &tv in t {
        let args: Vec<f32> = freqs.iter().map(|f| tv * f).collect();
        out.extend(crate::torchcpu::cos_f32(&args));
        out.extend(crate::torchcpu::sin_f32(&args));
    }
    out
}

impl Unet {
    pub fn load(s: &Source) -> Result<Self> {
        let mut input = vec![];
        let mut i = 0;
        loop {
            let l = layers_at(s, &format!("input_blocks.{i}"))?;
            if l.is_empty() {
                break;
            }
            input.push(l);
            i += 1;
        }
        let mut output = vec![];
        let mut i = 0;
        loop {
            let l = layers_at(s, &format!("output_blocks.{i}"))?;
            if l.is_empty() {
                break;
            }
            output.push(l);
            i += 1;
        }
        let time1 = lin(s, "time_embed.0")?;
        let model_channels = time1.w.dim(1)?;
        Ok(Unet {
            time1,
            time2: lin(s, "time_embed.2")?,
            label1: lin(s, "label_emb.0.0")?,
            label2: lin(s, "label_emb.0.2")?,
            input,
            middle: layers_at(s, "middle_block")?,
            output,
            out_norm: gn(s, "out.0", 1e-5)?,
            out_conv: conv(s, "out.2", 1)?,
            model_channels,
        })
    }

    pub(crate) fn run(layers: &[Layer], h: Tensor, emb: &Tensor, ctx: &Tensor, out_shape: Option<(usize, usize)>) -> Result<Tensor> {
        Self::run_couple(layers, h, emb, ctx, out_shape, None)
    }

    fn run_couple(
        layers: &[Layer],
        mut h: Tensor,
        emb: &Tensor,
        ctx: &Tensor,
        out_shape: Option<(usize, usize)>,
        couple: Option<(&CoupleCall, (usize, usize))>,
    ) -> Result<Tensor> {
        for l in layers {
            h = match l {
                Layer::Conv(c) | Layer::Down(c) => c.forward(&h)?,
                Layer::Res(r) => r.forward(&h, emb)?,
                Layer::Attn(t) => t.forward(&h, ctx, couple)?,
                Layer::Up(c) => {
                    let (_, _, hh, ww) = h.dims4()?;
                    let (oh, ow) = out_shape.unwrap_or((hh * 2, ww * 2));
                    c.forward(&nearest_to(&h, oh, ow)?)?
                }
            };
        }
        Ok(h)
    }

    /// x: (b, 4, h, w) f16, t: discrete timesteps, ctx: (b, L, 2048) f16, y: (b, 2816) f16.
    pub fn forward(&self, x: &Tensor, t: &[f32], ctx: &Tensor, y: &Tensor, dev: &Device) -> Result<Tensor> {
        self.forward_with(x, t, ctx, y, None, dev)
    }

    /// forward with a ControlNet's residuals (apply_control: the middle block's output and each
    /// skip as the output blocks take it, in place in f16)
    pub fn forward_with(&self, x: &Tensor, t: &[f32], ctx: &Tensor, y: &Tensor, ctrl: Option<&Control>, dev: &Device) -> Result<Tensor> {
        self.forward_couple(x, t, ctx, y, ctrl, None, dev)
    }

    /// forward with a ControlNet's residuals and TiledAttentionCouple's patch
    #[allow(clippy::too_many_arguments)]
    pub fn forward_couple(
        &self,
        x: &Tensor,
        t: &[f32],
        ctx: &Tensor,
        y: &Tensor,
        ctrl: Option<&Control>,
        couple: Option<&CoupleCall>,
        dev: &Device,
    ) -> Result<Tensor> {
        Ok(crate::profile::time("unet forward (total)", dev, || {
            self.forward_inner(x, t, ctx, y, ctrl, couple, dev).map_err(|e| candle_core::Error::Msg(e.to_string()))
        })?)
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_inner(
        &self,
        x: &Tensor,
        t: &[f32],
        ctx: &Tensor,
        y: &Tensor,
        ctrl: Option<&Control>,
        couple: Option<&CoupleCall>,
        dev: &Device,
    ) -> Result<Tensor> {
        crate::profile::scope("unet");
        let dt = x.dtype();
        // comfy's timestep_embedding runs on the card (CUDA's expf, cosf, sinf), then the model's dtype
        let temb = crate::kernels::torch_timestep_embedding(t, self.model_channels, dev)?.to_dtype(dt)?;
        let emb = self.time2.forward(&crate::kernels::torch_silu(&self.time1.forward(&temb)?)?)?;
        let emb = (emb + self.label2.forward(&crate::kernels::torch_silu(&self.label1.forward(y)?)?)?)?;
        let mut hs = vec![];
        let mut h = x.clone();
        let (_, _, lh, lw) = x.dims4()?;
        // the patched blocks: input 4, 5, 7 and 8, output 0 to 5
        let patch = |on: bool| couple.filter(|_| on).map(|c| (c, (lh, lw)));
        for (id, blk) in self.input.iter().enumerate() {
            h = Self::run_couple(blk, h, &emb, ctx, None, patch([4, 5, 7, 8].contains(&id)))?;
            hs.push(h.clone());
        }
        h = Self::run(&self.middle, h, &emb, ctx, None)?;
        let mut residuals: Vec<Tensor> = vec![];
        if let Some(c) = ctrl {
            h = (h + &c.middle)?;
            residuals = c.output.clone();
        }
        for (id, blk) in self.output.iter().enumerate() {
            let mut skip = hs.pop().context("unet: skip stack empty")?;
            // control['output'] is popped from its end, one per output block
            if let Some(r) = residuals.pop() {
                skip = (skip + r)?;
            }
            h = Tensor::cat(&[&h, &skip], 1)?;
            let out_shape = hs.last().map(|s| (s.dim(2).unwrap(), s.dim(3).unwrap()));
            h = Self::run_couple(blk, h, &emb, ctx, out_shape, patch(id < 6))?;
        }
        let h = self.out_conv.forward(&self.out_norm.forward_silu(&h)?)?;
        Ok(h.to_dtype(DType::F32)?)
    }
}

/// debug-unet CHECKPOINT REF (refs/unet_ref.py): the first UNet call of the README graph,
/// rustyui's embeddings against ComfyUI's, then every block on ComfyUI's input for it, then the
/// first ResBlock's and the first transformer block's ops
pub fn debug_unet(ckpt: &std::path::Path, refs: &std::path::Path, dev: &Device) -> Result<()> {
    let file = crate::model::File::open(ckpt)?;
    let src = Source {
        file: &file,
        map: crate::model::unet_map(),
        loras: vec![],
        lora_prefix: "lora_unet_",
        compute: DType::F16,
        dtype: DType::F16,
        dev,
        patching: Default::default(),
    };
    let unet = Unet::load(&src)?;
    let r = unsafe { candle_core::safetensors::MmapedSafetensors::new(refs)? };
    let get = |n: &str| -> Result<Tensor> { r.load(n, dev).with_context(|| format!("no {n} in the capture")) };
    let get16 = |n: &str| -> Result<Tensor> { Ok(get(n)?.to_dtype(DType::F16)?) };
    let report = |name: &str, ours: &Tensor, theirs: &Tensor| -> Result<()> {
        let ours = ours.to_dtype(DType::F32)?.reshape(theirs.dims())?;
        let d = (&ours - theirs)?.abs()?;
        let differ = d.ne(0f32)?.to_dtype(DType::F32)?.sum_all()?.to_scalar::<f32>()?;
        let max = d.flatten_all()?.max(0)?.to_scalar::<f32>()?;
        println!(
            "{name:44} {differ:>9} of {:>9} differ, max |d| {max:.3e}{}",
            theirs.elem_count(),
            if differ == 0.0 { " exact" } else { "" }
        );
        Ok(())
    };
    // the embeddings
    let ts: Vec<f32> = get("unet.timesteps")?.to_vec1()?;
    let temb = crate::kernels::torch_timestep_embedding(&ts, unet.model_channels, dev)?.to_dtype(DType::F16)?;
    report("timestep embedding (sinusoid, f16)", &temb, &get("time_embed.in")?)?;
    let te = unet.time2.forward(&crate::kernels::torch_silu(&unet.time1.forward(&get16("time_embed.in")?)?)?)?;
    report("time_embed (their sinusoid)", &te, &get("time_embed")?)?;
    let le = unet.label2.forward(&crate::kernels::torch_silu(&unet.label1.forward(&get16("unet.y")?)?)?)?;
    report("label_emb (their y)", &le, &get("label_emb")?)?;
    // the size part of y (pooled first, then six 256-wide sinusoids of the sizes), as rustyui builds it
    let y = get("unet.y")?;
    let (h8, w8) = (get("unet.x")?.dim(2)? * 8, get("unet.x")?.dim(3)? * 8);
    let ours = crate::graph::adm_for_debug(&y.narrow(0, 0, 1)?.narrow(1, 0, 1280)?, w8, h8, dev)?;
    report("y (sizes, rustyui's)", &ours.narrow(1, 1280, 1536)?, &y.narrow(0, 0, 1)?.narrow(1, 1280, 1536)?)?;
    let emb = (get16("time_embed")? + get16("label_emb")?)?;
    let ctx = get16("unet.context")?;
    // the blocks, each on ComfyUI's input for it
    let shape = |n: &str| -> Result<(usize, usize)> {
        let t = get(n)?;
        Ok((t.dim(2)?, t.dim(3)?))
    };
    for (i, blk) in unet.input.iter().enumerate() {
        let out = Unet::run(blk, get16(&format!("input_blocks.{i}.in"))?, &emb, &ctx, None)?;
        report(&format!("input_blocks.{i}"), &out, &get(&format!("input_blocks.{i}"))?)?;
    }
    report("middle_block", &Unet::run(&unet.middle, get16("middle_block.in")?, &emb, &ctx, None)?, &get("middle_block")?)?;
    let n_out = unet.output.len();
    for (i, blk) in unet.output.iter().enumerate() {
        // an up-sampling block goes to the size of the next skip (input block n_out - 2 - i)
        let next = if i + 1 < n_out { Some(shape(&format!("input_blocks.{}", n_out - 2 - i))?) } else { None };
        let out = Unet::run(blk, get16(&format!("output_blocks.{i}.in"))?, &emb, &ctx, next)?;
        report(&format!("output_blocks.{i}"), &out, &get(&format!("output_blocks.{i}"))?)?;
    }
    let fin = unet.out_conv.forward(&unet.out_norm.forward_silu(&get16(&format!("output_blocks.{}", n_out - 1))?)?)?;
    report("out (norm, SiLU, conv)", &fin, &get("unet.out")?)?;
    // inside input_blocks.1.0 (a ResBlock)
    if let Some(Layer::Res(rb)) = unet.input[1].first() {
        let p = "input_blocks.1.0";
        let x = get16(&format!("{p}.in_layers.0.in"))?;
        report("res: in_layers.0 (GroupNorm)", &rb.in_norm.forward(&x)?, &get(&format!("{p}.in_layers.0"))?)?;
        report("res: in_layers (norm, SiLU, conv)", &rb.in_conv.forward(&rb.in_norm.forward_silu(&x)?)?, &get(&format!("{p}.in_layers"))?)?;
        report(
            "res: emb_layers",
            &rb.emb.forward(&crate::kernels::torch_silu(&get16(&format!("{p}.emb_layers.in"))?)?)?,
            &get(&format!("{p}.emb_layers"))?,
        )?;
        let y = get16(&format!("{p}.out_layers.0.in"))?;
        report("res: out_layers.0 (GroupNorm)", &rb.out_norm.forward(&y)?, &get(&format!("{p}.out_layers.0"))?)?;
        report(
            "res: out_layers (norm, SiLU, conv)",
            &rb.out_conv.forward(&rb.out_norm.forward_silu(&y)?)?,
            &get(&format!("{p}.out_layers"))?,
        )?;
    }
    // inside input_blocks.4.1 (a SpatialTransformer) and its first block
    if let Some(Layer::Attn(tr)) = unet.input[4].get(1) {
        let p = "input_blocks.4.1";
        let x = get16(&format!("{p}.norm.in"))?;
        report("st: norm (GroupNorm)", &tr.norm.forward(&x)?, &get(&format!("{p}.norm"))?)?;
        let (b, c, h, w) = x.dims4()?;
        let t = get16(&format!("{p}.norm"))?.reshape((b, c, h * w))?.transpose(1, 2)?.contiguous()?;
        report("st: proj_in", &tr.proj_in.forward(&t)?, &get(&format!("{p}.proj_in"))?)?;
        let blk = &tr.blocks[0];
        let q = format!("{p}.transformer_blocks.0");
        report("tb: norm1", &blk.norm1.forward(&get16(&format!("{q}.norm1.in"))?)?, &get(&format!("{q}.norm1"))?)?;
        let n1 = get16(&format!("{q}.norm1"))?;
        report("tb: attn1.to_q", &blk.attn1.q.forward(&n1)?, &get(&format!("{q}.attn1.to_q"))?)?;
        report("tb: attn1.to_k", &blk.attn1.k.forward(&n1)?, &get(&format!("{q}.attn1.to_k"))?)?;
        report("tb: attn1.to_v", &blk.attn1.v.forward(&n1)?, &get(&format!("{q}.attn1.to_v"))?)?;
        let a = attention(
            &get16(&format!("{q}.attn1.to_q"))?,
            &get16(&format!("{q}.attn1.to_k"))?,
            &get16(&format!("{q}.attn1.to_v"))?,
            blk.attn1.heads,
            None,
        )?;
        report("tb: attn1 (FlashAttention)", &a, &get(&format!("{q}.attn1.to_out.0.in"))?)?;
        report(
            "tb: attn1.to_out",
            &blk.attn1.out.forward(&get16(&format!("{q}.attn1.to_out.0.in"))?)?,
            &get(&format!("{q}.attn1.to_out.0"))?,
        )?;
        report("tb: norm2", &blk.norm2.forward(&get16(&format!("{q}.norm2.in"))?)?, &get(&format!("{q}.norm2"))?)?;
        let n2 = get16(&format!("{q}.norm2"))?;
        report("tb: attn2.to_q", &blk.attn2.q.forward(&n2)?, &get(&format!("{q}.attn2.to_q"))?)?;
        report("tb: attn2.to_k (context)", &blk.attn2.k.forward(&ctx)?, &get(&format!("{q}.attn2.to_k"))?)?;
        report("tb: attn2.to_v (context)", &blk.attn2.v.forward(&ctx)?, &get(&format!("{q}.attn2.to_v"))?)?;
        let a = attention(
            &get16(&format!("{q}.attn2.to_q"))?,
            &get16(&format!("{q}.attn2.to_k"))?,
            &get16(&format!("{q}.attn2.to_v"))?,
            blk.attn2.heads,
            None,
        )?;
        report("tb: attn2 (FlashAttention)", &a, &get(&format!("{q}.attn2.to_out.0.in"))?)?;
        report(
            "tb: attn2.to_out",
            &blk.attn2.out.forward(&get16(&format!("{q}.attn2.to_out.0.in"))?)?,
            &get(&format!("{q}.attn2.to_out.0"))?,
        )?;
        report("tb: norm3", &blk.norm3.forward(&get16(&format!("{q}.norm3.in"))?)?, &get(&format!("{q}.norm3"))?)?;
        report("tb: ff.net.0.proj", &blk.ff_in.forward(&get16(&format!("{q}.norm3"))?)?, &get(&format!("{q}.ff.net.0.proj"))?)?;
        report("tb: GEGLU", &crate::kernels::geglu(&get16(&format!("{q}.ff.net.0.proj"))?)?, &get(&format!("{q}.ff.net.0"))?)?;
        report("tb: ff.net.2", &blk.ff_out.forward(&get16(&format!("{q}.ff.net.2.in"))?)?, &get(&format!("{q}.ff.net.2"))?)?;
        report("tb: whole block (their input)", &blk.forward(&get16(&format!("{q}.in"))?, &ctx, None)?, &get(&q)?)?;
        report("st: proj_out", &tr.proj_out.forward(&get16(&format!("{p}.proj_out.in"))?)?, &get(&format!("{p}.proj_out"))?)?;
    }
    Ok(())
}
