//! Reading weights out of a single-file checkpoint and LoRA files, memory-mapped.
//!
//! A `Source` hands a model its weights by the model's own names, cast to the dtype ComfyUI
//! keeps them in, with every LoRA folded in the way ComfyUI's server patches them (DynamicVRAM):
//! in the dtype the model computes in, `(strength * alpha/rank) * (up @ down)` added per LoRA in
//! order, each step rounded to that dtype. `Patching` asks for the classic ModelPatcher's fold
//! instead, and for hook LoRAs over it.

use anyhow::{bail, Context, Result};
use candle_core::safetensors::MmapedSafetensors;
use candle_core::{DType, Device, Tensor};
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::rc::Rc;

pub struct File {
    st: MmapedSafetensors,
    pub path: PathBuf,
    names: HashSet<String>,
    used: RefCell<HashSet<String>>,
}

impl File {
    pub fn open(path: &Path) -> Result<Rc<File>> {
        let st = unsafe { MmapedSafetensors::new(path) }.with_context(|| format!("open {}", path.display()))?;
        let names = st.tensors().into_iter().map(|(n, _)| n).collect();
        Ok(Rc::new(File { st, path: path.to_path_buf(), names, used: RefCell::new(HashSet::new()) }))
    }

    pub fn has(&self, name: &str) -> bool {
        self.names.contains(name)
    }

    pub fn load(&self, name: &str, dev: &Device) -> Result<Tensor> {
        self.st.load(name, dev).with_context(|| format!("{}: {name}", self.path.display()))
    }

    /// LoRA modules this file has that nothing asked for.
    pub fn unused_modules(&self) -> Vec<String> {
        let used = self.used.borrow();
        let mut out: Vec<String> = self
            .names
            .iter()
            .filter_map(|n| n.strip_suffix(".lora_down.weight"))
            .filter(|m| !used.contains(*m))
            .map(str::to_string)
            .collect();
        out.sort();
        out
    }
}

/// One CreateHookLora: a LoRA with its model and clip strengths. `id` is ComfyUI's hook_ref, which
/// a combine compares hooks by (clones keep it).
#[derive(Clone)]
pub struct WeightHook {
    pub id: u64,
    pub lora: PathBuf,
    pub strength_model: f64,
    /// what a hooked CLIP would patch its text encoders with, which ComfyUI never applies to an
    /// encode (Engine::encode)
    #[allow(dead_code)]
    pub strength_clip: f64,
}

/// ComfyUI's HookGroup (comfy/hooks.py): weight hooks in the order they were added
#[derive(Clone, Default)]
pub struct HookGroup {
    pub hooks: Vec<WeightHook>,
}

impl HookGroup {
    /// A new hook id, as each CreateHookLora run makes a new hook_ref
    pub fn next_id() -> u64 {
        use std::sync::atomic::{AtomicU64, Ordering};
        static NEXT: AtomicU64 = AtomicU64::new(1);
        NEXT.fetch_add(1, Ordering::Relaxed)
    }

    /// clone_and_combine: this group's hooks, then each of `other`'s it does not hold yet
    pub fn combined(&self, other: &HookGroup) -> HookGroup {
        let mut c = self.clone();
        for h in &other.hooks {
            if !c.hooks.iter().any(|x| x.id == h.id) {
                c.hooks.push(h.clone());
            }
        }
        c
    }

    /// The model-side patches it makes: each hook's LoRA with its model strength, in order
    /// (a hook at strength 0 adds nothing to a weight)
    pub fn model_loras(&self) -> Vec<(PathBuf, f64)> {
        self.hooks.iter().filter(|h| h.strength_model != 0.0).map(|h| (h.lora.clone(), h.strength_model)).collect()
    }
}

/// Where a model's weight lives in the file.
pub enum Mapped {
    Key(String),
    /// rows [start, start+len) of a fused tensor (OpenCLIP's in_proj)
    Rows(String, usize, usize),
}

pub struct Source<'a> {
    pub file: &'a File,
    pub map: Box<dyn Fn(&str) -> Mapped + 'a>,
    pub loras: Vec<(Rc<File>, f64)>,
    pub lora_prefix: &'static str,
    /// The dtype the model computes in, which ComfyUI casts a layer to before it patches it: f16
    /// for SDXL's UNet, f32 for the text encoders (their f16 weights cast up)
    pub compute: DType,
    pub dtype: DType,
    pub dev: &'a Device,
    pub patching: Patching<'a>,
}

/// How a load patches its weights beyond DynamicVRAM's fold of `loras` (the default: none of this)
#[derive(Default)]
pub struct Patching<'a> {
    /// Fold the LoRAs as the classic ModelPatcher does, which a sampler swaps to when any
    /// conditioning has hooks (CFGGuider.inner_set_conds' non-dynamic delegate, a second copy of
    /// the model loaded from the file): up and down cast to f32, their mm in f32, the product scaled
    /// in f32 and rounded once to the weight's dtype, then added in it (patch_weight_to_device with
    /// lora_compute_dtype, f16 on a CUDA card, and calculate_weight's f32 intermediate)
    pub classic: bool,
    /// Hook LoRAs (CreateHookLora) with their model strengths, in their HookGroup's order, patched
    /// over the folded weight as ModelPatcher.patch_hook_weight_to_device does: the weight cast to
    /// f32, each hook's scaled f32 product added in f32, the sum rounded back to the weight's dtype
    pub hooks: Vec<(Rc<File>, f64)>,
    /// The folded weights a hooked load starts from, by name (a `record` of the hookless load): a
    /// weight no hook touches is that same tensor, so it takes no memory twice
    pub base: Option<&'a RefCell<HashMap<String, Tensor>>>,
    /// Where every weight this load hands out is kept by name, for a later `base`
    pub record: Option<&'a RefCell<HashMap<String, Tensor>>>,
}

impl Source<'_> {
    fn raw(&self, name: &str) -> Result<Option<Tensor>> {
        match (self.map)(name) {
            Mapped::Key(k) => {
                if !self.file.has(&k) {
                    return Ok(None);
                }
                Ok(Some(self.file.load(&k, self.dev)?))
            }
            Mapped::Rows(k, start, len) => {
                if !self.file.has(&k) {
                    return Ok(None);
                }
                Ok(Some(self.file.load(&k, self.dev)?.narrow(0, start, len)?.contiguous()?))
            }
        }
    }

    pub fn weight_opt(&self, name: &str) -> Result<Option<Tensor>> {
        if let Some(base) = self.patching.base {
            let Some(w) = base.borrow().get(name).cloned() else { return Ok(None) };
            return Ok(Some(match name.strip_suffix(".weight") {
                Some(stem) => self.patch_hooks(w, stem)?,
                None => w,
            }));
        }
        let Some(t) = self.raw(name)? else { return Ok(None) };
        let t = t.to_dtype(self.dtype)?;
        let t = match name.strip_suffix(".weight") {
            Some(stem) if !self.loras.is_empty() => self.patch(t, stem)?,
            _ => t,
        };
        if let Some(r) = self.patching.record {
            r.borrow_mut().insert(name.to_string(), t.clone());
        }
        Ok(Some(t))
    }

    pub fn weight(&self, name: &str) -> Result<Tensor> {
        match self.weight_opt(name)? {
            Some(t) => Ok(t),
            None => bail!("{}: no weight for {name}", self.file.path.display()),
        }
    }

    /// `w` with the LoRAs patched in as the weight `stem` (its name without ".weight"), for a
    /// weight the model holds in another layout than the file (CLIP-G's text projection)
    pub fn patch_as(&self, w: Tensor, stem: &str) -> Result<Tensor> {
        if self.loras.is_empty() {
            return Ok(w);
        }
        self.patch(w, stem)
    }

    /// Every LoRA module name ComfyUI's key map takes for this weight, in the order it adds them
    /// (comfy/lora.py model_lora_keys_unet: the model's own name, then the diffusers ones); when a
    /// file has more than one, load_lora keeps the last.
    fn lora_modules(&self, stem: &str) -> Vec<String> {
        let mut names = vec![format!("{}{}", self.lora_prefix, stem.replace('.', "_"))];
        if self.lora_prefix == "lora_unet_" {
            names.extend(sdxl_diffusers_stems(stem).iter().map(|d| format!("lora_unet_{}", d.replace('.', "_"))));
        }
        names
    }

    /// A LoRA module's `up @ down` with both cast to `dt` (the mm on torch's GEMM for it), shaped
    /// as the weight, and its alpha / rank
    fn lora_product(&self, lora: &File, module: &str, dt: DType, shape: &candle_core::Shape) -> Result<(Tensor, f64)> {
        if lora.has(&format!("{module}.lora_mid.weight")) || lora.has(&format!("{module}.dora_scale")) {
            bail!("{}: {module} is a LoCon-mid or DoRA module, not read yet", lora.path.display());
        }
        let up = lora.load(&format!("{module}.lora_up.weight"), self.dev)?.to_dtype(dt)?;
        let down = lora.load(&format!("{module}.lora_down.weight"), self.dev)?.to_dtype(dt)?;
        let rank = down.dim(0)?;
        let alpha = if lora.has(&format!("{module}.alpha")) {
            let a = lora.load(&format!("{module}.alpha"), &Device::Cpu)?.to_dtype(DType::F32)?;
            a.flatten_all()?.to_vec1::<f32>()?[0] as f64 / rank as f64
        } else {
            1.0
        };
        let diff = crate::kernels::torch_mm(&up.flatten_from(1)?, &down.flatten_from(1)?)?.reshape(shape)?;
        Ok((diff, alpha))
    }

    fn patch(&self, w: Tensor, stem: &str) -> Result<Tensor> {
        let names = self.lora_modules(stem);
        let ct = self.compute;
        let mut acc: Option<Tensor> = None;
        for (lora, strength) in &self.loras {
            let Some(module) = names.iter().rev().find(|m| lora.has(&format!("{m}.lora_down.weight"))) else {
                continue;
            };
            let diff = if self.patching.classic {
                // the classic ModelPatcher: calculate_weight's f32 intermediate for up, down and
                // `(strength * alpha) * lora_diff`, rounded once to the weight's dtype (.type())
                let (diff, alpha) = self.lora_product(lora, module, DType::F32, w.shape())?;
                (diff * (strength * alpha))?.to_dtype(ct)?
            } else {
                // ComfyUI's DynamicVRAM loader patches a layer when it first casts it (comfy/ops.py
                // resolve_cast_module_with_vbar): LowVramPatch runs calculate_weight with the cast
                // weight's dtype as the intermediate, so up and down go to the compute dtype, their
                // mm runs on torch's GEMM in it, and `weight += ((strength * alpha) * lora_diff)`
                // rounds the product and the sum to it, each through f32 as torch's elementwise
                // kernels do
                let (diff, alpha) = self.lora_product(lora, module, ct, w.shape())?;
                let scale = strength * alpha;
                if scale != 1.0 {
                    (diff.to_dtype(DType::F32)? * scale)?.to_dtype(ct)?
                } else {
                    diff
                }
            };
            let base = match acc.take() {
                Some(a) => a,
                None => w.to_dtype(ct)?,
            };
            acc = Some((base.to_dtype(DType::F32)? + diff.to_dtype(DType::F32)?)?.to_dtype(ct)?);
            lora.used.borrow_mut().insert(module.clone());
        }
        // a patched weight stays in the compute dtype: that is what the layer's first use computes
        // with (the text encoders round it to f16 afterwards, clip::ClipText::settle)
        Ok(acc.unwrap_or(w))
    }

    /// The folded weight `w` with the hook LoRAs patched over it (patch_hook_weight_to_device):
    /// cast to f32, each hook's `(strength * alpha) * (up @ down)` (all f32) added in f32, the sum
    /// rounded once back to `w`'s dtype (stochastic_rounding is a plain cast for f16)
    fn patch_hooks(&self, w: Tensor, stem: &str) -> Result<Tensor> {
        let names = self.lora_modules(stem);
        let mut acc: Option<Tensor> = None;
        for (lora, strength) in &self.patching.hooks {
            let Some(module) = names.iter().rev().find(|m| lora.has(&format!("{m}.lora_down.weight"))) else {
                continue;
            };
            let (diff, alpha) = self.lora_product(lora, module, DType::F32, w.shape())?;
            let base = match acc.take() {
                Some(a) => a,
                None => w.to_dtype(DType::F32)?,
            };
            acc = Some((base + (diff * (strength * alpha))?)?);
            lora.used.borrow_mut().insert(module.clone());
        }
        Ok(match acc {
            Some(a) => a.to_dtype(w.dtype())?,
            None => w,
        })
    }
}

const UNET_MAP_ATTENTIONS: [&str; 6] = ["proj_in.weight", "proj_in.bias", "proj_out.weight", "proj_out.bias", "norm.weight", "norm.bias"];
const TRANSFORMER_BLOCKS: [&str; 20] = [
    "norm1.weight",
    "norm1.bias",
    "norm2.weight",
    "norm2.bias",
    "norm3.weight",
    "norm3.bias",
    "attn1.to_q.weight",
    "attn1.to_k.weight",
    "attn1.to_v.weight",
    "attn1.to_out.0.weight",
    "attn1.to_out.0.bias",
    "attn2.to_q.weight",
    "attn2.to_k.weight",
    "attn2.to_v.weight",
    "attn2.to_out.0.weight",
    "attn2.to_out.0.bias",
    "ff.net.0.proj.weight",
    "ff.net.0.proj.bias",
    "ff.net.2.weight",
    "ff.net.2.bias",
];
/// (the model's name, diffusers' name) for a ResBlock's weights
const UNET_MAP_RESNET: [(&str, &str); 12] = [
    ("in_layers.2.weight", "conv1.weight"),
    ("in_layers.2.bias", "conv1.bias"),
    ("emb_layers.1.weight", "time_emb_proj.weight"),
    ("emb_layers.1.bias", "time_emb_proj.bias"),
    ("out_layers.3.weight", "conv2.weight"),
    ("out_layers.3.bias", "conv2.bias"),
    ("skip_connection.weight", "conv_shortcut.weight"),
    ("skip_connection.bias", "conv_shortcut.bias"),
    ("in_layers.0.weight", "norm1.weight"),
    ("in_layers.0.bias", "norm1.bias"),
    ("out_layers.0.weight", "norm2.weight"),
    ("out_layers.0.bias", "norm2.bias"),
];
const UNET_MAP_BASIC: [(&str, &str); 18] = [
    ("label_emb.0.0.weight", "class_embedding.linear_1.weight"),
    ("label_emb.0.0.bias", "class_embedding.linear_1.bias"),
    ("label_emb.0.2.weight", "class_embedding.linear_2.weight"),
    ("label_emb.0.2.bias", "class_embedding.linear_2.bias"),
    ("label_emb.0.0.weight", "add_embedding.linear_1.weight"),
    ("label_emb.0.0.bias", "add_embedding.linear_1.bias"),
    ("label_emb.0.2.weight", "add_embedding.linear_2.weight"),
    ("label_emb.0.2.bias", "add_embedding.linear_2.bias"),
    ("input_blocks.0.0.weight", "conv_in.weight"),
    ("input_blocks.0.0.bias", "conv_in.bias"),
    ("out.0.weight", "conv_norm_out.weight"),
    ("out.0.bias", "conv_norm_out.bias"),
    ("out.2.weight", "conv_out.weight"),
    ("out.2.bias", "conv_out.bias"),
    ("time_embed.0.weight", "time_embedding.linear_1.weight"),
    ("time_embed.0.bias", "time_embedding.linear_1.bias"),
    ("time_embed.2.weight", "time_embedding.linear_2.weight"),
    ("time_embed.2.bias", "time_embedding.linear_2.bias"),
];

/// comfy.utils.unet_to_diffusers: (diffusers' name, the model's name) for every weight of a UNet
/// with these res blocks per level and transformer depths (the output ones taken from the end)
fn unet_to_diffusers(num_res_blocks: &[usize], depth: &[usize], depth_output: &[usize], depth_middle: usize) -> Vec<(String, String)> {
    let mut depth: std::collections::VecDeque<usize> = depth.iter().copied().collect();
    let mut depth_output = depth_output.to_vec();
    let mut map = vec![];
    let attention = |map: &mut Vec<(String, String)>, d: &str, m: &str, n: usize| {
        for b in UNET_MAP_ATTENTIONS {
            map.push((format!("{d}.{b}"), format!("{m}.{b}")));
        }
        for t in 0..n {
            for b in TRANSFORMER_BLOCKS {
                map.push((format!("{d}.transformer_blocks.{t}.{b}"), format!("{m}.transformer_blocks.{t}.{b}")));
            }
        }
    };
    for (x, &blocks) in num_res_blocks.iter().enumerate() {
        let mut n = 1 + (blocks + 1) * x;
        for i in 0..blocks {
            for (m, d) in UNET_MAP_RESNET {
                map.push((format!("down_blocks.{x}.resnets.{i}.{d}"), format!("input_blocks.{n}.0.{m}")));
            }
            let t = depth.pop_front().unwrap_or(0);
            if t > 0 {
                attention(&mut map, &format!("down_blocks.{x}.attentions.{i}"), &format!("input_blocks.{n}.1"), t);
            }
            n += 1;
        }
        for k in ["weight", "bias"] {
            map.push((format!("down_blocks.{x}.downsamplers.0.conv.{k}"), format!("input_blocks.{n}.0.op.{k}")));
        }
    }
    attention(&mut map, "mid_block.attentions.0", "middle_block.1", depth_middle);
    for (i, n) in [0, 2].into_iter().enumerate() {
        for (m, d) in UNET_MAP_RESNET {
            map.push((format!("mid_block.resnets.{i}.{d}"), format!("middle_block.{n}.{m}")));
        }
    }
    for (x, &blocks) in num_res_blocks.iter().rev().enumerate() {
        let mut n = (blocks + 1) * x;
        let l = blocks + 1;
        for i in 0..l {
            let mut c = 0;
            for (m, d) in UNET_MAP_RESNET {
                map.push((format!("up_blocks.{x}.resnets.{i}.{d}"), format!("output_blocks.{n}.0.{m}")));
            }
            c += 1;
            let t = depth_output.pop().unwrap_or(0);
            if t > 0 {
                c += 1;
                attention(&mut map, &format!("up_blocks.{x}.attentions.{i}"), &format!("output_blocks.{n}.1"), t);
            }
            if i == l - 1 {
                for k in ["weight", "bias"] {
                    map.push((format!("up_blocks.{x}.upsamplers.0.conv.{k}"), format!("output_blocks.{n}.{c}.conv.{k}")));
                }
            }
            n += 1;
        }
    }
    for (m, d) in UNET_MAP_BASIC {
        map.push((d.to_string(), m.to_string()));
    }
    map
}

/// The diffusers names (without ".weight") of SDXL's UNet weight `stem`, the names kohya LoRAs
/// trained on a diffusers checkpoint use (lora_unet_down_blocks_1_attentions_0_...)
fn sdxl_diffusers_stems(stem: &str) -> &'static [String] {
    static MAP: std::sync::OnceLock<std::collections::HashMap<String, Vec<String>>> = std::sync::OnceLock::new();
    let map = MAP.get_or_init(|| {
        let mut m: std::collections::HashMap<String, Vec<String>> = std::collections::HashMap::new();
        for (d, ldm) in unet_to_diffusers(&[2, 2, 2], &[0, 0, 2, 2, 10, 10], &[0, 0, 0, 2, 2, 2, 10, 10, 10], 10) {
            if let (Some(d), Some(ldm)) = (d.strip_suffix(".weight"), ldm.strip_suffix(".weight")) {
                m.entry(ldm.to_string()).or_default().push(d.to_string());
            }
        }
        m
    });
    map.get(stem).map(Vec::as_slice).unwrap_or(&[])
}

/// SDXL's UNet weights live under model.diffusion_model.
pub fn unet_map() -> Box<dyn Fn(&str) -> Mapped> {
    Box::new(|n| Mapped::Key(format!("model.diffusion_model.{n}")))
}

/// An SDXL ControlNet in diffusers' layout (xinsir's Union ProMax), by the names ComfyUI's
/// load_controlnet_state_dict gives its weights: the encoder half through unet_to_diffusers, the
/// hint encoder (controlnet_cond_embedding) as input_hint_block's even indices, the zero convs
/// (controlnet_down_blocks, controlnet_mid_block); anything else (the union parts) as the file names it
pub fn controlnet_map() -> Box<dyn Fn(&str) -> Mapped> {
    let mut map: std::collections::HashMap<String, String> = std::collections::HashMap::new();
    for (d, m) in unet_to_diffusers(&[2, 2, 2], &[0, 0, 2, 2, 10, 10], &[0, 0, 0, 2, 2, 2, 10, 10, 10], 10) {
        // label_emb is add_embedding in an SDXL file, not class_embedding
        if !d.starts_with("class_embedding") {
            map.insert(m, d);
        }
    }
    Box::new(move |n: &str| {
        if let Some(d) = map.get(n) {
            return Mapped::Key(d.clone());
        }
        let (stem, suffix) = n.rsplit_once('.').unwrap_or((n, ""));
        if let Some(i) = stem.strip_prefix("input_hint_block.").and_then(|i| i.parse::<usize>().ok()) {
            let d = match i {
                0 => "controlnet_cond_embedding.conv_in".to_string(),
                14 => "controlnet_cond_embedding.conv_out".to_string(),
                _ => format!("controlnet_cond_embedding.blocks.{}", i / 2 - 1),
            };
            return Mapped::Key(format!("{d}.{suffix}"));
        }
        if let Some(i) = stem.strip_prefix("zero_convs.").and_then(|r| r.strip_suffix(".0")) {
            return Mapped::Key(format!("controlnet_down_blocks.{i}.{suffix}"));
        }
        if stem == "middle_block_out.0" {
            return Mapped::Key(format!("controlnet_mid_block.{suffix}"));
        }
        Mapped::Key(n.to_string())
    })
}

pub fn vae_map() -> Box<dyn Fn(&str) -> Mapped> {
    Box::new(|n| Mapped::Key(format!("first_stage_model.{n}")))
}

pub fn clip_l_map() -> Box<dyn Fn(&str) -> Mapped> {
    Box::new(|n| Mapped::Key(format!("conditioner.embedders.0.transformer.{n}")))
}

/// OpenCLIP bigG, presented under transformers' names (what kohya's lora_te2_ keys use).
pub fn clip_g_map() -> Box<dyn Fn(&str) -> Mapped> {
    Box::new(|n| {
        let p = "conditioner.embedders.1.model";
        let parts: Vec<&str> = n.split('.').collect();
        match parts.as_slice() {
            ["text_model", "embeddings", "token_embedding", "weight"] => Mapped::Key(format!("{p}.token_embedding.weight")),
            ["text_model", "embeddings", "position_embedding", "weight"] => Mapped::Key(format!("{p}.positional_embedding")),
            ["text_model", "final_layer_norm", wb] => Mapped::Key(format!("{p}.ln_final.{wb}")),
            ["text_projection"] => Mapped::Key(format!("{p}.text_projection")),
            ["text_model", "encoder", "layers", i, rest @ ..] => {
                let r = format!("{p}.transformer.resblocks.{i}");
                match rest {
                    ["layer_norm1", wb] => Mapped::Key(format!("{r}.ln_1.{wb}")),
                    ["layer_norm2", wb] => Mapped::Key(format!("{r}.ln_2.{wb}")),
                    ["mlp", "fc1", wb] => Mapped::Key(format!("{r}.mlp.c_fc.{wb}")),
                    ["mlp", "fc2", wb] => Mapped::Key(format!("{r}.mlp.c_proj.{wb}")),
                    ["self_attn", "out_proj", wb] => Mapped::Key(format!("{r}.attn.out_proj.{wb}")),
                    ["self_attn", qkv, wb] => {
                        let which = match *qkv {
                            "q_proj" => 0,
                            "k_proj" => 1,
                            _ => 2,
                        };
                        let fused = if *wb == "weight" { "in_proj_weight" } else { "in_proj_bias" };
                        Mapped::Rows(format!("{r}.attn.{fused}"), which * 1280, 1280)
                    }
                    _ => Mapped::Key(format!("{r}.{}", rest.join("."))),
                }
            }
            _ => Mapped::Key(format!("{p}.{n}")),
        }
    })
}
