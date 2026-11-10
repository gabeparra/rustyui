//! ComfyUI-TiledAttentionCouple (Deathspike, 1.3.0), the hub's `imagen couple` graph:
//! TiledAttentionCouplePrompt (a base prompt and one per tile, split at BREAK, the tiles shuffled
//! by Python's random when a TiledAttentionCoupleShuffle is linked) and TiledAttentionCouple, which
//! encodes every prompt under its tile's mask (modules/core/config.py) and hands the model
//! AttentionCouple's cross-attention patch (modules/attention_couple.py; unet.rs's Couple).

use super::*;
use crate::unet::{Couple, CoupleSide};

/// The nodes this file runs
pub const TAC_NODES: &[&str] = &["TiledAttentionCouple", "TiledAttentionCouplePrompt", "TiledAttentionCoupleShuffle"];

/// TiledAttentionCouplePrompt's TILED_CONFIG
pub struct TiledConfig {
    divisions: Vec<f64>,
    horizontal: bool,
    positive_base: String,
    positive_tiles: Vec<String>,
    negative_base: String,
    negative_tiles: Vec<String>,
}

/// Python's str.strip(): what str.isspace() calls whitespace
fn py_strip(s: &str) -> &str {
    s.trim_matches(|c: char| c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c))
}

/// ConfigProvider.parse_text: the base before the first BREAK, then one prompt per tile, empty
/// ones added or extra ones dropped to the number of divisions
fn parse_text(value: &str, n: usize) -> (String, Vec<String>) {
    let parts: Vec<String> = value.split("BREAK").map(|p| py_strip(p).to_string()).collect();
    let mut tiles: Vec<String> = parts[1..].to_vec();
    tiles.resize(n, String::new());
    (parts[0].clone(), tiles)
}

/// Python's sum() of floats (3.12: Neumaier's compensated sum)
fn py_fsum(v: &[f64]) -> f64 {
    let (mut s, mut c) = (0.0f64, 0.0f64);
    for &x in v {
        let t = s + x;
        c += if s.abs() >= x.abs() { (s - t) + x } else { (x - t) + s };
        s = t;
    }
    if c != 0.0 && c.is_finite() {
        s + c
    } else {
        s
    }
}

impl Engine {
    pub(super) fn tac_node(&mut self, class: &str, inp: &serde_json::Map<String, Value>) -> Result<Option<Vec<Val>>> {
        Ok(Some(match class {
            "TiledAttentionCoupleShuffle" => {
                vec![Val::Int(self.arg(inp, "seed")?.as_u64().context("TiledAttentionCoupleShuffle: seed is not a whole number")? as i64)]
            }
            "TiledAttentionCouplePrompt" => {
                let division = self.text_arg(inp, "division")?;
                let divisions = division
                    .split(',')
                    .map(|x| {
                        py_strip(x).parse::<f64>().with_context(|| format!("TiledAttentionCouplePrompt: division {x:?} is not a number"))
                    })
                    .collect::<Result<Vec<f64>>>()?;
                let horizontal = self.text_arg(inp, "orientation")? == "horizontal";
                let (positive_base, mut positive_tiles) = parse_text(&self.text_arg(inp, "positive")?, divisions.len());
                let (negative_base, mut negative_tiles) = parse_text(&self.text_arg(inp, "negative")?, divisions.len());
                if inp.contains_key("shuffle") {
                    let Val::Int(seed) = self.input(inp, "shuffle")? else {
                        bail!("TiledAttentionCouplePrompt: shuffle is not a TiledAttentionCoupleShuffle")
                    };
                    crate::rng::Mt19937::python(seed as u64).python_shuffle(&mut positive_tiles);
                    crate::rng::Mt19937::python(seed as u64).python_shuffle(&mut negative_tiles);
                }
                vec![Val::TiledConfig(Rc::new(TiledConfig {
                    divisions,
                    horizontal,
                    positive_base,
                    positive_tiles,
                    negative_base,
                    negative_tiles,
                }))]
            }
            "TiledAttentionCouple" => self.tiled_attention_couple(inp)?,
            _ => return Ok(None),
        }))
    }

    /// TiledProvider.process: Config.process over the whole picture, then AttentionCouple
    fn tiled_attention_couple(&mut self, inp: &serde_json::Map<String, Value>) -> Result<Vec<Val>> {
        let Val::Model(m) = self.input(inp, "model")? else { bail!("TiledAttentionCouple: model is not an SDXL model") };
        let Val::Clip(clip, layer) = self.input(inp, "clip")? else { bail!("TiledAttentionCouple: clip is not an SDXL CLIP") };
        let Val::TiledConfig(cfg) = self.input(inp, "config")? else {
            bail!("TiledAttentionCouple: config is not a TiledAttentionCouplePrompt")
        };
        let (w, h) = (self.int_arg(inp, "width")?, self.int_arg(inp, "height")?);
        // validate: an SDXL model's latent size, 64
        anyhow::ensure!(w % 64 == 0, "TiledAttentionCouple: Width must be divisible by 64.");
        anyhow::ensure!(h % 64 == 0, "TiledAttentionCouple: Height must be divisible by 64.");
        // Config.process: the bases under a mask of ones (the negative's first), then each tile's
        // prompts under ones over the tile (MaskComposite's add onto zeros), positive then negative
        let rect = |x1: usize, y1: usize, x2: usize, y2: usize| -> Vec<f32> {
            let mut v = vec![0f32; w * h];
            for y in y1..y2 {
                v[y * w + x1..y * w + x2].fill(1.0);
            }
            v
        };
        let mut neg = vec![self.tac_encode(&clip, layer, &cfg.negative_base, rect(0, 0, w, h), w, h)?];
        let mut pos = vec![self.tac_encode(&clip, layer, &cfg.positive_base, rect(0, 0, w, h), w, h)?];
        let mut attention = false;
        let total = py_fsum(&cfg.divisions);
        let outer = if cfg.horizontal { w } else { h } as i64;
        let mut shift = 0i64;
        for (i, div) in cfg.divisions.iter().enumerate() {
            let base = (outer as f64 * (1.0 * div / total)).floor() as i64;
            let size = if i == cfg.divisions.len() - 1 { outer - shift } else { base };
            let (x1, y1, x2, y2) = if cfg.horizontal { (shift, 0, shift + size, h as i64) } else { (0, shift, w as i64, shift + size) };
            // Tile.overlap with the picture
            let (x1, y1, x2, y2) = (x1.max(0), y1.max(0), x2.min(w as i64), y2.min(h as i64));
            if x1 < x2 && y1 < y2 {
                let mask = rect(x1 as usize, y1 as usize, x2 as usize, y2 as usize);
                if cfg.positive_tiles.get(i).is_some_and(|t| !t.is_empty()) {
                    pos.push(self.tac_encode(&clip, layer, &cfg.positive_tiles[i], mask.clone(), w, h)?);
                    attention = true;
                }
                if cfg.negative_tiles.get(i).is_some_and(|t| !t.is_empty()) {
                    neg.push(self.tac_encode(&clip, layer, &cfg.negative_tiles[i], mask, w, h)?);
                    attention = true;
                }
            }
            shift += size;
        }
        pad_conds(&mut pos)?;
        pad_conds(&mut neg)?;
        if !attention {
            // "Latent": the prompts as they are, masks and all
            return Ok(vec![Val::Model(m), Val::Cond(Rc::new(pos)), Val::Cond(Rc::new(neg))]);
        }
        let neg_side = couple_side(&mut neg, &self.dev)?;
        let pos_side = couple_side(&mut pos, &self.dev)?;
        let mut m2 = (*m).clone();
        m2.couple = Some(Rc::new(Couple { sides: [pos_side, neg_side] }));
        Ok(vec![Val::Model(Rc::new(m2)), Val::Cond(Rc::new(vec![pos[0].clone()])), Val::Cond(Rc::new(vec![neg[0].clone()]))])
    }

    /// Config.encode: CLIPTextEncode, then ConditioningSetMask (default area, strength 1)
    fn tac_encode(&mut self, clip: &Stack, layer: i64, text: &str, mask: Vec<f32>, w: usize, h: usize) -> Result<Cond> {
        let mut c = self.encode(clip, layer, text)?;
        c.props.hooks = clip.cond_hooks.clone();
        c.props.mask = Some((Tensor::from_vec(mask, (1, h, w), &self.dev)?, 1.0));
        c.props.area_to_bounds = false;
        Ok(c)
    }
}

/// Config.pad: each prompt's tokens padded with zeros to the longest of its side
fn pad_conds(conds: &mut [Cond]) -> Result<()> {
    let longest = conds.iter().map(|c| c.cond.dim(1)).collect::<candle_core::Result<Vec<_>>>()?.into_iter().max().unwrap_or(0);
    for c in conds.iter_mut() {
        let (b, t, d) = c.cond.dims3()?;
        if t < longest {
            let zeros = Tensor::zeros((b, longest - t, d), c.cond.dtype(), c.cond.device())?;
            c.cond = Tensor::cat(&[&c.cond, &zeros], 1)?;
        }
    }
    Ok(())
}

/// AttentionCouple for one side: with several prompts, their masks in f16 times their strengths,
/// each over their sum (summed in f32 and rounded to f16, the quotient rounded to f16), their
/// contexts together, and the first prompt's mask dropped (it then runs as an unmasked prompt);
/// with one, its context alone and its mask kept
fn couple_side(conds: &mut [Cond], dev: &Device) -> Result<CoupleSide> {
    let ctx = Tensor::cat(&conds.iter().map(|c| &c.cond).collect::<Vec<_>>(), 0)?.to_dtype(DType::F16)?.to_device(dev)?;
    if conds.len() == 1 {
        return Ok(CoupleSide::new(ctx, None));
    }
    let f16 = |v: f32| half::f16::from_f32(v).to_f32();
    let mut masks = vec![];
    let (mut mh, mut mw) = (0, 0);
    for c in conds.iter() {
        let (m, strength) = c.props.mask.as_ref().context("TiledAttentionCouple: a tile prompt without its mask")?;
        let (_, h, w) = m.dims3()?;
        (mh, mw) = (h, w);
        masks.push(
            m.to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?
                .into_iter()
                .map(|v| f16(f16(v) * *strength as f32))
                .collect::<Vec<f32>>(),
        );
    }
    let n = mh * mw;
    let mut norm = Vec::with_capacity(masks.len() * n);
    let sum: Vec<f32> = (0..n).map(|i| f16(masks.iter().fold(0f32, |a, m| a + m[i]))).collect();
    for m in &masks {
        norm.extend(m.iter().zip(&sum).map(|(v, s)| f16(v / s)));
    }
    conds[0].props.mask = None;
    Ok(CoupleSide::new(ctx, Some((norm, mh, mw))))
}
