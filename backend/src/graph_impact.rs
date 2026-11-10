//! ComfyUI-Impact-Pack's nodes from a mask to a detailed picture (modules/impact), as
//! tools/inpaint.py's "only masked" graph chains them: ImpactGaussianBlurMask
//! (segs_nodes.GaussianBlurMask), MaskToSEGS (segs_nodes.py, core.mask_to_segs) and
//! DetailerForEach (impact_pack.py, core.enhance_detail, impact_sampling.ksampler_wrapper), with
//! ComfyUI's GrowMask and DifferentialDiffusion (comfy_extras/nodes_mask.py,
//! nodes_differential_diffusion.py) before and inside them. The arithmetic is impact.rs's; what
//! the detailer does not support yet (wildcards, hooks, inpaint models, tiled VAE passes, the SDE
//! samplers' own noise) stops the graph with its name.

use super::*;
use crate::impact;

/// The nodes this file runs
pub const IMPACT_NODES: &[&str] = &["GrowMask", "DifferentialDiffusion", "ImpactGaussianBlurMask", "MaskToSEGS", "DetailerForEach"];

/// DetailerForEach's settings, as enhance_detail takes them
struct Detail {
    guide_size: f64,
    guide_for_bbox: bool,
    max_size: f64,
    steps: usize,
    cfg: f64,
    sampler: String,
    scheduler: String,
    denoise: f64,
    noise_mask: bool,
    force_inpaint: bool,
    cycle: u64,
    noise_mask_feather: i64,
}

impl Engine {
    pub(super) fn impact_node(&mut self, class: &str, inp: &serde_json::Map<String, Value>) -> Result<Option<Vec<Val>>> {
        Ok(Some(match class {
            "GrowMask" => {
                // each mask of (-1, h, w) on its own
                let (v, n, h, w) = self.mask_values(inp, "mask")?;
                let expand = self.arg(inp, "expand")?.as_i64().context("GrowMask: expand is not a whole number")?;
                let tapered = self.flag_arg(inp, "tapered_corners")?;
                let mut out = Vec::with_capacity(v.len());
                for i in 0..n {
                    out.extend(impact::grow_mask(&v[i * h * w..(i + 1) * h * w], h, w, expand, tapered));
                }
                vec![Val::Mask(Tensor::from_vec(out, (n, h, w), &self.dev)?)]
            }
            "DifferentialDiffusion" => {
                let Val::Model(m) = self.input(inp, "model")? else { bail!("DifferentialDiffusion: model is not an SDXL model") };
                let strength = if inp.contains_key("strength") { self.float_arg(inp, "strength")? } else { 1.0 };
                let mut m2 = (*m).clone();
                m2.differential = Some(strength);
                vec![Val::Model(Rc::new(m2))]
            }
            "ImpactGaussianBlurMask" => {
                // make_3d_mask, then tensor_gaussian_blur_mask on (b, h, w, 1)
                let (v, n, h, w) = self.mask_values(inp, "mask")?;
                let k = self.arg(inp, "kernel_size")?.as_i64().context("ImpactGaussianBlurMask: kernel_size is not a whole number")?;
                let sigma = self.float_arg(inp, "sigma")?;
                let out = impact::tensor_gaussian_blur_mask(&v, n, h, w, k, sigma as f32);
                vec![Val::Mask(Tensor::from_vec(out, (n, h, w), &self.dev)?)]
            }
            "MaskToSEGS" => {
                // make_2d_mask leaves a batch of masks a batch: their segments joined
                let (v, n, h, w) = self.mask_values(inp, "mask")?;
                let combined = self.flag_arg(inp, "combined")?;
                let crop_factor = self.float_arg(inp, "crop_factor")?;
                let bbox_fill = self.flag_arg(inp, "bbox_fill")?;
                let drop_size = self.arg(inp, "drop_size")?.as_i64().context("MaskToSEGS: drop_size is not a whole number")?;
                let contour_fill = if inp.contains_key("contour_fill") { self.flag_arg(inp, "contour_fill")? } else { false };
                vec![Val::Segs(Rc::new(impact::mask_to_segs(&v, n, h, w, combined, crop_factor, bbox_fill, drop_size, contour_fill)))]
            }
            "DetailerForEach" => vec![self.detailer_for_each(inp)?],
            _ => return Ok(None),
        }))
    }

    /// A MASK input's values on the CPU, how many masks and their (h, w)
    fn mask_values(&mut self, inp: &serde_json::Map<String, Value>, name: &str) -> Result<(Vec<f32>, usize, usize, usize)> {
        let Val::Mask(m) = self.input(inp, name)? else { bail!("{name} is not a mask") };
        anyhow::ensure!(m.rank() >= 2, "{name}: a mask of {} dims", m.rank());
        let (h, w) = (m.dim(m.rank() - 2)?, m.dim(m.rank() - 1)?);
        let v = m.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        let n = v.len() / (h * w).max(1);
        Ok((v, n, h, w))
    }

    /// DetailerForEach.do_detail: each segment cut from the picture as it stands (earlier ones
    /// pasted in), its mask feathered for the paste, enhance_detail with the seed plus its index,
    /// and the result pasted back. The picture stays on the CPU in f32, as Impact keeps it.
    fn detailer_for_each(&mut self, inp: &serde_json::Map<String, Value>) -> Result<Val> {
        let Val::Image(img) = self.input(inp, "image")? else { bail!("DetailerForEach: image is not an image") };
        let (b, c, ih, iw) = img.dims4()?;
        anyhow::ensure!(b == 1, "DetailerForEach does not allow image batches");
        anyhow::ensure!(c == 3, "DetailerForEach: a picture of {c} channels");
        let Val::Segs(segs) = self.input(inp, "segs")? else { bail!("DetailerForEach: segs is not SEGS") };
        let Val::Model(model) = self.input(inp, "model")? else { bail!("DetailerForEach: model is not an SDXL model") };
        let Val::Vae(vae) = self.input(inp, "vae")? else { bail!("DetailerForEach: vae is not a VAE") };
        let Val::Cond(pos) = self.input(inp, "positive")? else { bail!("DetailerForEach: positive is not conditioning") };
        let Val::Cond(neg) = self.input(inp, "negative")? else { bail!("DetailerForEach: negative is not conditioning") };
        let int = |e: &mut Engine, n: &str| -> Result<i64> {
            e.arg(inp, n)?.as_i64().with_context(|| format!("DetailerForEach: {n} is not a whole number"))
        };
        let d = Detail {
            guide_size: self.float_arg(inp, "guide_size")?,
            guide_for_bbox: self.flag_arg(inp, "guide_size_for")?,
            max_size: self.float_arg(inp, "max_size")?,
            steps: self.int_arg(inp, "steps")?,
            cfg: self.float_arg(inp, "cfg")?,
            sampler: self.text_arg(inp, "sampler_name")?,
            scheduler: self.text_arg(inp, "scheduler")?,
            denoise: self.float_arg(inp, "denoise")?,
            noise_mask: self.flag_arg(inp, "noise_mask")?,
            force_inpaint: self.flag_arg(inp, "force_inpaint")?,
            cycle: int(self, "cycle")?.max(0) as u64,
            // do_detail's default when a graph leaves the optional input out
            noise_mask_feather: if inp.contains_key("noise_mask_feather") { int(self, "noise_mask_feather")? } else { 0 },
        };
        let seed = self.arg(inp, "seed")?.as_u64().context("DetailerForEach: seed is not a whole number")?;
        let feather = int(self, "feather")?;
        // an empty wildcard changes nothing; any other text (even a space) re-encodes the prompt
        anyhow::ensure!(
            self.text_arg(inp, "wildcard")?.is_empty(),
            "DetailerForEach: wildcards are not implemented yet (leave the wildcard empty)"
        );
        for (name, what) in [("inpaint_model", "inpaint_model"), ("tiled_encode", "tiled_encode"), ("tiled_decode", "tiled_decode")] {
            if inp.contains_key(name) && self.flag_arg(inp, name)? {
                bail!("DetailerForEach: {what} is not implemented yet");
            }
        }
        for name in ["detailer_hook", "scheduler_func_opt"] {
            anyhow::ensure!(!inp.contains_key(name), "DetailerForEach: {name} is not implemented yet");
        }
        // impact_sampling.ksampler gives these SDE samplers a Brownian tree over the whole
        // schedule, and uni_pc reports a step's x0 its own way
        const IMPACT_SDE: [&str; 6] =
            ["dpmpp_sde", "dpmpp_sde_gpu", "dpmpp_2m_sde", "dpmpp_2m_sde_gpu", "dpmpp_3m_sde", "dpmpp_3m_sde_gpu"];
        anyhow::ensure!(
            !IMPACT_SDE.contains(&d.sampler.as_str()) && !d.sampler.starts_with("uni_pc"),
            "DetailerForEach: sampler {} is not implemented yet in the detailer",
            d.sampler
        );
        let all: Vec<&Cond> = pos.iter().chain(neg.iter()).collect();
        anyhow::ensure!(
            !all.iter().any(|c| c.props.mask.is_some()),
            "DetailerForEach: masked conditioning (crop_condition_mask) is not implemented yet"
        );
        anyhow::ensure!(
            !all.iter().any(|c| c.props.control.is_some()),
            "DetailerForEach: a ControlNet in its conditioning is not implemented yet"
        );
        // segs_scale_match changes nothing for SEGS made at the picture's size
        anyhow::ensure!(
            (segs.h == ih && segs.w == iw) || segs.h == 0 || segs.w == 0,
            "DetailerForEach: SEGS made for {}x{} on a {iw}x{ih} picture (segs_scale_match is not implemented yet)",
            segs.w,
            segs.h
        );
        // apply_differential_diffusion when the noise mask is feathered
        let model = if d.noise_mask_feather > 0 && model.differential.is_none() {
            let mut m2 = (*model).clone();
            m2.differential = Some(1.0);
            Rc::new(m2)
        } else {
            model
        };
        self.clip = None;
        self.ltx_te = None;
        self.ltx_dit = None;
        let t = Instant::now();
        let mut image = img.permute((0, 2, 3, 1))?.flatten_all()?.to_vec1::<f32>()?;
        for (i, seg) in segs.segs.iter().enumerate() {
            let [x1, y1, x2, y2] = seg.crop;
            let (cw, ch) = (x2 - x1, y2 - y1);
            anyhow::ensure!(seg.mask.len() == cw * ch && x2 <= iw && y2 <= ih, "DetailerForEach: segment {i}'s mask does not fit its crop");
            let crop = impact::crop_hwc(&image, iw, 3, seg.crop);
            let paste_mask = impact::tensor_gaussian_blur_mask(&seg.mask, 1, ch, cw, feather, 10.0);
            if seg.mask.iter().all(|&v| v == 0.0) {
                eprintln!("[engine] detailer: segment skip [empty mask]");
                continue;
            }
            let noise_mask = d.noise_mask.then_some(seg.mask.as_slice());
            let seg_seed = seed.wrapping_add(i as u64);
            if let Some(out) = self.enhance_detail(&crop, cw, ch, seg.bbox, seg_seed, noise_mask, &model, &vae, &pos, &neg, &d)? {
                impact::tensor_paste(&mut image, iw, ih, &out, cw, ch, x1, y1, &paste_mask)?;
            }
        }
        self.timings.push((format!("detailer ({} segments)", segs.segs.len()), t.elapsed().as_secs_f64()));
        Ok(Val::Image(Tensor::from_vec(image, (1, ih, iw, 3), &self.dev)?.permute((0, 3, 1, 2))?.contiguous()?))
    }

    /// core.enhance_detail on one segment's crop (h, w, 3): upscaled so the bbox's shorter side
    /// (or the crop's) reaches guide_size, the longer result side at most max_size, with Pillow's
    /// LANCZOS on bytes; encoded from pil2tensor's NHWC memory; sampled as ksampler_wrapper does
    /// (KSampler's schedule for steps / denoise from step steps / denoise - steps, the noise mask
    /// moved to the card), cycle times, each from the last run's final x0, which separated_sample
    /// returns through process_latent_out on the CPU; decoded and resized back. None when Impact
    /// skips the segment.
    #[allow(clippy::too_many_arguments)]
    fn enhance_detail(
        &mut self,
        crop: &[f32],
        w: usize,
        h: usize,
        bbox: [i64; 4],
        seed: u64,
        mask: Option<&[f32]>,
        model: &Rc<Stack>,
        vae: &Path,
        pos: &Rc<Vec<Cond>>,
        neg: &Rc<Vec<Cond>>,
        d: &Detail,
    ) -> Result<Option<Vec<f32>>> {
        let noise_mask = match mask {
            Some(m) => {
                Some(Tensor::from_vec(impact::tensor_gaussian_blur_mask(m, 1, h, w, d.noise_mask_feather, 10.0), (1, h, w), &self.dev)?)
            }
            None => None,
        };
        let (bw, bh) = (bbox[2] - bbox[0], bbox[3] - bbox[1]);
        if !d.force_inpaint && bh as f64 >= d.guide_size && bw as f64 >= d.guide_size {
            eprintln!("[engine] detailer: segment skip (enough big)");
            return Ok(None);
        }
        let short = if d.guide_for_bbox { bw.min(bh) } else { w.min(h) as i64 };
        anyhow::ensure!(short > 0, "DetailerForEach: a segment of no size");
        let mut upscale = d.guide_size / short as f64;
        let mut new_w = (w as f64 * upscale) as usize;
        let mut new_h = (h as f64 * upscale) as usize;
        if new_w as f64 > d.max_size || new_h as f64 > d.max_size {
            upscale *= d.max_size / new_w.max(new_h) as f64;
            new_w = (w as f64 * upscale) as usize;
            new_h = (h as f64 * upscale) as usize;
        }
        if upscale <= 1.0 || new_w == 0 || new_h == 0 {
            if !d.force_inpaint {
                eprintln!("[engine] detailer: segment skip [determined upscale factor={upscale}]");
                return Ok(None);
            }
            (new_w, new_h) = (w, h);
        }
        eprintln!("[engine] detailer: segment upscale for ({bw}, {bh}) | crop region ({w}, {h}) -> ({new_w}, {new_h})");
        let up = impact::tensor_resize(crop, w, h, new_w, new_h);
        let up = Tensor::from_vec(up, (1, new_h, new_w, 3), &self.dev)?.permute((0, 3, 1, 2))?;
        let mut latent = self.vae_encode(&up, true, vae)?;
        for c in 0..d.cycle {
            let spec = SampleSpec {
                model,
                seed: seed.wrapping_add(c),
                steps: d.steps,
                cfg: d.cfg,
                sampler: &d.sampler,
                scheduler: &d.scheduler,
                denoise: d.denoise,
                pos,
                neg,
                mask: noise_mask.as_ref().map(|m| (m, true)),
            };
            let (_, x0) = self.sample_sdxl(&spec, &latent)?;
            // process_latent_out on the CPU: a true division by the scale factor
            let shape = x0.dims().to_vec();
            let v: Vec<f32> =
                x0.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?.into_iter().map(|x| x / SDXL_SCALE as f32).collect();
            latent = Tensor::from_vec(v, shape, &self.dev)?;
        }
        let px = self.vae_decode(&latent, vae)?;
        let (_, _, dh, dw) = px.dims4()?;
        let px = px.permute((0, 2, 3, 1))?.flatten_all()?.to_vec1::<f32>()?;
        Ok(Some(impact::tensor_resize(&px, dw, dh, w, h)))
    }
}
