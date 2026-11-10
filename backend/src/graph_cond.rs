//! Masks, hooks and the conditioning nodes that carry them, each as its source defines it:
//! ComfyUI's SolidMask, MaskComposite, FeatherMask, MaskToImage, ImageToMask and InvertMask
//! (comfy_extras/nodes_mask.py), CreateHookLora, CreateHookLoraModelOnly, CombineHooks2,
//! SetClipHooks, ConditioningSetProperties and ConditioningSetPropertiesAndCombine
//! (comfy_extras/nodes_hooks.py with comfy/hooks.py), ConditioningCombine and ConditioningSetMask
//! (nodes.py), ControlNetLoader and ControlNetApplyAdvanced (nodes.py) and SetUnionControlNetType
//! (comfy_extras/nodes_controlnet.py). The sampler's side of masks, hooks and controls is graph.rs's
//! MultiCondDenoiser, the hooked weights model.rs's Patching, the ControlNet controlnet.rs.

use super::*;

/// The nodes this file runs
pub const COND_NODES: &[&str] = &[
    "SolidMask",
    "MaskComposite",
    "FeatherMask",
    "MaskToImage",
    "ImageToMask",
    "InvertMask",
    "CreateHookLora",
    "CreateHookLoraModelOnly",
    "CombineHooks2",
    "SetClipHooks",
    "ConditioningSetProperties",
    "ConditioningSetPropertiesAndCombine",
    "ConditioningCombine",
    "ConditioningSetMask",
    "ControlNetLoader",
    "SetUnionControlNetType",
    "ControlNetApplyAdvanced",
];

/// A mask as (batch, height, width) and its values, read off the card
fn mask_values(m: &Tensor) -> Result<(usize, usize, usize, Vec<f32>)> {
    let (h, w) = (m.dim(m.rank() - 2)?, m.dim(m.rank() - 1)?);
    let v = m.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    Ok((v.len() / (h * w).max(1), h, w, v))
}

impl Engine {
    /// An optional input's value, None when nothing is linked to it
    fn linked(&mut self, inp: &serde_json::Map<String, Value>, name: &str) -> Result<Option<Val>> {
        match inp.get(name) {
            Some(Value::Array(l)) if l.len() == 2 && l[1].is_u64() => Ok(Some(self.input(inp, name)?)),
            _ => Ok(None),
        }
    }

    fn mask_in(&mut self, inp: &serde_json::Map<String, Value>, name: &str, node: &str) -> Result<Tensor> {
        match self.input(inp, name)? {
            Val::Mask(m) => Ok(m),
            _ => bail!("{node}: {name} is not a mask"),
        }
    }

    /// An optional HOOKS input (CombineHooks2 can hand on none)
    fn hooks_in(&mut self, inp: &serde_json::Map<String, Value>, name: &str, node: &str) -> Result<Option<Rc<HookGroup>>> {
        match self.linked(inp, name)? {
            None | Some(Val::Nothing) => Ok(None),
            Some(Val::Hooks(h)) => Ok(Some(h)),
            Some(_) => bail!("{node}: {name} is not hooks"),
        }
    }

    fn cond_in(&mut self, inp: &serde_json::Map<String, Value>, name: &str, node: &str) -> Result<Rc<Vec<Cond>>> {
        match self.input(inp, name)? {
            Val::Cond(c) => Ok(c),
            _ => bail!("{node}: {name} is not conditioning"),
        }
    }

    pub(super) fn cond_node(&mut self, class: &str, inp: &serde_json::Map<String, Value>) -> Result<Option<Vec<Val>>> {
        let dev = self.dev.clone();
        Ok(Some(match class {
            "SolidMask" => {
                // torch.full((1, height, width), value, dtype=float32)
                let value = self.float_arg(inp, "value")? as f32;
                let (w, h) = (self.int_arg(inp, "width")?, self.int_arg(inp, "height")?);
                vec![Val::Mask(Tensor::full(value, (1, h, w), &dev)?)]
            }
            "MaskComposite" => {
                // both as (-1, h, w); the source's part that fits combined into the destination's
                // from (x, y) (and, or and xor on the values rounded half to even, as booleans),
                // then the whole clamped to [0, 1]
                let (db, dh, dw, mut out) = mask_values(&self.mask_in(inp, "destination", class)?)?;
                let (sb, sh, sw, s) = mask_values(&self.mask_in(inp, "source", class)?)?;
                let (x, y) = (self.int_arg(inp, "x")?, self.int_arg(inp, "y")?);
                let op = self.text_arg(inp, "operation")?;
                anyhow::ensure!(
                    ["multiply", "add", "subtract", "and", "or", "xor"].contains(&op.as_str()),
                    "MaskComposite: operation {op} is not one of ComfyUI's"
                );
                anyhow::ensure!(sb == db || sb == 1, "MaskComposite: a source batch of {sb} does not fit a destination batch of {db}");
                let (right, bottom) = ((x + sw).min(dw), (y + sh).min(dh));
                anyhow::ensure!(right >= x && bottom >= y, "MaskComposite: ({x}, {y}) is outside the destination");
                let on = |v: f32| v.round_ties_even() != 0.0;
                for bi in 0..db {
                    let si = if sb == 1 { 0 } else { bi };
                    for r in y..bottom {
                        for col in x..right {
                            let v = s[si * sh * sw + (r - y) * sw + (col - x)];
                            let d = &mut out[bi * dh * dw + r * dw + col];
                            *d = match op.as_str() {
                                "multiply" => *d * v,
                                "add" => *d + v,
                                "subtract" => *d - v,
                                "and" => (on(*d) & on(v)) as u8 as f32,
                                "or" => (on(*d) | on(v)) as u8 as f32,
                                _ => (on(*d) ^ on(v)) as u8 as f32,
                            };
                        }
                    }
                }
                for v in out.iter_mut() {
                    *v = v.clamp(0.0, 1.0);
                }
                vec![Val::Mask(Tensor::from_vec(out, (db, dh, dw), &dev)?)]
            }
            "FeatherMask" => {
                // the edge columns and rows times (k + 1) / n counted from each edge (a double
                // rounded to f32, as torch multiplies a float tensor by a python float): left,
                // right, top, bottom in turn, each at most the mask's size
                let (b, h, w, mut v) = mask_values(&self.mask_in(inp, "mask", class)?)?;
                let left = self.int_arg(inp, "left")?.min(w);
                let right = self.int_arg(inp, "right")?.min(w);
                let top = self.int_arg(inp, "top")?.min(h);
                let bottom = self.int_arg(inp, "bottom")?.min(h);
                let col = |v: &mut Vec<f32>, c: usize, rate: f64| {
                    for bi in 0..b {
                        for r in 0..h {
                            v[bi * h * w + r * w + c] *= rate as f32;
                        }
                    }
                };
                for x in 0..left {
                    col(&mut v, x, (x as f64 + 1.0) / left as f64);
                }
                for x in 0..right {
                    col(&mut v, w - 1 - x, (x + 1) as f64 / right as f64);
                }
                let row = |v: &mut Vec<f32>, r: usize, rate: f64| {
                    for bi in 0..b {
                        for c in 0..w {
                            v[bi * h * w + r * w + c] *= rate as f32;
                        }
                    }
                };
                for y in 0..top {
                    row(&mut v, y, (y + 1) as f64 / top as f64);
                }
                for y in 0..bottom {
                    row(&mut v, h - 1 - y, (y + 1) as f64 / bottom as f64);
                }
                vec![Val::Mask(Tensor::from_vec(v, (b, h, w), &dev)?)]
            }
            "MaskToImage" => {
                // the mask in all three channels
                let m = self.mask_in(inp, "mask", class)?;
                let (h, w) = (m.dim(m.rank() - 2)?, m.dim(m.rank() - 1)?);
                let m = m.to_dtype(DType::F32)?.reshape(((), 1, h, w))?;
                let b = m.dim(0)?;
                vec![Val::Image(m.broadcast_as((b, 3, h, w))?.contiguous()?)]
            }
            "ImageToMask" => {
                let Val::Image(img) = self.input(inp, "image")? else { bail!("ImageToMask: image is not an image") };
                let channel = self.text_arg(inp, "channel")?;
                let c = match channel.as_str() {
                    "red" => 0,
                    "green" => 1,
                    "blue" => 2,
                    other => bail!("ImageToMask: channel {other} is not one an RGB picture has"),
                };
                vec![Val::Mask(img.narrow(1, c, 1)?.squeeze(1)?.contiguous()?)]
            }
            "InvertMask" => {
                // 1.0 - mask
                let m = self.mask_in(inp, "mask", class)?;
                vec![Val::Mask(m.to_dtype(DType::F32)?.affine(-1.0, 1.0)?)]
            }
            "CreateHookLora" | "CreateHookLoraModelOnly" => {
                // a group of the previous hooks then this LoRA's (clone_and_combine; a new empty
                // group without them); both strengths 0 hand the previous group on as it is.
                // The model-only node is CreateHookLora with strength_clip 0.
                let name = self.text_arg(inp, "lora_name")?;
                let sm = self.float_arg(inp, "strength_model")?;
                let sc = if class == "CreateHookLora" { self.float_arg(inp, "strength_clip")? } else { 0.0 };
                let prev = self.hooks_in(inp, "prev_hooks", class)?.unwrap_or_else(|| Rc::new(HookGroup::default()));
                if sm == 0.0 && sc == 0.0 {
                    vec![Val::Hooks(prev)]
                } else {
                    let lora = self.model_path("loras", &name)?;
                    let hook = HookGroup {
                        hooks: vec![model::WeightHook { id: HookGroup::next_id(), lora, strength_model: sm, strength_clip: sc }],
                    };
                    vec![Val::Hooks(Rc::new(prev.combined(&hook)))]
                }
            }
            "CombineHooks2" => {
                // combine_all_hooks: none gives none, one gives itself, more a new group of the
                // first's hooks then each next one's it lacks
                let mut out: Option<Rc<HookGroup>> = None;
                for name in ["hooks_A", "hooks_B"] {
                    if let Some(h) = self.hooks_in(inp, name, class)? {
                        out = Some(match out {
                            None => h,
                            Some(prev) => Rc::new(prev.combined(&h)),
                        });
                    }
                }
                vec![out.map_or(Val::Nothing, Val::Hooks)]
            }
            "SetClipHooks" => {
                // the CLIP marked as hooked (whose encodes run the shared text encoders, see
                // Engine::encode), with apply_to_conds the hooks every prompt it encodes carries
                let Val::Clip(c, layer) = self.input(inp, "clip")? else { bail!("SetClipHooks: clip is not a clip") };
                let apply = self.flag_arg(inp, "apply_to_conds")?;
                let schedule = self.flag_arg(inp, "schedule_clip")?;
                match self.hooks_in(inp, "hooks", class)? {
                    None => vec![Val::Clip(c, layer)],
                    Some(h) => {
                        anyhow::ensure!(!schedule, "SetClipHooks: schedule_clip is not implemented yet");
                        let mut s = (*c).clone();
                        s.text_hooked = true;
                        if apply {
                            s.cond_hooks = Some(h);
                        }
                        vec![Val::Clip(Rc::new(s), layer)]
                    }
                }
            }
            "ConditioningSetProperties" | "ConditioningSetPropertiesAndCombine" => {
                // set_conds_props(_and_combine): on each of cond_NEW's prompts the hooks (appended
                // to any it has: one combined group per pair in this node), then the mask with
                // strength as its mask_strength (a strength without a mask sets nothing); the
                // combining node puts them after cond's
                let new = self.cond_in(inp, "cond_NEW", class)?;
                let strength = self.float_arg(inp, "strength")?;
                let area = self.text_arg(inp, "set_cond_area")?;
                let mask = match self.linked(inp, "mask")? {
                    None => None,
                    Some(Val::Mask(m)) => Some(if m.rank() < 3 { m.unsqueeze(0)? } else { m }),
                    Some(_) => bail!("{class}: mask is not a mask"),
                };
                let hooks = self.hooks_in(inp, "hooks", class)?;
                anyhow::ensure!(self.linked(inp, "timesteps")?.is_none(), "{class}: timesteps is not implemented yet");
                let mut out =
                    if class == "ConditioningSetPropertiesAndCombine" { self.cond_in(inp, "cond", class)?.to_vec() } else { vec![] };
                let mut combined: Vec<(*const HookGroup, Rc<HookGroup>)> = vec![];
                for c in new.iter() {
                    let mut c = c.clone();
                    if let Some(h) = &hooks {
                        c.props.hooks = Some(match &c.props.hooks {
                            None => h.clone(),
                            Some(prev) => match combined.iter().find(|(k, _)| *k == Rc::as_ptr(prev)) {
                                Some((_, g)) => g.clone(),
                                None => {
                                    let g = Rc::new(prev.combined(h));
                                    combined.push((Rc::as_ptr(prev), g.clone()));
                                    g
                                }
                            },
                        });
                    }
                    if let Some(m) = &mask {
                        c.props.mask = Some((m.clone(), strength));
                        c.props.area_to_bounds = area != "default";
                    }
                    out.push(c);
                }
                vec![Val::Cond(Rc::new(out))]
            }
            "ConditioningCombine" => {
                let a = self.cond_in(inp, "conditioning_1", class)?;
                let b = self.cond_in(inp, "conditioning_2", class)?;
                vec![Val::Cond(Rc::new(a.iter().chain(b.iter()).cloned().collect()))]
            }
            "ConditioningSetMask" => {
                let conds = self.cond_in(inp, "conditioning", class)?;
                let m = self.mask_in(inp, "mask", class)?;
                let m = if m.rank() < 3 { m.unsqueeze(0)? } else { m };
                let strength = self.float_arg(inp, "strength")?;
                let bounds = self.text_arg(inp, "set_cond_area")? != "default";
                let out = conds
                    .iter()
                    .map(|c| {
                        let mut c = c.clone();
                        c.props.mask = Some((m.clone(), strength));
                        c.props.area_to_bounds = bounds;
                        c
                    })
                    .collect();
                vec![Val::Cond(Rc::new(out))]
            }
            "ControlNetLoader" => {
                let name = self.text_arg(inp, "control_net_name")?;
                vec![Val::ControlNet(Rc::new((self.model_path("controlnet", &name)?, vec![])))]
            }
            "SetUnionControlNetType" => {
                // the union model's control type by name (control_types.py); "auto" or a name it
                // does not know sets none
                let Val::ControlNet(cn) = self.input(inp, "control_net")? else {
                    bail!("SetUnionControlNetType: control_net is not a ControlNet")
                };
                let ty = self.text_arg(inp, "type")?;
                let types = crate::controlnet::UNION_TYPES.iter().position(|t| *t == ty).into_iter().collect();
                vec![Val::ControlNet(Rc::new((cn.0.clone(), types)))]
            }
            "ControlNetApplyAdvanced" => {
                // a control on every prompt of both sides; prompts that had the same control before
                // share the new one, which is applied over it (set_previous_controlnet); strength 0
                // hands both sides on untouched. The vae input matters only to latent ControlNets.
                let pos = self.cond_in(inp, "positive", class)?;
                let neg = self.cond_in(inp, "negative", class)?;
                let strength = self.float_arg(inp, "strength")?;
                if strength == 0.0 {
                    return Ok(Some(vec![Val::Cond(pos), Val::Cond(neg)]));
                }
                let Val::ControlNet(cn) = self.input(inp, "control_net")? else {
                    bail!("ControlNetApplyAdvanced: control_net is not a ControlNet")
                };
                let Val::Image(image) = self.input(inp, "image")? else { bail!("ControlNetApplyAdvanced: image is not an image") };
                let nhwc = self.image_nhwc(inp, "image")?;
                let range = (self.float_arg(inp, "start_percent")?, self.float_arg(inp, "end_percent")?);
                let mut made: Vec<(Option<*const crate::controlnet::ControlApply>, Rc<crate::controlnet::ControlApply>)> = vec![];
                let mut out = vec![];
                for conds in [&pos, &neg] {
                    let mut side = vec![];
                    for c in conds.iter() {
                        let prev = c.props.control.clone();
                        let key = prev.as_ref().map(Rc::as_ptr);
                        let ca = match made.iter().find(|(k, _)| *k == key) {
                            Some((_, ca)) => ca.clone(),
                            None => {
                                let ca = Rc::new(crate::controlnet::ControlApply {
                                    net: cn.0.clone(),
                                    types: cn.1.clone(),
                                    image: image.clone(),
                                    nhwc,
                                    strength,
                                    range,
                                    prev,
                                });
                                made.push((key, ca.clone()));
                                ca
                            }
                        };
                        let mut c = c.clone();
                        c.props.control = Some(ca);
                        side.push(c);
                    }
                    out.push(Val::Cond(Rc::new(side)));
                }
                out
            }
            _ => return Ok(None),
        }))
    }
}
