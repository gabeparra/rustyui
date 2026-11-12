//! Runs a ComfyUI API graph (the JSON our tools post to /prompt) with the nodes implemented
//! so far. Models are loaded when first needed and dropped as soon as the next stage starts:
//! text encoders before the UNet, the UNet before the VAE, so VRAM holds one model at a time.

use crate::clip::{Cond, SdxlClip};
use crate::controlnet::{ControlApply, ControlNet};

use crate::model::{self, File, HookGroup, Source};
use crate::prompt::Embeddings;
use crate::sampling::{self, Denoise, Discrete};
use crate::unet::{timestep_embedding, Unet};
use crate::upscale::Esrgan;
use crate::vae::{Decoder, Encoder};
use crate::{image, rng};
use anyhow::{bail, Context, Result};
use candle_core::{DType, Device, Tensor};
use serde_json::Value;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::time::Instant;

#[path = "graph_ltx.rs"]
mod ltx_nodes;
pub use ltx_nodes::{LTX_NODES, SHARED_NODES};
#[path = "graph_utils.rs"]
mod util_nodes;
pub use util_nodes::UTIL_NODES;
#[path = "graph_cond.rs"]
mod cond_nodes;
pub use cond_nodes::COND_NODES;
#[path = "graph_impact.rs"]
mod impact_nodes;
pub use impact_nodes::IMPACT_NODES;
#[path = "graph_tac.rs"]
mod tac_nodes;
pub use tac_nodes::TAC_NODES;
#[path = "graph_anima.rs"]
mod anima_nodes;
pub use anima_nodes::ANIMA_NODES;
use ltx_nodes::{AudioVal, DitCache, Guider, LtxCond, LtxLatent, VideoVal};

const SDXL_SCALE: f64 = 0.13025;

#[derive(Clone)]
struct Stack {
    ckpt: PathBuf,
    loras: Vec<(PathBuf, f64)>,
    /// a CLIP's apply_hooks_to_conds (SetClipHooks): every prompt it encodes carries these hooks
    cond_hooks: Option<Rc<HookGroup>>,
    /// a CLIP SetClipHooks gave hooks to, which encodes with the shared text encoders (Engine::encode)
    text_hooked: bool,
    /// DifferentialDiffusion's strength: its denoise_mask_function on the model
    differential: Option<f64>,
    /// TiledAttentionCouple's cross-attention patch
    couple: Option<Rc<crate::unet::Couple>>,
}

impl Stack {
    fn new(ckpt: PathBuf) -> Self {
        Stack { ckpt, loras: vec![], cond_hooks: None, text_hooked: false, differential: None, couple: None }
    }
}

/// The UNet left on the card: the weights a sampler runs (DynamicVRAM's fold, or the classic
/// ModelPatcher's when a conditioning has hooks), and for a classic load its weights by name and
/// the hook groups' weights made from them
struct LoadedUnet {
    key: String,
    base: Unet,
    weights: std::cell::RefCell<HashMap<String, Tensor>>,
    hooked: Vec<(String, Unet)>,
    /// the ControlNets its samplers use, by file (they leave the card with it)
    controlnets: Vec<(PathBuf, ControlNet)>,
}

#[derive(Clone)]
enum Val {
    Model(Rc<Stack>),
    Clip(Rc<Stack>, i64),
    Vae(PathBuf),
    Cond(Rc<Vec<Cond>>),
    Latent(Tensor),
    /// a LATENT with a noise_mask (SetLatentNoiseMask's): the samples and the mask, (b, 1, h, w)
    /// at the picture's size, f32; KSampler keeps it in its output as ComfyUI's dict copy does
    MaskedLatent(Tensor, Tensor),
    /// (b, 3, h, w) f32 in [0, 1], ComfyUI's IMAGE (a video's frames are its batch)
    Image(Tensor),
    UpscaleModel(PathBuf),
    // the LTX-2.3 values (graph_ltx.rs)
    /// LatentUpscaleModelLoader's latent upsampler (its file)
    LatentUpscaler(PathBuf),
    /// LTXAVTextEncoderLoader's CLIP: the Gemma 3 file and the checkpoint holding the projection
    LtxClip(Rc<(PathBuf, PathBuf)>),
    LtxCond(Rc<LtxCond>),
    /// UNETLoader's LTX-2.3 DiT, and LoRAs on it
    LtxModel(Rc<Stack>),
    /// LTXVAudioVAELoader's audio VAE (its file)
    AudioVae(PathBuf),
    /// ComfyUI's AUDIO: a waveform (B, C, samples) f32 and its sample rate
    Audio(Rc<AudioVal>),
    LtxLatent(Rc<LtxLatent>),
    Noise(u64),
    Sampler(String),
    Sigmas(Rc<Vec<f32>>),
    Guider(Rc<Guider>),
    /// (b, h, w) f32, ComfyUI's MASK
    Mask(Tensor),
    /// Impact Pack's SEGS (impact.rs)
    Segs(Rc<crate::impact::Segs>),
    /// TiledAttentionCouplePrompt's TILED_CONFIG
    TiledConfig(Rc<tac_nodes::TiledConfig>),
    /// CreateHookLora's HOOKS
    Hooks(Rc<HookGroup>),
    /// ControlNetLoader's CONTROL_NET: its file and SetUnionControlNetType's types
    ControlNet(Rc<(PathBuf, Vec<usize>)>),

    Video(Rc<VideoVal>),
    /// UNETLoader's Anima model (graph_anima.rs)
    AnimaModel(Rc<Stack>),
    /// CLIPLoader's Qwen3-0.6B, Anima's text encoder (its file)
    AnimaClip(Rc<PathBuf>),
    /// CLIPTextEncode on Qwen3-0.6B
    AnimaCond(Rc<anima_nodes::AnimaCond>),
    // ComfyUI's INT, FLOAT, STRING and BOOLEAN outputs, which can feed a widget input by a link
    Int(i64),
    #[allow(dead_code)] // no node makes one yet
    Float(f64),
    #[allow(dead_code)]
    Str(String),
    #[allow(dead_code)]
    Bool(bool),
    /// an output with nothing in it (PlaySoundKJ's with no input)
    Nothing,
}

pub struct Engine {
    pub dev: Device,
    models: PathBuf,
    out_dir: PathBuf,
    sms: u32,
    graph: serde_json::Map<String, Value>,
    memo: HashMap<(String, usize), Vec<Val>>,
    /// ComfyUI's cache between graphs: a node's outputs (and the files it saved) by its
    /// signature, kept for the nodes of the last graph only, as ComfyUI's classic cache keeps them
    cache: HashMap<u64, (Vec<Val>, Vec<PathBuf>)>,
    sigs: HashMap<String, u64>,
    used: std::collections::HashSet<u64>,
    /// the LTX-2.3 DiT left on the card for the next graph
    ltx_dit: Option<DitCache>,
    /// the Anima DiT left on the card for the next graph, by its checkpoint and LoRAs (stack_key)
    anima: Option<(String, crate::anima::AnimaDit)>,
    /// Anima's two tokenizers, built once
    anima_tok: Option<Rc<crate::anima_tokens::AnimaTokenizer>>,
    /// the smallest latent (batch x height x width) at which a sampler's resident hook-group
    /// weights ran the card out of memory: from that size on each group's are made per call
    hooks_lean: Option<usize>,
    /// the smallest video latent (frames x height x width) whose decode ran out of memory beside
    /// the DiT: from that size on the DiT goes first rather than after a failed attempt
    ltx_decode_oom: Option<usize>,
    /// the last prompt's text context (the connectors' output) for a checkpoint: it outlives the
    /// DiT leaving the card, so a new seed does not run the connectors again
    /// the text context last computed: its checkpoint, the projected text it came from (by tensor,
    /// so prompts that only add guides or a frame rate share it), the context
    ltx_context: Vec<(PathBuf, candle_core::TensorId, Tensor)>,
    files: HashMap<PathBuf, Rc<File>>,
    clip: Option<(String, SdxlClip)>,
    /// the checkpoints whose model a sampler has loaded since the engine started: ComfyUI keeps
    /// such a model's sigma table on the card, and a scheduler there computes on it
    sigmas_on_card: std::collections::HashSet<PathBuf>,
    /// per checkpoint, the CLIP (its LoRAs) whose unhooked encode last ran its text encoders,
    /// which a hooked CLIP encodes with (Engine::encode)
    clip_prepared: HashMap<PathBuf, Stack>,
    /// the CLIPs (by their key) that have encoded since the engine started: ComfyUI keeps their
    /// encoders loaded with the LoRA patches settled, so a reload starts settled
    clip_settled: std::collections::HashSet<String>,
    unet: Option<LoadedUnet>,
    /// the SDXL VAE's decoder and encoder and the last upscale model, by file: like the text
    /// encoders and the UNet they stay on the card between nodes and graphs while there is room
    /// for what runs next (Engine::make_room), as ComfyUI keeps its loaded models
    vae_dec: Option<(PathBuf, Rc<Decoder>)>,
    vae_enc: Option<(PathBuf, Rc<Encoder>)>,
    esrgan: Option<(PathBuf, Rc<Esrgan>)>,
    /// RUSTYUI_ONE_MODEL=1: the old way, one model on the card at a time
    one_model: bool,
    /// SAM3Segment's model, kept between graphs as the node's processor cache keeps it
    sam3: Option<Rc<crate::sam3::Sam3>>,
    /// Gemma 3 and the projection's checkpoint, kept between an LTX graph's prompts
    ltx_te: Option<(String, crate::gemma::Gemma, candle_core::safetensors::MmapedSafetensors)>,
    embs: Embeddings,
    pub saved: Vec<PathBuf>,
    /// extra PNG text chunks for this graph's SaveImage (the request's extra_pnginfo)
    pub pnginfo: Vec<(String, String)>,
    pub timings: Vec<(String, f64)>,
    pub peak_vram_mb: f64,
    /// RUSTYUI_DUMP_NODES: every node's output tensors this graph, written there at its end, named
    /// as refs/graph_nodes_ref.py names ComfyUI's
    dump: Option<HashMap<String, Tensor>>,
}

/// a model Engine::make_room keeps on the card
#[derive(Clone, Copy, PartialEq, Eq)]
enum Keep {
    Nothing,
    Anima,
    Esrgan,
    VaeEnc,
    VaeDec,
    Clip,
    Unet,
}

/// the order Engine::make_room lets models go in
const EVICT_ORDER: [Keep; 6] = [Keep::Anima, Keep::Esrgan, Keep::VaeEnc, Keep::VaeDec, Keep::Clip, Keep::Unet];

/// VRAM to load both SDXL text encoders (1.6 GB in f16) and encode a prompt
const CLIP_NEED: usize = 3 << 30;
/// VRAM to load an SDXL UNet (4.9 GB in f16) with its LoRAs folded in
const UNET_NEED: usize = 11 << 29;
/// ImageUpscaleWithModel's estimate for a 4x model: (512 * 512 * 3) * 4 bytes * 4 * 384, and the model
const UPSCALE_NEED: usize = 512 * 512 * 3 * 4 * 4 * 384 + (128 << 20);

fn stack_key(s: &Stack, extra: &str) -> String {
    let mut k = format!("{}|{extra}", s.ckpt.display());
    for (p, w) in &s.loras {
        k.push_str(&format!("|{}@{w}", p.display()));
    }
    k
}

/// One node output's tensors for RUSTYUI_DUMP_NODES, as f32 on the CPU, named as ComfyUI's are
/// flattened (an IMAGE (b, h, w, 3), a LATENT's samples, a NestedTensor's parts n0 / n1)
fn dump_val(key: &str, v: &Val, d: &mut HashMap<String, Tensor>) -> Result<()> {
    let mut put = |k: String, t: &Tensor| -> Result<()> {
        let t = if t.dtype().is_float() { t.to_dtype(DType::F32)? } else { t.clone() };
        d.insert(k, t.to_device(&Device::Cpu)?);
        Ok(())
    };
    match v {
        Val::Latent(t) => put(format!("{key}.samples"), t)?,
        Val::MaskedLatent(t, m) => {
            put(format!("{key}.samples"), t)?;
            put(format!("{key}.noise_mask"), m)?;
        }
        Val::Image(t) => put(key.to_string(), &t.permute((0, 2, 3, 1))?.contiguous()?)?,
        Val::Mask(t) => put(key.to_string(), t)?,
        Val::LtxLatent(l) => {
            for (k, t) in l.parts() {
                put(format!("{key}.{k}"), t)?;
            }
        }
        Val::LtxCond(c) => put(format!("{key}.0.0"), c.tensor())?,
        Val::Cond(cs) => {
            for (i, c) in cs.iter().enumerate() {
                put(format!("{key}.{i}.0"), &c.cond)?;
                put(format!("{key}.{i}.1.pooled_output"), &c.pooled)?;
                if let Some((m, _)) = &c.props.mask {
                    put(format!("{key}.{i}.1.mask"), m)?;
                }
            }
        }
        Val::Audio(a) => put(format!("{key}.waveform"), &a.waveform)?,
        Val::AnimaCond(c) => {
            for (k, t) in c.parts()? {
                put(format!("{key}.{k}"), &t)?;
            }
        }
        _ => {}
    }
    Ok(())
}

impl Engine {
    pub fn new(out_dir: &Path) -> Result<Self> {
        let dev = Device::new_cuda(0)?;
        let models = crate::paths::models_dir();
        let sms = {
            use candle_core::cuda_backend::cudarc::driver::{sys::CUdevice_attribute, CudaContext};
            let ctx = CudaContext::new(0)?;
            ctx.attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)? as u32
        };
        Ok(Engine {
            dev,
            embs: Embeddings::new(vec![models.join("embeddings")]),
            models,
            out_dir: out_dir.to_path_buf(),
            sms,
            graph: serde_json::Map::new(),
            memo: HashMap::new(),
            cache: HashMap::new(),
            sigs: HashMap::new(),
            used: std::collections::HashSet::new(),
            ltx_dit: None,
            anima: None,
            anima_tok: None,
            ltx_decode_oom: None,
            hooks_lean: None,

            ltx_context: vec![],
            files: HashMap::new(),
            clip: None,
            clip_prepared: HashMap::new(),
            clip_settled: std::collections::HashSet::new(),
            sigmas_on_card: std::collections::HashSet::new(),

            unet: None,
            vae_dec: None,
            vae_enc: None,
            esrgan: None,
            one_model: std::env::var_os("RUSTYUI_ONE_MODEL").is_some_and(|v| v != "0"),
            sam3: None,
            ltx_te: None,
            saved: vec![],
            pnginfo: vec![],
            timings: vec![],
            dump: None,
            peak_vram_mb: 0.0,
        })
    }

    fn vram(&mut self) {
        use candle_core::cuda_backend::cudarc::driver::CudaContext;
        if let Ok(ctx) = CudaContext::new(0) {
            if let Ok((free, total)) = ctx.mem_get_info() {
                let used = (total - free) as f64 / 1048576.0;
                self.peak_vram_mb = self.peak_vram_mb.max(used);
            }
        }
    }

    /// free VRAM in bytes, as cudaMemGetInfo reports it
    fn free_vram(&self) -> Result<usize> {
        match &self.dev {
            Device::Cuda(d) => Ok(d.cuda_stream().context().mem_get_info().map_err(|e| anyhow::anyhow!("free VRAM: {e:?}"))?.0),
            _ => Ok(usize::MAX),
        }
    }

    /// Lets the image models on the card go until `need` bytes are free, keeping `keep`: the Anima
    /// DiT first (another family's), then the SDXL models, the quickest to load again first. The
    /// video models always go (they need most of the card). With RUSTYUI_ONE_MODEL every other
    /// model goes, as before.
    fn make_room(&mut self, need: usize, keep: Keep) -> Result<()> {
        self.ltx_te = None;
        self.ltx_dit = None;
        let mut dropped = false;
        for k in EVICT_ORDER {
            if k == keep || !self.holds(k) {
                continue;
            }
            if !self.one_model {
                // what was freed (here or by the node before) goes back from the stream's memory
                // pool at a synchronize, and only then does cudaMemGetInfo count it as free
                self.dev.synchronize()?;
                dropped = false;
                if self.free_vram()? >= need {
                    break;
                }
            }
            match k {
                Keep::Anima => self.anima = None,
                Keep::Esrgan => self.esrgan = None,
                Keep::VaeEnc => self.vae_enc = None,
                Keep::VaeDec => self.vae_dec = None,
                Keep::Clip => self.clip = None,
                Keep::Unet => self.unet = None,
                Keep::Nothing => {}
            }
            dropped = true;
        }
        if dropped {
            self.dev.synchronize()?;
        }
        Ok(())
    }

    /// whether that model is on the card
    fn holds(&self, k: Keep) -> bool {
        match k {
            Keep::Anima => self.anima.is_some(),
            Keep::Esrgan => self.esrgan.is_some(),
            Keep::VaeEnc => self.vae_enc.is_some(),
            Keep::VaeDec => self.vae_dec.is_some(),
            Keep::Clip => self.clip.is_some(),
            Keep::Unet => self.unet.is_some(),
            Keep::Nothing => false,
        }
    }

    /// Every image model but `keep` off the card, after an attempt ran out of memory beside them;
    /// false when there was nothing to let go (the error stands)
    fn room_after_oom(&mut self, keep: Keep) -> Result<bool> {
        if !EVICT_ORDER.into_iter().any(|k| k != keep && self.holds(k)) {
            return Ok(false);
        }
        self.make_room(usize::MAX, keep)?;
        eprintln!("[engine] out of memory beside the models kept on the card: let them go and ran again");
        Ok(true)
    }

    fn file(&mut self, path: &Path) -> Result<Rc<File>> {
        if let Some(f) = self.files.get(path) {
            return Ok(f.clone());
        }
        let f = File::open(path)?;
        self.files.insert(path.to_path_buf(), f.clone());
        Ok(f)
    }

    fn model_path(&self, folder: &str, name: &str) -> Result<PathBuf> {
        let rel = crate::paths::contained(name).with_context(|| format!("{folder}/{name:?}: not a file name inside the models folder"))?;
        let p = self.models.join(folder).join(rel);
        if p.is_file() {
            return Ok(p);
        }
        bail!("{folder}/{name} not found under {}", self.models.display())
    }

    /// Whether ComfyUI holds the IMAGE linked at `name` as (B, H, W, C) memory rather than a view
    /// of NCHW memory, which is what torch's convolutions see once a node permutes it to NCHW (RIFE
    /// VFI's encoder runs channels-last on the first). LoadImage, ResizeImagesByLongerEdge and
    /// LTXVPreprocess make NHWC memory; VAEDecode, ImageUpscaleWithModel and lanczos resizes NCHW;
    /// bilinear resizes, ImageFromBatch and DetailerForEach keep their input's (each checked
    /// against ComfyUI's node, 2026-10-15). Anything else counts as NCHW.
    pub(super) fn image_nhwc(&self, inputs: &serde_json::Map<String, Value>, name: &str) -> Result<bool> {
        let mut link = inputs.get(name).cloned();
        for _ in 0..64 {
            let Some(Value::Array(l)) = link else { return Ok(false) };
            let Some(id) = l.first().and_then(|v| v.as_str().map(str::to_string).or_else(|| v.as_u64().map(|n| n.to_string()))) else {
                return Ok(false);
            };
            let (class, inp) = self.node(&id)?;
            match class.as_str() {
                "LoadImage" | "ResizeImagesByLongerEdge" | "LTXVPreprocess" | "DepthAnythingV2Preprocessor" | "SAM3Segment" => {
                    return Ok(true)
                }
                "ImageFromBatch" | "DetailerForEach" => link = inp.get("image").cloned(),
                "ImageScale" | "ImageScaleBy" | "ImageResizeKJv2" => {
                    if inp.get("upscale_method").and_then(|v| v.as_str()) != Some("bilinear") {
                        return Ok(false);
                    }
                    link = inp.get("image").cloned();
                }
                _ => return Ok(false),
            }
        }
        Ok(false)
    }

    fn node(&self, id: &str) -> Result<(String, serde_json::Map<String, Value>)> {
        let n = self.graph.get(id).with_context(|| format!("no node {id}"))?;
        let class = n["class_type"].as_str().context("node without class_type")?.to_string();
        let inputs = n["inputs"].as_object().cloned().unwrap_or_default();
        Ok((class, inputs))
    }

    fn input(&mut self, inputs: &serde_json::Map<String, Value>, name: &str) -> Result<Val> {
        match &inputs.get(name) {
            Some(Value::Array(link)) if link.len() == 2 => {
                let id = link[0].as_str().map(str::to_string).unwrap_or_else(|| link[0].to_string());
                let out = link[1].as_u64().context("bad link")? as usize;
                let vals = self.eval(&id)?;
                vals.get(out).cloned().with_context(|| format!("node {id} has no output {out}"))
            }
            _ => bail!("input {name} is not a link"),
        }
    }

    /// An input's value: its literal, or what the INT, FLOAT, STRING or BOOLEAN output it is linked
    /// to holds (in ComfyUI a widget input can be driven by a link, GetImageSize's width into a
    /// latent's)
    fn arg(&mut self, inputs: &serde_json::Map<String, Value>, name: &str) -> Result<Value> {
        match inputs.get(name) {
            Some(Value::Array(link)) if link.len() == 2 && link[1].is_u64() => match self.input(inputs, name)? {
                Val::Int(v) => Ok(serde_json::json!(v)),
                Val::Float(v) => Ok(serde_json::json!(v)),
                Val::Str(v) => Ok(serde_json::json!(v)),
                Val::Bool(v) => Ok(serde_json::json!(v)),
                _ => bail!("input {name} is linked to an output that is not a number, text or switch"),
            },
            Some(v) => Ok(v.clone()),
            None => bail!("missing input {name}"),
        }
    }

    fn text_arg(&mut self, inputs: &serde_json::Map<String, Value>, name: &str) -> Result<String> {
        self.arg(inputs, name)?.as_str().map(str::to_string).with_context(|| format!("{name} is not text"))
    }

    fn int_arg(&mut self, inputs: &serde_json::Map<String, Value>, name: &str) -> Result<usize> {
        self.arg(inputs, name)?.as_u64().map(|v| v as usize).with_context(|| format!("{name} is not a whole number"))
    }

    fn float_arg(&mut self, inputs: &serde_json::Map<String, Value>, name: &str) -> Result<f64> {
        self.arg(inputs, name)?.as_f64().with_context(|| format!("{name} is not a number"))
    }

    fn flag_arg(&mut self, inputs: &serde_json::Map<String, Value>, name: &str) -> Result<bool> {
        self.arg(inputs, name)?.as_bool().with_context(|| format!("{name} is not true or false"))
    }

    /// ComfyUI's cache key for a node: its class, its literal inputs and, through each link, the
    /// key of the node the link comes from, so a node is reused only while nothing upstream has
    /// changed. LoadImage also keys on its file's size and time (ComfyUI's IS_CHANGED hashes the file).
    fn signature(&mut self, id: &str, depth: usize) -> Result<u64> {
        use std::hash::{Hash, Hasher};
        if let Some(s) = self.sigs.get(id) {
            return Ok(*s);
        }
        anyhow::ensure!(depth < 10_000, "the graph loops back on itself at node {id}");
        let (class, inp) = self.node(id)?;
        let mut h = std::collections::hash_map::DefaultHasher::new();
        class.hash(&mut h);
        let mut keys: Vec<&String> = inp.keys().collect();
        keys.sort();
        for k in keys {
            k.hash(&mut h);
            match &inp[k] {
                Value::Array(link) if link.len() == 2 && link[1].is_u64() => {
                    let up = link[0].as_str().map(str::to_string).unwrap_or_else(|| link[0].to_string());
                    self.signature(&up, depth + 1)?.hash(&mut h);
                    link[1].as_u64().hash(&mut h);
                }
                v => v.to_string().hash(&mut h),
            }
        }
        if class == "LoadImage" {
            let name = inp.get("image").and_then(Value::as_str).unwrap_or("");
            let path = self.models.parent().unwrap_or(&self.models).join("input").join(name.strip_suffix(" [input]").unwrap_or(name));
            if let Ok(m) = std::fs::metadata(&path) {
                m.len().hash(&mut h);
                m.modified().ok().hash(&mut h);
            }
        }
        let sig = h.finish();
        self.sigs.insert(id.to_string(), sig);
        Ok(sig)
    }

    /// The order ComfyUI's ExecutionList runs the nodes these outputs need in (comfy_execution/
    /// graph.py), which matters beyond the editor: the first text encode after its model loads
    /// computes with the LoRAs' f32 patches and the later ones with f16-rounded weights. Each
    /// output is added in turn by add_node's stack (a node's linked inputs pushed in the prompt's
    /// order, so the last is followed first), a node whose output is in the cache is left out, and
    /// then, of the nodes whose inputs have all run, ux_friendly_pick_node takes an output node,
    /// else one an output waits on, else one two links from an output, else the first pending
    /// (none of rustyui's nodes is async, which it would take first too).
    fn comfy_order(&mut self, outputs: &[String]) -> Result<Vec<String>> {
        let mut pending: Vec<String> = vec![];
        let mut block_count: HashMap<String, usize> = HashMap::new();
        let mut blocking: HashMap<String, Vec<String>> = HashMap::new();
        for out in outputs {
            let (mut stack, mut links) = (vec![out.clone()], vec![]);
            while let Some(u) = stack.pop() {
                if block_count.contains_key(&u) {
                    continue;
                }
                pending.push(u.clone());
                block_count.insert(u.clone(), 0);
                blocking.insert(u.clone(), vec![]);
                let (_, inp) = self.node(&u)?;
                for v in inp.values() {
                    if let Value::Array(link) = v {
                        if link.len() == 2 && link[1].is_u64() {
                            let from = link[0].as_str().map(str::to_string).unwrap_or_else(|| link[0].to_string());
                            if !self.is_cached(&from) {
                                stack.push(from.clone());
                            }
                            links.push((from, u.clone()));
                        }
                    }
                }
            }
            for (from, to) in links {
                if self.is_cached(&from) {
                    continue;
                }
                let b = blocking.get_mut(&from).context("a link from a node that is not pending")?;
                if !b.contains(&to) {
                    b.push(to.clone());
                    *block_count.get_mut(&to).context("a link to a node that is not pending")? += 1;
                }
            }
        }
        let class = |s: &Self, id: &str| s.graph.get(id).and_then(|n| n["class_type"].as_str()).unwrap_or("").to_string();
        let is_output = |s: &Self, id: &str| OUTPUT_NODES.contains(&class(s, id).as_str());
        let mut order = vec![];
        while !pending.is_empty() {
            let available: Vec<String> = pending.iter().filter(|n| block_count[*n] == 0).cloned().collect();
            anyhow::ensure!(!available.is_empty(), "the graph loops back on itself");
            let waits = |n: &String| blocking[n].iter().any(|b| is_output(self, b));
            let waits2 = |n: &String| blocking[n].iter().any(|b| blocking.get(b).is_some_and(|bb| bb.iter().any(|c| is_output(self, c))));
            let pick = available
                .iter()
                .find(|n| is_output(self, n))
                .or_else(|| available.iter().find(|n| waits(n)))
                .or_else(|| available.iter().find(|n| waits2(n)))
                .unwrap_or(&available[0])
                .clone();
            pending.retain(|n| n != &pick);
            for b in blocking.remove(&pick).unwrap_or_default() {
                *block_count.get_mut(&b).context("a node that is not pending")? -= 1;
            }
            order.push(pick);
        }
        Ok(order)
    }

    /// Whether this node's outputs are in the cache (ComfyUI's ExecutionList.is_cached)
    fn is_cached(&mut self, id: &str) -> bool {
        self.signature(id, 0).is_ok_and(|s| self.cache.contains_key(&s))
    }

    fn eval(&mut self, id: &str) -> Result<Vec<Val>> {
        if let Some(v) = self.memo.get(&(id.to_string(), 0)) {
            return Ok(v.clone());
        }
        let sig = self.signature(id, 0)?;
        self.used.insert(sig);
        if let Some((v, files)) = self.cache.get(&sig).cloned() {
            self.saved.extend(files);
            self.memo.insert((id.to_string(), 0), v.clone());
            return Ok(v);
        }
        let saved_before = self.saved.len();
        let (class, inp) = self.node(id)?;
        // ComfyUI's order: a node runs once everything it takes has, so the editor sees each node
        // start and finish in turn (the loaders only name their files, so nothing loads sooner)
        for v in inp.values() {
            if let Value::Array(link) = v {
                if link.len() == 2 && link[1].is_u64() {
                    let up = link[0].as_str().map(str::to_string).unwrap_or_else(|| link[0].to_string());
                    self.eval(&up)?;
                }
            }
        }
        // an interrupt takes effect between nodes too (ComfyUI's before_node_execution)
        crate::progress::check()?;
        let executing = crate::progress::Executing::new(id, &class);
        let t0 = Instant::now();
        let out = match self.compute(id, &class, &inp) {
            // a node that ran out of memory beside the models kept from earlier nodes runs once
            // more without them, as it would have when only one model stayed on the card
            Err(e) if ltx_nodes::is_oom(&e) && self.room_after_oom(Keep::Nothing)? => self.compute(id, &class, &inp)?,
            r => r?,
        };
        let _ = t0;
        executing.done();
        if let Some(d) = self.dump.as_mut() {
            for (i, v) in out.iter().enumerate() {
                dump_val(&format!("{id}.{i}"), v, d)?;
            }
        }
        self.memo.insert((id.to_string(), 0), out.clone());
        // big results (a video's frames) are not kept: they would hold VRAM until the next graph
        if out.iter().map(val_bytes).sum::<usize>() <= 256 << 20 {
            let files = self.saved[saved_before..].to_vec();
            self.cache.insert(sig, (out.clone(), files));
        }
        Ok(out)
    }

    /// What one node computes, from its inputs (eval has run the nodes they link to)
    fn compute(&mut self, id: &str, class: &str, inp: &serde_json::Map<String, Value>) -> Result<Vec<Val>> {
        Ok(match class {
            "CheckpointLoaderSimple" => {
                let name = self.text_arg(inp, "ckpt_name")?;
                let path = self.model_path("checkpoints", &name)?;
                let s = Rc::new(Stack::new(path.clone()));
                vec![Val::Model(s.clone()), Val::Clip(s, -2), Val::Vae(path)]
            }
            "LoraLoader" | "LoraLoaderModelOnly" => {
                let name = self.text_arg(inp, "lora_name")?;
                let path = self.model_path("loras", &name)?;
                let sm = self.float_arg(inp, "strength_model")?;
                // which model: 0 SDXL, 1 LTX (model only), 2 Anima
                let (m, kind) = match self.input(inp, "model")? {
                    Val::Model(m) => (m, 0),
                    Val::LtxModel(m) if class == "LoraLoaderModelOnly" => (m, 1),
                    Val::AnimaModel(m) => (m, 2),
                    _ => bail!("LoraLoader: model input is not a model"),
                };
                let mut m2 = (*m).clone();
                if sm != 0.0 {
                    m2.loras.push((path.clone(), sm));
                }
                let m2 = Rc::new(m2);
                let mut out = vec![match kind {
                    1 => Val::LtxModel(m2),
                    2 => Val::AnimaModel(m2),
                    _ => Val::Model(m2),
                }];
                if class == "LoraLoader" && kind == 2 {
                    // Anima's text encoder: a LoRA without lora_te keys leaves it as it is
                    let clip = self.input(inp, "clip")?;
                    anyhow::ensure!(matches!(clip, Val::AnimaClip(_)), "LoraLoader: an Anima model with a clip that is not Qwen3-0.6B");
                    let st = unsafe { candle_core::safetensors::MmapedSafetensors::new(&path)? };
                    let te = st.tensors().into_iter().any(|(k, _)| !k.starts_with("lora_unet_"));
                    anyhow::ensure!(
                        !te || self.float_arg(inp, "strength_clip")? == 0.0,
                        "LoraLoader: {name} has text-encoder keys; LoRAs on Anima's Qwen3 are not implemented yet"
                    );
                    out.push(clip);
                } else if class == "LoraLoader" {
                    let sc = self.float_arg(inp, "strength_clip")?;
                    let Val::Clip(c, layer) = self.input(inp, "clip")? else { bail!("LoraLoader: clip input is not a clip") };
                    let mut c2 = (*c).clone();
                    if sc != 0.0 {
                        c2.loras.push((path, sc));
                    }
                    out.push(Val::Clip(Rc::new(c2), layer));
                }
                out
            }
            "CLIPSetLastLayer" => {
                let layer = self.arg(inp, "stop_at_clip_layer")?.as_i64().context("stop_at_clip_layer is not a whole number")?;
                let Val::Clip(c, _) = self.input(inp, "clip")? else { bail!("CLIPSetLastLayer: not a clip") };
                vec![Val::Clip(c, layer)]
            }
            "CLIPTextEncode" => {
                let text = self.text_arg(inp, "text")?;
                match self.input(inp, "clip")? {
                    Val::Clip(c, layer) => {
                        let mut cond = self.encode(&c, layer, &text)?;
                        // CLIP.add_hooks_to_dict: a CLIP's apply_hooks_to_conds go into each prompt
                        cond.props.hooks = c.cond_hooks.clone();
                        vec![Val::Cond(Rc::new(vec![cond]))]
                    }
                    Val::LtxClip(files) => vec![self.ltx_encode(&files, &text)?],
                    Val::AnimaClip(file) => vec![self.anima_encode(&file, &text)?],
                    _ => bail!("CLIPTextEncode: not a clip"),
                }
            }
            "ConditioningConcat" => {
                let Val::Cond(to) = self.input(inp, "conditioning_to")? else { bail!("ConditioningConcat: not conditioning") };
                let Val::Cond(from) = self.input(inp, "conditioning_from")? else { bail!("ConditioningConcat: not conditioning") };
                let from0 = &from[0].cond;
                let mut out = vec![];
                for c in to.iter() {
                    out.push(Cond { cond: Tensor::cat(&[&c.cond, from0], 1)?, pooled: c.pooled.clone(), props: c.props.clone() });
                }
                vec![Val::Cond(Rc::new(out))]
            }
            "EmptyLatentImage" => {
                let w = self.int_arg(inp, "width")?;
                let h = self.int_arg(inp, "height")?;
                let b = self.int_arg(inp, "batch_size")?;
                vec![Val::Latent(Tensor::zeros((b, 4, h / 8, w / 8), DType::F32, &self.dev)?)]
            }
            "KSampler" if matches!(self.input(inp, "model")?, Val::AnimaModel(_)) => vec![self.anima_ksampler(inp)?],
            "KSampler" => {
                let (z, mask) = self.ksampler(inp)?;
                vec![match mask {
                    Some(m) => Val::MaskedLatent(z, m),
                    None => Val::Latent(z),
                }]
            }
            "VAEDecode" if matches!(self.input(inp, "samples")?, Val::LtxLatent(_)) => {
                let Val::LtxLatent(l) = self.input(inp, "samples")? else { unreachable!() };
                let Val::Vae(path) = self.input(inp, "vae")? else { bail!("VAEDecode: not a vae") };
                vec![self.ltx_decode(&l, &path)?]
            }
            "VAEDecode" if matches!(self.input(inp, "samples")?, Val::Latent(ref z) if z.rank() == 5) => {
                let Val::Latent(z) = self.input(inp, "samples")? else { unreachable!() };
                let Val::Vae(path) = self.input(inp, "vae")? else { bail!("VAEDecode: not a vae") };
                vec![Val::Image(self.wan_decode(&z, &path)?)]
            }
            "VAEDecode" => {
                let z = match self.input(inp, "samples")? {
                    Val::Latent(z) | Val::MaskedLatent(z, _) => z,
                    _ => bail!("VAEDecode: not a latent"),
                };
                let Val::Vae(path) = self.input(inp, "vae")? else { bail!("VAEDecode: not a vae") };
                vec![Val::Image(self.vae_decode(&z, &path)?)]
            }
            "UpscaleModelLoader" => {
                let name = self.text_arg(inp, "model_name")?;
                vec![Val::UpscaleModel(self.model_path("upscale_models", &name)?)]
            }
            "ImageUpscaleWithModel" => {
                let Val::UpscaleModel(path) = self.input(inp, "upscale_model")? else {
                    bail!("ImageUpscaleWithModel: not an upscale model")
                };
                let Val::Image(img) = self.input(inp, "image")? else { bail!("ImageUpscaleWithModel: not an image") };
                // a picture ComfyUI holds as NHWC memory runs the whole model channels-last
                let nhwc = self.image_nhwc(inp, "image")?;
                let t = Instant::now();
                // ImageUpscaleWithModel's own estimate: the model and 384 times a 512-pixel tile's
                // f32 input times the scale
                if self.esrgan.as_ref().is_some_and(|(p, _)| p != &path) {
                    self.esrgan = None;
                }
                self.make_room(UPSCALE_NEED, Keep::Esrgan)?;
                let model = match &self.esrgan {
                    Some((p, m)) if p == &path => m.clone(),
                    _ => {
                        self.esrgan = None;
                        let m = Rc::new(Esrgan::load(&path, &self.dev)?);
                        self.esrgan = Some((path.clone(), m.clone()));
                        m
                    }
                };
                let run = |model: &Esrgan| -> Result<Tensor> {
                    let mut outs = vec![];
                    for i in 0..img.dim(0)? {
                        outs.push(model.upscale(&img.narrow(0, i, 1)?, nhwc)?);
                    }
                    Ok(Tensor::cat(&outs, 0)?)
                };
                let out = match run(&model) {
                    Err(e) if ltx_nodes::is_oom(&e) && self.room_after_oom(Keep::Esrgan)? => run(&model)?,
                    r => r?,
                };
                self.dev.synchronize()?;
                self.vram();
                self.timings.push((format!("upscale x{}", model.scale), t.elapsed().as_secs_f64()));
                vec![Val::Image(out)]
            }
            "ImageScale" => {
                let method = self.text_arg(inp, "upscale_method")?;
                let w = self.int_arg(inp, "width")?;
                let h = self.int_arg(inp, "height")?;
                let crop = self.text_arg(inp, "crop")?;
                let Val::Image(img) = self.input(inp, "image")? else { bail!("ImageScale: not an image") };
                if !matches!(method.as_str(), "lanczos" | "bilinear") || crop != "disabled" || w == 0 || h == 0 {
                    bail!("ImageScale: only lanczos or bilinear to an explicit size, crop disabled, is implemented (got {method}, {w}x{h}, crop {crop})");
                }
                let t = Instant::now();
                let (b, _, ih, iw) = img.dims4()?;
                let data = img.permute((0, 2, 3, 1))?.flatten_all()?.to_vec1::<f32>()?;
                let mut out = Vec::with_capacity(b * h * w * 3);
                for i in 0..b {
                    let one = &data[i * ih * iw * 3..(i + 1) * ih * iw * 3];
                    if method == "bilinear" {
                        out.extend(crate::resample::resize_bilinear(one, iw, ih, w, h));
                    } else {
                        // np.clip(255. * image, 0, 255).astype(np.uint8), then PIL's resize
                        let px: Vec<u8> = one.iter().map(|&x| (255.0f32 * x).clamp(0.0, 255.0) as u8).collect();
                        let r = crate::resample::resize_lanczos(&px, iw, ih, w, h);
                        out.extend(r.into_iter().map(|v| v as f32 / 255.0));
                    }
                }
                let t2 = Tensor::from_vec(out, (b, h, w, 3), &self.dev)?.permute((0, 3, 1, 2))?.contiguous()?;
                self.timings.push((format!("{method} resize"), t.elapsed().as_secs_f64()));
                vec![Val::Image(t2)]
            }
            "VAEEncode" => {
                let Val::Image(img) = self.input(inp, "pixels")? else { bail!("VAEEncode: not an image") };
                let Val::Vae(path) = self.input(inp, "vae")? else { bail!("VAEEncode: not a vae") };
                // a picture ComfyUI holds as NHWC memory keeps the encoder's residual stream channels-last
                let nhwc = self.image_nhwc(inp, "pixels")?;
                vec![Val::Latent(self.vae_encode(&img, nhwc, &path)?)]
            }
            "SaveImage" => {
                let prefix = self.text_arg(inp, "filename_prefix")?;
                let Val::Image(img) = self.input(inp, "images")? else { bail!("SaveImage: not an image") };
                let graph = image::ascii_json(&Value::Object(self.graph.clone()));
                let saved = image::save_png(&img, &self.out_dir, &prefix, &graph, &self.pnginfo)?;
                self.saved.extend(saved);
                vec![]
            }
            other => match self.anima_node(other, inp)?.map_or_else(|| self.ltx_node(other, inp), |v| Ok(Some(v)))? {
                Some(out) => out,
                None => match self.util_node(other, inp)? {
                    Some(out) => out,
                    None => match self.cond_node(other, inp)? {
                        Some(out) => out,
                        None => match self.impact_node(other, inp)? {
                            Some(out) => out,
                            None => match self.tac_node(other, inp)? {
                                Some(out) => out,
                                None => bail!("node type {other} (node {id}) is not implemented yet"),
                            },
                        },
                    },
                },
            },
        })
    }

    fn encode(&mut self, c: &Stack, layer: i64, text: &str) -> Result<Cond> {
        // SetClipHooks clones the CLIP with disable_dynamic, whose patcher loads a fresh copy of the
        // text encoders and patches the hooks into it, but CLIP.clone keeps the shared
        // cond_stage_model, and that is what encodes. A hooked CLIP's prompt runs the checkpoint's
        // text encoders as the last unhooked encode left them (its LoRAs, settled), unpatched if
        // none has run; the hooks never reach the text (the reference ComfyUI server, 2026-10-15)
        let c = if c.text_hooked {
            self.clip_prepared.get(&c.ckpt).cloned().unwrap_or_else(|| Stack::new(c.ckpt.clone()))
        } else {
            self.clip_prepared.insert(c.ckpt.clone(), c.clone());
            c.clone()
        };
        let c = &c;
        let key = stack_key(c, "clip");
        if self.clip.as_ref().map(|(k, _)| k != &key).unwrap_or(true) {
            self.clip = None;
            self.make_room(CLIP_NEED, Keep::Nothing)?;
            let t = Instant::now();
            let file = self.file(&c.ckpt)?;
            let mut loras = vec![];
            for (p, w) in &c.loras {
                loras.push((self.file(p)?, *w));
            }
            let dev = self.dev.clone();
            let clip = {
                let src_l = Source {
                    file: &file,
                    map: model::clip_l_map(),
                    loras: loras.clone(),
                    lora_prefix: "lora_te1_",
                    compute: DType::F32,
                    dtype: DType::F16,
                    dev: &dev,
                    patching: Default::default(),
                };
                let src_g = Source {
                    file: &file,
                    map: model::clip_g_map(),
                    loras: loras.clone(),
                    lora_prefix: "lora_te2_",
                    compute: DType::F32,
                    dtype: DType::F16,
                    dev: &dev,
                    patching: Default::default(),
                };
                SdxlClip::load(&src_l, &src_g)?
            };
            self.timings.push(("load text encoders".into(), t.elapsed().as_secs_f64()));
            // encoders that have encoded before stay settled in ComfyUI, which keeps them loaded
            // where rustyui let them go for the UNet or the VAE (a TiledAttentionCouple at the
            // hires size encodes again after the first sampler)
            let mut clip = clip;
            if self.clip_settled.contains(&key) {
                clip.settle_now()?;
            }
            for (p, _) in &c.loras {
                let unused = self.file(p)?.unused_modules();
                let te: Vec<&String> = unused.iter().filter(|u| u.starts_with("lora_te")).collect();
                if !te.is_empty() {
                    eprintln!("[engine] {}: {} text encoder LoRA modules matched no weight (first: {})", p.display(), te.len(), te[0]);
                }
            }
            self.clip = Some((key.clone(), clip));
        }
        let t = Instant::now();
        let clip = &mut self.clip.as_mut().unwrap().1;
        let cond = clip.encode(text, layer, &mut self.embs, &self.dev)?;
        self.clip_settled.insert(key);
        self.timings.push(("encode prompt".into(), t.elapsed().as_secs_f64()));
        Ok(cond)
    }

    fn ksampler(&mut self, inp: &serde_json::Map<String, Value>) -> Result<(Tensor, Option<Tensor>)> {
        // each widget its literal or what it is linked to (a primitive, a seed node)
        let seed = self.arg(inp, "seed")?.as_u64().context("seed is not a whole number")?;
        let steps = self.int_arg(inp, "steps")?;
        let cfg = self.float_arg(inp, "cfg")?;
        let sampler = self.text_arg(inp, "sampler_name")?;
        let scheduler = self.text_arg(inp, "scheduler")?;
        let denoise = if inp.contains_key("denoise") { self.float_arg(inp, "denoise")? } else { 1.0 };
        let Val::Cond(pos) = self.input(inp, "positive")? else { bail!("KSampler: positive is not conditioning") };
        let Val::Cond(neg) = self.input(inp, "negative")? else { bail!("KSampler: negative is not conditioning") };
        let (latent, noise_mask) = match self.input(inp, "latent_image")? {
            Val::Latent(z) => (z, None),
            Val::MaskedLatent(z, m) => (z, Some(m)),
            _ => bail!("KSampler: latent_image is not a latent"),
        };
        let Val::Model(m) = self.input(inp, "model")? else { bail!("KSampler: model is not a model") };
        // common_ksampler leaves the noise mask on the CPU for prepare_mask
        let spec = SampleSpec {
            model: &m,
            seed,
            steps,
            cfg,
            sampler: &sampler,
            scheduler: &scheduler,
            denoise,
            pos: &pos,
            neg: &neg,
            mask: noise_mask.as_ref().map(|t| (t, false)),
        };
        let (out, _) = self.sample_sdxl(&spec, &latent)?;
        Ok((out, noise_mask))
    }

    /// common_ksampler from the model on: the schedule, the noise from the seed, CFGGuider's
    /// conditioning (one prompt a side the plain CFG path, anything else _calc_cond_batch), the
    /// model loaded with its hooks and ControlNets, KSamplerX0Inpaint around it for a noise mask,
    /// then the sampler. Returns the samples after process_latent_out (on the card) and the
    /// denoised latent the sampler's callback last saw (before it).
    fn sample_sdxl(&mut self, s: &SampleSpec, latent: &Tensor) -> Result<(Tensor, Tensor)> {
        let (m, pos, neg) = (s.model, s.pos, s.neg);
        let (seed, steps, cfg, denoise) = (s.seed, s.steps, s.cfg, s.denoise);
        let (sampler, scheduler) = (s.sampler.to_string(), s.scheduler.to_string());
        anyhow::ensure!(!pos.is_empty() && !neg.is_empty(), "KSampler: a side has no conditioning");
        let all: Vec<&Cond> = pos.iter().chain(neg.iter()).collect();
        anyhow::ensure!(!all.iter().any(|c| c.props.area_to_bounds), "set_cond_area \"mask bounds\" is not implemented yet");
        // one prompt a side with nothing set runs the plain CFG path, anything else ComfyUI's
        // _calc_cond_batch (MultiCondDenoiser)
        let plain = pos.len() == 1 && neg.len() == 1 && pos[0].props.is_plain() && neg[0].props.is_plain();
        // CFGGuider.inner_set_conds: a conditioning with hooks swaps the model for the classic
        // ModelPatcher's copy of it (DynamicVRAM does not run hooks)
        let classic = all.iter().any(|c| c.props.hooks.is_some());
        self.load_unet(m, classic)?;
        {
            // the sampler's activations: about 64 KiB a latent pixel for each of the two CFG
            // halves, plus a GiB (measured peaks are about 52 KiB at 1024x1792), beside the UNet
            let (b, _, h, w) = latent.dims4()?;
            self.make_room((1 << 30) + 2 * 65536 * b * h * w, Keep::Unet)?;
        }
        // the hook groups in the order _calc_cond_batch meets them (the positive's, then the
        // negative's), each with the LoRAs its weights add to the base (none: the base itself)
        let mut groups: Vec<(Option<Rc<HookGroup>>, Vec<(PathBuf, f64)>)> = vec![];
        for c in &all {
            if !groups.iter().any(|(g, _)| same_hooks(g, &c.props.hooks)) {
                groups.push((c.props.hooks.clone(), c.props.hooks.as_ref().map(|h| h.model_loras()).unwrap_or_default()));
            }
        }
        let ckpt_file = self.file(&m.ckpt)?;
        let mut hook_files: Vec<Vec<(Rc<File>, f64)>> = vec![];
        for (_, loras) in &groups {
            let mut f = vec![];
            for (p, s) in loras {
                f.push((self.file(p)?, *s));
            }
            hook_files.push(f);
        }
        let hooked = hook_files.iter().any(|f| !f.is_empty());
        // the ControlNets the prompts use: each control and every one it was applied over
        let mut applies: Vec<Rc<ControlApply>> = vec![];
        for c in &all {
            let mut a = c.props.control.clone();
            while let Some(ca) = a {
                if !applies.iter().any(|x| Rc::ptr_eq(x, &ca)) {
                    applies.push(ca.clone());
                }
                a = ca.prev.clone();
            }
        }
        for ca in &applies {
            if !self.unet.as_ref().context("KSampler: no UNet loaded")?.controlnets.iter().any(|(p, _)| p == &ca.net) {
                let t = Instant::now();
                let f = self.file(&ca.net)?;
                let net = ControlNet::load(&f, &self.dev)?;
                self.timings.push(("load controlnet".into(), t.elapsed().as_secs_f64()));
                self.unet.as_mut().context("KSampler: no UNet loaded")?.controlnets.push((ca.net.clone(), net));
            }
        }

        let disc = Discrete::sdxl();
        let skip_uncond = skips_uncond(cfg, &sampler);
        // the schedule where ComfyUI has the model's sigma table when it makes the sampler: on the
        // card once a sampler has loaded this checkpoint's model (Discrete::schedule)
        let card = self.sigmas_on_card.contains(&m.ckpt).then_some(&self.dev);
        let sigmas = if denoise >= 0.9999 {
            disc.schedule(&scheduler, steps, card)?
        } else {
            let total = (steps as f64 / denoise) as usize;
            let all = disc.schedule(&scheduler, total, card)?;
            all[all.len() - (steps + 1)..].to_vec()
        };
        let (b, c, h, w) = latent.dims4()?;
        let noise = Tensor::from_vec(rng::torch_randn_cpu(seed, b * c * h * w), (b, c, h, w), &self.dev)?;
        let nonzero = latent.abs()?.sum_all()?.to_scalar::<f32>()? > 0.0;
        let latent_in = if nonzero { latent.affine(SDXL_SCALE, 0.0)? } else { latent.clone() };
        let s0 = sigmas[0];
        let max_denoise = ((disc.sigma_max() - s0).abs() <= 1e-5 * disc.sigma_max().abs().max(s0.abs())) || s0 > disc.sigma_max();
        let scale = if max_denoise { (1.0f32 + s0 * s0).sqrt() } else { s0 };
        let x = noise.broadcast_mul(&Tensor::new(scale, &self.dev)?)?.broadcast_add(&latent_in)?;

        let t = Instant::now();
        // the hook groups' weights stay on the card beside the base, unless they ran it out of
        // memory at this size or a smaller one: then each group's are made for its own calls
        let size = b * h * w;
        // (RUSTYUI_HOOKS_LEAN=1 makes them per call from the start, to test that path)
        let mut lean = hooked && (self.hooks_lean.is_some_and(|n| size >= n) || std::env::var_os("RUSTYUI_HOOKS_LEAN").is_some());
        let out = loop {
            let mut resident: Vec<Option<usize>> = vec![];
            let mut made = Ok(());
            if !lean {
                for (_, loras) in &groups {
                    if loras.is_empty() {
                        resident.push(None);
                        continue;
                    }
                    match self.hooked_unet(m, loras) {
                        Ok(i) => resident.push(Some(i)),
                        Err(e) => {
                            made = Err(e);
                            break;
                        }
                    }
                }
            }
            let result = made.and_then(|_| -> Result<(Tensor, Tensor)> {
                let loaded = self.unet.as_ref().context("KSampler: no UNet loaded")?;
                let den: Box<dyn Denoise + '_> = if plain {
                    Box::new(
                        CfgDenoiser::new(&loaded.base, &disc, &pos[0], &neg[0], cfg as f32, skip_uncond, (w * 8, h * 8), &self.dev)?
                            .with_couple(m.couple.clone()),
                    )
                } else {
                    let weights = groups
                        .iter()
                        .enumerate()
                        .map(|(k, (g, _))| {
                            let wts = match resident.get(k).copied().flatten() {
                                Some(i) => Weights::Resident(&loaded.hooked[i].1),
                                None if hook_files[k].is_empty() || !lean => Weights::Resident(&loaded.base),
                                None => Weights::Rebuilt { file: ckpt_file.clone(), hooks: hook_files[k].clone(), base: &loaded.weights },
                            };
                            (g.clone(), wts)
                        })
                        .collect();
                    let nets = loaded.controlnets.iter().map(|(p, n)| (p.clone(), n)).collect();
                    Box::new(
                        MultiCondDenoiser::new(
                            weights,
                            &disc,
                            pos,
                            neg,
                            cfg as f32,
                            skip_uncond,
                            (w * 8, h * 8),
                            &[b, c, h, w],
                            nets,
                            &applies,
                            &self.dev,
                        )?
                        .with_couple(m.couple.clone()),
                    )
                };
                // a noise mask: prepare_mask stretches it to the latent, and KSamplerX0Inpaint holds
                // the kept part to the clean latent around each model call
                let den: Box<dyn Denoise + '_> = match s.mask {
                    Some((nm, on_card)) => {
                        let mask = sampling::reshape_mask(nm, &[b, c, h, w], on_card)?.to_dtype(DType::F32)?;
                        let differential = match m.differential {
                            Some(strength) => Some(Differential::new(&disc, &sigmas, strength, &self.dev)?),
                            None => None,
                        };
                        Box::new(InpaintDenoiser {
                            keep: mask.affine(-1.0, 1.0)?,
                            mask,
                            latent: latent_in.clone(),
                            noise: noise.clone(),
                            inner: den,
                            differential,
                        })
                    }
                    None => den,
                };
                let mut rec = Recorder { inner: den, last: None };
                let mut last = Instant::now();
                let mut progress = |i: usize, n: usize| -> Result<()> {
                    crate::progress::step(i, n)?;
                    if i == 1 || i == n || last.elapsed().as_secs_f64() > 5.0 {
                        eprintln!("[engine] step {i}/{n}  {:.1}s", t.elapsed().as_secs_f64());
                        last = Instant::now();
                    }
                    Ok(())
                };
                let out = sampling::sample(&sampler, &mut rec, x.clone(), &sigmas, seed, self.sms, &self.dev, &mut progress)?;
                Ok((out, rec.last.context("the sampler never called the model")?))
            });
            match result {
                // beside the text encoders or a VAE kept from before: they go and it runs again
                Err(e) if ltx_nodes::is_oom(&e) && self.room_after_oom(Keep::Unet)? => {}
                Err(e) if hooked && !lean && ltx_nodes::is_oom(&e) => {
                    eprintln!("[engine] the hook groups' weights do not fit beside the sampler at this size: making each group's for its own calls");
                    self.hooks_lean = Some(self.hooks_lean.map_or(size, |n| n.min(size)));
                    if let Some(u) = self.unet.as_mut() {
                        u.hooked.clear();
                    }
                    self.dev.synchronize()?;
                    lean = true;
                }
                r => break r?,
            }
        };
        self.dev.synchronize()?;
        self.vram();
        // DynamicVRAM moved the model's buffers to the card as it loaded it, and keeps them there
        // until it unloads the model (rustyui's /free, which ends the engine); a hooked sampler ran
        // the classic copy instead and left this one where it was
        if !classic {
            self.sigmas_on_card.insert(m.ckpt.clone());
        }
        let secs = t.elapsed().as_secs_f64();
        self.timings.push((format!("sample {steps} steps ({:.2} s/step)", secs / steps as f64), secs));
        // process_latent_out
        let (out, last) = out;
        Ok((out.affine(1.0 / SDXL_SCALE, 0.0)?, last))
    }

    /// VAE.encode of an IMAGE (b, 3, h, w): vae_encode_crop_pixels (each side centre-cropped to
    /// a multiple of 8), then the encoder, its residual stream channels-last when ComfyUI holds
    /// the picture as NHWC memory
    fn vae_encode(&mut self, img: &Tensor, nhwc: bool, path: &Path) -> Result<Tensor> {
        let t = Instant::now();
        let (_, _, h, w) = img.dims4()?;
        let (ch, cw) = (h / 8 * 8, w / 8 * 8);
        let img = img.narrow(2, (h % 8) / 2, ch)?.narrow(3, (w % 8) / 2, cw)?.contiguous()?;
        // VAE.memory_used_encode: 1767 bytes a pixel for the SDXL VAE in bf16, ComfyUI's estimate
        let (b, _, _, _) = img.dims4()?;
        if self.vae_enc.as_ref().is_some_and(|(p, _)| p != path) {
            self.vae_enc = None;
        }
        self.make_room(1767 * 2 * b * ch * cw, Keep::VaeEnc)?;
        let enc = match &self.vae_enc {
            Some((p, e)) if p == path => e.clone(),
            _ => {
                self.vae_enc = None;
                let file = self.file(path)?;
                let dev = self.dev.clone();
                let src = Source {
                    file: &file,
                    map: model::vae_map(),
                    loras: vec![],
                    lora_prefix: "",
                    compute: DType::BF16,
                    dtype: DType::BF16,
                    dev: &dev,
                    patching: Default::default(),
                };
                let e = Rc::new(Encoder::load(&src)?);
                self.vae_enc = Some((path.to_path_buf(), e.clone()));
                e
            }
        };
        let z = match enc.encode(&img, nhwc) {
            Err(e) if ltx_nodes::is_oom(&e) && self.room_after_oom(Keep::VaeEnc)? => enc.encode(&img, nhwc)?,
            r => r?,
        };
        self.dev.synchronize()?;
        self.vram();
        self.timings.push(("vae encode".into(), t.elapsed().as_secs_f64()));
        Ok(z)
    }

    /// VAE.decode of an SDXL latent to an IMAGE (b, 3, h, w)
    fn vae_decode(&mut self, z: &Tensor, path: &Path) -> Result<Tensor> {
        let t = Instant::now();
        // VAE.memory_used_decode: 2178 * 64 bytes a latent pixel for the SDXL VAE in bf16
        let (b, _, h, w) = z.dims4()?;
        if self.vae_dec.as_ref().is_some_and(|(p, _)| p != path) {
            self.vae_dec = None;
        }
        self.make_room(2178 * 64 * 2 * b * h * w, Keep::VaeDec)?;
        let dec = match &self.vae_dec {
            Some((p, d)) if p == path => d.clone(),
            _ => {
                self.vae_dec = None;
                let file = self.file(path)?;
                let dev = self.dev.clone();
                let src = Source {
                    file: &file,
                    map: model::vae_map(),
                    loras: vec![],
                    lora_prefix: "",
                    compute: DType::BF16,
                    dtype: DType::BF16,
                    dev: &dev,
                    patching: Default::default(),
                };
                let d = Rc::new(Decoder::load(&src)?);
                self.vae_dec = Some((path.to_path_buf(), d.clone()));
                d
            }
        };
        let img = match dec.decode(z) {
            Err(e) if ltx_nodes::is_oom(&e) && self.room_after_oom(Keep::VaeDec)? => dec.decode(z)?,
            r => r?,
        };
        self.dev.synchronize()?;
        self.vram();
        self.timings.push(("vae decode".into(), t.elapsed().as_secs_f64()));
        Ok(img)
    }

    /// This model stack's UNet on the card: DynamicVRAM's fold of its LoRAs, or with `classic` the
    /// classic ModelPatcher's, keeping its weights by name for hooked loads
    fn load_unet(&mut self, m: &Stack, classic: bool) -> Result<()> {
        let key = stack_key(m, if classic { "unet-classic" } else { "unet" });
        if self.unet.as_ref().is_some_and(|u| u.key == key) {
            return Ok(());
        }
        // the UNet goes beside the text encoders when there is room for it
        self.unet = None;
        self.make_room(UNET_NEED, Keep::Nothing)?;
        let t = Instant::now();
        let file = self.file(&m.ckpt)?;
        let mut loras = vec![];
        for (p, w) in &m.loras {
            loras.push((self.file(p)?, *w));
        }
        let dev = self.dev.clone();
        let weights = std::cell::RefCell::new(HashMap::new());
        let base = {
            let patching = model::Patching { classic, record: classic.then_some(&weights), ..Default::default() };
            let src = Source {
                file: &file,
                map: model::unet_map(),
                loras,
                lora_prefix: "lora_unet_",
                compute: DType::F16,
                dtype: DType::F16,
                dev: &dev,
                patching,
            };
            Unet::load(&src)?
        };
        self.dev.synchronize()?;
        self.vram();
        self.timings.push((format!("load unet (+loras{})", if classic { ", classic patching" } else { "" }), t.elapsed().as_secs_f64()));
        self.unet = Some(LoadedUnet { key, base, weights, hooked: vec![], controlnets: vec![] });
        for (p, _) in &m.loras {
            self.report_unused(p)?;
        }
        Ok(())
    }

    /// The loaded classic UNet with a hook group's LoRAs patched over it (ModelPatcher.patch_hooks;
    /// MaxSpeed keeps them on the card as ComfyUI's cached_hook_patches do), sharing every weight
    /// they leave alone; its index in LoadedUnet::hooked
    fn hooked_unet(&mut self, m: &Stack, loras: &[(PathBuf, f64)]) -> Result<usize> {
        let key = loras.iter().map(|(p, s)| format!("{}@{s}", p.display())).collect::<Vec<_>>().join("|");
        if let Some(i) = self.unet.as_ref().context("hooks without a UNet")?.hooked.iter().position(|(k, _)| k == &key) {
            return Ok(i);
        }
        let t = Instant::now();
        let file = self.file(&m.ckpt)?;
        let mut files = vec![];
        for (p, s) in loras {
            files.push((self.file(p)?, *s));
        }
        let dev = self.dev.clone();
        let unet = {
            let loaded = self.unet.as_ref().context("hooks without a UNet")?;
            let patching = model::Patching { hooks: files, base: Some(&loaded.weights), ..Default::default() };
            let src = Source {
                file: &file,
                map: model::unet_map(),
                loras: vec![],
                lora_prefix: "lora_unet_",
                compute: DType::F16,
                dtype: DType::F16,
                dev: &dev,
                patching,
            };
            Unet::load(&src)?
        };
        self.dev.synchronize()?;
        self.vram();
        self.timings.push(("patch hook loras".into(), t.elapsed().as_secs_f64()));
        for (p, _) in loras {
            self.report_unused(p)?;
        }
        let loaded = self.unet.as_mut().context("hooks without a UNet")?;
        loaded.hooked.push((key, unet));
        Ok(loaded.hooked.len() - 1)
    }

    fn report_unused(&mut self, p: &Path) -> Result<()> {
        let f = self.file(p)?;
        let unused = f.unused_modules();
        let unet_unused: Vec<&String> = unused.iter().filter(|u| u.starts_with("lora_unet_")).collect();
        if !unet_unused.is_empty() {
            eprintln!("[engine] {}: {} UNet LoRA modules matched no weight (first: {})", p.display(), unet_unused.len(), unet_unused[0]);
        }
        Ok(())
    }
}

/// Whether two conditionings' hooks are the same group (ComfyUI keys by the object)
fn same_hooks(a: &Option<Rc<HookGroup>>, b: &Option<Rc<HookGroup>>) -> bool {
    match (a, b) {
        (None, None) => true,
        (Some(x), Some(y)) => Rc::ptr_eq(x, y),
        _ => false,
    }
}

/// What one sampler run takes: KSampler's widgets, the model and prompts, and a noise mask with
/// whether ComfyUI has it on the card when prepare_mask stretches it to the latent (Impact's
/// samplers move it there first; common_ksampler leaves it on the CPU)
struct SampleSpec<'a> {
    model: &'a Stack,
    seed: u64,
    steps: usize,
    cfg: f64,
    sampler: &'a str,
    scheduler: &'a str,
    denoise: f64,
    pos: &'a Rc<Vec<Cond>>,
    neg: &'a Rc<Vec<Cond>>,
    mask: Option<(&'a Tensor, bool)>,
}

/// DifferentialDiffusion's denoise_mask_function (comfy_extras/nodes_differential_diffusion.py):
/// each model call masks where the noise mask reaches a threshold, the call's timestep's place
/// between the first step's and the last's (sigma_min's when the last sigma is below it), the
/// timesteps from the model's sigma table, on the card while it samples (CUDA's logf); with a
/// strength under 1 (and not 0) blended back with the noise mask
struct Differential<'a> {
    disc: &'a Discrete,
    ts_from: i64,
    ts_to: i64,
    strength: f64,
    dev: &'a Device,
}

impl<'a> Differential<'a> {
    fn new(disc: &'a Discrete, sigmas: &[f32], strength: f64, dev: &'a Device) -> Result<Self> {
        let (Some(&first), Some(&last)) = (sigmas.first(), sigmas.last()) else { bail!("DifferentialDiffusion: no sigmas") };
        let to = if last > disc.sigma_min() { last } else { disc.sigma_min() };
        Ok(Differential { disc, ts_from: disc.timestep_on_card(first, dev)?, ts_to: disc.timestep_on_card(to, dev)?, strength, dev })
    }

    /// the mask a call at `sigma` samples with
    fn mask(&self, mask: &Tensor, sigma: f32) -> Result<Tensor> {
        let current = self.disc.timestep_on_card(sigma, self.dev)?;
        // int64 timesteps, the quotient in float32
        let threshold = (current - self.ts_to) as f32 / (self.ts_from - self.ts_to) as f32;
        let binary = mask.ge(threshold as f64)?.to_dtype(DType::F32)?;
        if self.strength != 0.0 && self.strength < 1.0 {
            Ok((binary.affine(self.strength as f32 as f64, 0.0)? + mask.affine((1.0 - self.strength) as f32 as f64, 0.0)?)?)
        } else {
            Ok(binary)
        }
    }
}

/// Keeps the last denoised latent the sampler was handed: what a k-diffusion sampler's callback
/// last saw as x0 (every sampler the detailer allows calls the model last for its last step)
struct Recorder<'a> {
    inner: Box<dyn Denoise + 'a>,
    last: Option<Tensor>,
}

impl Denoise for Recorder<'_> {
    fn uncond_denoised(&self) -> Option<Tensor> {
        self.inner.uncond_denoised()
    }

    fn denoised(&mut self, x: &Tensor, sigma: f32) -> Result<Tensor> {
        let d = self.inner.denoised(x, sigma)?;
        self.last = Some(d.clone());
        Ok(d)
    }
}

/// KSamplerX0Inpaint around the CFG denoiser for an eps model: going in, x * mask + (noise *
/// sigma + latent) * (1 - mask) (scale_latent_inpaint is noise_scaling, the sampler's own noise);
/// coming out, out * mask + latent * (1 - mask); `latent` the clean latent after process_latent_in;
/// the mask DifferentialDiffusion's for the call when the model has it
struct InpaintDenoiser<'a> {
    inner: Box<dyn Denoise + 'a>,
    mask: Tensor,
    keep: Tensor,
    latent: Tensor,
    noise: Tensor,
    differential: Option<Differential<'a>>,
}

impl Denoise for InpaintDenoiser<'_> {
    /// a post-CFG function runs inside the model call, before KSamplerX0Inpaint's blend
    fn uncond_denoised(&self) -> Option<Tensor> {
        self.inner.uncond_denoised()
    }

    fn denoised(&mut self, x: &Tensor, sigma: f32) -> Result<Tensor> {
        let made;
        let (mask, keep) = match &self.differential {
            None => (&self.mask, &self.keep),
            Some(d) => {
                let m = d.mask(&self.mask, sigma)?;
                let k = m.affine(-1.0, 1.0)?;
                made = (m, k);
                (&made.0, &made.1)
            }
        };
        let s = Tensor::new(sigma, x.device())?;
        let scaled = (self.noise.broadcast_mul(&s)? + &self.latent)?;
        let x = ((x * mask)? + (scaled * keep)?)?;
        let out = self.inner.denoised(&x, sigma)?;
        Ok(((out * mask)? + (&self.latent * keep)?)?)
    }
}

/// x, sigma -> CFG-combined denoised latent, cond and uncond batched when their lengths allow.
struct CfgDenoiser<'a> {
    unet: &'a Unet,
    disc: &'a Discrete,
    ctx: Vec<Tensor>, // [cond, uncond] contexts, f16, padded by repetition when batched
    y: Vec<Tensor>,
    batched: bool,
    /// cfg 1: only the positive runs (skips_uncond)
    skip_uncond: bool,
    cfg: f32,
    dev: &'a Device,
    /// the last call's uncond prediction as a denoised latent (Denoise::uncond_denoised)
    last_uncond: Option<Tensor>,
    /// TiledAttentionCouple's patch on the model
    couple: Option<Rc<crate::unet::Couple>>,
}

/// sampling_function's cfg 1 optimization: at a cfg math.isclose to 1 ComfyUI passes no negative to
/// calc_cond_batch, so the UNet runs on the positive alone and the negative's prediction is the zeros
/// its empty side divides to; the CFG++ samplers' post-cfg functions turn it off
/// (disable_cfg1_optimization)
fn skips_uncond(cfg: f64, sampler: &str) -> bool {
    (cfg - 1.0).abs() <= 1e-9 * cfg.abs().max(1.0) && !sampler.ends_with("_cfg_pp")
}

/// adm, for unet::debug_unet
pub(crate) fn adm_for_debug(pooled: &Tensor, width: usize, height: usize, dev: &Device) -> Result<Tensor> {
    adm(pooled, width, height, dev)
}

fn adm(pooled: &Tensor, width: usize, height: usize, dev: &Device) -> Result<Tensor> {
    let sizes = [height as f32, width as f32, 0.0, 0.0, height as f32, width as f32];
    let mut flat = vec![];
    for s in sizes {
        flat.extend(timestep_embedding(&[s], 256));
    }
    let flat = Tensor::from_vec(flat, (1, 6 * 256), dev)?;
    Ok(Tensor::cat(&[pooled, &flat], 1)?.to_dtype(DType::F16)?)
}

fn lcm(a: usize, b: usize) -> usize {
    let (mut x, mut y) = (a, b);
    while y != 0 {
        (x, y) = (y, x % y);
    }
    a / x * b
}

impl<'a> CfgDenoiser<'a> {
    #[allow(clippy::too_many_arguments)]
    fn new(
        unet: &'a Unet,
        disc: &'a Discrete,
        pos: &Cond,
        neg: &Cond,
        cfg: f32,
        skip_uncond: bool,
        size: (usize, usize),
        dev: &'a Device,
    ) -> Result<Self> {
        let (lc, lu) = (pos.cond.dim(1)?, neg.cond.dim(1)?);
        let m = lcm(lc, lu);
        // ComfyUI's limit on padding by repetition is 4x; RUSTYUI_BATCH_CFG=1 lifts it (same maths, one pass)
        let batched = !skip_uncond && (m / lc.min(lu) <= 4 || std::env::var_os("RUSTYUI_BATCH_CFG").is_some());
        let fit =
            |t: &Tensor, len: usize| -> Result<Tensor> { Ok(if batched && len < m { t.repeat((1, m / len, 1))? } else { t.clone() }) };
        Ok(CfgDenoiser {
            unet,
            disc,
            ctx: vec![fit(&pos.cond, lc)?.to_dtype(DType::F16)?, fit(&neg.cond, lu)?.to_dtype(DType::F16)?],
            y: vec![adm(&pos.pooled, size.0, size.1, dev)?, adm(&neg.pooled, size.0, size.1, dev)?],
            batched,
            skip_uncond,
            cfg,
            dev,
            last_uncond: None,
            couple: None,
        })
    }

    fn with_couple(mut self, couple: Option<Rc<crate::unet::Couple>>) -> Self {
        self.couple = couple;
        self
    }

    /// one UNet call, its batch's chunks the given sides (cond_or_uncond: 0 the positive)
    fn call(&self, x: &Tensor, t: &[f32], ctx: &Tensor, y: &Tensor, sides: &[usize]) -> Result<Tensor> {
        match &self.couple {
            None => self.unet.forward(x, t, ctx, y, self.dev),
            Some(c) => {
                self.unet.forward_couple(x, t, ctx, y, None, Some(&crate::unet::CoupleCall { couple: c, sides: sides.to_vec() }), self.dev)
            }
        }
    }
}

impl Denoise for CfgDenoiser<'_> {
    fn denoised(&mut self, x: &Tensor, sigma: f32) -> Result<Tensor> {
        let div = Tensor::new((sigma * sigma + 1.0).sqrt(), self.dev)?;
        let xc = x.broadcast_div(&div)?.to_dtype(DType::F16)?;
        let t = self.disc.timestep(sigma);
        let sig = Tensor::new(sigma, self.dev)?;
        // a batch of b latents: each conditioning repeated to b (comfy.utils.repeat_to_batch_size),
        // cond's b then uncond's b in one call when they batch
        let b = x.dim(0)?;
        let (ctx, y) = if b == 1 {
            (self.ctx.clone(), self.y.clone())
        } else {
            (
                self.ctx.iter().map(|c| c.repeat((b, 1, 1))).collect::<candle_core::Result<Vec<_>>>()?,
                self.y.iter().map(|y| y.repeat((b, 1))).collect::<candle_core::Result<Vec<_>>>()?,
            )
        };
        if self.skip_uncond {
            let cond_out = self.call(&xc, &vec![t; b], &ctx[0], &y[0], &[0])?;
            let cond = (x - cond_out.broadcast_mul(&sig)?)?;
            let uncond = x.zeros_like()?;
            let cfg = Tensor::new(self.cfg, self.dev)?;
            let out = (&uncond + (&cond - &uncond)?.broadcast_mul(&cfg)?)?;
            self.last_uncond = Some(uncond);
            return Ok(out);
        }
        let (cond_out, uncond_out) = if self.batched {
            // _calc_cond_batch reverses its list, so the batch is uncond's b, then cond's b
            let xb = Tensor::cat(&[&xc, &xc], 0)?;
            let ctx = Tensor::cat(&[&ctx[1], &ctx[0]], 0)?;
            let y = Tensor::cat(&[&y[1], &y[0]], 0)?;
            let out = self.call(&xb, &vec![t; 2 * b], &ctx, &y, &[1, 0])?;
            (out.narrow(0, b, b)?, out.narrow(0, 0, b)?)
        } else {
            (self.call(&xc, &vec![t; b], &ctx[0], &y[0], &[0])?, self.call(&xc, &vec![t; b], &ctx[1], &y[1], &[1])?)
        };
        let cond = (x - cond_out.broadcast_mul(&sig)?)?;
        let uncond = (x - uncond_out.broadcast_mul(&sig)?)?;
        let cfg = Tensor::new(self.cfg, self.dev)?;
        let out = (&uncond + (&cond - &uncond)?.broadcast_mul(&cfg)?)?;
        self.last_uncond = Some(uncond);
        Ok(out)
    }

    fn uncond_denoised(&self) -> Option<Tensor> {
        self.last_uncond.clone()
    }
}

/// A conditioning as _calc_cond_batch runs it: its context (f16), its adm, the side it counts on
/// (0 the positive, 1 the negative) and the mult its prediction counts with: its mask stretched to
/// the latent, times mask_strength, over every channel (None: ones)
struct Entry {
    ctx: Tensor,
    y: Tensor,
    side: usize,
    mult: Option<Tensor>,
    /// its ControlNetApplyAdvanced: prompts batch only with others holding the same one
    control: Option<Rc<ControlApply>>,
}

/// The weights a hook group's calls run with
enum Weights<'a> {
    Resident(&'a Unet),
    /// made for each call from the base's and let go after it, on a card without room to keep
    /// them (ComfyUI patches a group's in and out around its calls the same way when short)
    Rebuilt {
        file: Rc<File>,
        hooks: Vec<(Rc<File>, f64)>,
        base: &'a std::cell::RefCell<HashMap<String, Tensor>>,
    },
}

/// x, sigma -> the CFG-combined denoised latent from any number of conditionings a side, as
/// ComfyUI's _calc_cond_batch: grouped by their hooks, each group through its own weights; of a
/// group, the conditionings that batch with its first run in one call in reverse order; each
/// prediction times its mult is summed into its side, and each side divided by its summed mults
/// (which start at 1e-37). The batch is taken whole: ComfyUI shrinks it only when the card is short.
struct MultiCondDenoiser<'a> {
    groups: Vec<(Weights<'a>, Vec<Entry>)>,
    disc: &'a Discrete,
    cfg: f32,
    dev: &'a Device,
    last_uncond: Option<Tensor>,
    /// the ControlNets the prompts use, by file
    nets: Vec<(PathBuf, &'a ControlNet)>,
    /// every control (and the ones it was applied over) with its hint at this latent's size and
    /// the sigmas it acts between (percent_to_sigma of start and end)
    controls: Vec<(Rc<ControlApply>, Tensor, f32, f32)>,
    /// TiledAttentionCouple's patch on the model
    couple: Option<Rc<crate::unet::Couple>>,
}

/// Whether two prompts carry the same control (ComfyUI compares the objects)
fn same_control(a: &Option<Rc<ControlApply>>, b: &Option<Rc<ControlApply>>) -> bool {
    match (a, b) {
        (None, None) => true,
        (Some(x), Some(y)) => Rc::ptr_eq(x, y),
        _ => false,
    }
}

impl<'a> MultiCondDenoiser<'a> {
    #[allow(clippy::too_many_arguments)]
    fn new(
        groups: Vec<(Option<Rc<HookGroup>>, Weights<'a>)>,
        disc: &'a Discrete,
        pos: &[Cond],
        neg: &[Cond],
        cfg: f32,
        skip_uncond: bool,
        size: (usize, usize),
        latent: &[usize; 4],
        nets: Vec<(PathBuf, &'a ControlNet)>,
        applies: &[Rc<ControlApply>],
        dev: &'a Device,
    ) -> Result<Self> {
        let [b, c, h, w] = *latent;
        let keys: Vec<Option<Rc<HookGroup>>> = groups.iter().map(|(k, _)| k.clone()).collect();
        let mut out: Vec<(Weights<'a>, Vec<Entry>)> = groups.into_iter().map(|(_, u)| (u, vec![])).collect();
        // at cfg 1 the negative's prompts are not run at all (skips_uncond)
        for (side, conds) in [pos, if skip_uncond { &[][..] } else { neg }].into_iter().enumerate() {
            for cond in conds {
                let g = keys.iter().position(|k| same_hooks(k, &cond.props.hooks)).context("a conditioning's hooks have no weights")?;
                let mult = match &cond.props.mask {
                    None => None,
                    Some((m, strength)) => {
                        // resolve_areas_and_cond_masks_multidim: the mask on the card, stretched
                        // to the latent (bilinear, torch's kernel); get_area_and_mult: the latent's
                        // batch of it times mask_strength, repeated over batch and channels
                        let m = m.to_device(dev)?.to_dtype(DType::F32)?;
                        let m = if m.rank() == 2 { m.unsqueeze(0)? } else { m };
                        let (mb, mh, mw) = m.dims3()?;
                        let m = if (mh, mw) != (h, w) {
                            crate::kernels::torch_upsample_bilinear2d_f32(&m.unsqueeze(1)?, h, w, None)?.squeeze(1)?
                        } else {
                            m
                        };
                        let m = (m.narrow(0, 0, mb.min(b))? * *strength)?;
                        let mb = m.dim(0)?;
                        anyhow::ensure!(b % mb == 0, "a mask batch of {mb} does not fit a latent batch of {b}");
                        Some(m.unsqueeze(1)?.repeat((b / mb, c, 1, 1))?.contiguous()?)
                    }
                };
                out[g].1.push(Entry {
                    ctx: cond.cond.to_dtype(DType::F16)?,
                    y: adm(&cond.pooled, size.0, size.1, dev)?,
                    side,
                    mult,
                    control: cond.props.control.clone(),
                });
            }
        }
        // RUSTYUI_DUMP_MULTS=FILE: every masked conditioning's mult by group and entry, written to
        // FILE.<h>x<w>.safetensors (a check against torch's own interpolate)
        if let Some(path) = std::env::var_os("RUSTYUI_DUMP_MULTS") {
            let mut d = HashMap::new();
            for (g, (_, entries)) in out.iter().enumerate() {
                for (e, en) in entries.iter().enumerate() {
                    if let Some(m) = &en.mult {
                        d.insert(format!("g{g}.e{e}"), m.to_device(&Device::Cpu)?);
                    }
                }
            }
            candle_core::safetensors::save(&d, PathBuf::from(path).with_extension(format!("{h}x{w}.safetensors")))?;
        }
        // ControlNet.get_control's hint (common_upscale to the latent's size, once per sampler)
        // and pre_run's timestep range, with the model's sigma table on the card
        let mut controls = vec![];
        for ca in applies {
            let hint = crate::controlnet::prepare_hint(&ca.image, h, w, dev)?;
            let hi = crate::controlnet::percent_to_sigma(disc, ca.range.0, dev)?;
            let lo = crate::controlnet::percent_to_sigma(disc, ca.range.1, dev)?;
            controls.push((ca.clone(), hint, hi, lo));
        }
        Ok(MultiCondDenoiser { groups: out, disc, cfg, dev, last_uncond: None, nets, controls, couple: None })
    }

    fn with_couple(mut self, couple: Option<Rc<crate::unet::Couple>>) -> Self {
        self.couple = couple;
        self
    }

    /// ControlNet.get_control for a batch: the controls it was applied over first, then its own
    /// residuals if this sigma is in its range, each times its strength (in f16), the earlier
    /// ones' added (control_merge)
    #[allow(clippy::too_many_arguments)]
    fn control_for(
        &self,
        ca: &Rc<ControlApply>,
        x: &Tensor,
        t: &[f32],
        ctx: &Tensor,
        y: &Tensor,
        sigma: f32,
    ) -> Result<Option<crate::unet::Control>> {
        let prev = match &ca.prev {
            Some(p) => self.control_for(p, x, t, ctx, y, sigma)?,
            None => None,
        };
        let (_, hint, hi, lo) = self.controls.iter().find(|(c, ..)| Rc::ptr_eq(c, ca)).context("a control without its hint")?;
        if sigma > *hi || sigma < *lo {
            return Ok(prev);
        }
        let net = self.nets.iter().find(|(p, _)| p == &ca.net).map(|(_, n)| *n).context("a ControlNet that was not loaded")?;
        let c = net.forward(x, t, ctx, y, hint, ca.nhwc, &ca.types, self.dev)?;
        let mut output = c.output.iter().map(|o| crate::controlnet::scale_f16(o, ca.strength)).collect::<Result<Vec<_>>>()?;
        let mut middle = crate::controlnet::scale_f16(&c.middle, ca.strength)?;
        if let Some(p) = prev {
            for (o, q) in output.iter_mut().zip(p.output) {
                *o = (q + &*o)?;
            }
            middle = (p.middle + middle)?;
        }
        Ok(Some(crate::unet::Control { output, middle }))
    }
}

impl Denoise for MultiCondDenoiser<'_> {
    fn denoised(&mut self, x: &Tensor, sigma: f32) -> Result<Tensor> {
        let div = Tensor::new((sigma * sigma + 1.0).sqrt(), self.dev)?;
        let xc = x.broadcast_div(&div)?.to_dtype(DType::F16)?;
        let t = self.disc.timestep(sigma);
        let sig = Tensor::new(sigma, self.dev)?;
        let b = x.dim(0)?;
        let mut sum = [x.zeros_like()?, x.zeros_like()?];
        let mut count = [Tensor::full(1e-37f32, x.shape(), self.dev)?, Tensor::full(1e-37f32, x.shape(), self.dev)?];
        let lift = std::env::var_os("RUSTYUI_BATCH_CFG").is_some();
        for (weights, entries) in &self.groups {
            let built;
            let unet: &Unet = match weights {
                Weights::Resident(u) => u,
                Weights::Rebuilt { file, hooks, base } => {
                    let patching = model::Patching { hooks: hooks.clone(), base: Some(base), ..Default::default() };
                    let src = Source {
                        file,
                        map: model::unet_map(),
                        loras: vec![],
                        lora_prefix: "lora_unet_",
                        compute: DType::F16,
                        dtype: DType::F16,
                        dev: self.dev,
                        patching,
                    };
                    built = Unet::load(&src)?;
                    &built
                }
            };
            let mut to_run: Vec<usize> = (0..entries.len()).collect();
            while !to_run.is_empty() {
                // can_concat_cond: contexts whose lengths pad to their lcm within 4x
                let l1 = entries[to_run[0]].ctx.dim(1)?;
                let mut to_batch = vec![];
                for (i, &e) in to_run.iter().enumerate() {
                    let l2 = entries[e].ctx.dim(1)?;
                    if (l1 == l2 || lift || lcm(l1, l2) / l1.min(l2) <= 4) && same_control(&entries[to_run[0]].control, &entries[e].control)
                    {
                        to_batch.push(i);
                    }
                }
                to_batch.reverse();
                let batch: Vec<usize> = to_batch.iter().map(|&i| to_run.remove(i)).collect();
                // cond_cat: each context repeated to the lcm of their lengths, and to the latent's batch
                let mut m = 1;
                for &e in &batch {
                    m = lcm(m, entries[e].ctx.dim(1)?);
                }
                let (mut ctx, mut y) = (vec![], vec![]);
                for &e in &batch {
                    let en = &entries[e];
                    let l = en.ctx.dim(1)?;
                    let cx = if l < m { en.ctx.repeat((1, m / l, 1))? } else { en.ctx.clone() };
                    ctx.push(if b > 1 { cx.repeat((b, 1, 1))? } else { cx });
                    y.push(if b > 1 { en.y.repeat((b, 1))? } else { en.y.clone() });
                }
                let n = batch.len();
                let xb = if n > 1 { Tensor::cat(&vec![&xc; n], 0)? } else { xc.clone() };
                let (ts, ctx, y) = (vec![t; n * b], Tensor::cat(&ctx, 0)?, Tensor::cat(&y, 0)?);
                let ctrl = match &entries[batch[0]].control {
                    Some(ca) => self.control_for(ca, &xb, &ts, &ctx, &y, sigma)?,
                    None => None,
                };
                // cond_or_uncond: the side of each prompt in the batch
                let cc = self
                    .couple
                    .as_ref()
                    .map(|c| crate::unet::CoupleCall { couple: c, sides: batch.iter().map(|&e| entries[e].side).collect() });
                let out = unet.forward_couple(&xb, &ts, &ctx, &y, ctrl.as_ref(), cc.as_ref(), self.dev)?;
                for (k, &e) in batch.iter().enumerate() {
                    let en = &entries[e];
                    let den = (x - out.narrow(0, k * b, b)?.broadcast_mul(&sig)?)?;
                    let s = en.side;
                    match &en.mult {
                        Some(mult) => {
                            sum[s] = (&sum[s] + (den * mult)?)?;
                            count[s] = (&count[s] + mult)?;
                        }
                        None => {
                            sum[s] = (&sum[s] + den)?;
                            count[s] = (&count[s] + 1.0)?;
                        }
                    }
                }
            }
        }
        let cond = (&sum[0] / &count[0])?;
        let uncond = (&sum[1] / &count[1])?;
        let cfg = Tensor::new(self.cfg, self.dev)?;
        let out = (&uncond + (&cond - &uncond)?.broadcast_mul(&cfg)?)?;
        self.last_uncond = Some(uncond);
        Ok(out)
    }

    fn uncond_denoised(&self) -> Option<Tensor> {
        self.last_uncond.clone()
    }
}

/// Bytes of tensor data a value holds
fn val_bytes(v: &Val) -> usize {
    let t = |t: &Tensor| t.elem_count() * t.dtype().size_in_bytes();
    match v {
        Val::Latent(x) | Val::Image(x) | Val::Mask(x) => t(x),
        Val::MaskedLatent(x, m) => t(x) + t(m),
        Val::Video(v) => v.bytes(),
        Val::Audio(a) => t(&a.waveform),
        Val::LtxLatent(l) => l.bytes(),
        Val::LtxCond(c) => c.bytes(),
        Val::AnimaCond(c) => c.bytes(),
        _ => 0,
    }
}

/// The nodes a graph is run for: what they need is evaluated, nothing else
pub const OUTPUT_NODES: &[&str] = &["SaveImage", "SaveVideo", "PlaySoundKJ", "PreviewAny"];

/// Node types this engine runs besides LTX_NODES; a graph with any other is refused before it starts.
pub const NODES: &[&str] = &[
    "CheckpointLoaderSimple",
    "LoraLoader",
    "LoraLoaderModelOnly",
    "CLIPSetLastLayer",
    "CLIPTextEncode",
    "ConditioningConcat",
    "EmptyLatentImage",
    "KSampler",
    "VAEDecode",
    "VAEEncode",
    "UpscaleModelLoader",
    "ImageUpscaleWithModel",
    "ImageScale",
    "SaveImage",
];

pub fn unsupported(graph: &Value) -> Vec<String> {
    let mut out: Vec<String> = graph
        .as_object()
        .map(|m| {
            m.values()
                .filter_map(|n| n["class_type"].as_str())
                .filter(|c| {
                    !NODES.contains(c)
                        && !LTX_NODES.contains(c)
                        && !UTIL_NODES.contains(c)
                        && !COND_NODES.contains(c)
                        && !IMPACT_NODES.contains(c)
                        && !TAC_NODES.contains(c)
                        && !ANIMA_NODES.contains(c)
                })
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default();
    out.sort();
    out.dedup();
    out
}

impl Engine {
    /// Runs every SaveImage and SaveVideo in the graph; returns (node id, files written) per output node.
    /// Loaded models stay cached for the next graph (one in VRAM at a time).
    pub fn run(&mut self, graph: Value) -> Result<Vec<(String, Vec<PathBuf>)>> {
        self.run_outputs(graph, None, |_, _| {})
    }

    /// `run` for `only` these output nodes when given (ComfyUI's partial_execution_targets),
    /// telling `each` the files of each output node as it finishes.
    pub fn run_outputs(
        &mut self,
        graph: Value,
        only: Option<&[String]>,
        mut each: impl FnMut(&str, &[PathBuf]),
    ) -> Result<Vec<(String, Vec<PathBuf>)>> {
        self.graph = match graph {
            Value::Object(m) => m,
            _ => bail!("the graph is not a JSON object of nodes"),
        };
        self.memo.clear();
        self.sigs.clear();
        self.used.clear();
        self.timings.clear();
        self.dump = std::env::var_os("RUSTYUI_DUMP_NODES").map(|_| HashMap::new());
        let bad = unsupported(&Value::Object(self.graph.clone()));
        if !bad.is_empty() {
            bail!("not implemented in rustyui yet: {}", bad.join(", "));
        }
        let mut outputs: Vec<String> = self
            .graph
            .iter()
            .filter(|(id, n)| OUTPUT_NODES.iter().any(|o| n["class_type"] == *o) && only.is_none_or(|o| o.contains(id)))
            .map(|(id, _)| id.clone())
            .collect();
        outputs.sort_by_key(|id| id.parse::<u64>().unwrap_or(u64::MAX));
        if outputs.is_empty() {
            bail!("the graph has no SaveImage or SaveVideo node");
        }
        // ComfyUI's execution_cached: the nodes whose outputs the last graphs left in the cache
        let ids: Vec<String> = self.graph.keys().cloned().collect();
        let cached: Vec<String> = ids.into_iter().filter(|id| self.signature(id, 0).is_ok_and(|s| self.cache.contains_key(&s))).collect();
        crate::progress::cached(&cached);
        let mut result = vec![];
        for id in self.comfy_order(&outputs)? {
            let before = self.saved.len();
            self.eval(&id)?;
            if outputs.contains(&id) {
                let files = self.saved[before..].to_vec();
                each(&id, &files);
                result.push((id, files));
            }
        }
        // every node this graph has (its signature was taken, run or not) keeps its outputs: a
        // node found in the cache is not evaluated, nor is anything above it
        let mut used = std::mem::take(&mut self.used);
        used.extend(self.sigs.values().copied());
        self.cache.retain(|k, _| used.contains(k));
        if let (Some(d), Some(path)) = (self.dump.take(), std::env::var_os("RUSTYUI_DUMP_NODES")) {
            candle_core::safetensors::save(&d, PathBuf::from(&path))?;
            eprintln!("[engine] {} node tensors written to {}", d.len(), PathBuf::from(path).display());
        }
        Ok(result)
    }

    /// Drops every cached model (VRAM back to the pool).
    pub fn free(&mut self) {
        self.vae_dec = None;
        self.vae_enc = None;
        self.esrgan = None;
        self.sam3 = None;
        self.clip = None;
        self.ltx_te = None;
        self.ltx_dit = None;
        self.unet = None;
    }
}
