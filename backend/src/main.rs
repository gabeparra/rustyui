//! rustyui-backend: rustyui's server and renderer. It runs ComfyUI API graphs, from ComfyUI
//! clients and from the editors, in Rust on candle, and serves the editors (ComfyUI's, forked, from
//! ../ui/dist at /, and rustyui's first from ../frontend/dist at /simple/, compiled in by build.rs).
//!
//!     rustyui-backend serve [--listen 127.0.0.1:8190] [--idle 90] [--exit-idle S] [--models DIR] [--out DIR] [--frontend DIR]
//!                           [--prefix NAME] [--mirror HOST:DIR] [--writer GGUF [--writer-tokenizer FILE]] [--no-video]
//!                                                   the server: ComfyUI's API and the editors
//!     rustyui-backend run graph.json --out DIR      a ComfyUI API graph, once
//!     rustyui-backend from-png ref.png --out DIR    the graph inside a ComfyUI PNG, rendered again
//!     rustyui-backend compare a.png b.png           how far apart two renders are, in numbers
//!     rustyui-backend tokens "prompt"               the CLIP-L / CLIP-G token chunks
//!     rustyui-backend t2v "prompt" [--width 448] [--height 640] [--frames 49] [--fps 30] [--seed N] [--out FILE.mp4]
//!                                                   a silent LTX-2.3 clip (the base pass), to an MP4
//!     rustyui-backend write "idea" --model GGUF [--current PROMPT] [--tokenizer FILE]
//!                                                   the prompt writer, once
//!
//! `--models DIR` (or `RUSTYUI_MODELS`) is a folder laid out as ComfyUI's `models/`; the default
//! is `./models`. `--out DIR` (or `RUSTYUI_OUTPUT`) is where renders go; the default is `./output`.

// Lints this code base breaks on purpose:
// - the float constants are copied digit for digit from PyTorch, Cephes, curand and the other
//   libraries they reproduce, so they read the same as the code they come from;
// - kernel launchers take one argument per kernel parameter, and some results are big tuples;
// - `drop((view, ...))` after a launch keeps device views alive until the launch has returned;
// - index loops follow the C and Python code they port, index for index;
// - hand-written clamps keep their NaN behaviour, which `clamp` would change.
#![allow(
    clippy::excessive_precision,
    clippy::approx_constant,
    clippy::too_many_arguments,
    clippy::type_complexity,
    clippy::drop_non_drop,
    clippy::needless_range_loop,
    clippy::explicit_counter_loop,
    clippy::manual_clamp,
    clippy::unit_arg,
    clippy::result_large_err
)]
mod anima;
mod anima_tokens;
mod canny;
mod clip;
mod depth_anything;

mod controlnet;

mod brownian;
mod gemma;
mod graph;
mod image;
mod impact;
mod kernels;
mod ltx;
mod ltx_audio;
mod ltx_upsampler;
mod ltx_vae;
mod ltx_video;
mod model;
mod nn;
mod paths;
mod profile;
mod progress;
mod prompt;
mod qwen3;
mod resample;
mod rife;
mod rng;
mod sam3;
mod sampling;
mod server;
mod tokenizer;
mod torchcpu;
mod unet;
mod upscale;
mod vae;
mod wan_vae;
mod writer;

use anyhow::{bail, Context, Result};
use std::path::PathBuf;
use std::time::Instant;

fn arg<'a>(args: &'a [String], name: &str) -> Option<&'a str> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).map(String::as_str)
}

fn render(graph: serde_json::Value, out: PathBuf) -> Result<()> {
    let t = Instant::now();
    let mut eng = graph::Engine::new(&out)?;
    let result = eng.run(graph)?;
    for (what, secs) in &eng.timings {
        eprintln!("[engine] {what}: {secs:.2}s");
    }
    profile::report();
    eprintln!("[engine] peak VRAM seen {:.0} MB, total {:.1}s", eng.peak_vram_mb, t.elapsed().as_secs_f64());
    for (_, files) in &result {
        for p in files {
            println!("{}", p.display());
        }
    }
    Ok(())
}

use paths::models_dir;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    // --models and --out win over the environment, and reach every part that reads it
    if let Some(dir) = arg(&args, "--models") {
        std::env::set_var("RUSTYUI_MODELS", dir);
    }
    if let Some(dir) = arg(&args, "--out") {
        std::env::set_var("RUSTYUI_OUTPUT", dir);
    }
    let out = paths::output_dir();
    match args.get(1).map(String::as_str) {
        Some("run") => {
            let path = args.get(2).context("run GRAPH.json --out DIR")?;
            let graph: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(path)?)?;
            render(graph, out)
        }
        Some("from-png") => {
            let path = PathBuf::from(args.get(2).context("from-png IMG.png --out DIR")?);
            let png = image::read_png(&path)?;
            let text = png.texts.iter().find(|(k, _)| k == "prompt").map(|(_, v)| v.clone()).context("no ComfyUI graph (\"prompt\" text) in that PNG")?;
            render(serde_json::from_str(&text)?, out)
        }
        Some("serve") => server::serve(server::Options {
            listen: arg(&args, "--listen").unwrap_or("127.0.0.1:8190").to_string(),
            out: out.clone(),
            idle: arg(&args, "--idle").unwrap_or("90").parse()?,
            exit_idle: arg(&args, "--exit-idle").map(str::parse::<u64>).transpose()?,
            frontend: arg(&args, "--frontend").map(PathBuf::from),
            prefix: arg(&args, "--prefix").unwrap_or("rustyui").to_string(),
            mirror: arg(&args, "--mirror").map(str::to_string),
            writer: arg(&args, "--writer").map(|g| {
                let tokenizer = arg(&args, "--writer-tokenizer").map(PathBuf::from).unwrap_or_else(|| writer::default_tokenizer(&models_dir()));
                (PathBuf::from(g), tokenizer)
            }),
            no_video: args.iter().any(|a| a == "--no-video"),
        }),
        Some("write") => {
            let usage = "write \"idea\" --model GGUF [--current PROMPT] [--tokenizer FILE]";
            let idea = args.get(2).context(usage)?;
            let model = arg(&args, "--model").map(PathBuf::from).context(usage)?;
            let tokenizer = arg(&args, "--tokenizer").map(PathBuf::from).unwrap_or_else(|| writer::default_tokenizer(&models_dir()));
            let mut w = writer::Writer::load(&model, &tokenizer, &candle_core::Device::new_cuda(0)?)?;
            let seed = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)?.as_nanos() as u64;
            let r = w.write(idea, arg(&args, "--current").unwrap_or(""), seed)?;
            eprintln!("[writer] {} tokens in {:.1}s ({:.0} tokens/s)", r.tokens, r.seconds, r.tokens as f64 / r.seconds.max(1e-9));
            println!("{}", r.prompt);
            Ok(())
        }
        Some("t2v") => {
            let prompt = args.get(2).context("t2v \"prompt\" [--width 448] [--height 640] [--frames 49] [--fps 30] [--seed N] [--out FILE.mp4]")?;
            let seed: u64 = match arg(&args, "--seed") {
                Some(s) => s.parse()?,
                None => std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)?.as_micros() as u64 % (1 << 48),
            };
            let out = match arg(&args, "--out") {
                Some(f) if f.ends_with(".mp4") => PathBuf::from(f),
                _ => out.join(format!("rustyui-t2v_{seed}.mp4")),
            };
            let o = ltx_video::T2v {
                prompt: prompt.clone(),
                width: arg(&args, "--width").unwrap_or("448").parse()?,
                height: arg(&args, "--height").unwrap_or("640").parse()?,
                frames: arg(&args, "--frames").unwrap_or("49").parse()?,
                fps: arg(&args, "--fps").unwrap_or("30").parse()?,
                steps: arg(&args, "--steps").unwrap_or("8").parse()?,
                seed,
                out,
            };
            println!("{}", ltx_video::text_to_video(&o, &models_dir())?.display());
            Ok(())
        }
        Some("i2v") => {
            let usage = "i2v IMAGE.png \"prompt\" [--strength 1] [--width 448] [--height 640] [--frames 49] [--fps 30] [--seed N] [--out FILE.mp4]";
            let start = PathBuf::from(args.get(2).context(usage)?);
            let prompt = args.get(3).context(usage)?;
            let seed: u64 = match arg(&args, "--seed") {
                Some(s) => s.parse()?,
                None => std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)?.as_micros() as u64 % (1 << 48),
            };
            let out = match arg(&args, "--out") {
                Some(f) if f.ends_with(".mp4") => PathBuf::from(f),
                _ => out.join(format!("rustyui-i2v_{seed}.mp4")),
            };
            let o = ltx_video::T2v {
                prompt: prompt.clone(),
                width: arg(&args, "--width").unwrap_or("448").parse()?,
                height: arg(&args, "--height").unwrap_or("640").parse()?,
                frames: arg(&args, "--frames").unwrap_or("49").parse()?,
                fps: arg(&args, "--fps").unwrap_or("30").parse()?,
                steps: arg(&args, "--steps").unwrap_or("8").parse()?,
                seed,
                out,
            };
            let strength: f64 = arg(&args, "--strength").unwrap_or("1").parse()?;
            println!("{}", ltx_video::image_to_video(&o, &start, strength, &models_dir())?.display());
            Ok(())
        }
        Some("debug-conv") => {
            // torch's cuDNN convolutions (refs/conv_ref.py) against kernels::cudnn_conv
            let refs = args.get(2).context("debug-conv REF.safetensors (refs/conv_ref.py)")?;
            let dev = candle_core::Device::new_cuda(0)?;
            let st = unsafe { candle_core::safetensors::MmapedSafetensors::new(refs)? };
            for name in ["up3d_in", "up3d_mid", "up2d", "tmp3d", "sdxl2d", "vae3d_f32"] {
                let get = |k: &str| st.load(&format!("{name}.{k}"), &dev);
                let (x, w, b) = (get("x")?, get("w")?, get("b")?);
                let ns = x.rank() - 2;
                let y = kernels::cudnn_conv(&x, &w, &vec![1; ns], &vec![1; ns], &vec![1; ns])?;
                ltx::compare(&format!("{name} {:?}", x.dims()), &y, &get("y")?)?;
                let mut shape = vec![1usize; x.rank()];
                shape[1] = b.dim(0)?;
                ltx::compare(&format!("{name} with its bias"), &y.broadcast_add(&b.reshape(shape)?)?, &get("yb")?)?;
            }
            print!("{}", kernels::cudnn_conv_plans());
            Ok(())
        }
        Some("debug-conv-nhwc") => {
            // one channels-last convolution saved by torch (x, w, y; NCHW memory in the file) against
            // kernels::cudnn_conv_nhwc, and against y64 (torch's float64 result) if the file has it
            let usage = "debug-conv-nhwc REF.safetensors STRIDE PADDING";
            let refs = args.get(2).context(usage)?;
            let stride: usize = args.get(3).context(usage)?.parse()?;
            let pad: usize = args.get(4).context(usage)?.parse()?;
            let dev = candle_core::Device::new_cuda(0)?;
            let st = unsafe { candle_core::safetensors::MmapedSafetensors::new(refs)? };
            let (x, w) = (st.load("x", &dev)?, st.load("w", &dev)?);
            let y = kernels::cudnn_conv_nhwc(&x, &w, &[pad, pad], &[stride, stride], &[1, 1])?;
            ltx::compare(&format!("conv {:?} * {:?}", x.dims(), w.dims()), &y, &st.load("y", &dev)?)?;
            if let Ok(y64) = st.load("y64", &dev) {
                ltx::compare("against float64", &y, &y64)?;
            }
            print!("{}", kernels::cudnn_conv_plans());
            Ok(())
        }
        Some("debug-group-norm") => {

            // torch's GroupNorm, and SiLU after it (refs/gn_ref.py), against kernels::group_norm
            let refs = args.get(2).context("debug-group-norm REF.safetensors (refs/gn_ref.py)")?;
            let dev = candle_core::Device::new_cuda(0)?;
            let st = unsafe { candle_core::safetensors::MmapedSafetensors::new(refs)? };
            for (name, eps) in [("up_bf16", 1e-5), ("small_bf16", 1e-5), ("sdxl_f16", 1e-5), ("vae_f32", 1e-6), ("small_f32", 1e-6)] {
                let get = |k: &str| st.load(&format!("{name}.{k}"), &dev);
                let (x, w, b) = (get("x")?, get("w")?, get("b")?);
                ltx::compare(&format!("{name} {:?}", x.dims()), &kernels::group_norm(&x, &w, &b, 32, eps, false)?, &get("y")?)?;
                ltx::compare(&format!("{name}, then SiLU"), &kernels::group_norm(&x, &w, &b, 32, eps, true)?, &get("ys")?)?;
            }
            Ok(())
        }
        Some("debug-audio") => {
            let refs = args.get(2).context("debug-audio REF.safetensors (refs/audio_ref.py) [--vae FILE]")?;
            let vae = arg(&args, "--vae").map(PathBuf::from).unwrap_or_else(|| models_dir().join("checkpoints/ltx-2.3-audio-vae.safetensors"));
            ltx_audio::debug_audio(std::path::Path::new(refs), &vae)
        }
        Some("debug-upsampler") => ltx_upsampler::debug_upsampler(std::path::Path::new(args.get(2).context("debug-upsampler REF.safetensors (refs/upsampler_ref.py)")?), &models_dir()),
        Some("debug-rife") => {
            let ckpt = args.get(2).context("debug-rife CKPT.pth REF.safetensors (refs/rife_ref.py)")?;
            let reference = args.get(3).context("debug-rife CKPT.pth REF.safetensors (refs/rife_ref.py)")?;
            rife::debug_rife(std::path::Path::new(ckpt), std::path::Path::new(reference))
        }
        Some("debug-beta-schedule") => {
            // the beta schedule's table indices for 1..=N steps, one JSON list a line
            let n: usize = args.get(2).context("debug-beta-schedule MAX_STEPS")?.parse()?;
            let d = sampling::Discrete::sdxl();
            for steps in 1..=n {
                println!("{}", serde_json::to_string(&d.beta_indices(steps))?);
            }
            Ok(())
        }
        Some("debug-preprocess") => ltx_video::debug_preprocess(std::path::Path::new(args.get(2).context("debug-preprocess REF.safetensors (refs/preprocess_ref.py)")?)),
        Some("debug-guide-attn") => ltx::debug_guide_attn(args.get(2).context("debug-guide-attn REF.safetensors (refs/guide_attn_ref.py)")?),
        Some("debug-ltx-lora") => {
            let usage = "debug-ltx-lora CHECKPOINT LORA_REF.safetensors LORA.safetensors...";
            let ckpt = args.get(2).context(usage)?;
            let refs = args.get(3).context(usage)?;
            let loras: Vec<PathBuf> = args[4..].iter().map(PathBuf::from).collect();
            anyhow::ensure!(!loras.is_empty(), "{usage}");
            ltx::debug_lora(ckpt, refs, &loras)
        }
        Some("debug-ltx-i2v") => {
            let usage = "debug-ltx-i2v CHECKPOINT I2V_REF.safetensors IMAGE.png [--vae FILE]";
            let ckpt = PathBuf::from(args.get(2).context(usage)?);
            let refs = PathBuf::from(args.get(3).context(usage)?);
            let image = PathBuf::from(args.get(4).context(usage)?);
            let vae = arg(&args, "--vae").map(PathBuf::from).unwrap_or_else(|| models_dir().join("vae/ltx-2.3-video-vae.safetensors"));
            ltx_video::debug_i2v(&ckpt, &refs, &image, &vae, args.iter().any(|a| a == "--ancestral"))
        }
        Some("debug-ltx-guide") => {
            let usage = "debug-ltx-guide CHECKPOINT GUIDE_REF.safetensors IMAGE.png [--strength 0.7] [--vae FILE]";
            let ckpt = PathBuf::from(args.get(2).context(usage)?);
            let refs = PathBuf::from(args.get(3).context(usage)?);
            let image = PathBuf::from(args.get(4).context(usage)?);
            let vae = arg(&args, "--vae").map(PathBuf::from).unwrap_or_else(|| models_dir().join("vae/ltx-2.3-video-vae.safetensors"));
            ltx_video::debug_guide(&ckpt, &refs, &image, &vae, arg(&args, "--strength").unwrap_or("0.7").parse()?)
        }
        Some("debug-ltx-vae-enc") => {
            // the video VAE encoder against ComfyUI's encode of the same pixels (refs/vae_enc_ref.py)
            let refs = args.get(2).context("debug-ltx-vae-enc REF.safetensors [--vae FILE]")?;
            let vae = arg(&args, "--vae").map(PathBuf::from).unwrap_or_else(|| models_dir().join("vae/ltx-2.3-video-vae.safetensors"));
            let dev = candle_core::Device::new_cuda(0)?;
            let st = unsafe { candle_core::safetensors::MmapedSafetensors::new(refs)? };
            let enc = ltx_vae::VideoEncoder::load(&vae, &dev)?;
            for which in ["one", "nine"] {
                let t = Instant::now();
                let got = enc.encode(&st.load(&format!("{which}.image"), &dev)?)?;
                dev.synchronize()?;
                let want = st.load(&format!("{which}.latent"), &dev)?;
                let d = (&got - &want)?;
                let max = d.abs()?.flatten_all()?.max(0)?.to_scalar::<f32>()?;
                let rel = (d.sqr()?.sum_all()?.to_scalar::<f32>()? / want.sqr()?.sum_all()?.to_scalar::<f32>()?).sqrt();
                println!("{which}: {:?} in {:.2}s, max |diff| {max:.5}, relative {rel:.2e}", got.dims(), t.elapsed().as_secs_f64());
            }
            Ok(())
        }
        Some("debug-ltx-vae-stream") => {
            // the streaming video VAE against ComfyUI's decode and encode at a size where it chunks
            // (refs/vae_stream_ref.py)
            let refs = args.get(2).context("debug-ltx-vae-stream REF.safetensors [--vae FILE]")?;
            let vae = arg(&args, "--vae").map(PathBuf::from).unwrap_or_else(|| models_dir().join("vae/ltx-2.3-video-vae.safetensors"));
            let dev = candle_core::Device::new_cuda(0)?;
            let st = unsafe { candle_core::safetensors::MmapedSafetensors::new(refs)? };
            let exact = |name: &str, got: &candle_core::Tensor, want: &candle_core::Tensor| -> Result<()> {
                let d = (got - want)?.abs()?;
                let differ = d.ne(0f32)?.to_dtype(candle_core::DType::F32)?.sum_all()?.to_scalar::<f32>()?;
                let (max, mean, psnr) = ltx_vae::compare(got, want)?;
                println!("{name}: {:?}, {differ} of {} values differ, max {max:.6} mean {mean:.8} PSNR {psnr:.2} dB", got.dims(), got.elem_count());
                Ok(())
            };
            let dec = ltx_vae::VideoDecoder::load(&vae, &dev)?;
            let latent = st.load("latent", &dev)?;
            let t = Instant::now();
            let got = dec.decode(&latent)?;
            dev.synchronize()?;
            eprintln!("[vae] decode {:?} -> {:?} in {:.2}s", latent.dims(), got.dims(), t.elapsed().as_secs_f64());
            let want = st.load("decoded", &dev)?.squeeze(0)?;
            exact("decode", &got, &want)?;
            for f in [0, 1, got.dim(0)? / 2, got.dim(0)? - 1] {
                exact(&format!("  frame {f}"), &got.get(f)?, &want.get(f)?)?;
            }
            drop((got, want, dec));
            let enc = ltx_vae::VideoEncoder::load(&vae, &dev)?;
            let pixels = st.load("encode_in", &dev)?;
            let t = Instant::now();
            let got = enc.encode(&pixels)?;
            dev.synchronize()?;
            eprintln!("[vae] encode {:?} -> {:?} in {:.2}s", pixels.dims(), got.dims(), t.elapsed().as_secs_f64());
            exact("encode", &got, &st.load("encoded", &dev)?)?;
            Ok(())
        }
        Some("debug-ltx-vae") => {
            // the video VAE decoder against ComfyUI's decode of the same latent (refs/sample_ref.py)
            let refs = args.get(2).context("debug-ltx-vae REF.safetensors [--vae FILE]")?;
            let vae = arg(&args, "--vae").map(PathBuf::from).unwrap_or_else(|| models_dir().join("vae/ltx-2.3-video-vae.safetensors"));
            let dev = candle_core::Device::new_cuda(0)?;
            let st = unsafe { candle_core::safetensors::MmapedSafetensors::new(refs)? };
            let latent = st.load("vae.decode_in", &dev)?;
            let want = st.load("vae.decode_out", &dev)?.squeeze(0)?;
            let t = Instant::now();
            let dec = ltx_vae::VideoDecoder::load(&vae, &dev)?;
            eprintln!("[vae] loaded in {:.1}s", t.elapsed().as_secs_f64());
            let t = Instant::now();
            let got = dec.decode(&latent)?;
            dev.synchronize()?;
            eprintln!("[vae] {:?} -> {:?} in {:.2}s", latent.dims(), got.dims(), t.elapsed().as_secs_f64());
            let (max, mean, psnr) = ltx_vae::compare(&got, &want)?;
            println!("all frames: max {max:.6} mean {mean:.7} PSNR {psnr:.2} dB (pixels in 0..1)");
            for f in 0..got.dim(0)? {
                let (max, mean, psnr) = ltx_vae::compare(&got.get(f)?, &want.get(f)?)?;
                println!("frame {f}: max {max:.6} mean {mean:.7} PSNR {psnr:.2} dB");
            }
            Ok(())
        }
        Some("compare") => {
            let (a, b) = (args.get(2).context("compare A.png B.png")?, args.get(3).context("compare A.png B.png")?);
            println!("{}", image::compare(&PathBuf::from(a), &PathBuf::from(b))?);
            Ok(())
        }
        Some("tokens") => {
            let text = args.get(2).context("tokens \"prompt\"")?;
            let tok = tokenizer::ClipTokenizer::new()?;
            let mut embs = prompt::Embeddings::new(vec![models_dir().join("embeddings")]);
            for (name, pad, key) in [("l", tokenizer::EOS, "clip_l"), ("g", 0, "clip_g")] {
                let enc = prompt::Encoder { tok: &tok, pad, embedding_key: key, max_length: 77 };
                for (i, chunk) in enc.tokenize(text, &mut embs).iter().enumerate() {
                    let ids: Vec<String> = chunk
                        .iter()
                        .map(|(t, w)| match t {
                            prompt::Tok::Id(id) if *w == 1.0 => id.to_string(),
                            prompt::Tok::Id(id) => format!("{id}@{w}"),
                            prompt::Tok::Vec(_) => "<emb>".into(),
                        })
                        .collect();
                    println!("{name}[{i}] {}", ids.join(" "));
                }
            }
            Ok(())
        }
        Some("debug-lanczos") => {
            // debug-lanczos IN.png W H OUT.png: PIL-exact resize, for checking against PIL
            let (inp, w, h, outp) = (args.get(2).context("in")?, args.get(3).context("w")?.parse::<usize>()?, args.get(4).context("h")?.parse::<usize>()?, args.get(5).context("out")?);
            let png = image::read_png(&PathBuf::from(inp))?;
            let r = resample::resize_lanczos(&png.rgb, png.width as usize, png.height as usize, w, h);
            let file = std::io::BufWriter::new(std::fs::File::create(outp)?);
            let mut enc = ::png::Encoder::new(file, w as u32, h as u32);
            enc.set_color(::png::ColorType::Rgb);
            enc.set_depth(::png::BitDepth::Eight);
            enc.write_header()?.write_image_data(&r)?;
            Ok(())
        }
        Some("debug-wan-vae") => {
            let usage = "debug-wan-vae VAE.safetensors REF.safetensors (refs/wan_vae_ref.py)";
            let (vae, reference) = (args.get(2).context(usage)?, args.get(3).context(usage)?);
            wan_vae::debug_decode(std::path::Path::new(vae), std::path::Path::new(reference), &candle_core::Device::new_cuda(0)?)
        }
        Some("debug-anima-tokens") => {
            let reference = args.get(2).context("debug-anima-tokens REF.json (refs/anima_tokens_ref.py)")?;
            anima_tokens::debug_tokens(std::path::Path::new(reference), &prompt::Embeddings::new(vec![models_dir().join("embeddings")]))
        }
        Some("debug-gemma-tokens") => {
            // debug-gemma-tokens TE.safetensors PROMPT.json [REF.safetensors]: tokenize like ComfyUI
            let te = PathBuf::from(args.get(2).context("text encoder file")?);
            let prompt: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(args.get(3).context("prompt json")?)?)?;
            let text = prompt["prompt"].as_str().context("prompt json needs \"prompt\"")?;
            let st = unsafe { candle_core::safetensors::MmapedSafetensors::new(&te)? };
            let proto = st.load("spiece_model", &candle_core::Device::Cpu)?.to_vec1::<u8>()?;
            let tok = gemma::GemmaTokenizer::from_proto(&proto)?;
            let (ids, n) = tok.encode_padded(text)?;
            println!("padded {} real {} first real {:?} last {:?}", ids.len(), n, &ids[ids.len() - n..ids.len() - n + 8], &ids[ids.len() - 5..]);
            if let Some(r) = args.get(4) {
                let refs = unsafe { candle_core::safetensors::MmapedSafetensors::new(r)? };
                let want: Vec<i64> = refs.load("ids", &candle_core::Device::Cpu)?.to_vec1::<i64>()?;
                let got: Vec<i64> = ids.iter().map(|&x| x as i64).collect();
                let diff = want.iter().zip(&got).filter(|(a, b)| a != b).count();
                println!("vs ComfyUI: {} ids, {} differ{}", want.len(), diff + want.len().abs_diff(got.len()), if diff == 0 && want.len() == got.len() { " (identical)" } else { "" });
            }
            Ok(())
        }
        Some("debug-dequant") => {
            // debug-dequant GEMMA.safetensors REF.safetensors: our dequantised weights against the
            // ones ComfyUI's forward multiplies by, element for element
            let dev = candle_core::Device::new_cuda(0)?;
            let g = gemma::Gemma::open(&PathBuf::from(args.get(2).context("gemma file")?))?;
            let refs = unsafe { candle_core::safetensors::MmapedSafetensors::new(args.get(3).context("reference")?)? };
            for (key, layer) in [("fp8_l10_gate", "model.layers.10.mlp.gate_proj"), ("nvfp4_l0_q", "model.layers.0.self_attn.q_proj")] {
                let ours = g.debug_linear(layer, &dev)?;
                let theirs = refs.load(key, &dev)?;
                let differ = ours.ne(&theirs)?.to_dtype(candle_core::DType::F32)?.sum_all()?.to_scalar::<f32>()?;
                let maxd = (&ours - &theirs)?.abs()?.flatten_all()?.max(0)?.to_scalar::<f32>()?;
                println!("{layer}: {differ} of {} elements differ, max |diff| {maxd:e}", ours.elem_count());
            }
            Ok(())
        }
        Some("debug-ltx-proj") => {
            // debug-ltx-proj PROJ.safetensors TE_REF.safetensors PROJ_REF.safetensors: the projection
            // split at its GEMMs, against what ComfyUI's two linears received and returned
            use candle_core::DType;
            let dev = candle_core::Device::new_cuda(0)?;
            let proj = unsafe { candle_core::safetensors::MmapedSafetensors::new(args.get(2).context("projection file")?)? };
            let te = unsafe { candle_core::safetensors::MmapedSafetensors::new(args.get(3).context("te reference")?)? };
            let pr = unsafe { candle_core::safetensors::MmapedSafetensors::new(args.get(4).context("projection reference")?)? };
            let (xv, xa) = gemma::ltx_projection_inputs(&te.load("hidden", &dev)?.squeeze(0)?)?;
            let count = |a: &candle_core::Tensor, b: &candle_core::Tensor| -> Result<f32> {
                Ok(a.ne(b)?.to_dtype(DType::F32)?.sum_all()?.to_scalar::<f32>()?)
            };
            for (which, ours_in) in [("video", xv), ("audio", xa)] {
                let their_in = pr.load(&format!("{which}_in"), &dev)?.squeeze(0)?;
                let their_out = pr.load(&format!("{which}_out"), &dev)?.squeeze(0)?;
                let w = proj.load(&format!("text_embedding_projection.{which}_aggregate_embed.weight"), &dev)?;
                let b = proj.load(&format!("text_embedding_projection.{which}_aggregate_embed.bias"), &dev)?;
                println!("{which}: our GEMM input differs from ComfyUI's on {} of {}", count(&ours_in, &their_in)?, their_in.elem_count());
                // the same input through our bf16 GEMM, and through an f32-accumulated one
                let lt = kernels::torch_linear(&their_in.to_dtype(DType::BF16)?, &w, &b)?.to_dtype(DType::F32)?;
                let exact = their_in.matmul(&w.to_dtype(DType::F32)?.t()?)?.broadcast_add(&b.to_dtype(DType::F32)?)?
                    .to_dtype(DType::BF16)?.to_dtype(DType::F32)?;
                println!("   our bf16 GEMM vs exact: {} differ; vs ComfyUI's: {} differ; exact vs ComfyUI's: {} differ (of {})",
                    count(&lt, &exact)?, count(&lt, &their_out)?, count(&exact, &their_out)?, their_out.elem_count());
            }
            Ok(())
        }
        Some("debug-ipp-resize") => {
            // IPP's uint8 bicubic resize against cv2's (refs/ipp_resize_ref.py)
            depth_anything::debug_ipp_resize(&PathBuf::from(args.get(2).context("debug-ipp-resize REF.safetensors")?))
        }
        Some("debug-depth") => {
            // Depth Anything V2 stage by stage against refs/depth_anything_ref.py
            let usage = "debug-depth PICTURE CKPT RESOLUTION REF";
            let pic = PathBuf::from(args.get(2).context(usage)?);
            let ckpt = PathBuf::from(args.get(3).context(usage)?);
            let res: usize = args.get(4).context(usage)?.parse()?;
            let refs = PathBuf::from(args.get(5).context(usage)?);
            depth_anything::debug_depth(&pic, &ckpt, res, &refs, &candle_core::Device::new_cuda(0)?)
        }
        Some("debug-impact-blur") => {
            // GrowMask and Impact's mask blur against refs/impact_blur_ref.py, and against
            // the small masks refs/impact_segs_ref.py saves
            anyhow::ensure!(args.len() > 2, "debug-impact-blur REF.safetensors...");
            for r in &args[2..] {
                impact::debug_blur(&PathBuf::from(r))?;
            }
            Ok(())
        }
        Some("debug-impact-segs") => {
            // MaskToSEGS against Impact's own (refs/impact_segs_ref.py)
            impact::debug_segs(&PathBuf::from(args.get(2).context("debug-impact-segs REF.safetensors")?))
        }
        Some("debug-sam3") => {

            // SAM3Segment stage by stage against refs/sam3_ref.py
            let usage = "debug-sam3 PICTURE PROMPT REF [MODE CONFIDENCE MAX_SEGMENTS PICK BLUR]";
            let pic = PathBuf::from(args.get(2).context(usage)?);
            let prompt = args.get(3).context(usage)?;
            let refs = PathBuf::from(args.get(4).context(usage)?);
            sam3::debug_sam3(&pic, prompt, &refs, &models_dir(), &candle_core::Device::new_cuda(0)?, &args[5.min(args.len())..])

        }
        Some("debug-schedule") => {

            // the SDXL sigma table's schedules as f32 bit patterns, one list per step count, for
            // comparing with ComfyUI's calculate_sigmas (refs/cmp_schedule.py); RUSTYUI_SIGMAS_ON_CARD=1
            // as a model ComfyUI has loaded gives them
            let usage = "debug-schedule SCHEDULER STEPS...";
            let sched = args.get(2).context(usage)?;
            let disc = sampling::Discrete::sdxl();
            let card = match std::env::var_os("RUSTYUI_SIGMAS_ON_CARD") {
                Some(_) => Some(candle_core::Device::new_cuda(0)?),
                None => None,
            };
            let mut out = serde_json::Map::new();
            for s in &args[3..] {
                let sig = disc.schedule(sched, s.parse()?, card.as_ref())?;

                out.insert(s.clone(), sig.iter().map(|v| v.to_bits()).collect::<Vec<u32>>().into());
            }
            println!("{}", serde_json::Value::Object(out));
            Ok(())
        }
        Some("debug-powf") => {
            // Sleef's powf against torch's: PAIRS.bin holds (x, y, torch.pow(x, y)) as little-endian
            // f32 triples, every x from the vectorized part of a tensor (refs/cmp_schedule.py --powf)
            let data = std::fs::read(args.get(2).context("debug-powf TRIPLES.bin")?)?;
            let f = |i: usize| f32::from_le_bytes(data[i * 4..i * 4 + 4].try_into().unwrap());
            let (mut bad, n) = (0usize, data.len() / 12);
            for k in 0..n {
                let (x, y, want) = (f(3 * k), f(3 * k + 1), f(3 * k + 2));
                let got = torchcpu::sleef_powf(x, y);
                if got.to_bits() != want.to_bits() {
                    if bad < 10 {
                        println!("powf({x:e} [{:#x}], {y}) = {got:e} [{:#x}], torch {want:e} [{:#x}]", x.to_bits(), got.to_bits(), want.to_bits());
                    }
                    bad += 1;
                }
            }
            println!("{bad} of {n} differ");
            Ok(())
        }
        Some("debug-sdxl-vae") => {
            // SDXL's VAE decoder block by block against refs/sdxl_vae_ref.py (the README lighthouse graph's decode)
            let usage = "debug-sdxl-vae CHECKPOINT VAE_REF";
            vae::debug_vae(&PathBuf::from(args.get(2).context(usage)?), &PathBuf::from(args.get(3).context(usage)?), &candle_core::Device::new_cuda(0)?)
        }
        Some("debug-esrgan") => {
            // ImageUpscaleWithModel's ESRGAN tile by tile against refs/esrgan_ref.py
            let usage = "debug-esrgan MODEL.pth ESRGAN_REF";
            upscale::debug_esrgan(&PathBuf::from(args.get(2).context(usage)?), &PathBuf::from(args.get(3).context(usage)?), &candle_core::Device::new_cuda(0)?)
        }
        Some("debug-unet") => {
            // SDXL's UNet block by block against refs/unet_ref.py (the README lighthouse graph's first call)
            let usage = "debug-unet CHECKPOINT UNET_REF";
            unet::debug_unet(&PathBuf::from(args.get(2).context(usage)?), &PathBuf::from(args.get(3).context(usage)?), &candle_core::Device::new_cuda(0)?)
        }
        Some("debug-clip") => {
            // SDXL's text encoders op by op against refs/clip_ref.py (the README lighthouse graph's prompt)
            let usage = "debug-clip CHECKPOINT CLIP_REF [GRAPH.json]";
            let ckpt = PathBuf::from(args.get(2).context(usage)?);
            let refs = PathBuf::from(args.get(3).context(usage)?);
            let graph: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(args.get(4).map(String::as_str).unwrap_or("backend/refs/fixtures/readme-graph.json"))?)?;
            let g = graph.get("prompt").unwrap_or(&graph);
            let prompt = g["2"]["inputs"]["text"].as_str().context("node 2 has no text")?;
            clip::debug_clip(&ckpt, &refs, prompt, &candle_core::Device::new_cuda(0)?)
        }
        Some("debug-gemma-layer") => {
            // Gemma's layers 0 and 5 op by op against refs/gemma_layer_ref.py
            let usage = "debug-gemma-layer GEMMA.safetensors REF.safetensors";
            let g = gemma::Gemma::open(&PathBuf::from(args.get(2).context(usage)?))?;
            g.debug_layers(&PathBuf::from(args.get(3).context(usage)?), &candle_core::Device::new_cuda(0)?)
        }
        Some("debug-ltx-te") => {
            // debug-ltx-te GEMMA.safetensors PROJ.safetensors PROMPT.json REF.safetensors
            let dev = candle_core::Device::new_cuda(0)?;
            let t0 = Instant::now();
            let g = gemma::Gemma::open(&PathBuf::from(args.get(2).context("gemma file")?))?;
            let proj = unsafe { candle_core::safetensors::MmapedSafetensors::new(args.get(3).context("projection file")?)? };
            let prompt: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(args.get(4).context("prompt json")?)?)?;
            let (ids, n) = g.tok.encode_padded(prompt["prompt"].as_str().context("prompt")?)?;
            let hidden = g.hidden_states(&ids, n, &dev)?;
            dev.synchronize()?;
            let t1 = t0.elapsed().as_secs_f64();
            let cond = gemma::ltx_project(&hidden, &proj, &dev)?;
            dev.synchronize()?;
            println!("gemma {t1:.2}s, total {:.2}s; hidden {:?} cond {:?}", t0.elapsed().as_secs_f64(), hidden.dims(), cond.dims());
            let refs = unsafe { candle_core::safetensors::MmapedSafetensors::new(args.get(5).context("reference")?)? };
            let rh = refs.load("hidden", &dev)?.squeeze(0)?;
            let rc = refs.load("cond", &dev)?;
            let stat = |a: &candle_core::Tensor, b: &candle_core::Tensor| -> Result<(f32, f32)> {
                let d = (a - b)?;
                let maxd = d.abs()?.flatten_all()?.max(0)?.to_scalar::<f32>()?;
                let rel = (d.sqr()?.sum_all()?.to_scalar::<f32>()? / b.sqr()?.sum_all()?.to_scalar::<f32>()?).sqrt();
                Ok((maxd, rel))
            };
            for l in [0usize, 1, 2, 6, 12, 24, 36, 47, 48] {
                let (m, r) = stat(&hidden.get(l)?, &rh.get(l)?)?;
                println!("  hidden[{l:>2}] max |diff| {m:.6}  relative {r:.2e}");
            }
            let (m, r) = stat(&cond, &rc)?;
            println!("  cond       max |diff| {m:.6}  relative {r:.2e}");
            // the projection alone, fed ComfyUI's own hidden states
            let not_bf16 = rc.ne(&rc.to_dtype(candle_core::DType::BF16)?.to_dtype(candle_core::DType::F32)?)?
                .to_dtype(candle_core::DType::F32)?.sum_all()?.to_scalar::<f32>()?;
            println!("  reference cond: {not_bf16} of {} values are not bf16 values", rc.elem_count());
            let (m, r) = stat(&gemma::ltx_project(&rh, &proj, &dev)?, &rc)?;
            println!("  projection alone, bf16: max |diff| {m:.6}  relative {r:.2e}");
            let (m, r) = stat(&gemma::ltx_project_f32(&rh, &proj, &dev)?, &rc)?;
            println!("  projection alone, f32:  max |diff| {m:.6}  relative {r:.2e}");
            Ok(())
        }
        Some("debug-fp8") => {
            // debug-fp8 REF.safetensors: our fp8 input rounding and fp8 GEMM against ComfyUI's
            // (fp8_ref.py: real LTX-2.3 layers, fixed inputs), and both against an exact product
            use candle_core::{DType, Tensor};
            let dev = candle_core::Device::new_cuda(0)?;
            let r = unsafe { candle_core::safetensors::MmapedSafetensors::new(args.get(2).context("reference")?)? };
            let one = Tensor::new(&[1f32], &dev)?;
            let count = |a: &Tensor, b: &Tensor| -> Result<f32> { Ok(a.ne(b)?.to_dtype(DType::F32)?.sum_all()?.to_scalar::<f32>()?) };
            for i in 0.. {
                let c = format!("c{i}");
                let Ok(x) = r.load(&format!("{c}.x"), &dev) else { break };
                let (w, ws, b) = (r.load(&format!("{c}.w"), &dev)?, r.load(&format!("{c}.w_scale"), &dev)?, r.load(&format!("{c}.bias"), &dev)?);
                let q = kernels::quant_fp8(&x)?;
                let qd = count(&q, &r.load(&format!("{c}.qx"), &dev)?)?;
                let t = Instant::now();
                let y = kernels::fp8_linear(&x, &w, &ws, &one, Some(&b))?;
                dev.synchronize()?;
                let secs = t.elapsed().as_secs_f64();
                let theirs = r.load(&format!("{c}.y"), &dev)?;
                let exact = r.load(&format!("{c}.exact"), &dev)?.to_dtype(DType::BF16)?;
                let maxd = (y.to_dtype(DType::F32)? - theirs.to_dtype(DType::F32)?)?.abs()?.flatten_all()?.max(0)?.to_scalar::<f32>()?;
                let n = y.elem_count() as f32;
                println!("{c} {:?} -> {:?}: input bytes differ {qd}; vs ComfyUI {:.4}% differ (max |d| {maxd:e}); \
                          ours vs exact {:.2}%, ComfyUI vs exact {:.2}%  [{:.1} ms incl. first-call setup]",
                    x.dims(), y.dims(), 100.0 * count(&y, &theirs)? / n, 100.0 * count(&y, &exact)? / n,
                    100.0 * count(&theirs, &exact)? / n, secs * 1e3);
            }
            Ok(())
        }
        Some("debug-ltx-attn") => {
            // debug-ltx-attn REF2.safetensors: our flash attention on the q, k, v each attention call
            // in ComfyUI's blocks received (dit_ref2.py), against what PyTorch's SDPA returned
            use candle_core::DType;
            let dev = candle_core::Device::new_cuda(0)?;
            let r = unsafe { candle_core::safetensors::MmapedSafetensors::new(args.get(2).context("reference")?)? };
            let subs = ["attn1", "attn2", "audio_attn1", "audio_attn2", "audio_to_video_attn", "video_to_audio_attn"];
            for b in [0, 2] {
                for sub in subs {
                    let key = format!("b{b}.{sub}.sdpa0");
                    let Ok(q) = r.load(&format!("{key}.q"), &dev) else { continue };
                    let (k, v, theirs) = (r.load(&format!("{key}.k"), &dev)?, r.load(&format!("{key}.v"), &dev)?, r.load(&format!("{key}.out"), &dev)?);
                    let (bsz, tq, inner) = q.dims3()?;
                    let tk = k.dim(1)?;
                    let d = inner / 32;
                    let scale = (1.0f64 / (d as f64).sqrt()) as f32;
                    let ours = candle_flash_attn::flash_attn(&q.reshape((bsz, tq, 32, d))?, &k.reshape((bsz, tk, 32, d))?, &v.reshape((bsz, tk, 32, d))?, scale, false)?
                        .reshape((bsz, tq, inner))?;
                    let differ = ours.ne(&theirs)?.to_dtype(DType::F32)?.sum_all()?.to_scalar::<f32>()?;
                    let maxd = (ours.to_dtype(DType::F32)? - theirs.to_dtype(DType::F32)?)?.abs()?.flatten_all()?.max(0)?.to_scalar::<f32>()?;
                    println!("{key}: q {tq} x k {tk}, head dim {d}: {differ} of {} differ, max |d| {maxd:e}", ours.elem_count());
                }
            }
            Ok(())
        }
        Some("debug-ltx-block") => {
            // debug-ltx-block CHECKPOINT REF2.safetensors: LTX-2.3's DiT blocks 0 and 2, piece by
            // piece and whole, fed what ComfyUI fed them (dit_ref2.py), against its results
            ltx::debug_block(args.get(2).context("checkpoint")?, args.get(3).context("reference")?)
        }
        Some("debug-ltx-io") => {
            // debug-ltx-io CHECKPOINT REF2.safetensors: LTX-2.3's timestep embeddings, patchify,
            // positions and output head against ComfyUI's (dit_ref2.py)
            ltx::debug_io(args.get(2).context("checkpoint")?, args.get(3).context("reference")?)
        }
        Some("debug-ltx-forward") => {
            // debug-ltx-forward CHECKPOINT REF2.safetensors: one whole LTX-2.3 DiT call (blocks
            // streamed) from ComfyUI's inputs, against its output (dit_ref2.py)
            ltx::debug_forward(args.get(2).context("checkpoint")?, args.get(3).context("reference")?)
        }
        Some("debug-ltx-batch") => {
            // debug-ltx-batch CHECKPOINT REF: the server's batched DiT call (CFG's negative and
            // prompt together) from its inputs, stage by stage (dit_batch_ref.py)
            ltx::debug_batch(args.get(2).context("checkpoint")?, args.get(3).context("reference")?)
        }
        Some("debug-ltx-connector") => {
            // debug-ltx-connector CHECKPOINT REF2.safetensors: LTX-2.3's two text connectors on
            // ComfyUI's projected text, against its output (dit_ref2.py)
            ltx::debug_connector(args.get(2).context("checkpoint")?, args.get(3).context("reference")?)
        }
        Some("debug-ltx-sample") => {
            // debug-ltx-sample CHECKPOINT REF2 SAMPLE_REF: LTX-2.3's scheduler, noise and euler
            // steps against ComfyUI's (sample_ref.py), from its text context
            ltx::debug_sample(args.get(2).context("checkpoint")?, args.get(3).context("ref2")?, args.get(4).context("sample reference")?)
        }
        Some("debug-randn") => {
            let seed: u64 = arg(&args, "--seed").unwrap_or("0").parse()?;
            let n: usize = arg(&args, "--n").unwrap_or("16").parse()?;
            let v = if args.iter().any(|a| a == "--cuda") {
                let sms: u32 = arg(&args, "--sms").unwrap_or("66").parse()?;
                let calls: usize = arg(&args, "--calls").unwrap_or("1").parse()?;
                let mut g = rng::CudaRandn::new(seed, sms, 1536);
                let mut v = vec![];
                for _ in 0..calls {
                    v = g.randn(n);
                }
                v
            } else {
                rng::torch_randn_cpu(seed, n)
            };
            let sum: f64 = v.iter().map(|x| *x as f64).sum();
            println!("first {:?}\nsum {sum:.9}", &v[..8.min(v.len())]);
            Ok(())
        }
        _ => bail!("usage: rustyui-backend serve [--listen ADDR] [--idle S] [--exit-idle S] [--models DIR] [--out DIR] [--frontend DIR] [--prefix NAME] [--mirror HOST:DIR] [--writer GGUF] [--no-video] | write IDEA --model GGUF | run GRAPH.json --out DIR | from-png IMG.png --out DIR | compare A.png B.png | tokens TEXT"),
    }
}
