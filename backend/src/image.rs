//! Pixels out the way ComfyUI's SaveImage makes them, and two PNGs compared by numbers.

use anyhow::{bail, Context, Result};
use candle_core::Tensor;
use std::fs;
use std::io::BufWriter;
use std::path::{Path, PathBuf};

/// JSON with every non-ASCII character escaped (\uXXXX), as Python's json.dumps writes it:
/// a PNG text chunk is Latin-1, so a prompt with an emoji in it would otherwise fail to save.
pub fn ascii_json(v: &serde_json::Value) -> String {
    let mut out = String::new();
    for c in v.to_string().chars() {
        if c.is_ascii() {
            out.push(c);
        } else {
            let mut units = [0u16; 2];
            for u in c.encode_utf16(&mut units) {
                out.push_str(&format!("\\u{:04x}", u));
            }
        }
    }
    out
}

/// (b, 3, h, w) in [0, 1] -> `trunc(255 * clip(x))` (SaveImage's numpy cast), written as
/// `{prefix}_{n:05}_.png` with the graph in a "prompt" text chunk, like ComfyUI, and one more
/// chunk per `extra` entry (the request's extra_pnginfo, ComfyUI's "workflow" and the like).
pub fn save_png(img: &Tensor, dir: &Path, prefix: &str, graph: &str, extra: &[(String, String)]) -> Result<Vec<PathBuf>> {
    let (b, c, h, w) = img.dims4()?;
    anyhow::ensure!(c == 3, "expected 3 channels, got {c}");
    let data = img.permute((0, 2, 3, 1))?.flatten_all()?.to_vec1::<f32>()?;
    // inside `dir`, as ComfyUI's SaveImage insists ("Saving image outside the output folder is
    // not allowed"): a relative prefix without `..`, under no symlink that leads out of `dir`
    let rel = crate::paths::contained(prefix).with_context(|| format!("filename_prefix {prefix:?} leaves the output folder"))?;
    let target = dir.join(rel);
    let folder = target.parent().unwrap_or(dir).to_path_buf();
    let stem = target.file_name().and_then(|s| s.to_str()).unwrap_or("rustyui").to_string();
    fs::create_dir_all(dir)?;
    if !crate::paths::below(dir, &folder) {
        bail!("filename_prefix {prefix:?} leads out of {}", dir.display());
    }
    fs::create_dir_all(&folder)?;
    let mut counter = 1;
    for entry in fs::read_dir(&folder)? {
        let name = entry?.file_name().to_string_lossy().to_string();
        if let Some(rest) = name.strip_prefix(&format!("{stem}_")) {
            if let Some(num) = rest.strip_suffix("_.png").and_then(|n| n.parse::<usize>().ok()) {
                counter = counter.max(num + 1);
            }
        }
    }
    let mut out = vec![];
    for i in 0..b {
        let px: Vec<u8> = data[i * h * w * 3..(i + 1) * h * w * 3].iter().map(|&x| (255.0f32 * x).clamp(0.0, 255.0) as u8).collect();
        let path = folder.join(format!("{stem}_{counter:05}_.png"));
        counter += 1;
        let file = BufWriter::new(fs::File::create(&path)?);
        let mut enc = png::Encoder::new(file, w as u32, h as u32);
        enc.set_color(png::ColorType::Rgb);
        enc.set_depth(png::BitDepth::Eight);
        enc.add_text_chunk("prompt".to_string(), graph.to_string())?;
        enc.add_text_chunk("generator".to_string(), "rustyui-backend".to_string())?;
        for (key, text) in extra {
            enc.add_text_chunk(key.clone(), text.clone())?;
        }
        let mut writer = enc.write_header()?;
        writer.write_image_data(&px)?;
        writer.finish()?;
        out.push(path);
    }
    Ok(out)
}

/// 8-bit RGB as a PNG in memory, with no text chunks (a thumbnail carries no graph).
pub fn encode_png(rgb: &[u8], w: u32, h: u32) -> Result<Vec<u8>> {
    let mut out = vec![];
    let mut enc = png::Encoder::new(&mut out, w, h);
    enc.set_color(png::ColorType::Rgb);
    enc.set_depth(png::BitDepth::Eight);
    enc.set_compression(png::Compression::Fast);
    let mut writer = enc.write_header()?;
    writer.write_image_data(rgb)?;
    writer.finish()?;
    Ok(out)
}

pub struct Decoded {
    pub width: u32,
    pub height: u32,
    pub rgb: Vec<u8>,
    pub texts: Vec<(String, String)>,
}

pub fn read_png(path: &Path) -> Result<Decoded> {
    let dec = png::Decoder::new(std::io::BufReader::new(fs::File::open(path).with_context(|| path.display().to_string())?));
    let mut reader = dec.read_info()?;
    let mut buf = vec![0; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf)?;
    let texts: Vec<(String, String)> = reader
        .info()
        .uncompressed_latin1_text
        .iter()
        .map(|t| (t.keyword.clone(), t.text.clone()))
        .chain(reader.info().utf8_text.iter().filter_map(|t| t.get_text().ok().map(|s| (t.keyword.clone(), s))))
        .collect();
    let channels = match info.color_type {
        png::ColorType::Rgb => 3,
        png::ColorType::Rgba => 4,
        other => bail!("{}: color type {other:?} not handled", path.display()),
    };
    let rgb: Vec<u8> = buf[..info.buffer_size()].chunks(channels).flat_map(|p| p[..3].to_vec()).collect();
    Ok(Decoded { width: info.width, height: info.height, rgb, texts })
}

/// Objective differences only: mean and max absolute difference per channel value, PSNR,
/// and how many values differ at all or by more than 8 levels.
pub fn compare(a: &Path, b: &Path) -> Result<String> {
    let (x, y) = (read_png(a)?, read_png(b)?);
    if (x.width, x.height) != (y.width, y.height) {
        bail!("sizes differ: {}x{} vs {}x{}", x.width, x.height, y.width, y.height);
    }
    let n = x.rgb.len() as f64;
    let mut sum = 0f64;
    let mut sq = 0f64;
    let mut max = 0u8;
    let mut differ = 0usize;
    let mut over8 = 0usize;
    for (p, q) in x.rgb.iter().zip(&y.rgb) {
        let d = p.abs_diff(*q);
        sum += d as f64;
        sq += (d as f64) * (d as f64);
        max = max.max(d);
        differ += (d > 0) as usize;
        over8 += (d > 8) as usize;
    }
    let mse = sq / n;
    let psnr = if mse == 0.0 { f64::INFINITY } else { 10.0 * (255.0f64 * 255.0 / mse).log10() };
    Ok(format!(
        "{}x{}  mean |diff| {:.3}  max {}  PSNR {:.2} dB  values differing {:.2}%  by >8 levels {:.3}%",
        x.width,
        x.height,
        sum / n,
        max,
        psnr,
        100.0 * differ as f64 / n,
        100.0 * over8 as f64 / n
    ))
}

#[cfg(test)]
mod tests {
    #[test]
    fn png_text_is_ascii_json() {
        let (wolf, e) = (char::from_u32(0x1F43A).unwrap(), char::from_u32(0xE9).unwrap());
        let v = serde_json::json!({"text": format!("wolf {wolf} caf{e} \"quoted\"")});
        let s = super::ascii_json(&v);
        assert!(s.is_ascii(), "{s}");
        // two UTF-16 surrogates for the wolf, one for the e, two escaped quotes
        assert_eq!(s.matches('\\').count(), 5, "{s}");
        assert!(s.contains("d83d") && s.contains("dc3a") && s.contains("00e9"), "{s}");
        assert_eq!(serde_json::from_str::<serde_json::Value>(&s).unwrap(), v);
    }
}
