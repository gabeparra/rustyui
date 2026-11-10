//! ComfyUI's routes for files, as server.py and api_server/routes/internal answer them: /view for
//! the input, temp and output folders, /upload/image, the model folders (/models,
//! /view_metadata, /internal/folder_paths) and /internal/files. The input and temp folders are
//! ComfyUI's own, beside `models/`, so a picture uploaded here is there for ComfyUI too.

use super::*;

/// folder_paths.supported_pt_extensions
const PT: &[&str] = &[".ckpt", ".pt", ".pt2", ".bin", ".pth", ".safetensors", ".pkl", ".sft"];

/// folder_paths' own folders, in its order: (name, folders below models/, extensions; none = any file)
const FOLDERS: &[(&str, &[&str], &[&str])] = &[
    ("checkpoints", &["checkpoints"], PT),
    ("configs", &["configs"], &[".yaml"]),
    ("loras", &["loras"], PT),
    ("vae", &["vae"], PT),
    ("text_encoders", &["text_encoders", "clip"], PT),
    ("diffusion_models", &["unet", "diffusion_models"], PT),
    ("clip_vision", &["clip_vision"], PT),
    ("style_models", &["style_models"], PT),
    ("embeddings", &["embeddings"], PT),
    ("diffusers", &["diffusers"], &[]),
    ("vae_approx", &["vae_approx"], PT),
    ("controlnet", &["controlnet", "t2i_adapter"], PT),
    ("gligen", &["gligen"], PT),
    ("upscale_models", &["upscale_models"], PT),
    ("latent_upscale_models", &["latent_upscale_models"], PT),
    // custom node packs and datasets have no folder here: rustyui's nodes are its own
    ("custom_nodes", &[], &[]),
    ("datasets", &[], &[]),
    ("hypernetworks", &["hypernetworks"], PT),
    ("photomaker", &["photomaker"], PT),
    ("classifiers", &["classifiers"], &[]),
    ("model_patches", &["model_patches"], PT),
    ("audio_encoders", &["audio_encoders"], PT),
    ("background_removal", &["background_removal"], PT),
    ("frame_interpolation", &["frame_interpolation"], PT),
    ("geometry_estimation", &["geometry_estimation"], PT),
    ("optical_flow", &["optical_flow"], PT),
    ("detection", &["detection"], PT),
];

/// folder_paths.map_legacy
fn legacy(name: &str) -> &str {
    match name {
        "unet" => "diffusion_models",
        "clip" => "text_encoders",
        n => n,
    }
}

/// A model folder's directories and extensions: folder_paths' own, or (as a custom node pack
/// registers one) a folder of that name under models/
fn folder(srv: &Server, name: &str) -> Option<(Vec<PathBuf>, &'static [&'static str])> {
    let name = legacy(name);
    if let Some((_, dirs, exts)) = FOLDERS.iter().find(|(n, _, _)| *n == name) {
        return Some((dirs.iter().map(|d| srv.models.join(d)).collect(), exts));
    }
    let dir = srv.models.join(contained(name)?);
    dir.is_dir().then(|| (vec![dir], PT))
}

/// Every folder name: folder_paths' own, then the other folders under models/
fn folder_names(srv: &Server) -> Vec<String> {
    let taken: Vec<&str> = FOLDERS.iter().flat_map(|(n, dirs, _)| std::iter::once(*n).chain(dirs.iter().copied())).collect();
    let mut extra: Vec<String> = fs::read_dir(&srv.models)
        .map(|rd| rd.flatten().filter(|e| e.path().is_dir()).map(|e| e.file_name().to_string_lossy().to_string()).collect())
        .unwrap_or_default();
    extra.retain(|n| !taken.contains(&n.as_str()) && !n.starts_with('.'));
    extra.sort();
    FOLDERS.iter().map(|(n, _, _)| n.to_string()).chain(extra).collect()
}

/// folder_paths.get_filename_list: every file of the folder with one of its extensions, as
/// relative paths, sorted
fn filenames(dirs: &[PathBuf], exts: &[&str]) -> Vec<String> {
    let mut out: Vec<String> = dirs.iter().flat_map(|d| list(d, if exts.is_empty() { &[""] } else { exts })).collect();
    out.sort();
    out.dedup();
    out
}

/// ComfyUI's folders by type: input and temp beside models/ as ComfyUI's are, output where
/// renders go
fn dir_by_type(srv: &Server, ty: &str) -> Option<PathBuf> {
    let base = srv.models.parent().unwrap_or(&srv.models);
    match ty {
        "output" => Some(srv.out.clone()),
        "input" => Some(base.join("input")),
        "temp" => Some(base.join("temp")),
        _ => None,
    }
}

/// folder_paths.annotated_filepath: "name.png [input]" names its folder
fn annotated(name: &str) -> (&str, Option<&'static str>) {
    for (suffix, ty) in [(" [output]", "output"), (" [input]", "input"), (" [temp]", "temp")] {
        if let Some(n) = name.strip_suffix(suffix) {
            return (n, Some(ty));
        }
    }
    (name, None)
}

/// os.path.splitext
fn split_ext(name: &str) -> (&str, &str) {
    match name.rfind('.') {
        Some(i) if name[..i].chars().any(|c| c != '.') => (&name[..i], &name[i..]),
        _ => (name, ""),
    }
}

/// The parts of a multipart/form-data body: (name, file name, content)
fn form_parts<'a>(ctype: &str, body: &'a [u8]) -> Option<Vec<(String, Option<String>, &'a [u8])>> {
    let boundary = ctype.split(';').map(str::trim).find_map(|p| p.strip_prefix("boundary="))?.trim_matches('"');
    let delim = format!("--{boundary}").into_bytes();
    let find = |from: usize, needle: &[u8]| body.get(from..)?.windows(needle.len()).position(|w| w == needle).map(|i| i + from);
    let between = [b"\r\n".as_slice(), &delim].concat();
    let mut pos = find(0, &delim)? + delim.len();
    let mut parts = vec![];
    while !body.get(pos..)?.starts_with(b"--") {
        pos += if body[pos..].starts_with(b"\r\n") { 2 } else { 0 };
        let head_end = find(pos, b"\r\n\r\n")?;
        let head = std::str::from_utf8(&body[pos..head_end]).ok()?;
        let end = find(head_end + 4, &between)?;
        let (mut name, mut file) = (None, None);
        for line in head.split("\r\n").filter(|l| l.to_ascii_lowercase().starts_with("content-disposition:")) {
            for p in line.split(';').skip(1).map(str::trim) {
                if let Some(v) = p.strip_prefix("name=") {
                    name = Some(v.trim_matches('"').to_string());
                } else if let Some(v) = p.strip_prefix("filename=") {
                    file = Some(v.trim_matches('"').to_string());
                }
            }
        }
        parts.push((name?, file, &body[head_end + 4..end]));
        pos = end + between.len();
    }
    Some(parts)
}

/// POST /upload/image: image_upload. The file goes to the folder its type names (input unless
/// said), in its subfolder; a different file of the same name is kept and this one numbered
/// "name (1).png", unless overwrite is asked; the same file again is not written twice.
fn upload_image(srv: &Server, ctype: &str, body: &[u8]) -> (u16, Value) {
    let Some(parts) = form_parts(ctype, body) else { return (400, json!({"error": "not a multipart form"})) };
    let text = |n: &str| parts.iter().find(|p| p.0 == n).map(|p| String::from_utf8_lossy(p.2).to_string());
    let Some((_, Some(filename), data)) = parts.iter().find(|p| p.0 == "image") else { return (400, json!({"error": "no image"})) };
    let ty = text("type").filter(|t| !t.is_empty()).unwrap_or_else(|| "input".into());
    let ty = if ty == "temp" || ty == "output" { ty } else { "input".to_string() };
    let subfolder = text("subfolder").unwrap_or_default();
    let Some(base) = dir_by_type(srv, &ty) else { return (400, json!({"error": "no such folder"})) };
    // one plain name, in a subfolder that stays below the folder
    let plain = contained(filename).is_some_and(|p| p.components().count() == 1);
    let dir = match (subfolder.is_empty(), contained(&subfolder)) {
        (true, _) => Some(base.clone()),
        (false, Some(s)) => Some(base.join(s)),
        (false, None) => None,
    };
    let (Some(dir), true) = (dir, plain) else { return (400, json!({"error": "that name leaves the folder"})) };
    if fs::create_dir_all(&dir).is_err() {
        return (500, json!({"error": "cannot make the folder"}));
    }
    let overwrite = matches!(text("overwrite").as_deref(), Some("true") | Some("1"));
    let (stem, ext) = split_ext(filename);
    let mut name = filename.clone();
    let mut same = false;
    if !overwrite {
        let mut i = 1;
        while dir.join(&name).exists() {
            if fs::read(dir.join(&name)).is_ok_and(|b| b == *data) {
                same = true;
                break;
            }
            name = format!("{stem} ({i}){ext}");
            i += 1;
        }
    }
    if !same {
        let tmp = dir.join(format!(".rustyui-upload-{}.tmp", std::process::id()));
        if fs::write(&tmp, data).and_then(|_| fs::rename(&tmp, dir.join(&name))).is_err() {
            let _ = fs::remove_file(&tmp);
            return (500, json!({"error": "cannot write the file"}));
        }
    }
    (200, json!({"name": name, "subfolder": subfolder, "type": ty}))
}

/// A safetensors file's __metadata__ (view_metadata), if its header is under 1 MiB as ComfyUI asks
fn safetensors_metadata(path: &Path) -> Option<Value> {
    let mut f = fs::File::open(path).ok()?;
    let mut n = [0u8; 8];
    f.read_exact(&mut n).ok()?;
    let n = u64::from_le_bytes(n);
    if n > 1 << 20 {
        return None;
    }
    let mut head = vec![0u8; n as usize];
    f.read_exact(&mut head).ok()?;
    serde_json::from_slice::<Value>(&head).ok()?.get("__metadata__").cloned()
}

/// The routes of this file; Err hands the request back when `api` is not one of them
pub(super) fn route(
    srv: &Server,
    req: tiny_http::Request,
    method: &str,
    api: &str,
    q: &str,
    body: &[u8],
) -> Result<(), tiny_http::Request> {
    match (method, api) {
        ("GET", "/view") => {
            let filename = query(q, "filename");
            let (name, named) = annotated(&filename);
            // the name without its folders, as ComfyUI's os.path.basename
            let name = name.rsplit('/').next().unwrap_or(name);
            let ty = named
                .map(str::to_string)
                .unwrap_or_else(|| Some(query(q, "type")).filter(|t| !t.is_empty()).unwrap_or_else(|| "output".into()));
            let Some(dir) = dir_by_type(srv, &ty) else { return Ok(reply(req, 400, "text/plain", vec![])) };
            match picture(&dir, name, &query(q, "subfolder")) {
                Some(file) => {
                    let ctype = match file.extension().and_then(|e| e.to_str()).map(str::to_ascii_lowercase).as_deref() {
                        Some("jpg") | Some("jpeg") => "image/jpeg",
                        Some("webp") => "image/webp",
                        Some("gif") => "image/gif",
                        Some("mp4") => "video/mp4",
                        Some("webm") => "video/webm",
                        _ => "image/png",
                    };
                    match fs::read(&file) {
                        Ok(b) => user_routes::reply_headers(
                            req,
                            200,
                            ctype,
                            b,
                            &[("Content-Disposition", &format!("filename=\"{}\"", name.replace('\\', "\\\\").replace('"', "\\\"")))],
                        ),
                        Err(_) => json_reply(req, 404, json!({"error": "not readable"})),
                    }
                }
                None => json_reply(req, 404, json!({"error": "not a picture in that folder"})),
            }
        }
        ("POST", "/upload/image") => {
            let ctype = header(&req, "Content-Type").unwrap_or("").to_string();
            let (code, v) = upload_image(srv, &ctype, body);
            json_reply(req, code, v)
        }
        ("GET", "/models") => json_reply(req, 200, json!(folder_names(srv))),
        ("GET", p) if p.starts_with("/models/") => match folder(srv, &percent_decode(&p["/models/".len()..])) {
            Some((dirs, exts)) => json_reply(req, 200, json!(filenames(&dirs, exts))),
            None => reply(req, 404, "text/plain", vec![]),
        },
        ("GET", p) if p.starts_with("/view_metadata/") => {
            let filename = query(q, "filename");
            let found = folder(srv, &percent_decode(&p["/view_metadata/".len()..]))
                .filter(|_| filename.ends_with(".safetensors"))
                .and_then(|(dirs, _)| dirs.iter().find_map(|d| Some(d.join(contained(&filename)?)).filter(|f| f.is_file())))
                .and_then(|f| safetensors_metadata(&f));
            match found {
                Some(meta) => json_reply(req, 200, meta),
                None => reply(req, 404, "text/plain", vec![]),
            }
        }
        ("GET", "/internal/folder_paths") => {
            let map: serde_json::Map<String, Value> = FOLDERS
                .iter()
                .map(|(n, dirs, _)| (n.to_string(), json!(dirs.iter().map(|d| srv.models.join(d)).collect::<Vec<_>>())))
                .collect();
            json_reply(req, 200, Value::Object(map))
        }
        ("GET", p) if p.starts_with("/internal/files/") => {
            let ty = &p["/internal/files/".len()..];
            let dir = ["output", "input", "temp"].contains(&ty).then(|| dir_by_type(srv, ty)).flatten();
            let Some(dir) = dir else { return Ok(json_reply(req, 400, json!({"error": "Invalid directory type"}))) };
            // the folder's own files (not its subfolders), newest first
            let mut files: Vec<(SystemTime, String)> = fs::read_dir(&dir)
                .map(|rd| {
                    rd.flatten()
                        .filter(|e| e.file_type().is_ok_and(|t| t.is_file()) && !e.file_name().to_string_lossy().starts_with('.'))
                        .map(|e| {
                            (e.metadata().and_then(|m| m.modified()).unwrap_or(UNIX_EPOCH), e.file_name().to_string_lossy().to_string())
                        })
                        .collect()
                })
                .unwrap_or_default();
            files.sort_by_key(|f| std::cmp::Reverse(f.0));
            json_reply(req, 200, json!(files.into_iter().map(|(_, n)| format!("{n} [{ty}]")).collect::<Vec<_>>()))
        }
        _ => return Err(req),
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn multipart_forms() {
        let body = b"--XyZ\r\nContent-Disposition: form-data; name=\"image\"; filename=\"a b.png\"\r\nContent-Type: image/png\r\n\r\n\x89PNG\r\n--x\r\n--XyZ\r\nContent-Disposition: form-data; name=\"overwrite\"\r\n\r\ntrue\r\n--XyZ--\r\n";
        let parts = form_parts("multipart/form-data; boundary=XyZ", body).unwrap();
        assert_eq!(parts.len(), 2);
        assert_eq!((parts[0].0.as_str(), parts[0].1.as_deref(), parts[0].2), ("image", Some("a b.png"), b"\x89PNG\r\n--x".as_slice()));
        assert_eq!((parts[1].0.as_str(), parts[1].1.as_deref(), parts[1].2), ("overwrite", None, b"true".as_slice()));
        assert!(form_parts("multipart/form-data; boundary=other", body).is_none());
    }

    #[test]
    fn names_as_python_splits_them() {
        assert_eq!(split_ext("a.b.png"), ("a.b", ".png"));
        assert_eq!(split_ext("noext"), ("noext", ""));
        assert_eq!(split_ext(".hidden"), (".hidden", ""));
        assert_eq!(annotated("x.png [input]"), ("x.png", Some("input")));
        assert_eq!(annotated("x.png"), ("x.png", None));
        assert_eq!(legacy("unet"), "diffusion_models");
    }
}
