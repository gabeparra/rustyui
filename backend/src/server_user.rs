//! ComfyUI's routes for the editor's own state, as app/user_manager.py, app/app_settings.py,
//! server.py and api_server/routes/internal answer them: users, settings, user files (saved
//! workflows, user.css), the editor's extension scripts, feature flags, node translations,
//! subgraph blueprints, node replacements, custom node templates and the log panel. The user folder is
//! ComfyUI's own (`<ComfyUI>/user/default`, beside `models/`), so workflows and settings saved in
//! ComfyUI open here and the other way round. One user, as ComfyUI without --multi-user.

use super::*;

/// ComfyUI's single user's folder
fn user_dir(srv: &Server) -> PathBuf {
    srv.models.parent().unwrap_or(&srv.models).join("user").join("default")
}

/// A user file named in a URL (percent-encoded, "/" as %2F), kept inside the user folder
fn user_path(srv: &Server, encoded: &str) -> Option<PathBuf> {
    let name = percent_decode(encoded);
    let rel = contained(&name)?;
    Some(user_dir(srv).join(rel))
}

fn file_info(path: &Path, base: &Path) -> Value {
    let meta = fs::metadata(path).ok();
    let ms = |t: Option<std::io::Result<SystemTime>>| -> u64 {
        t.and_then(|r| r.ok()).and_then(|t| t.duration_since(UNIX_EPOCH).ok()).map(|d| d.as_millis() as u64).unwrap_or(0)
    };
    json!({
        "path": path.strip_prefix(base).unwrap_or(path).to_string_lossy().replace('\\', "/"),
        "size": meta.as_ref().map(|m| m.len()).unwrap_or(0),
        "modified": ms(meta.as_ref().map(|m| m.modified())),
        "created": ms(meta.as_ref().map(|m| m.created())),
    })
}

fn read_settings(srv: &Server) -> serde_json::Map<String, Value> {
    fs::read_to_string(user_dir(srv).join("comfy.settings.json")).ok().and_then(|s| serde_json::from_str(&s).ok()).unwrap_or_default()
}

/// comfy.settings.json as ComfyUI writes it (json.dumps(indent=4))
fn save_settings(srv: &Server, settings: &serde_json::Map<String, Value>) -> std::io::Result<()> {
    let dir = user_dir(srv);
    fs::create_dir_all(&dir)?;
    let mut out = vec![];
    let fmt = serde_json::ser::PrettyFormatter::with_indent(b"    ");
    let mut ser = serde_json::Serializer::with_formatter(&mut out, fmt);
    serde::Serialize::serialize(settings, &mut ser).map_err(std::io::Error::other)?;
    fs::write(dir.join("comfy.settings.json"), out)
}

/// Every file below `dir`, recursively or not
fn walk_files(dir: &Path, recurse: bool, out: &mut Vec<PathBuf>) {
    let Ok(rd) = fs::read_dir(dir) else { return };
    for e in rd.flatten() {
        let p = e.path();
        if p.is_dir() {
            if recurse {
                walk_files(&p, true, out);
            }
        } else if p.is_file() {
            out.push(p);
        }
    }
}

/// The editor's own extension scripts (`extensions/**/*.js` of its build), as /extensions lists them
fn extension_scripts(srv: &Server) -> Vec<String> {
    let mut out: Vec<String> = match srv.frontend.as_deref() {
        Some(dir) => {
            let mut files = vec![];
            walk_files(&dir.join("extensions"), true, &mut files);
            files
                .into_iter()
                .filter(|p| p.extension().is_some_and(|e| e == "js"))
                .filter_map(|p| p.strip_prefix(dir).ok().map(|r| format!("/{}", r.to_string_lossy())))
                .collect()
        }
        None => editor::EDITOR
            .iter()
            .filter(|(p, _)| p.starts_with("extensions/") && p.ends_with(".js"))
            .map(|(p, _)| format!("/{p}"))
            .collect(),
    };
    out.sort();
    out
}

const MODEL_EXTENSIONS: [&str; 8] = [".bin", ".ckpt", ".pkl", ".pt", ".pt2", ".pth", ".safetensors", ".sft"];

/// The card's name and memory (total, free) from nvidia-smi, kept 5 s: system_stats is polled, and
/// asking CUDA itself would hold a context (and its VRAM) open in an idle backend
fn gpu_info() -> (String, u64, u64) {
    static CACHE: Mutex<Option<(Instant, (String, u64, u64))>> = Mutex::new(None);
    let mut c = CACHE.lock().unwrap();
    if let Some((t, v)) = c.as_ref() {
        if t.elapsed() < Duration::from_secs(5) {
            return v.clone();
        }
    }
    let out = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=name,memory.total,memory.free", "--format=csv,noheader,nounits"])
        .output()
        .ok()
        .map(|o| String::from_utf8_lossy(&o.stdout).to_string())
        .unwrap_or_default();
    let f: Vec<&str> = out.lines().next().unwrap_or("").split(',').map(str::trim).collect();
    let mib = |s: Option<&&str>| s.and_then(|v| v.parse::<u64>().ok()).unwrap_or(0) << 20;
    let v = (f.first().map(|s| s.to_string()).unwrap_or_default(), mib(f.get(1)), mib(f.get(2)));
    *c = Some((Instant::now(), v.clone()));
    v
}

/// MemTotal and MemAvailable from /proc/meminfo, in bytes
fn ram_info() -> (u64, u64) {
    let info = fs::read_to_string("/proc/meminfo").unwrap_or_default();
    let field = |name: &str| -> u64 {
        info.lines()
            .find(|l| l.starts_with(name))
            .and_then(|l| l.split_whitespace().nth(1))
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(0)
            * 1024
    };
    (field("MemTotal:"), field("MemAvailable:"))
}

/// ComfyUI's /system_stats, filled in for rustyui (the ComfyUI version is the one whose API it answers)
fn system_stats() -> Value {
    let (ram_total, ram_free) = ram_info();
    let (gpu, vram_total, vram_free) = gpu_info();
    json!({
        "system": {
            "os": "linux", "ram_total": ram_total, "ram_free": ram_free,
            "comfyui_version": "0.39.0", "required_frontend_version": "1.55.14",
            "installed_templates_version": "", "required_templates_version": "", "comfy_package_versions": [],
            "python_version": format!("rustyui-backend {}", env!("CARGO_PKG_VERSION")), "pytorch_version": "candle 0.11",
            "embedded_python": false, "deploy_environment": "rustyui",
            "argv": std::env::args().collect::<Vec<_>>(),
        },
        "devices": [{
            "name": format!("cuda:0 {gpu}"), "type": "cuda", "index": 0,
            "vram_total": vram_total, "vram_free": vram_free, "torch_vram_total": 0, "torch_vram_free": 0,
        }],
    })
}

/// A reply with headers beyond the usual ones (a user file goes out as a download)
pub(super) fn reply_headers(req: tiny_http::Request, code: u16, ctype: &str, body: Vec<u8>, extra: &[(&str, &str)]) {
    let h = |k: &str, v: &str| tiny_http::Header::from_bytes(k.as_bytes(), v.as_bytes()).unwrap();
    let mut resp = tiny_http::Response::from_data(body)
        .with_status_code(code)
        .with_header(h("Content-Type", ctype))
        .with_header(h("Cache-Control", "no-store"));
    for (k, v) in extra {
        resp = resp.with_header(h(k, v));
    }
    let _ = req.respond(resp);
}

/// ComfyUI's feature flags (/features, and the websocket's feature_flags), with what rustyui does
/// not do turned off: no latent previews yet, no model type tags, no manager, no node
/// replacements, no asset database
pub(super) fn features() -> Value {
    json!({"supports_preview_metadata": false, "supports_model_type_tags": false, "max_upload_size": 104857600,
           "extension": {}, "node_replacements": false, "assets": false})
}

fn query_flag(q: &str, key: &str, default: bool) -> bool {
    match query(q, key).to_lowercase().as_str() {
        "" => default,
        v => v == "true",
    }
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
        ("GET", "/users") => json_reply(req, 200, json!({"storage": "server", "migrated": user_dir(srv).exists()})),
        ("POST", "/users") => json_reply(req, 400, json!({"error": "rustyui has one user"})),
        ("GET", "/settings") => json_reply(req, 200, Value::Object(read_settings(srv))),
        ("GET", p) if p.starts_with("/settings/") => {
            let id = percent_decode(&p["/settings/".len()..]);
            json_reply(req, 200, read_settings(srv).get(&id).cloned().unwrap_or(Value::Null))
        }
        ("POST", "/settings") => match serde_json::from_slice::<serde_json::Map<String, Value>>(body) {
            Ok(new) => {
                let mut s = read_settings(srv);
                s.extend(new);
                match save_settings(srv, &s) {
                    Ok(()) => reply(req, 200, "text/plain", vec![]),
                    Err(e) => json_reply(req, 500, json!({"error": e.to_string()})),
                }
            }
            Err(_) => reply(req, 400, "text/plain", b"settings must be a JSON object".to_vec()),
        },
        ("POST", p) if p.starts_with("/settings/") => {
            let id = percent_decode(&p["/settings/".len()..]);
            match serde_json::from_slice::<Value>(body) {
                Ok(v) if !id.is_empty() => {
                    let mut s = read_settings(srv);
                    s.insert(id, v);
                    match save_settings(srv, &s) {
                        Ok(()) => reply(req, 200, "text/plain", vec![]),
                        Err(e) => json_reply(req, 500, json!({"error": e.to_string()})),
                    }
                }
                _ => reply(req, 400, "text/plain", vec![]),
            }
        }
        ("GET", "/userdata") => {
            let dir = query(q, "dir");
            if dir.is_empty() {
                return Ok(reply(req, 400, "text/plain", b"Directory not provided".to_vec()));
            }
            let Some(path) = user_path(srv, &dir) else { return Ok(reply(req, 403, "text/plain", b"Invalid directory".to_vec())) };
            if !path.exists() {
                return Ok(reply(req, 404, "text/plain", b"Directory not found".to_vec()));
            }
            let (recurse, full, split) =
                (query_flag(q, "recurse", false), query_flag(q, "full_info", false), query_flag(q, "split", false));
            let mut files = vec![];
            walk_files(&path, recurse, &mut files);
            files.sort();
            let list: Vec<Value> = files
                .iter()
                .map(|f| {
                    if full {
                        return file_info(f, &path);
                    }
                    let rel = f.strip_prefix(&path).unwrap_or(f).to_string_lossy().replace('\\', "/");
                    if split {
                        let mut parts = vec![json!(rel)];
                        parts.extend(rel.split('/').map(|s| json!(s)));
                        Value::Array(parts)
                    } else {
                        json!(rel)
                    }
                })
                .collect();
            json_reply(req, 200, Value::Array(list))
        }
        ("GET", "/v2/userdata") => {
            let rel = percent_decode(&query(q, "path"));
            let base = user_dir(srv);
            let path = if rel.is_empty() { Some(base.clone()) } else { contained(&rel).map(|r| base.join(r)) };
            match path {
                Some(p) if p.is_dir() => {
                    let mut items = vec![];
                    if let Ok(rd) = fs::read_dir(&p) {
                        for e in rd.flatten() {
                            let ep = e.path();
                            let relp = ep.strip_prefix(&base).unwrap_or(&ep).to_string_lossy().replace('\\', "/");
                            let name = e.file_name().to_string_lossy().to_string();
                            if ep.is_dir() {
                                items.push(json!({"name": name, "path": relp, "type": "directory"}));
                            } else {
                                let info = file_info(&ep, &base);
                                items.push(json!({"name": name, "path": relp, "type": "file", "size": info["size"], "modified": info["modified"].as_u64().map(|m| m as f64 / 1000.0)}));
                            }
                        }
                    }
                    json_reply(req, 200, Value::Array(items))
                }
                Some(p) if !p.exists() => reply(req, 404, "text/plain", b"Path not found".to_vec()),
                _ => reply(req, 400, "text/plain", b"Invalid path".to_vec()),
            }
        }
        ("POST", p) if p.starts_with("/userdata/") && p.contains("/move/") => {
            let rest = &p["/userdata/".len()..];
            let (from, to) = rest.split_once("/move/").unwrap_or((rest, ""));
            let (Some(src), Some(dst)) = (user_path(srv, from), user_path(srv, to)) else {
                return Ok(reply(req, 403, "text/plain", vec![]));
            };
            if !src.is_file() {
                return Ok(reply(req, 404, "text/plain", vec![]));
            }
            if !query_flag(q, "overwrite", true) && dst.exists() {
                return Ok(reply(req, 409, "text/plain", b"File already exists".to_vec()));
            }
            if let Some(d) = dst.parent() {
                let _ = fs::create_dir_all(d);
            }
            match fs::rename(&src, &dst) {
                Ok(()) if query_flag(q, "full_info", false) => json_reply(req, 200, file_info(&dst, &user_dir(srv))),
                Ok(()) => json_reply(req, 200, json!(dst.strip_prefix(user_dir(srv)).unwrap_or(&dst).to_string_lossy())),
                Err(e) => json_reply(req, 400, json!({"error": e.to_string()})),
            }
        }
        ("GET", p) if p.starts_with("/userdata/") => {
            let Some(path) = user_path(srv, &p["/userdata/".len()..]) else { return Ok(reply(req, 403, "text/plain", vec![])) };
            let is_css = path == user_dir(srv).join("user.css");
            match fs::read(&path) {
                // user files are downloads, never inline content, except the editor's user.css
                Ok(b) if is_css => reply(req, 200, "text/css", b),
                Err(_) if is_css => reply(req, 200, "text/css", vec![]),
                Ok(b) => reply_headers(
                    req,
                    200,
                    "application/octet-stream",
                    b,
                    &[("X-Content-Type-Options", "nosniff"), ("Content-Disposition", "attachment")],
                ),
                Err(_) => reply(req, 404, "text/plain", b"File not found".to_vec()),
            }
        }
        ("POST", p) if p.starts_with("/userdata/") => {
            let Some(path) = user_path(srv, &p["/userdata/".len()..]) else { return Ok(reply(req, 403, "text/plain", vec![])) };
            if !query_flag(q, "overwrite", true) && path.exists() {
                return Ok(reply(req, 409, "text/plain", b"File already exists".to_vec()));
            }
            let dir = path.parent().map(Path::to_path_buf).unwrap_or_else(|| user_dir(srv));
            let tmp = dir.join(format!(".rustyui-{}.tmp", std::process::id()));
            let written = fs::create_dir_all(&dir).and_then(|_| fs::write(&tmp, body)).and_then(|_| fs::rename(&tmp, &path));
            match written {
                Ok(()) if query_flag(q, "full_info", false) => json_reply(req, 200, file_info(&path, &user_dir(srv))),
                Ok(()) => json_reply(req, 200, json!(path.strip_prefix(user_dir(srv)).unwrap_or(&path).to_string_lossy())),
                Err(_) => {
                    let _ = fs::remove_file(&tmp);
                    reply(req, 400, "text/plain", b"Invalid filename".to_vec())
                }
            }
        }
        ("DELETE", p) if p.starts_with("/userdata/") => match user_path(srv, &p["/userdata/".len()..]) {
            Some(path) if path.is_file() => match fs::remove_file(&path) {
                Ok(()) => reply(req, 204, "text/plain", vec![]),
                Err(e) => json_reply(req, 500, json!({"error": e.to_string()})),
            },
            Some(_) => reply(req, 404, "text/plain", vec![]),
            None => reply(req, 403, "text/plain", vec![]),
        },
        ("GET", "/system_stats") => json_reply(req, 200, system_stats()),
        // the model library: every folder under models/, and a folder's files
        ("GET", "/experiment/models") => {
            let mut folders: Vec<Value> = fs::read_dir(&srv.models)
                .map(|rd| {
                    rd.flatten()
                        .filter(|e| e.path().is_dir())
                        .map(|e| json!({"name": e.file_name().to_string_lossy(), "folders": [e.path()], "extensions": MODEL_EXTENSIONS}))
                        .collect()
                })
                .unwrap_or_default();
            folders.sort_by_key(|v| v["name"].as_str().unwrap_or("").to_string());
            json_reply(req, 200, Value::Array(folders))
        }
        ("GET", p) if p.starts_with("/experiment/models/") => {
            let folder = percent_decode(&p["/experiment/models/".len()..]);
            let Some(dir) = contained(&folder).map(|r| srv.models.join(r)) else { return Ok(reply(req, 403, "text/plain", vec![])) };
            if !dir.is_dir() {
                return Ok(json_reply(req, 404, json!({"error": "no such model folder"})));
            }
            let mut files = vec![];
            walk_files(&dir, true, &mut files);
            files.sort();
            let list: Vec<Value> = files
                .iter()
                .filter(|f| f.extension().is_some_and(|e| MODEL_EXTENSIONS.contains(&format!(".{}", e.to_string_lossy()).as_str())))
                .map(|f| {
                    let info = file_info(f, &dir);
                    let secs = |v: &Value| v.as_u64().map(|m| m as f64 / 1000.0);
                    json!({"name": info["path"], "pathIndex": 0, "modified": secs(&info["modified"]), "created": secs(&info["created"]), "size": info["size"]})
                })
                .collect();
            json_reply(req, 200, Value::Array(list))
        }
        // the editor's stylesheet link: the user's own user.css, else an empty one
        ("GET", "/user.css") => reply(req, 200, "text/css", fs::read(user_dir(srv).join("user.css")).unwrap_or_default()),
        ("GET", "/extensions") => json_reply(req, 200, json!(extension_scripts(srv))),
        ("GET", "/features") => json_reply(req, 200, features()),
        // no custom node packs bring translations, blueprints, replacements or templates yet
        ("GET", "/i18n") => json_reply(req, 200, json!({})),
        ("GET", "/global_subgraphs") => json_reply(req, 200, json!({})),
        ("GET", p) if p.starts_with("/global_subgraphs/") => json_reply(req, 404, json!({"error": "no such blueprint"})),
        ("GET", "/node_replacements") => json_reply(req, 200, json!({})),
        ("GET", "/workflow_templates") => json_reply(req, 200, json!({})),
        // the log panel: this backend's log is the journal's (journalctl --user -u rustyui-backend),
        // not kept here, so the panel starts empty and is told nothing
        ("GET", "/internal/logs") => json_reply(req, 200, json!("")),
        ("GET", "/internal/logs/raw") => json_reply(req, 200, json!({"entries": [], "size": {"cols": 80, "rows": 24}})),
        ("PATCH", "/internal/logs/subscribe") => reply(req, 200, "text/plain", vec![]),
        _ => return Err(req),
    }
    Ok(())
}
