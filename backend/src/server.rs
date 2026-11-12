//! `rustyui-backend serve`: rustyui's server, as ComfyUI's is. One process on 127.0.0.1:8190 that
//! runs the graphs, answers ComfyUI's HTTP API for any ComfyUI client and serves the
//! editors: ComfyUI's, forked (ui/), at /, and rustyui's first (frontend/: the prompt writer, a
//! layout for a phone) at /simple/. Both are compiled in; `--frontend DIR` serves a fresh build
//! of the one at / while working on it.
//!
//!   POST /prompt          {"prompt": graph, "extra_data": {"extra_pnginfo": {...}}, "client_id"}
//!                                                    -> {"prompt_id", "number", "node_errors"}
//!   GET  /history[/<id>], /jobs[/<id>]               -> ComfyUI's history entries, its jobs
//!   GET  /queue, /prompt, /system_stats, /embeddings, /object_info[/<Node>], /view?filename=...
//!   GET  /progress                                   -> the node running now and its step
//!   POST /free, /interrupt, /queue, /history, /jobs/<id>/cancel, /upload/image
//!   GET  /ws                                         -> ComfyUI's websocket events
//!
//! Each of those answers under /api/ too, as in ComfyUI, and so do the routes ComfyUI's own
//! editor uses for its settings, user files and model lists (server_user.rs, server_jobs.rs,
//! server_files.rs). The first editor's own routes are /rustyui/status and /rustyui/history
//! (every prompt run here, kept in a file, with the graph it ran); the editor itself is at /.
//!
//! A worker thread runs the graphs in order and keeps models loaded between them. Before a job
//! it waits for ComfyUI on the same card to be idle and asks it to unload, so the two never hold
//! the card at once. After `--idle` seconds with nothing to do, or on /free, the worker ends and
//! takes its models, workspaces and handles with it, which gives the card back to the desktop
//! apart from the CUDA context (about 220 MiB, as ComfyUI keeps its own); the server stays up and
//! starts a worker again for the next prompt.
//!
//! Under systemd (`rustyui-backend.socket`) it starts on demand: systemd holds 127.0.0.1:8190 and
//! starts the backend on the first connection, and the backend exits after `--exit-idle` seconds
//! with no request and nothing to run, so it costs nothing at all until it is used again. An open
//! editor polls every few seconds; that keeps it up only while it holds no finished session's
//! CUDA context (see `done_for_now`).
//!
//! It answers this box only. It refuses to listen anywhere but loopback, and refuses what a web
//! page elsewhere could send: a Host that is not a loopback name (DNS rebinding puts the page's
//! own name on 127.0.0.1), an Origin that is not the editor's own, or a request the browser marks
//! as coming from another site (Sec-Fetch-Site). The one exception is opening the editor from a
//! link, a top-level navigation to "/", which reads and changes nothing. Python tools and curl
//! send none of these headers.

use crate::graph::{self, Engine};
use anyhow::{bail, Context, Result};
use serde_json::{json, Value};
use std::collections::{HashMap, VecDeque};
use std::fs;
use std::io::Read;
use std::net::{IpAddr, SocketAddr, ToSocketAddrs};
use std::path::{Component, Path, PathBuf};
use std::sync::{Arc, Condvar, Mutex};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

const COMFY: &str = "http://127.0.0.1:8188";

// ComfyUI's routes for the editor's own state, its jobs API and its files
#[path = "server_files.rs"]
mod files;
#[path = "server_jobs.rs"]
mod jobs;
#[path = "server_user.rs"]
mod user_routes;

/// the editors, compiled in (build.rs): EDITOR from ../ui/dist, SIMPLE from ../frontend/dist
mod editor {
    include!(concat!(env!("OUT_DIR"), "/editor.rs"));
    include!(concat!(env!("OUT_DIR"), "/simple.rs"));
}

/// what an editor's page says when the backend was built without it
const NO_EDITOR: &str = "<!doctype html><meta charset=utf-8><title>rustyui</title>\
<p>This rustyui-backend was built without this editor. Build it (ui/UPSTREAM.md for the one at /, \
<code>cd ~/rustyui/frontend &amp;&amp; bun install --frozen-lockfile &amp;&amp; bun run build</code> for the one at /simple/), then build \
and install the backend again. ComfyUI's API answers here either way.</p>";

/// the editor's history entries kept; the oldest go first (the pictures stay where they are)
const KEEP: usize = 300;

#[derive(Clone)]
struct Job {
    id: String,
    /// ComfyUI's number: the queue runs the lowest first (a prompt queued at the front takes
    /// minus the next number)
    number: f64,
    graph: Value,
    /// PNG text chunks from extra_data.extra_pnginfo, already ASCII JSON
    extra: Vec<(String, String)>,
    /// ComfyUI's extra_data as its queue and history show it: client_id, extra_pnginfo,
    /// create_time (ms) and the rest of what the client sent, less the sensitive keys
    extra_data: Value,
    /// the output nodes to run (ComfyUI's outputs_to_execute)
    outputs: Vec<String>,
}

impl Job {
    /// ComfyUI's queue item: [number, prompt_id, prompt, extra_data, outputs_to_execute]
    fn item(&self) -> Value {
        json!([number_json(self.number), self.id, self.graph, self.extra_data, self.outputs])
    }

    /// The websocket client its events go to
    fn client(&self) -> Option<&str> {
        self.extra_data["client_id"].as_str().filter(|c| !c.is_empty())
    }
}

/// A number as Python's json writes it: a whole one stays an int
fn number_json(n: f64) -> Value {
    if n.fract() == 0.0 && n.abs() < 9.0e15 {
        json!(n as i64)
    } else {
        json!(n)
    }
}

/// Python's truth of a JSON value ("clear": true, 1, "yes", ...)
fn truthy(v: &Value) -> bool {
    match v {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64() != Some(0.0),
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
    }
}

/// A request for the prompt writer, answered on `reply` by the worker.
struct WriteJob {
    idea: String,
    current: String,
    reply: std::sync::mpsc::Sender<std::result::Result<crate::writer::Written, String>>,
}

struct State {
    queue: VecDeque<Job>,
    /// prompt-writer requests, served before the next render (they are short and someone waits)
    writes: VecDeque<WriteJob>,
    running: Option<Job>,
    /// ComfyUI's history, for the tools and the editor: the last 500, in memory, `order` oldest first
    history: HashMap<String, Value>,
    order: VecDeque<String>,
    /// ComfyUI's prompt counter (the next number)
    number: u64,
    free: bool,
    last: Instant,
    /// whether a worker (and with it the models and the CUDA context) is up
    worker: bool,
    /// the editor's history: every prompt run here, oldest first, kept in a file
    /// {id, number, title, queued, status, outputs, error, seconds, finished, graph, workflow}
    ui: Vec<Value>,
    /// bumped on every change to `ui`, so the editor refetches only then
    ui_version: u64,
    /// whether a worker has run in this process: its CUDA context stays until the process exits
    cuda: bool,
}

struct Server {
    st: Mutex<State>,
    cv: Condvar,
    /// when the last request came in, and how many are being answered (for --exit-idle)
    last_request: Mutex<Instant>,
    /// the same, leaving out an open editor's polling (status, history, progress, queue)
    last_use: Mutex<Instant>,
    in_flight: std::sync::atomic::AtomicUsize,
    out: PathBuf,
    models: PathBuf,
    idle: u64,
    frontend: Option<PathBuf>,
    ui_file: PathBuf,
    /// the SaveImage prefix the editor starts from (a machine that mirrors its renders to another
    /// can give its own, so the two never write the same name)
    prefix: String,
    /// `HOST:DIR`: every render is also copied there, below the same subfolder (`--mirror`)
    mirror: Option<String>,
    /// the prompt writer's model and tokenizer files (`--writer`), when this backend has one
    writer: Option<(PathBuf, PathBuf)>,
    /// `--no-video`: no LTX nodes here
    no_video: bool,
}

/// `rustyui-backend serve`'s flags.
pub struct Options {
    pub listen: String,
    pub out: PathBuf,
    pub idle: u64,
    /// seconds with no request and nothing queued, running or loaded before the process exits
    /// (only when systemd started it, unless given)
    pub exit_idle: Option<u64>,
    pub frontend: Option<PathBuf>,
    pub prefix: String,
    pub mirror: Option<String>,
    pub writer: Option<(PathBuf, PathBuf)>,
    /// neither list nor run the video nodes (`--no-video`, for a machine whose card should only do
    /// images), apart from those an image graph uses too
    pub no_video: bool,
}

fn now() -> f64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs_f64()).unwrap_or(0.0)
}

/// A random (version 4) UUID, as ComfyUI's uuid.uuid4() makes a prompt's id
fn new_id() -> String {
    let mut b = [0u8; 16];
    if let Ok(mut f) = fs::File::open("/dev/urandom") {
        let _ = f.read_exact(&mut b);
    }
    b[6] = (b[6] & 0x0f) | 0x40;
    b[8] = (b[8] & 0x3f) | 0x80;
    let h: String = b.iter().map(|x| format!("{x:02x}")).collect();
    format!("{}-{}-{}-{}-{}", &h[0..8], &h[8..12], &h[12..16], &h[16..20], &h[20..32])
}

fn outputs_of(graph: &Value) -> Vec<String> {
    graph
        .as_object()
        .map(|m| m.iter().filter(|(_, n)| graph::OUTPUT_NODES.iter().any(|o| n["class_type"] == *o)).map(|(k, _)| k.clone()).collect())
        .unwrap_or_default()
}

/// A name for a prompt the editor did not title: its SaveImage prefix
fn title_of(graph: &Value) -> Value {
    graph
        .as_object()
        .into_iter()
        .flatten()
        .find(|(_, n)| graph::OUTPUT_NODES.iter().any(|o| n["class_type"] == *o))
        .and_then(|(_, n)| n["inputs"]["filename_prefix"].as_str())
        .map(|p| json!(p))
        .unwrap_or(Value::Null)
}

// ── the editor's history ─────────────────────────────────────────────────────

fn save_ui(srv: &Server, s: &mut State) {
    let excess = s.ui.len().saturating_sub(KEEP);
    s.ui.drain(..excess);
    if let Some(dir) = srv.ui_file.parent() {
        let _ = fs::create_dir_all(dir);
    }
    let tmp = srv.ui_file.with_extension("json.tmp");
    if fs::write(&tmp, Value::Array(s.ui.clone()).to_string()).is_ok() {
        let _ = fs::rename(&tmp, &srv.ui_file);
    }
    s.ui_version += 1;
}

fn ui_update(srv: &Server, s: &mut State, id: &str, fields: Value) {
    if let Some(e) = s.ui.iter_mut().find(|e| e["id"] == id) {
        for (k, v) in fields.as_object().into_iter().flatten() {
            e[k] = v.clone();
        }
        save_ui(srv, s);
    }
}

// ── the worker ───────────────────────────────────────────────────────────────

/// Waits for ComfyUI on this card to finish what it is doing, then asks it to unload.
fn comfy_yield() {
    let agent = ureq::AgentBuilder::new().timeout(Duration::from_secs(5)).build();
    let deadline = Instant::now() + Duration::from_secs(1800);
    loop {
        match agent.get(&format!("{COMFY}/queue")).call().ok().and_then(|r| r.into_json::<Value>().ok()) {
            Some(q) => {
                let busy = q["queue_running"].as_array().map_or(0, Vec::len) + q["queue_pending"].as_array().map_or(0, Vec::len);
                if busy == 0 || Instant::now() > deadline {
                    break;
                }
                eprintln!("[rustyui] ComfyUI is busy with {busy} job(s); waiting");
                thread::sleep(Duration::from_secs(3));
            }
            None => return, // not running: nothing to wait for or free
        }
    }
    let _ = agent.post(&format!("{COMFY}/free")).send_json(json!({"unload_models": true, "free_memory": true}));
}

/// Starts a worker when there is work and none is up. Call with the state locked.
fn wake(srv: &Arc<Server>, s: &mut State) {
    srv.cv.notify_all();
    if !s.worker && (!s.queue.is_empty() || !s.writes.is_empty()) {
        s.worker = true;
        s.cuda = true;
        let srv = srv.clone();
        thread::spawn(move || worker(srv));
    }
}

/// One GPU session: the jobs run on an inner thread, so its thread-locals (cuBLAS, cuBLASLt and
/// cuDNN handles and their workspaces) are gone when it ends, then candle's cached kernel
/// parameters. What stays is the CUDA context itself, about 220 MiB: cuBLAS and cuDNN keep
/// their own (static) CUDA runtime, which holds it until the process exits, as ComfyUI's does.
/// The worker is marked down only after all that, and a prompt that came meanwhile starts the
/// next session.
fn worker(srv: Arc<Server>) {
    let (why, ending) = {
        let srv = srv.clone();
        thread::spawn(move || run_jobs(&srv)).join().unwrap_or_else(|_| ("a job panicked".to_string(), Instant::now()))
    };
    candle_core::cuda_backend::clear_param_cache();
    // from the worker's decision to stop to here: the engine's drop, then the job thread's end
    // (its cuBLAS and cuDNN handles and workspaces) and candle's kernel parameters
    eprintln!("[rustyui] {why}: models released, session ended in {:.1} ms", ending.elapsed().as_secs_f64() * 1e3);
    let mut s = srv.st.lock().unwrap();
    s.worker = false;
    // a prompt that came in while the context was going
    wake(&srv, &mut s);
}

enum Next {
    Render(Job),
    Write(WriteJob),
}

/// Runs the queue until `idle` seconds pass without work or a /free, and says which and when it
/// decided to stop.
fn run_jobs(srv: &Arc<Server>) -> (String, Instant) {
    let mut engine: Option<Engine> = None;
    let mut writer: Option<crate::writer::Writer> = None;
    loop {
        let next = {
            let mut s = srv.st.lock().unwrap();
            loop {
                if let Some(w) = s.writes.pop_front() {
                    s.free = false;
                    break Next::Write(w);
                }
                if let Some(j) = s.queue.pop_front() {
                    emit_status(s.queue.len() + 1);
                    s.free = false;
                    s.running = Some(j.clone());
                    let id = j.id.clone();
                    ui_update(srv, &mut s, &id, json!({"status": "running", "started": now()}));
                    break Next::Render(j);
                }
                let idle = s.last.elapsed() > Duration::from_secs(srv.idle);
                if idle || s.free {
                    let ending = Instant::now();
                    s.free = false;
                    drop(s);
                    drop(engine);
                    drop(writer);
                    return (if idle { format!("idle for {}s", srv.idle) } else { "asked to free".to_string() }, ending);
                }
                s = srv.cv.wait_timeout(s, Duration::from_secs(1)).unwrap().0;
            }
        };
        comfy_yield();
        let job = match next {
            Next::Write(w) => {
                let answer = (|| {
                    let (model, tokenizer) = srv.writer.clone().ok_or_else(|| anyhow::anyhow!("this backend has no prompt writer"))?;
                    // the writer and a render take turns on the card: the 12B holds about 14 GB
                    engine = None;
                    if writer.is_none() {
                        writer = Some(crate::writer::Writer::load(&model, &tokenizer, &candle_core::Device::new_cuda(0)?)?);
                    }
                    let seed = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_nanos() as u64).unwrap_or(0);
                    writer.as_mut().unwrap().write(&w.idea, &w.current, seed)
                })();
                if let Ok(r) = &answer {
                    eprintln!("[rustyui] wrote a prompt: {} tokens in {:.1}s", r.tokens, r.seconds);
                }
                let _ = w.reply.send(answer.map_err(|e| format!("{e:#}")));
                srv.st.lock().unwrap().last = Instant::now();
                continue;
            }
            Next::Render(job) => job,
        };
        use crate::progress::Outcome;
        // a render does not share the card with the writer (reloaded in a few seconds when asked)
        writer = None;
        crate::progress::start(&job.id, job.client());
        let t = Instant::now();
        let out_dir = srv.out.clone();
        // each output node's files go out as it finishes, as ComfyUI's executed messages do
        let each = |node: &str, files: &[PathBuf]| {
            if !files.is_empty() {
                crate::progress::executed(node, &output_of(files, &out_dir));
            }
        };
        let result = (|| {
            if engine.is_none() {
                engine = Some(Engine::new(&srv.out)?);
            }
            let e = engine.as_mut().unwrap();
            e.pnginfo = job.extra.clone();
            e.run_outputs(job.graph.clone(), Some(&job.outputs), each)
        })();
        let (node, class) = crate::progress::last_node().unwrap_or_default();
        let secs = t.elapsed().as_secs_f64();
        let (mut outputs, mut meta) = (serde_json::Map::new(), serde_json::Map::new());
        let (success, messages, ui) = match result {
            Ok(files) => {
                let mut ui_outputs = vec![];
                for (node, paths) in files.iter().filter(|(_, p)| !p.is_empty()) {
                    let output = output_of(paths, &srv.out);
                    for img in output["images"].as_array().into_iter().flatten() {
                        ui_outputs.push(json!({"node": node, "filename": img["filename"], "subfolder": img["subfolder"]}));
                    }
                    meta.insert(node.clone(), json!({"node_id": node, "display_node": node, "parent_node": null, "real_node_id": node}));
                    outputs.insert(node.clone(), output);
                }
                let messages = crate::progress::finished(Outcome::Success);
                if let Some(e) = engine.as_ref() {
                    let parts: Vec<String> = e.timings.iter().map(|(w, s)| format!("{w} {s:.2}s")).collect();
                    eprintln!("[rustyui] {} done in {secs:.1}s: {}", job.id, parts.join(", "));
                }
                if let Some(dest) = srv.mirror.clone() {
                    let (out, files): (PathBuf, Vec<PathBuf>) = (srv.out.clone(), files_of(&outputs, &srv.out));
                    thread::spawn(move || mirror(&dest, &out, &files));
                }
                (true, messages, json!({"status": "success", "outputs": ui_outputs, "error": null, "seconds": secs, "finished": now()}))
            }
            Err(e) => {
                let msg = format!("{e:#}");
                let messages = if e.chain().any(|c| c.is::<crate::progress::Interrupted>()) {
                    eprintln!("[rustyui] {} interrupted after {secs:.1}s, in node {node} ({class})", job.id);
                    crate::progress::finished(Outcome::Interrupted(&node, &class))
                } else {
                    eprintln!("[rustyui] {} failed after {secs:.1}s: {msg}", job.id);
                    crate::progress::finished(Outcome::Error(&node, &class, &msg))
                };
                (false, messages, json!({"status": "error", "outputs": [], "error": msg, "seconds": secs, "finished": now()}))
            }
        };
        // ComfyUI's order: the history entry, then the queue's status to everyone, then the client
        // is told the prompt is done
        let mut s = srv.st.lock().unwrap();
        s.running = None;
        record(&mut s, &job, Value::Object(outputs), Value::Object(meta), success, messages);
        emit_status(s.queue.len());
        ui_update(srv, &mut s, &job.id, ui);
        s.last = Instant::now();
        drop(s);
        crate::progress::done();
    }
}

/// A finished prompt's history entry, as ComfyUI's task_done writes it; the newest 500 are kept
fn record(s: &mut State, job: &Job, outputs: Value, meta: Value, success: bool, messages: Vec<Value>) {
    let status = json!({"status_str": if success { "success" } else { "error" }, "completed": success, "messages": messages});
    s.history.insert(job.id.clone(), json!({"prompt": job.item(), "outputs": outputs, "status": status, "meta": meta}));
    s.order.retain(|o| o != &job.id);
    s.order.push_back(job.id.clone());
    while s.order.len() > 500 {
        if let Some(old) = s.order.pop_front() {
            s.history.remove(&old);
        }
    }
}

/// An output node's files as ComfyUI's history and executed messages list them; a video goes
/// under "images" too, flagged as ComfyUI's SaveVideo flags it
fn output_of(files: &[PathBuf], out: &Path) -> Value {
    let images: Vec<Value> = files
        .iter()
        .map(|p| {
            let sub = p.parent().and_then(|d| d.strip_prefix(out).ok()).map(|d| d.to_string_lossy().to_string()).unwrap_or_default();
            json!({"filename": p.file_name().map(|f| f.to_string_lossy().to_string()).unwrap_or_default(), "subfolder": sub, "type": "output"})
        })
        .collect();
    if files.iter().any(|p| p.extension().is_some_and(|e| e == "mp4")) {
        json!({"images": images, "animated": [true]})
    } else {
        json!({"images": images})
    }
}

/// Stops the prompt the worker has: at its next step or node once it has started, else as soon
/// as it starts (it may still be waiting for ComfyUI or the card)
fn stop_running(id: &str) {
    if !crate::progress::interrupt(Some(id)) {
        crate::progress::interrupt_at_start(id);
    }
}

/// GET /history: ComfyUI's get_history, oldest first; with max_items and no offset, the newest
/// max_items
fn history_page(s: &State, max_items: Option<usize>, offset: Option<i64>) -> Value {
    let mut offset = offset.unwrap_or(-1);
    if offset < 0 {
        if let Some(m) = max_items {
            offset = s.order.len() as i64 - m as i64;
        }
    }
    let mut out = serde_json::Map::new();
    for (i, id) in s.order.iter().enumerate() {
        if i as i64 >= offset {
            if let Some(e) = s.history.get(id) {
                out.insert(id.clone(), e.clone());
            }
            if max_items.is_some_and(|m| out.len() >= m) {
                break;
            }
        }
    }
    Value::Object(out)
}

/// The files a job's ComfyUI outputs name, below `out`.
fn files_of(outputs: &serde_json::Map<String, Value>, out: &Path) -> Vec<PathBuf> {
    outputs
        .values()
        .flat_map(|o| o["images"].as_array().cloned().unwrap_or_default())
        .filter_map(|i| Some(out.join(i["subfolder"].as_str()?).join(i["filename"].as_str()?)))
        .collect()
}

/// rsync's arguments for copying `rel` (below the output folder) to the same place below `dest`
/// (`HOST:DIR`). Never overwrites: a name already there stays as it is. -i says what it did.
fn mirror_args(dest: &str, file: &Path, rel: &Path) -> Vec<String> {
    let target = format!("{}/{}", dest.trim_end_matches('/'), rel.to_string_lossy());
    vec![
        "-i".into(),
        "--ignore-existing".into(),
        "--mkpath".into(),
        "--chmod=F644".into(),
        "--timeout=60".into(),
        "-e".into(),
        "ssh -o BatchMode=yes -o ConnectTimeout=10".into(),
        file.to_string_lossy().into(),
        target,
    ]
}

/// Copies finished renders to another machine's output folder (`--mirror host:ComfyUI/output`),
/// so a render made here also shows up there. Never
/// fatal, never overwrites, and says what happened to each file.
fn mirror(dest: &str, out: &Path, files: &[PathBuf]) {
    for file in files {
        let Ok(rel) = file.strip_prefix(out) else { continue };
        match std::process::Command::new("rsync").args(mirror_args(dest, file, rel)).output() {
            Ok(o) if o.status.success() && o.stdout.is_empty() => {
                eprintln!("[rustyui] {} is already in {dest}; that one was left as it is", rel.display())
            }
            Ok(o) if o.status.success() => eprintln!("[rustyui] {} copied to {dest}", rel.display()),
            Ok(o) => eprintln!("[rustyui] {} not copied to {dest}: {}", rel.display(), String::from_utf8_lossy(&o.stderr).trim()),
            Err(e) => eprintln!("[rustyui] {} not copied to {dest}: rsync: {e}", rel.display()),
        }
    }
}

/// A PNG of the output folder made `width` wide (never wider than it was); other pictures as they are.
fn thumbnail(file: &Path, width: usize) -> Result<(&'static str, Vec<u8>)> {
    let ext = file.extension().and_then(|e| e.to_str()).map(str::to_ascii_lowercase).unwrap_or_default();
    if ext == "mp4" || ext == "webm" {
        // a video's first frame, `width` wide (never wider than the video), as a PNG
        let out = std::process::Command::new("ffmpeg")
            .args(["-v", "error", "-i"])
            .arg(file)
            .args(["-frames:v", "1", "-vf", &format!("scale='min({width},iw)':-2"), "-f", "image2pipe", "-vcodec", "png", "-"])
            .output()?;
        anyhow::ensure!(out.status.success() && !out.stdout.is_empty(), "ffmpeg could not read {}", file.display());
        return Ok(("image/png", out.stdout));
    }
    if ext != "png" {
        let ctype = if ext == "webp" { "image/webp" } else { "image/jpeg" };
        return Ok((ctype, fs::read(file)?));
    }
    let img = crate::image::read_png(file)?;
    let (w, h) = (img.width as usize, img.height as usize);
    if w <= width {
        return Ok(("image/png", fs::read(file)?));
    }
    let height = ((h * width) as f64 / w as f64).round().max(1.0) as usize;
    let small = crate::resample::resize_lanczos(&img.rgb, w, h, width, height);
    Ok(("image/png", crate::image::encode_png(&small, width as u32, height as u32)?))
}

// ── what the API lists ───────────────────────────────────────────────────────

/// Files under `dir` (following symlinks) with one of the extensions, as ComfyUI lists them:
/// relative paths with forward slashes, sorted.
fn list(dir: &Path, exts: &[&str]) -> Vec<String> {
    fn walk(base: &Path, d: &Path, exts: &[&str], out: &mut Vec<String>, depth: usize) {
        let Ok(rd) = fs::read_dir(d) else { return };
        for e in rd.flatten() {
            let p = e.path();
            if p.is_dir() {
                if depth < 8 {
                    walk(base, &p, exts, out, depth + 1);
                }
            } else if p.is_file() {
                let name = p.file_name().map(|f| f.to_string_lossy().to_lowercase()).unwrap_or_default();
                if exts.iter().any(|x| name.ends_with(x)) {
                    if let Ok(rel) = p.strip_prefix(base) {
                        out.push(rel.to_string_lossy().replace('\\', "/"));
                    }
                }
            }
        }
    }
    let mut out = vec![];
    walk(dir, dir, exts, &mut out, 0);
    out.sort();
    out
}

/// ComfyUI's /object_info for the nodes the backend implements: each input's type, default and
/// range, combos of what is installed (or of what the backend supports, for samplers and the
/// like), the outputs, and input_order (serde_json sorts object keys, so order travels apart).
/// `prefix` is the SaveImage prefix the editor starts from (`--prefix`).
fn object_info(models: &Path, node: Option<&str>, prefix: &str, no_video: bool) -> Value {
    let files = |dir: &str, exts: &[&str]| json!([list(&models.join(dir), exts)]);
    let ty = |t: &str| json!([t]);
    let int = |d: i64, min: i64, max: i64| json!(["INT", {"default": d, "min": min, "max": max, "step": 1}]);
    let float =
        |d: f64, min: f64, max: f64, step: f64| json!(["FLOAT", {"default": d, "min": min, "max": max, "step": step, "round": 0.01}]);
    let strength = || float(1.0, -100.0, 100.0, 0.01);
    let size = |d: i64, step: i64| json!(["INT", {"default": d, "min": 16, "max": 16384, "step": step}]);
    // (class, display name, category, description, inputs in order, outputs)
    let nodes: Vec<(&str, &str, &str, &str, Vec<(&str, Value)>, Vec<&str>)> = vec![
        (
            "CheckpointLoaderSimple",
            "Load Checkpoint",
            "loaders",
            "An SDXL checkpoint: its model, text encoders and VAE.",
            vec![("ckpt_name", files("checkpoints", &[".safetensors", ".ckpt"]))],
            vec!["MODEL", "CLIP", "VAE"],
        ),
        (
            "LoraLoader",
            "Load LoRA",
            "loaders",
            "A LoRA folded into the model and the text encoders.",
            vec![
                ("model", ty("MODEL")),
                ("clip", ty("CLIP")),
                ("lora_name", files("loras", &[".safetensors"])),
                ("strength_model", strength()),
                ("strength_clip", strength()),
            ],
            vec!["MODEL", "CLIP"],
        ),
        (
            "LoraLoaderModelOnly",
            "Load LoRA (model only)",
            "loaders",
            "A LoRA folded into the model only.",
            vec![("model", ty("MODEL")), ("lora_name", files("loras", &[".safetensors"])), ("strength_model", strength())],
            vec!["MODEL"],
        ),
        (
            "CLIPSetLastLayer",
            "CLIP Set Last Layer",
            "conditioning",
            "Which text-encoder layer the prompt is read from (-2 is SDXL's usual).",
            vec![("clip", ty("CLIP")), ("stop_at_clip_layer", int(-1, -24, -1))],
            vec!["CLIP"],
        ),
        (
            "CLIPTextEncode",
            "CLIP Text Encode (Prompt)",
            "conditioning",
            "A prompt, with ComfyUI's (word:1.2) weights and embedding:name.",
            vec![("text", json!(["STRING", {"multiline": true, "dynamicPrompts": true}])), ("clip", ty("CLIP"))],
            vec!["CONDITIONING"],
        ),
        (
            "ConditioningConcat",
            "Conditioning (Concat)",
            "conditioning",
            "Two prompts joined end to end (what A1111's BREAK does).",
            vec![("conditioning_to", ty("CONDITIONING")), ("conditioning_from", ty("CONDITIONING"))],
            vec!["CONDITIONING"],
        ),
        (
            "EmptyLatentImage",
            "Empty Latent Image",
            "latent",
            "A blank latent of this size to sample into.",
            vec![("width", size(1024, 8)), ("height", size(1024, 8)), ("batch_size", int(1, 1, 4096))],
            vec!["LATENT"],
        ),
        (
            "KSampler",
            "KSampler",
            "sampling",
            "Denoises the latent with the model, guided by the prompts.",
            vec![
                ("model", ty("MODEL")),
                ("seed", json!(["INT", {"default": 0, "min": 0, "max": u64::MAX, "control_after_generate": true}])),
                ("steps", int(20, 1, 10000)),
                ("cfg", float(8.0, 0.0, 100.0, 0.1)),
                (
                    "sampler_name",
                    json!([[
                        "euler",
                        "euler_ancestral",
                        "euler_ancestral_cfg_pp",
                        "dpmpp_2m",
                        "dpmpp_2m_sde",
                        "dpmpp_2m_sde_gpu",
                        "er_sde"
                    ]]),
                ),
                ("scheduler", json!([["normal", "karras", "sgm_uniform", "simple", "beta"]])),
                ("positive", ty("CONDITIONING")),
                ("negative", ty("CONDITIONING")),
                ("latent_image", ty("LATENT")),
                ("denoise", float(1.0, 0.0, 1.0, 0.01)),
            ],
            vec!["LATENT"],
        ),
        ("VAEDecode", "VAE Decode", "latent", "Latent to pixels.", vec![("samples", ty("LATENT")), ("vae", ty("VAE"))], vec!["IMAGE"]),
        (
            "VAEEncode",
            "VAE Encode",
            "latent",
            "Pixels to latent (for a second pass).",
            vec![("pixels", ty("IMAGE")), ("vae", ty("VAE"))],
            vec!["LATENT"],
        ),
        (
            "UpscaleModelLoader",
            "Load Upscale Model",
            "loaders",
            "An ESRGAN-family upscaler such as Remacri.",
            vec![("model_name", files("upscale_models", &[".pth", ".safetensors"]))],
            vec!["UPSCALE_MODEL"],
        ),
        (
            "ImageUpscaleWithModel",
            "Upscale Image (using Model)",
            "image/upscaling",
            "Upscales with the model, tiled 512/32 like ComfyUI.",
            vec![("upscale_model", ty("UPSCALE_MODEL")), ("image", ty("IMAGE"))],
            vec!["IMAGE"],
        ),
        (
            "ImageScale",
            "Upscale Image",
            "image/upscaling",
            "Resizes to an exact size (lanczos, as Pillow does it).",
            vec![
                ("image", ty("IMAGE")),
                ("upscale_method", json!([["lanczos", "bilinear"]])),
                ("width", json!(["INT", {"default": 512, "min": 0, "max": 16384, "step": 1}])),
                ("height", json!(["INT", {"default": 512, "min": 0, "max": 16384, "step": 1}])),
                ("crop", json!([["disabled"]])),
            ],
            vec!["IMAGE"],
        ),
        (
            "SaveImage",
            "Save Image",
            "image",
            "Writes PNGs to the output folder, the graph inside each.",
            vec![("images", ty("IMAGE")), ("filename_prefix", json!(["STRING", {"default": prefix}]))],
            vec![],
        ),
    ];
    debug_assert!(nodes.iter().all(|n| graph::NODES.contains(&n.0)) && nodes.len() == graph::NODES.len());
    let mut all = serde_json::Map::new();
    for (class, display, category, description, inputs, outputs) in nodes {
        if node.is_some_and(|want| want != class) {
            continue;
        }
        let order: Vec<&str> = inputs.iter().map(|(n, _)| *n).collect();
        let required: serde_json::Map<String, Value> = inputs.into_iter().map(|(n, v)| (n.to_string(), v)).collect();
        let mut entry = json!({
            "input": {"required": required},
            "input_order": {"required": order},
            "output": outputs, "output_is_list": vec![false; outputs.len()], "output_name": outputs,
            "name": class, "display_name": display, "description": description, "category": category,
            "output_node": class == "SaveImage", "python_module": "rustyui-backend",
        });
        if class == "SaveImage" {
            entry["input"]["hidden"] = json!({"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"});
        }
        all.insert(class.to_string(), entry);
    }
    // the LTX and utility nodes: ComfyUI's own definitions (assets/comfy_object_info.json, taken
    // from its /object_info, choices cut to what rustyui runs and no file names kept), with the
    // file lists filled from this machine's folders
    let mut ltx: serde_json::Map<String, Value> = serde_json::from_str(COMFY_OBJECT_INFO).unwrap_or_default();
    let input_dir = models.parent().unwrap_or(models).join("input");
    // RIFE VFI's checkpoints sit in the custom node's own folder; the ones rife.rs runs
    let rife: Vec<String> = list(&models.parent().unwrap_or(models).join("custom_nodes/ComfyUI-Frame-Interpolation/ckpts/rife"), &[".pth"])
        .into_iter()
        .filter(|n| n == "rife47.pth" || n == "rife49.pth")
        .collect();
    // Depth Anything V2's checkpoints wherever depth_anything::checkpoint finds them (not the
    // Giant: its blocks are not ported)
    let depth: Vec<String> = ["depth_anything_v2_vitl.pth", "depth_anything_v2_vitb.pth", "depth_anything_v2_vits.pth"]
        .into_iter()
        .filter(|n| crate::depth_anything::checkpoint(models, n).is_ok())
        .map(String::from)
        .collect();
    // CLIPLoader: the text encoders rustyui loads through it, Qwen3-0.6B (Anima's), by their keys
    let qwen3: Vec<String> = list(&models.join("text_encoders"), &[".safetensors"])
        .into_iter()
        .filter(|n| {
            let Ok(st) = (unsafe { candle_core::safetensors::MmapedSafetensors::new(models.join("text_encoders").join(n)) }) else {
                return false;
            };
            st.get("model.layers.27.self_attn.q_norm.weight").is_ok()
                && st.get("model.layers.28.self_attn.q_norm.weight").is_err()
                && st.get("model.embed_tokens.weight").is_ok_and(|t| t.shape() == [151936, 1024])
        })
        .collect();
    let fill: [(&str, &str, Vec<String>); 15] = [
        ("CLIPLoader", "clip_name", qwen3),
        ("ControlNetLoader", "control_net_name", list(&models.join("controlnet"), &[".safetensors"])),
        ("CreateHookLora", "lora_name", list(&models.join("loras"), &[".safetensors"])),
        ("CreateHookLoraModelOnly", "lora_name", list(&models.join("loras"), &[".safetensors"])),
        ("LTXAVTextEncoderLoader", "text_encoder", list(&models.join("text_encoders"), &[".safetensors"])),
        ("LTXAVTextEncoderLoader", "ckpt_name", list(&models.join("checkpoints"), &[".safetensors", ".ckpt"])),
        (
            "UNETLoader",
            "unet_name",
            [list(&models.join("diffusion_models"), &[".safetensors"]), list(&models.join("unet"), &[".safetensors"])].concat(),
        ),
        ("VAELoader", "vae_name", list(&models.join("vae"), &[".safetensors"])),
        ("LTXVAudioVAELoader", "ckpt_name", list(&models.join("checkpoints"), &[".safetensors"])),
        ("LoadImage", "image", list(&input_dir, &[".png"])),
        ("LoadImageMask", "image", list(&input_dir, &[".png"])),
        ("LatentUpscaleModelLoader", "model_name", list(&models.join("latent_upscale_models"), &[".safetensors"])),
        ("KSamplerSelect", "sampler_name", vec!["euler".to_string(), "euler_ancestral".to_string()]),
        ("RIFE VFI", "ckpt_name", rife),
        ("DepthAnythingV2Preprocessor", "ckpt_name", depth),
    ];
    for (class, input, mut names) in fill {
        names.sort();
        names.dedup();
        let Some(inputs) = ltx.get_mut(class).and_then(|d| d.get_mut("input")) else { continue };
        let part = if inputs.get("required").and_then(|r| r.get(input)).is_some() { "required" } else { "optional" };
        if let Some(spec) = inputs.get_mut(part).and_then(|r| r.get_mut(input)) {
            set_choices(spec, names);
        }
    }
    let video = graph::LTX_NODES.iter().filter(|c| !no_video || graph::SHARED_NODES.contains(c));
    for class in video
        .chain(graph::UTIL_NODES.iter())
        .chain(graph::COND_NODES.iter())
        .chain(graph::IMPACT_NODES.iter())
        .chain(graph::TAC_NODES.iter())
        .chain(graph::ANIMA_NODES.iter())
    {
        if node.is_none_or(|want| want == *class) {
            if let Some(entry) = ltx.remove(*class) {
                all.insert(class.to_string(), entry);
            }
        }
    }
    Value::Object(all)
}

const COMFY_OBJECT_INFO: &str = include_str!("../assets/comfy_object_info.json");

/// A combo's choices, in either of ComfyUI's forms: `[[...], {..}]` or `["COMBO", {"options": [...]}]`
fn set_choices(spec: &mut Value, names: Vec<String>) {
    if spec[0].is_array() {
        spec[0] = json!(names);
    } else if spec[0] == "COMBO" {
        spec[1]["options"] = json!(names);
    }
}

/// Every file name in the graph (an output prefix, a model) that would leave its folder, so a
/// graph is refused when it arrives rather than after it has rendered.
fn escaping_names(graph: &Value) -> Vec<String> {
    let mut bad = vec![];
    for (id, node) in graph.as_object().into_iter().flatten() {
        for key in
            ["filename_prefix", "ckpt_name", "lora_name", "model_name", "vae_name", "unet_name", "clip_name", "text_encoder", "image"]
        {
            if let Some(name) = node["inputs"].get(key).and_then(Value::as_str) {
                if crate::paths::contained(name).is_none() {
                    bad.push(format!("node {id} {key} {name:?}"));
                }
            }
        }
    }
    bad
}

// ── who may connect ──────────────────────────────────────────────────────────

/// `listen` resolved, and refused unless every address it names is loopback.
pub fn loopback_addr(listen: &str) -> Result<SocketAddr> {
    let addrs: Vec<SocketAddr> = listen.to_socket_addrs().with_context(|| format!("cannot resolve {listen}"))?.collect();
    match addrs.first() {
        Some(a) if addrs.iter().all(|a| a.ip().is_loopback()) => Ok(*a),
        _ => bail!("refusing to listen on {listen}: rustyui serves this box only (127.0.0.1 or [::1])"),
    }
}

fn header<'a>(req: &'a tiny_http::Request, name: &'static str) -> Option<&'a str> {
    req.headers().iter().find(|h| h.field.equiv(name)).map(|h| h.value.as_str())
}

/// "127.0.0.1:8190", "[::1]:8190", "localhost" -> the name, without port or brackets.
fn host_name(host: &str) -> &str {
    match host.strip_prefix('[') {
        Some(rest) => rest.split(']').next().unwrap_or(""),
        None => host.rsplit_once(':').map_or(host, |(name, _)| name),
    }
}

fn loopback_name(name: &str) -> bool {
    name.eq_ignore_ascii_case("localhost") || name.parse::<IpAddr>().is_ok_and(|ip| ip.is_loopback())
}

/// Why this request is turned away, or None. `h` reads a header.
fn refusal(method: &str, path: &str, h: &dyn Fn(&'static str) -> Option<String>) -> Option<String> {
    let host = h("Host");
    if let Some(host) = &host {
        if !loopback_name(host_name(host)) {
            return Some(format!("Host {host} is not this box"));
        }
    }
    let open_editor = method == "GET"
        && matches!(path, "/" | "/index.html" | "/simple" | "/simple/" | "/simple/index.html")
        && h("Sec-Fetch-Mode").as_deref() == Some("navigate")
        && h("Sec-Fetch-Dest").is_none_or(|d| d == "document");
    if let Some(origin) = h("Origin") {
        let own = host.as_ref().map(|hh| format!("http://{hh}"));
        if Some(&origin) != own.as_ref() && !open_editor {
            return Some(format!("Origin {origin} is another site"));
        }
    }
    if let Some(site) = h("Sec-Fetch-Site") {
        if site != "same-origin" && site != "none" && !open_editor {
            return Some(format!("a {site} request"));
        }
    }
    None
}

// ── files below the output folder ────────────────────────────────────────────

/// `name` as a path that stays below the folder it is joined to: relative, no `..`, no root.
fn contained(name: &str) -> Option<PathBuf> {
    let p = Path::new(name);
    let plain = p.components().all(|c| matches!(c, Component::Normal(_) | Component::CurDir));
    (plain && p.components().any(|c| matches!(c, Component::Normal(_)))).then(|| p.to_path_buf())
}

/// A picture /view may hand out: below its folder (output, input or temp) once symlinks are
/// followed, and an image or video by its extension.
fn picture(out: &Path, filename: &str, subfolder: &str) -> Option<PathBuf> {
    let rel = match (contained(filename), subfolder.is_empty()) {
        (Some(f), true) => f,
        (Some(f), false) => contained(subfolder)?.join(f),
        (None, _) => return None,
    };
    let real = fs::canonicalize(out.join(rel)).ok()?;
    let base = fs::canonicalize(out).ok()?;
    let ext = real.extension()?.to_str()?.to_ascii_lowercase();
    (real.starts_with(base) && real.is_file() && ["png", "jpg", "jpeg", "webp", "gif", "mp4", "webm"].contains(&ext.as_str()))
        .then_some(real)
}

fn query(q: &str, key: &str) -> String {
    q.split('&').find_map(|kv| kv.strip_prefix(&format!("{key}="))).map(percent_decode).unwrap_or_default()
}

fn percent_decode(s: &str) -> String {
    let b = s.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        match b[i] {
            b'%' if i + 2 < b.len() => match u8::from_str_radix(std::str::from_utf8(&b[i + 1..i + 3]).unwrap_or("zz"), 16) {
                Ok(v) => {
                    out.push(v);
                    i += 3;
                }
                Err(_) => {
                    out.push(b'%');
                    i += 1;
                }
            },
            b'+' => {
                out.push(b' ');
                i += 1;
            }
            c => {
                out.push(c);
                i += 1;
            }
        }
    }
    String::from_utf8_lossy(&out).to_string()
}

// ── replies ──────────────────────────────────────────────────────────────────

/// ComfyUI's "status" websocket message: how many prompts are queued or running
fn emit_status(queue_remaining: usize) {
    crate::progress::emit("status", json!({"status": {"exec_info": {"queue_remaining": queue_remaining}}}));
}

/// GET /ws: ComfyUI's websocket. The connection is handed to a thread that greets the client with
/// its sid and the queue, and the server's feature flags (ComfyUI sends those in answer to the
/// client's, the one message a client sends, which this server does not read), then forwards
/// every event progress sends it (and pings while idle); a client that has gone is dropped at
/// the next write.
fn serve_ws(srv: &Arc<Server>, req: tiny_http::Request, q: &str) {
    let key = req.headers().iter().find(|h| h.field.equiv("Sec-WebSocket-Key")).map(|h| h.value.as_str().to_string());
    let Some(key) = key else { return reply(req, 400, "text/plain", b"not a websocket request".to_vec()) };
    let accept = tungstenite::handshake::derive_accept_key(key.as_bytes());
    let h = |k: &str, v: &str| tiny_http::Header::from_bytes(k.as_bytes(), v.as_bytes()).unwrap();
    let resp = tiny_http::Response::empty(tiny_http::StatusCode(101))
        .with_header(h("Upgrade", "websocket"))
        .with_header(h("Connection", "Upgrade"))
        .with_header(h("Sec-WebSocket-Accept", &accept));
    let stream = req.upgrade("websocket", resp);
    let sid = match query(q, "clientId") {
        id if !id.is_empty() => id,
        _ => new_id().replace('-', ""),
    };
    let remaining = {
        let s = srv.st.lock().unwrap();
        s.queue.len() + usize::from(s.running.is_some())
    };
    thread::spawn(move || {
        use tungstenite::{protocol::Role, Message, WebSocket};
        let mut ws = WebSocket::from_raw_socket(stream, Role::Server, None);
        let (tx, rx) = std::sync::mpsc::channel::<String>();
        let hello = json!({"type": "status", "data": {"status": {"exec_info": {"queue_remaining": remaining}}, "sid": sid}});
        let flags = json!({"type": "feature_flags", "data": user_routes::features()});
        if ws.send(Message::text(hello.to_string())).is_err() || ws.send(Message::text(flags.to_string())).is_err() {
            return;
        }
        crate::progress::subscribe(&sid, tx);
        loop {
            let sent = match rx.recv_timeout(Duration::from_secs(20)) {
                Ok(msg) => ws.send(Message::text(msg)),
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => ws.send(Message::Ping(Vec::new())),
                Err(_) => break,
            };
            if sent.is_err() {
                break;
            }
        }
    });
}

fn reply(req: tiny_http::Request, code: u16, ctype: &str, body: Vec<u8>) {
    reply_cached(req, code, ctype, body, "no-store")
}

fn reply_cached(req: tiny_http::Request, code: u16, ctype: &str, body: Vec<u8>, cache: &str) {
    let h = |k: &str, v: &str| tiny_http::Header::from_bytes(k.as_bytes(), v.as_bytes()).unwrap();
    let resp = tiny_http::Response::from_data(body)
        .with_status_code(code)
        .with_header(h("Content-Type", ctype))
        .with_header(h("Cache-Control", cache))
        .with_header(h("X-Content-Type-Options", "nosniff"))
        .with_header(h("X-Frame-Options", "DENY"));
    let _ = req.respond(resp);
}

fn json_reply(req: tiny_http::Request, code: u16, v: Value) {
    reply(req, code, "application/json", v.to_string().into_bytes());
}

fn refuse_prompt(req: tiny_http::Request, kind: &str, msg: &str) {
    json_reply(req, 400, json!({"error": {"type": kind, "message": msg, "details": msg}, "node_errors": {}}));
}

fn content_type(name: &str) -> &'static str {
    match name.rsplit_once('.').map(|(_, e)| e.to_ascii_lowercase()).as_deref() {
        Some("html") => "text/html; charset=utf-8",
        Some("js") | Some("mjs") => "text/javascript; charset=utf-8",
        Some("css") => "text/css; charset=utf-8",
        Some("json") | Some("map") => "application/json",
        Some("svg") => "image/svg+xml",
        Some("png") => "image/png",
        Some("ico") => "image/x-icon",
        Some("woff2") => "font/woff2",
        Some("woff") => "font/woff",
        Some("txt") => "text/plain; charset=utf-8",
        _ => "application/octet-stream",
    }
}

/// A file of an editor ("" is index.html): ComfyUI's, forked, below / (from `--frontend DIR`
/// when given), and rustyui's first below /simple/; compiled in.
fn editor_file(dir: Option<&Path>, path: &str) -> Option<(&'static str, Vec<u8>)> {
    let (files, path, dir) = match path.strip_prefix("/simple") {
        Some(rest) if rest.is_empty() || rest.starts_with('/') => (editor::SIMPLE, rest, None),
        _ => (editor::EDITOR, path, dir),
    };
    let rel = match path.trim_start_matches('/') {
        "" => "index.html",
        p => p,
    };
    let body = match dir {
        Some(d) => fs::read(d.join(contained(rel)?)).ok(),
        None => files.iter().find(|(p, _)| *p == rel).map(|(_, b)| b.to_vec()),
    };
    match body {
        Some(b) => Some((content_type(rel), b)),
        None if rel == "index.html" => Some((content_type(rel), NO_EDITOR.as_bytes().to_vec())),
        None => None,
    }
}

// ── routes ───────────────────────────────────────────────────────────────────

/// POST /prompt: checked, queued, and entered in the editor's history, as ComfyUI's post_prompt:
/// the client's own prompt id if it sent one, its client_id kept for its events, a number from
/// the counter (negative at the front of the queue) unless one is given, and only the output
/// nodes partial_execution_targets names, when it does.
fn queue_prompt(srv: &Arc<Server>, req: tiny_http::Request, body: &[u8]) {
    let parsed: Value = match serde_json::from_slice(body) {
        Ok(v) => v,
        Err(e) => return json_reply(req, 400, json!({"error": {"type": "invalid_json", "message": e.to_string()}, "node_errors": {}})),
    };
    let Some(g) = parsed.get("prompt").cloned() else {
        let error = json!({"type": "no_prompt", "message": "No prompt provided", "details": "No prompt provided", "extra_info": {}});
        return json_reply(req, 400, json!({"error": error, "node_errors": {}}));
    };
    let asked_id = match parsed.get("prompt_id") {
        None | Some(Value::Null) => None,
        Some(Value::String(p)) if jobs::valid_job_id(p) => Some(p.clone()),
        Some(_) => {
            let error = json!({"type": "invalid_prompt_id", "message": "prompt_id must be a valid UUID",
                               "details": "prompt_id must be a UUID string in canonical lowercase hyphenated form; omit it to let the server generate one",
                               "extra_info": {}});
            return json_reply(req, 400, json!({"error": error, "node_errors": {}}));
        }
    };
    let names = escaping_names(&g);
    if !names.is_empty() {
        return refuse_prompt(req, "prompt_outputs_failed_validation", &format!("names that leave their folder: {}", names.join(", ")));
    }
    let mut extra = vec![];
    for (k, v) in parsed["extra_data"]["extra_pnginfo"].as_object().into_iter().flatten() {
        let ok = (1..=79).contains(&k.len()) && k.chars().all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-');
        if !ok || k == "prompt" || k == "generator" {
            let msg = format!("extra_pnginfo key {k:?}: a PNG text keyword is 1-79 of A-Z a-z 0-9 _ -, and not prompt or generator");
            return refuse_prompt(req, "invalid_extra_pnginfo", &msg);
        }
        extra.push((k.clone(), crate::image::ascii_json(v)));
    }
    let bad = graph::unsupported(&g);
    let video: Vec<&str> = if srv.no_video {
        let mut v: Vec<&str> = g
            .as_object()
            .into_iter()
            .flatten()
            .filter_map(|(_, n)| n["class_type"].as_str())
            .filter(|c| graph::LTX_NODES.contains(c) && !graph::SHARED_NODES.contains(c))
            .collect();
        v.sort();
        v.dedup();
        v
    } else {
        vec![]
    };
    if !g.is_object() {
        return refuse_prompt(req, "invalid_prompt", "the prompt is not an object of nodes");
    }
    if let Some((nid, node)) = g
        .as_object()
        .into_iter()
        .flatten()
        .filter(|(_, n)| n["class_type"].as_str().is_some_and(|c| bad.iter().any(|b| b == c)))
        .min_by_key(|(k, _)| (k.parse::<u64>().unwrap_or(u64::MAX), k.to_string()))
    {
        // ComfyUI's answer for a node type it does not have, naming the first such node
        let class = node["class_type"].as_str().unwrap_or("");
        let title = node["_meta"]["title"].as_str().unwrap_or(class);
        let error = json!({"type": "missing_node_type", "message": format!("Node '{title}' not found. rustyui does not run {} yet.", bad.join(", ")),
                           "details": format!("Node ID '#{nid}'"), "extra_info": {"node_id": nid, "class_type": class, "node_title": title}});
        return json_reply(req, 400, json!({"error": error, "node_errors": {}}));
    }
    if !video.is_empty() {
        let msg = format!("this backend runs no video nodes (--no-video): {}", video.join(", "));
        return refuse_prompt(req, "prompt_outputs_failed_validation", &msg);
    }
    // the output nodes to run: every one, or those partial_execution_targets names
    let targets: Option<Vec<String>> = parsed
        .get("partial_execution_targets")
        .and_then(Value::as_array)
        .map(|a| a.iter().map(|v| v.as_str().map(str::to_string).unwrap_or_else(|| v.to_string())).collect());
    let mut outputs = outputs_of(&g);
    if let Some(t) = &targets {
        outputs.retain(|o| t.contains(o));
    }
    outputs.sort_by_key(|id| (id.parse::<u64>().unwrap_or(u64::MAX), id.clone()));
    if outputs.is_empty() {
        let error = json!({"type": "prompt_no_outputs", "message": "Prompt has no outputs",
                           "details": "the graph has no SaveImage or SaveVideo node to run", "extra_info": {}});
        return json_reply(req, 400, json!({"error": error, "node_errors": {}}));
    }
    // ComfyUI's extra_data: the client's, its client_id, where it came from, less the sensitive
    // keys (rustyui keeps none of them), and when it was queued
    let mut extra_data = match parsed.get("extra_data") {
        Some(Value::Object(m)) => Value::Object(m.clone()),
        _ => json!({}),
    };
    if let Some(c) = parsed.get("client_id") {
        extra_data["client_id"] = c.clone();
    }
    if extra_data.get("comfy_usage_source").is_none() {
        if let Some(source) = header(&req, "Comfy-Usage-Source") {
            extra_data["comfy_usage_source"] = json!(source);
        }
    }
    if let Some(m) = extra_data.as_object_mut() {
        m.remove("auth_token_comfy_org");
        m.remove("api_key_comfy_org");
    }
    extra_data["create_time"] = json!((now() * 1000.0) as u64);
    let mut s = srv.st.lock().unwrap();
    let number = match parsed.get("number").and_then(Value::as_f64) {
        Some(n) => n,
        None => {
            let n = s.number as f64;
            s.number += 1;
            if truthy(&parsed["front"]) {
                -n
            } else {
                n
            }
        }
    };
    let id = asked_id.unwrap_or_else(new_id);
    let title = match parsed.get("title") {
        Some(Value::String(t)) if !t.is_empty() => json!(t),
        _ => title_of(&g),
    };
    // the editor's own graph, in ComfyUI's workflow format as ComfyUI's frontend sends it (the
    // first editor sent its own format under "rustyui")
    let pnginfo = &parsed["extra_data"]["extra_pnginfo"];
    let workflow = if pnginfo["workflow"].is_object() { &pnginfo["workflow"] } else { &pnginfo["rustyui"] };
    s.ui.push(json!({
        "id": id, "number": number_json(number), "title": title, "queued": now(), "status": "queued", "outputs": [],
        "error": null, "seconds": null, "graph": g, "workflow": workflow,
    }));
    save_ui(srv, &mut s);
    // ComfyUI's queue is a heap on the number: the lowest runs first (equal ones in turn here)
    let at = s.queue.iter().position(|j| j.number > number).unwrap_or(s.queue.len());
    s.queue.insert(at, Job { id: id.clone(), number, graph: g, extra, extra_data, outputs });
    emit_status(s.queue.len() + usize::from(s.running.is_some()));
    s.last = Instant::now();
    wake(srv, &mut s);
    drop(s);
    json_reply(req, 200, json!({"prompt_id": id, "number": number_json(number), "node_errors": {}}));
}

/// POST /rustyui/write {"idea", "prompt"}: the prompt writer's answer {"prompt", "tokens", "seconds"},
/// once the worker gets to it (writes go before renders). The idea, or the current prompt to work
/// from, may be empty but not both.
fn write_prompt(srv: &Arc<Server>, req: tiny_http::Request, body: &[u8]) {
    if srv.writer.is_none() {
        return json_reply(req, 404, json!({"error": {"message": "this backend has no prompt writer (rustyui-backend serve --writer)"}}));
    }
    let v: Value = serde_json::from_slice(body).unwrap_or(Value::Null);
    let idea = v["idea"].as_str().unwrap_or("").chars().take(2000).collect::<String>();
    let current = v["prompt"].as_str().unwrap_or("").chars().take(4000).collect::<String>();
    if idea.trim().is_empty() && current.trim().is_empty() {
        return json_reply(req, 400, json!({"error": {"message": "nothing to write from: give an idea, or a prompt to work on"}}));
    }
    let (tx, rx) = std::sync::mpsc::channel();
    {
        let mut s = srv.st.lock().unwrap();
        s.writes.push_back(WriteJob { idea, current, reply: tx });
        s.last = Instant::now();
        wake(srv, &mut s);
    }
    match rx.recv_timeout(Duration::from_secs(600)) {
        Ok(Ok(w)) => json_reply(req, 200, json!({"prompt": w.prompt, "tokens": w.tokens, "seconds": w.seconds})),
        Ok(Err(e)) => json_reply(req, 500, json!({"error": {"message": e}})),
        Err(_) => json_reply(req, 504, json!({"error": {"message": "the writer did not answer within 10 minutes"}})),
    }
}

/// Counts a request in flight from when it arrives until it is answered, and notes when the last
/// one came, and the last that was more than an open editor polling.
struct InFlight<'a> {
    srv: &'a Server,
    used: bool,
}

impl<'a> InFlight<'a> {
    fn new(srv: &'a Server) -> Self {
        srv.in_flight.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        *srv.last_request.lock().unwrap() = Instant::now();
        Self { srv, used: false }
    }

    fn used(&mut self) {
        self.used = true;
        *self.srv.last_use.lock().unwrap() = Instant::now();
    }
}

impl Drop for InFlight<'_> {
    fn drop(&mut self) {
        *self.srv.last_request.lock().unwrap() = Instant::now();
        if self.used {
            *self.srv.last_use.lock().unwrap() = Instant::now();
        }
        self.srv.in_flight.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
    }
}

/// The requests an open editor (or a tool waiting) repeats every few seconds whether or not
/// anyone is doing anything.
fn polling(method: &str, api: &str) -> bool {
    method == "GET"
        && matches!(api, "/rustyui/status" | "/rustyui/history" | "/progress" | "/queue" | "/system_stats" | "/jobs" | "/prompt")
}

fn handle(srv: &Arc<Server>, mut req: tiny_http::Request) {
    let mut counted = InFlight::new(srv);
    // a HEAD is answered as its GET (tiny_http leaves the body out), as aiohttp answers it for ComfyUI
    let method = match req.method().to_string().to_uppercase() {
        m if m == "HEAD" => "GET".to_string(),
        m => m,
    };
    let url = req.url().to_string();
    let (path, q) = url.split_once('?').unwrap_or((url.as_str(), ""));
    let (path, q) = (path.to_string(), q.to_string());
    let why = {
        let r = &req;
        refusal(&method, &path, &|name| header(r, name).map(str::to_string))
    };
    if let Some(why) = why {
        eprintln!("[rustyui] refused {method} {path}: {why}");
        return reply(req, 403, "text/plain; charset=utf-8", format!("rustyui answers this box only: {why}\n").into_bytes());
    }
    // ComfyUI's websocket (progress, the queue, finished outputs), before any body is read: an
    // upgrade request has none, and reading would wait on the open connection
    let upgrade = req.headers().iter().any(|h| h.field.equiv("Upgrade") && h.value.as_str().eq_ignore_ascii_case("websocket"));
    if upgrade && method == "GET" && matches!(path.as_str(), "/ws" | "/api/ws") {
        return serve_ws(srv, req, &q);
    }
    // as much as ComfyUI's default --max-upload-size (100 MB) and the form around it
    let mut body = vec![];
    let _ = req.as_reader().take(101 << 20).read_to_end(&mut body);
    // ComfyUI answers its API under /api/ as well
    let api = match path.strip_prefix("/api") {
        Some(rest) if rest.starts_with('/') => rest,
        _ => path.as_str(),
    };
    let api = if api.len() > 1 { api.trim_end_matches('/') } else { api };
    if !polling(&method, api) {
        counted.used();
    }
    // the editor's own state (settings, user files, extension scripts and the like), the jobs
    // API, and files (views, uploads, model folders)
    let req = match user_routes::route(srv, req, &method, api, &q, &body) {
        Ok(()) => return,
        Err(req) => req,
    };
    let req = match jobs::route(srv, req, &method, api, &q, &body) {
        Ok(()) => return,
        Err(req) => req,
    };
    let req = match files::route(srv, req, &method, api, &q, &body) {
        Ok(()) => return,
        Err(req) => req,
    };
    match (method.as_str(), api) {
        // ComfyUI's API
        ("GET", "/queue") => {
            let s = srv.st.lock().unwrap();
            let running: Vec<Value> = s.running.iter().map(Job::item).collect();
            let pending: Vec<Value> = s.queue.iter().map(Job::item).collect();
            drop(s);
            json_reply(req, 200, json!({"queue_running": running, "queue_pending": pending}));
        }
        ("POST", "/queue") => {
            // {"clear": true} empties the queue, {"delete": [ids]} takes those off it
            let v: Value = serde_json::from_slice(&body).unwrap_or(Value::Null);
            let mut s = srv.st.lock().unwrap();
            let mut gone: Vec<String> = vec![];
            if truthy(&v["clear"]) {
                gone.extend(s.queue.drain(..).map(|j| j.id));
            }
            for id in v["delete"].as_array().into_iter().flatten().filter_map(Value::as_str) {
                if let Some(at) = s.queue.iter().position(|j| j.id == id) {
                    gone.extend(s.queue.remove(at).map(|j| j.id));
                }
            }
            for id in &gone {
                ui_update(srv, &mut s, id, json!({"status": "error", "error": "taken off the queue", "finished": now()}));
            }
            if !gone.is_empty() {
                emit_status(s.queue.len() + usize::from(s.running.is_some()));
            }
            drop(s);
            reply(req, 200, "text/plain", vec![]);
        }
        ("GET", "/prompt") => {
            let s = srv.st.lock().unwrap();
            let remaining = s.queue.len() + usize::from(s.running.is_some());
            drop(s);
            json_reply(req, 200, json!({"exec_info": {"queue_remaining": remaining}}));
        }
        ("POST", "/prompt") => queue_prompt(srv, req, &body),
        ("GET", p) if p == "/history" || p.starts_with("/history/") => {
            let s = srv.st.lock().unwrap();
            let v = match p.strip_prefix("/history/") {
                Some(id) => s.history.get(id).map(|e| json!({ id: e })).unwrap_or(json!({})),
                None => history_page(&s, query(&q, "max_items").parse().ok(), query(&q, "offset").parse().ok()),
            };
            drop(s);
            json_reply(req, 200, v);
        }
        ("POST", "/history") => {
            // {"clear": true} forgets the history, {"delete": [ids]} those entries (the files stay)
            let v: Value = serde_json::from_slice(&body).unwrap_or(Value::Null);
            let mut s = srv.st.lock().unwrap();
            if truthy(&v["clear"]) {
                s.history.clear();
                s.order.clear();
            }
            for id in v["delete"].as_array().into_iter().flatten().filter_map(Value::as_str) {
                s.history.remove(id);
                s.order.retain(|o| o != id);
            }
            drop(s);
            reply(req, 200, "text/plain", vec![]);
        }
        ("GET", "/embeddings") => {
            let names: Vec<String> = list(&srv.models.join("embeddings"), &[".safetensors", ".pt", ".bin"])
                .into_iter()
                .map(|n| n.rsplit_once('.').map(|(a, _)| a.to_string()).unwrap_or(n))
                .collect();
            json_reply(req, 200, json!(names));
        }
        ("GET", "/object_info") => json_reply(req, 200, object_info(&srv.models, None, &srv.prefix, srv.no_video)),
        // the class percent-decoded, as aiohttp's route gives it ("RIFE%20VFI")
        ("GET", p) if p.starts_with("/object_info/") => {
            json_reply(req, 200, object_info(&srv.models, Some(&percent_decode(&p["/object_info/".len()..])), &srv.prefix, srv.no_video))
        }
        ("GET", "/progress") => json_reply(req, 200, json!({"executing": crate::progress::json()})),
        ("POST", "/free") => {
            // the worker ends once the current prompt (if any) is done, taking the models and
            // the CUDA context with it
            let mut s = srv.st.lock().unwrap();
            let loaded = s.worker;
            s.free = true;
            srv.cv.notify_all();
            drop(s);
            json_reply(req, 200, json!({"loaded": loaded}));
        }
        ("POST", "/interrupt") => {
            // the prompt running stops at its next step or node; with a prompt_id, only if that
            // one is running
            let v: Value = serde_json::from_slice(&body).unwrap_or(Value::Null);
            let target = v["prompt_id"].as_str().filter(|p| !p.is_empty()).map(str::to_string);
            let s = srv.st.lock().unwrap();
            match (target, s.running.as_ref().map(|j| j.id.clone())) {
                (Some(t), Some(r)) if t == r => stop_running(&r),
                (Some(t), _) => eprintln!("[rustyui] {t} is not running; not interrupted"),
                (None, Some(r)) => stop_running(&r),
                (None, None) => {
                    crate::progress::interrupt(None);
                }
            }
            drop(s);
            reply(req, 200, "text/plain", vec![]);
        }

        // the editor's own
        ("POST", "/rustyui/write") => write_prompt(srv, req, &body),
        ("GET", "/rustyui/thumb") => match picture(&srv.out, &query(&q, "filename"), &query(&q, "subfolder")) {
            // a small copy for lists on a phone (the same pictures /view hands out, no others)
            Some(file) => match thumbnail(&file, query(&q, "w").parse::<usize>().unwrap_or(320).clamp(32, 1024)) {
                Ok((ctype, body)) => reply_cached(req, 200, ctype, body, "private, max-age=86400"),
                Err(e) => json_reply(req, 500, json!({"error": {"message": format!("{e:#}")}})),
            },
            None => json_reply(req, 404, json!({"error": "not a picture in the output folder"})),
        },
        ("GET", "/rustyui/status") => {
            let s = srv.st.lock().unwrap();
            let v = json!({
                "engine": "up", "loaded": s.worker, "running": s.running.is_some() as u8, "pending": s.queue.len(),
                "writer": srv.writer.is_some(),
                "executing": crate::progress::json(), "history_version": s.ui_version,
            });
            drop(s);
            json_reply(req, 200, v);
        }
        ("GET", "/rustyui/history") => {
            // newest first, without the graphs (GET /rustyui/history/<id> has them)
            let s = srv.st.lock().unwrap();
            let items: Vec<Value> =
                s.ui.iter()
                    .rev()
                    .map(|e| {
                        let mut e = e.clone();
                        if let Some(m) = e.as_object_mut() {
                            m.remove("graph");
                            m.remove("workflow");
                        }
                        e
                    })
                    .collect();
            let v = json!({"items": items, "version": s.ui_version});
            drop(s);
            json_reply(req, 200, v);
        }
        ("POST", "/rustyui/history/forget") => {
            // takes entries off the list; the pictures are not touched
            let ids: Vec<String> = serde_json::from_slice::<Value>(&body)
                .ok()
                .and_then(|v| v["ids"].as_array().cloned())
                .unwrap_or_default()
                .iter()
                .filter_map(|v| v.as_str().map(str::to_string))
                .collect();
            let mut s = srv.st.lock().unwrap();
            s.ui.retain(|e| {
                !ids.iter().any(|id| e["id"] == id.as_str()) || matches!(e["status"].as_str(), Some("queued") | Some("running"))
            });
            save_ui(srv, &mut s);
            drop(s);
            json_reply(req, 200, json!({"ok": true}));
        }
        ("GET", p) if p.starts_with("/rustyui/history/") => {
            let id = &p["/rustyui/history/".len()..];
            let found = srv.st.lock().unwrap().ui.iter().find(|e| e["id"] == id).cloned();
            match found {
                Some(e) => json_reply(req, 200, e),
                None => json_reply(req, 404, json!({"error": {"message": "no such prompt in the history"}})),
            }
        }

        // the simple editor's files are relative to /simple/, so its address without the slash
        // goes there (a relative Location, which also holds behind a proxy's prefix)
        ("GET", "/simple") if path == "/simple" => {
            let to = tiny_http::Header::from_bytes(&b"Location"[..], &b"simple/"[..]).unwrap();
            let _ = req.respond(tiny_http::Response::empty(308).with_header(to));
        }
        // the editors: anything else a GET asks for is one of their files (a bundle's file names
        // change with its content, so those are kept)
        ("GET", _) => match editor_file(srv.frontend.as_deref(), &path) {
            Some((ctype, body)) if path.contains("/assets/") => reply_cached(req, 200, ctype, body, "private, max-age=31536000, immutable"),
            Some((ctype, body)) => reply(req, 200, ctype, body),
            None => json_reply(req, 404, json!({"error": format!("no route {method} {path}")})),
        },
        _ => json_reply(req, 404, json!({"error": format!("no route {method} {path}")})),
    }
}

/// The listening socket systemd passes (socket activation: LISTEN_FDS=1 for this process, fd 3),
/// refused unless it is on loopback.
fn activated_listener() -> Result<Option<std::net::TcpListener>> {
    let ours = std::env::var("LISTEN_PID").ok().and_then(|p| p.parse::<u32>().ok()) == Some(std::process::id());
    let fds = std::env::var("LISTEN_FDS").ok().and_then(|n| n.parse::<i32>().ok()).unwrap_or(0);
    if !ours || fds < 1 {
        return Ok(None);
    }
    use std::os::fd::FromRawFd;
    let listener = unsafe { std::net::TcpListener::from_raw_fd(3) };
    let addr = listener.local_addr().context("the socket systemd passed")?;
    if !addr.ip().is_loopback() {
        bail!("systemd passed a socket on {addr}: rustyui serves this box only (127.0.0.1 or [::1])");
    }
    Ok(Some(listener))
}

pub fn serve(o: Options) -> Result<()> {
    let Options { listen, out, idle, exit_idle, frontend, prefix, mirror, writer, no_video } = o;
    let activated = activated_listener()?;
    let exit_idle = exit_idle.or(activated.as_ref().map(|_| 120));
    let home = PathBuf::from(std::env::var("HOME").unwrap_or_default());
    let models = crate::paths::models_dir();
    let state_dir = std::env::var_os("XDG_STATE_HOME").map(PathBuf::from).unwrap_or_else(|| home.join(".local/state")).join("rustyui");
    let ui_file = state_dir.join("history.json");
    // the editor's history from the last run; what was still queued or running then is lost
    let mut ui: Vec<Value> = fs::read_to_string(&ui_file)
        .ok()
        .and_then(|t| serde_json::from_str::<Value>(&t).ok())
        .and_then(|v| v.as_array().cloned())
        .unwrap_or_default();
    for e in ui.iter_mut().filter(|e| matches!(e["status"].as_str(), Some("queued") | Some("running"))) {
        e["status"] = json!("lost");
        e["error"] = json!("the backend stopped before this prompt finished; its picture, if any, is in the output folder");
    }
    let srv = Arc::new(Server {
        st: Mutex::new(State {
            queue: VecDeque::new(),
            writes: VecDeque::new(),
            running: None,
            history: HashMap::new(),
            order: VecDeque::new(),
            number: 0,
            free: false,
            last: Instant::now(),
            worker: false,
            ui,
            // from the clock, so an open editor sees a restart as a change
            ui_version: (now() * 1000.0) as u64,
            cuda: false,
        }),
        cv: Condvar::new(),
        last_request: Mutex::new(Instant::now()),
        last_use: Mutex::new(Instant::now()),
        in_flight: std::sync::atomic::AtomicUsize::new(0),
        out,
        models,
        idle,
        frontend,
        ui_file,
        prefix,
        mirror,
        writer,
        no_video,
    });
    let (server, addr) = match activated {
        Some(listener) => {
            let addr = listener.local_addr()?;
            (tiny_http::Server::from_listener(listener, None).map_err(|e| anyhow::anyhow!("systemd's socket: {e}"))?, addr)
        }
        None => {
            let addr = loopback_addr(&listen)?;
            (tiny_http::Server::http(addr).map_err(|e| anyhow::anyhow!("cannot listen on {addr}: {e}"))?, addr)
        }
    };
    eprintln!(
        "[rustyui] http://{addr}/ (the editor and ComfyUI's API), writing {}_NNNNN_.png to {}{}, models released after {idle}s idle{}{}",
        srv.prefix,
        srv.out.display(),
        srv.mirror.as_ref().map(|m| format!(" and copying each to {m}")).unwrap_or_default(),
        exit_idle.map(|e| format!(", exits after {e}s with nothing to do")).unwrap_or_default(),
        srv.frontend.as_ref().map(|d| format!(", frontend from {}", d.display())).unwrap_or_default()
    );
    // the exit is decided here, between requests, so none is taken in and then dropped
    let exit_idle = exit_idle.map(Duration::from_secs);
    loop {
        match server.recv_timeout(Duration::from_secs(2)) {
            Ok(Some(req)) => {
                let srv = srv.clone();
                thread::spawn(move || handle(&srv, req));
            }
            Ok(None) => {
                if let Some(after) = exit_idle {
                    if done_for_now(&srv, after) {
                        eprintln!("[rustyui] nothing to do for {}s; exiting (the next request starts it again)", after.as_secs());
                        std::process::exit(0);
                    }
                }
            }
            Err(e) => bail!("accepting connections: {e}"),
        }
    }
}

/// Whether the process can go: nothing queued, running, loaded or being answered, and either no
/// requests at all for `after`, or only an open editor polling while the CUDA context of a
/// finished session (about 220 MiB of the card) is held for nothing. Exiting gives that back; the
/// editor's next poll starts a fresh process, which has none until it runs something.
fn done_for_now(srv: &Server, after: Duration) -> bool {
    let s = srv.st.lock().unwrap();
    let quiet = !s.worker
        && s.queue.is_empty()
        && s.writes.is_empty()
        && s.running.is_none()
        && srv.in_flight.load(std::sync::atomic::Ordering::SeqCst) == 0;
    let unasked = srv.last_request.lock().unwrap().elapsed() > after;
    let unused = s.cuda && srv.last_use.lock().unwrap().elapsed() > after;
    quiet && (unasked || unused)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn headers<'a>(pairs: &'a [(&'static str, &'a str)]) -> impl Fn(&'static str) -> Option<String> + 'a {
        move |name| pairs.iter().find(|(k, _)| k.eq_ignore_ascii_case(name)).map(|(_, v)| v.to_string())
    }

    #[test]
    fn listens_on_loopback_only() {
        assert!(loopback_addr("127.0.0.1:8190").is_ok());
        assert!(loopback_addr("[::1]:8190").is_ok());
        for bad in ["0.0.0.0:8190", "[::]:8190", "10.0.0.5:8190", "192.168.0.10:8190"] {
            assert!(loopback_addr(bad).is_err(), "{bad} should be refused");
        }
    }

    #[test]
    fn tools_and_the_editor_are_allowed() {
        // Python tools and curl: a loopback Host, nothing else
        assert_eq!(refusal("POST", "/prompt", &headers(&[("Host", "127.0.0.1:8190")])), None);
        assert_eq!(refusal("GET", "/history/x", &headers(&[])), None);
        // the editor's own requests
        let own = [("Host", "127.0.0.1:8190"), ("Origin", "http://127.0.0.1:8190"), ("Sec-Fetch-Site", "same-origin")];
        assert_eq!(refusal("POST", "/api/prompt", &headers(&own)), None);
        assert_eq!(refusal("GET", "/", &headers(&[("Host", "localhost:8190"), ("Sec-Fetch-Site", "none")])), None);
        // opening the editor from a link elsewhere
        let nav =
            [("Host", "127.0.0.1:8190"), ("Sec-Fetch-Site", "cross-site"), ("Sec-Fetch-Mode", "navigate"), ("Sec-Fetch-Dest", "document")];
        assert_eq!(refusal("GET", "/", &headers(&nav)), None);
    }

    #[test]
    fn web_pages_are_refused() {
        let cases: Vec<(&str, &str, Vec<(&'static str, &str)>)> = vec![
            ("POST", "/prompt", vec![("Host", "127.0.0.1:8190"), ("Origin", "https://evil.example")]),
            ("POST", "/api/prompt", vec![("Host", "127.0.0.1:8190"), ("Origin", "http://127.0.0.1:8188")]),
            ("POST", "/prompt", vec![("Host", "127.0.0.1:8190"), ("Origin", "null")]),
            ("GET", "/queue", vec![("Host", "evil.example:8190")]),
            ("GET", "/view", vec![("Host", "127.0.0.1:8190"), ("Sec-Fetch-Site", "cross-site"), ("Sec-Fetch-Mode", "no-cors")]),
            ("GET", "/rustyui/history", vec![("Host", "127.0.0.1:8190"), ("Sec-Fetch-Site", "same-site")]),
            ("GET", "/assets/index.js", vec![("Host", "127.0.0.1:8190"), ("Sec-Fetch-Site", "cross-site"), ("Sec-Fetch-Mode", "no-cors")]),
            ("GET", "/assets/index.js", vec![("Host", "127.0.0.1:8190"), ("Sec-Fetch-Site", "cross-site"), ("Sec-Fetch-Mode", "navigate")]),
        ];
        for (m, p, hs) in cases {
            assert!(refusal(m, p, &headers(&hs)).is_some(), "{m} {p} {hs:?}");
        }
    }

    #[test]
    fn host_names() {
        assert!(loopback_name(host_name("127.0.0.1:8190")));
        assert!(loopback_name(host_name("localhost:8190")));
        assert!(loopback_name(host_name("[::1]:8190")));
        assert!(loopback_name(host_name("127.0.0.1")));
        for bad in ["evil.example:8190", "myhost:8190", "10.0.0.5:8190", "2130706433:8190", "[fe80::1]:8190"] {
            assert!(!loopback_name(host_name(bad)), "{bad} should not count as loopback");
        }
    }

    #[test]
    fn graphs_that_write_or_read_outside_are_refused() {
        let ok = json!({"9": {"class_type": "SaveImage", "inputs": {"filename_prefix": "portraits/x", "images": ["8", 0]}},
                        "4": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "sd_xl_base_1.0.safetensors"}}});
        assert!(escaping_names(&ok).is_empty());
        let bad = json!({"9": {"class_type": "SaveImage", "inputs": {"filename_prefix": "/home/user/Pictures/x", "images": ["8", 0]}},
                         "4": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "../../../.ssh/id_ed25519"}},
                         "5": {"class_type": "LoraLoader", "inputs": {"lora_name": "ok.safetensors"}}});
        assert_eq!(escaping_names(&bad).len(), 2);
    }

    #[test]
    fn pictures_stay_in_the_output_folder() {
        let root = std::env::temp_dir().join(format!("rustyui-test-{}", std::process::id()));
        let (out, elsewhere) = (root.join("out"), root.join("elsewhere"));
        fs::create_dir_all(out.join("sub")).unwrap();
        fs::create_dir_all(&elsewhere).unwrap();
        fs::write(out.join("a.png"), b"x").unwrap();
        fs::write(out.join("sub/b.png"), b"x").unwrap();
        fs::write(out.join("notes.txt"), b"x").unwrap();
        fs::write(elsewhere.join("c.png"), b"x").unwrap();
        std::os::unix::fs::symlink(&elsewhere, out.join("link")).unwrap();
        assert!(picture(&out, "a.png", "").is_some());
        assert!(picture(&out, "b.png", "sub").is_some());
        for (f, s) in
            [("notes.txt", ""), ("c.png", "link"), ("../elsewhere/c.png", ""), ("/etc/passwd", ""), ("c.png", "/tmp"), ("missing.png", "")]
        {
            assert!(picture(&out, f, s).is_none(), "{f} {s}");
        }
        fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn query_strings() {
        assert_eq!(query("filename=rustyui_00001_.png&subfolder=", "filename"), "rustyui_00001_.png");
        assert_eq!(query("filename=a%20b.png&subfolder=x%2Fy", "subfolder"), "x/y");
        assert_eq!(query("filename=a%20b.png", "filename"), "a b.png");
        assert_eq!(query("fresh=1", "filename"), "");
    }

    #[test]
    fn editor_files_stay_in_their_folder() {
        let dir = std::env::temp_dir().join(format!("rustyui-editor-test-{}", std::process::id()));
        fs::create_dir_all(dir.join("assets")).unwrap();
        fs::write(dir.join("index.html"), b"<!doctype html>").unwrap();
        fs::write(dir.join("assets/index-abc.js"), b"x").unwrap();
        fs::write(dir.parent().unwrap().join(format!("rustyui-secret-{}", std::process::id())), b"s").unwrap();
        assert_eq!(editor_file(Some(&dir), "/").unwrap(), ("text/html; charset=utf-8", b"<!doctype html>".to_vec()));
        assert_eq!(editor_file(Some(&dir), "/assets/index-abc.js").unwrap().0, "text/javascript; charset=utf-8");
        let secret = format!("/../rustyui-secret-{}", std::process::id());
        for bad in [secret.as_str(), "/assets/missing.js", "//etc/passwd", "/assets/../../x"] {
            assert!(editor_file(Some(&dir), bad).is_none(), "{bad}");
        }
        // without the editor, "/" still answers, saying how to build it
        let empty = dir.join("empty");
        fs::create_dir_all(&empty).unwrap();
        assert!(String::from_utf8(editor_file(Some(&empty), "/").unwrap().1).unwrap().contains("bun run build"));
        fs::remove_file(dir.parent().unwrap().join(format!("rustyui-secret-{}", std::process::id()))).unwrap();
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn only_real_use_keeps_a_used_process() {
        for p in ["/rustyui/status", "/rustyui/history", "/progress", "/queue", "/system_stats"] {
            assert!(polling("GET", p), "{p}");
        }
        for (m, p) in
            [("POST", "/prompt"), ("GET", "/view"), ("GET", "/history/x"), ("GET", "/rustyui/history/x"), ("GET", "/"), ("POST", "/free")]
        {
            assert!(!polling(m, p), "{m} {p}");
        }
    }

    #[test]
    fn renders_are_copied_without_overwriting() {
        let args = mirror_args("host:ComfyUI/output/", Path::new("/o/sub/a_00001_.png"), Path::new("sub/a_00001_.png"));
        assert!(args.contains(&"--ignore-existing".to_string()) && args.contains(&"--mkpath".to_string()));
        assert_eq!(args.last().unwrap(), "host:ComfyUI/output/sub/a_00001_.png");
        let outputs = json!({"16": {"images": [{"filename": "a_00001_.png", "subfolder": "sub", "type": "output"},
                                               {"filename": "b_00001_.png", "subfolder": "", "type": "output"}]}});
        let files = files_of(outputs.as_object().unwrap(), Path::new("/o"));
        assert_eq!(files, vec![PathBuf::from("/o/sub/a_00001_.png"), PathBuf::from("/o/b_00001_.png")]);
    }

    #[test]
    fn thumbnails_shrink_and_never_grow() {
        let dir = std::env::temp_dir().join(format!("rustyui-thumb-test-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let (w, h) = (64usize, 32usize);
        let rgb: Vec<u8> = (0..w * h * 3).map(|i| (i % 251) as u8).collect();
        let file = dir.join("p.png");
        fs::write(&file, crate::image::encode_png(&rgb, w as u32, h as u32).unwrap()).unwrap();
        let (ctype, small) = thumbnail(&file, 16).unwrap();
        fs::write(dir.join("t.png"), &small).unwrap();
        let t = crate::image::read_png(&dir.join("t.png")).unwrap();
        assert_eq!((ctype, t.width, t.height), ("image/png", 16, 8));
        // asked for more than it has: the file as it is
        assert_eq!(thumbnail(&file, 512).unwrap().1, fs::read(&file).unwrap());
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn untitled_prompts_take_their_save_prefix() {
        let g = json!({"9": {"class_type": "SaveImage", "inputs": {"filename_prefix": "portraits/lighthouse", "images": ["8", 0]}}});
        assert_eq!(title_of(&g), json!("portraits/lighthouse"));
        assert_eq!(title_of(&json!({})), Value::Null);
    }
}
