//! What the worker is doing right now, and ComfyUI's websocket events about it. For GET /progress:
//! the prompt, the node being evaluated and, inside a sampler, the step. For the editor, as
//! execution.py and comfy_execution/progress.py send them: execution_start and execution_cached,
//! executing for each node that runs, progress for each step, progress_state (ComfyUI's progress
//! registry: every node started in the prompt with its value, max and state), executed for each
//! output, then execution_success, execution_error or execution_interrupted. The status messages
//! among those are kept for the prompt's history entry, as ComfyUI keeps them. Written by the
//! worker thread, read by the HTTP threads.
//!
//! Each event goes where ComfyUI sends it: to the websocket client that queued the prompt (its
//! client_id), or, for a prompt that came without one (the API tools), executing, progress and
//! executed go to every client and the status messages to none. "status" (the queue's length)
//! always goes to every client.

use serde_json::{json, Value};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

struct Now {
    prompt: String,
    /// the websocket client that queued the prompt (extra_data.client_id)
    client: Option<String>,
    /// (node id, class) of every node being evaluated, outermost first
    stack: Vec<(String, String)>,
    /// the node entered last (the one a failure happened in, once the stack has unwound)
    last: Option<(String, String)>,
    value: usize,
    max: usize,
    /// ComfyUI's progress registry: each node started in this prompt, in the order started,
    /// with its value, max and state (running, finished or error)
    nodes: Vec<(String, f64, f64, &'static str)>,
    /// the nodes that ran to the end (ComfyUI's `executed`)
    executed: Vec<String>,
    /// the status messages the history keeps, [event, data]
    messages: Vec<Value>,
}

static NOW: Mutex<Option<Now>> = Mutex::new(None);

/// POST /interrupt: the running prompt stops at its next step or node, as ComfyUI's does
static INTERRUPT: AtomicBool = AtomicBool::new(false);

/// A prompt the worker has taken off the queue but not started yet (it waits for ComfyUI and the
/// card first), to stop as soon as it starts
static STOP_AT_START: Mutex<Option<String>> = Mutex::new(None);

/// What an interrupted prompt fails with
#[derive(Debug)]
pub struct Interrupted;

impl std::fmt::Display for Interrupted {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Processing interrupted")
    }
}

impl std::error::Error for Interrupted {}

/// The websocket clients, by sid, each with its queue (server.rs drains each into its socket)
static CLIENTS: Mutex<Vec<(String, std::sync::mpsc::Sender<String>)>> = Mutex::new(vec![]);

/// A client connected with this sid; one already there under the same sid is replaced, as
/// ComfyUI replaces a reconnecting tab's old socket
pub fn subscribe(sid: &str, tx: std::sync::mpsc::Sender<String>) {
    let mut c = CLIENTS.lock().unwrap();
    c.retain(|(s, _)| s != sid);
    c.push((sid.to_string(), tx));
}

/// One of ComfyUI's websocket messages, {"type", "data"}, to the client `to` or, with None, to
/// every client (gone ones dropped)
pub fn send(kind: &str, data: Value, to: Option<&str>) {
    let msg = json!({"type": kind, "data": data}).to_string();
    CLIENTS.lock().unwrap().retain(|(sid, tx)| to.is_some_and(|t| t != sid) || tx.send(msg.clone()).is_ok());
}

/// A message to every client
pub fn emit(kind: &str, data: Value) {
    send(kind, data, None)
}

fn now_ms() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_millis() as u64).unwrap_or(0)
}

/// execution.py's add_message: stamped, kept for the history, and sent to the prompt's client
/// (or, when `broadcast`, to every client if the prompt has none)
fn add_message(n: &mut Now, event: &str, mut data: Value, broadcast: bool) {
    data["timestamp"] = json!(now_ms());
    n.messages.push(json!([event, data]));
    if n.client.is_some() || broadcast {
        send(event, data, n.client.as_deref());
    }
}

/// ComfyUI's progress_state: every node started so far, to the prompt's client (or every client)
fn send_state(n: &Now) {
    let nodes: serde_json::Map<String, Value> = n
        .nodes
        .iter()
        .map(|(id, value, max, state)| {
            (
                id.clone(),
                json!({"value": value, "max": max, "state": state, "node_id": id, "prompt_id": n.prompt,
                                "display_node_id": id, "parent_node_id": null, "real_node_id": id}),
            )
        })
        .collect();
    send("progress_state", json!({"prompt_id": n.prompt, "nodes": nodes}), n.client.as_deref());
}

fn set_state(n: &mut Now, id: &str, f: impl FnOnce(&mut (String, f64, f64, &'static str))) {
    if !n.nodes.iter().any(|e| e.0 == id) {
        n.nodes.push((id.to_string(), 0.0, 1.0, "running"));
    }
    if let Some(e) = n.nodes.iter_mut().find(|e| e.0 == id) {
        f(e);
    }
}

/// A prompt starts: the interrupt flag is cleared (one sent while nothing ran is not kept for
/// the next prompt, as in ComfyUI) unless this prompt was stopped before it started, and
/// execution_start goes out
pub fn start(prompt: &str, client: Option<&str>) {
    let stopped = STOP_AT_START.lock().unwrap().take().is_some_and(|p| p == prompt);
    INTERRUPT.store(stopped, Ordering::SeqCst);
    let mut n = Now {
        prompt: prompt.to_string(),
        client: client.map(str::to_string),
        stack: vec![],
        last: None,
        value: 0,
        max: 0,
        nodes: vec![],
        executed: vec![],
        messages: vec![],
    };
    add_message(&mut n, "execution_start", json!({"prompt_id": prompt}), false);
    *NOW.lock().unwrap() = Some(n);
}

/// The nodes whose outputs were already cached when the prompt started
pub fn cached(nodes: &[String]) {
    if let Some(n) = NOW.lock().unwrap().as_mut() {
        let prompt = n.prompt.clone();
        add_message(n, "execution_cached", json!({"nodes": nodes, "prompt_id": prompt}), false);
    }
}

/// An output node's files are written: executed, with its outputs as the history has them
pub fn executed(node: &str, output: &Value) {
    if let Some(n) = NOW.lock().unwrap().as_ref() {
        send("executed", json!({"node": node, "display_node": node, "output": output, "prompt_id": n.prompt}), n.client.as_deref());
    }
}

/// How a prompt ended
pub enum Outcome<'a> {
    Success,
    /// (node id, class, message): the node it failed in
    Error(&'a str, &'a str, &'a str),
    /// (node id, class): the node it was stopped in
    Interrupted(&'a str, &'a str),
}

/// The prompt ended: execution_success, execution_error or execution_interrupted. Returns the
/// status messages for its history entry.
pub fn finished(outcome: Outcome) -> Vec<Value> {
    let mut guard = NOW.lock().unwrap();
    let Some(n) = guard.as_mut() else { return vec![] };
    let prompt = n.prompt.clone();
    let executed = json!(n.executed);
    match outcome {
        Outcome::Success => add_message(n, "execution_success", json!({"prompt_id": prompt}), false),
        Outcome::Error(node, class, message) => {
            set_state(n, node, |e| e.3 = "error");
            add_message(
                n,
                "execution_error",
                json!({"prompt_id": prompt, "node_id": node, "node_type": class, "executed": executed, "exception_message": message,
                       "exception_type": "RuntimeError", "traceback": [], "current_inputs": {}, "current_outputs": []}),
                false,
            )
        }
        Outcome::Interrupted(node, class) => add_message(
            n,
            "execution_interrupted",
            json!({"prompt_id": prompt, "node_id": node, "node_type": class, "executed": executed}),
            true,
        ),
    }
    n.messages.clone()
}

/// The prompt is in the history: its client is told the queue has moved on (executing with no
/// node, as main.py sends it after task_done), and nothing runs any more
pub fn done() {
    let mut now = NOW.lock().unwrap();
    if let Some(n) = now.as_ref().filter(|n| n.client.is_some()) {
        send("executing", json!({"node": null, "prompt_id": n.prompt}), n.client.as_deref());
    }
    *now = None;
}

/// Stops the running prompt at its next step or node. With an id, only if that prompt is the one
/// running; says whether it was.
pub fn interrupt(prompt: Option<&str>) -> bool {
    let now = NOW.lock().unwrap();
    let running = now.as_ref().map(|n| n.prompt.as_str());
    if prompt.is_none() || (running.is_some() && running == prompt) {
        INTERRUPT.store(true, Ordering::SeqCst);
        return running.is_some();
    }
    false
}

/// Stops `prompt` as soon as it starts (the worker has it but has not started it yet)
pub fn interrupt_at_start(prompt: &str) {
    *STOP_AT_START.lock().unwrap() = Some(prompt.to_string());
}

/// Fails with Interrupted once /interrupt has asked
pub fn check() -> anyhow::Result<()> {
    if INTERRUPT.load(Ordering::SeqCst) {
        return Err(Interrupted.into());
    }
    Ok(())
}

/// One sampler step finished (`value` of `max`). Like ComfyUI's progress hook, it is where an
/// interrupt takes effect.
pub fn step(value: usize, max: usize) -> anyhow::Result<()> {
    check()?;
    let mut now = NOW.lock().unwrap();
    if let Some(n) = now.as_mut() {
        n.value = value;
        n.max = max;
        let node = n.stack.last().map(|(id, _)| id.clone()).unwrap_or_default();
        set_state(n, &node, |e| {
            e.1 = value as f64;
            e.2 = max as f64;
        });
        send("progress", json!({"value": value, "max": max, "prompt_id": n.prompt, "node": node}), n.client.as_deref());
        send_state(n);
    }
    Ok(())
}

/// Marks a node as being evaluated for as long as the guard lives: it finishes with `done`, and
/// one dropped without it (an error) is not counted as executed.
pub struct Executing {
    id: String,
    finished: bool,
}

impl Executing {
    pub fn new(id: &str, class: &str) -> Executing {
        let mut now = NOW.lock().unwrap();
        if let Some(n) = now.as_mut() {
            n.stack.push((id.to_string(), class.to_string()));
            n.last = Some((id.to_string(), class.to_string()));
            n.value = 0;
            n.max = 0;
            set_state(n, id, |e| *e = (id.to_string(), 0.0, 1.0, "running"));
            send("executing", json!({"node": id, "display_node": id, "prompt_id": n.prompt}), n.client.as_deref());
            send_state(n);
        }
        Executing { id: id.to_string(), finished: false }
    }

    pub fn done(mut self) {
        self.finished = true;
    }
}

impl Drop for Executing {
    fn drop(&mut self) {
        if let Some(n) = NOW.lock().unwrap().as_mut() {
            n.stack.pop();
            if self.finished {
                set_state(n, &self.id, |e| {
                    e.1 = e.2;
                    e.3 = "finished";
                });
                n.executed.push(self.id.clone());
                send_state(n);
            }
        }
    }
}

/// The node entered last, (id, class): where a failure happened, for an execution_error
pub fn last_node() -> Option<(String, String)> {
    NOW.lock().unwrap().as_ref().and_then(|n| n.last.clone())
}

/// {"prompt_id", "node", "class_type", "value", "max"} while a prompt runs, else null.
pub fn json() -> Value {
    match &*NOW.lock().unwrap() {
        Some(n) => {
            let (node, class) = n.stack.last().cloned().unwrap_or_default();
            json!({"prompt_id": n.prompt, "node": node, "class_type": class, "value": n.value, "max": n.max})
        }
        None => Value::Null,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_interrupt_waits_for_its_prompt() {
        // nothing runs: a targeted interrupt is not kept, a global one is cleared by the next start
        assert!(!interrupt(Some("a")));
        assert!(check().is_ok());
        interrupt(None);
        assert!(check().is_err());
        start("a", None);
        assert!(check().is_ok());
        assert!(!interrupt(Some("b")));
        assert!(check().is_ok());
        assert!(interrupt(Some("a")));
        let e = step(1, 2).unwrap_err();
        assert!(e.chain().any(|c| c.is::<Interrupted>()));
        let messages = finished(Outcome::Interrupted("3", "KSampler"));
        let events: Vec<&str> = messages.iter().map(|m| m[0].as_str().unwrap()).collect();
        assert_eq!(events, ["execution_start", "execution_interrupted"]);
        assert!(messages.iter().all(|m| m[1]["timestamp"].is_u64()));
        done();
        // stopped while it waited to start: it stops at its first node; another prompt is not
        interrupt_at_start("b");
        start("b", None);
        assert!(check().is_err());
        start("c", None);
        assert!(check().is_ok());
        done();
    }
}
