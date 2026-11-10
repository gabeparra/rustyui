//! ComfyUI's jobs API, as server.py's /api/jobs routes answer it over comfy_execution/jobs.py: the
//! queue and the history as one list of jobs, which the editor's queue and history panels read,
//! one job's detail (its outputs and the workflow to load again), and cancelling a job whatever
//! state it is in.

use super::*;

/// The jobs API's statuses (JobStatus.ALL)
const STATUSES: [&str; 5] = ["pending", "in_progress", "completed", "failed", "cancelled"];
/// Media the editor can preview, then the file kinds it previews by extension
const PREVIEWABLE: [&str; 5] = ["images", "video", "audio", "3d", "text"];
const THREE_D: [&str; 5] = [".obj", ".fbx", ".gltf", ".glb", ".usdz"];
const TEXT: [&str; 3] = [".txt", ".md", ".json"];
/// How much of a text output a job's preview carries
const TEXT_PREVIEW_MAX: usize = 1024;

/// jobs.py's validate_job_id: a UUID in canonical lowercase hyphenated form
pub(super) fn valid_job_id(id: &str) -> bool {
    id.len() == 36
        && id
            .char_indices()
            .all(|(i, c)| if [8, 13, 18, 23].contains(&i) { c == '-' } else { c.is_ascii_digit() || ('a'..='f').contains(&c) })
}

/// comfy_api's prune_dict: the object without its null fields
fn prune(v: Value) -> Value {
    match v {
        Value::Object(m) => Value::Object(m.into_iter().filter(|(_, v)| !v.is_null()).collect()),
        v => v,
    }
}

fn has_ext(name: &str, exts: &[&str]) -> bool {
    let low = name.to_lowercase();
    exts.iter().any(|e| low.ends_with(e))
}

/// Python's str() of a value as the history holds it
fn py_str(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        v => v.to_string(),
    }
}

/// normalize_output_item: a dict as it is, a 3D file's name as a file entry, anything else None
fn output_item(item: &Value) -> Option<Value> {
    match item {
        Value::String(s) if has_ext(s, &THREE_D) => Some(json!({"filename": s, "type": "output", "subfolder": "", "mediaType": "3d"})),
        Value::Object(_) => Some(item.clone()),
        _ => None,
    }
}

/// normalize_outputs: 3D file names made file entries, null items dropped
fn normalize_outputs(outputs: &Value) -> Value {
    let Some(nodes) = outputs.as_object() else { return outputs.clone() };
    let node = |o: &Value| -> Value {
        let Some(media) = o.as_object() else { return o.clone() };
        Value::Object(
            media
                .iter()
                .map(|(kind, items)| match items.as_array() {
                    Some(list) if kind != "animated" => (
                        kind.clone(),
                        Value::Array(list.iter().filter(|i| !i.is_null()).map(|i| output_item(i).unwrap_or_else(|| i.clone())).collect()),
                    ),
                    _ => (kind.clone(), items.clone()),
                })
                .collect(),
        )
    };
    Value::Object(nodes.iter().map(|(id, o)| (id.clone(), node(o))).collect())
}

fn is_previewable(media: &str, item: &Value) -> bool {
    if PREVIEWABLE.contains(&media) {
        return true;
    }
    let format = item["format"].as_str().unwrap_or("");
    if format.starts_with("video/") || format.starts_with("audio/") {
        return true;
    }
    let name = item["filename"].as_str().unwrap_or("");
    has_ext(name, &THREE_D) || has_ext(name, &TEXT)
}

fn is_text_preview(media: &str, item: &Value) -> bool {
    media == "text" || has_ext(item["filename"].as_str().unwrap_or(""), &TEXT)
}

fn text_preview(text: &str) -> Value {
    if text.chars().count() <= TEXT_PREVIEW_MAX {
        json!({"content": text})
    } else {
        json!({"content": text.chars().take(TEXT_PREVIEW_MAX).collect::<String>(), "truncated": true})
    }
}

/// get_outputs_summary: how many outputs a job has, and the one its row previews (a saved
/// picture or video first, then any other visual, then a saved text file, then raw text)
fn outputs_summary(outputs: &Value) -> (usize, Value) {
    let mut count = 0;
    let (mut preview, mut fallback, mut text_file, mut text) = (Value::Null, Value::Null, Value::Null, Value::Null);
    for (node, media) in outputs.as_object().into_iter().flatten() {
        for (kind, items) in media.as_object().into_iter().flatten() {
            let Some(items) = items.as_array() else { continue };
            if kind == "animated" {
                continue;
            }
            for item in items {
                let item = if item.is_object() {
                    item.clone()
                } else {
                    match output_item(item) {
                        Some(i) => i,
                        None => {
                            // text content: only a preview, not an output
                            if kind == "text" && preview.is_null() && text.is_null() {
                                let value = match item {
                                    Value::Array(a) => a.first().map(py_str).unwrap_or_default(),
                                    v => py_str(v),
                                };
                                let mut enriched = text_preview(&value);
                                enriched["nodeId"] = json!(node);
                                enriched["mediaType"] = json!(kind);
                                text = enriched;
                            }
                            continue;
                        }
                    }
                };
                count += 1;
                if !preview.is_null() || !is_previewable(kind, &item) {
                    continue;
                }
                let mut enriched = item.clone();
                enriched["nodeId"] = json!(node);
                if item.get("mediaType").is_none() {
                    enriched["mediaType"] = json!(kind);
                }
                if is_text_preview(kind, &item) {
                    if text_file.is_null() {
                        text_file = enriched;
                    }
                } else if item["type"] == "output" {
                    preview = enriched;
                } else if fallback.is_null() {
                    fallback = enriched;
                }
            }
        }
    }
    (count, [preview, fallback, text_file, text].into_iter().find(|v| !v.is_null()).unwrap_or(Value::Null))
}

/// count_previewable_outputs: the outputs the expanded view shows
fn previewable_count(outputs: &Value) -> usize {
    let mut n = 0;
    for media in outputs.as_object().into_iter().flatten().map(|(_, m)| m) {
        for (kind, items) in media.as_object().into_iter().flatten() {
            if kind == "animated" {
                continue;
            }
            for item in items.as_array().into_iter().flatten() {
                let item = if item.is_object() { Some(item.clone()) } else { output_item(item) };
                if item.is_some_and(|i| is_previewable(kind, &i)) {
                    n += 1;
                }
            }
        }
    }
    n
}

/// normalize_queue_item: a queued or running prompt ([number, id, prompt, extra_data, outputs]) as a job
pub(super) fn queue_job(item: &Value, status: &str) -> Value {
    let extra = &item[3];
    prune(json!({
        "id": item[1], "status": status, "priority": item[0], "create_time": extra["create_time"],
        "outputs_count": 0, "previewable_outputs_count": 0, "workflow_id": extra["extra_pnginfo"]["workflow"]["id"],
    }))
}

/// normalize_history_item: a finished prompt as a job; with its outputs, status and workflow for
/// the detail route
pub(super) fn history_job(id: &str, entry: &Value, include_outputs: bool) -> Value {
    let prompt = &entry["prompt"];
    let extra = &prompt[3];
    let status_info = &entry["status"];
    let outputs = if entry["outputs"].is_object() { entry["outputs"].clone() } else { json!({}) };
    let (count, preview) = outputs_summary(&outputs);
    let (mut error, mut start, mut end, mut interrupted) = (Value::Null, Value::Null, Value::Null, false);
    for m in status_info["messages"].as_array().into_iter().flatten() {
        let (Some(event), Some(data)) = (m.get(0).and_then(Value::as_str), m.get(1)) else { continue };
        if !data.is_object() {
            continue;
        }
        match event {
            "execution_start" => start = data["timestamp"].clone(),
            "execution_success" | "execution_error" | "execution_interrupted" => {
                end = data["timestamp"].clone();
                if event == "execution_error" {
                    error = data.clone();
                } else if event == "execution_interrupted" {
                    interrupted = true;
                }
            }
            _ => {}
        }
    }
    let status = match status_info["status_str"].as_str() {
        Some("error") if interrupted => "cancelled",
        Some("error") => "failed",
        _ => "completed",
    };
    let mut job = prune(json!({
        "id": id, "status": status, "priority": prompt[0], "create_time": extra["create_time"],
        "execution_start_time": start, "execution_end_time": end, "execution_error": error,
        "outputs_count": count, "previewable_outputs_count": previewable_count(&outputs), "preview_output": preview,
        "workflow_id": extra["extra_pnginfo"]["workflow"]["id"],
    }));
    if include_outputs {
        job["outputs"] = normalize_outputs(&outputs);
        job["execution_status"] = status_info.clone();
        job["workflow"] = json!({"prompt": prompt[2], "extra_data": extra});
    }
    job
}

/// apply_sorting (a stable sort, as Python's)
fn sort_jobs(jobs: &mut [Value], by: &str, desc: bool) {
    let key = |j: &Value| -> f64 {
        if by == "execution_duration" {
            let (s, e) = (j["execution_start_time"].as_f64().unwrap_or(0.0), j["execution_end_time"].as_f64().unwrap_or(0.0));
            if s != 0.0 && e != 0.0 {
                e - s
            } else {
                0.0
            }
        } else {
            j["create_time"].as_f64().unwrap_or(0.0)
        }
    };
    jobs.sort_by(|a, b| {
        let o = key(a).partial_cmp(&key(b)).unwrap_or(std::cmp::Ordering::Equal);
        if desc {
            o.reverse()
        } else {
            o
        }
    });
}

/// The queue as ComfyUI's routes see it: the running prompt, the pending ones, and the history
/// in the order it was written
fn snapshot(srv: &Server) -> (Vec<Value>, Vec<Value>, Vec<(String, Value)>) {
    let s = srv.st.lock().unwrap();
    let running = s.running.iter().map(Job::item).collect();
    let queued = s.queue.iter().map(Job::item).collect();
    let history = s.order.iter().filter_map(|id| Some((id.clone(), s.history.get(id)?.clone()))).collect();
    (running, queued, history)
}

/// A query parameter, or None when it is not given at all
fn param(q: &str, key: &str) -> Option<String> {
    q.split('&').find_map(|kv| if kv == key { Some(String::new()) } else { kv.strip_prefix(&format!("{key}=")).map(percent_decode) })
}

/// GET /jobs: every job, filtered, sorted and paged as get_jobs and get_all_jobs do
fn list_jobs(srv: &Server, q: &str) -> (u16, Value) {
    let bad = |msg: String| (400, json!({"error": msg}));
    let statuses: Vec<String> = match param(q, "status") {
        Some(s) if !s.is_empty() => {
            let wanted: Vec<String> = s.split(',').map(|x| x.trim().to_lowercase()).filter(|x| !x.is_empty()).collect();
            let invalid: Vec<&str> = wanted.iter().map(String::as_str).filter(|x| !STATUSES.contains(x)).collect();
            if !invalid.is_empty() {
                return bad(format!("Invalid status value(s): {}. Valid values: {}", invalid.join(", "), STATUSES.join(", ")));
            }
            wanted
        }
        _ => STATUSES.iter().map(|s| s.to_string()).collect(),
    };
    let workflow_id = param(q, "workflow_id").filter(|w| !w.is_empty());
    let sort_by = param(q, "sort_by").unwrap_or_else(|| "created_at".into()).to_lowercase();
    let sort_order = param(q, "sort_order").unwrap_or_else(|| "desc".into()).to_lowercase();
    if sort_by != "created_at" && sort_by != "execution_duration" {
        return bad("sort_by must be 'created_at' or 'execution_duration'".into());
    }
    if sort_order != "asc" && sort_order != "desc" {
        return bad("sort_order must be 'asc' or 'desc'".into());
    }
    let limit = match param(q, "limit") {
        None => None,
        Some(l) => match l.trim().parse::<i64>() {
            Ok(n) if n > 0 => Some(n as usize),
            Ok(_) => return bad("limit must be a positive integer".into()),
            Err(_) => return bad("limit must be an integer".into()),
        },
    };
    let offset = match param(q, "offset") {
        None => 0,
        Some(o) => match o.trim().parse::<i64>() {
            Ok(n) => n.max(0) as usize,
            Err(_) => return bad("offset must be an integer".into()),
        },
    };
    let (running, queued, history) = snapshot(srv);
    let wants = |s: &str| statuses.iter().any(|x| x == s);
    let mut jobs = vec![];
    if wants("in_progress") {
        jobs.extend(running.iter().map(|i| queue_job(i, "in_progress")));
    }
    if wants("pending") {
        jobs.extend(queued.iter().map(|i| queue_job(i, "pending")));
    }
    if ["completed", "failed", "cancelled"].iter().any(|s| wants(s)) {
        for (id, entry) in &history {
            let job = history_job(id, entry, false);
            if job["status"].as_str().is_some_and(wants) {
                jobs.push(job);
            }
        }
    }
    if let Some(w) = &workflow_id {
        jobs.retain(|j| j["workflow_id"].as_str() == Some(w.as_str()));
    }
    sort_jobs(&mut jobs, &sort_by, sort_order == "desc");
    let total = jobs.len();
    let page: Vec<Value> = jobs.into_iter().skip(offset).take(limit.unwrap_or(usize::MAX)).collect();
    let has_more = offset + page.len() < total;
    (200, json!({"jobs": page, "pagination": {"offset": offset, "limit": limit, "total": total, "has_more": has_more}}))
}

/// get_job: a finished job with its outputs, else the running or pending one
fn job_detail(srv: &Server, id: &str) -> Option<Value> {
    let (running, queued, history) = snapshot(srv);
    if let Some((_, entry)) = history.iter().find(|(k, _)| k == id) {
        return Some(history_job(id, entry, true));
    }
    if let Some(item) = running.iter().find(|i| i[1] == id) {
        return Some(queue_job(item, "in_progress"));
    }
    queued.iter().find(|i| i[1] == id).map(|i| queue_job(i, "pending"))
}

/// cancel_job: a running job is interrupted, a pending one taken off the queue, a finished or
/// unknown one left alone. Says whether it acted.
pub(super) fn cancel(srv: &Server, id: &str) -> bool {
    let mut s = srv.st.lock().unwrap();
    if s.running.as_ref().is_some_and(|j| j.id == id) {
        stop_running(id);
        return true;
    }
    if let Some(at) = s.queue.iter().position(|j| j.id == id) {
        s.queue.remove(at);
        ui_update(srv, &mut s, id, json!({"status": "error", "error": "taken off the queue", "finished": now()}));
        emit_status(s.queue.len() + usize::from(s.running.is_some()));
        return true;
    }
    false
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
        ("GET", "/jobs") => {
            let (code, v) = list_jobs(srv, q);
            json_reply(req, code, v)
        }
        ("POST", "/jobs/cancel") => {
            let parsed: Option<Value> = serde_json::from_slice(body).ok();
            let Some(parsed) = parsed else { return Ok(json_reply(req, 400, json!({"error": "Request body must be valid JSON"}))) };
            let Some(ids) = parsed.get("job_ids").and_then(Value::as_array) else {
                return Ok(json_reply(req, 400, json!({"error": "job_ids must be a list"})));
            };
            let invalid: Vec<String> = ids.iter().filter(|v| !v.as_str().is_some_and(valid_job_id)).map(py_str).collect();
            if !invalid.is_empty() {
                return Ok(json_reply(req, 400, json!({"error": "job_ids contains invalid id(s)", "invalid_ids": invalid})));
            }
            let mut cancelled = false;
            for id in ids.iter().filter_map(Value::as_str) {
                cancelled |= cancel(srv, id);
            }
            json_reply(req, 200, json!({"cancelled": cancelled}))
        }
        ("POST", p) if p.starts_with("/jobs/") && p.ends_with("/cancel") => {
            let id = percent_decode(&p["/jobs/".len()..p.len() - "/cancel".len()]);
            if id.is_empty() || id.contains('/') {
                return Ok(json_reply(req, 404, json!({"error": format!("no route {method} {p}")})));
            }
            json_reply(req, 200, json!({"cancelled": cancel(srv, &id)}))
        }
        ("GET", p) if p.starts_with("/jobs/") && !p["/jobs/".len()..].contains('/') => {
            let id = percent_decode(&p["/jobs/".len()..]);
            match job_detail(srv, &id) {
                Some(job) => json_reply(req, 200, job),
                None => json_reply(req, 404, json!({"error": "Job not found"})),
            }
        }
        _ => return Err(req),
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(status: &str, messages: Value, outputs: Value) -> Value {
        json!({
            "prompt": [3, "id", {"1": {}}, {"create_time": 1000, "extra_pnginfo": {"workflow": {"id": "wf"}}}, ["9"]],
            "outputs": outputs,
            "status": {"status_str": status, "completed": status == "success", "messages": messages},
        })
    }

    #[test]
    fn job_ids_are_canonical_uuids() {
        assert!(valid_job_id("0b9f1c2e-6a7d-4e8f-9a0b-1c2d3e4f5a6b"));
        assert!(!valid_job_id("0B9F1C2E-6A7D-4E8F-9A0B-1C2D3E4F5A6B"));
        assert!(!valid_job_id("0b9f1c2e6a7d4e8f9a0b1c2d3e4f5a6b"));
        assert!(!valid_job_id("{0b9f1c2e-6a7d-4e8f-9a0b-1c2d3e4f5a6b}"));
    }

    #[test]
    fn history_entries_become_jobs() {
        let pic = json!({"9": {"images": [{"filename": "a.png", "subfolder": "", "type": "output"}]}});
        let ok = history_job(
            "id",
            &entry(
                "success",
                json!([["execution_start", {"prompt_id": "id", "timestamp": 5}],
                                                              ["execution_success", {"prompt_id": "id", "timestamp": 9}]]),
                pic.clone(),
            ),
            false,
        );
        assert_eq!(ok["status"], "completed");
        assert_eq!((ok["execution_start_time"].as_u64(), ok["execution_end_time"].as_u64()), (Some(5), Some(9)));
        assert_eq!((ok["outputs_count"].as_u64(), ok["previewable_outputs_count"].as_u64()), (Some(1), Some(1)));
        assert_eq!(
            ok["preview_output"],
            json!({"filename": "a.png", "subfolder": "", "type": "output", "nodeId": "9", "mediaType": "images"})
        );
        assert_eq!((ok["priority"].as_u64(), ok["create_time"].as_u64(), ok["workflow_id"].as_str()), (Some(3), Some(1000), Some("wf")));
        assert!(ok.get("execution_error").is_none() && ok.get("outputs").is_none());
        let stopped = history_job("id", &entry("error", json!([["execution_interrupted", {"timestamp": 7}]]), json!({})), true);
        assert_eq!(stopped["status"], "cancelled");
        assert_eq!(stopped["workflow"]["prompt"], json!({"1": {}}));
        let failed = history_job("id", &entry("error", json!([["execution_error", {"node_id": "3", "timestamp": 7}]]), json!({})), false);
        assert_eq!((failed["status"].as_str(), failed["execution_error"]["node_id"].as_str()), (Some("failed"), Some("3")));
    }

    #[test]
    fn previews_prefer_saved_media_over_text() {
        let outputs = json!({"1": {"text": ["hello"]}, "2": {"images": [{"filename": "t.png", "type": "temp"}]},
                             "3": {"images": [{"filename": "o.png", "type": "output"}], "animated": [true]}});
        let (count, preview) = outputs_summary(&outputs);
        assert_eq!(count, 2);
        assert_eq!(preview["filename"], "o.png");
        let (_, text) = outputs_summary(&json!({"1": {"text": ["hello"]}}));
        assert_eq!(text, json!({"content": "hello", "nodeId": "1", "mediaType": "text"}));
    }

    #[test]
    fn queued_prompts_become_jobs() {
        let item = json!([-2, "id", {}, {"create_time": 5, "client_id": "c"}, ["9"]]);
        assert_eq!(
            queue_job(&item, "pending"),
            json!({"id": "id", "status": "pending", "priority": -2, "create_time": 5,
                                                         "outputs_count": 0, "previewable_outputs_count": 0})
        );
        let mut jobs =
            vec![json!({"id": "a", "create_time": 1}), json!({"id": "b", "create_time": 3}), json!({"id": "c", "create_time": 1})];
        sort_jobs(&mut jobs, "created_at", true);
        let ids: Vec<&str> = jobs.iter().map(|j| j["id"].as_str().unwrap()).collect();
        assert_eq!(ids, ["b", "a", "c"]);
    }
}
