"""Talking to rustyui-backend: queue a graph, follow it to the end, fetch what it made and place it
where ComfyUI looks for outputs. Standard library only, so it is tested without ComfyUI
(tests/test_client.py)."""

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid


class RustyError(Exception):
    """Something rustyui-backend refused or reported, worded for the person who queued the graph."""


def _message(raw):
    """The error a rustyui (or ComfyUI) JSON reply carries, node errors included, or None."""
    try:
        body = json.loads(raw)
    except (ValueError, TypeError):
        return None
    if not isinstance(body, dict):
        return None
    lines = []
    error = body.get("error")
    if isinstance(error, dict):
        msg = error.get("message") or error.get("type")
        details = error.get("details")
        lines.append(f"{msg}: {details}" if details and details != msg else str(msg))
    elif error:
        lines.append(str(error))
    for node, info in (body.get("node_errors") or {}).items():
        for e in (info or {}).get("errors", []):
            text = e.get("message", "")
            if e.get("details") and e.get("details") != text:
                text += f": {e['details']}"
            lines.append(f"node {node} ({info.get('class_type', '?')}): {text}")
    return "\n".join(lines) or None


def _call(url, path, body=None, timeout=30):
    data = None if body is None else json.dumps(body).encode()
    headers = {"Content-Type": "application/json"} if data is not None else {}
    req = urllib.request.Request(url.rstrip("/") + path, data=data, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read()
    except urllib.error.HTTPError as e:
        raw = e.read()
        raise RustyError(_message(raw) or f"rustyui-backend answered HTTP {e.code} to {path}") from None
    except (urllib.error.URLError, OSError) as e:
        reason = getattr(e, "reason", e)
        raise RustyError(
            f"rustyui-backend is not answering at {url} ({reason}). Start it with "
            "`rustyui-backend serve --models <ComfyUI>/models --out <ComfyUI>/output`, or set the node's url."
        ) from None


def _json(url, path, body=None, timeout=30):
    raw = _call(url, path, body, timeout)
    return json.loads(raw) if raw else {}


def queue(url, graph, extra_pnginfo=None):
    """Sends an API graph; returns its prompt id. The mark tells the backend this comes from
    inside a ComfyUI job, so it does not wait for ComfyUI's queue (which holds that job)."""
    extra = {"rustyui_from_comfyui": True}
    if extra_pnginfo:
        extra["extra_pnginfo"] = extra_pnginfo
    body = {"prompt": graph, "client_id": f"comfyui-rustyui-{uuid.uuid4().hex[:12]}", "extra_data": extra}
    answer = _json(url, "/prompt", body)
    if answer.get("node_errors"):
        raise RustyError(_message(json.dumps(answer)) or "rustyui-backend refused the graph")
    pid = answer.get("prompt_id")
    if not pid:
        raise RustyError(f"rustyui-backend gave no prompt id: {answer}")
    return pid


def cancel(url, pid):
    """Takes the prompt off rustyui's queue, or stops it if it has started."""
    for path, body in (("/queue", {"delete": [pid]}), ("/interrupt", {"prompt_id": pid})):
        try:
            _call(url, path, body, timeout=5)
        except RustyError:
            pass


def wait(url, pid, timeout, interrupted=lambda: False, progress=lambda value, total: None, poll=0.5):
    """Follows the prompt until rustyui records it. Returns its history entry, or None when
    `interrupted()` turned true (the prompt is cancelled on rustyui's side too)."""
    deadline = time.monotonic() + timeout
    while True:
        entry = _json(url, f"/history/{urllib.parse.quote(pid)}").get(pid)
        if entry:
            return entry
        if interrupted():
            cancel(url, pid)
            return None
        if time.monotonic() > deadline:
            cancel(url, pid)
            raise RustyError(f"rustyui-backend did not finish within {timeout} s; the prompt was cancelled")
        try:
            now = _json(url, "/progress", timeout=5).get("executing") or {}
        except RustyError:
            now = {}
        if now.get("prompt_id") == pid and now.get("max"):
            progress(now.get("value", 0), now["max"])
        time.sleep(poll)


def outputs(entry):
    """The files of a finished history entry as ComfyUI lists them, and whether they are video.
    Raises RustyError with the backend's own message when the prompt failed."""
    status = entry.get("status") or {}
    if status.get("status_str") != "success":
        for name, data in status.get("messages") or []:
            if name == "execution_error":
                msg = str(data.get("exception_message") or "no message").strip()
                if data.get("node_id"):
                    raise RustyError(f"rustyui-backend failed at node {data['node_id']} ({data.get('node_type')}): {msg}")
                raise RustyError(f"rustyui-backend failed before running a node: {msg}")
        raise RustyError(f"rustyui-backend did not finish the prompt: {status.get('status_str') or 'no status'}")
    files, animated = [], False
    for out in (entry.get("outputs") or {}).values():
        files.extend(i for i in out.get("images", []) if isinstance(i, dict) and i.get("filename"))
        animated = animated or any(out.get("animated", []))
    return files, animated


def fetch(url, item):
    query = urllib.parse.urlencode(
        {"filename": item["filename"], "subfolder": item.get("subfolder", ""), "type": item.get("type", "output")}
    )
    return _call(url, f"/view?{query}", timeout=120)


def _inside(base, subfolder, filename):
    """base/subfolder/filename, or None if that would leave base"""
    if not filename or "/" in filename or "\\" in filename or filename in (".", ".."):
        return None
    parts = [p for p in subfolder.replace("\\", "/").split("/") if p not in ("", ".")]
    if subfolder.startswith(("/", "\\")) or ".." in parts:
        return None
    return os.path.join(base, *parts, filename)


def place(data, item, output_dir, temp_dir):
    """Puts a file rustyui made where ComfyUI's /view finds it; returns the entry for ComfyUI's UI.

    When rustyui writes into this ComfyUI's output folder (the usual setup), the file is already
    there and is left as it is. Otherwise it is written there under the same name, unless another
    file has that name, in which case it goes to the temp folder."""
    sub = item.get("subfolder", "")
    target = _inside(output_dir, sub, item["filename"])
    if target is None:
        raise RustyError(f"rustyui-backend named a file outside its folder: {sub!r} / {item['filename']!r}")
    if os.path.isfile(target):
        with open(target, "rb") as f:
            if f.read() == data:
                return {"filename": item["filename"], "subfolder": sub, "type": "output"}
        sub = os.path.join("rustyui", sub) if sub else "rustyui"
        target, kind = _inside(temp_dir, sub, item["filename"]), "temp"
    else:
        kind = "output"
    os.makedirs(os.path.dirname(target), exist_ok=True)
    with open(target, "wb") as f:
        f.write(data)
    return {"filename": item["filename"], "subfolder": sub, "type": kind}
