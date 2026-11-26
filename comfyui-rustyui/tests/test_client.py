"""client.py against a stand-in rustyui-backend. No ComfyUI, GPU or network needed:
python3 -m unittest discover -s comfyui-rustyui/tests"""

import json
import os
import sys
import tempfile
import threading
import unittest
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import client  # noqa: E402

PNG = b"\x89PNG\r\n\x1a\n-not-a-real-picture-"


class Fake(BaseHTTPRequestHandler):
    """Answers as rustyui does: /prompt, /history/<id>, /progress, /view, /queue, /interrupt."""

    def log_message(self, *a):
        pass

    def reply(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        st = self.server.state
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])) or b"{}")
        st["posts"].append((self.path, body))
        if self.path == "/prompt":
            if st["refuse"]:
                return self.reply(400, {"error": {"type": "prompt_outputs_failed_validation", "message": "Prompt outputs failed validation"},
                                        "node_errors": {"4": {"class_type": "FancyNode", "errors": [{"message": "node not ported yet", "details": "FancyNode"}]}}})
            return self.reply(200, {"prompt_id": "p1", "number": 0, "node_errors": {}})
        self.reply(200, b"", "text/plain")

    def do_GET(self):
        st = self.server.state
        path, _, query = self.path.partition("?")
        if path == "/history/p1":
            st["polls"] += 1
            return self.reply(200, {"p1": st["entry"]} if st["polls"] > st["after"] else {})
        if path == "/progress":
            return self.reply(200, {"executing": {"prompt_id": "p1", "node": "3", "class_type": "KSampler", "value": 4, "max": 8}})
        if path == "/view":
            q = urllib.parse.parse_qs(query)
            st["views"].append(q)
            return self.reply(200, PNG, "image/png")
        self.reply(404, {"error": "no route"})


def done(images, animated=False):
    out = {"images": images}
    if animated:
        out["animated"] = [True]
    return {"status": {"status_str": "success", "completed": True, "messages": []}, "outputs": {"9": out}}


class ClientTest(unittest.TestCase):
    def setUp(self):
        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), Fake)
        self.srv.state = {"posts": [], "views": [], "polls": 0, "after": 2, "refuse": False,
                          "entry": done([{"filename": "rustyui_00001_.png", "subfolder": "sub", "type": "output"}])}
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.srv.server_address[1]}"
        self.dir = tempfile.TemporaryDirectory()
        self.out = os.path.join(self.dir.name, "output")
        self.tmp = os.path.join(self.dir.name, "temp")

    def tearDown(self):
        self.srv.shutdown()
        self.srv.server_close()
        self.dir.cleanup()

    def test_a_graph_goes_there_and_its_pictures_come_back(self):
        pid = client.queue(self.url, {"9": {"class_type": "SaveImage"}}, {"workflow": {"nodes": []}})
        path, body = self.srv.state["posts"][0]
        self.assertEqual((pid, path), ("p1", "/prompt"))
        self.assertTrue(body["extra_data"]["rustyui_from_comfyui"])
        self.assertEqual(body["extra_data"]["extra_pnginfo"], {"workflow": {"nodes": []}})
        seen = []
        entry = client.wait(self.url, pid, 30, progress=lambda v, m: seen.append((v, m)), poll=0.01)
        self.assertIn((4, 8), seen)
        files, animated = client.outputs(entry)
        self.assertFalse(animated)
        shown = client.place(client.fetch(self.url, files[0]), files[0], self.out, self.tmp)
        self.assertEqual(shown, {"filename": "rustyui_00001_.png", "subfolder": "sub", "type": "output"})
        with open(os.path.join(self.out, "sub", "rustyui_00001_.png"), "rb") as f:
            self.assertEqual(f.read(), PNG)
        self.assertEqual(self.srv.state["views"][0]["subfolder"], ["sub"])

    def test_a_shared_output_folder_is_left_alone(self):
        os.makedirs(os.path.join(self.out, "sub"))
        with open(os.path.join(self.out, "sub", "a.png"), "wb") as f:
            f.write(PNG)
        item = {"filename": "a.png", "subfolder": "sub", "type": "output"}
        self.assertEqual(client.place(PNG, item, self.out, self.tmp)["type"], "output")
        self.assertFalse(os.path.exists(self.tmp))

    def test_another_file_of_that_name_is_not_overwritten(self):
        os.makedirs(self.out)
        with open(os.path.join(self.out, "a.png"), "wb") as f:
            f.write(b"someone else's")
        shown = client.place(PNG, {"filename": "a.png", "subfolder": ""}, self.out, self.tmp)
        self.assertEqual(shown, {"filename": "a.png", "subfolder": "rustyui", "type": "temp"})
        with open(os.path.join(self.out, "a.png"), "rb") as f:
            self.assertEqual(f.read(), b"someone else's")

    def test_names_that_leave_the_folder_are_refused(self):
        for sub, name in [("..", "a.png"), ("/etc", "a.png"), ("a/../..", "x.png"), ("", "../a.png"), ("", "")]:
            with self.assertRaises(client.RustyError, msg=(sub, name)):
                client.place(PNG, {"filename": name, "subfolder": sub}, self.out, self.tmp)

    def test_videos_keep_their_flag(self):
        files, animated = client.outputs(done([{"filename": "v_00001_.mp4", "subfolder": "", "type": "output"}], True))
        self.assertTrue(animated)
        self.assertEqual(files[0]["filename"], "v_00001_.mp4")

    def test_refusals_name_the_node(self):
        self.srv.state["refuse"] = True
        with self.assertRaises(client.RustyError) as e:
            client.queue(self.url, {}, None)
        self.assertIn("node 4 (FancyNode): node not ported yet: FancyNode", str(e.exception))

    def test_failures_carry_the_backends_message(self):
        entry = {"status": {"status_str": "error", "completed": False, "messages": [
            ["execution_start", {}], ["execution_error", {"node_id": "3", "node_type": "KSampler", "exception_message": "out of memory"}]]}}
        with self.assertRaises(client.RustyError) as e:
            client.outputs(entry)
        self.assertEqual(str(e.exception), "rustyui-backend failed at node 3 (KSampler): out of memory")

    def test_a_failure_before_any_node_says_so(self):
        entry = {"status": {"status_str": "error", "messages": [["execution_error", {"node_id": "", "node_type": "", "exception_message": "no CUDA driver\n"}]]}}
        with self.assertRaises(client.RustyError) as e:
            client.outputs(entry)
        self.assertEqual(str(e.exception), "rustyui-backend failed before running a node: no CUDA driver")

    def test_an_interrupt_cancels_it_there_too(self):
        self.srv.state["after"] = 10**9
        self.assertIsNone(client.wait(self.url, "p1", 30, interrupted=lambda: True, poll=0.01))
        self.assertEqual([p for p, _ in self.srv.state["posts"]], ["/queue", "/interrupt"])
        self.assertEqual(self.srv.state["posts"][0][1], {"delete": ["p1"]})

    def test_a_backend_that_is_not_running_says_how_to_start_it(self):
        with self.assertRaises(client.RustyError) as e:
            client.queue("http://127.0.0.1:9", {}, None)
        self.assertIn("rustyui-backend serve", str(e.exception))


if __name__ == "__main__":
    unittest.main()
