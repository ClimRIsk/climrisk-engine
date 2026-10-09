"""A model call has a hard total deadline: a server that keeps trickling bytes cannot hold an answer for minutes."""
import http.server
import json
import threading
import time

from cri.resident import answer as AN


def _serve(handler):
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


class Trickle(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        self.send_response(200)
        self.send_header("Content-Length", "100000")
        self.end_headers()
        try:
            for _ in range(200):                      # one byte per 0.1 s: never silent for long, so a plain socket timeout never fires
                self.wfile.write(b" ")
                self.wfile.flush()
                time.sleep(0.1)
        except Exception:                             # noqa: BLE001
            pass

    def log_message(self, *a):
        pass


class Good(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.dumps({"message": {"content": "ok [1]"}, "load_duration": 2_000_000_000, "eval_duration": 3_000_000_000, "eval_count": 30, "total_duration": 6_000_000_000}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


def test_trickling_server_is_cut_off_at_the_deadline(monkeypatch):
    srv = _serve(Trickle)
    monkeypatch.setattr(AN, "_OLLAMA", f"http://127.0.0.1:{srv.server_address[1]}")
    t = time.time()
    try:
        AN._ollama_post("/api/chat", b"{}", 1.5)
        assert False, "should have timed out"
    except TimeoutError as e:
        assert "did not answer within" in str(e)
    assert time.time() - t < 4
    srv.shutdown()


def test_normal_call_returns_timing(monkeypatch):
    srv = _serve(Good)
    monkeypatch.setattr(AN, "_OLLAMA", f"http://127.0.0.1:{srv.server_address[1]}")
    r = AN._ollama_post("/api/chat", b"{}", 5)
    tm = AN._timing(r)
    assert r["message"]["content"] == "ok [1]" and tm["load"] == 2.0 and tm["generate"] == 3.0 and tm["output_tokens"] == 30
    srv.shutdown()


def test_case_llm_falls_back_when_the_model_stalls(monkeypatch):
    srv = _serve(Trickle)
    monkeypatch.setattr(AN, "_OLLAMA", f"http://127.0.0.1:{srv.server_address[1]}")
    monkeypatch.setattr(AN, "LLM_DEADLINE_S", 1.0)
    from cri.resident import casebase as CB
    ps = [{"id": 1, "text": "x", "title": "t", "url": "u"}]
    monkeypatch.setattr(CB, "system_prompt", lambda ps: "sys")
    t = time.time()
    assert AN._case_llm("q", ps, "m", None) is None and "within" in AN.LAST_LLM["why"]
    assert time.time() - t < 4
    srv.shutdown()


def test_warm_unloads_other_resident_models(monkeypatch):
    calls = []

    class Ps(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            body = json.dumps({"models": [{"name": "climrisk-llm-v5b:latest"}, {"name": "climrisk-llm-v2:latest"}]}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            calls.append(json.loads(self.rfile.read(n)))
            body = b"{}"
            self.send_response(200)
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    srv = _serve(Ps)
    monkeypatch.setattr(AN, "_OLLAMA", f"http://127.0.0.1:{srv.server_address[1]}")
    assert AN.unload_others("climrisk-llm-v5b:latest") == ["climrisk-llm-v2:latest"]
    assert calls == [{"model": "climrisk-llm-v2:latest", "prompt": "", "stream": False, "keep_alive": 0}]
    srv.shutdown()
