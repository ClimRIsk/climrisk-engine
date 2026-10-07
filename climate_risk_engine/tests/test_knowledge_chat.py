"""The learning agent: knowledge index, cited answers, optional local model, learn-on-demand, report generation in chat."""
import http.server
import json
import threading
import time

import pytest

PAGE = """<html><head><title>Storm surge primer</title></head><body><nav>menu menu</nav><article>
<p>A storm surge is a rise of sea water above the normal tide, pushed onto the coast by the wind and low pressure of a tropical cyclone. It is the main cause of death and damage on low flat coasts such as the Bengal delta.</p>
<p>River flooding is different: it comes from rain falling inland and flowing downstream, and it can arrive days after the rain, long after the storm has passed. A site near an estuary can see both together.</p>
<p>Engineers separate the two because the defences differ: sea walls and raised plant against surge, drainage capacity and upstream storage against river floods. Both are screened with terrain, water level and building footprint data before design.</p>
</article><footer>footer</footer></body></html>"""


@pytest.fixture()
def kn(monkeypatch, tmp_path):
    monkeypatch.setenv("CRI_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("CRI_LLM", "0")
    from cri.resident import knowledge as K
    return K


@pytest.fixture()
def site():
    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200); self.send_header("Content-Type", "text/html"); self.end_headers(); self.wfile.write(PAGE.encode())
        def log_message(self, *a): pass
    srv = http.server.HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}/surge"
    srv.shutdown()


def test_chunker_keeps_headings_and_drops_boilerplate(kn):
    text = "== Causes ==\n" + ("Wind pushes water onshore and low pressure lets the sea rise. " * 12) + "\n\n== References ==\nsomething"
    ps = kn._chunk(kn._CUT.sub("", "\n" + text))
    assert ps and ps[0].startswith("Causes.") and "References" not in " ".join(ps) and "==" not in ps[0]
    t, body = kn._strip_html(PAGE)
    assert t == "Storm surge primer" and "menu" not in body and "footer" not in body and "storm surge is a rise" in body


def test_learn_a_page_then_answer_with_the_source(kn, site):
    from cri.resident import answer as AN
    r = kn.learn_url(site)
    assert r["ok"] and r["passages"] >= 1
    assert kn.stats()["passages"] >= 1
    out = AN.ask("what is the difference between a storm surge and river flooding?", allow_learn=False)
    assert out["ok"] and "[1]" in out["reply"] and out["mode"] == "extractive"
    assert out["sources"][0]["url"] == site and out["sources"][0]["title"] == "Storm surge primer"
    assert "rain" in out["reply"].lower() or "wind" in out["reply"].lower()


def test_nothing_relevant_is_said_not_invented(kn):
    from cri.resident import answer as AN
    out = AN.ask("what is the airspeed velocity of an unladen swallow?", allow_learn=False)
    assert not out["ok"] and out["sources"] == [] and "will not guess" in out["reply"]


def test_title_match_and_stemming_rank_the_right_page(kn, monkeypatch):
    kn._store("https://x/storm", "Storm surge (Wikipedia)", "wikipedia", ["A storm surge is a coastal flood phenomenon of rising water driven by tropical cyclones and low pressure systems over the sea " * 2])
    kn._store("https://x/katrina", "Hurricane Katrina (Wikipedia)", "wikipedia", ["The surge in New Orleans flooded neighbourhoods near the river after the levees failed and the storm passed over the city again " * 2])
    top = kn.search("what is a storm surge?", 3)
    assert top[0]["title"].startswith("Storm surge")
    assert kn.has("flooding", "the flood rose")
    assert kn._terms("Saffir-Simpson scale work") == ["saffir", "simpson", "scale"]


def test_local_model_path_requires_citations(kn, site, monkeypatch):
    """A fake Ollama: a cited answer is used; an uncited one is rejected and the extractive answer takes over."""
    from cri.resident import answer as AN
    kn.learn_url(site)
    state = {"reply": "A surge is wind-driven sea water; a river flood is inland rain [1]."}

    class O(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200); self.end_headers(); self.wfile.write(json.dumps({"models": [{"name": "llama3.2:3b"}]}).encode())
        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0)); self.rfile.read(n)
            self.send_response(200); self.end_headers(); self.wfile.write(json.dumps({"message": {"content": state["reply"]}}).encode())
        def log_message(self, *a): pass
    srv = http.server.HTTPServer(("127.0.0.1", 0), O)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    monkeypatch.setenv("CRI_LLM", "1"); monkeypatch.setattr(AN, "_OLLAMA", f"http://127.0.0.1:{srv.server_port}"); AN._LLM_CACHE.update(t=0.0, v=None)
    try:
        assert AN.llm_status(force=True) == {"available": True, "model": "llama3.2:3b", "why": "ok"}
        a = AN.ask("difference between storm surge and river flooding", allow_learn=False)
        assert a["ok"] and a["mode"].startswith("language model") and a["reply"] == state["reply"]
        state["reply"] = "It is obviously different, trust me."                    # no citation: must not be shown as sourced
        b = AN.ask("difference between storm surge and river flooding", allow_learn=False)
        assert b["ok"] and b["mode"] == "extractive" and "[" in b["reply"]
    finally:
        srv.shutdown(); AN._LLM_CACHE.update(t=0.0, v=None)


def test_agent_answers_open_questions_and_learns_commands(kn, site, monkeypatch):
    from cri.resident import agent as AG
    ag = AG.Agent(twin_builder=lambda **k: {}, analyser=lambda t: {}, scanner=lambda **k: {}, lab=lambda *a: {})
    monkeypatch.setattr(kn, "learn_seed", lambda *a, **k: {"new_passages": 0})
    r = ag.handle("s", f"learn from {site}")
    assert r["job"]
    for _ in range(50):
        j = AG.job_status(r["job"])
        if j["status"] != "running":
            break
        time.sleep(0.1)
    assert j["status"] == "done" and "passages" in j["reply"]
    q = ag.handle("s", "why is a storm surge different from river flooding?")
    assert q["job"]
    for _ in range(80):
        j = AG.job_status(q["job"])
        if j["status"] != "running":
            break
        time.sleep(0.1)
    assert j["status"] == "done" and "[1]" in j["reply"] and any(b["type"] == "sources" for b in j["blocks"])
    kb = ag.handle("s", "what do you know?")
    assert "passages" in kb["reply"] and kb["intent"] == "knowledge"
    assert "Forgot" in ag.handle("s", f"forget {site}")["reply"]


def test_chat_writes_the_report_files(tmp_path, monkeypatch):
    monkeypatch.setenv("CRI_DATA_DIR", str(tmp_path))
    from cri import story as S, story_reports as SR
    from cri.resident import agent as AG

    def fake(rtype, payload, template=None, brand=None):
        st = S.Story("Physical risk", "What the water does to Test Plant", "A short standfirst.")
        st.section("loud", "The loud risk").p("Water rose [[S:a]].").facts([("1.2 m", "peak depth")]).source("a", "Test source")
        return st.d
    monkeypatch.setattr(SR, "build", fake)
    ag = AG.Agent(twin_builder=lambda **k: {}, analyser=lambda t: {}, scanner=lambda **k: {}, lab=lambda *a: {})
    r = ag.handle("s", "write a physical risk report for 22.2645, 91.805")
    assert r["job"] and "Writing the physical report" in r["reply"]
    for _ in range(100):
        j = AG.job_status(r["job"])
        if j["status"] != "running":
            break
        time.sleep(0.1)
    assert j["status"] == "done", j
    files = next(b for b in j["blocks"] if b["type"] == "files")["items"]
    assert files[0]["url"].startswith("/resident/file/") and files[0]["url"].endswith(".html")
    assert "What the water does to Test Plant" in j["reply"] and any(b["type"] == "table" for b in j["blocks"])
    assert (AG.report_dir() / files[0]["url"].rsplit("/", 1)[1]).read_text(encoding="utf-8").count("Test Plant") >= 1


def test_file_route_serves_only_report_names(tmp_path, monkeypatch):
    monkeypatch.setenv("CRI_DATA_DIR", str(tmp_path))
    from fastapi.testclient import TestClient
    from cri.api.main import app
    from cri.resident import agent as AG
    (AG.report_dir() / "demo_1.html").write_text("<p>hi</p>", encoding="utf-8")
    c = TestClient(app)
    assert c.get("/resident/file/demo_1.html").status_code == 200
    assert c.get("/resident/file/..%2Fresident.db").status_code == 404
    assert c.get("/resident/file/resident.db").status_code == 404
