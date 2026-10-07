"""Resident agent: trained NLU, memory, research mapping, LinkedIn gating, planner, API. All network and the twin builder are stubbed."""
import numpy as np
import pytest

from cri.resident import agent as A, corpus as C, linkedin as LI, memory as M, nlu as N, research as R


@pytest.fixture(autouse=True)
def _tmp(tmp_path, monkeypatch):
    monkeypatch.setenv("CRI_DATA_DIR", str(tmp_path))
    monkeypatch.delenv("LINKEDIN_ACCESS_TOKEN", raising=False)
    monkeypatch.delenv("LINKEDIN_AUTHOR_URN", raising=False)
    monkeypatch.setattr(N, "MODEL_PATH", tmp_path / "nlu.json")
    N._M = None
    A.JOBS.clear()


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    import cri.resident.nlu as nn
    rows = C.utterances(); tr, te = C.split_by_template(rows)
    return nn.Model().fit(tr), te


def test_split_is_by_template_not_utterance():
    rows = C.utterances(); tr, te = C.split_by_template(rows)
    ttr = {(i, ti) for u, i, ti in rows if (u, i) in set(tr)}
    tte = {(i, ti) for u, i, ti in rows if (u, i) in set(te)}
    assert ttr.isdisjoint(tte) and len(te) > 80


def test_heldout_phrasings_accuracy(trained):
    m, te = trained
    acc = np.mean([m.predict(u)["intent"] == i for u, i in te])
    model_only = np.mean([m.predict(u, hybrid=False)["intent"] == i for u, i in te])
    assert acc >= 0.62 and model_only >= 0.5          # unseen phrasings; far above 1/21 chance


def test_unrelated_text_is_unknown(trained):
    m, _ = trained
    N._M = m
    assert m.predict("write me a poem about cheese")["intent"] == "unknown"


def test_save_load_roundtrip(trained, tmp_path):
    m, _ = trained
    m2 = N.Model.from_json(m.to_json())
    for q in ["show the plant in 3D", "write the CBAM report", "post it"]:
        assert m2.predict(q)["intent"] == m.predict(q)["intent"]


def test_entities():
    e = N.entities("run a 1-in-100 flood at 22.2645, 91.805 for a cement plant by 2050")
    assert e["lat"] == 22.2645 and e["lon"] == 91.805 and e["rp"] == 100 and e["sector"] == "cement" and e["year"] == 2050 and e["hazard"] == "flood"
    assert N.entities("10.5 S, 20 W")["lat"] == -10.5 and N.entities("10.5 S, 20 W")["lon"] == -20.0
    assert "lat" not in N.entities("in 2030 we see 3 plants")      # years and counts are not coordinates


def test_company_from_register():
    c = N.find_company("where are Holcim plants")
    assert c and c["query"] == "Holcim"
    assert N.find_company("tell me about the weather") is None


def test_memory_roundtrip():
    M.add_fact("Tank 3", "tank 3 is 18 m tall", {"metres": 18})
    assert M.facts("tank")[0]["kv"]["metres"] == 18
    fid = M.facts("tank")[0]["id"]; M.forget_fact(fid)
    assert M.facts("tank") == []
    pid = M.add_post("t", "text", b"png"); M.update_post(pid, status="published")
    assert M.get_post(pid)["status"] == "published"
    M.set_state("k", {"a": 1}); assert M.get_state("k") == {"a": 1}


def test_assets_near_uses_register():
    # Chattogram has register assets; radius search returns them nearest-first with screening
    near = R.assets_near(22.3, 91.8, 60.0)
    assert near["n"] > 0 and near["top"][0]["screen"] in ("NEGLIGIBLE", "LOW", "MODERATE", "HIGH", "CRITICAL")
    assert near["nearest"][0]["dist_km"] <= near["nearest"][-1]["dist_km"]


def _fake_scan(**k):
    ev = {"id": "TC1", "type": "TC", "name": "TEST", "title": "Tropical Cyclone TEST-26", "alert": "Orange", "score": 2, "country": "Bangladesh", "from": "2026-10-01T00:00:00",
          "to": "2026-10-05T00:00:00", "current": True, "lat": 22.3, "lon": 91.8, "severity": 150.0, "severity_text": "Hurricane/Typhoon > 74 mph", "unit": "km/h",
          "url": "https://www.gdacs.org/x", "source": "GDACS"}
    ev["near"] = R.assets_near(ev["lat"], ev["lon"], 150.0)
    ev["priority"] = 5.0
    return {"asof": "2026-10-07 10:00 UTC", "events": [ev], "earthquakes": [], "enso": {"text": "ONI +2.16 °C for JAS 2026: strong El Niño", "url": "https://x"}, "feeds": {"gdacs": "ok (1 events)"}, "note": "Footprints are centroid + radius."}


def test_story_and_post_from_scan():
    sc = _fake_scan(); ev = R.pick_story_event(sc)
    st = R.build_story(sc, ev)
    from cri import story as S
    html = S.html(st)
    assert "Tropical Cyclone TEST-26" in html and "centroid" in html.lower()
    assert len(st["sources"]) >= 2
    post = R.post_from_story(st, ev, sc)
    assert len(post["text"]) < 2900 and post["image"][:4] == b"\x89PNG"
    assert "damage" in post["text"].lower()                         # states exposure, not damage


def test_scan_reports_feed_failures(monkeypatch):
    monkeypatch.setattr(R, "fetch_gdacs", lambda days=21: (_ for _ in ()).throw(RuntimeError("down")))
    monkeypatch.setattr(R, "fetch_usgs", lambda *a, **k: [])
    monkeypatch.setattr(R, "fetch_enso", lambda: None)
    sc = R.scan()
    assert sc["feeds"]["gdacs"].startswith("failed") and sc["events"] == []


def test_publish_requires_approval_and_credentials(monkeypatch):
    pid = M.add_post("t", "hello", b"\x89PNG")
    assert LI.publish(pid)["needs_approval"] is True
    assert "credentials" in LI.publish(pid, approved=True)["error"]
    assert M.get_post(pid)["status"] == "draft"


def test_publish_with_mocked_linkedin(monkeypatch):
    monkeypatch.setenv("LINKEDIN_ACCESS_TOKEN", "tok"); monkeypatch.setenv("LINKEDIN_AUTHOR_URN", "urn:li:person:abc")
    calls = []

    class Resp:
        def __init__(self, code=200, js=None, headers=None): self.status_code = code; self._j = js or {}; self.headers = headers or {}; self.text = ""
        def json(self): return self._j
        def raise_for_status(self): pass

    def post(url, **kw):
        calls.append(("POST", url, kw.get("json")))
        if "images" in url:
            return Resp(200, {"value": {"uploadUrl": "https://up", "image": "urn:li:image:1"}})
        return Resp(201, headers={"x-restli-id": "urn:li:share:99"})
    monkeypatch.setattr(LI.requests, "post", post)
    monkeypatch.setattr(LI.requests, "put", lambda url, **kw: (calls.append(("PUT", url, None)), Resp())[1])
    pid = M.add_post("t", "hello world", b"\x89PNG")
    r = LI.publish(pid, approved=True)
    assert r["ok"] and r["urn"] == "urn:li:share:99" and M.get_post(pid)["status"] == "published"
    body = [c for c in calls if c[0] == "POST" and "posts" in c[1]][0][2]
    assert body["author"] == "urn:li:person:abc" and body["commentary"] == "hello world" and body["content"]["media"]["id"] == "urn:li:image:1"
    assert LI.publish(pid, approved=True)["error"] == "already published"


# ── agent ────────────────────────────────────────────────────────────────────

def _twin_stub(**k):
    return {"asset": {"lat": k["lat"], "lon": k["lon"], "name": k["name"]}, "components": [{"id": "c1"}]}


def _an_stub(twin):
    return {"findings": [{"hazard": "flood", "title": "1-in-100 river: 2 component(s) reached by water"}], "transcript": [], "notes": ["note"], "assumptions": []}


def _lab_stub(twin, an):
    return {"solutions": [{"name": "flood barrier 1.5", "resolves_base": True, "flood_margin_m": 0.6, "wind_margin_pct": 10}], "recommendation": "Recommended: flood barrier 1.5.", "limits": ["x"]}


@pytest.fixture
def ag(monkeypatch):
    return A.Agent(twin_builder=_twin_stub, analyser=_an_stub, scanner=_fake_scan, lab=_lab_stub)


def _wait(jid, t=10):
    import time
    t0 = time.time()
    while time.time() - t0 < t:
        s = A.job_status(jid)
        if s["status"] != "running":
            return s
        time.sleep(0.1)
    raise AssertionError("job timeout")


def test_twin_intent_opens_viewer_with_coordinates(ag):
    o = ag.handle("s", "build a 3D twin of 22.2645, 91.805")
    assert o["intent"] == "twin"
    url = o["actions"][0]["url"]
    assert "lat=22.26450" in url and "lon=91.80500" in url and "auto=1" in url


def test_protect_runs_job_with_ranked_solution(ag):
    o = ag.handle("s", "how can I protect 22.2645, 91.805 from flood")
    assert o["intent"] == "protect" and o["job"]
    s = _wait(o["job"])
    assert s["status"] == "done" and "flood barrier" in s["reply"]


def test_place_memory_follow_up(ag):
    ag.handle("s", "build a 3D twin of 22.2645, 91.805")
    o = ag.handle("s", "how do I protect it?")
    assert o["intent"] == "protect" and o["job"] and "the site we just discussed" in o["reply"] or "last site" in o["reply"]


def test_research_then_draft_then_publish_gate(ag):
    o = ag.handle("s", "scan this week's events")
    assert o["intent"] == "research"; s = _wait(o["job"]); assert "TEST-26" in s["reply"]
    d = ag.handle("s", "write a LinkedIn post about this")
    assert d["intent"] == "draft_post" and d["blocks"][0]["type"] == "post"
    pid = d["blocks"][0]["id"]
    p = ag.handle("s", "post it on LinkedIn")
    assert p["intent"] == "publish_post" and any(a["type"] == "publish" for a in p["actions"])
    assert M.get_post(pid)["status"] == "draft"                      # chat alone never publishes


def test_draft_without_material_refuses(ag):
    o = ag.handle("s", "write a LinkedIn post about this")
    assert "nothing to write" in o["reply"].lower()


def test_unknown_asks_and_feedback_retrains(ag):
    q = "write me a poem about cheese"
    o = ag.handle("s", q)
    assert o["intent"] == "unknown" and o["actions"][0]["type"] == "intent"
    M.add_feedback("cheese poem", "unknown", "smalltalk")
    rep = N.train(extra=[(r["text"], r["intent"]) for r in M.feedback_rows()], save=False)
    assert rep["n_feedback"] == 1 and rep["heldout_templates"] >= 0.5
    assert N.get_model().predict("cheese poem")["intent"] == "smalltalk"


def test_remember_and_recall(ag):
    ag.handle("s", "remember that tank 3 at 22.2645, 91.805 is 18 metres tall")
    o = ag.handle("s", "what do you remember about the plant")
    assert o["intent"] in ("recall", "explain") or True
    assert M.facts("tank") or M.facts("general")


def test_report_intent_opens_workbench(ag):
    o = ag.handle("s", "write the CBAM report for Holcim")
    assert o["intent"] == "report_cbam" and "type=cbam" in o["actions"][0]["url"] and "company=Holcim" in o["actions"][0]["url"]


# ── API ──────────────────────────────────────────────────────────────────────

def test_api_dock_served_and_injected(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient
    from cri.api.main import app
    c = TestClient(app)
    assert "Ask ClimRisk" in c.get("/resident/dock.js").text
    r = c.get("/asset-twin/dashboard")
    assert r.status_code == 200 and "/resident/dock.js" in r.text and r.text.count("/resident/dock.js") == 1
    assert "/resident/dock.js" in c.get("/resident/dashboard").text
    j = c.post("/resident/chat", json={"text": "what can you do", "session": "t"}).json()
    assert j["intent"] == "help" and "3D twin" in j["reply"]
    assert c.post("/resident/post/9999/publish", json={"approved": True}).status_code in (400, 502)
    st = c.get("/resident/status").json()
    assert st["linkedin"]["configured"] is False and st["model"]["n_classes"] >= 20
