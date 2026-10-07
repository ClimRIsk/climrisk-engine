"""Autopilot: the resident agent working unattended. Feeds, analyst and twin stubbed; inbox, dedupe, gating, commands and start-up are real."""
import time

import pytest

from cri.resident import agent as A, autopilot as AP, memory as M, nlu as N, research as R


@pytest.fixture(autouse=True)
def _tmp(tmp_path, monkeypatch):
    monkeypatch.setenv("CRI_DATA_DIR", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(N, "MODEL_PATH", tmp_path / "nlu.json")
    N._M = None; A._AGENT = None if hasattr(A, "_AGENT") else None


EVENT = {"id": "TC1", "type": "TC", "title": "Tropical Cyclone TEST", "name": "TEST", "alert": "Orange", "country": "India", "from": "2026-10-01", "to": "2026-10-05", "current": True,
         "lat": 17.5, "lon": 83.3, "severity": 120, "url": "", "score": 2.0,
         "near": {"n": 2, "emissions_t": 3e6, "radius_km": 100.0, "sectors": {"power": 2}, "nearest": [{"name": "A", "dist_km": 10}],
                  "top": [{"i": 1, "name": "Plant A", "country": "IND", "sector": "power", "lat": 17.58, "lon": 83.29, "dist_km": 10, "emissions_t": 2e6, "elev_m": 5, "rp100_ms": 47.0, "screen": "MODERATE"}]}}


def _scan(*a, **k):
    return {"asof": "2026-10-07 10:00 UTC", "events": [dict(EVENT)], "earthquakes": [], "enso": {"text": "ONI +2.16 strong El Niño", "url": "u"}, "feeds": {"gdacs": "ok (1 events)"}, "note": "n"}


def test_inbox_dedupes_marks_read_and_caps():
    a = AP.inbox_add("x", "t", "d", key="k1"); b = AP.inbox_add("x", "t", "d", key="k1")
    assert a and b is None and len(AP.inbox()) == 1 and AP.status()["unread"] == 1
    assert AP.mark_read() == 1 and AP.status()["unread"] == 0
    for i in range(AP.MAX_INBOX + 10):
        AP.inbox_add("x", f"t{i}", "", key=f"q{i}")
    assert len(AP.inbox()) == AP.MAX_INBOX


def test_sites_registered_once_and_removed():
    AP.register_site(22.2645, 91.805, "Chattogram"); AP.register_site(22.2646, 91.8051, "dup")
    assert len(AP.sites()) == 1
    AP.remove_site("chattogram"); assert AP.sites() == []


def test_cycle_scans_triages_drafts_and_never_publishes(monkeypatch):
    monkeypatch.setattr(R, "scan", _scan)
    monkeypatch.setattr("cri.finance.analyst.analyse_asset", lambda *a, **k: {
        "asset": {"value": {"basis": "proxy"}}, "physical": {"cyclone": {"severity": "HIGH", "rp100_ms": 61.0, "category_100yr": "Cat 4"}},
        "financial": {"physical": {"aal_pct": 1.5}, "scenarios": {"Delayed transition": {"total_npv": {"mean": 1e8, "p95": 3e8}}}}})
    r = AP.run_cycle("test")
    assert r["ok"] and any(l.startswith("research") for l in r["log"]) and any(l.startswith("drafts") for l in r["log"])
    kinds = {i["kind"] for i in AP.inbox()}
    assert {"event", "triage", "draft", "briefing"} <= kinds
    ps = M.posts(10)
    assert len(ps) == 1 and ps[0]["status"] == "draft" and ps[0]["published_ts"] is None       # drafted, never published
    AP.run_cycle("again")
    assert len(M.posts(10)) == 1 and sum(1 for i in AP.inbox() if i["kind"] == "event") == 1    # no duplicate work
    assert AP.status()["cycles"] == 2 and AP.status()["last_run"]


def test_cycle_survives_feed_failure(monkeypatch):
    def boom(*a, **k): raise RuntimeError("offline")
    monkeypatch.setattr(R, "scan", boom)
    r = AP.run_cycle("offline")
    assert r["ok"] and any("research failed" in l for l in r["log"])


def test_watched_site_in_reach_runs_twin_and_lab(monkeypatch):
    monkeypatch.setattr(R, "scan", _scan)
    monkeypatch.setattr("cri.finance.analyst.analyse_asset", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("skip")))
    calls = {}
    class FakeAgent:
        def twin_builder(self, **k): calls["twin"] = k; return {"components": []}
        def analyser(self, tw): return {"findings": [{"title": "River flood: 3 reached", "hazard": "flood"}], "notes": []}
        def lab(self, tw, an): calls["lab"] = 1; return {"solutions": [{"name": "Raise equipment", "resolves_base": True, "flood_margin_m": 0.6, "wind_margin_pct": None}], "recommendation": "Raise equipment 2.6 m."}
    monkeypatch.setattr(A, "get_agent", lambda: FakeAgent())
    AP.register_site(17.58, 83.29, "Plant A", "power"); AP.register_site(-30.0, 20.0, "Far away")
    AP.run_cycle("test")
    site = [i for i in AP.inbox() if i["kind"] == "site"]
    assert len(site) == 1 and "Plant A" in site[0]["title"] and calls["lab"] == 1 and calls["twin"]["name"] == "Plant A"
    assert any(a["type"] == "open" for a in site[0]["actions"])


def test_chat_commands_inbox_watch_and_run(monkeypatch):
    AP.inbox_add("event", "Cyclone X", "detail", key="e")
    ag = A.Agent(twin_builder=lambda **k: {}, analyser=lambda t: {"findings": []}, scanner=_scan, lab=lambda *a: {})
    out = ag.handle("s", "what did you find while I was away?")
    assert "new item" in out["reply"] and AP.status()["unread"] == 0
    w = ag.handle("s", "watch Holcim")
    assert "Watching" in w["reply"]
    from cri.api.agents import _load_watchlist
    assert any("holcim" in x["company_name"].lower() for x in _load_watchlist())
    monkeypatch.setattr(AP, "run_cycle", lambda reason="": None)
    assert "Started a cycle" in ag.handle("s", "run the autopilot now")["reply"]


def test_start_is_idempotent_and_can_be_disabled(monkeypatch):
    monkeypatch.setenv("CRI_AUTOPILOT", "0")
    assert AP.start() is False
    monkeypatch.setenv("CRI_AUTOPILOT", "1"); monkeypatch.setenv("CRI_AUTOPILOT_DELAY_S", "60")
    try:
        assert AP.start() is True and AP.start() is False
    finally:
        AP.stop(); time.sleep(0.05)
        AP._THREAD = None
