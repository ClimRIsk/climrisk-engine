import os, tempfile, time
import pytest
from cri import siteprofile as SP
from cri.resident import agent as AG


def _p():
    ok = SP._entry("cyclone", "Cyclone wind", "modelled", value=21.9, unit="m/s", severity="LOW")
    nf = SP._entry("sealevel", "Sea level", "none_found")
    nl = SP._entry("earthquake", "Earthquake shaking", "null", reason="too sparse")
    return {"hazards": {"cyclone": ok, "sealevel": nf, "earthquake": nl}, "counts": {"modelled": 1, "constructed": 0, "none_found": 1, "null": 1},
            "rules": ["r"], "history": {"flood_events": {"events": [], "outside_dense_coverage": True}}, "null_hazards": ["earthquake"], "top_severity": "LOW"}


def test_rows_show_null_not_zero():
    rows = {r[0]: r for r in AG.profile_rows(_p())}
    assert rows["Earthquake shaking"][2] == "null" and rows["Earthquake shaking"][3] == "not assessed"
    assert rows["Sea level"][2].startswith("0") and "absent" in rows["Sea level"][2]
    assert rows["Cyclone wind"][2] == "21.9 m/s"


def test_twin_cache_roundtrip():
    os.environ["CRI_HOME"] = tempfile.mkdtemp()
    assert SP.cached_twin(10.0, 20.0) is None
    SP.save_twin(10.0, 20.0, {"history": {"windstorms": [1]}, "scenarios": {"flood": {"x": 1}}})
    t = SP.cached_twin(10.004, 20.004)
    assert t and t["history"]["windstorms"] == [1]
    assert SP.cached_twin(11.0, 20.0) is None


def test_run_site_profile_uses_twin_and_reports(monkeypatch):
    os.environ["CRI_HOME"] = tempfile.mkdtemp()
    seen = {}
    def fake(lat, lon, name, twin=None, progress=None, **kw):
        seen["twin"] = twin
        return _p()
    monkeypatch.setattr(SP, "profile", fake)
    SP.save_twin(5.0, 6.0, {"history": {}, "scenarios": {}})
    out = AG.run_site_profile({"lat": 5.0, "lon": 6.0, "name": "X"})
    assert seen["twin"] is not None
    assert "NULL" in out["reply"] or "null" in out["reply"].lower()
    assert out["profile"]["null_hazards"] == ["earthquake"]
    assert any(b["type"] == "note" and "outside the regions" in b.get("text", "") for b in out["blocks"])


def test_profile_regex_triggers():
    rx = AG.Agent._PROFILE_RX if hasattr(AG, "Agent") else None
    assert rx is not None
    for t in ["all hazards at 22.26, 91.80", "hazard profile for Chattogram", "how risky is 9.47, 77.91", "what are the risks at Dhaka"]:
        assert rx.search(t), t
    assert not rx.search("write the CBAM report for Holcim")


def test_agent_routes_bare_coordinates_to_profile(monkeypatch):
    os.environ["CRI_HOME"] = tempfile.mkdtemp()
    monkeypatch.setattr(SP, "profile", lambda lat, lon, name, twin=None, progress=None, **kw: _p())
    ag = AG.Agent(twin_builder=lambda **k: {}, analyser=lambda t: {}, scanner=lambda **k: {}, lab=lambda *a: {})
    out = ag.handle("s1", "9.4713, 77.9107")
    assert out["intent"] == "site_profile" and out.get("job")
    for _ in range(50):
        st = AG.job_status(out["job"])
        if st["status"] != "running":
            break
        time.sleep(0.1)
    assert st["status"] == "done", st
    assert "null" in " ".join(str(b) for b in st["blocks"]).lower()
    out2 = ag.handle("s1", "all hazards at 22.26, 91.80")
    assert out2["intent"] == "site_profile"
    # an ordinary single-hazard request is still routed to the lab, not the profile
    out3 = ag.handle("s1", "build a flood model for 22.2645, 91.805")
    assert out3.get("intent") == "modellab"
