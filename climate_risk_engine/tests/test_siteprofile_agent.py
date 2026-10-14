import os, tempfile, time
import pytest
from cri import siteprofile as SP
from cri.resident import agent as AG


@pytest.fixture(autouse=True)
def _data(monkeypatch, tmp_path):
    monkeypatch.setenv("CRI_DATA_DIR", str(tmp_path))
    yield


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


def test_agent_stores_declared_sources_and_reruns_air(monkeypatch):
    from cri.analysis import local_air as LA
    os.environ["CRI_HOME"] = tempfile.mkdtemp()
    seen = {}
    def fake_profile(lat, lon, name, only=None, use_cache=True, **kw):
        seen["only"] = only
        ent = SP._entry("air_local", "Air", "constructed", value=20.4, severity="MODERATE",
                        metrics={"record": {"years_used": [2023, 2025], "annual_pm25": 19.0, "days_above_who_24h": 219.0, "days_above_naaqs_24h": 0.3, "diwali": [{"year": 2023, "ratio": 3.4}]},
                                 "increment": {"central": {"min": 0.1, "max": 1.4}, "high": {"max": 7.3}},
                                 "sources": [{"kind": "cement", "provenance": "declared", "modelled": True, "scenarios": {"central": {"pm25_annual_ug_m3": {"min": 0.0, "max": 1.0}}, "high": {"pm25_annual_ug_m3": {"max": 5.2}}}}]})
        return {"hazards": {"air_local": ent}}
    monkeypatch.setattr(SP, "profile", fake_profile)
    ag = AG.Agent(twin_builder=lambda **k: {}, analyser=lambda t: {}, scanner=lambda **k: {}, lab=lambda *a: {})
    out = ag.handle("s2", "we are really close to a cement manufacturing unit and a thermal power plant of 13 MW near 9.4713, 77.9107, and fireworks industries")
    assert out["intent"] == "local_sources" and out["job"]
    for _ in range(50):
        st = AG.job_status(out["job"])
        if st["status"] != "running":
            break
        time.sleep(0.1)
    assert st["status"] == "done", st
    assert seen["only"] == ["air_local", "water_stress"]
    d = LA.declared(9.4713, 77.9107)
    assert {x["kind"] for x in d} == {"cement", "thermal", "fireworks"}
    assert "not verified" in st["reply"] and "lower bound" in st["reply"]
