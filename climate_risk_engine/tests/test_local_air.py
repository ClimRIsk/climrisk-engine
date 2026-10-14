import datetime as dt, os, tempfile
import numpy as np
import pytest
from cri.analysis import local_air as LA


def _raw(days=900, pm=20.0, start="2023-01-01", wind_dir=45.0):
    t0 = dt.datetime.fromisoformat(start)
    ts = [(t0 + dt.timedelta(hours=i)).isoformat(timespec="minutes") for i in range(days * 24)]
    n = len(ts)
    aq = {"time": ts, "pm2_5": [pm] * n, "pm10": [pm * 1.5] * n, "sulphur_dioxide": [5.0] * n, "nitrogen_dioxide": [9.0] * n}
    met = {"time": ts, "wind_speed_10m": [3.0] * n, "wind_direction_10m": [wind_dir] * n, "cloud_cover": [40] * n, "is_day": [1 if 6 <= (i % 24) < 18 else 0 for i in range(n)]}
    return {"aq": aq, "met": met, "cell": (9.5, 77.9)}


def test_parse_declaration():
    s = LA.parse_declaration("we are close to a cement manufacturing unit and a thermal power plant of 13 MW about 3 km north-east, and fireworks industries")
    kinds = {x["kind"] for x in s}
    assert kinds == {"thermal", "cement", "fireworks"}
    th = next(x for x in s if x["kind"] == "thermal")
    assert th["assumptions"]["mw"] == 13.0 and th["distance_km"] == 3.0 and th["bearing_deg"] == 45
    assert LA.parse_declaration("what is the weather") == []


def test_record_stats_and_severity():
    r = LA.record_stats(_raw())
    assert r["annual_pm25"] == 20.0 and r["days_above_naaqs_24h"] == 0.0 and r["days_above_who_24h"] == 365.0
    assert LA.severity_pm25(4) == "NEGLIGIBLE" and LA.severity_pm25(19) == "MODERATE" and LA.severity_pm25(41) == "CRITICAL"


def test_emission_scales_with_capacity():
    a = LA.emission("thermal", {"mw": 13})["pm_g_s"]["central"]
    b = LA.emission("thermal", {"mw": 26})["pm_g_s"]["central"]
    assert abs(b / a - 2) < 0.01
    assert LA.emission("fireworks") is None
    assert LA.emission("cement", {"clinker_mtpa": 2})["pm_g_s"]["low"] == pytest.approx(2 * LA.emission("cement")["pm_g_s"]["low"], rel=0.01)


def test_plume_direction_matters():
    raw = _raw(wind_dir=45.0)                      # wind FROM north-east carries a north-east source to the site
    src = [{"kind": "cement", "distance_km": 3.0, "bearing_deg": 45}]
    up = LA.screen(9.5, 77.9, src, raw)[0]["scenarios"]["central"]["pm25_annual_ug_m3"]["max"]
    src2 = [{"kind": "cement", "distance_km": 3.0, "bearing_deg": 225}]
    away = LA.screen(9.5, 77.9, src2, raw)[0]["scenarios"]["central"]["pm25_annual_ug_m3"]["max"]
    assert up > 0.5 and away < 0.01


def test_unknown_geometry_is_swept_and_ordered():
    s = LA.screen(9.5, 77.9, [{"kind": "thermal", "distance_km": None, "bearing_deg": None}], _raw())[0]
    assert s["geometry_swept"]["bearings_deg"] and len(s["geometry_swept"]["distances_km"]) == 3
    c = s["scenarios"]
    assert c["low"]["pm25_annual_ug_m3"]["max"] <= c["central"]["pm25_annual_ug_m3"]["max"] <= c["high"]["pm25_annual_ug_m3"]["max"]
    assert "so2_uncontrolled" in c


def test_assess_statuses_and_unverified_flag():
    os.environ["CRI_HOME"] = tempfile.mkdtemp()
    a = LA.assess(9.5, 77.9, osm=False, raw=_raw(), sources_override=[])
    assert a["status"] == "modelled" and a["annual_pm25_total_central"] == 20.0 and not a["sources"]
    srcs = LA.parse_declaration("a cement unit and fireworks units nearby")
    b = LA.assess(9.5, 77.9, osm=False, raw=_raw(), sources_override=srcs)
    assert b["status"] == "constructed" and b["annual_pm25_total_central"] > 20.0
    assert "cement" in b["unverified_sources"] and "fireworks" in b["unverified_sources"]
    assert any("fireworks" in x["kind"] and not x["modelled"] for x in b["sources"])
    assert "fugitive" in " ".join(b["not_modelled"])


def test_unreadable_record_is_null():
    def boom(*a, **k):
        raise OSError("down")
    os.environ["CRI_CACHE_DIR"] = tempfile.mkdtemp()
    a = LA.assess(1.234, 5.678, osm=False, fetch=boom, sources_override=[])
    assert a["status"] == "null"


def test_declare_roundtrip():
    os.environ["CRI_HOME"] = tempfile.mkdtemp()
    LA.declare(9.47, 77.91, LA.parse_declaration("a cement unit nearby"))
    LA.declare(9.47, 77.91, LA.parse_declaration("thermal power plant 13 MW"))
    d = LA.declared(9.47, 77.91)
    assert {x["kind"] for x in d} == {"cement", "thermal"} and all("declared" in x["provenance"] for x in d)


from cri.analysis import site_sources as SS


def _reg(lat, lon, r, limit=200):
    return {"top": [{"name": "Test Cement Plant", "lat": lat + 0.02, "lon": lon + 0.01, "sector": "cement", "type": "integrated dry", "emissions_t": 1.74e6},
                    {"name": "Solar farm", "lat": lat, "lon": lon + 0.05, "sector": "electricity-generation", "type": "solar", "emissions_t": 0}]}


def test_register_discovery_and_inference():
    os.environ["CRI_CACHE_DIR"] = tempfile.mkdtemp()
    d = SS.discover(10.0, 20.0, osm=False, places=False, register=_reg)
    kinds = [s["kind"] for s in d["sources"]]
    assert kinds.count("cement") == 1 and "thermal" in kinds            # solar ignored, captive power inferred
    cem = next(s for s in d["sources"] if s["kind"] == "cement")
    assert cem["confidence"] == "high" and 2 < cem["distance_km"] < 4 and cem["bearing_deg"] < 90
    th = next(s for s in d["sources"] if s["kind"] == "thermal")
    assert th["confidence"] == "inferred" and "inference" in th["provenance"]


def test_place_knowledge_is_strict():
    import json as _j
    def fake_get(url, *a, **k):
        if "wikidata" in url:
            return {"results": {"bindings": [{"c": {"value": "Point(77.8 9.45)"}, "iLabel": {"value": "Sivakasi"}, "a": {"value": "https://en.wikipedia.org/wiki/Sivakasi"}},
                                              {"c": {"value": "Point(77.9 9.50)"}, "iLabel": {"value": "Roadville"}, "a": {"value": "https://en.wikipedia.org/wiki/Roadville"}}]}}
        return {"query": {"pages": {"1": {"title": "Sivakasi", "extract": "The city is known for firecrackers and match factories that produce 70% of the output."},
                                    "2": {"title": "Roadville", "extract": "The village has cement roads and a school. It is known for its temple."}}}}
    os.environ["CRI_CACHE_DIR"] = tempfile.mkdtemp()
    out, note = SS.from_places(9.47, 77.91, 25.0, get=fake_get)
    assert [o["kind"] for o in out] == ["fireworks"] and out[0]["confidence"] == "low" and out[0]["cluster"]


def test_no_firms_key_is_stated_not_hidden():
    os.environ.pop("NASA_FIRMS_MAP_KEY", None)
    s, note = SS.from_firms(1.0, 2.0)
    assert s == [] and "no FIRMS key" in note


def test_assess_uses_discovered_sources_and_user_overrides(monkeypatch):
    os.environ["CRI_HOME"] = tempfile.mkdtemp()
    os.environ["CRI_CACHE_DIR"] = tempfile.mkdtemp()
    monkeypatch.setattr(SS, "discover", lambda lat, lon, **k: SS.discover.__wrapped__(lat, lon, **k) if False else {"sources": [
        {"kind": "cement", "name": "Found Cement", "lat": 9.52, "lon": 77.92, "distance_km": 2.4, "bearing_deg": 134, "provenance": "register", "confidence": "high", "co2e_t_per_year": 1.74e6},
        {"kind": "thermal", "name": "captive (inferred)", "distance_km": 2.4, "bearing_deg": 134, "provenance": "inference", "confidence": "inferred", "assumptions": {"mw": 15.0}}], "log": ["x"], "radius_km": 25})
    a = LA.assess(9.5, 77.9, raw=_raw(wind_dir=134.0))
    cem = next(s for s in a["sources"] if s["kind"] == "cement")
    assert cem["emission"]["assumptions"]["clinker_mtpa"] == 2.0 and cem["wind_from_source_pct"] > 90
    assert a["status"] == "constructed" and "thermal" in a["unverified_sources"]
    LA.declare(9.5, 77.9, LA.parse_declaration("a thermal power plant of 13 MW"))        # the user's word replaces the inferred one
    b = LA.assess(9.5, 77.9, raw=_raw(wind_dir=134.0))
    th = [s for s in b["sources"] if s["kind"] == "thermal"]
    assert len(th) == 1 and th[0]["emission"]["assumptions"]["mw"] == 13.0 and th[0]["confidence"] == "declared"
