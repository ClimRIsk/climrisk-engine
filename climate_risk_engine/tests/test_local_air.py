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
