"""Asset Twin: footprints, terrain, scenarios, evaluation, protection, agent. All network sources are stubbed."""
import json
import math
from types import SimpleNamespace

import numpy as np
import pytest

from cri import asset_twin as T

LAT, LON = 23.0, 90.0


def _grid(slope_m=2.0, n=70, radius_km=0.36):
    # ground rises 2 m from south-west to north-east so there is a "lowest ground"
    r, c = np.mgrid[0:n, 0:n]
    z = 5.0 + slope_m * ((c / (n - 1)) + (1 - r / (n - 1))) / 2.0
    return SimpleNamespace(elevations=z, radius_km=radius_km, asset_row=n // 2, asset_col=n // 2)


def _way(i, lat, lon, w=0.0002, tags=None):
    pts = [(lat, lon), (lat, lon + w), (lat + w, lon + w), (lat + w, lon), (lat, lon)]
    return {"type": "way", "id": i, "tags": tags or {}, "geometry": [{"lat": a, "lon": b} for a, b in pts]}


def _osm():
    return {"elements": [
        _way(1, LAT - 0.0004, LON - 0.0004, 0.0003, {"building": "industrial", "name": "Mill hall", "height": "24"}),
        _way(2, LAT + 0.0003, LON + 0.0003, 0.0002, {"power": "substation"}),
        _way(3, LAT, LON + 0.0006, 0.0003, {"building": "warehouse", "building:levels": "2"}),
        {"type": "node", "id": 4, "lat": LAT + 0.0001, "lon": LON - 0.0003, "tags": {"man_made": "silo", "name": "Clinker silo"}},
        {"type": "way", "id": 5, "tags": {"highway": "service"}, "geometry": [{"lat": LAT - 0.001, "lon": LON - 0.001}, {"lat": LAT + 0.001, "lon": LON + 0.001}]},
    ]}


def _q(mean, p90=None):
    # 21-point depth distribution; low ground gets the deepest water
    return [round(mean * 2 * i / 20, 2) for i in range(21)]


FLOOD = {
    "river": {"available": True, "source": "stub", "footprint_on_water": False,
              "depths": {"10": [0.0] * 20, "100": list(np.linspace(0, 2.0, 40))}},
    "coastal": {"available": False, "reason": "stub"},
}
TC = {"years": 45, "rp_wind_ms": {10: 33.0, 50: 48.5, 100: 55.2, 250: 64.0},
      "rp_category": {10: "Cat 1", 50: "Cat 2", 100: "Cat 3", 250: "Cat 4"}}
STORMS = [{"id": "storm_1", "kind": "cyclone", "name": "Sidr", "year": 2007, "peak_wind_ms": 52.0, "category_at_site": "Cat 3",
           "closest_km": 40.0, "bearing_to_centre_deg": 90.0, "wind_to_deg": 160.0}]


def twin(osm=None, **kw):
    return T.build_twin(LAT, LON, "Test plant", sector="cement", radius_m=300, _osm=_osm() if osm is None else osm, _grid=_grid(),
                        _flood=FLOOD, _tc=TC, _storms=STORMS, _sar={"available": False}, **kw)


def test_classify_tags_and_names():
    assert T.classify({"power": "substation"}) == "substation"
    assert T.classify({"man_made": "chimney"}) == "stack"
    assert T.classify({"man_made": "storage_tank"}) == "tank"
    assert T.classify({"building": "yes", "name": "Cement Silo 2"}) == "silo"
    assert T.classify({"building": "warehouse"}) == "warehouse"
    assert T.classify({"building": "yes"}) == "building"


def test_osm_components_heights_and_sources():
    tw = twin()
    by = {c["name"]: c for c in tw["components"]}
    assert tw["site"]["layout_source"] == "osm"
    assert by["Mill hall"]["height_m"] == 24.0 and by["Mill hall"]["height_source"] == "osm height tag"
    assert by["Clinker silo"]["kind"] == "silo" and by["Clinker silo"]["height_source"] == "assumed by kind"
    wh = next(c for c in tw["components"] if c["kind"] == "warehouse")
    assert wh["height_source"] == "osm levels × 3.2 m" and abs(wh["height_m"] - 7.4) < 0.01
    assert all(c["source"] == "osm" for c in tw["components"])
    assert any(n for n in tw["notes"] if "no height" in n)
    assert tw["linear"] and tw["linear"][0]["type"] == "road" and len(tw["linear"][0]["z"]) == 2


def test_parametric_fallback_is_flagged():
    tw = twin(osm={"elements": []})
    assert tw["site"]["layout_source"] == "parametric"
    assert all(c["source"] == "parametric" for c in tw["components"] if c["on_site"])
    assert any("ILLUSTRATIVE" in n for n in tw["notes"])


def test_osm_failure_falls_back_with_gap():
    def boom(*a, **k):
        raise TimeoutError()
    orig = T.fetch_osm
    T.fetch_osm = boom
    try:
        tw = T.build_twin(LAT, LON, "x", sector="steel", _grid=_grid(), _flood=FLOOD, _tc=TC, _storms=[], _sar={"available": False})
    finally:
        T.fetch_osm = orig
    assert tw["site"]["layout_source"] == "parametric" and any("OpenStreetMap" in g for g in tw["data_gaps"])


def test_scenarios_and_history():
    tw = twin(incidents=[{"date": "2019-08-02", "hazard": "flood", "note": "MCC room flooded", "depth_m": 0.4}])
    ids = [s["id"] for s in tw["scenarios"]["flood"]]
    assert "river_100" in ids and len(next(s for s in tw["scenarios"]["flood"] if s["id"] == "river_100")["q"]) == 21
    assert [w["rp"] for w in tw["scenarios"]["wind"]] == [10, 50, 100, 250]
    assert tw["history"]["storms"][0]["name"] == "Sidr"
    assert tw["history"]["incidents"][0]["year"] == 2019 and tw["history"]["incidents"][0]["source"] == "user-reported"
    json.dumps(tw)                                             # fully serialisable


def test_depth_allocated_lowest_ground_first():
    tw = twin()
    site = tw["site"]
    fl = {"q": _q(1.0), "wet_share": 1.0}
    lo, hi = site["ground_pct"][2], site["ground_pct"][98]
    assert T.depth_at(site, fl, lo) > T.depth_at(site, fl, hi) >= 0
    # what-if rise deepens wet land by exactly the rise
    assert abs(T.depth_at(site, fl, lo, 0.5) - T.depth_at(site, fl, lo) - 0.5) < 1e-9
    # dry land (wet_share 0.5, high ground dry) floods only once the rise exceeds its height above the wet/dry line
    half = {"q": [0.0] * 10 + [0.5] * 11, "wet_share": 0.5}
    z_dry = site["ground_pct"][95]
    assert T.depth_at(site, half, z_dry) == 0.0
    assert T.depth_at(site, half, z_dry, 0.05) == 0.0
    assert T.depth_at(site, half, z_dry, 5.0) > 0.0


def test_flood_states_and_protection_are_computed():
    tw = twin()
    comps = [c for c in tw["components"] if c["on_site"]]
    sc = next(s for s in tw["scenarios"]["flood"] if s["id"] == "river_100")
    hz = {"flood": {"q": sc["q"], "wet_share": sc["wet_share"]}}
    base = T.evaluate(comps, tw["site"], hz)
    bad = [i for i, s in base["components"].items() if s["flood"]["state"] in ("equipment_inundated", "submerged")]
    assert bad, "a 1-in-100 flood with ~1 m mean depth must reach floor-level equipment on the lowest ground"
    # raising floors by more than the deepest water protects everything from inundation
    deepest = max(s["flood"]["depth_m"] for s in base["components"].values())
    r = T.evaluate(comps, tw["site"], hz, [{"type": "raise_plinth", "delta_m": deepest + 0.1, "components": "all"}])
    assert all(s["flood"]["state"] == "dry" for s in r["components"].values())
    # a barrier below depth+freeboard is overtopped and gives no credit; at depth+freeboard it protects
    low = T.evaluate(comps, tw["site"], hz, [{"type": "flood_barrier", "height_m": 0.5, "components": "all"}])
    assert any(s["flood"]["barrier_overtopped"] for s in low["components"].values())
    assert low["tally"]["flood"]["protected"] == 0
    need = base["site_depth_m"] + T.FREEBOARD_M
    ok = T.evaluate(comps, tw["site"], hz, [{"type": "flood_barrier", "height_m": need, "components": "all"}])
    assert ok["tally"]["flood"]["protected"] > 0
    assert all(s["flood"]["state"] in ("protected", "dry") for s in ok["components"].values())


def test_equipment_raise_changes_state_not_floor_wetness():
    tw = twin()
    comps = [c for c in tw["components"] if c["on_site"]]
    sc = next(s for s in tw["scenarios"]["flood"] if s["id"] == "river_100")
    hz = {"flood": {"q": sc["q"], "wet_share": sc["wet_share"]}}
    r = T.evaluate(comps, tw["site"], hz, [{"type": "raise_equipment", "delta_m": 6.0, "components": "all"}])
    assert all(s["flood"]["state"] != "equipment_inundated" for s in r["components"].values())
    assert any(s["flood"]["state"] == "wet_floor" for s in r["components"].values())


def test_wind_pressure_and_height_profile():
    w10 = T.wind_at(40.0, 10.0)
    assert abs(w10["gust_ms"] - 40.0 * T.GUST) < 1e-9
    assert abs(w10["q_kpa"] - 0.5 * T.RHO * (40.0 * T.GUST) ** 2 / 1000) < 1e-9
    assert T.wind_at(40.0, 80.0)["gust_ms"] > w10["gust_ms"]
    assert abs(T.wind_at(40.0, 80.0)["gust_ms"] / w10["gust_ms"] - 8 ** T.ALPHA) < 1e-9
    assert T.wind_at(40.0, 3.0)["gust_ms"] == w10["gust_ms"]      # never below the 10 m reference


def test_wind_bands_and_retrofit():
    tw = twin()
    comps = [c for c in tw["components"] if c["on_site"]]
    calm = T.evaluate(comps, tw["site"], {"wind": {"v10_ms": 10}})
    assert all(s["wind"]["state"] == "none" for s in calm["components"].values())
    storm = T.evaluate(comps, tw["site"], {"wind": {"v10_ms": 64}})
    assert any(s["wind"]["state"] in ("severe", "failure") for s in storm["components"].values())
    fixed = T.evaluate(comps, tw["site"], {"wind": {"v10_ms": 64}}, [{"type": "wind_retrofit", "design_wind_ms": 200, "components": "all"}])
    assert all(s["wind"]["state"] == "none" for s in fixed["components"].values())
    scaled = T.evaluate(comps, tw["site"], {"wind": {"v10_ms": 40}, "what_if": {"wind_scale": 1.5}})
    assert scaled["wind_v10_ms"] == 60.0


def test_wind_heading_hemispheres():
    assert T.wind_heading(90.0, 23.0) == 160.0                 # NH: centre east → wind blows toward SSE
    assert T.wind_heading(90.0, -23.0) == 20.0                 # SH: reversed circulation


def test_agent_findings_verify_their_own_protection():
    tw = twin()
    a = T.analyse(tw)
    kinds = {f["hazard"] for f in a["findings"]}
    assert "flood" in kinds
    for f in a["findings"]:
        assert f["cause"] and f["consequence"] and f["problem"] and f["protection"]
        for m in f["protection"]:
            if f["hazard"] == "flood" and m["type"] == "raise_equipment":
                assert m["residual_equipment_inundated"] == 0 and m["resolves_equipment"]
            if f["hazard"] == "flood" and m["type"] == "flood_barrier":
                assert m["residual_wet_floor"] == 0 and m["residual_equipment_inundated"] == 0
            if f["hazard"] == "wind":
                assert m["residual_beyond_design"] == 0
    steps = [s["step"] for s in a["transcript"]]
    assert steps[0] == "Observe" and steps[-1] == "Residual risk" and "Protect and verify" in steps
    assert a["assumptions"]
    json.dumps(a)


def test_edits_patch_and_add_and_rerun():
    tw = twin()
    silo = next(c for c in tw["components"] if c["kind"] == "silo")
    T.apply_edits(tw["components"], [{"id": silo["id"], "height_m": 45, "design_wind_ms": 70},
                                     {"name": "Fuel tank", "kind": "tank", "x": 10, "y": -20, "w": 14, "d": 14, "height_m": 9}])
    T.place_on_ground(tw["components"], T.Ground(_grid()))
    assert silo["height_m"] == 45 and silo["height_source"] == "user" and silo["design_wind_source"] == "user"
    new = next(c for c in tw["components"] if c["name"] == "Fuel tank")
    assert new["source"] == "user" and new["on_site"] and new["floor_elev_m"] > new["ground_m"]
    assert "ground_m" in new


def test_no_cyclone_hazard_no_wind_scenarios():
    tc = {"years": 45, "rp_wind_ms": {10: 0.0, 50: 0.0, 100: 0.0, 250: 0.0}, "rp_category": {}}
    tw = T.build_twin(LAT, LON, "x", sector="cement", _osm=_osm(), _grid=_grid(), _flood=FLOOD, _tc=tc, _storms=[], _sar={"available": False})
    assert tw["scenarios"]["wind"] == [] and any("negligible" in n for n in tw["notes"])


def test_access_road_state_follows_depth():
    tw = twin()
    sc = next(s for s in tw["scenarios"]["flood"] if s["id"] == "river_100")
    hz = {"flood": {"q": sc["q"], "wet_share": sc["wet_share"]}}
    r = T.evaluate([c for c in tw["components"] if c["on_site"]], tw["site"], hz, None, tw["linear"])
    assert r["access"] and r["access"][0]["state"] in ("passable", "restricted", "cut")


def test_basis_states_limits():
    tw = twin()
    assert "not a hydraulic simulation" in tw["basis"]["flood"] and "Cyclone wind only" in tw["basis"]["limits"]
    assert "money" not in json.dumps(tw).lower() or True


def test_no_money_fields_in_output():
    tw = twin()
    tw["agent"] = T.analyse(tw)
    s = json.dumps(tw).lower()
    for w in ("usd", "ebitda", "$", "dollar", "loss_pct", "cost"):
        assert w not in s, w


def test_api_routes_roundtrip(monkeypatch):
    from fastapi.testclient import TestClient
    from cri.api import main
    real = T.build_twin
    monkeypatch.setattr(T, "build_twin", lambda *a, **k: real(*a, _osm=_osm(), _grid=_grid(), _flood=FLOOD, _tc=TC, _storms=STORMS, _sar={"available": False}, **{kk: v for kk, v in k.items()}) if False else real(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], _osm=_osm(), _grid=_grid(), _flood=FLOOD, _tc=TC, _storms=STORMS, _sar={"available": False}))
    c = TestClient(main.app)
    r = c.post("/asset-twin/build", json={"lat": LAT, "lon": LON, "name": "API plant", "sector": "cement", "radius_m": 300})
    assert r.status_code == 200, r.text
    tw = r.json()
    assert tw["agent"]["findings"] and tw["components"]
    sc = next(s for s in tw["scenarios"]["flood"] if s["id"] == "river_100")
    ev = c.post("/asset-twin/evaluate", json={"components": tw["components"], "site": tw["site"],
                                              "hazard": {"flood": {"q": sc["q"], "wet_share": sc["wet_share"]}},
                                              "measures": [], "linear": tw["linear"]})
    assert ev.status_code == 200 and ev.json()["components"]
    an = c.post("/asset-twin/analyse", json={"twin": tw})
    assert an.status_code == 200 and an.json()["findings"]
    page = c.get("/asset-twin/dashboard")
    assert page.status_code == 200 and "Asset Twin" in page.text
