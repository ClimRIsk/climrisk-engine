"""Rain-driven flooding: gauge-anchored design rain, the catchment-scale run, the twin's rain scenario and the site-profile entries. No network: every external read is injected."""
from __future__ import annotations

import datetime as dt
from types import SimpleNamespace

import numpy as np
import pytest

from cri import asset_twin as AT, hydraulics as H, siteprofile as SP
from cri.finance import pluvial as PV, rain_design as RD


def _gauge_csv(years=range(1980, 2021), big=None):
    """Daily PRCP csv in the NCEI shape (STATION,NAME,DATE,PRCP); modest rain every day, one wet day per year, `big` = {date: mm}."""
    rows = ["STATION,NAME,DATE,PRCP"]
    d = dt.date(min(years), 1, 1)
    end = dt.date(max(years), 12, 31)
    while d <= end:
        mm = 2.0 + (d.timetuple().tm_yday % 7)
        if d.month == 10 and d.day == 15:
            mm = 80.0 + (d.year % 10) * 6.0
        if big and d.isoformat() in big:
            mm = big[d.isoformat()]
        rows.append(f"X,TEST,{d.isoformat()},{mm}")
        d += dt.timedelta(days=1)
    return "\n".join(rows)


def test_gauge_annual_maxima_and_three_day_total(monkeypatch):
    from cri.modellab import stationcheck as SC
    monkeypatch.setattr(SC, "nearest", lambda *a, **k: [{"id": "X", "name": "Test Gauge", "km": 3.0, "first": 1980, "last": 2020}])
    ser = RD.daily_gauge(12.9, 80.2, fetch=lambda u: _gauge_csv(big={"2015-12-02": 345.0}))
    assert ser["km"] == 3.0 and "Test Gauge" in ser["source"]
    am1, am3 = RD.annual_maxima(ser, 1), RD.annual_maxima(ser, 3)
    assert am1[2015][0] == pytest.approx(345.0) and am1[2015][1] == "2015-12-02"
    assert am3[2015][0] >= 345.0


def test_gauge_gap_is_not_undercounted(monkeypatch):
    from cri.modellab import stationcheck as SC
    monkeypatch.setattr(SC, "nearest", lambda *a, **k: [{"id": "X", "name": "Test Gauge", "km": 3.0, "first": 1980, "last": 2020}])
    txt = _gauge_csv().split("\n")
    txt = [r for r in txt if ",1999-06-1" not in r]                 # a ten-day hole in 1999
    ser = RD.daily_gauge(12.9, 80.2, fetch=lambda u: "\n".join(txt))
    i = ser["t"].index("1999-06-10")
    assert ser["mm"][i] is None


def test_design_takes_the_gauge_when_it_exceeds_the_grids(monkeypatch):
    from cri.modellab import stationcheck as SC
    monkeypatch.setattr(SC, "nearest", lambda *a, **k: [{"id": "X", "name": "Test Gauge", "km": 3.0, "first": 1980, "last": 2020}])
    grid = {"source": "grid", "t": [], "mm": []}
    d = dt.date(1991, 1, 1)
    while d <= dt.date(2020, 12, 31):
        grid["t"].append(d.isoformat()); grid["mm"].append(1.0 + (40.0 if d.month == 10 and d.day == 15 else 0.0))
        d += dt.timedelta(days=1)
    monkeypatch.setattr(RD, "daily_era5", lambda *a, **k: grid)
    monkeypatch.setattr(RD, "daily_power", lambda *a, **k: grid)
    monkeypatch.setattr(RD, "daily_gauge", lambda la, lo, **k: _orig_gauge(la, lo, fetch=lambda u: _gauge_csv(big={"2015-12-02": 345.0})))
    out = RD.design(12.9, 80.2)
    assert out["available"] and out["d1_mm"] > 100.0
    assert "gauge" in out["source_used"]["1d"].lower()
    assert out["by_source"]["gauge"]["km"] == 3.0
    assert out["largest_in_record"]["1d"]["mm"] == pytest.approx(345.0)
    assert out["uplift_if_sources_disagree"]["1d"] > 1.5


_orig_gauge = RD.daily_gauge


def test_design_survives_a_failing_gauge(monkeypatch):
    grid = {"source": "grid", "t": [], "mm": []}
    d = dt.date(1991, 1, 1)
    while d <= dt.date(2020, 12, 31):
        grid["t"].append(d.isoformat()); grid["mm"].append(1.0 + (30.0 if d.month == 7 and d.day == 4 else 0.0)); d += dt.timedelta(days=1)
    monkeypatch.setattr(RD, "daily_era5", lambda *a, **k: grid)
    monkeypatch.setattr(RD, "daily_power", lambda *a, **k: grid)

    def boom(*a, **k):
        raise RuntimeError("no gauge within 60 km")
    monkeypatch.setattr(RD, "daily_gauge", boom)
    out = RD.design(52.0, 4.0)
    assert out["available"] and "gauge" in out["failed_sources"] and "gauge" not in out["by_source"]


def _bowl_dem(n=121, cell=40.0, depth=6.0):
    """A bowl: hill rim around a flat floor; the pin is on the floor."""
    y, x = np.mgrid[:n, :n]
    r = np.hypot(x - n // 2, y - n // 2) * cell
    z = 20.0 + np.clip((r - 600.0), 0, None) * 0.02 - 0.0
    return SimpleNamespace(elevations=z.astype(float), asset_row=n // 2, asset_col=n // 2, cell_size_m=cell, source="synthetic bowl")


def _rain(d1=300.0):
    return {"available": True, "rp": 100, "d1_mm": d1, "d3_mm": d1 * 1.5, "by_source": {"gauge": {"d1_mm": d1, "km": 4.0, "source": "gauge"}, "era5": {"d1_mm": d1 / 1.7, "source": "era5"}},
            "source_used": {"1d": "rain gauge Test", "3d": "rain gauge Test"}, "largest_in_record": {"1d": {"mm": d1, "date": "2015-12-02", "source": "g"}}}


def test_surface_water_bounds_and_floor_vs_rim():
    dem = _bowl_dem()
    pin = PV.site_surface_water(0.0, 0.0, rain=_rain(), dem=dem, radius_km=2.4, cell_cap=60, use_cache=False)
    assert pin["available"]
    assert pin["depth_m"] >= pin["depth_working_m"] >= 0.0          # drains overwhelmed can only be as deep or deeper
    assert pin["depth_m"] > 0.05 and pin["rating"] != "NEGLIGIBLE"  # water collects on the bowl floor
    assert pin["depth_near_max_m"] >= pin["depth_m"] >= 0.0 and pin["depth_pin_cell_m"] >= 0.0 and pin["window_m"] >= 100.0   # headline = window mean, never above the deepest cell
    assert "mean over the cells" in pin["depth_basis"]
    assert pin["rain"]["d1_mm"] == 300.0 and pin["rain"]["gauge_d1_mm"] == 300.0 and pin["rain"]["gauge_km"] == 4.0
    assert pin["rain"]["grid_best_d1_mm"] == pytest.approx(300.0 / 1.7, abs=0.2)
    hill = SimpleNamespace(**{**vars(dem), "asset_row": 5, "asset_col": 60})
    rim = PV.site_surface_water(0.0, 0.0, rain=_rain(), dem=hill, radius_km=2.4, cell_cap=60, use_cache=False)
    assert rim["depth_m"] < pin["depth_m"]


def test_surface_water_unavailable_rain_is_not_zero():
    r = PV.site_surface_water(0.0, 0.0, rain={"available": False, "reason": "no rainfall record could be read"}, dem=_bowl_dem(), use_cache=False)
    assert r["available"] is False and "rainfall" in r["reason"]


def test_surface_water_more_rain_is_not_shallower():
    dem = _bowl_dem()
    a = PV.site_surface_water(0.0, 0.0, rain=_rain(120.0), dem=dem, radius_km=2.4, cell_cap=60, bounds=False, use_cache=False)
    b = PV.site_surface_water(0.0, 0.0, rain=_rain(340.0), dem=dem, radius_km=2.4, cell_cap=60, bounds=False, use_cache=False)
    assert b["depth_m"] >= a["depth_m"] and b["depth_working_m"] is None


class _G:
    """Ground-like: z, dx, dy, ac, ar, rows, cols (north-up, pin at (ar, ac))."""
    def __init__(self, z, dx):
        self.z = np.asarray(z, float); self.rows, self.cols = self.z.shape; self.dx = self.dy = dx
        self.ac, self.ar = (self.cols - 1) / 2.0, (self.rows - 1) / 2.0


def _sw_field(level):
    zc = np.full((20, 20), 10.0); zc[8:12, 8:12] = 7.0                       # a 4x4 coarse depression
    hc = np.where(zc < 9, level - zc, 0.0); hc[hc < 0] = 0.0
    return {"available": True, "_z": zc, "_h_max": hc, "_asset": (10, 10), "_dx": 50.0, "cell_m": 50.0, "radius_km": 0.5, "mass_error": 0.0, "depth_m": 2.0}


def test_rain_field_downscales_water_surface_to_site_ground():
    z = np.full((40, 40), 10.0)
    z[15:25, 15:25] = 6.0                                                      # lower pit in the middle than the coarse cell mean (7 m)
    z[0:5, 0:5] = 12.0                                                        # high corner
    fld = H.rain_field(0.0, 0.0, _G(z, 10.0), sw=_sw_field(9.0))
    d = fld["depth"]
    assert d[20, 20] > 2.5                                                     # surface 9 m over a 6 m pit
    assert d[2, 2] == 0.0                                                      # above the surface, dry
    assert fld["wse_range"][0] == pytest.approx(9.0, abs=0.6)


def test_rain_field_none_when_no_run():
    assert H.rain_field(0.0, 0.0, _G(np.zeros((10, 10)), 5.0), sw={"available": False}) is None


def test_simulate_rain_shape_matches_solver_contract_and_barrier_does_not_help():
    z = np.full((40, 40), 10.0); z[15:25, 15:25] = 6.0
    g = _G(z, 10.0)
    comps = [{"id": "a", "x": 0.0, "y": 0.0, "footprint": [[-8, -8], [8, -8], [8, 8], [-8, 8]]}]
    fld = H.rain_field(0.0, 0.0, g, sw=_sw_field(9.0))
    sc = {"id": "rain_100", "kind": "rain", "rp": 100, "label": "rain"}
    s0 = H.simulate_rain(0.0, 0.0, g, comps, sc, _field=fld)
    s1 = H.simulate_rain(0.0, 0.0, g, comps, sc, barrier_m=3.0, barrier_radius_m=60.0, _field=fld)
    assert s0["dry"] is False and s0["comp"]["a"]["depth_at_ground_m"] > 1.0
    assert s0["comp"]["a"]["velocity_ms"] is None and s0["comp"]["a"]["arrival_min"] is None and s0["frames"] is None
    assert s1["comp"]["a"]["depth_at_ground_m"] == s0["comp"]["a"]["depth_at_ground_m"]
    assert s1["barrier_note"] and "trap" in s1["barrier_note"]
    assert len(s0["hmax_cm"]) == 40 * 40 and len(s0["block"]) == 40 * 40
    s2 = H.simulate_rain(0.0, 0.0, g, comps, sc, add_m=0.5, _field=fld)
    assert s2["comp"]["a"]["depth_at_ground_m"] == pytest.approx(s0["comp"]["a"]["depth_at_ground_m"] + 0.5, abs=0.02)


def test_simulate_rain_without_a_run_is_flagged_not_zero():
    g = _G(np.zeros((10, 10)), 5.0)
    # when the catchment run is unavailable the contract is `failed` set and dry True (never an invented depth)
    import cri.hydraulics as HH
    orig = HH.rain_field
    HH.rain_field = lambda *a, **k: None
    try:
        out = HH.simulate_rain(0.0, 0.0, g, [{"id": "a", "x": 0, "y": 0, "footprint": []}], {"id": "rain_100", "kind": "rain", "rp": 100})
    finally:
        HH.rain_field = orig
    assert out["dry"] is True and out.get("failed")


def test_twin_picks_rain_and_explains_it():
    sc = {"id": "rain_100", "kind": "rain", "rp": 100, "label": "Heavy rain", "depth_m": 0.3, "depth_p90_m": 1.0, "wet_share": 0.3, "q": [0.0] * 21,
          "rain": {"d1_mm": 345.0, "gauge_d1_mm": 345.0, "gauge_km": 4.0}, "radius_km": 6.0}
    tw = {"scenarios": {"flood": [sc, {"id": "river_100", "kind": "river", "rp": 100, "year": None, "depth_m": 1.0}]}}
    ids = [s["id"] for s in AT._key_flood(tw, solver=True)]
    assert ids == ["river_100", "rain_100"]
    txt = AT._flood_cause(sc, {}, True, ["c"], {"c": {"id": "c", "ground_m": 6.2}}, tw)
    assert "345" in txt and "rain gauge" in txt and "wall does not keep" in txt


def test_site_profile_surface_water_entry(monkeypatch):
    fake = {"available": True, "depth_m": 0.26, "depth_working_m": 0.25, "rating": "MODERATE", "depth_pin_cell_m": 0.01, "depth_near_max_m": 0.97, "window_m": 201, "cell_m": 66.9, "radius_km": 6.0, "builtup_share": 0.58,
            "rain": {"d1_mm": 324.9, "gauge_d1_mm": 324.9, "gauge_km": 4.0, "grid_best_d1_mm": 202.5, "largest_in_record": {"mm": 344.9, "date": "2015-12-02"}, "source_used": "gauge"}}
    monkeypatch.setattr(PV, "site_surface_water", lambda *a, **k: fake)
    e = SP._h_surface_water(SP.Ctx(12.98, 80.22, "Velachery"))
    assert e["status"] == "modelled" and e["severity"] == "MODERATE" and e["metrics"]["range"] == "0.25–0.26 m" and e["metrics"]["deepest_cell_nearby_m"] == 0.97
    assert "202" in e["method"] and "Mudichur" in e["caveat"]
    monkeypatch.setattr(PV, "site_surface_water", lambda *a, **k: {"available": False, "reason": "terrain unavailable"})
    n = SP._h_surface_water(SP.Ctx(12.98, 80.22, "Velachery"))
    assert n["status"] == "null" and n["severity"] is None and n["value"] is None


def test_river_low_rating_raised_by_terrain_second_opinion(monkeypatch):
    c = SP.Ctx(50.5, 7.0, "Ahrweiler")
    monkeypatch.setattr(SP, "_terrain_rank", lambda cc: ({"rank": 0.9, "hand_m_1.0": 1.2, "heldout_auc": 0.76}, {}))
    e = SP._river_modelled(c, "river_flood", "River flood", "flood", 0.0, 0.0, {"rps": [10, 100]}, [])
    assert e["severity"] == "MODERATE" and e["metrics"]["raised_from"] == "NEGLIGIBLE" and "Raised to MODERATE" in e["method"]
    monkeypatch.setattr(SP, "_terrain_rank", lambda cc: ({"rank": 0.2, "hand_m_1.0": 30.0, "heldout_auc": 0.76}, {}))
    e2 = SP._river_modelled(SP.Ctx(1.0, 1.0, "hill"), "river_flood", "River flood", "flood", 0.0, 0.0, {"rps": [10]}, [])
    assert e2["severity"] == "NEGLIGIBLE" and "raised_from" not in e2["metrics"]
    e3 = SP._river_modelled(SP.Ctx(2.0, 2.0, "deep"), "river_flood", "River flood", "flood", 1.5, 0.9, {"rps": [10]}, [])
    assert e3["severity"] == "HIGH"                                        # a mapped depth is never lowered or recomputed


def test_sensitivity_reports_rain_response_and_saturation(monkeypatch):
    # the variants run through the public function with a patched site_surface_water, for speed; the stand-in fills to a 1 m spill height
    calls = []

    def fake(lat, lon, rp=100, rain=None, rain_scale=1.0, cn=75.0, manning=0.05, bounds=False, drainage_mm_h=12.0, **k):
        calls.append(rain_scale)
        d = min(1.0, 0.01 * (rain or _rain())["d1_mm"] * rain_scale)          # fills to a 1 m spill height
        return {"available": True, "depth_m": d, "depth_working_m": d * 0.95 if bounds else None, "depth_near_max_m": d, "depth_pin_cell_m": d}
    monkeypatch.setattr(PV, "site_surface_water", fake)
    out = PV.surface_water_sensitivity(0.0, 0.0, rain=_rain(300.0), rain_for_rp=lambda r: _rain(300.0))
    assert out["available"] and out["base_m"] == 1.0
    assert out["saturates_at_rain_mm"] is not None and out["saturates_at_rain_mm"] < 0.7 * 300.0
    assert "terrain" in out["text"] and "ponding flag" in out["text"]
    assert [x["rain_mm"] for x in out["rain_response"]] == sorted(x["rain_mm"] for x in out["rain_response"])
    assert any(r["input"] == "curve_number" for r in out["ranking"]) and all(r["input"] != "rain_response" for r in out["ranking"])
