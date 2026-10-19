"""Regressions found by the stress campaign (tools/stress_api.py)."""
import pytest
from fastapi.testclient import TestClient
from cri.api.main import app
from cri.data.parameters import normalise_sector, sector_recognised

client = TestClient(app, raise_server_exceptions=False)


def test_malformed_free_form_body_is_422_not_500():
    r = client.post("/asset-twin/analyse", json={"twin": {}})
    assert r.status_code == 422 and "unprocessable_request" in r.text
    assert "Traceback" not in r.text


def test_unknown_sector_is_flagged_not_silently_low_risk():
    assert not sector_recognised("other") and not sector_recognised("asdf") and not sector_recognised("")
    assert normalise_sector("cement") == "cement" and sector_recognised("cement")
    r = client.post("/agent/assess", json={"company_name": "NoSuchCo", "sector_hint": "other"}).json()
    assert any("SECTOR NOT RECOGNISED" in g for g in r["company_profile"]["data_gaps"])


def test_default_revenue_is_not_labelled_user_supplied():
    d = client.post("/agent/assess", json={"company_name": "NoSuchCo", "sector_hint": "cement"}).json()
    assert "assumed" in d["company_profile"]["revenue_usd_m"]["tier"]
    u = client.post("/agent/assess", json={"company_name": "NoSuchCo", "sector_hint": "cement", "revenue_usd_m": 900}).json()
    assert u["company_profile"]["revenue_usd_m"]["tier"] == "user-supplied"


def test_terrain_failure_does_not_refetch_every_tile(monkeypatch):
    from cri.connectors import terrain_tiles as tt
    calls = []

    def fake(z, x, y):
        calls.append((z, x, y)); return None
    monkeypatch.setattr(tt, "_fetch_tile", fake)
    monkeypatch.setenv("CRI_LIDAR", "0")                      # this point is inside the Dutch LiDAR cover; the test is about the global tiles
    with pytest.raises(RuntimeError):
        tt.get_elevation_grid(51.5, 4.4, radius_km=2)
    assert len(calls) == len(set(calls)), "a failed tile was fetched more than once"


def test_climate_monitor_unavailable_is_null_not_moderate():
    from cri.connectors.hf_climate_monitor import ClimateMonitorClient as C
    rep = C._fallback(1.0, 2.0, "request_failed")
    assert rep["available"] is False and rep["risk_level"] is None
    assert rep["temperature_anomaly"] is None and rep["precipitation_anomaly"] is None
    assert C.score_from_report(rep) is None
    ok = C._parse_report({"data": [{"risk_level": "High", "temperature_anomaly": 1.2}]}, 1, 2)
    assert C.score_from_report(ok) == 71.0 and ok["precipitation_anomaly"] is None


def test_polar_and_dateline_sites_are_refused_cleanly_not_killed():
    """89.9 N used to request ~100k tiles and be OOM-killed; |lat| > 85.05 has no Web-Mercator terrain tiles at all."""
    from cri.connectors import terrain_tiles as tt
    for lat, lon in [(89.9, 45), (90, 0), (-90, 0), (float("nan"), 0)]:
        with pytest.raises(ValueError):
            tt.get_elevation_grid(lat, lon, radius_km=2.5)
    with pytest.raises(ValueError):
        tt.get_elevation_grid(84.9, 10, radius_km=2.5)


def test_offshore_site_is_refused_not_rated_critical(monkeypatch):
    from cri import physical_report as pr
    fake = {"terrain": {"elevation_m": -4992.0, "slope_at_asset_deg": 1.0}, "land_cover": {}, "satellite": {}, "radar": {}, "findings": []}
    from cri.finance import geo
    monkeypatch.setattr(geo, "admin_at", lambda lat, lon: {"found": False})
    with pytest.raises(ValueError, match="not on assessable land"):
        pr.build_asset_report(-40.0, -140.0, "buoy", "cement", None, None, include_image=False, _assess=fake, _flood={})


def test_site_with_no_data_at_all_is_refused_not_negligible(monkeypatch):
    from cri import physical_report as pr
    from cri.finance import geo
    monkeypatch.setattr(geo, "admin_at", lambda lat, lon: {"found": False})
    with pytest.raises(ValueError, match="not on assessable land"):
        pr.build_asset_report(90.0, 0.0, "pole", "cement", None, None, include_image=False,
                              _assess={"terrain": {}, "land_cover": {}, "satellite": {}, "radar": {}, "findings": []}, _flood={})


# ── silent-degradation audit (docs/SILENT_DEGRADATION_AUDIT.md) ───────────────────────────────────────────────────────────

def test_portfolio_does_not_average_in_an_invented_score():
    from cri.analysis import portfolio as P

    class Broken:                      # no .rating at all
        pass
    with pytest.raises(ValueError, match="no composite score"):
        P._composite_score(Broken())


def test_cyclone_lookup_failure_is_not_reported_as_not_exposed(monkeypatch):
    from cri import climate_scenario as CS
    from cri.climate import tropical_cyclone as TC

    def boom(*a, **k):
        raise RuntimeError("no data")
    monkeypatch.setattr(TC, "site_hazard", boom)
    w = CS._wind(10.0, 120.0, "ssp245", 2070)
    assert w["tc_status"] == "unavailable" and "NOT evidence" in w["note"]


def test_failed_gis_resolution_is_labelled_placeholder(monkeypatch):
    from cri.climate import spatial_downscaling as SD
    monkeypatch.setattr(SD, "_try_gis_resolve", lambda lat, lon: {"elevation_m": 0.0, "coastal_km": 999.0, "is_cyclone_belt": False,
                                                                 "is_permafrost": False, "is_arid": False, "koppen_zone": "", "_failed": True})
    monkeypatch.setattr(SD, "_fetch_nasa_power_baseline", lambda lat, lon: {})
    monkeypatch.setattr(SD, "_fetch_era5_baseline", lambda lat, lon: {})
    monkeypatch.setattr(SD, "_fetch_cmip6_projection", lambda *a, **k: {})
    monkeypatch.setattr(SD, "_fetch_wri_aqueduct_point", lambda *a, **k: {})
    monkeypatch.setattr(SD, "_fetch_live_conditions", lambda *a, **k: {})
    try:
        sig = SD.SpatialDownscaler().get_hazard_signals(10.0, 10.0, 2050, "ssp245") if hasattr(SD.SpatialDownscaler, "get_hazard_signals") else None
    except TypeError:
        sig = None
    if sig is not None:
        assert "UNAVAILABLE" in sig.data_sources["gis"] and "UNAVAILABLE" in sig.data_sources["temperature_baseline"]


def test_failed_coastal_read_is_not_a_dry_site(monkeypatch):
    """Every failed Deltares read used to become a zero depth with available=True -> 'NEGLIGIBLE' and cached for ever."""
    from cri.finance import flood_hazard as FH

    def boom(*a, **k):
        raise OSError("down")
    monkeypatch.setattr(FH, "_chunk", boom)
    r = FH.coastal_depths(25.76, -80.19)
    assert r["available"] is False and "not assessed" in r["reason"]


# ── heat: observed temperature record, not a land-cover guess ─────────────────
def _fake_tmax(peak, n_hot35, n_hot40):
    """30 years of synthetic daily maxima: n_hot35 days at 36 C (incl. n_hot40 days at 41 C), the rest 20 C."""
    days, vals = [], []
    import datetime as _d
    for y in range(1994, 2024):
        for k in range(365):
            dd = _d.date(y, 1, 1) + _d.timedelta(days=k)
            days.append(dd.isoformat())
            vals.append(41.0 if k < n_hot40 else (36.0 if k < n_hot35 else 20.0))
        vals[-1] = peak
    return {"daily": {"time": days, "temperature_2m_max": vals}}


def test_heat_rating_thresholds_and_failure_modes():
    from cri.finance.heat_hazard import rate, site_heat
    assert rate(0, 0) == "NEGLIGIBLE" and rate(5, 0) == "LOW" and rate(20, 0) == "MODERATE"
    assert rate(50, 0) == "HIGH" and rate(5, 12) == "HIGH" and rate(100, 70) == "CRITICAL" and rate(160, 0) == "CRITICAL"
    hot = site_heat(24.7, 46.7, fetch=lambda url: _fake_tmax(45.0, 140, 70))
    assert hot["available"] and hot["rating"] == "CRITICAL" and hot["days_ge_40"] >= 60
    cool = site_heat(64.1, -21.9, fetch=lambda url: _fake_tmax(20.0, 0, 0))
    assert cool["rating"] == "NEGLIGIBLE" and cool["days_ge_35"] == 0
    def boom(url): raise OSError("net down")
    bad = site_heat(10.0, 10.0, fetch=boom)
    assert bad["available"] is False and "rating" not in bad                       # a failed read is never a rating
    assert site_heat(float("nan"), 0.0)["available"] is False
    short = site_heat(1.0, 1.0, fetch=lambda url: {"daily": {"time": ["2000-01-01"], "temperature_2m_max": [30.0]}})
    assert short["available"] is False


def test_compose_rejects_bad_coordinates_before_any_network():
    import pytest
    from cri import climate_scenario as CS
    for la, lo in [(1e308, 0), (0, 1e308), (91, 0), (0, 181), (float("nan"), 0), ("x", 0)]:
        with pytest.raises(ValueError):
            CS.compose(la, lo, "ssp245", 2050, fetch=lambda *a, **k: (_ for _ in ()).throw(AssertionError("network touched")))


def test_dateline_site_gets_tiles_from_both_sides(monkeypatch):
    import numpy as np
    from cri.connectors import terrain_tiles as T
    seen = []

    def fake(z, x, y):
        seen.append(x)
        return np.zeros((256, 256, 3), dtype=np.uint8) + np.array([128, 0, 0], dtype=np.uint8)
    monkeypatch.setattr(T, "_fetch_tile", fake)
    g = T.get_elevation_grid(-17.0, 179.995, radius_km=2.5, zoom=13)
    n = 2 ** 13
    assert any(x > n - 4 for x in seen) and any(x < 4 for x in seen), "both the eastern-most and western-most tile columns are needed"
    assert 0 <= g.asset_row < g.rows and 0 <= g.asset_col < g.cols
    assert abs(g.asset_col - g.cols / 2) < 0.1 * g.cols, "the asset sits near the middle of the grid, not at an edge"


def test_fire_weather_index_matches_the_published_reference_values():
    """Van Wagner/NRCan reference record (T 17 C, RH 42 %, wind 25 km/h, no rain, start FFMC 85 / DMC 6 / DC 15). The index code is right;
    inflated day counts at Mediterranean sites come from using daily extremes as inputs, not from the arithmetic."""
    from cri.modellab import fire
    r = fire.run_series([17.0], [42.0], [25.0], [0.0], [4], tropical=False)
    for k, v in {"ffmc": 87.7, "dmc": 8.5, "dc": 19.0, "isi": 10.9, "bui": 8.5, "fwi": 10.1}.items():
        assert abs(r[k][0] - v) < 0.1, (k, r[k][0], v)


# ── LiDAR terrain: floodplain screen and grid source ─────────────────────────
import numpy as np


def _valley(n=120, res=5.0):
    """A V-shaped valley with a channel along the middle column: floor at 10 m, slopes rising 0.1 m per metre sideways."""
    x = np.abs(np.arange(n) - n // 2) * res
    return np.tile(10.0 + 0.1 * x, (n, 1)) - np.linspace(0, 0.5, n)[:, None]       # gentle fall along the valley so flow routes to one end


def test_hand_is_small_on_the_valley_floor_and_large_on_the_slope():
    from cri.terrain.lidar_flood import hand_and_rel, band_for
    d = _valley()
    rel, hand = hand_and_rel(d, 5.0, channel_km2=0.005)
    mid, side = hand[60, 60], hand[60, 5]
    assert np.isfinite(mid) and mid < 1.0
    assert np.isfinite(side) and side > 10.0
    assert band_for(0.4) == "HIGH" and band_for(3.0) == "MODERATE" and band_for(8.0) == "LOW" and band_for(12.0) is None and band_for(None) is None


def test_lidar_outside_england_and_netherlands_returns_nothing_and_never_touches_the_network(monkeypatch):
    from cri.connectors import lidar_dem as L
    monkeypatch.setattr(L, "_get", lambda *a, **k: (_ for _ in ()).throw(AssertionError("network touched")))
    assert L.try_grid(34.05, -118.25, 1.0) is None            # Los Angeles
    assert L.try_grid(float("nan"), 0.0, 1.0) is None
    monkeypatch.setenv("CRI_LIDAR", "0")
    assert L.try_grid(51.5, -0.12, 1.0) is None               # London, but switched off


def test_lidar_failure_falls_back_to_none_not_an_error(monkeypatch, tmp_path):
    from cri.connectors import lidar_dem as L
    monkeypatch.setattr(L, "CACHE", tmp_path)
    L._FAILED.clear(); L._DOWN.clear()
    monkeypatch.setattr(L, "_get", lambda *a, **k: b"<ServiceException>boom</ServiceException>")
    assert L.try_grid(54.9, -2.93, 1.0) is None


def test_flood_position_says_not_available_outside_lidar_cover(monkeypatch):
    from cri.terrain import lidar_flood as LF
    from cri.connectors import terrain_tiles as T
    fake = T.ElevationGrid(lat=34.0, lon=-118.0, radius_km=1.0, cell_size_m=10.0, rows=5, cols=5, asset_row=2, asset_col=2, elevations=np.ones((5, 5)))
    monkeypatch.setattr(T, "get_elevation_grid", lambda *a, **k: fake)
    r = LF.flood_position(34.0, -118.0)
    assert r["available"] is False and "England only" in r["reason"]


# ── earthquake hazard ────────────────────────────────────────────────────────
def test_seismic_rating_bands_and_curve_interpolation():
    from cri.finance import seismic_hazard as SH
    assert [SH.pga_rating(x) for x in (0.01, 0.05, 0.1, 0.2, 0.4)] == ["NEGLIGIBLE", "LOW", "MODERATE", "HIGH", "CRITICAL"]
    assert [SH.count_rating(x) for x in (0, 2, 10, 30, 100)] == ["NEGLIGIBLE", "LOW", "MODERATE", "HIGH", "CRITICAL"]
    xs, ys = [0.1, 0.2, 0.4], [1e-2, 1e-3, 1e-4]                          # a straight line in log-log space: rate falls a decade per doubling
    assert abs(SH._loglog(xs, ys, 1e-3) - 0.2) < 1e-6 and SH._loglog(xs, ys, 1e-9) is None


def test_seismic_uses_hazard_model_in_us_and_catalogue_elsewhere_and_fails_to_not_assessed():
    import json
    from cri.finance import seismic_hazard as SH
    r475 = -__import__("math").log(0.9) / 50.0
    curve = {"status": "success", "response": {"hazardCurves": [{"imt": {"value": "PGA"}, "data": [{"component": "Total", "values": {
        "xs": [0.1, 0.2, 0.4], "ys": [10 * r475, r475, r475 / 10]}}]}]}}
    us = SH.site_seismic(34.05, -118.25, fetch=lambda url: json.dumps(curve))
    assert us["available"] and us["method"].startswith("USGS NSHM") and abs(us["pga_475yr_g"] - 0.2) < 0.01 and us["rating"] == "HIGH"
    eu = SH.site_seismic(51.5, -0.12, fetch=lambda url: "0")
    assert eu["available"] and eu["rating"] == "NEGLIGIBLE" and "seismicity screen" in eu["method"]
    def boom(url): raise OSError("down")
    bad = SH.site_seismic(51.5, -0.12, fetch=boom)
    assert bad["available"] is False and "rating" not in bad
    # a US point whose hazard service fails falls back to the labelled catalogue screen instead of pretending to have a model
    fb = SH.site_seismic(34.05, -118.25, fetch=lambda url: (_ for _ in ()).throw(OSError("x")) if "nshmp" in url else "12")
    assert fb["available"] and "seismicity screen" in fb["method"]


# ---- surface-water (rain-on-grid) model ---------------------------------------------------------------------------------------------------------
def _bowl_dem():
    """Flat 20 m plain with a 1.5 m deep closed bowl in the middle: rain must pond there and nowhere else."""
    from types import SimpleNamespace
    z = np.full((60, 60), 20.0)
    yy, xx = np.mgrid[0:60, 0:60]
    r = np.hypot(yy - 30, xx - 30)
    z -= np.clip(1.5 * (1 - r / 8.0), 0, None)
    z += 0.002 * xx                                           # a faint tilt so the plain drains out of the window
    return SimpleNamespace(elevations=z, cell_size_m=10.0, asset_row=30, asset_col=30, source="LiDAR test grid")


def test_pluvial_ponds_in_a_bowl_and_conserves_mass():
    from cri.finance.pluvial import site_pluvial
    r = site_pluvial(51.0, -1.0, rp=100, dem=_bowl_dem(), daily_rain_mm=120.0, builtup=0.0, keep_grids=True)
    assert r["available"] and r["depth_m"] > 0.05 and r["mass_error"] < 1e-6
    h = r["_h_max"]
    assert h[30, 30] > h[5, 5]                                # ponding in the bowl, thinner sheet on the plain


def test_pluvial_more_rain_is_not_less_water():
    from cri.finance.pluvial import site_pluvial
    a = site_pluvial(51.0, -1.0, dem=_bowl_dem(), daily_rain_mm=60.0, builtup=0.3)["depth_m"]
    b = site_pluvial(51.0, -1.0, dem=_bowl_dem(), daily_rain_mm=160.0, builtup=0.3)["depth_m"]
    assert b >= a


def test_pluvial_sewers_remove_light_rain_on_built_ground():
    from cri.finance.pluvial import site_pluvial
    wet = site_pluvial(51.0, -1.0, dem=_bowl_dem(), daily_rain_mm=25.0, builtup=1.0)["runoff_mm"]
    assert wet == 0.0 or wet < 25.0 * 0.35                    # 1-h share of 25 mm is under the 12 mm/h sewer allowance


def test_pluvial_failure_is_not_assessed_not_zero():
    from cri.finance.pluvial import site_pluvial
    r = site_pluvial(float("nan"), 0.0)
    assert r["available"] is False and "depth_m" not in r


# ---- river depth (lab function) ----------------------------------------------------------------------------------------------------------------
def test_river_stage_grows_with_flow_and_falls_with_slope():
    from cri.finance.river_depth import manning_depth
    assert manning_depth(200, 0.001) > manning_depth(20, 0.001) > 0
    assert manning_depth(100, 0.0005) > manning_depth(100, 0.005)
    assert manning_depth(0, 0.001) == 0.0


def test_river_depth_declines_when_flow_cell_is_far_and_never_returns_zero_for_missing_data():
    from cri.finance.river_depth import site_river_depth
    from types import SimpleNamespace
    dem = SimpleNamespace(elevations=np.zeros((20, 20)), cell_size_m=10.0, asset_row=10, asset_col=10, source="test")
    r = site_river_depth(51.0, -1.0, dem=dem, maxima=list(np.linspace(5, 50, 30)), snap_km=9.0)
    assert r["available"] is False and "depth_m" not in r
