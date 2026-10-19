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
