"""Physical Risk Assessment report: content, no financial figures, sector grouping, delivery formats."""
import json
import os

import pytest

os.environ.setdefault("CRI_API_KEYS", "")
from fastapi.testclient import TestClient  # noqa: E402

from cri import physical_report as PR  # noqa: E402
from cri.api.main import app  # noqa: E402

client = TestClient(app)
REAL = PR.build_asset_report


@pytest.fixture(autouse=True)
def _offline_heat_and_coast(monkeypatch):
    """These tests inject the site assessment; the heat record and sea-distance reads must not touch the network either."""
    from cri.finance import heat_hazard
    monkeypatch.setattr(heat_hazard, "site_heat", lambda lat, lon, fetch=None: {
        "available": True, "rating": "MODERATE", "days_ge_35": 20.0, "days_ge_40": 2.0, "annual_max_mean_c": 38.0, "record_max_c": 41.0, "years": "1994-2023", "n_years": 30})
    from cri.finance import seismic_hazard
    monkeypatch.setattr(seismic_hazard, "site_seismic", lambda lat, lon, vs30=760, fetch=None: {"available": True, "method": "seismicity screen (recorded earthquakes)",
        "m55_within_150km_since_1973": 0, "rate_per_year": 0.0, "rating": "NEGLIGIBLE", "source": "test catalogue"})
    from cri.terrain import lidar_flood
    monkeypatch.setattr(lidar_flood, "flood_position", lambda lat, lon, radius_km=2.5: {"available": False, "reason": "test: LiDAR not used"})
    from cri.finance import pluvial
    monkeypatch.setattr(pluvial, "site_pluvial", lambda *a, **k: {"available": False, "reason": "test"})
    from cri.finance import fire_behaviour
    monkeypatch.setattr(fire_behaviour, "site_fire_behaviour", lambda *a, **k: {"available": False, "reason": "test"})
    from cri.modellab import slr
    monkeypatch.setattr(slr, "coast_info", lambda *a, **k: {"elevation_m": 4.0, "sea_km": None, "sea_bearing": None})


def stub_ai(tc="MODERATE", flood="HIGH"):
    ev = lambda m, v, u="": {"metric": m, "value": v, "unit": u, "source": "test"}      # noqa: E731
    return {
        "terrain": {"elevation_m": 4.0, "terrain_position": "flat", "slope_at_asset_deg": 0.5, "upslope_area_km2": 2.0,
                    "elevation_head_m": 6.0},
        "satellite": {"nearest_water_m": 120.0, "scene_date": "2025-11-01"},
        "land_cover": {"available": True, "class_at_asset": "built_up", "built_up_within_250m": 0.6,
                       "tree_cover_within_1km": 0.1},
        "radar": {}, "errors": {}, "sources": ["test source"],
        "findings": [
            {"hazard": "flood", "severity": flood, "score": 4, "mechanisms": ["pluvial"], "evidence": [ev("Built-up", 60, "%")],
             "actions": ["Audit on-site drainage capacity."]},
            {"hazard": "tropical_cyclone", "severity": tc, "score": 3, "mechanisms": ["cyclone_wind"],
             "evidence": [ev("1-in-50-year wind", "45 m/s — Cat 2")], "actions": ["Verify roof fixings."]},
            {"hazard": "wildfire", "severity": "NEGLIGIBLE", "evidence": [ev("Tree cover", 5, "%")], "actions": []},
            {"hazard": "landslide", "severity": "NEGLIGIBLE", "evidence": [], "actions": []},
            {"hazard": "debris_flow", "severity": "NEGLIGIBLE", "evidence": [], "actions": []},
            {"hazard": "urban_heat", "severity": "LOW", "evidence": [ev("Built-up", 60, "%")], "actions": []},
        ]}


FLOOD = {"river": {"available": True, "footprint_on_water": False, "source": "JRC", "rps": [10, 100, 500],
                   "depths": {"10": [0.0, 0.0], "100": [1.2, 1.4], "500": [2.5, 2.7]}},
         "coastal": {"available": False, "reason": "site above 30 m — coastal flooding not assessed"}}


def build(year=1980, sector="cement", **kw):
    return PR.build_asset_report(22.3569, 91.81, "Test plant", sector, "integrated dry", year,
                                 include_image=False, _assess=stub_ai(**kw), _flood=FLOOD)


def test_depth_rating_thresholds():
    assert PR._depth_rating(0.0, False) == "NEGLIGIBLE"
    assert PR._depth_rating(0.2, False) == "LOW"
    assert PR._depth_rating(0.6, False) == "MODERATE"
    assert PR._depth_rating(1.3, False) == "HIGH"
    assert PR._depth_rating(2.5, False) == "CRITICAL"
    assert PR._depth_rating(1.3, True) == "MODERATE"          # defended to 1-in-100 or better → one level lower


def test_site_zone_age_issues_and_measures_present():
    r = build()
    assert "35" not in r["site"]["summary"] and "4 m above sea level" in r["site"]["summary"]
    river = next(z for z in r["zones"] if z["key"] == "river_flood")
    assert "1-in-100" in river["zone"] and "1.3 m" in river["zone"] and river["table"]["rows"]
    assert r["age"]["age_2050"] == 70 and "beyond the upper end" in r["age"]["statement"]
    assert r["age"]["design_life_years"]["typical"] == 40
    hazards = {i["hazard"] for i in r["issues"]}
    assert {"river_flood", "surface_water", "cyclone"} <= hazards
    river_issue = next(i for i in r["issues"] if i["hazard"] == "river_flood")
    assert "kiln" in river_issue["components_at_risk"] and "1.3 m" in river_issue["what_will_happen"][0]
    assert river_issue["age_note"]                          # 46 years old → ageing mechanism applies
    assert r["measures"] and r["measures"][0]["when"] == "Immediately"
    assert any("1.8 m" in a for m in r["measures"] for a in m["actions"])      # 1.3 + 0.5 freeboard


def test_no_age_means_no_age_notes_and_prompt():
    r = build(year=None)
    assert "not provided" in r["age"]["statement"]
    assert all(i["age_note"] is None for i in r["issues"])


def test_young_asset_gets_no_ageing_note():
    r = build(year=2022)
    assert all(i["age_note"] is None for i in r["issues"])
    assert "within the typical design life" in r["age"]["statement"]


def test_no_financial_figures_anywhere():
    r = build()
    text = json.dumps(r).lower() + PR.render_html(r).lower()
    for bad in ("$", "usd", "ebitda", "npv", "% of value", "annual loss", "asset value", "damage ratio"):
        assert bad not in text, bad


def test_sector_profiles_change_the_consequences():
    assert PR.profile_key("cement", "") == "cement" and PR.profile_key("real estate", "office") == "real_estate"
    assert PR.profile_key("electricity-generation", "coal") == "power" and PR.profile_key("x", "y") == "generic"
    cement, building = build(sector="cement"), build(sector="real estate")
    c = next(i for i in cement["issues"] if i["hazard"] == "river_flood")["what_will_happen"][-1]
    b = next(i for i in building["issues"] if i["hazard"] == "river_flood")["what_will_happen"][-1]
    assert "kiln" in c and "basement" in b and c != b


def test_overall_rating_ignores_water_stress_context():
    r = build(tc="NEGLIGIBLE", flood="NEGLIGIBLE")
    assert r["overall_rating"] in ("HIGH", "MODERATE")        # river flood from the stub still drives it


def test_portfolio_groups_by_sector_and_renders(monkeypatch):
    def fake(lat, lon, name, sector, atype, year, include_image=True):
        return REAL(lat, lon, name, sector, atype, year, include_image=False,
                    _assess=stub_ai(), _flood=FLOOD)
    monkeypatch.setattr(PR, "build_asset_report", fake)
    rep = PR.build_portfolio_report([{"lat": 22.3, "lon": 91.8, "name": "A", "sector": "cement", "commissioned_year": 1990},
                                     {"lat": 22.4, "lon": 91.9, "name": "B", "sector": "cement"},
                                     {"lat": 22.5, "lon": 91.7, "name": "C", "sector": "real estate"}], client="ACME")
    assert {b["sector"] for b in rep["sectors"]} == {"cement", "real_estate"}
    cem = next(b for b in rep["sectors"] if b["sector"] == "cement")
    assert cem["count"] == 2 and "River flooding" in cem["high_hazards"]
    html = PR.render_html(rep)
    assert "Summary by sector" in html and "ACME" in html and "<img" not in html
    assert PR.render_pdf(rep)[:4] == b"%PDF"


def test_failed_asset_is_reported_not_fatal(monkeypatch):
    def fake(lat, lon, *a, **k):
        if lat > 50:
            raise RuntimeError("boom")
        return REAL(lat, lon, "ok", "cement", None, None, include_image=False, _assess=stub_ai(), _flood=FLOOD)
    monkeypatch.setattr(PR, "build_asset_report", fake)
    rep = PR.build_portfolio_report([{"lat": 22.3, "lon": 91.8}, {"lat": 60.0, "lon": 10.0, "name": "bad"}])
    assert rep["n_assets"] == 1 and len(rep["failed"]) == 1 and "boom" in rep["failed"][0]


def test_api_formats_and_validation(monkeypatch):
    def fake(lat, lon, name, sector, atype, year, include_image=True):
        return REAL(lat, lon, name, sector, atype, year, include_image=False,
                    _assess=stub_ai(), _flood=FLOOD)
    monkeypatch.setattr(PR, "build_asset_report", fake)
    body = {"assets": [{"lat": 22.3, "lon": 91.8, "name": "Plant 1", "sector": "cement", "commissioned_year": 1994}], "client": "ACME"}
    j = client.post("/reports/physical-risk", json=body)
    assert j.status_code == 200 and j.json()["assets"][0]["asset"]["name"] == "Plant 1"
    h = client.post("/reports/physical-risk.html", json=body)
    assert h.status_code == 200 and "How the site sits" in h.text
    p = client.post("/reports/physical-risk.pdf", json=body)
    assert p.status_code == 200 and p.content[:4] == b"%PDF"
    assert client.post("/reports/physical-risk", json={"assets": []}).status_code == 422
    assert client.post("/reports/physical-risk", json={"assets": [{"lat": 95, "lon": 0}]}).status_code == 422


def _dry_low_coast_report(monkeypatch, coast_info):
    from cri.modellab import slr
    monkeypatch.setattr(slr, "coast_info", coast_info)
    ai = stub_ai()
    ai["terrain"]["elevation_m"] = 2.0
    flood = {"river": FLOOD["river"], "coastal": {"available": True, "rps": [10, 100], "years": [2018, 2050],
             "depths": {"2018_10": [0.0], "2018_100": [0.0], "2050_10": [0.0], "2050_100": [0.0]}}}
    return PR.build_asset_report(25.77, -80.19, "Tower", "real estate", "office", 2005, include_image=False, _assess=ai, _flood=flood)


def test_low_lying_urban_site_with_no_surge_gets_a_verify_caution(monkeypatch):
    """Sea distance unknown (tile read fails): the rating is left as the map says, but the report must say it is unconfirmed."""
    def boom(*a, **k):
        raise OSError("no tiles")
    r = _dry_low_coast_report(monkeypatch, boom)
    assert r["cautions"] and "unconfirmed" in r["cautions"][0]
    assert "Verify:" in PR.render_html(r)


def test_dry_surge_map_on_very_low_ground_next_to_the_sea_is_floored_not_cleared(monkeypatch):
    r = _dry_low_coast_report(monkeypatch, lambda *a, **k: {"elevation_m": 2.0, "sea_km": 0.0, "sea_bearing": 90})
    z = next(z for z in r["zones"] if z["key"] == "coastal_flood")
    assert z["rating"] == "MODERATE" and z["zone"].startswith("Elevation screen, not a modelled flood depth")
    assert z["rating_basis"] == "elevation screen floor"


def test_elevation_screen_ignores_inland_ground_far_from_the_sea(monkeypatch):
    r = _dry_low_coast_report(monkeypatch, lambda *a, **k: {"elevation_m": 2.0, "sea_km": 40.0, "sea_bearing": 90})
    z = next(z for z in r["zones"] if z["key"] == "coastal_flood")
    assert z["rating"] == "NEGLIGIBLE" and "rating_basis" not in z


def test_dashboard_served():
    r = client.get("/reports/physical-risk/dashboard")
    assert r.status_code == 200 and "Physical Risk Report" in r.text and "no financial figures" in r.text.lower()


def test_water_zone_text_names_what_drives_the_rating(monkeypatch):
    """Regression: a HIGH rating (from drought risk) sat beside text saying catchment water risk was Low."""
    from cri.finance import analyst
    monkeypatch.setattr(analyst, "_water", lambda lat, lon, sector: {
        "admin1": "Test", "overall": {"label": "Low (0-1)", "score": 0.5},
        "water_stress": {"label": "Low (<10%)", "score": 0.5}, "drought": {"label": "Medium - High (0.6-0.8)", "score": 3.0},
        "groundwater_decline": {"label": "Insignificant Trend", "score": 0}, "water_stress_path_bau": {}})
    r = build(tc="NEGLIGIBLE", flood="NEGLIGIBLE")
    wz = next(z for z in r["zones"] if z["key"] == "water")
    assert wz["rating"] == "HIGH"
    assert "Baseline water stress is Low" in wz["zone"] and "drought risk is Medium - High" in wz["zone"]
    assert "follows the higher of the two (drought risk)" in wz["zone"]
    assert "high drought risk" in r["headline"] and "high water stress" not in r["headline"]
