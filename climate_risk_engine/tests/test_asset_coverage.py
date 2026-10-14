"""Water-stress finding, hazard coverage table and honest heat labelling in the Asset Intelligence output."""
from __future__ import annotations

from cri.analysis import asset_intelligence as ai
from cri.resident.brief import site_brief

HOME = (9.4712866, 77.9106587)       # Virudhunagar district, Tamil Nadu: not a wildfire place, but water-stressed


def test_home_site_water_stress_is_critical_and_basin_scale():
    f = ai._water_stress(*HOME)
    assert f["hazard"] == "water_stress" and f["severity"] == "CRITICAL"
    assert f["detail"]["indicator_score"] >= 4
    assert ai._BASIS["water_stress"] == "basin_screening"
    assert "basin" in f["caveat"].lower() and "not directly validated" in f["caveat"]
    assert any("business-as-usual" in e["metric"] for e in f["evidence"])
    assert f["mechanisms"] == ["water_stress"] and ai._ACTIONS["water_stress"]


def test_open_ocean_is_not_assessed_not_low():
    assert ai._water_stress(0.0, -30.0) is None
    cov = ai._coverage([], {}, water_found=False)
    ws = [c for c in cov if c["hazard"] == "water_stress"][0]
    assert ws["status"] == "not_assessed"


def test_failed_layer_is_missing_not_low():
    flood = {"hazard": "flood", "severity": "NEGLIGIBLE", "basis": "modelled"}
    cov = ai._coverage([flood], {"terrain": "boom"}, water_found=False)
    by = {c["hazard"]: c for c in cov}
    assert by["flood"]["status"] == "missing"
    assert "not the same as low risk" in by["flood"]["note"]


def test_unmodelled_heat_and_groundwater_are_listed_as_not_assessed():
    by = {c["hazard"]: c for c in ai._coverage([], {}, water_found=True)}
    for k in ("heat_stress", "surface_urban_heat", "groundwater_site"):
        assert by[k]["status"] == "not_assessed"


def test_heat_finding_says_it_is_land_cover_screening_only():
    f = ai._urban_heat(9.47, {"within_1km": {"built_up": 0.05}, "tree_cover_within_1km": 0.3})
    assert f["severity"] in ("NEGLIGIBLE", "LOW")
    assert "does NOT mean the place is not hot" in f["caveat"]
    assert any("not modelled" in str(e["value"]) for e in f["evidence"])
    assert ai._BASIS["urban_heat"] == "proxy"


def test_headline_without_material_hazard_is_not_an_all_clear():
    h = ai._headline("Site", {}, {}, {}, [])
    assert "not an all-clear" in h


def test_water_stress_leads_headline_when_top_material():
    f = ai._water_stress(*HOME)
    h = ai._headline("Home", {}, {}, {}, [f])
    assert "water stress is the leading exposure" in h.lower() and "CRITICAL" in h


def test_brief_lists_unrated_items_and_water_mechanism():
    f = ai._water_stress(*HOME)
    f["actions"] = ai._ACTIONS["water_stress"]
    a = {"asset_name": "Home", "lat": HOME[0], "lon": HOME[1], "risk_signal": "CRITICAL", "headline": "h",
         "findings": [f], "coverage": ai._coverage([f], {}, True)}
    titles = [s["title"] for s in site_brief(a)["sections"]]
    text = site_brief(a)["narrative"]
    assert "Not rated in this result" in titles
    assert "Water stress (basin scale): CRITICAL" in titles
    assert "not this site's own" in text or "not this site" in text


def test_physical_report_water_rating_matches_panel_rule():
    # same rule: max(baseline stress, drought) → CRITICAL at the home site
    from cri.finance import analyst
    w = analyst._water(HOME[0], HOME[1], None)
    q = [s for s in (w["water_stress"]["score"], w["drought"]["score"]) if s is not None]
    assert max(q) >= 4
