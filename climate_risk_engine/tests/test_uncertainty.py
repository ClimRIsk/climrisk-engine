import pytest

from cri import uncertainty as U


def _design():
    return {"available": True, "d1_mm": 325.0, "by_source": {"era5": {"d1_mm": 202.0}, "power": {"d1_mm": 191.0}, "gauge": {"d1_mm": 325.0, "km": 4.0}}}


def test_rain_sources_quantifies_disagreement_and_says_who():
    e = U.rain_sources(_design())
    assert e["status"] == "quantified" and e["low"] == 191.0 and e["high"] == 325.0 and e["relative_range"] == pytest.approx(0.26 * 0 + (325 - 191) / 258.0, abs=0.01)
    assert "gauge 4.0 km" in e["source"] and "point" in e["note"]


def test_single_source_or_missing_is_not_assessed_not_zero():
    one = U.rain_sources({"available": True, "d1_mm": 100.0, "by_source": {"era5": {"d1_mm": 100.0}}})
    assert one["status"] == "not_assessed" and one["low"] is None and "second" in one["reason"]
    assert U.rain_sources({"available": False, "reason": "network"})["status"] == "not_assessed"
    assert U.rain_sources(None)["status"] == "not_assessed"


def test_scenario_and_sampling_entries():
    s = U.scenario_spread({"p10": 1.02, "median": 1.12, "p90": 1.31}, "SSP2-4.5")
    assert s["status"] == "quantified" and s["low"] == 1.02 and s["high"] == 1.31 and "SSP2-4.5" in s["method"]
    assert U.scenario_spread(None)["status"] == "not_assessed"
    r = U.return_level_sampling((250.0, 420.0), central=325.0)
    assert r["status"] == "quantified" and r["relative_range"] == pytest.approx(170 / 335, abs=0.01)
    assert U.return_level_sampling(None)["status"] == "not_assessed"


def test_damage_curve_transfer_uses_real_jrc_curves():
    e = U.damage_curve_transfer(1.0, "asia", "industrial")
    assert e["status"] == "quantified" and 0.0 <= e["low"] <= e["central"] <= e["high"] <= 1.0
    assert "not the published confidence band" in e["note"]
    assert U.damage_curve_transfer(1.0, "asia", "not_a_class")["status"] == "not_assessed"


def test_map_disagreement_counts_steps_and_needs_two_sources():
    e = U.map_disagreement({"JRC river map": "NEGLIGIBLE", "terrain model": "MODERATE", "rain run": "HIGH"})
    assert e["low"] == 0 and e["high"] == 3 and "3 step" in e["note"]
    assert U.map_disagreement({"JRC river map": "LOW"})["status"] == "not_assessed"
    assert U.map_disagreement({"a": "bogus", "b": "LOW"})["status"] == "not_assessed"


def test_ledger_ranks_quantified_lists_not_assessed_and_does_not_combine():
    led = U.site_ledger(12.98, 80.22, depth_m=1.0, design=_design(), scenario_rain={"p10": 1.02, "median": 1.12, "p90": 1.31}, bootstrap=(250.0, 420.0),
                        ratings={"JRC river map": "HIGH", "rain run": "MODERATE"}, fetch_rain=False)
    assert led["counts"]["quantified"] >= 5 and led["counts"]["not_assessed"] >= 5
    rr = [x["relative_range"] for x in led["ranking"]]
    assert rr == sorted(rr, reverse=True) and led["dominant_quantified"]
    assert "NOT combined" in led["combination"]
    kinds = {e["kind"] for e in led["entries"] if e["status"] == "not_assessed"}
    assert "internal_variability" in kinds and "scenario" in kinds
    tc = next(e for e in led["entries"] if e["key"] == "tc_sources")
    assert tc["status"] == "not_assessed" and "no cyclone wind" in tc["reason"]


def test_nothing_supplied_is_all_not_assessed_never_a_made_up_range():
    led = U.site_ledger(0.0, 0.0, fetch_rain=False)
    assert led["counts"]["quantified"] == 0 and led["ranking"] == [] and led["dominant_quantified"] is None
    assert all(e["low"] is None and e["high"] is None for e in led["entries"] if e["status"] == "not_assessed")


def test_endpoint():
    from fastapi.testclient import TestClient
    from cri.api.main import app
    r = TestClient(app).post("/uncertainty/ledger", json={"lat": 1.0, "lon": 1.0, "depth_m": 1.0, "fetch_rain": False, "ratings": {"a": "LOW", "b": "HIGH"}})
    if r.status_code in (401, 403):
        pytest.skip("endpoint needs an API key in this environment")
    assert r.status_code == 200 and r.json()["counts"]["quantified"] >= 2


def test_tc_sources_needs_two_independent_sources():
    one = U.tc_sources({"IBTrACS + Holland": {100: 62.0}})
    assert one["status"] == "not_assessed" and "independent second event set" in one["reason"]
    two = U.tc_sources({"IBTrACS + Holland": {100: 62.0}, "IRIS": {100: 70.5}})
    assert two["status"] == "quantified" and two["low"] == 62.0 and two["high"] == 70.5 and "IRIS" in two["source"]
    assert U.tc_sources({"a": {100: 60.0}, "b": {50: 55.0}})["status"] == "not_assessed"        # no common return period
    assert U.tc_sources({"a": {100: 60.0}, "b": {100: float("nan")}})["status"] == "not_assessed"
    led = U.site_ledger(20.0, 120.0, tc_winds={"A": {100: 60.0}, "B": {100: 75.0}}, fetch_rain=False)
    assert any(r["key"] == "tc_sources" for r in led["ranking"])


def _prof(**h):
    return {"hazards": h, "flood_events": {"available": True, "events": []}}


def _he(label, sev, status="modelled"):
    return {"label": label, "status": status, "severity": sev, "method": "m"}


def test_flood_concordance_worst_credible_not_average_and_null_is_not_dry():
    p = _prof(river_flood=_he("River flood", "NEGLIGIBLE"), surface_water=_he("Rain-driven flooding", "HIGH"),
              coastal_flood={"label": "Coastal", "status": "null", "severity": None, "reason": "no coastal map"})
    e = U.flood_concordance(p)
    assert e["worst"] == "HIGH" and e["agreement"] == "disagree" and e["status"] == "quantified"
    assert e["low"] == 0 and e["high"] == 3
    assert [x["key"] for x in e["not_assessed"]] == ["coastal_flood"] and "Not assessed: Coastal" in e["verdict"]
    assert "worst credible" in e["verdict"].lower()


def test_flood_concordance_single_source_and_none():
    one = U.flood_concordance(_prof(river_flood=_he("River flood", "LOW")))
    assert one["agreement"] == "single_source" and one["status"] == "not_assessed" and "Not cross-checked" in one["verdict"]
    none = U.flood_concordance(_prof(river_flood={"label": "River flood", "status": "null", "severity": None, "reason": "x"}))
    assert none["agreement"] == "none" and "NOT ASSESSED" in none["verdict"]
    assert U.flood_concordance(None)["agreement"] == "none"


def test_flood_concordance_warns_when_recorded_events_contradict_models():
    p = _prof(river_flood=_he("River flood", "NEGLIGIBLE"), surface_water=_he("Rain-driven flooding", "LOW"))
    p["flood_events"] = {"available": True, "events": [{"date": "2021-07-14", "bbox_contains_point": True}, {"date": "2019-01-01", "bbox_contains_point": False}]}
    e = U.flood_concordance(p)
    assert e["agreement"] == "agree" and e["observed"]["events_with_box_over_site"] == 1 and "Caution: 1 recorded flood event" in e["verdict"]


def test_site_ledger_uses_profile_for_flood_concordance():
    p = _prof(river_flood=_he("River flood", "MODERATE"), surface_water=_he("Rain-driven flooding", "HIGH"))
    led = U.site_ledger(12.98, 80.22, fetch_rain=False, profile=p)
    keys = [e["key"] for e in led["entries"]]
    assert "flood_concordance" in keys and "map_disagreement" not in keys


def test_every_peril_is_checked_and_single_source_perils_say_so():
    p = _prof(cyclone=_he("Tropical cyclone", "HIGH"), wind=_he("Extreme wind", "MODERATE"),
              thermal_stress=_he("Heat", "HIGH"), heat=_he("WBGT", "HIGH"), earthquake=_he("Earthquake", "MODERATE"),
              water_stress=_he("Aqueduct", "HIGH"), drought={"label": "Drought", "status": "null", "severity": None, "reason": "no data"})
    h = U.hazard_concordance(p)
    assert h["perils"]["wind"]["agreement"] == "agree" and h["perils"]["heat"]["agreement"] == "agree"
    assert h["perils"]["seismic"]["agreement"] == "single_source" and "only one source" in h["perils"]["seismic"]["verdict"]
    assert h["perils"]["water"]["agreement"] == "single_source" and h["perils"]["water"]["not_assessed"][0]["key"] == "drought"
    assert "seismic" in h["single_source"] and "flood" in h["no_answer"] and "wind" in h["cross_checked"]


def test_ledger_gets_one_concordance_entry_per_peril_that_ran_and_no_zero_for_missing():
    p = _prof(river_flood=_he("River flood", "HIGH"), surface_water=_he("Rain", "LOW"), earthquake=_he("Earthquake", "LOW"))
    led = U.site_ledger(10.0, 20.0, fetch_rain=False, profile=p)
    keys = {e["key"] for e in led["entries"]}
    assert {"flood_concordance", "seismic_concordance"} <= keys and "wind_concordance" not in keys
    f = next(e for e in led["entries"] if e["key"] == "flood_concordance")
    assert f["agreement"] == "disagree" and f["worst"] == "HIGH"


def test_shared_inputs_are_named_when_sources_agree():
    p = _prof(landslide_slope=_he("Slope", "HIGH"), landslide=_he("Physics", "HIGH"))
    e = U.peril_concordance(p, "slope")
    assert e["agreement"] == "agree" and "same terrain model" in e["verdict"]


def test_a_study_that_ran_without_a_severity_is_unrated_not_unassessed():
    p = _prof(thermal_stress=_he("Heat screening", "HIGH"), heat={"label": "WBGT", "status": "modelled", "severity": None, "value": 31.2, "unit": "C"})
    e = U.peril_concordance(p, "heat")
    assert e["agreement"] == "single_source" and e["unrated"][0]["value"] == 31.2 and e["not_assessed"] == []
    assert "Ran but not rated" in e["verdict"]
    only = U.peril_concordance(_prof(heat={"label": "WBGT", "status": "modelled", "severity": None, "value": 31.2, "unit": "C"}), "heat")
    assert only["agreement"] == "unrated" and "NOT ASSESSED" not in only["verdict"]
