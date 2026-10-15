"""Linked stress test: pieces work alone, work together, count costs jointly, and keep missing inputs null."""
import copy

import pytest

from cri import stresstest as ST


def fake_hazards(lat, lon, name):
    return {"hazards": {"cyclone": {"status": "modelled", "label": "Tropical cyclone wind", "value": 54.6, "unit": "m/s", "severity": "HIGH",
                                    "metrics": {"rp_wind_ms": {"10": 35.3, "50": 48.8, "100": 54.6, "250": 62.2}}},
                        "coastal_flood": {"status": "null", "label": "Coastal flood", "value": None, "reason": "no surge data"}}}


def ex():
    return copy.deepcopy(ST.WORKED_EXAMPLE)


def test_wind_is_placed_on_the_site_return_period_curve():
    rp = {"10": 35.3, "50": 48.8, "100": 54.6, "250": 62.2}
    assert ST.return_period_of_wind(54.6, rp)["rp_years"] == 100
    mid = ST.return_period_of_wind(58.0, rp)["rp_years"]
    assert 100 < mid < 250
    assert ST.return_period_of_wind(64.0, rp)["bound"] == ">250"
    assert ST.return_period_of_wind(20.0, rp)["bound"] == "<10"
    assert ST.return_period_of_wind(50, {})["status"] == "null"


def test_operations_standalone_fuel_gap_and_capacity():
    o = ST.operations({"grid_outage_days": 9, "generator_fuel_hours": 48, "access_cut_days": 5, "workers_absent_pct": 80})
    assert o["fuel_gap_days"] == 7 and o["full_capacity_from_day"] == 9
    assert o["capacity_by_day"][0] == pytest.approx(0.2) and o["capacity_by_day"][3] == 0.0 and o["capacity_by_day"][10] == 1.0
    assert ST.operations({})["status"] == "null"


def test_supply_buffer_only_drawn_once_plant_runs_and_alternates_are_counted():
    s = ST.supply({"delay_weeks_min": 4, "delay_weeks_max": 6, "alternates": [{"name": "A", "preapproved": False, "backlog_weeks": 8, "qualification_days": 14}]},
                  {"component_buffer_days": 14}, None, restart_day=9)
    assert s["shortfall_days"] == {"if_delay_min": 5.0, "if_delay_max": 19.0}
    assert s["earliest_alternate_day"] == 70 and s["alternate_beats_supplier"] is False
    assert s["buffer_gap_days"]["if_delay_max"] == 19.0
    pre = ST.supply({"delay_weeks_max": 6, "alternates": [{"name": "B", "preapproved": True, "backlog_weeks": 1}]}, {"component_buffer_days": 14})
    assert pre["earliest_alternate_day"] == 7 and pre["preapproved_alternates"] == 1


def test_missing_buffer_is_null_not_zero():
    s = ST.supply({"delay_weeks_min": 4, "delay_weeks_max": 6}, {})
    assert s["shortfall_days"] == {"if_delay_min": None, "if_delay_max": None} and "component_buffer_days" in s["inputs_missing"]


def test_freight_counts_tariff_and_penalty_together_and_exposes_the_interaction():
    sp = ex()
    f = ST.freight_decision(sp["batch"], sp["tariff"], sp["penalty"], restart_day=40)
    sea, air = f["options"]
    assert air["tariff_triggered"] and not sea["tariff_triggered"]
    assert air["costs_usd"]["tariff_usd"] == pytest.approx(0.15 * sp["batch"]["value_usd"])
    assert air["emissions_t"] > 50 * sea["emissions_t"]                       # air per t·km is far above ship
    i = f["interaction"]
    assert i["air_minus_sea_total_usd"] == pytest.approx(air["total_extra_cost_usd"] - sea["total_extra_cost_usd"], abs=0.02)
    assert i["tariff_triggered_by_air_usd"] > 0 and i["freight_premium_of_air_usd"] > 0
    assert f["best"]["total_extra_cost_usd"] <= min(sea["total_extra_cost_usd"], air["total_extra_cost_usd"])


def test_missing_cost_input_is_never_counted_as_zero():
    sp = ex()
    del sp["batch"]["air_usd_per_t"]
    f = ST.freight_decision(sp["batch"], sp["tariff"], sp["penalty"], 40)
    assert f["options"][1]["total_extra_cost_usd"] is None and "air_freight_usd" in f["options"][1]["components_missing"]
    assert f["options"][1]["profit_usd"] is None


def test_tariff_without_threshold_is_not_guessed():
    sp = ex()
    sp["tariff"] = {"rate_pct": 15, "trigger": {"type": "intensity_threshold"}}
    f = ST.freight_decision(sp["batch"], sp["tariff"], sp["penalty"], 40)
    assert f["options"][1]["costs_usd"]["tariff_usd"] is None and "threshold" in f["options"][1]["tariff_note"]


def test_full_run_answers_four_questions_and_flags_governance_gap():
    r = ST.run(ex(), hazard_fn=fake_hazards)
    ids = [a["id"] for a in r["answers"]]
    assert ids == ["visibility", "financial_buffers", "inventory_buffer", "governance"]
    gov = r["answers"][3]
    assert gov["status"] == "partial" and "governance.decision_owner" in gov["inputs_missing"]
    assert r["sites"][0]["storm_placement"]["bound"] == ">250"                # Cat 4 midpoint 64 m/s > the engine's 250-year wind
    assert r["ship_ready_day"]["high"] >= r["ship_ready_day"]["low"] > 0
    assert "joint" in " ".join(r["rules"]).lower() or "together" in " ".join(r["rules"]).lower()


def test_run_with_nothing_but_a_storm_keeps_everything_else_null():
    r = ST.run({"hazard": {"category": 4}, "sites": [{"name": "Hub"}]}, hazard_fn=fake_hazards)
    assert r["operations"]["status"] == "null" and r["supply"]["status"] == "null" and r["freight"]["status"] == "null"
    assert r["sites"][0]["status"] == "null"
    assert all(a["status"] in ("null", "partial") for a in r["answers"])


def test_bad_inputs_are_rejected():
    with pytest.raises(ST.StressError):
        ST.run({"hazard": {"category": 9}, "sites": []})
    with pytest.raises(ST.StressError):
        ST.operations({"grid_outage_days": -1})


def test_chat_text_is_recognised_and_numbers_extracted():
    text = ('The Scenario: "The Squeeze on District 4". A Category 4 typhoon makes landfall; backup generators only have 48 hours of fuel on-site. '
            'A Tier-2 supplier was hit and estimates a 4-to-6 week delay. Shifting from sea freight to air freight will trigger a 15% green border tariff. '
            'Stress-test questions: visibility, financial buffers, inventory buffer, governance. Phase 1, Phase 2, Phase 3.')
    assert ST.looks_like_stress_scenario(text)
    s = ST.extract_spec(text)
    assert s["hazard"]["category"] == 4 and s["event"]["generator_fuel_hours"] == 48
    assert (s["supplier"]["delay_weeks_min"], s["supplier"]["delay_weeks_max"]) == (4, 6)
    assert s["tariff"]["rate_pct"] == 15 and s["tariff"]["trigger"]["type"] == "any_air"
    assert not ST.looks_like_stress_scenario("What is a storm surge?")
    assert not ST.looks_like_stress_scenario("What is the heat stress risk at 29.7, -95.0?")


def test_resident_routes_a_scenario_to_the_stress_test(monkeypatch, tmp_path):
    from cri.resident import agent as A
    monkeypatch.setattr(ST, "_DRAFT", tmp_path / "d.json")
    ag = A.get_agent() if hasattr(A, "get_agent") else A.Agent()
    text = ('Scenario: a Category 4 typhoon hits our Zone 4 hub; generators have 48 hours of fuel. A Tier-2 supplier faces a 4 to 6 week delay. '
            'Air freight instead of sea freight triggers a 15% border tariff. Stress-test it: visibility, buffers, governance. Phase 1 Phase 2 Phase 3 '
            'with much more narrative text to pass the length check, including logistics and shipping details of the batch.')
    r = ag._stress_commands("s-test", text)
    assert r and r["intent"] == "stress_test" and r["actions"][0]["url"].startswith("/stress-test/dashboard")
    assert any(b.get("type") == "table" for b in r["blocks"])
    assert ST.load_draft()["tariff"]["rate_pct"] == 15
    assert ag._stress_commands("s-test", "build a 3D twin at 29.7255, -95.0") is None


def test_training_rows_are_engine_correct_and_teach_nulls():
    import random
    from cri.llmtrain import dataset as D, evaluate as E, gold as G
    rows = D.build_stress(random.Random(3), 80, "train")
    assert len(rows) >= 60 and any(r["meta"]["nulls"] for r in rows) and any(not r["meta"]["nulls"] for r in rows)
    for r in rows:
        ref = r["messages"][-1]["content"]
        s = E.score_row(r, ref) if hasattr(E, "score_row") else None
        facts = r["meta"]["facts"] + r["messages"][1]["content"]
        assert E.nums(ref) <= E.nums(facts) | {"0", "1", "2", "3", "4", "5", "10"}, ref        # the reference answer invents no number
        if r["meta"]["nulls"]:
            assert "not modelled" in ref.lower() or "no decision owner" in ref.lower() or "not known" in ref.lower()
    assert sum(1 for g in G.GOLD if g[0] == "stress") >= 10
