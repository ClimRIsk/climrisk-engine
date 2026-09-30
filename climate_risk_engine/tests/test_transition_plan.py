"""Transition plan builder tests."""
import os

import pytest

os.environ.setdefault("CRI_API_KEYS", "")
from fastapi.testclient import TestClient  # noqa: E402

from cri.api.main import app  # noqa: E402
from cri.transition_plan import PlanError, build, from_dict  # noqa: E402

c = TestClient(app)
BASE = {"organisation": "T", "base_year": 2024, "base_emissions_t": {"scope1": 100000, "scope2": 50000, "scope3": 0},
        "near_term_year": 2030, "near_term_reduction_pct": 42, "internal_carbon_price_usd_t": {"2024": 100, "2050": 100}}


def lever(**kw):
    d = {"name": "L", "scope": "scope1", "abatement_t_per_year": 10000, "start_year": 2025, "ramp_years": 1,
         "capex_usd": 0.0, "opex_change_usd_per_year": 0.0, "lifetime_years": 20}
    d.update(kw)
    return d


def test_no_levers_misses_target():
    r = build(from_dict(dict(BASE)))
    assert r["near_term"]["met"] is False and r["near_term"]["target_t"] == pytest.approx(87000)
    assert any(ch["level"] == "error" for ch in r["checks"])


def test_levers_meet_target_and_ramp():
    r = build(from_dict(dict(BASE, levers=[lever(abatement_t_per_year=70000, ramp_years=2)])))
    i25, i26 = r["years"].index(2025), r["years"].index(2026)
    assert r["planned_t"]["scope1"][i25] == pytest.approx(65000)      # half ramp in first year
    assert r["planned_t"]["scope1"][i26] == pytest.approx(30000)
    assert r["near_term"]["met"]


def test_abatement_capped_at_bau_and_overlap_flagged():
    r = build(from_dict(dict(BASE, levers=[lever(abatement_t_per_year=80000), lever(name="M", abatement_t_per_year=80000)])))
    assert min(r["planned_t"]["scope1"]) == 0
    assert any("overlap" in ch["message"] for ch in r["checks"])


def test_mac_arithmetic_and_reinvestment():
    r = build(from_dict(dict(BASE, discount_rate=0.07, levers=[lever(capex_usd=1_000_000, opex_change_usd_per_year=-50_000,
                                                                   lifetime_years=10, abatement_t_per_year=1000)])))
    crf = 0.07 / (1 - 1.07 ** -10)
    assert r["levers"][0]["abatement_cost_usd_per_t"] == pytest.approx((1e6 * crf - 50000) / 1000, abs=0.1)
    capex_years = [y for y, v in zip(r["years"], r["capex_usd_by_year"]) if v]
    assert capex_years == [2025, 2035, 2045]                           # initial + two reinvestments


def test_avoided_carbon_cost_uses_price_path():
    r = build(from_dict(dict(BASE, discount_rate=0.0, end_year=2026, near_term_year=2026, levers=[lever(abatement_t_per_year=1000)])))
    assert r["levers"][0]["npv_avoided_carbon_cost_usd"]["Internal carbon price"] == pytest.approx(2 * 1000 * 100)


def test_net_zero_residual_check():
    r = build(from_dict(dict(BASE, net_zero_year=2050, net_zero_reduction_pct=90,
                             levers=[lever(abatement_t_per_year=70000)])))
    assert any("residual emissions" in ch["message"] for ch in r["checks"])


def test_validation():
    with pytest.raises(PlanError):
        build(from_dict(dict(BASE, near_term_year=2020)))
    with pytest.raises(PlanError):
        build(from_dict(dict(BASE, levers=[lever(scope="scope4")])))


def test_api_and_inventory_seed():
    assert c.post("/transition-plan", json=dict(BASE, levers=[lever()])).status_code == 200
    inv = {"organisation": "Inv Co", "reporting_year": 2025, "entities": [{"id": "E", "name": "E", "country": "GBR"}],
           "records": [{"entity": "E", "category": "s1_stationary", "method": "factor",
                        "factor_id": "desnz26:fuels-gaseous-fuels-natural-gas-kwh-gross-cv", "quantity": 1e7, "evidence": "x"}],
           "category_status": {f"s3_c{k}": {"status": "excluded", "reason": "t"} for k in range(1, 16)}}
    r = c.post("/transition-plan", json={"inventory": inv, "near_term_year": 2030, "near_term_reduction_pct": 42,
                                         "internal_carbon_price_usd_t": {"2025": 80, "2050": 80}})
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["organisation"] == "Inv Co" and j["base_year"] == 2025 and j["bau_t"]["scope1"][0] > 1000
    assert c.get("/transition-plan/dashboard").status_code == 200
    assert c.post("/transition-plan", json=dict(BASE, near_term_reduction_pct=0)).status_code == 422
