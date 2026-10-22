"""ESRS E1 gap assessment and the CSRD template's no-invented-facts rule."""
import math

import pytest

from cri.regulatory.esrs_e1_gap import BY_ID, CHECKLIST, assess_esrs_e1


def _row(out, item_id):
    return next(r for r in out["items"] if r["id"] == item_id)


def test_nothing_provided_everything_missing_and_materiality_warned():
    out = assess_esrs_e1()
    assert out["summary"]["provided"] == 0 and out["summary"]["coverage"] == 0.0
    assert out["summary"]["missing"] + out["summary"]["not_applicable"] == len(CHECKLIST)
    assert any("Materiality of E1 not stated" in w for w in out["warnings"])


def test_zero_is_provided_none_and_blank_are_missing():
    out = assess_esrs_e1({"E1-6.scope1": 0, "E1-6.total_ghg": None, "E1-1.16a": "   "}, e1_material=True)
    assert _row(out, "E1-6.scope1")["status"] == "provided" and _row(out, "E1-6.scope1")["value"] == 0
    assert _row(out, "E1-6.total_ghg")["status"] == "missing"
    assert _row(out, "E1-1.16a")["status"] == "missing"


@pytest.mark.parametrize("bad", [-1, "12", True, float("nan"), float("inf")])
def test_bad_numbers_are_invalid_gaps(bad):
    out = assess_esrs_e1({"E1-6.scope1": bad})
    assert _row(out, "E1-6.scope1")["status"] == "invalid"
    assert "E1-6.scope1" in out["gaps"]


def test_bool_kind_rejects_strings():
    out = assess_esrs_e1({"E1-1.16i": "yes"})
    assert _row(out, "E1-1.16i")["status"] == "invalid"


def test_conditional_price_only_when_price_scheme_used():
    off = assess_esrs_e1({"E1-8.uses_price": False})
    assert _row(off, "E1-8.price")["status"] == "not_applicable"
    on = assess_esrs_e1({"E1-8.uses_price": True})
    assert _row(on, "E1-8.price")["status"] == "missing"
    unknown = assess_esrs_e1({})
    assert _row(unknown, "E1-8.price")["status"] == "missing"   # unanswered gate does not excuse the item


def test_engine_proxy_is_never_provided_and_is_labelled():
    out = assess_esrs_e1({}, {"scope1": 123456.0})
    r = _row(out, "E1-6.scope1")
    assert r["status"] == "proxy_only" and r["engine_estimate"] == 123456.0 and "value" not in r
    assert out["summary"]["provided"] == 0
    assert "E1-6.scope1" in out["gaps"]


def test_company_value_beats_proxy():
    out = assess_esrs_e1({"E1-6.scope1": 100.0}, {"scope1": 999.0})
    r = _row(out, "E1-6.scope1")
    assert r["status"] == "provided" and r["value"] == 100.0 and "engine_estimate" not in r


def test_unknown_ids_reported_not_counted():
    out = assess_esrs_e1({"E1-6.scope9": 5})
    assert any("E1-6.scope9" in w for w in out["warnings"])
    assert out["summary"]["provided"] == 0


def test_not_material_needs_explanation():
    assert any("no explanation" in w for w in assess_esrs_e1(e1_material=False)["warnings"])
    assert not assess_esrs_e1(e1_material=False, e1_not_material_explanation="Assessed immaterial: see IRO register")["warnings"]


def test_coverage_arithmetic_and_by_requirement():
    vals = {"E1-6.scope1": 1.0, "E1-6.scope2_location": 2.0, "E1-6.scope2_market": 3.0,
            "E1-6.scope3_total": 4.0, "E1-6.total_ghg": 10.0, "E1-6.ghg_intensity": 0.1}
    out = assess_esrs_e1(vals, e1_material=True)
    assert out["by_requirement"]["E1-6"]["coverage"] == 1.0
    assert out["summary"]["provided"] == 6
    n = out["summary"]["items_applicable"]
    assert math.isclose(out["summary"]["coverage"], round(6 / n, 3))


def test_checklist_ids_unique_and_wellformed():
    assert len(BY_ID) == len(CHECKLIST)
    for i in CHECKLIST:
        assert i.kind in {"text", "number", "bool", "list"}
        if i.when:
            assert i.when[0] in BY_ID


def test_csrd_template_invents_no_company_facts():
    """Regression: the template used to hard-code targets, 'Committed' net zero, SBTi status and a True materiality flag."""
    from cri.outcomes.disclosure import generate_csrd
    from cri.api.main import COMPANY_REGISTRY, SCENARIO_REGISTRY
    from cri.engine.orchestrator import run as run_engine
    company = next(iter(COMPANY_REGISTRY.values()))
    runs = [run_engine(company=company, scenario=SCENARIO_REGISTRY[k]) for k in ("nze_2050", "delayed_transition", "current_policies")]
    r = generate_csrd(company, *runs).to_dict()["sections"]
    assert r["E1-1_transition_plan"]["has_transition_plan"] is None
    assert r["E1-4_targets"]["ghg_targets"] is None
    assert r["E1-8_internal_carbon_price"]["uses_internal_price"] is None
    assert r["E1-9_financial_effects"]["material_financial_effects"] is None
    assert r["E1-6_ghg_emissions"]["reported"]["scope_1_gross_tco2e"] is None
    txt = repr(r)
    for banned in ("Committed", "Pending SBTi", "reduction_pct", "IEA APS"):
        assert banned not in txt
    mine = generate_csrd(company, *runs, provided={"E1-6.scope1": 42.0, "E1-1.has_plan": False}).to_dict()["sections"]
    assert mine["E1-6_ghg_emissions"]["reported"]["scope_1_gross_tco2e"] == 42.0
    assert mine["E1-1_transition_plan"]["has_transition_plan"] is False
