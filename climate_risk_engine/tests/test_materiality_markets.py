"""Double materiality register and carbon-market module tests."""
import os

import pytest

os.environ.setdefault("CRI_API_KEYS", "")
from fastapi.testclient import TestClient  # noqa: E402

from cri.api.main import app  # noqa: E402
from cri.carbon_markets import MarketError, ets_position, screen_credits  # noqa: E402
from cri.materiality import IRO, MaterialityError, assess, score, suggest_e1  # noqa: E402

c = TestClient(app)


def test_impact_scoring_rules():
    actual = score(IRO("a", "E1", "x", "negative_impact", "d", actual=True, scale=4, scope=3, irremediability=5))
    assert actual["impact_score"] == pytest.approx(4.0)
    pot = score(IRO("b", "E1", "x", "negative_impact", "d", scale=4, scope=4, irremediability=4, likelihood=1))
    assert pot["impact_score"] == pytest.approx(2.0)
    hr = score(IRO("c", "S1", "x", "negative_impact", "d", human_rights=True, scale=4, scope=4, irremediability=4, likelihood=1))
    assert hr["impact_score"] == pytest.approx(4.0)        # severity takes precedence (ESRS 1 §45)
    fin = score(IRO("d", "E1", "x", "risk", "d", magnitude=4, likelihood=4))
    assert fin["financial_score"] == pytest.approx(4.0) and fin["impact_score"] is None


def test_assess_topics_and_checks():
    r = assess([IRO("a", "E1", "x", "risk", "d", magnitude=4, likelihood=4, evidence="e"),
                IRO("b", "E3", "x", "positive_impact", "d", actual=True, scale=1, scope=2, evidence="e",
                    stakeholder_ratings=[{"group": "ngo", "rating": 5}])], threshold=3)
    assert r["material_topics"] == ["E1"]
    assert any("stakeholder view differs" in ch["message"] for ch in r["checks"])
    with pytest.raises(MaterialityError):
        assess([IRO("a", "E1", "x", "risk", "d", magnitude=6, likelihood=1)])
    with pytest.raises(MaterialityError):
        assess([IRO("a", "X9", "x", "risk", "d", magnitude=3, likelihood=3)])


def test_suggest_e1():
    s = suggest_e1({"total_market": 1000}, 0.5, 80)
    assert {i["id"] for i in s} == {"E1-GHG", "E1-PHYS", "E1-PRICE", "E1-ENERGY"}


def test_ets_balance_and_cbam_phase_out():
    r = ets_position([{"name": "k", "sector": "cement", "verified_emissions_t": 1000, "free_allocation_t": 800,
                       "allowances_held_t": 100}], 2025, eua_price=70)
    assert r["totals"]["balance_t"] == 100 and r["totals"]["cost_to_cover"] == 7000 and r["currency"] == "EUR"
    proj = {p["year"]: p for p in r["installations"][0]["projection"]}
    assert proj[2026]["free_allocation_t"] == 780 and proj[2034]["free_allocation_t"] == 0
    r2 = ets_position([{"name": "p", "sector": "refining", "verified_emissions_t": 1000, "free_allocation_t": 800}], 2025, 70)
    assert r2["installations"][0]["projection"][-1]["free_allocation_t"] == 800     # not a CBAM sector
    with pytest.raises(MarketError):
        ets_position([], 2025)


def test_credit_screen_rules():
    r = screen_credits([
        {"id": "ok", "project_type": "DACCS", "quantity_t": 10, "vintage": 2025, "ccp_label": True, "retired": True,
         "intended_use": "neutralisation"},
        {"id": "bad", "project_type": "renewable energy", "quantity_t": 90, "vintage": 2015, "intended_use": "product_claim"}],
        current_year=2026)
    lots = {l["id"]: l for l in r["lots"]}
    assert lots["ok"]["status"] == "ok" and "neutralisation" in lots["ok"]["suitable_uses"]
    assert lots["bad"]["status"] == "issue" and any("2024/825" in f for f in lots["bad"]["flags"])
    assert r["shares"]["removals"] == pytest.approx(0.1)


def test_api():
    assert c.post("/materiality/assess", json={"iros": [{"id": "a", "topic": "E1", "subtopic": "x", "type": "risk",
                                                           "description": "d", "magnitude": 3, "likelihood": 3}]}).status_code == 200
    assert c.post("/materiality/assess", json={"iros": [{"id": "a", "topic": "E1", "subtopic": "x", "type": "risk",
                                                           "description": "d"}]}).status_code == 422
    assert c.post("/carbon-markets/ets-position", json={"installations": [{"name": "k", "sector": "cement",
                  "verified_emissions_t": 10, "free_allocation_t": 5}], "compliance_year": 2025}).status_code == 200
    assert c.post("/carbon-markets/credits", json={"lots": [{"id": "x", "project_type": "biochar", "quantity_t": 1}]}).status_code == 200
    for p in ("/materiality/dashboard", "/carbon-markets/dashboard"):
        assert c.get(p).status_code == 200
