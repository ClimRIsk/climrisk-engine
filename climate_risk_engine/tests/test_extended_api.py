"""Every endpoint the website and desktop app advertise must answer 200 on the local engine."""
import os

import pytest

os.environ.setdefault("CRI_API_KEYS", "")          # auth off for tests
from fastapi.testclient import TestClient          # noqa: E402

from cri.api.main import app                       # noqa: E402

c = TestClient(app)
POS = [{"company_id": "shell", "company_name": "Shell", "exposure_usd_m": 120, "sector": "oil_gas", "region": "EU",
        "physical_score": 62, "transition_score": 78},
       {"company_id": "heineken", "company_name": "Heineken", "exposure_usd_m": 80, "sector": "beverages",
        "region": "EU", "physical_score": 55, "transition_score": 40}]
CP = [{"company_id": "shell", "company_name": "Shell", "sector": "oil_gas", "ead_usd_m": 100, "baseline_pd": 0.01,
       "physical_score": 60, "transition_score": 75}]

CASES = [
    ("get", "/companies", None),
    ("get", "/stress/event/catalogue", None),
    ("post", "/regulatory/sfdr-pai", {"company_id": "x", "company_name": "X", "sector_key": "cement", "revenue_eur_m": 900,
                                       "ev_eur_m": 1500, "scope1_tco2e": 2e6, "scope2_tco2e": 1e5}),
    ("post", "/regulatory/eu-taxonomy", {"company_id": "x", "company_name": "X", "sector_key": "cement", "revenue_eur_m": 900,
                                          "capex_eur_m": 120}),
    ("post", "/regulatory/pcaf", {"counterparty_id": "x", "counterparty_name": "X", "sector_key": "cement",
                                   "asset_class": "business_loans", "outstanding_amount_eur_m": 50,
                                   "enterprise_value_eur_m": 1500, "scope1_tco2e": 2e6}),
    ("post", "/stress/event", {"company_id": "x", "company_name": "X", "sector_key": "cement", "asset_lat": 29.76,
                               "asset_lon": -95.37, "ev_usd_m": 1500, "revenue_usd_m": 900}),
    ("post", "/physical/slr", {"asset_id": "a", "asset_name": "Port", "asset_lat": 51.9, "asset_lon": 4.3,
                               "asset_elevation_m": 2.0, "ev_usd_m": 500}),
    ("post", "/physical/biodiversity", {"company_id": "x", "company_name": "X", "sector_key": "mining",
                                        "asset_lat": -22.3, "asset_lon": 118.35}),
    ("post", "/portfolio/benchmark", {"positions": POS, "benchmark_name": "test index",
                                      "benchmark_sector_weights": {"oil_gas": 5, "beverages": 5, "technology": 25,
                                                                   "Financial Services": 15, "Industrials": 50}}),
    ("get", "/physical/water-risk?lat=13.08&lon=80.27&sector=cement", None),
    ("post", "/portfolio/counterparty", {"positions": [
        {"counterparty_id": "a", "counterparty_name": "A", "sector_key": "cement", "region": "EU", "exposure_usd_m": 50,
         "ev_usd_m": 900, "revenue_usd_m": 600},
        {"counterparty_id": "b", "counterparty_name": "B", "sector_key": "utilities", "region": "US", "exposure_usd_m": 70,
         "ev_usd_m": 2000, "revenue_usd_m": 900}]}),
    ("post", "/portfolio/risk", {"positions": POS, "run_stress": False}),
    ("post", "/portfolio/credit", {"obligors": CP}),
    ("post", "/portfolio/rules", {"positions": POS}),
    ("post", "/screen/triage", {"company_name": "Holcim", "sector_key": "cement", "country": "CH"}),
    ("post", "/compare/scenarios", {"company_id": "x", "company_name": "X", "sector_key": "cement", "asset_lat": 29.76,
                                    "asset_lon": -95.37, "ev_usd_m": 1500, "revenue_usd_m": 900,
                                    "scope1_emissions_mt_co2e": 2.0}),
    ("post", "/deal/brief", {"company_name": "X", "sector_key": "cement", "country": "DE", "ev_usd_m": 1500,
                             "revenue_usd_m": 900}),
    ("post", "/sensitivity", {"company_name": "X", "sector_key": "cement", "scope1_mt_co2e": 2.0, "revenue_usd_m": 900,
                              "ev_usd_m": 1500}),
    ("post", "/agent/stranded-assets", {"company_name": "X", "sector": "coal", "carrying_value_usd_m": 900,
                                        "revenue_usd_m": 600}),
    ("post", "/agent/cbam", {"company_name": "X", "sector": "cement", "revenue_usd_m": 900}),
    ("post", "/agent/water-risk", {"company_name": "X", "sector": "beverages", "aqueduct_score": 3.5, "revenue_usd_m": 900}),
    ("post", "/assets/batch", {"assets": [{"asset_id": "a1", "name": "Plant", "lat": 29.76, "lon": -95.37,
                                           "sector": "chemicals", "replacement_cost_usd_m": 500,
                                           "annual_revenue_usd_m": 300}], "scenarios": ["cp"]}),
    ("post", "/decision", {"company_id": "heineken"}),
]


@pytest.mark.parametrize("method,path,body", CASES, ids=[p for _, p, _ in CASES])
def test_endpoint_answers(method, path, body):
    r = c.get(path) if method == "get" else c.post(path, json=body)
    assert r.status_code == 200, f"{path}: {r.status_code} {r.text[:400]}"
    assert r.content


def test_companies_have_assets_and_financials():
    d = c.get("/companies").json()
    assert d and all("financials" in x and "assets" in x for x in d)


MORE = [
    ("post", "/runs/export", {"company_id": "heineken", "scenario_id": "nze_2050"}),
    ("post", "/reports/pdf", {"company_id": "heineken"}),
    ("post", "/predictive/alerts", {"company_id": "x", "company_name": "X", "annual_revenue_usd": 5e8,
                                    "assets": [{"id": "a", "name": "A", "lat": 29.76, "lon": -95.37}]}),
    ("post", "/portfolio/optimise", {"positions": POS}),
    ("post", "/portfolio/validate", {"observations": [{"predicted_pd": 0.02, "defaulted": 0, "predicted_loss": 1.0,
                                                       "actual_loss": 0.8}] * 20}),
    ("post", "/regulatory/pcaf/portfolio", {"exposures": [CASES[4][2]]}),
    ("post", "/physical/slr/portfolio", {"assets": [CASES[6][2]]}),
    ("post", "/screen/triage/batch", {"companies": [{"company_name": "Holcim", "sector_key": "cement", "country": "CH"}]}),
    ("post", "/regulatory/sfdr-pai/export", dict(CASES[2][2], file_format="xlsx")),
    ("post", "/regulatory/sfdr-pai/export", dict(CASES[2][2], file_format="csv")),
    ("post", "/portfolio/delta", {"run_id_prior": "a", "run_id_current": "b", "positions": [
        {"company_id": "x", "company_name": "X", "prior_score": 0.4, "current_score": 0.55, "exposure_usd_m": 10}]}),
    ("post", "/companies", {"id": "test_co_ext", "name": "Test Co", "sector": "cement", "revenue_usd_m": 500,
                            "scope1_tco2": 1e6, "asset_lat": 51.9, "asset_lon": 4.3}),
    ("get", "/validation/scorecard", None),
]


@pytest.mark.parametrize("method,path,body", MORE, ids=[p for _, p, _ in MORE])
def test_more_endpoints_answer(method, path, body):
    r = c.get(path) if method == "get" else c.post(path, json=body)
    assert r.status_code in (200, 201), f"{path}: {r.status_code} {r.text[:400]}"
