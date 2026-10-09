"""OED import / ORD-style export and the Entity -> Facility -> Collateral credit bridge."""
import numpy as np
import pytest

from cri import creditbridge as CBR
from cri import insurance as I
from cri import oasis as O

LOC = """PortNumber,AccNumber,LocNumber,LocName,Latitude,Longitude,CountryCode,BuildingTIV,OtherTIV,ContentsTIV,BITIV,ConstructionCode,OccupancyCode,LocPerilsCovered,LocDed6All,LocDedType6All,LocLimit6All,LocLimitType6All,LocCurrency
1,A1,1,Plant North,22.0,91.0,BD,600000,0,300000,100000,5050,3000,WTC;OO1,0.05,2,1,2,USD
1,A1,2,Plant South,22.4,91.4,BD,500000,0,0,0,5000,3000,WTC,10000,0,400000,0,USD
1,A2,1,Depot,0,0,BD,100000,0,0,0,5000,1000,WTC,,,,,USD
1,A2,2,Depot EU,22.1,91.1,BD,100000,0,0,0,5000,1000,WTC,,,,,EUR
1,A2,3,Quake only,22.2,91.2,BD,100000,0,0,0,5000,1000,QEQ,,,,,USD
1,A2,4,Odd deductible,22.3,91.3,BD,200000,0,0,0,5000,1000,WTC,0.1,1,,,USD
1,A1,1,Plant North dup,22.0,91.0,BD,1,0,0,0,5000,3000,WTC,,,,,USD
"""


def test_read_oed_maps_sums_and_reports_everything_it_drops():
    r = O.read_oed(O.parse_csv(LOC), [{"PortNumber": "1", "AccNumber": "A1", "LayerLimit": "5000000"}], occupancy_map={"3000": "cement"})
    assert r["read"] == 7 and r["loaded"] == 3 and r["skipped"] == 4
    n, s, odd = r["sites"]
    assert n["tiv_usd"] == 1_000_000 and n["sector"] == "cement" and n["covers"] == ["flood", "wind"]
    assert n["deductible_pct"] == pytest.approx(5.0) and n["limit_pct"] == pytest.approx(100.0)          # OED fractions become percents
    assert s["deductible_usd"] == 10000 and s["limit_usd"] == 400000 and s["sector"] == "cement" and s["covers"] == ["wind"]
    assert "deductible_usd" not in odd and "deductible_pct" not in odd                                  # type 1 (fraction of loss) cannot be applied
    issues = {w["issue"] for w in r["warnings"]}
    assert {"bad_coordinates", "currency_without_rate", "no_modelled_peril", "term_type_not_supported", "policy_terms_not_applied", "duplicate_location_id"} <= issues
    assert r["meta"][0]["ConstructionCode"] == "5050" and r["meta"][0]["coverage_tiv"]["bi"] == 100000
    assert r["hierarchy"] == {"portfolios": 1, "accounts": 2, "locations": 3, "account_rows_supplied": 1}
    for s_ in r["sites"]:
        I.normalise_site(s_, 0)                                                                         # the engine accepts every loaded row


def test_unmapped_occupancy_is_flagged_not_guessed_and_fx_converts():
    rows = O.parse_csv(LOC)[:1]
    r = O.read_oed(rows)
    assert r["sites"][0]["sector"] == "other" and any(w["issue"] == "construction_occupancy_not_used" for w in r["warnings"])
    rows[0]["LocCurrency"] = "EUR"
    assert O.read_oed(rows, fx={"EUR": 1.1})["sites"][0]["tiv_usd"] == pytest.approx(1_100_000)


def _hz():
    return {"cont": "asia", "tc_region": "NI", "iso3": "BGD", "notes": [], "wind": {"lam": 0.6, "sigma": 9.0, "rp100_ms": 40.0},
            "river": {"rps": [10, 50, 100, 500], "depths": {str(rp): [d] * 40 + [0.0] * 60 for rp, d in {10: 0.2, 50: 0.6, 100: 0.9, 500: 2.0}.items()}, "protection": 0.0}, "coast": None}


def test_ord_tables_have_portfolio_account_and_location_levels():
    imp = O.read_oed(O.parse_csv(LOC)[:2], occupancy_map={"3000": "cement"})
    res = I.run(imp["sites"], hazards=[_hz(), _hz()], n_years=2000, reinsurance_program={"quota_share": 0.2})
    t = O.ord_tables(res, imp["meta"])
    assert {t_["Level"] for t_ in t["summary_info"]} == {"portfolio", "account", "location"}
    port = next(a for a in t["alt"] if a["SummaryId"] == 1 and a["Perspective"].startswith("gross"))
    assert port["MeanLoss"] == pytest.approx(res["stats"]["gross"]["aal"]) and port["SDLoss"] is not None
    acc = next(a for a in t["alt"] if a["SummaryId"] == 201)
    assert acc["MeanLoss"] == pytest.approx(sum(r["aal"] for r in res["sites"]))                        # one account holds both sites
    names = {(e["EPTypeName"], e["ReturnPeriod"]) for e in t["ept"] if e["Perspective"].startswith("gross")}
    assert ("OEP", 100) in names and ("AEP", 250) in names
    assert any(e["Perspective"] == "net of reinsurance" for e in t["ept"])
    csv_text = O.to_csv(t["ept"])
    assert csv_text.splitlines()[0].startswith("SummaryId,Perspective") and len(csv_text.splitlines()) == len(t["ept"]) + 1
    assert any("not certified" in n for n in t["notes"])


def test_lei_check_digits():
    assert CBR.validate_lei("5493001KJTIIGC8Y1R12")                                                     # a published LEI (Bloomberg Finance L.P.)
    assert not CBR.validate_lei("5493001KJTIIGC8Y1R13")
    assert not CBR.validate_lei("short") and not CBR.validate_lei("")


def _book():
    def coll(i, lat, lon, v, insured=False):
        c = {"id": f"c{i}", "name": f"asset {i}", "lat": lat, "lon": lon, "value_usd": v, "sector": "cement", "insured": insured}
        if insured:
            c.update(deductible_pct=2.0, limit_pct=100.0)
        return c
    return [
        {"id": "E1", "name": "Acme", "lei": "5493001KJTIIGC8Y1R12", "financials": {"revenue": 1000.0e6, "ebitda": 200e6, "net_debt": 500e6, "interest_expense": 40e6, "equity": 400e6, "total_assets": 1200e6},
         "facilities": [{"id": "F1", "type": "term_loan", "drawn": 300e6, "covenants": {"max_net_debt_to_ebitda": 3.0, "min_interest_cover": 4.0},
                         "collateral": [coll(1, 22.0, 91.0, 250e6), coll(2, 22.1, 91.1, 150e6, insured=True)]}]},
        {"id": "E2", "name": "Beta", "lei": "BADLEI", "facilities": [{"id": "F2", "drawn": 50e6, "collateral": [coll(3, 22.2, 91.2, 80e6), {"id": "bad", "lat": 99, "lon": 0, "value_usd": 1}]}]},
    ]


def test_credit_bridge_levels_ltv_and_flags():
    r = CBR.assess(_book(), return_period=100, n_years=3000, hazards=[_hz()] * 3)
    assert {p["issue"] for p in r["schema_problems"]} == {"lei_check_digits_fail", "collateral_invalid"}
    e1, e2 = r["entities"]
    f1 = e1["facilities"][0]
    assert f1["collateral_value"] == 400e6 and f1["ltv_before"] == pytest.approx(0.75)
    sc = f1["loss_1in100"]
    assert sc["ground_up"] > 0 and 0 <= sc["insured"] <= sc["ground_up"] and sc["uninsured"] == pytest.approx(sc["ground_up"] - sc["insured"])
    assert f1["ltv_after"] > f1["ltv_before"]                                                           # losses raise loan-to-value
    assert f1["insured_share_of_value_pct"] == 37.5
    fe = e1["financial_effect"]
    assert fe["after"]["net_debt"] > fe["before"]["net_debt"] and fe["loss"]["asset_damage"] == pytest.approx(sc["ground_up"])
    assert e2["facilities"][0]["insured_share_of_value_pct"] == 0.0 and e2["facilities"][0]["collateral_assets"] == 1
    p = r["portfolio"]
    assert p["collateral_assets"] == 3 and p["diversification"]["portfolio_1in_rp_loss"] <= p["diversification"]["sum_of_entity_1in_rp_losses"] + 1e-6
    assert "probability of default" in r["not_produced"] and any("not an LGD" in x for x in r["limits"])


def test_uninsured_collateral_gets_no_recovery_and_errors_are_clear():
    book = _book()[:1]
    book[0]["facilities"][0]["collateral"] = [dict(book[0]["facilities"][0]["collateral"][0])]
    r = CBR.assess(book, hazards=[_hz()], n_years=2000)
    assert r["entities"][0]["facilities"][0]["loss_1in100"]["insured"] == 0.0
    with pytest.raises(ValueError):
        CBR.assess([], n_years=1000)
    with pytest.raises(ValueError, match="no valid collateral"):
        CBR.assess([{"id": "x", "facilities": [{"collateral": [{"lat": 1, "lon": 2}]}]}])
    with pytest.raises(ValueError):
        CBR.assess(_book(), return_period=5, hazards=[_hz()] * 3)


def test_endpoints(monkeypatch):
    from fastapi import Depends, FastAPI
    from fastapi.testclient import TestClient
    from cri.resident.api import make_router
    monkeypatch.setattr(I, "fetch_hazards", lambda sites, **k: [_hz() for _ in sites])
    prot, _ = make_router(Depends(lambda: None))
    app = FastAPI()
    app.include_router(prot)
    c = TestClient(app)
    r = c.post("/resident/oed/import", json={"location": LOC, "occupancy_map": {"3000": "cement"}}).json()
    assert r["loaded"] == 3 and r["warnings"]
    assert c.post("/resident/oed/import", json={}).status_code == 422
    run = c.post("/resident/oed/run", json={"location": LOC, "occupancy_map": {"3000": "cement"}, "n_years": 1000}).json()
    assert run["result"]["n_sites"] == 3 and "ept" in run["ord_csv"] and run["ord"]["summary_info"]
    assert "entity" in c.get("/resident/credit/schema").json()
    cr = c.post("/resident/credit/assess", json={"entities": _book(), "n_years": 1000}).json()
    assert cr["portfolio"]["collateral_assets"] == 3
    assert c.post("/resident/credit/assess", json={"entities": []}).status_code == 422
