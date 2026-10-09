"""Risk-manager tools: audit trail, covenant translation, exposure quality, ISSB draft, validation pack."""
import json

import pytest

from cri import riskmgmt as RM


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("CRI_HOME", str(tmp_path))
    return tmp_path


def test_audit_log_records_hash_and_detects_changed_input(home):
    inp = {"sites": [{"lat": 1, "lon": 2}], "seed": 7}
    rec = RM.audit_log("insurer_run", inp, {"aal": 12.5})
    assert rec and rec["input_sha256"] == RM.fingerprint({"seed": 7, "sites": [{"lon": 2, "lat": 1}]})        # key order does not matter
    assert RM.audit_recent(5)[0]["id"] == rec["id"]
    assert RM.audit_check(rec["id"], inp)["same_input"] is True
    assert RM.audit_check(rec["id"], {**inp, "seed": 8})["same_input"] is False
    assert RM.audit_check("nope", inp)["found"] is False


def test_audit_never_raises(monkeypatch):
    monkeypatch.setenv("CRI_HOME", "/proc/definitely/not/writable")
    assert RM.audit_log("x", {}, {}) is None


FIN = {"revenue": 1000.0, "ebitda": 200.0, "net_debt": 500.0, "interest_expense": 40.0, "equity": 400.0, "total_assets": 1200.0}


def test_translate_arithmetic_and_breach_flags():
    r = RM.translate(FIN, {"asset_damage": 100.0, "insured_recovery": 40.0, "bi_revenue_lost": 200.0, "bi_margin": 0.3, "extra_costs": 10.0},
                     {"max_net_debt_to_ebitda": 3.0, "min_interest_cover": 4.0, "min_equity_ratio": 0.30})
    assert r["loss"]["uninsured_asset_loss"] == 60.0 and r["loss"]["ebitda_hit"] == 70.0
    assert r["after"]["ebitda"] == 130.0 and r["after"]["net_debt"] == 630.0                       # 500 + 60 + 70, all debt funded
    assert r["before"]["net_debt_to_ebitda"] == 2.5 and round(r["after"]["net_debt_to_ebitda"], 3) == round(630 / 130, 3)
    by = {t["covenant"]: t for t in r["covenants"]}
    assert by["Net debt / EBITDA"]["status"] == "breach"
    assert by["EBITDA / interest"]["status"] == "pass, under 15% headroom" or by["EBITDA / interest"]["status"] == "breach"
    assert by["Equity / total assets"]["status"] in ("breach", "pass", "pass, under 15% headroom")
    assert r["ebitda_cushion"]["ebitda_floor"] == pytest.approx(500 / 3.0)
    assert r["assumptions"] and "not a forecast" in r["status"]


def test_translate_default_margin_is_stated_and_no_loss_means_no_change():
    r = RM.translate(FIN, {"bi_revenue_lost": 100.0}, {"max_net_debt_to_ebitda": 3.0})
    assert r["loss"]["ebitda_hit"] == pytest.approx(20.0) and any("average EBITDA margin" in a for a in r["assumptions"])
    z = RM.translate(FIN, {}, {"max_net_debt_to_ebitda": 3.0})
    assert z["after"]["ebitda"] == FIN["ebitda"] and z["breaches"] == []


def test_translate_rejects_bad_input():
    with pytest.raises(ValueError):
        RM.translate({"ebitda": 1}, {})
    with pytest.raises(ValueError):
        RM.translate(FIN, {"asset_damage": "lots"})
    with pytest.raises(ValueError):
        RM.translate(FIN, {"tax_rate": 1.5})


def test_translate_negative_ebitda_is_a_leverage_breach():
    r = RM.translate({**FIN, "ebitda": 50.0}, {"extra_costs": 80.0}, {"max_net_debt_to_ebitda": 3.0})
    assert r["after"]["ebitda"] < 0 and r["covenants"][0]["status"] == "breach"


def test_exposure_quality_flags_the_usual_problems():
    sites = [{"name": "ok", "lat": 25.77, "lon": -80.19, "tiv_usd": 1e7, "country": "US", "sector": "retail"},
             {"name": "null island", "lat": 0, "lon": 0, "tiv_usd": 1e7},
             {"name": "off", "lat": 95, "lon": 10, "tiv_usd": 1e7},
             {"name": "no tiv", "lat": 10.5, "lon": 20.5},
             {"name": "neg", "lat": 10.55, "lon": 20.55, "tiv_usd": -5},
             {"name": "dup", "lat": 25.77, "lon": -80.19, "tiv_usd": 2e7},
             {"name": "city", "lat": 25, "lon": -80, "tiv_usd": 1e7},
             {"name": "units", "lat": 30.12, "lon": -90.34, "tiv_usd": 1e12, "limit_usd": 5, "deductible_usd": 10}]
    q = RM.exposure_quality(sites)
    checks = {i["check"] for i in q["issues"]}
    assert {"null_island", "coord_range", "tiv_missing", "tiv_nonpositive", "duplicate_location", "coord_precision", "tiv_outlier", "limit_le_deductible"} <= checks
    assert q["rows_with_errors"] == 5 and q["verdict"] == "model after fixing errors" and q["usable_rows_pct"] == 37.5
    clean = RM.exposure_quality([{"name": f"s{i}", "lat": 10.11 + i, "lon": 20.22, "tiv_usd": 1e6, "country": "X", "sector": "y"} for i in range(5)])
    assert clean["verdict"] == "fit to model" and clean["errors"] == 0
    assert RM.exposure_quality([])["verdict"] == "do not model: no usable rows"


def test_issb_draft_marks_unknowns_and_cites_engine_outputs():
    ins = {"n_sites": 3, "tiv": 3e7, "stats": {"gross": {"aal": 250000.0, "rows": [{"rp": 100, "aep": 5e6, "oep": 4e6}]}}, "climate": {"ssp": "ssp245", "year": 2070, "aal": {"today": 1.0, "scenario": 1.2, "change_pct": 20.0}, "confidence": "medium"}}
    d = RM.issb_s2_draft({"insurance": ins}, {"name": "Acme", "reporting_year": 2025, "emissions": {"scope1": 1200}})
    md = d["markdown"]
    assert "250,000" in md and "engine: insurance.stats.gross.aal" in md and "+20.0%" in md
    assert "management to complete" in md and d["lines_to_complete"] >= 4
    lines = [l for s in d["sections"] for l in s["lines"]]
    s1 = next(l for l in lines if l["text"].startswith("Scope 1"))
    s2 = next(l for l in lines if l["text"].startswith("Scope 2"))
    assert s1["source"] == "supplied by the company" and s2["source"] == "management to complete"
    empty = RM.issb_s2_draft({})
    assert "250,000" not in empty["markdown"] and empty["lines_from_engine_or_company"] >= 0


def test_validation_pack_uses_files_and_states_gaps(tmp_path):
    demo = tmp_path / "demo.json"
    demo.write_text(json.dumps({"summary": {"questions": 24, "passed": 24, "pass_rate": 1.0, "median_s": 0.4, "p95_s": 2.1}}))
    p = RM.validation_pack(str(demo), str(tmp_path / "missing.json"), "26 passed")
    md = p["markdown"]
    assert "24/24 questions passed" in md and "26 passed" in md and "No evaluation file supplied" in md
    assert "NOT been validated" in md and p["has_demo"] and not p["has_eval"]


def test_risk_endpoints(home):
    from fastapi import Depends, FastAPI
    from fastapi.testclient import TestClient
    from cri.resident.api import make_router
    prot, _ = make_router(Depends(lambda: None))
    app = FastAPI()
    app.include_router(prot)
    c = TestClient(app)
    r = c.post("/resident/risk/translate", json={"fin": FIN, "loss": {"asset_damage": 10.0}, "covenants": {"max_net_debt_to_ebitda": 3.0}})
    assert r.status_code == 200 and r.json()["after"]["net_debt"] == 510.0
    assert c.post("/resident/risk/translate", json={"fin": {}, "loss": {}}).status_code == 422
    assert c.post("/resident/risk/exposure-quality", json={"sites": [{"lat": 1.11, "lon": 2.22, "tiv_usd": 5}]}).json()["n_sites"] == 1
    assert c.post("/resident/risk/exposure-quality", json={}).status_code == 422
    assert "markdown" in c.post("/resident/risk/issb-draft", json={"outputs": {}}).json()
    RM.audit_log("insurer_run", {"a": 1}, {"aal": 1})
    assert c.get("/resident/audit").json()["records"]
    assert "markdown" in c.get("/resident/risk/validation-pack", params={"demo": "../../etc/passwd"}).json()
