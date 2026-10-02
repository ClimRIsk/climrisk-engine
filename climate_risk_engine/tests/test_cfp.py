"""ISO 14067 CFP study calculator — hand-checked arithmetic."""
import os

import pytest

os.environ.setdefault("CRI_API_KEYS", "")
from fastapi.testclient import TestClient  # noqa: E402

from cri.api.main import app  # noqa: E402
from cri.lca import cfp, products  # noqa: E402
from cri.lca.engine import ElectricityUse, FuelUse, Inventory, ProcessEmission, calculate  # noqa: E402

client = TestClient(app)


def simple():
    inv = Inventory("1 t widget")
    inv.process.append(ProcessEmission("process", 100.0, 2))
    inv.electricity.append(ElectricityUse(100, None, 500.0, 2, "power"))      # 100 kWh × 500 g = 50 kg
    return inv


def test_gate_equals_engine_and_boundaries_add_up():
    inv = simple()
    spec = {"boundary": "cradle_to_grave",
            "distribution": [{"mode": "truck", "t": 1, "km": 100}],                       # 100 × (0.10163 + 0.02359)
            "use": {"service_life_years": 10, "other": [{"kg_co2e_per_year": 3.0}]},     # 30
            "end_of_life": [{"material": "Construction > Concrete", "t": 1, "shares": {"landfill": 1.0}}]}  # 1.2704
    r = cfp.run_study(inv, spec, "widget")
    assert r["by_stage"]["A1–A3 product stage"] == pytest.approx(150.0)
    assert r["by_stage"]["A4 distribution"] == pytest.approx(12.522, abs=0.01)
    assert r["by_stage"]["B use stage"] == pytest.approx(30.0)
    assert r["by_stage"]["C end of life"] == pytest.approx(1.2704, abs=0.001)
    assert r["cfp_kg_co2e"] == pytest.approx(150 + 12.522 + 30 + 1.2704, abs=0.02)
    assert r["boundary"]["is_full_cfp"] is True
    assert r["total_by_boundary_if_all_supplied"]["cradle_to_gate"] == pytest.approx(calculate(inv)["total_kg_co2e"])


def test_partial_cfp_labelled_and_stages_excluded():
    r = cfp.run_study(simple(), {"boundary": "cradle_to_gate"}, "widget")
    assert r["boundary"]["is_full_cfp"] is False and r["boundary"]["label"].startswith("Partial CFP")
    assert set(r["by_stage"]) == {"A1–A3 product stage"}
    assert r["cfp_kg_co2e"] == pytest.approx(150.0)


def test_aircraft_reported_separately_and_rf_uplift():
    spec = {"boundary": "cradle_to_customer", "distribution": [{"mode": "air", "t": 0.01, "km": 9000}]}
    r = cfp.run_study(simple(), spec, "widget")
    air = 0.01 * 9000 * (0.5313 + 0.13516)
    assert r["by_ghg_type"]["aircraft_ghg"] == pytest.approx(air, abs=0.05)
    assert r["by_ghg_type"]["fossil_ghg"] == pytest.approx(150.0)
    assert r["aircraft_rf_uplift_kg_co2e"] == pytest.approx(0.01 * 9000 * (0.89939 - 0.5313), abs=0.05)
    assert r["cfp_kg_co2e"] == pytest.approx(150 + air, abs=0.05)


def test_allocation_shares_and_scaling():
    a = {"method": "economic", "product": {"mass_t": 1, "price_per_t": 100, "energy_gj": 10},
         "coproducts": [{"name": "slag", "mass_t": 0.5, "price_per_t": 40, "energy_gj": 0.1}]}
    sh = cfp.allocation_shares(a)
    assert sh["mass"]["share"] == pytest.approx(1 / 1.5)
    assert sh["economic"]["share"] == pytest.approx(100 / 120)
    r = cfp.run_study(simple(), {"allocation": a}, "widget")
    assert r["allocation"]["product_share"] == pytest.approx(100 / 120, abs=1e-4)
    assert r["by_stage"]["A1–A3 product stage"] == pytest.approx(150 * 100 / 120, abs=0.01)
    assert {"mass", "energy", "economic"} == set(r["allocation"]["all_methods"])
    with pytest.raises(cfp.StudyError):
        cfp.run_study(simple(), {"allocation": {"method": "energy", "product": {"mass_t": 1},
                                                "coproducts": [{"name": "x", "mass_t": 1}]}}, "widget")


def test_cutoff_arithmetic():
    spec = {"cutoff": {"criterion_pct": 1, "total_input_mass_t": 2.0,
                       "excluded": [{"name": "additive", "mass_t": 0.01, "kgco2e_per_t": 1000}]}}
    r = cfp.run_study(simple(), spec, "widget")
    assert r["cutoff"]["excluded_mass_pct"] == pytest.approx(0.5)
    assert r["cutoff"]["excluded_emissions_pct"] == pytest.approx(100 * 10 / 160, abs=0.01)   # 10 kg vs 150 + 10
    assert r["cutoff"]["status"] == "criterion exceeded"
    assert cfp.run_study(simple(), {}, "widget")["cutoff"]["status"] == "not assessed"


def test_use_stage_grid_pathway_declines():
    use = {"service_life_years": 20, "electricity": [{"kwh_per_year": 1000, "country": "DEU"}]}
    stat = cfp.run_study(simple(), {"boundary": "cradle_to_grave", "use": dict(use, grid_basis="static")}, "w")
    nz = cfp.run_study(simple(), {"boundary": "cradle_to_grave", "use": dict(use, grid_basis="Net Zero 2050")}, "w")
    assert nz["by_stage"]["B use stage"] < stat["by_stage"]["B use stage"]
    g = products.grid_factor("DEU")[0] if hasattr(products, "grid_factor") else None
    from cri.lca.engine import grid_factor
    assert stat["by_stage"]["B use stage"] == pytest.approx(20 * 1000 * grid_factor("DEU")[0] / 1000, rel=1e-6)
    with pytest.raises(cfp.StudyError):
        cfp.run_study(simple(), {"boundary": "cradle_to_grave", "use": dict(use, grid_basis="Imaginary")}, "w")


def test_biogenic_net_zero_by_default_and_override():
    inv = Inventory("1 t")
    inv.fuels.append(FuelUse("wood", 10.0, 2))
    r = cfp.run_study(inv, {}, "w")
    assert r["by_ghg_type"]["biogenic_co2_emissions"] > 0
    assert r["by_ghg_type"]["biogenic_co2_emissions"] + r["by_ghg_type"]["biogenic_co2_removals"] == pytest.approx(0, abs=0.02)
    r0 = cfp.run_study(inv, {"biogenic_uptake_share": 0.0}, "w")
    assert r0["cfp_kg_co2e"] == pytest.approx(r["cfp_kg_co2e"] + r["by_ghg_type"]["biogenic_co2_emissions"], abs=0.05)


def test_luc_included_and_validation():
    r = cfp.run_study(simple(), {"luc_kgco2e": 20}, "w")
    assert r["by_ghg_type"]["direct_land_use_change"] == 20 and r["cfp_kg_co2e"] == pytest.approx(170.0, abs=0.02)
    with pytest.raises(cfp.StudyError):
        cfp.run_study(simple(), {"boundary": "nope"}, "w")
    with pytest.raises(cfp.StudyError):
        cfp.run_study(simple(), {"boundary": "cradle_to_grave", "end_of_life": [
            {"material": "Construction > Concrete", "t": 1, "shares": {"landfill": 0.4}}]}, "w")
    with pytest.raises(cfp.StudyError):
        cfp.run_study(simple(), {"boundary": "cradle_to_grave", "end_of_life": [
            {"material": "Unobtainium", "t": 1, "shares": {"landfill": 1}}]}, "w")


def test_uncertainty_ordering_hotspots_and_dq_and_reproducibility():
    r = cfp.run_study(simple(), {"boundary": "cradle_to_gate"}, "w")
    u = r["uncertainty"]
    assert u["p5"] < u["p50"] < u["p95"] and u["p5"] < r["cfp_kg_co2e"] < u["p95"]
    assert r["hotspots_to_80pct"][0]["item"] == "process" and r["hotspots_to_80pct"][-1]["cumulative_pct"] >= 80
    assert len(r["data_quality"]["criteria"]) == 10 and 1 <= r["data_quality"]["overall_score"] <= 5
    assert r["fingerprint"] == cfp.run_study(simple(), {"boundary": "cradle_to_gate"}, "w")["fingerprint"]
    assert r["verification"]["status"] == "not reviewed"


def test_report_contains_clause_9_sections_and_xlsx():
    r = cfp.run_study(simple(), {"commissioner": "ACME"}, "widget")
    rep = cfp.report(r, {"commissioner": "ACME"})
    assert {"1_general", "2_goal", "3_scope", "4_inventory_and_data", "5_results", "6_data_quality",
            "7_interpretation", "8_review_and_communication"} <= set(rep["sections"])
    assert "ACME" in rep["markdown"] and "kg CO2e" in rep["markdown"]
    assert cfp.report_xlsx(r, rep)[:2] == b"PK"


def test_api_study_cement_end_to_end():
    body = {"product": "cement",
            "params": {"country": "IND", "clinker_ratio": 0.75, "kiln_thermal_gj_per_t_clinker": 3.4,
                       "fuel_mix": {"coal_bituminous": 0.7, "petcoke": 0.3}, "kwh_per_t_clinker": 60,
                       "kwh_per_t_cement_grinding": 40, "clinker_cao": 0.65, "clinker_mgo": 0.015},
            "study": {"boundary": "cradle_to_grave", "distribution": [{"mode": "truck", "t": 1, "km": 250}],
                      "end_of_life": [{"material": "Construction > Concrete", "t": 1,
                                       "shares": {"closed_loop": 0.7, "landfill": 0.3}}]}}
    r = client.post("/lca/study", json=body)
    assert r.status_code == 200, r.text
    d = r.json()
    gate = client.post("/lca/cement", json=body["params"]).json()["footprint_kg_co2e_per_t"]
    assert d["by_stage"]["A1–A3 product stage"] == pytest.approx(gate, abs=0.1)
    assert d["by_stage"]["A4 distribution"] == pytest.approx(250 * (0.10163 + 0.02359), abs=0.01)
    assert d["by_stage"]["C end of life"] == pytest.approx(0.7 * 1.014 + 0.3 * 1.2704, abs=0.01)
    assert client.post("/lca/study/report", json=body).json()["report"]["markdown"].startswith("# Cement")
    x = client.post("/lca/study/report.xlsx", json=body)
    assert x.status_code == 200 and x.content[:2] == b"PK"
    assert client.get("/lca/eol-options").json()["materials"]
    assert client.post("/lca/study", json=dict(body, params={"country": "IND"})).status_code == 422
    assert client.post("/lca/study", json=dict(body, study={"boundary": "x"})).status_code == 422
