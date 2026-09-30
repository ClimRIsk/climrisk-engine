"""Tests for the LCA module (cri.lca) and its API endpoints."""
import math

import pytest
from fastapi.testclient import TestClient

from cri.api.main import app
from cri.lca import products
from cri.lca.cement import CementPlant, Constituent, assess, calcination_kg_per_t_clinker, inventory
from cri.lca.engine import (ElectricityUse, FuelUse, Inventory, LOW_CARBON_FLOOR_G_KWH, ProcessEmission,
                            TransportLeg, calculate, factors, grid_factor, uncertainty)

client = TestClient(app)


def plant(**kw):
    base = dict(name="t", country="DEU", clinker_ratio=0.74, kiln_thermal_gj_per_t_clinker=3.5,
                fuel_mix={"coal_bituminous": 0.5, "petcoke": 0.2, "wood": 0.3},
                kwh_per_t_clinker=70, kwh_per_t_cement_grinding=40)
    base.update(kw)
    return CementPlant(**base)


# ── factor database ──────────────────────────────────────────────────────────

def test_factor_database_has_sources_and_values():
    f = factors()
    assert f["gwp"]["CH4_fossil"] == pytest.approx(29.8) and f["gwp"]["N2O"] == pytest.approx(273)
    assert f["fuels"]["natural_gas"]["co2_kg_per_gj"] == pytest.approx(56.1)
    assert f["fuels"]["wood"]["biogenic"] is True
    assert len(f["grid"]["countries"]) > 150
    assert f["process"]["clinker_default_kgco2_per_t"]["value"] == 525


def test_grid_factor_known_and_unknown_country():
    g_fra, src = grid_factor("FRA")
    g_pol, _ = grid_factor("POL")
    assert g_fra < 100 < g_pol and "Ember" in src
    g_world, src_w = grid_factor("XXX")
    assert g_world == factors()["grid"]["world_g_per_kwh"] and "world" in src_w.lower()


# ── core calculator ──────────────────────────────────────────────────────────

def test_fuel_combustion_arithmetic():
    inv = Inventory("1 t", fuels=[FuelUse("natural_gas", 10.0, 1)])
    r = calculate(inv)
    f = factors()["fuels"]["natural_gas"]
    gwp = factors()["gwp"]
    expect = 10 * (f["co2_kg_per_gj"] + f["ch4_kg_per_gj"] * gwp["CH4_fossil"] + f["n2o_kg_per_gj"] * gwp["N2O"])
    assert r["by_stage"]["A3 fuel combustion (fossil CO2, CH4, N2O)"] == pytest.approx(expect)
    assert r["by_stage"]["A3 fuel supply chain (well-to-tank)"] == pytest.approx(10 * f["wtt_kgco2e_per_gj"])


def test_biogenic_co2_is_separate():
    r = calculate(Inventory("1 t", fuels=[FuelUse("wood", 10.0, 1)]))
    assert r["biogenic_co2_kg"] == pytest.approx(10 * factors()["fuels"]["wood"]["co2_kg_per_gj"])
    # only CH4/N2O count toward the total
    assert r["total_kg_co2e"] < 0.05 * r["biogenic_co2_kg"]
    assert any("no well-to-tank" in e for e in r["exclusions"])


def test_capture_only_on_capturable_fossil_and_process_co2():
    inv = Inventory("1 t", process=[ProcessEmission("calc", 500, 1)], fuels=[FuelUse("wood", 10, 1)],
                    electricity=[ElectricityUse(100, "DEU", 400, 1)], capture_rate=0.9)
    r = calculate(inv)
    assert r["captured_kg_co2"] == pytest.approx(450)
    assert r["total_kg_co2e"] == pytest.approx(r["gross_kg_co2e"] - 450)


def test_transport_and_materials():
    inv = Inventory("1 t", materials=[__import__("cri.lca.engine", fromlist=["Material"]).Material("slag", 0.2, 50, "EPD", 2)],
                    transport=[TransportLeg("truck", 1.0, 100, 2)])
    r = calculate(inv)
    fr = factors()["freight"]["truck"]
    assert r["by_stage"]["A1 purchased materials"] == pytest.approx(10)
    assert r["by_stage"]["A2 inbound transport"] == pytest.approx(100 * (fr["kgco2e_per_tkm"] + fr["wtt_kgco2e_per_tkm"]))


def test_grid_multiplier_respects_floor_and_ignores_supplier_factor():
    grid = Inventory("1 t", electricity=[ElectricityUse(1000, "POL", None, 1)])
    ppa = Inventory("1 t", electricity=[ElectricityUse(1000, "POL", 10.0, 1)])
    assert calculate(grid, grid_multiplier=0.0)["total_kg_co2e"] == pytest.approx(LOW_CARBON_FLOOR_G_KWH)
    assert calculate(ppa, grid_multiplier=0.0)["total_kg_co2e"] == pytest.approx(10.0)
    assert calculate(ppa, grid_multiplier=1.0)["total_kg_co2e"] == pytest.approx(10.0)


def test_uncertainty_band_brackets_deterministic_and_widens_with_poor_data():
    good = Inventory("1 t", process=[ProcessEmission("p", 500, 1)], fuels=[FuelUse("natural_gas", 5, 1)])
    poor = Inventory("1 t", process=[ProcessEmission("p", 500, 5)], fuels=[FuelUse("natural_gas", 5, 5)])
    ug, up = uncertainty(good, n=4000), uncertainty(poor, n=4000)
    assert ug["p5"] < ug["deterministic"] < ug["p95"]
    assert (up["p95"] - up["p5"]) > 3 * (ug["p95"] - ug["p5"])


# ── cement ───────────────────────────────────────────────────────────────────

def test_calcination_tier2_arithmetic():
    p = plant(clinker_cao=0.65, clinker_mgo=0.015, ckd_t_per_t_clinker=0.0, raw_meal_toc=0.0)
    kg, basis, _ = calcination_kg_per_t_clinker(p)
    expect = (0.65 * 44.01 / 56.08 + 0.015 * 44.01 / 40.30) * 1000
    assert kg == pytest.approx(expect, rel=1e-3) and "Tier 2" in basis


def test_calcination_default_when_no_composition():
    kg, basis, _ = calcination_kg_per_t_clinker(plant(raw_meal_toc=0.0))
    assert kg == pytest.approx(525) and "default" in basis


def test_fuel_mix_must_sum_to_one_and_known_fuels():
    with pytest.raises(ValueError):
        inventory(plant(fuel_mix={"coal_bituminous": 0.5}))
    with pytest.raises(ValueError):
        inventory(plant(fuel_mix={"coal": 1.0}))


def test_cement_result_plausible_and_consistent():
    r = assess(plant(), n=500)
    assert 450 < r["footprint_kg_co2e_per_t"] < 900       # global range for grey cement
    assert r["uncertainty"]["p5"] < r["footprint_kg_co2e_per_t"] < r["uncertainty"]["p95"]
    assert sum(r["by_stage"].values()) == pytest.approx(r["footprint_kg_co2e_per_t"] + r["captured_co2_kg_per_t"], abs=0.5)
    assert r["biogenic_co2_kg_per_t"] > 0
    assert r["trajectory_with_plan"] is None and r["trajectory"]["plan_applied"] is False
    assert "Below 2°C" in r["trajectory"]["scenarios"]


def test_cement_plan_lowers_trajectory():
    r = assess(plant(plan_target_year=2035, plan_clinker_ratio=0.60, plan_capture_rate=0.9,
                     plan_capture_start_year=2040), n=300)
    base = r["trajectory"]["scenarios"]["Current Policies"]
    plan = r["trajectory_with_plan"]["scenarios"]["Current Policies"]
    assert base[0] == pytest.approx(plan[0], rel=0.01)
    assert plan[-1] < 0.35 * base[-1]


def test_plan_needs_target_year():
    with pytest.raises(ValueError):
        assess(plant(plan_clinker_ratio=0.6), n=100)


def test_cbam_not_applicable_inside_eu_and_positive_outside():
    assert assess(plant(), n=100)["cbam"]["applies"] is False
    r = assess(plant(country="IND", fuel_mix={"coal_bituminous": 1.0}), n=100, cbam_year=2030)
    c = r["cbam"]
    assert c["applies"] is True and c["cbam_phase_in_share"] == pytest.approx(0.485)
    expect = (c["direct_tco2_per_t_cement"] + c["indirect_tco2_per_t_cement"]) * c["eu_ets_price_usd_t"] * 0.485
    assert c["indicative_cbam_cost_usd_per_t_cement"] == pytest.approx(expect - 0, rel=0.01)


def test_constituent_without_factor_is_excluded():
    r = assess(plant(constituents=[Constituent("GGBS", 0.2, None)]), n=100)
    assert any("GGBS" in e for e in r["exclusions"])


# ── other products ───────────────────────────────────────────────────────────

def test_steel_routes():
    bof = products.steel("bf_bof", 150, "CHN", n=200)
    eaf = products.steel("eaf", 450, "FRA", n=200)
    assert bof["by_stage"]["A3 process emissions"] == pytest.approx(1460)
    assert eaf["footprint_kg_co2e_per_t"] < 0.2 * bof["footprint_kg_co2e_per_t"]
    with pytest.raises(ValueError):
        products.steel("open_hearth", 100)


def test_aluminium_electricity_dominates_and_exclusions_listed():
    coal = products.aluminium(14500, "CHN", n=200)
    hydro = products.aluminium(14500, "ISL", n=200)
    assert coal["footprint_kg_co2e_per_t"] > 3 * hydro["footprint_kg_co2e_per_t"]
    assert len(coal["exclusions"]) == 2
    full = products.aluminium(14500, "ISL", alumina_t=1.9, alumina_kgco2e_per_t=600, pfc_kgco2e_per_t=300, n=200)
    assert full["exclusions"] == []


def test_ammonia_tier1_and_urea_deduction():
    a = products.ammonia(100, "USA", n=200)
    b = products.ammonia(100, "USA", co2_recovered_for_urea_kg_per_t=700, n=200)
    assert a["footprint_kg_co2e_per_t"] - b["footprint_kg_co2e_per_t"] == pytest.approx(700, abs=1)


def test_custom_rejects_unknown_fuel():
    with pytest.raises(ValueError):
        products.custom("1 t", [{"fuel": "unobtanium", "gj": 1}], [], [], [], [])


# ── API ──────────────────────────────────────────────────────────────────────

def test_api_factors():
    r = client.get("/lca/factors")
    assert r.status_code == 200 and "natural_gas" in r.json()["fuels"]


def test_api_cement_requires_plant_inputs():
    assert client.post("/lca/cement", json={"country": "DEU"}).status_code == 422


def test_api_cement_ok_and_validation():
    body = dict(country="IND", clinker_ratio=0.7, kiln_thermal_gj_per_t_clinker=3.2,
                fuel_mix={"coal_bituminous": 0.7, "petcoke": 0.3}, kwh_per_t_clinker=65,
                kwh_per_t_cement_grinding=35, cbam_year=2028)
    r = client.post("/lca/cement", json=body)
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["cbam"]["applies"] and math.isfinite(j["footprint_kg_co2e_per_t"])
    bad = dict(body, fuel_mix={"coal_bituminous": 0.5})
    assert client.post("/lca/cement", json=bad).status_code == 422


@pytest.mark.parametrize("path,body", [
    ("/lca/steel", {"route": "eaf", "kwh_per_t": 450, "country": "DEU"}),
    ("/lca/aluminium", {"kwh_per_t": 14000, "country": "NOR"}),
    ("/lca/ammonia", {"kwh_per_t": 100, "country": "USA"}),
    ("/lca/custom", {"declared_unit": "1 t glass", "country": "ITA", "fuels": [{"fuel": "natural_gas", "gj": 6}],
                     "electricity": [{"kwh": 300}], "process": [{"name": "carbonates", "kg_co2e": 120}]}),
])
def test_api_products(path, body):
    r = client.post(path, json=body)
    assert r.status_code == 200, r.text
    assert r.json()["footprint_kg_co2e_per_t"] > 0


def test_api_dashboard_served():
    r = client.get("/lca/dashboard")
    assert r.status_code == 200 and "Product Carbon Footprint" in r.text
