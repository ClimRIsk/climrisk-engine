"""Real-estate decarbonisation module tests."""
import os

import pytest

os.environ.setdefault("CRI_API_KEYS", "")
from fastapi.testclient import TestClient  # noqa: E402

from cri.api.main import app  # noqa: E402
from cri.carbon import factors as CF  # noqa: E402
from cri.real_estate import Building, REError, assess_building, assess_portfolio  # noqa: E402

c = TestClient(app)


def office(**kw):
    d = dict(id="b", name="b", country="GBR", property_type="Office", floor_area_m2=1000, electricity_kwh=100000,
             natural_gas_kwh=50000, data_year=2025)
    d.update(kw)
    return Building(**d)


def test_intensity_arithmetic():
    r = assess_building(office())
    g = CF.electricity_location("GBR")["g_per_kwh"]
    gas = CF.desnz("desnz26:fuels-gaseous-fuels-natural-gas-kwh-gross-cv", "AR6")["kgco2e_per_unit"]
    assert r["eui_kwh_m2"] == pytest.approx(150)
    assert r["carbon_intensity_kgco2e_m2"] == pytest.approx((100000 * g / 1000 + 50000 * gas) / 1000, rel=1e-3)


def test_grid_decarbonises_and_retrofit_electrifies():
    base = assess_building(office(), scenario="Net Zero 2050")
    retro = assess_building(office(retrofits=[{"year": 2028, "energy_reduction_pct": 20, "electrify_heat": True}]),
                            scenario="Net Zero 2050")
    assert base["trajectory_kgco2e_m2"][-1] < base["trajectory_kgco2e_m2"][0]
    i = base["years"].index(2030)
    assert retro["trajectory_kgco2e_m2"][i] < base["trajectory_kgco2e_m2"][i]


def test_stranding_year():
    r = assess_building(office(), {2025: 100, 2050: 0}, "test", carbon_price_usd_t=100)
    s = r["stranding"]
    assert s["aligned_today"] is True and s["stranding_year"] is not None and s["stranding_year"] > 2025
    assert s["excess_cost_usd_at_price"] == pytest.approx(s["cumulative_excess_tco2e"] * 100, rel=1e-3)
    assert assess_building(office(), {2025: 1000, 2050: 1000})["stranding"]["stranding_year"] is None


def test_regulatory_checks():
    nl = assess_building(office(country="NLD", epc_rating="D"))
    assert any(x["rule"].startswith("NL offices") and x["status"] == "breach" for x in nl["regulatory"])
    uk = assess_building(office(epc_rating="F"))
    assert any(x["rule"].startswith("MEES") and x["status"] == "breach" for x in uk["regulatory"])
    assert any(x["rule"].startswith("MEES") and x["status"] == "ok" for x in assess_building(office(epc_rating="C"))["regulatory"])
    fr = assess_building(office(country="FRA", reference_year_final_energy_kwh=400000))
    assert any(x["rule"].startswith("Décret") and x["status"] == "ok" for x in fr["regulatory"])


def test_validation_and_portfolio():
    with pytest.raises(REError):
        assess_building(office(floor_area_m2=0))
    with pytest.raises(REError):
        assess_building(office(country="DEU", district_heat_kwh=1000))
    p = assess_portfolio([office(id="a"), office(id="b", floor_area_m2=3000)], {"*": {"source": "t", "kgco2e_m2": {"2025": 1, "2050": 0}}})
    assert p["n_buildings"] == 2 and p["share_of_area_stranded_by_year_pct"][2025] == 100.0


def test_api():
    body = {"buildings": [{"id": "x", "name": "x", "country": "USA", "us_state": "CA", "property_type": "Office",
                           "floor_area_m2": 2000, "electricity_kwh": 300000}],
            "pathways": {"USA|Office": {"source": "t", "kgco2e_m2": {"2025": 50, "2050": 2}}}}
    r = c.post("/real-estate/assess", json=body)
    assert r.status_code == 200 and r.json()["buildings"][0]["grid"]["source"].startswith("US EPA eGRID2023 subregion CAMX")
    assert c.post("/real-estate/assess", json={"buildings": [{"id": "x"}]}).status_code == 422
    assert c.get("/real-estate/dashboard").status_code == 200
