"""Climate scenario composer, its application to the 3D twin, and the agent's free-form scenario command (all data stubbed)."""
import time

import pytest

from cri import climate_scenario as CS
from tests import test_asset_twin as TA

RAIN = {"p10": 0.95, "median": 1.14, "p90": 1.33, "baseline_mm_day": 90.0, "cell": [22.25, 91.75]}
SEA = {"available": True, "location": "gauge X", "km": 5.0, "gauge": True, "sea_km": 4.0, "elevation_m": 6.0, "add_m": {"p17": 0.15, "median": 0.30, "p83": 0.50, "p95": 0.63}}


@pytest.fixture()
def stub(monkeypatch):
    monkeypatch.setattr(CS, "_rain", lambda *a, **k: dict(RAIN))
    monkeypatch.setattr(CS, "_heat", lambda *a, **k: {"p10": 1.0, "median": 2.0, "p90": 3.0, "baseline_c": 35.0})
    monkeypatch.setattr(CS, "_sea", lambda *a, **k: dict(SEA))
    monkeypatch.setattr(CS, "_wind", lambda lat, lon, ssp, year: {"tc_exposed": True, "warming_k": 2.0, "scale": {"median": 1.05, "low": 1.01, "high": 1.10}, "note": "x"})


def test_period_and_warming():
    assert CS.period_of(2070) == "2060-2079" and CS.period_of(2099) == "2080-2099" and CS.period_of(2025) == "2020-2039"
    assert CS.warming_at("ssp585", 2090) == pytest.approx(3.55) and CS.warming_at("ssp585", 2015) == 0.0 and CS.warming_at("ssp126", 2200) == pytest.approx(0.95)


def test_compose_derives_river_factor_and_grades(stub):
    c = CS.compose(22.27, 91.80, "ssp585", 2070)
    assert c["river_depth_factor"]["median"] == pytest.approx(1.14 ** 0.6)
    assert {x["factor"] for x in c["confidence"]} >= {"Heavier extreme rain", "River flood depth", "Sea-level rise", "Cyclone wind"}
    assert next(x for x in c["confidence"] if x["factor"] == "River flood depth")["level"] == "low"
    with pytest.raises(ValueError):
        CS.compose(0, 0, "ssp999", 2070)


def test_forcing_coastal_uses_sea_level_and_river_uses_manning(stub):
    comp = CS.compose(22.27, 91.80, "ssp245", 2070)
    tw = {"scenarios": {"flood": [{"id": "river_100", "kind": "river", "label": "River 1-in-100", "rp": 100, "year": None, "depth_m": 1.0, "depth_p90_m": 1.8, "wet_share": 0.5, "q": [0] * 21},
                                   {"id": "coastal_2020_100", "kind": "coastal", "label": "Coast 1-in-100", "rp": 100, "year": 2020, "depth_m": 1.0, "depth_p90_m": 1.5, "wet_share": 0.4, "q": [0] * 21}]}}
    f = CS.twin_forcing(tw, comp, 100, "coastal", "median"); assert f["scenario_id"] == "coastal_2020_100" and f["add_m"] == pytest.approx(0.30)
    assert CS.twin_forcing(tw, comp, 100, "coastal", "high")["add_m"] == pytest.approx(0.50)
    f = CS.twin_forcing(tw, comp, 100, "river", "median"); assert f["scenario_id"] == "river_100" and f["add_m"] == pytest.approx(round(1.8 * (1.14 ** 0.6 - 1), 2))
    assert f["wind_scale"] == pytest.approx(1.05)
    assert CS.twin_forcing({"scenarios": {"flood": []}}, comp)["scenario_id"] is None


def test_apply_to_real_twin_more_water_means_not_fewer_wet_structures(stub):
    tw = TA.twin()
    comp = CS.compose(TA.LAT, TA.LON, "ssp585", 2070)
    r = CS.apply_to_twin(tw, comp, 100, "river", "high", run_solver=False)
    assert r["forcing"]["scenario_id"] == "river_100" and r["forcing"]["add_m"] > 0
    aff = lambda x: x["wet"] + x["equipment_inundated"]                # a wet floor can become an inundated one: judge the sum
    assert aff(r["future"]) >= aff(r["now"]) and r["future"]["equipment_inundated"] >= r["now"]["equipment_inundated"] and r["future"]["solver"] is False


def test_agent_understands_free_form_scenario(stub, monkeypatch):
    from cri.resident import agent as AG
    ag = AG.Agent(twin_builder=lambda **k: {}, analyser=lambda t: {}, scanner=lambda **k: {}, lab=lambda *a: {})
    monkeypatch.setattr(AG, "resolve_place", lambda t, e, s: {"lat": 22.27, "lon": 91.80, "name": "Chattogram"})
    r = ag.handle("s", "what does the worst case SSP5-8.5 world look like at Chattogram in 2070 for a 1-in-100 flood?")
    assert r.get("job") and r["intent"] == "scenario"
    for _ in range(100):
        j = AG.job_status(r["job"])
        if j["status"] != "running":
            break
        time.sleep(0.1)
    assert j["status"] == "done", j
    res = AG.JOBS[r["job"]]["result"]
    assert "climate=ssp585:2070:100" in res["actions"][0]["url"] and res["blocks"][0]["rows"][0][3]
    assert ag.handle("s", "what is the flood risk at Dhaka") .get("intent") != "scenario"
