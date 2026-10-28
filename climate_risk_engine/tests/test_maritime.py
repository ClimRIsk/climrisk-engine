"""Voyage model: scaling laws, scope rules, nothing defaulted, ship records, sentence parsing."""
import math
import os

import pytest

os.environ.setdefault("CRI_API_KEYS", "")
from fastapi.testclient import TestClient          # noqa: E402

from cri.api.main import app                       # noqa: E402
from cri.lca import marine_fuels as MF             # noqa: E402
from cri.maritime import answer as A, vessel as VS, voyage as V   # noqa: E402

SG = {"name": "Singapore", "lat": 1.26, "lon": 103.8, "country_code": "SG"}
RT = {"name": "Rotterdam", "lat": 51.95, "lon": 4.14, "country_code": "NL"}
HM = {"name": "Hamburg", "lat": 53.54, "lon": 9.98, "country_code": "DE"}
SH = {"name": "Shanghai", "lat": 31.2, "lon": 121.5, "country_code": "CN"}
SHIP = {"name": "T", "reference_speed_kn": 14.0, "fuel_kg_per_nm_ref": 180.0, "gt": 80000}


def test_cube_law_scaling_and_reference():
    r = V.assess_voyage(SHIP, 1000, 14, "hfo", SG, RT, 2026)
    assert r["fuel"]["t"] == pytest.approx(180.0)                         # at the reference speed: record x distance
    r2 = V.assess_voyage(SHIP, 1000, 7, "hfo", SG, RT, 2026)
    assert r2["fuel"]["t"] == pytest.approx(180.0 / 4)                    # half speed, quarter fuel for the same distance
    assert r2["voyage"]["days"] == pytest.approx(2 * r["voyage"]["days"])


def test_emissions_match_the_fuel_library():
    r = V.assess_voyage(SHIP, 1000, 14, "hfo", SG, RT, 2026)
    w = MF.wtw("hfo")
    assert r["emissions"]["ttw_co2e_t"] == pytest.approx(180.0 * 1e6 * w["lcv_mj_per_g"] * w["ttw"] / 1e6)
    assert r["emissions"]["wtw_co2e_t"] > r["emissions"]["ttw_co2e_t"]


def test_scope_rules():
    assert V.scope_share("NL", "DE", V.EEA) == 1.0
    assert V.scope_share("SG", "NL", V.EEA) == 0.5
    assert V.scope_share("SG", "CN", V.EEA) == 0.0
    assert V.scope_share("NO", "NL", V.EEA) == 1.0 and V.scope_share("NO", "NL", V.EU27) == 0.5     # Norway: ETS yes, FuelEU Member State no
    assert V.scope_share(None, "NL", V.EEA) is None


def test_ets_phase_in_and_basis():
    r25 = V.assess_voyage(SHIP, 1000, 14, "hfo", SG, RT, 2025, {"eua_price": 80, "currency": "EUR"})
    r26 = V.assess_voyage(SHIP, 1000, 14, "hfo", SG, RT, 2026, {"eua_price": 80, "currency": "EUR"})
    assert r25["ets"]["phase_in"] == 0.7 and r26["ets"]["phase_in"] == 1.0
    assert r25["ets"]["chargeable_t"] == pytest.approx(r25["emissions"]["ttw_co2_t"] * 0.5 * 0.7)       # CO2 only before 2026
    assert r26["ets"]["chargeable_t"] == pytest.approx(r26["emissions"]["ttw_co2e_t"] * 0.5)
    assert r26["ets"]["cost"] == pytest.approx(r26["ets"]["chargeable_t"] * 80) and r26["ets"]["price_currency"] == "EUR"


def test_out_of_scope_voyage_has_no_ets_or_fueleu():
    r = V.assess_voyage(SHIP, 1000, 14, "hfo", SG, SH, 2026)
    assert r["ets"]["chargeable_t"] == 0 and r["fueleu"]["energy_in_scope_mj"] == 0
    assert r["scenarios"]["diversion"] is None or r["scenarios"]["diversion"]["extra_distance_pct"] is not None


def test_below_5000_gt_is_out_of_ets():
    r = V.assess_voyage({**SHIP, "gt": 3000}, 1000, 14, "hfo", SG, RT, 2026)
    assert r["ets"]["chargeable_t"] == 0 and "5,000" in r["ets"]["status"]


def test_fueleu_deficit_for_hfo_and_surplus_for_low_ghg_fuel():
    r = V.assess_voyage(SHIP, 1000, 14, "hfo", RT, HM, 2026)
    assert r["fueleu"]["scope_share"] == 1.0 and r["fueleu"]["compliance_balance_tco2e"] < 0
    expect = (r["fueleu"]["limit_gco2e_per_mj"] - r["fueleu"]["intensity_gco2e_per_mj"]) * r["fueleu"]["energy_in_scope_mj"] / 1e6
    assert r["fueleu"]["compliance_balance_tco2e"] == pytest.approx(expect)
    assert r["fueleu"]["penalty_eur"] is None                                                 # no rate given, none invented


def test_costs_only_from_given_inputs():
    r = V.assess_voyage(SHIP, 1000, 14, "hfo", SG, RT, 2026, {})
    assert r["cost"]["lines"] == [] or all("ETS" in l["item"] for l in r["cost"]["lines"])
    assert "bunker price (USD/t)" in r["cost"]["missing_inputs"]
    r = V.assess_voyage(SHIP, 1000, 14, "hfo", SG, RT, 2026, {"bunker_usd_per_t": 500, "charter_usd_per_day": 10000})
    bunker = next(l for l in r["cost"]["lines"] if l["item"] == "Bunker fuel")
    assert bunker["usd"] == pytest.approx(180 * 500)
    assert not any("charter" in m for m in r["cost"]["missing_inputs"])


def test_missing_record_is_an_error_not_a_guess():
    with pytest.raises(V.VoyageInputError) as e:
        V.assess_voyage({"name": "X"}, 1000, 14, "hfo", SG, RT, 2026)
    assert e.value.missing


def test_distance_default_is_flagged_lower_bound():
    r = V.assess_voyage(SHIP, None, 14, "hfo", SG, RT, 2026)
    assert any("great-circle" in f for f in r["flags"]) and r["voyage"]["distance_nm"] == pytest.approx(r["voyage"]["great_circle_nm"])
    r = V.assess_voyage(SHIP, 100, 14, "hfo", SG, RT, 2026)
    assert any("shorter than the great-circle" in f for f in r["flags"])


def test_fuel_not_given_is_flagged_and_alternatives_listed():
    r = V.assess_voyage(SHIP, 1000, 14, None, SG, RT, 2026)
    assert any("Fuel type not given" in f for f in r["flags"]) and len(r["emissions"]["alt_reference_fuels"]) == 2


def test_diversion_default_only_on_suez_relevant_routes():
    assert V.suez_relevant(SG, RT) and not V.suez_relevant(SG, SH) and not V.suez_relevant(RT, HM)
    assert V.assess_voyage(SHIP, 1000, 14, "hfo", SG, RT, 2026)["scenarios"]["diversion"]["extra_distance_pct"] == 30.0
    assert V.assess_voyage(SHIP, 1000, 14, "hfo", RT, HM, 2026)["scenarios"]["diversion"] is None
    d = V.assess_voyage(SHIP, 1000, 14, "hfo", RT, HM, 2026, diversion_pct=10)["scenarios"]["diversion"]
    assert d["distance_nm"] == pytest.approx(1100) and "your figure" in d["basis"]


def test_delay_recovery_speed_is_distance_over_remaining_time():
    r = V.assess_voyage(SHIP, 3360, 14, "hfo", SG, RT, 2026, delay_days=[2])   # 3360/14 = 240 h = 10 d
    x = r["scenarios"]["delay"][0]
    assert x["speed_to_recover_kn"] == pytest.approx(3360 / (8 * 24))
    assert x["extra_fuel_t"] > 0 and x["label"] == "your figure"


def test_fleet_loader_matches_columns_and_reports_the_rest():
    csv = "IMO Number,Name,Ship type,Gross tonnage,Total fuel consumption [m tonnes],Total distance travelled [n miles],Total time spent at sea [hours],Colour\n9000001,ALPHA,Bulk carrier,60000,\"4,000\",\"40,000\",\"3,200\",red\n9000002,ALPHA TWO,Tanker,70000,,,,\n"
    L = VS.load_fleet_csv(csv)
    assert "Colour" in L["ignored_columns"] and L["matched_columns"]["imo"] == "IMO Number"
    a, b = L["ships"]
    assert a["reference_speed_kn"] == pytest.approx(12.5) and a["fuel_kg_per_nm_ref"] == pytest.approx(100.0) and a["missing"] == []
    assert set(b["missing"]) == {"reference_speed_kn", "fuel_kg_per_nm_ref"}
    assert VS.find("9000001", fleets=L["ships"])["vessel"]["name"] == "ALPHA"
    amb = VS.find("alpha", fleets=L["ships"])
    assert amb["vessel"] is None or amb["vessel"]["name"] == "ALPHA"
    assert VS.find("9999999", fleets=L["ships"])["vessel"] is None


def test_find_with_no_fleet_files_says_how_to_add(tmp_path):
    f = VS.find("9000001", data_dir=tmp_path)
    assert f["vessel"] is None and "data/vessels" in f["why"]


# ── sentence parsing and the agent-facing answer ─────────────────────────────

def _geo(q):
    t = q.lower().replace("port of ", "")
    return {"singapore": {"lat": 1.26, "lon": 103.8, "country_code": "sg"}, "rotterdam": {"lat": 51.95, "lon": 4.14, "country_code": "nl"}}.get(t)


def test_sentence_parsing():
    p = A.parse("The ship MV Aurora is docked at the Port of Singapore and is moving to Rotterdam, 16000 NM away at 14 knots, burns 45 t/day at 14 knots, bunker $600/t, charter $20,000/day, using VLSFO")
    assert (p["name"], p["origin"], p["dest"], p["distance_nm"], p["speed_kn"]) == ("Aurora", "Singapore", "Rotterdam", 16000.0, 14.0)
    assert p["daily_t"] == 45 and p["market"] == {"bunker_usd_per_t": 600.0, "charter_usd_per_day": 20000.0} and p["fuel"] == "hfo"
    assert A.matches("a ship docked in Mumbai port is moving to Hamburg port, 16000 NM")
    assert not A.matches("what is the flood risk at 22.2, 91.8")


def test_answer_without_a_ship_record_gives_what_it_can_and_asks_for_the_rest(tmp_path):
    r = A.answer("Vessel Ever Given is docked in Singapore and moving to Rotterdam, 16000 NM at 14 knots", _geo, data_dir=tmp_path)
    assert r["result"] is None
    assert "fuel burn" in r["reply"] and "50%" in r["reply"] and "47.6 days" in r["reply"]
    txt = " ".join(b.get("text", "") for b in r["blocks"])
    assert "I have not guessed a consumption" in txt


def test_answer_with_consumption_runs_the_full_model(tmp_path):
    r = A.answer("MV Aurora is docked at Singapore and moving to Rotterdam, 16000 NM at 14 knots, burns 45 t/day at 14 knots, bunker $600/t, charter $20000/day", _geo, data_dir=tmp_path)
    x = r["result"]
    assert x is not None and x["fuel"]["t"] == pytest.approx(45 * 16000 / (14 * 24))
    assert "EU ETS" in r["reply"] and r["blocks"]


def test_answer_unknown_port_is_reported(tmp_path):
    r = A.answer("ship X is docked at Nowhereville and moving to Rotterdam, 100 NM", _geo, data_dir=tmp_path)
    assert r["result"] is None and "Nowhereville" in r["reply"]


def test_api_route_needs_a_ship_record_and_returns_422_with_what_is_missing():
    c = TestClient(app)
    body = {"origin": SG, "dest": RT, "distance_nm": 1000, "speed_kn": 14}
    r = c.post("/maritime/voyage", json=body)
    assert r.status_code == 422 and r.json()["detail"]["missing"]
    r = c.post("/maritime/voyage", json={**body, "reference_speed_kn": 14, "daily_consumption_t": 45, "fuel": "hfo", "year": 2026, "bunker_usd_per_t": 600})
    assert r.status_code == 200
    j = r.json()
    assert j["fuel"]["t"] == pytest.approx(45 * 1000 / (14 * 24))
    assert c.get("/maritime/fleet").status_code == 200


def test_agent_routes_a_ship_sentence_to_the_voyage_model(monkeypatch, tmp_path):
    import time
    from cri.resident import agent as AG
    monkeypatch.setattr(AG, "geocode", _geo)
    monkeypatch.setattr(VS, "DATA_DIR", tmp_path)
    out = AG.Agent().handle("t-voyage", "The ship Aurora is docked at Singapore and moving to Rotterdam, 16000 NM at 14 knots, burns 45 t/day at 14 knots")
    assert out["intent"] == "voyage" and out["job"]
    for _ in range(100):
        st = AG.job_status(out["job"])
        if st["status"] != "running":
            break
        time.sleep(0.2)
    assert st["status"] == "done", st
    assert "EU ETS" in st["reply"] and st["blocks"]
