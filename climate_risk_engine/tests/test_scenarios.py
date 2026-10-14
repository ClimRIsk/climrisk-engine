"""Scenario suite: client questions across sectors and regions, checked for contract (nothing silently left out), plus unit tests for the scope tables."""
import os
import tempfile

import pytest

os.environ.setdefault("CRI_CACHE_DIR", tempfile.mkdtemp())

from cri import perilscope as P
from cri import scenario_suite as SS
from cri import siteprofile as SP

IDS = [s["id"] for s in SS.scenarios()]


@pytest.mark.parametrize("sid", IDS)
def test_scenario_contract_offline(sid):
    r = SS.check_offline(SS.get(sid), n_years=300)
    assert r["passed"], r["findings"]
    for row in r["result"]["sites"]:
        sm = row["peril_scope"]["summary"]
        assert sm["primary"] >= 1 and sm["statement"]
        assert sm["sector"] != "generic"


def test_engine_table_matches_hazard_registry():
    keys = set(SP.hazard_keys())
    for p, e in P.ENGINE.items():
        assert e["status"] in ("loss_model", "hazard_only", "screen", "none")
        assert all(k in keys for k in e["keys"]), (p, e["keys"])
        if e["status"] == "none":
            assert not e["keys"], p


def test_every_sector_and_region_peril_is_in_the_engine_table():
    for sec, items in P.SECTOR.items():
        for p, role, ev in items:
            assert p in P.ENGINE, (sec, p)
            assert role in ("primary", "secondary") and ev.startswith(("src:", "judgement:"))
    for rg, ps in P.REGION.items():
        assert all(p in P.ENGINE for p in ps), rg


def test_regions_use_country_not_boxes():
    assert P.regions(45.40, -75.70) == ["canada"]                      # Ottawa is not the United States
    assert "south_southeast_asia" in P.regions(-2.6, 121.4)            # Sulawesi
    assert "south_southeast_asia" in P.regions(1.30, 103.64)           # Singapore
    assert "gulf_middle_east" in P.regions(25.2, 55.27)
    assert "us" in P.regions(29.76, -95.37)
    assert "europe" in P.regions(50.94, 6.96) and "north_sea" not in P.regions(50.94, 6.96)


def test_cyclone_not_listed_where_none_can_occur():
    names = {x["peril"] for x in P.scope("offshore wind farm", 55.68, 7.63)["perils"]}
    assert "tropical_cyclone" not in names
    names = {x["peril"] for x in P.scope("offshore wind farm", 24.05, 120.15)["perils"]}
    assert "tropical_cyclone" in names


def test_sector_recognition():
    for text, want in (("offshore wind farm", "offshore_wind"), ("wind turbines onshore", "onshore_wind"), ("aluminium smelter", "heavy_industry"), ("petrochemical plant", "oil_gas_refining"),
                       ("manufacturing plant", "manufacturing_general"), ("copper mine", "mining"), ("data centre", "data_centre"), ("power plant", "power_grid"), ("garment factory", "textile_garment"),
                       ("thing", "generic"), (None, "generic")):
        assert P.sector_key(text) == want, (text, P.sector_key(text))


def test_freeze_is_primary_for_gulf_coast_refinery_and_has_no_loss_model():
    sc = P.scope("refinery", 29.72, -95.0)
    fr = next(x for x in sc["perils"] if x["peril"] == "extreme_cold_freeze")
    assert fr["role"] == "primary" and fr["engine_status"] == "hazard_only" and not fr["in_price"]
    assert "supply_chain_cbi" in sc["summary"]["primary_not_modelled"]


def test_solar_hail_is_primary_and_unmodelled():
    sc = P.scope("solar farm", 31.95, -102.1)
    h = next(x for x in sc["perils"] if x["peril"] == "severe_convective_hail")
    assert h["role"] == "primary" and h["engine_status"] == "none" and "Midway" in h["why"]


def test_permafrost_listed_for_northern_canada_only():
    assert "permafrost_thaw" in {x["peril"] for x in P.scope("diamond mine", 64.5, -110.3)["perils"]}
    assert "permafrost_thaw" not in {x["peril"] for x in P.scope("residential property", 51.05, -114.07)["perils"]}


def test_offshore_typhoon_site_is_withheld_not_priced_with_north_sea_curve(monkeypatch):
    from cri import insurance as I
    from cri.analysis import offshore_wind as OW
    storm = {"status": "constructed", "years": [1981, 2024], "return_levels": {}, "storm_damage_ratio_aal": {"benign": 1e-4, "central": 6e-4, "severe": 2.5e-3}, "access": {},
             "cold": {"status": "null", "reason": "x"}, "gumbel": {"wind_hub": [30.0, 2.5], "hs": [6.0, 0.7]}, "design": {"vref_ms": 50.0, "hs50_m": 9.0}, "assumptions": {}, "limits": [], "wind_source": "t", "wave_source": "t"}
    monkeypatch.setattr(OW, "hazard", lambda *a, **k: storm)
    claims = [{"amount": 2e6, "cause": "cable"}] * 5
    site = {"name": "Changhua", "lat": 24.05, "lon": 120.15, "tiv_usd": 2e9, "sector": "offshore wind farm", "offshore": {"turbines": 60, "claims": claims, "claims_exposure_turbine_years": 300}}
    r = I.run([site], n_years=300, tc_fn=lambda la, lo: {"lam": 0.8, "sigma": 9.0, "rp100_ms": 60.0}, flood_fn=lambda *a: {})
    row = r["sites"][0]
    assert row["technical_premium"] is None
    assert any("typhoon" in x["peril"] for x in row["not_assessed"])
    # the same farm in the North Sea (no cyclone basin) is priced
    site2 = dict(site, name="Horns Rev", lat=55.68, lon=7.63)
    r2 = I.run([site2], n_years=300, tc_fn=lambda la, lo: {"lam": 0.8}, flood_fn=lambda *a: {})
    assert r2["sites"][0]["technical_premium"] is not None
    assert r2["premium"]["scope_note"]                                       # lightning etc. still named as outside the number


def test_group_portfolio_states_what_the_number_leaves_out():
    r = SS.check_offline(SS.get("group_multi"), n_years=300)["result"]
    assert r["premium"]["scope_note"] and r["premium"]["sites_with_primary_perils_outside_price"]
    assert r["premium"]["complete"] is False


def _fake_cold(winter_min=-5.0, hard_year=2021, hard_min=-12.0):
    import datetime as dt, math
    days = []; d = dt.date(1980, 1, 1)
    while d <= dt.date(2024, 12, 31):
        days.append(d); d += dt.timedelta(days=1)
    tn = []
    for x in days:
        base = 12 + 14 * math.sin(2 * math.pi * (x.timetuple().tm_yday - 110) / 365.25)
        v = base - ((x.year * 37) % 11) * (0.55 if (x.month == 1 and x.day == 15) else 0.0)
        if x.year == hard_year and x.month == 2 and 14 <= x.day <= 16:
            v = hard_min
        tn.append(v)
    def fake(url, **k):
        return {"daily": {"time": [x.isoformat() for x in days], "temperature_2m_max": [v + 8 for v in tn], "temperature_2m_min": tn, "temperature_2m_mean": [v + 4 for v in tn],
                          "relative_humidity_2m_mean": [70] * len(days), "precipitation_sum": [0.0] * len(days)}}
    return fake


def test_cold_extremes_return_levels_and_design_check(monkeypatch, tmp_path):
    monkeypatch.setenv("CRI_CACHE_DIR", str(tmp_path))
    import datetime as dt
    from cri.analysis import cold_extremes as CE
    a = CE.assess(29.0, -95.0, design_min_c=-6.0, today=dt.date(2025, 6, 1), fetch=_fake_cold())
    assert a["status"] == "modelled" and a["annual_minimum_c"]["coldest"] == -12.0 and a["annual_minimum_c"]["coldest_year"] == 2021
    rl = a["return_levels_c"]
    assert rl[250] < rl[100] < rl[50] < rl[10]                                   # colder with return period
    assert a["design_check"]["years_in_record_below_design"] >= 1 and a["design_check"]["annual_exceedance_probability"] > 0
    assert CE.assess(40.0, -80.0, today=dt.date(2025, 6, 1), fetch=lambda u, **k: (_ for _ in ()).throw(RuntimeError("down")))["status"] == "null"


def test_cold_extreme_is_a_registered_hazard_and_null_not_zero():
    assert "cold_extreme" in SP.hazard_keys()
    import cri.analysis.cold_extremes as CE
    orig = CE.assess
    CE.assess = lambda *a, **k: {"status": "null", "reason": "down; cold extremes not assessed (not zero)"}
    try:
        e = SP._h_cold(SP.Ctx(29.0, -95.0, "x"))
    finally:
        CE.assess = orig
    assert e["status"] == "null" and e["value"] is None and e["severity"] is None


def test_flood_cover_flags_surface_water_lower_bound_even_with_a_map():
    from cri import insurance as I
    river = {"rps": [10, 100], "depths": {"10": [0.0], "100": [0.0]}, "protection": 0.0}
    hz = {"wind": {"lam": 0.0, "sigma": None, "rp100_ms": None}, "river": river, "coast": None, "cont": "europe", "tc_region": "ROW", "iso3": "DEU", "notes": []}
    r = I.run([{"name": "Ahrweiler", "lat": 50.5385, "lon": 7.1127, "tiv_usd": 1e7, "covers": ["flood"], "hazard": hz}], n_years=300)
    row = r["sites"][0]
    assert any("surface-water" in x["peril"] for x in row["not_assessed"]) and row["coverage_complete"] is False
    assert r["premium"]["technical_total"] is not None and "LOWER BOUND" in r["premium"]["coverage_warning"]     # a number, flagged as a lower bound
