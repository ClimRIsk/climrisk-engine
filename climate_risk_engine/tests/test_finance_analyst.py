"""Risk Analyst (cri.finance) — offline tests: no network (flood reads are stubbed)."""
import numpy as np
import pytest

from cri.finance import asset_value, geo, policy, transition, vulnerability
from cri.finance.distribution import simulate

NO_FLOOD = {"river": {"available": False, "reason": "test"}, "coastal": {"available": False, "reason": "test"}}


def test_admin_lookup_gives_jurisdiction_and_defences():
    a = geo.admin_at(51.874, 4.331)                       # Rotterdam
    assert a["iso3"] == "NLD" and a["name"] == "Zuid-Holland"
    assert a["prot_coast_rp"] >= 1000                      # Dutch delta works
    assert geo.admin_at(40.7, -73.9)["name"] == "New York"
    assert geo.continent("NLD") == "europe" and geo.continent("MEX") == "southamerica"
    assert geo.tc_region("PHL") == "WP2" and geo.tc_region("DEU") == "ROW"


def test_vulnerability_curves():
    assert float(vulnerability.tc_damage(25.7, 89.2)) == 0.0
    d = vulnerability.tc_damage([30, 50, 70, 90], 66.3)
    assert np.all(np.diff(d) > 0) and d[-1] < 1
    assert abs(float(vulnerability.flood_damage(1.0, "europe")) - 0.27) < 1e-9
    assert float(vulnerability.flood_damage(20, "asia")) == 1.0
    assert 0 < float(vulnerability.flood_damage(1.0, "oceania")) < 1        # mean-curve fallback


def test_policy_matches_state_national_and_eu_rules():
    ny = policy.applicable("USA", "New York", "electricity-generation", "gas", "power:gas")
    assert ny["price_setting_instrument"].startswith("RGGI") and ny["effective_price_usd_t"] > 0
    assert not ny["cbam_exporter"]                                          # US power is not EU-grid linked
    tx = policy.applicable("USA", "Texas", "electricity-generation", "coal", "power:coal")
    assert tx["effective_price_usd_t"] == 0 and not tx["in_force"]
    nl = policy.applicable("NLD", "Zuid-Holland", "oil-and-gas-refining", "Deep Conversion", "refining")
    assert "EU ETS industry" in nl["charged_share"]["rule"]                 # EU ETS allocation governs
    cn = policy.applicable("CHN", "Guangdong", "cement", "integrated dry", "cement")
    assert cn["price_setting_instrument"] == "China national ETS" and cn["cbam_exporter"]
    assert cn["charged_share"]["mode"] < 0.5                                # benchmark system
    eu_steel = policy.applicable("DEU", "Nordrhein-Westfalen", "iron-and-steel", "BF/BOF", "steel:BF/BOF")
    lo, mo, hi = policy.charged_share_path(eu_steel["charged_share"], [2026, 2030, 2034, 2040])
    assert lo[0] == pytest.approx(0.025) and lo[2] == 1.0 and lo[3] == 1.0  # CBAM phase-out of free allocation


def test_asset_value_ranges():
    v = asset_value.estimate("electricity-generation", "coal", 1000, 5e6, "USA")
    assert v["quality"] == "benchmark" and v["low"] < v["mode"] < v["high"]
    assert asset_value.estimate("electricity-generation", "coal", 1000, 5e6, "CHN")["mode"] < v["mode"]
    assert asset_value.estimate("x", "", None, None, "USA", value_usd=5e8)["mode"] == 5e8
    assert asset_value.estimate("ghgrp-221", "", None, 1e5, "USA")["quality"] == "proxy"


def test_ngfs_drivers():
    d = transition.drivers("IND", "electricity-generation", "coal", "power:coal")
    assert d["activity_variable"] == "Secondary Energy|Electricity|Coal"
    assert set(d["price"]) >= {"Net Zero 2050", "Current Policies"} and len(d["models"]) == 3
    nz = np.array(list(d["price"]["Net Zero 2050"].values()))
    cp = np.array(list(d["price"]["Current Policies"].values()))
    assert (nz >= 0).all() and nz[:, -1].mean() > cp[:, -1].mean()         # NZ prices end higher
    assert transition.drivers("ATA", "cement", "", "cement")["world_fallback"]


def _case(value=1e9, emissions=1e6, price=50.0, river=None):
    Y = len(transition.YEARS)
    drv = transition.drivers("DEU", "cement", "", "cement")
    return {"value": {"low": value, "mode": value, "high": value},
            "tc": {"lam": 0.0, "sigma": None, "region": "ROW"}, "river": river,
            "transition": {"emissions_t": emissions, "explicit_price": price,
                           "charged_share_path": (np.ones(Y),) * 3, "drivers": drv,
                           "pass_through_key": "cement", "activity_key": "cement"}}


def test_distribution_basics():
    r = simulate(_case(emissions=0.0), n=500)
    assert r["physical"]["aal_pct"] == 0
    assert all(s["total_npv"]["mean"] == 0 for s in r["scenarios"].values())
    r1, r2 = simulate(_case(), n=500, seed=3), simulate(_case(), n=500, seed=3)
    assert r1["scenarios"]["Net Zero 2050"]["total_npv"] == r2["scenarios"]["Net Zero 2050"]["total_npv"]
    s = r1["scenarios"]["Net Zero 2050"]
    assert s["total_npv"]["p5"] <= s["total_npv"]["p50"] <= s["total_npv"]["p95"] <= s["total_npv"]["p99"]
    # closure option: transition loss never exceeds asset value
    big = simulate(_case(value=1e8, emissions=5e7), n=500)
    assert all(v["transition_loss"]["p99"] <= 1e8 + 1 for v in big["scenarios"].values())


def test_flood_defences_hold_until_overtopped():
    river = {"rps": [10, 20, 50, 75, 100, 200, 500], "damage": [0.1, 0.2, 0.3, 0.35, 0.4, 0.45, 0.5]}
    open_ = simulate(_case(emissions=0.0, river=dict(river, protection=0)), n=2000)
    walled = simulate(_case(emissions=0.0, river=dict(river, protection=1e6)), n=2000)
    assert open_["physical"]["aal_pct"] > 1.0 and walled["physical"]["aal_pct"] == 0.0


def test_analyse_asset_end_to_end_offline():
    from cri.finance.analyst import analyse_asset, nearest_register_asset
    i = nearest_register_asset(29.4829, -95.6313)                           # W A Parish, Texas
    assert i is not None
    r = analyse_asset(29.4829, -95.6313, register_index=i, n=400, _flood=NO_FLOOD)
    assert r["asset"]["iso3"] == "USA" and r["asset"]["admin1"] == "Texas"
    assert r["physical"]["cyclone"]["lam"] > 0                              # Gulf coast storms
    assert r["policy"]["effective_price_usd_t"] == 0
    assert "Delayed transition" in r["financial"]["scenarios"]
    assert any(a["hazard"] == "stranding" for a in r["actions"])            # coal plant
    assert r["headline"] and len(r["steps"]) == 5
