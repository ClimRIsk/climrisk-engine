"""Abatement response and earnings layer — offline, hand-checked arithmetic."""
import numpy as np
import pytest

from cri.finance import abatement, earnings, transition
from cri.finance.distribution import simulate

NO_FLOOD = {"river": {"available": False, "reason": "test"}, "coastal": {"available": False, "reason": "test"}}
Y = len(transition.YEARS)


def _lever(**kw):
    d = dict(name="L", share=0.3, mac=40.0, lead=2, avail=2026, life=20, opex_share=0.6)
    d.update(kw)
    return abatement.user_levers([d])


def _sample(levers, n=4, seed=1):
    return abatement.sample(levers, np.random.default_rng(seed), n)


def test_no_price_no_abatement():
    S = _sample(_lever())
    R = abatement.respond(np.zeros((4, Y)), np.full((4, Y), 1000.0), S, 0.07)
    assert R["frac"].sum() == 0 and R["cost"].sum() == 0 and R["capex"].sum() == 0


def test_lever_starts_lead_years_after_trigger_and_costs_mac_per_tonne():
    S = _sample(_lever(share=0.3, mac=40.0, lead=2))
    q = np.zeros((4, Y))
    q[:, 5:] = 100.0                                   # price reaches 100 in year index 5 (2031)
    Ae = np.full((4, Y), 1000.0)
    R = abatement.respond(q, Ae, S, 0.07)
    assert (R["starts"] == 7).all()                    # trigger 5 + lead 2
    assert (R["frac"][:, :7] == 0).all() and np.allclose(R["frac"][:, 7:], 0.3)
    assert np.allclose(R["cost"][:, 7], 1000 * 0.3 * 40.0)                   # tonnes abated × levelised cost
    assert np.allclose(R["opex"][:, 7], 0.6 * 1000 * 0.3 * 40.0)
    af = (1 - 1.07 ** -20) / 0.07                                              # capital part, paid the year before start
    assert np.allclose(R["capex"][:, 6], 0.4 * 40.0 * 300 * af)
    assert R["capex"][:, :6].sum() == 0 and R["capex"][:, 7:].sum() == 0


def test_availability_year_gates_the_lever():
    S = _sample(_lever(avail=2030, lead=1))
    R = abatement.respond(np.full((4, Y), 500.0), np.full((4, Y), 1.0), S, 0.07)
    assert (R["starts"] == 4 + 1).all()                # available from 2030 (index 4) even though price was high from day 1


def test_price_below_cost_never_triggers_and_shares_capped():
    S = _sample(_lever(mac=200.0))
    assert abatement.respond(np.full((4, Y), 150.0), np.ones((4, Y)), S, 0.07)["frac"].sum() == 0
    many = abatement.user_levers([{"name": f"l{i}", "share": 0.5, "mac": 10} for i in range(4)])
    assert (_sample(many)["share"].sum(axis=1) <= abatement.MAX_ABATED + 1e-9).all()


def test_library_covers_sectors_and_user_levers_validate():
    assert abatement.library_for("cement", "cement")["levers"]
    assert abatement.library_for("steel", "steel:EAF")["key"] == "steel:EAF"
    assert abatement.library_for(None, None)["levers"] == []
    with pytest.raises(ValueError):
        abatement.user_levers([{"name": "bad", "share": 1.4, "mac": 10}])
    with pytest.raises(ValueError):
        abatement.user_levers([{"name": "bad", "share": 0.2, "mac": [50, 10, 90]}])


def _case(abate=True, emissions=1e6, value=1e9):
    drv = transition.drivers("DEU", "cement", "", "cement")
    return {"value": {"low": value, "mode": value, "high": value}, "tc": {"lam": 0.0, "sigma": None, "region": "ROW"},
            "transition": {"emissions_t": emissions, "explicit_price": 50.0, "charged_share_path": (np.ones(Y),) * 3,
                           "drivers": drv, "pass_through_key": "cement", "activity_key": "cement",
                           "abatement": {"enabled": abate, "levers": abatement.library_for("cement", "cement")["levers"]}}}


def test_abatement_never_worse_on_average_and_off_switch_matches_unabated():
    on, off = simulate(_case(True), n=800, seed=5), simulate(_case(False), n=800, seed=5)
    for s, sc in on["scenarios"].items():
        assert sc["total_npv"]["mean"] <= sc["total_npv_unabated"]["mean"] + 1.0, s
        assert off["scenarios"][s]["total_npv"]["mean"] == pytest.approx(sc["total_npv_unabated"]["mean"], rel=1e-9)
        assert off["scenarios"][s]["abatement"] == {"enabled": False}
    nz = on["scenarios"]["Net Zero 2050"]
    assert nz["total_npv"]["mean"] < nz["total_npv_unabated"]["mean"]                # a real response in NZE
    assert nz["abatement"]["abated_share_by_year"]["2050"] > 0.3
    assert on["scenarios"]["Current Policies"]["abatement"]["abated_share_by_year"]["2050"] <= \
        nz["abatement"]["abated_share_by_year"]["2050"]


def test_zero_emissions_gives_no_abatement_block():
    r = simulate(_case(True, emissions=0.0), n=300)
    assert all(sc["abatement"] == {"enabled": False} for sc in r["scenarios"].values())


# ── earnings ────────────────────────────────────────────────────────────────

def _fin(**kw):
    d = {"revenue_usd": 100e6, "ebitda_usd": 20e6, "da_usd": 5e6, "interest_usd": 2e6, "net_debt_usd": 10e6,
         "book_value_usd": 100e6, "tax_rate": 0.25}
    d.update(kw)
    return earnings.normalise(d, "cement")


def _ctx(n=3, pay=10e6):
    return {"Amin": np.ones((n, Y)), "pay": np.full((n, Y), pay), "ab_opex": np.zeros((n, Y)), "capex": np.zeros((n, Y)),
            "pt": np.full((n, 1), 0.5), "disc": 1 / 1.07 ** (transition.YEARS - 2025), "years": transition.YEARS,
            "discount": 0.07, "drv": {"energy": {}}, "scenario": "Net Zero 2050", "models": ["m"], "mi": np.zeros(n, int)}


def test_earnings_arithmetic_by_hand():
    out = earnings.compute(_fin(covenant_max_leverage=1.5), _ctx())
    # EBITDA 20m − 10m carbon = 10m every year; after-tax cash shortfall 7.5m a year adds to net debt
    assert out["ebitda"]["median"][0] == pytest.approx(10e6)
    assert out["net_debt"]["median"][0] == pytest.approx(10e6 + 7.5e6)
    assert out["net_debt"]["median"][1] == pytest.approx(10e6 + 15e6)
    assert out["leverage"]["median"][0] == pytest.approx(1.75) and out["leverage"]["baseline"] == pytest.approx(0.5)
    interest0 = 2e6 + 7.5e6 * (2e6 / 10e6)                                      # implied rate 20 % on the extra debt
    assert out["coverage_ebitda"]["median"][0] == pytest.approx(10e6 / interest0, abs=0.01)   # reported to 2 d.p.
    cr = out["credit"]
    assert cr["baseline"]["rating"] == "AA" and cr["baseline"]["coverage_ebit"] == pytest.approx(7.5)   # (20−5)/2
    assert earnings.RATINGS[int(np.searchsorted(earnings.ICR_BOUNDS, 7.5))] == "AA"
    assert cr["rating_median"][0] == "B-"                                        # EBIT 5m / interest 3.5m = 1.43
    assert cr["pd_mean"][0] == pytest.approx(0.0327)
    assert cr["ecl_mean_usd"][0] == pytest.approx(0.0327 * 0.45 * 17.5e6, rel=1e-3)
    assert out["covenants"]["prob_in_breach"][0] == 1.0 and out["covenants"]["first_breach_year_median"] == 2026
    assert out["channels_pv_mean"]["carbon_payments"] == pytest.approx(
        10e6 * (1 / 1.07 ** (transition.YEARS - 2025)).sum())


def test_baseline_rating_and_pd_floor():
    out = earnings.compute(_fin(), _ctx(pay=0.0))
    assert out["credit"]["baseline"] == {"coverage_ebit": 7.5, "rating": "AA", "pd": 0.0003}   # Basel floor on AA (0.02 %)
    assert out["credit"]["rating_notches_vs_baseline_median"][0] == 0
    assert (np.array(out["credit"]["pd_mean"]) >= 0.0003).all()
    assert out["impairment"]["climate_attributable"]["mean"] == 0


def test_climate_impairment_is_scenario_minus_baseline():
    out = earnings.compute(_fin(book_value_usd=100e6), _ctx())
    ann = (1 - 1.07 ** -25) / 0.07
    assert out["impairment"]["value_in_use"]["p50"] == pytest.approx(7.5e6 * ann, rel=1e-6)   # no sustaining capex given
    assert out["impairment"]["value_in_use_baseline"] == pytest.approx(15e6 * ann, rel=1e-6)
    assert out["impairment"]["climate_attributable"]["mean"] == pytest.approx(100e6 - 7.5e6 * ann, rel=1e-6)
    assert out["impairment"]["baseline_already_impaired"] is False
    assert any("Sustaining capex" in n for n in out["notes"])


def test_plant_closes_when_ebitda_turns_negative():
    out = earnings.compute(_fin(), _ctx(pay=30e6))                              # carbon 30m > EBITDA 20m
    assert out["prob_closed"][0] == 1.0 and out["ebitda"]["median"][5] == 0.0
    assert out["channels_pv_mean"]["carbon_payments"] == 0.0                    # payments stop after closure


def test_energy_channels_use_ngfs_price_differences():
    n = 3
    ctx = _ctx(n, pay=0.0)
    ctx["models"] = ["m"]
    ctx["drv"] = {"energy": {"Net Zero 2050": {"m": {"electricity": {"delta_usd_gj": np.full(Y, 5.0), "ratio": np.ones(Y)},
                                                       "coal": {"delta_usd_gj": np.full(Y, 2.0), "ratio": np.full(Y, 0.5)}}}}}
    f = _fin(electricity_mwh=100000, fuel="coal", fuel_gj=1e6, commodity="coal", price_realisation=1.0)
    out = earnings.compute(f, ctx)
    # power: 5 $/GJ × 3.6 GJ/MWh × 100 000 MWh × (1 − 0.5 pass-through) = 0.9m; fuel: 2 × 1e6 × 0.5 = 1m;
    # commodity: 100m revenue × (0.5 − 1) = −50m  → EBITDA falls by 51.9m ⇒ closure
    assert out["prob_closed"][0] == 1.0
    ctx2 = dict(ctx)
    f2 = _fin(electricity_mwh=100000, fuel="coal", fuel_gj=1e6, commodity="none")
    out2 = earnings.compute(f2, ctx2)
    assert out2["ebitda"]["median"][0] == pytest.approx(20e6 - 0.9e6 - 1.0e6)


def test_financials_validation():
    for bad in ({"revenue_usd": 0, "ebitda_usd": 1}, {"ebitda_usd": 1}, {"revenue_usd": 1e6, "ebitda_usd": 1e5, "tax_rate": 0.9},
                {"revenue_usd": 1e6, "ebitda_usd": 1e5, "fuel_gj": 100}, {"revenue_usd": 1e6, "ebitda_usd": 1e5, "fuel": "wood"},
                {"revenue_usd": "abc", "ebitda_usd": 1}):
        with pytest.raises(earnings.FinancialsError):
            earnings.normalise(bad, "cement")
    assert earnings.normalise({"revenue_usd": 1e6, "ebitda_usd": 1e5}, "coal-mining")["commodity"] == "coal"
    assert earnings.normalise({"revenue_usd": 1e6, "ebitda_usd": 1e5}, "oil-and-gas")["commodity"] == "oil"


def test_analyse_asset_with_financials_offline():
    from cri.finance.analyst import analyse_asset
    f = {"revenue_usd": 300e6, "ebitda_usd": 75e6, "da_usd": 25e6, "interest_usd": 12e6, "net_debt_usd": 150e6,
         "book_value_usd": 400e6, "tax_rate": 0.25, "electricity_mwh": 150000, "fuel": "coal", "fuel_gj": 4e6,
         "sustaining_capex_usd": 15e6, "covenant_max_leverage": 3.5, "covenant_min_icr": 3.0}
    kw = dict(name="Test cement", sector="cement", atype="plant", emissions_t=6e5, value_usd=5e8, iso3="PHL", n=400,
              _flood=NO_FLOOD)
    r = analyse_asset(16.2, 120.4, financials=f, **kw)
    d = r["financial"]["scenarios"]["Delayed transition"]
    assert d["earnings"]["credit"]["baseline"]["rating"] == "A-" and d["abatement"]["enabled"]
    assert any(a["hazard"] == "abatement" for a in r["actions"])
    assert any("Earnings" in line for line in r["narrative"])
    assert "earnings_input" in r and not any("held at today" in str(r["transition"]["assumptions"]["emissions"]) for _ in [0])
    plain = analyse_asset(16.2, 120.4, **kw)
    assert plain["financial"]["scenarios"]["Delayed transition"]["earnings"] is None
    assert any("No financials" in c for c in plain["caveats"])
    off = analyse_asset(16.2, 120.4, abatement_on=False, **kw)
    assert off["financial"]["scenarios"]["Delayed transition"]["abatement"] == {"enabled": False}
    assert off["transition"]["abatement"]["enabled"] is False
    with pytest.raises(ValueError):
        analyse_asset(16.2, 120.4, financials={"revenue_usd": 100}, **kw)
    own = analyse_asset(16.2, 120.4, levers=[{"name": "Own CCS", "share": 0.5, "mac": [30, 40, 50], "lead": 3}], **kw)
    assert [l["name"] for l in own["transition"]["abatement"]["levers"]] == ["Own CCS"]
    assert own["transition"]["abatement"]["user_levers"] is True
