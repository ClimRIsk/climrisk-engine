import math
from statistics import NormalDist

import pytest

from cri.climate import structural_credit as SC

N = NormalDist()


def test_implied_vol_reproduces_the_baseline_pd():
    s = SC.implied_asset_vol(0.01, 0.6)
    dd = (math.log(1 / 0.6) - 0.5 * s * s) / s
    assert N.cdf(-dd) == pytest.approx(0.01, abs=1e-7)


def test_shock_pd_matches_a_hand_calculation():
    pd0, L, s = 0.01, 0.6, 0.10
    sig = SC.implied_asset_vol(pd0, L)
    dd1 = (math.log((1 - s) / L) - 0.5 * sig ** 2) / sig
    r = SC.shocked_pd(pd0, L, s)
    assert r["pd1"] == pytest.approx(N.cdf(-dd1), rel=1e-9) and r["pd1"] > pd0
    assert SC.shocked_pd(pd0, L, 0.0)["pd1"] == pytest.approx(pd0, abs=1e-7)
    assert SC.shocked_pd(pd0, L, 0.2)["pd1"] > r["pd1"]                       # monotone in the shock


def test_lgd_by_hand_and_wrong_way_effect():
    o = dict(ead=100.0, baseline_pd=0.02, leverage=0.5, lgd_unsecured=0.45, collateral_value=150.0, recovery_haircut=0.3,
             physical_value_loss_fraction=0.2, collateral_damage_fraction=0.2)
    r = SC.assess(o)
    assert r["lgd"]["baseline"] == pytest.approx(0.45 * (100 - 150 * 0.7) / 100) or r["lgd"]["baseline"] == 0.0     # 150*0.7 = 105 > EAD: fully covered
    assert r["lgd"]["baseline"] == pytest.approx(0.0)
    assert r["lgd"]["climate"] == pytest.approx(0.45 * (100 - 150 * 0.8 * 0.7) / 100)                              # 84 recoverable: 0.45 * 16/100
    assert r["pd"]["climate"] > r["pd"]["baseline"] and r["ecl"]["climate"] > r["ecl"]["baseline"] == 0.0


def test_missing_inputs_are_reported_not_zeroed():
    r = SC.assess(dict(ead=100.0, baseline_pd=0.02, leverage=0.5))
    assert r["pd"]["climate"] == pytest.approx(r["pd"]["baseline"], abs=1e-9)
    assert r["ecl"] is None and r["rwa"] is None
    joined = " | ".join(r["not_assessed"])
    assert "physical shock" in joined and "transition shock" in joined and "lgd_unsecured" in joined
    with pytest.raises(SC.CreditError):
        SC.assess(dict(ead=1.0, baseline_pd=0.02))


def test_transition_shock_is_cost_over_ebitda():
    r = SC.assess(dict(ead=100.0, baseline_pd=0.02, leverage=0.5, emissions_tco2e=1e6, carbon_price_usd_per_t=50.0, ebitda=2e8, carbon_cost_pass_through=0.4, lgd_unsecured=0.45))
    assert r["shocks"]["transition"] == pytest.approx(1e6 * 50 * 0.6 / 2e8)
    assert r["pd"]["climate"] > r["pd"]["baseline"]


def test_irb_formula_known_point():
    # Basel II benchmark risk weights for corporate exposures: PD 1.00 %, LGD 45 %, M 2.5 years -> 92.32 % (the figure as I recall it from the BCBS table; the formula was coded from the text
    # of para 272 and agrees with it independently). The 1.06 factor of Basel II applies to the aggregate RWA, so it enters here only as the scaling input.
    assert SC.irb_rwa(0.01, 0.45, 1.0, 2.5, 1.0) == pytest.approx(0.9232, abs=0.001)
    assert SC.irb_rwa(0.01, 0.45, 100.0, 2.5, 1.06) == pytest.approx(92.32 * 1.06, abs=0.1)
    assert SC.irb_rwa(0.05, 0.45, 1.0) > SC.irb_rwa(0.01, 0.45, 1.0) > SC.irb_rwa(0.001, 0.45, 1.0)


def test_portfolio_and_errors():
    p = SC.portfolio([dict(ead=100.0, baseline_pd=0.02, leverage=0.5, lgd_unsecured=0.45, physical_value_loss_fraction=0.1),
                      dict(ead=50.0, baseline_pd=0.02, leverage=0.5)])
    assert p["obligors_without_lgd"] == 1 and p["ecl"]["increase"] > 0 and "LGD only" in p["note"]
    with pytest.raises(SC.CreditError):
        SC.implied_asset_vol(0.02, 1.2)
