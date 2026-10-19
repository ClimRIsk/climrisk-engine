"""Property tests (hypothesis): physical invariants that must hold for ANY input, not only the benchmark cases."""
import math
import numpy as np
import pytest
from hypothesis import given, settings, strategies as st, HealthCheck

from cri.modellab import hydrology as hy, evt, slr, flood, utci as U
from cri import wind_extreme as we

S = settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])


@S
@given(p=st.floats(0, 2000), cn=st.floats(1, 100), ia=st.floats(0, 0.5))
def test_scs_runoff_bounded_by_rainfall(p, cn, ia):
    q = hy.scs_cn_runoff(p, cn, ia)
    assert math.isfinite(q) and 0.0 <= q <= p + 1e-9


@S
@given(p=st.floats(0, 1000), dp=st.floats(0, 500), cn=st.floats(30, 98))
def test_scs_runoff_monotone_in_rain(p, dp, cn):
    assert hy.scs_cn_runoff(p + dp, cn) >= hy.scs_cn_runoff(p, cn) - 1e-9


@S
@given(p=st.floats(1, 600), cn=st.floats(30, 97), dcn=st.floats(0, 1))
def test_scs_runoff_monotone_in_cn(p, cn, dcn):
    assert hy.scs_cn_runoff(p, min(98, cn + dcn)) >= hy.scs_cn_runoff(p, cn) - 1e-9


@S
@given(xi=st.floats(-50, 200), a=st.floats(0.01, 80), t1=st.floats(1.5, 500), f=st.floats(1.0, 20))
def test_wind_return_level_monotone_in_return_period(xi, a, t1, f):
    assert we.return_level(xi, a, t1 * f) >= we.return_level(xi, a, t1) - 1e-9


@S
@given(x=st.lists(st.floats(0, 80, allow_nan=False), min_size=12, max_size=60))
def test_gumbel_lmom_never_returns_nan(x):
    xi, a = we.gumbel_lmom(np.array(x))
    assert math.isfinite(xi) and math.isfinite(a)


@S
@given(loc=st.floats(0, 100), scale=st.floats(0.1, 30), t=st.floats(1.5, 1000), f=st.floats(1.0, 10))
def test_evt_level_monotone_and_above_location(loc, scale, t, f):
    m = (loc, scale)
    assert evt.level(m, t * f) >= evt.level(m, t) - 1e-9


@S
@given(rise=st.floats(0, 5), tb=st.floats(0.005, 0.5), d=st.floats(0, 3))
def test_bruun_retreat_nonneg_and_monotone(rise, tb, d):
    assert slr.bruun_retreat(rise, tb) >= 0 and slr.bruun_retreat(rise + d, tb) >= slr.bruun_retreat(rise, tb) - 1e-12


@S
@given(t=st.floats(-60, 60), rh=st.floats(-10, 150), v=st.floats(-5, 60), tmrt=st.floats(-80, 120))
def test_utci_is_finite_or_explicit_nan_never_inf(t, rh, v, tmrt):
    u = float(U.utci(t, rh, v, tmrt))
    assert not math.isinf(u)
    if -50 <= t <= 50 and -30 <= tmrt - t <= 70:
        assert math.isfinite(u) and -120 < u < 90       # UTCI reaches about -100 C at -50 C air with a gale


@settings(max_examples=12, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(seed=st.integers(0, 10_000), rain=st.floats(0, 5e-5), n=st.sampled_from([0.02, 0.035, 0.1]),
       slope=st.floats(0, 0.02))
def test_flood_solver_conserves_mass_and_stays_nonnegative(seed, rain, n, slope):
    rng = np.random.default_rng(seed)
    ny, nx = 14, 12
    z = (np.arange(ny)[:, None] * slope * 20.0) + rng.normal(0, 0.05, (ny, nx))
    r = flood.simulate(z, 20.0, 900.0, rain_m_s=rain, manning_n=n, open_boundary=False)
    assert np.all(np.isfinite(r.h_end)) and r.h_end.min() >= 0.0
    assert r.mass_error < 1e-6 or r.vol_in == 0


# ── insurance portfolio: determinism, finiteness, linearity in values, deductible ladder ──────────────────────────────────
import test_insurance as _TI   # noqa: E402  (reuses the hazard fixtures)


@settings(max_examples=10, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(tiv=st.floats(1e3, 1e10), lam=st.floats(0.05, 3.0), sigma=st.floats(3.0, 15.0), seed=st.integers(1, 10_000))
def test_insurance_run_is_deterministic_finite_and_linear_in_values(tiv, lam, sigma, seed):
    def go(t):
        return _TI.run([_TI.site(tiv=t, h={"lam": lam, "sigma": sigma})], n_years=1500, seed=seed)
    a, b, c = go(tiv), go(tiv), go(2 * tiv)
    la = a["deductible_ladder"]
    assert la == b["deductible_ladder"], "same seed must give identical results"
    aal = [r["aal"] for r in la]
    assert all(math.isfinite(x) and x >= 0 for x in aal)
    assert all(aal[i + 1] <= aal[i] + 1e-6 * max(1.0, aal[0]) for i in range(len(aal) - 1)), "a higher deductible cannot raise the AAL"
    assert abs(c["deductible_ladder"][0]["aal"] - 2 * aal[0]) <= 1e-6 * max(1.0, aal[0]), "losses scale with insured value"
