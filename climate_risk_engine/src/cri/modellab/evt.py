"""
Extreme-value hazard model with a hindcast test — the engine's generic way to build AND validate any hazard defined by "the worst value in a year"
(extreme heat, rainfall, wind, fire weather, drought intensity, cold).

  1. Fit a Gumbel (EV1) and a GEV by maximum likelihood; keep the GEV only if a likelihood-ratio test says the extra shape parameter earns its place.
  2. Hindcast: fit on the first two-thirds of the record; predict the last third. Two tests that cannot be argued with:
       - probability-integral transform: F_train(x_test) must look uniform (Kolmogorov–Smirnov),
       - exceedance counts: the train T-year level should be exceeded n_test/T times (binomial test) for T = 2, 5, 10.
  3. Trend: does the location parameter drift with time (LRT)? If so, report both the stationary and the trend-adjusted level, the latter flagged as extrapolation.
  4. Return levels (10, 50, 100 y) with bootstrap confidence intervals.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

EULER = 0.5772156649015329


def _stats():
    from scipy import stats
    return stats


def gumbel_fit(x) -> tuple:
    """MLE Gumbel (loc, scale)."""
    st = _stats()
    loc, scale = st.gumbel_r.fit(np.asarray(x, float))
    return float(loc), float(scale)


def gev_fit(x) -> tuple:
    st = _stats()
    c, loc, scale = st.genextreme.fit(np.asarray(x, float))
    return float(c), float(loc), float(scale)


def level(model: tuple, T: float) -> float:
    st = _stats()
    p = 1 - 1.0 / T
    if len(model) == 2:
        return float(st.gumbel_r.ppf(p, *model))
    return float(st.genextreme.ppf(p, *model))


def cdf(model: tuple, x):
    st = _stats()
    return st.gumbel_r.cdf(x, *model) if len(model) == 2 else st.genextreme.cdf(x, *model)


def loglik(model: tuple, x) -> float:
    st = _stats()
    return float(np.sum(st.gumbel_r.logpdf(x, *model) if len(model) == 2 else st.genextreme.logpdf(x, *model)))


def best_model(x) -> dict:
    """Gumbel unless the GEV is significantly better (LRT, 1 df, p < 0.05)."""
    st = _stats()
    x = np.asarray(x, float)
    g = gumbel_fit(x)
    out = {"kind": "Gumbel", "params": g, "ll": loglik(g, x), "lrt_p": None}
    try:
        v = gev_fit(x)
        lr = 2 * (loglik(v, x) - out["ll"])
        p = float(st.chi2.sf(max(lr, 0), 1))
        out["lrt_p"] = p
        if p < 0.05 and abs(v[0]) < 0.5:                       # reject wild shapes from short records
            out.update(kind="GEV", params=v, ll=loglik(v, x))
    except Exception:                                          # noqa: BLE001
        pass
    return out


def bootstrap_levels(x, Ts=(10, 50, 100), n: int = 200, seed: int = 7) -> dict:
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    res = {T: [] for T in Ts}
    for _ in range(n):
        s = rng.choice(x, size=len(x), replace=True)
        try:
            m = gumbel_fit(s)
        except Exception:                                      # noqa: BLE001
            continue
        for T in Ts:
            res[T].append(level(m, T))
    return {T: (float(np.percentile(v, 5)), float(np.percentile(v, 95))) for T, v in res.items() if v}


def hindcast(x, frac: float = 0.67) -> dict:
    st = _stats()
    x = np.asarray(x, float)
    n = len(x)
    cut = max(8, int(round(n * frac)))
    tr, te = x[:cut], x[cut:]
    if len(te) < 5:
        return {"ok": False, "reason": f"only {len(te)} years to test on"}
    bm = best_model(tr)
    m = bm["params"]
    pit = np.asarray(cdf(m, te), float)
    ks = st.kstest(pit, "uniform")
    rows = []
    for T in (2, 5, 10):
        lv = level(m, T)
        k = int(np.sum(te > lv))
        exp = len(te) / T
        p = float(st.binomtest(k, len(te), 1.0 / T).pvalue)
        rows.append({"T": T, "level": lv, "observed": k, "expected": exp, "p": p})
    mean_pit = float(np.mean(pit))
    ok = ks.pvalue > 0.05 and all(r["p"] > 0.05 / 3 for r in rows)          # Bonferroni across the three exceedance tests
    return {"ok": True, "passed": bool(ok), "n_train": int(cut), "n_test": int(len(te)), "model": bm["kind"], "params": [float(v) for v in m], "ks_p": float(ks.pvalue),
            "mean_pit": mean_pit, "exceedances": rows, "pit": pit.tolist(), "test_mean": float(te.mean()), "train_mean": float(tr.mean()),
            "bias_note": ("The recent years are systematically higher than the early record predicted: the hazard is increasing." if mean_pit > 0.65 and ks.pvalue < 0.1 else
                          "The recent years are systematically lower than the early record predicted." if mean_pit < 0.35 and ks.pvalue < 0.1 else "")}


def trend_test(years, x) -> dict:
    """Stationary Gumbel vs Gumbel with location linear in time (LRT, 1 df)."""
    st = _stats()
    from scipy.optimize import minimize
    y = np.asarray(years, float); y = y - y.mean()
    x = np.asarray(x, float)
    l0, s0 = gumbel_fit(x)
    ll0 = float(np.sum(st.gumbel_r.logpdf(x, l0, s0)))

    def nll(p):
        loc, slope, ls = p
        return -float(np.sum(st.gumbel_r.logpdf(x, loc + slope * y, math.exp(ls))))
    r = minimize(nll, [l0, 0.0, math.log(s0)], method="Nelder-Mead", options={"maxiter": 4000, "xatol": 1e-6, "fatol": 1e-8})
    ll1 = -float(r.fun)
    lr = 2 * (ll1 - ll0)
    p = float(st.chi2.sf(max(lr, 0), 1))
    loc, slope, ls = r.x
    return {"slope_per_decade": float(slope * 10), "p": p, "significant": bool(p < 0.05), "loc0": float(loc), "scale": float(math.exp(ls)), "ymean": float(np.mean(years)), "stationary": (l0, s0)}


def trend_level(tr: dict, year: float, T: float) -> float:
    st = _stats()
    loc = tr["loc0"] + tr["slope_per_decade"] / 10.0 * (year - tr["ymean"])
    return float(st.gumbel_r.ppf(1 - 1.0 / T, loc, tr["scale"]))


# ── verification ─────────────────────────────────────────────────────────────

def bench_recovers_known_parameters() -> dict:
    rng = np.random.default_rng(1)
    x = rng.gumbel(40.0, 6.0, 4000)
    loc, sc = gumbel_fit(x)
    err = max(abs(loc - 40) / 40, abs(sc - 6) / 6)
    return {"name": "Fit recovers a known Gumbel (4000 synthetic years)", "metric": "worst relative parameter error", "value": err, "limit": 0.03, "ok": err < 0.03, "note": "A fitter that cannot recover the truth from plentiful data cannot be trusted on 30 years."}


def bench_return_level_formula() -> dict:
    loc, sc, T = 30.0, 5.0, 100.0
    exact = loc - sc * math.log(-math.log(1 - 1 / T))
    got = level((loc, sc), T)
    rel = abs(got - exact) / exact
    return {"name": "Return level equals μ − β ln(−ln(1 − 1/T))", "metric": "relative error", "value": rel, "limit": 1e-9, "ok": rel < 1e-9, "note": "The closed form of the Gumbel quantile."}


def bench_hindcast_is_calibrated() -> dict:
    """On stationary synthetic records the hindcast must NOT cry wolf: its uniformity test should pass about 95 % of the time."""
    ok = 0
    n = 80
    for s in range(n):
        x = np.random.default_rng(100 + s).gumbel(25.0, 4.0, 45)
        h = hindcast(x)
        ok += int(h["passed"])
    rate = ok / n
    return {"name": "Hindcast test is calibrated (stationary synthetic records)", "metric": "fraction of good models that pass the full hindcast (expect ≈ 0.9)", "value": rate, "limit": 0.8, "ok": rate >= 0.8, "note": "A validation test that fails good models, or passes bad ones, is worthless."}


def bench_hindcast_detects_a_trend() -> dict:
    """…and it must catch a real trend: records with a strong upward drift should be flagged."""
    caught = 0
    n = 30
    for s in range(n):
        rng = np.random.default_rng(300 + s)
        x = rng.gumbel(25.0, 3.0, 45) + np.linspace(0, 12, 45)
        h = hindcast(x)
        caught += int(not h["passed"])
    rate = caught / n
    return {"name": "Hindcast test detects a strong trend", "metric": "fraction of trending records flagged", "value": rate, "limit": 0.6, "ok": rate >= 0.6, "note": "Power: a drift of 12 units over 45 years must not slip through."}


def run_benchmarks() -> list:
    return [bench_return_level_formula(), bench_recovers_known_parameters(), bench_hindcast_is_calibrated(), bench_hindcast_detects_a_trend()]
