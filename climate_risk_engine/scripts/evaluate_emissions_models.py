"""
Held-out evaluation of Scope 1 estimators against REAL reported emissions.

Compares, on companies the candidate model never saw (grouped K-fold):
  A. parametric   — current IPCC/literature sector-intensity table × revenue
  B. (removed) XGBoost trained on synthetic rows from table A — identical to A
  C. calibrated   — sector log-intensity fitted on observed data, shrunk toward
                    the table prior when a sector has few companies
  D. calibrated+size — C plus a pooled log-revenue elasticity

Metric: error in log10 space (a model that is 2× off scores 0.30 either way).
Reported: median |log10 error|, share of companies within 2× / within 3×, bias.

Scope: "trusted" heavy-industry sectors where EPA GHGRP (US facilities
> 25 kt) captures most of a US-domestic company's Scope 1. Foreign filers
are excluded. Other sectors are lower bounds and are not used.

Usage: python scripts/evaluate_emissions_models.py [--csv data/training/emissions_ghgrp_sec_2023.csv]
Writes data/training/emissions_eval_<year>.json and prints a table.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cri.ml import emissions_estimator as ee  # noqa: E402

TRUSTED = ["utilities", "oil_gas", "chemicals", "cement", "steel", "mining", "pulp_paper", "coal"]
PRIOR_STRENGTH = 5.0      # pseudo-companies of weight given to the table prior
YEAR = 2023


def load(csv: Path) -> pd.DataFrame:
    d = pd.read_csv(csv)
    d = d[(~d.foreign_filer) & d.sector.isin(TRUSTED) & (d.revenue_usd_m > 50) & (d.scope1_t > 0)].copy()
    d["log_int"] = np.log10(d.scope1_t / d.revenue_usd_m)          # t per $M
    d["log_rev"] = np.log10(d.revenue_usd_m)
    d["obs_mt"] = d.scope1_t / 1e6
    return d.reset_index(drop=True)


def prior_log_int(sector: str) -> float:
    med, _, _ = ee._sector_lookup(sector)
    return math.log10(med * 0.97 ** (YEAR - 2020))     # same trend factor the table applies


# ── candidate models ─────────────────────────────────────────────────────────

def pred_parametric(test: pd.DataFrame, train: pd.DataFrame) -> np.ndarray:
    return np.array([ee._table_estimate(r.revenue_usd_m, r.sector, "US", None, YEAR, None, None).scope1_mt_co2e
                     for r in test.itertuples()])


def fit_calibrated(train: pd.DataFrame, size_term: bool) -> dict:
    """Empirical-Bayes sector means of log intensity (+ optional pooled size slope)."""
    slope = 0.0
    if size_term and len(train) > 20:
        # within-sector slope of log_int on log_rev (demeaned by sector)
        g = train.groupby("sector")
        x = train.log_rev - g.log_rev.transform("mean")
        y = train.log_int - g.log_int.transform("mean")
        slope = float((x * y).sum() / max((x * x).sum(), 1e-9))
    resid = train.log_int - slope * (train.log_rev - train.log_rev.mean())
    sectors = {}
    for s in TRUSTED:
        r = resid[train.sector == s]
        n = len(r)
        prior = prior_log_int(s)
        mean = (r.sum() + PRIOR_STRENGTH * prior) / (n + PRIOR_STRENGTH) if n else prior
        sd = float(r.std(ddof=1)) if n >= 3 else 0.45
        sectors[s] = {"log10_intensity": float(mean), "n": int(n), "resid_sd": sd}
    return {"slope": slope, "log_rev_center": float(train.log_rev.mean()), "sectors": sectors}


def predict_calibrated(m: dict, rev_m: float, sector: str) -> float:
    s = m["sectors"].get(sector)
    li = s["log10_intensity"] if s else prior_log_int(sector)
    li += m["slope"] * (math.log10(rev_m) - m["log_rev_center"])
    return 10 ** li * rev_m / 1e6


def pred_cal(size: bool):
    def f(test, train):
        m = fit_calibrated(train, size)
        return np.array([predict_calibrated(m, r.revenue_usd_m, r.sector) for r in test.itertuples()])
    return f


# ── evaluation ───────────────────────────────────────────────────────────────

def kfold_eval(d: pd.DataFrame, fn, k: int = 5, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    folds = rng.permutation(len(d)) % k
    pred = np.zeros(len(d))
    for f in range(k):
        te, tr = d[folds == f], d[folds != f]
        pred[folds == f] = fn(te, tr)
    return pred


def score(obs: np.ndarray, pred: np.ndarray) -> dict:
    e = np.log10(np.maximum(pred, 1e-9)) - np.log10(obs)
    return {"median_abs_log10_err": round(float(np.median(np.abs(e))), 3),
            "within_2x": round(float(np.mean(np.abs(e) <= math.log10(2))), 3),
            "within_3x": round(float(np.mean(np.abs(e) <= math.log10(3))), 3),
            "median_bias_x": round(float(10 ** np.median(e)), 2),
            "n": int(len(obs))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(ROOT / "data/training/emissions_ghgrp_sec_2023.csv"))
    a = ap.parse_args()
    d = load(Path(a.csv))
    # B (XGBoost on synthetic rows) was removed in estimator 2.0: xgboost was not
    # installed in production, so it always fell back to A and scored identically.
    models = {"A_parametric_table": pred_parametric,
              "C_calibrated": pred_cal(False), "D_calibrated_plus_size": pred_cal(True)}
    res, preds = {}, {}
    for name, fn in models.items():
        preds[name] = kfold_eval(d, fn)
        res[name] = score(d.obs_mt.values, preds[name])
    by_sector = {}
    for s in TRUSTED:
        m = d.sector == s
        if m.sum() >= 3:
            by_sector[s] = {n: score(d.obs_mt.values[m], p[m]) for n, p in preds.items()}
    final = fit_calibrated(d, size_term=res["D_calibrated_plus_size"]["median_abs_log10_err"]
                           < res["C_calibrated"]["median_abs_log10_err"])
    out = {"dataset": Path(a.csv).name, "year": YEAR, "trusted_sectors": TRUSTED,
           "n_companies": len(d), "overall": res, "by_sector": by_sector, "final_calibration": final}
    Path(ROOT / f"data/training/emissions_eval_{YEAR}.json").write_text(json.dumps(out, indent=1))

    print(f"Held-out (5-fold) Scope 1 accuracy on {len(d)} US companies, trusted sectors")
    print(f"{'model':26s} {'med |log10 err|':>15s} {'within 2x':>10s} {'within 3x':>10s} {'bias (x)':>9s}")
    for n, r in res.items():
        print(f"{n:26s} {r['median_abs_log10_err']:15.3f} {r['within_2x']:10.0%} {r['within_3x']:10.0%} {r['median_bias_x']:9.2f}")
    print("\nper sector (median |log10 err|): ", )
    for s, r in by_sector.items():
        print(f"  {s:11s} n={r['A_parametric_table']['n']:3d}  " +
              "  ".join(f"{k[:1]}={v['median_abs_log10_err']:.2f}" for k, v in r.items()))
    print("\nfinal calibrated intensities (t CO2e / $M revenue):")
    for s, v in final["sectors"].items():
        print(f"  {s:11s} {10 ** v['log10_intensity']:8.0f}  (table {10 ** prior_log_int(s):6.0f}, n={v['n']})")
    print(f"  size slope: {final['slope']:+.3f}")


if __name__ == "__main__":
    main()
