"""
cri/eventlearn_tc.py: learn from recorded US tropical cyclones how total loss relates to the storm's wind at the coast, and test it on storms it never saw.

Truth: NOAA NCEI "Billion-Dollar Weather and Climate Disasters" (US public domain), CPI-adjusted TOTAL economic cost per event (insured plus uninsured, direct and some
indirect; it is not an insured-loss figure). File: data/events/noaa_billion_tropical_cyclones.csv (the Tropical Cyclone rows, copied from
https://www.ncei.noaa.gov/access/billions/events-US-1980-2024.csv).
Hazard: NOAA IBTrACS best tracks already in the engine (cri/data/ibtracs_since1980.npz). Feature = peak 1-minute sustained wind (kt) over the storm's track points inside the
contiguous-US window (23-47 N, 64-100 W) and within 100 km of land. Storms that never come inside that window (Puerto Rico, Hawaii, Guam, Sandy's transition) are listed and left out.
Model: log10(cost) = a + b * wind_kt (+ c * (year - 2000)), ordinary least squares. Tested leave-one-storm-out: each storm is predicted by a model fitted on all the others.

    PYTHONPATH=src python -m cri.eventlearn_tc [OUT.json]
    from cri.eventlearn_tc import predict;  predict(wind_kt=115, year=2026)
"""
from __future__ import annotations

import csv
import json
import math
import re
import statistics as st
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
CSV = _ROOT / "data" / "events" / "noaa_billion_tropical_cyclones.csv"
NPZ = Path(__file__).resolve().parent / "data" / "ibtracs_since1980.npz"
WINDOW = dict(lat=(23.0, 47.0), lon=(-100.0, -64.0), d2l_km=100.0)


@lru_cache(maxsize=1)
def storms() -> list[dict]:
    zz = np.load(NPZ, allow_pickle=False)
    z = {k: zz[k] for k in ("storm", "name", "season", "lat", "lon", "vmax", "d2l")}                 # an npz re-reads an array on every access: load each once
    out, skipped = [], []
    for r in csv.DictReader(open(CSV)):
        m = re.match(r"(?:Hurricane|Tropical Storm|Typhoon) (\w+) \((\w+) (\d{4})\)", r["name"])
        name, year = m.group(1).upper(), int(m.group(3))
        idx = [i for i in range(len(z["name"])) if z["name"][i] == name and int(z["season"][i]) == year]
        if not idx:
            skipped.append((r["name"], "not in IBTrACS"))
            continue
        sel = z["storm"] == idx[0]
        ok = sel & (z["lon"] > WINDOW["lon"][0]) & (z["lon"] < WINDOW["lon"][1]) & (z["lat"] > WINDOW["lat"][0]) & (z["lat"] < WINDOW["lat"][1]) & (z["d2l"] <= WINDOW["d2l_km"])
        if not ok.any() or not np.isfinite(z["vmax"][ok]).any():
            skipped.append((r["name"], "no track point inside the contiguous-US window"))
            continue
        out.append(dict(name=r["name"], year=year, wind_kt=float(np.nanmax(z["vmax"][ok])), cost_musd=float(r["cpi_adjusted_cost_musd"])))
    storms.skipped = skipped                                              # type: ignore[attr-defined]
    return out


def _fit(rows, use_year):
    X = np.array([[1.0, r["wind_kt"]] + ([r["year"] - 2000.0] if use_year else []) for r in rows])
    y = np.array([math.log10(r["cost_musd"]) for r in rows])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    return beta, float(np.sqrt(resid @ resid / max(1, len(y) - X.shape[1])))


def _spearman(a, b):
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def loo(use_year=True) -> dict:
    rows = storms()
    pred, obs, base = [], [], []
    detail = []
    for i, r in enumerate(rows):
        tr = rows[:i] + rows[i + 1:]
        beta, _ = _fit(tr, use_year)
        x = np.array([1.0, r["wind_kt"]] + ([r["year"] - 2000.0] if use_year else []))
        p = float(x @ beta)
        b = st.median(math.log10(t["cost_musd"]) for t in tr)
        o = math.log10(r["cost_musd"])
        pred.append(p); obs.append(o); base.append(b)
        detail.append(dict(storm=r["name"], wind_kt=r["wind_kt"], observed_musd=round(r["cost_musd"]), predicted_musd=round(10 ** p), factor=round(10 ** (p - o), 2)))
    err = [abs(p - o) for p, o in zip(pred, obs)]
    berr = [abs(b - o) for b, o in zip(base, obs)]
    f = lambda e, k: round(100 * sum(x <= math.log10(k) for x in e) / len(e))
    return dict(n=len(rows), median_factor_error=round(10 ** st.median(err), 2), baseline_median_factor_error=round(10 ** st.median(berr), 2),
                within_factor_2_pct=f(err, 2), within_factor_3_pct=f(err, 3), baseline_within_factor_3_pct=f(berr, 3),
                rank_correlation_spearman=round(_spearman(pred, obs), 2), worst=sorted(detail, key=lambda d: -abs(math.log10(d["factor"])))[:6], detail=detail)


def predict(wind_kt: float, year: int = 2026) -> dict:
    """Expected total US economic cost (CPI-adjusted, today's dollars of the data year) for a storm with this coastal wind, with a 10-90% factor range from the fit's residual."""
    rows = storms()
    beta, s = _fit(rows, True)
    mu = float(beta @ np.array([1.0, wind_kt, year - 2000.0]))
    lo, hi = 10 ** (mu - 1.2816 * s), 10 ** (mu + 1.2816 * s)
    return dict(wind_kt=wind_kt, expected_musd=round(10 ** mu), p10_musd=round(lo), p90_musd=round(hi), n_storms=len(rows), residual_sd_log10=round(s, 2),
                caveat="total economic cost of a US landfalling storm, not an insured loss; rain-driven storms (Harvey) and surge (Sandy, Katrina) sit far from a wind-only curve")


def main():
    r = dict(window=WINDOW, n_storms=len(storms()), left_out=[dict(storm=a, reason=b) for a, b in storms.skipped],               # type: ignore[attr-defined]
             wind_and_year=loo(True), wind_only=loo(False))
    beta, s = _fit(storms(), True)
    r["fit_wind_and_year"] = dict(intercept_log10=round(float(beta[0]), 3), per_kt_log10=round(float(beta[1]), 4), per_year_log10=round(float(beta[2]), 4), residual_sd_log10=round(s, 3),
                                  multiplier_per_10kt=round(10 ** (10 * float(beta[1])), 2), multiplier_per_decade=round(10 ** (10 * float(beta[2])), 2))
    r["examples"] = {str(k): predict(k) for k in (65, 90, 115, 140)}
    for k in ("wind_and_year", "wind_only"):
        r[k] = {a: b for a, b in r[k].items() if a != "detail"} | {"detail_file": "see eventlearn_tc_detail.json"}
    if len(sys.argv) > 1:
        Path(sys.argv[1]).write_text(json.dumps(r, indent=1))
        Path(sys.argv[1]).with_name("eventlearn_tc_detail.json").write_text(json.dumps(loo(True)["detail"], indent=1))
    print(json.dumps(r, indent=1))


if __name__ == "__main__":
    main()
