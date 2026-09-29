"""
Export the runtime files the EmissionsEstimator loads (shipped inside the package):

  src/cri/data/emissions_calibration.json   sector intensities fitted on observed
                                            data + measured residual spread +
                                            held-out accuracy (from evaluate_emissions_models.py)
  src/cri/data/observed_emissions_us.json   company-level observed Scope 1 (EPA GHGRP)
                                            keyed by ticker / CIK / normalised names

Run after build_emissions_dataset.py and evaluate_emissions_models.py.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
YEAR = 2023
TRUSTED = ["utilities", "oil_gas", "chemicals", "cement", "steel", "mining", "pulp_paper", "coal"]

_SUFFIX = re.compile(
    r"\b(the|inc|incorporated|corp|corporation|co|company|companies|llc|l\.l\.c|lp|l\.p|ltd|limited|plc|"
    r"holdings?|group|sa|nv|ag|se|na|usa|us|trust|partners)\b")


def norm(name: str) -> str:
    s = str(name).lower().replace("&", " and ")
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    s = _SUFFIX.sub(" ", s)
    return re.sub(r"\s+", " ", s).strip()


def main():
    ev = json.loads((ROOT / f"data/training/emissions_eval_{YEAR}.json").read_text())
    cal = {
        "version": f"ghgrp-sec-{YEAR}",
        "year": YEAR,
        "unit": "t CO2e Scope 1 per $M revenue",
        "trusted_sectors": TRUSTED,
        "size_slope_log10": ev["final_calibration"]["slope"],
        "log_rev_center": ev["final_calibration"]["log_rev_center"],
        "sectors": ev["final_calibration"]["sectors"],
        "heldout_accuracy": ev["overall"],
        "n_companies": ev["n_companies"],
        "source": "EPA GHGRP 2023 direct emitters (parent-company attributed) + SEC EDGAR revenue",
    }
    out_dir = ROOT / "src/cri/data"
    (out_dir / "emissions_calibration.json").write_text(json.dumps(cal, indent=1))

    d = pd.read_csv(ROOT / f"data/training/emissions_ghgrp_sec_{YEAR}.csv")
    recs = []
    for r in d.itertuples():
        names = {norm(r.sec_title)} | {norm(n) for n in str(r.ghgrp_names).split(" | ") if n}
        complete = (r.sector in TRUSTED) and not bool(r.foreign_filer)
        recs.append({
            "cik": int(r.cik), "ticker": str(r.ticker).upper(), "name": r.sec_title,
            "names_norm": sorted(n for n in names if n),
            "sector": r.sector, "revenue_usd_m": float(r.revenue_usd_m),
            "scope1_mt": round(float(r.scope1_t) / 1e6, 4),
            "n_facilities": int(r.n_facilities),
            # complete: heavy-industry, US-domestic → GHGRP ≈ full Scope 1
            # lower_bound: other sectors / foreign HQ → only large US facilities counted
            "coverage": "complete" if complete else "lower_bound",
        })
    obs = {"year": YEAR, "source": "US EPA GHGRP (facility-level, parent-attributed by ownership share)",
           "scope": "Scope 1 direct emissions from US facilities > 25 kt CO2e/yr; excludes supplier & biogenic",
           "companies": recs}
    (out_dir / "observed_emissions_us.json").write_text(json.dumps(obs))
    print(f"calibration: {len(cal['sectors'])} sectors | observed: {len(recs)} companies "
          f"({sum(r['coverage'] == 'complete' for r in recs)} complete)")


if __name__ == "__main__":
    main()
