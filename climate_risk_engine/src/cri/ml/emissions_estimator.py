"""
Scope 1 / 2 / 3 Emissions Estimator — observed-data first, calibrated fallback
──────────────────────────────────────────────────────────────────────────────
Order of precedence for Scope 1
  1. Caller-supplied value (scope1_known)                    → "verified"
  2. Observed EPA GHGRP emissions for the company            → "verified"
     (matched by ticker or name; 246 US companies, 2023)       coverage "complete"
                                                              for US heavy industry,
                                                              "lower_bound" otherwise
  3. Sector intensity × revenue                              → "ai_estimated"
     • 8 heavy-industry sectors: intensities fitted on observed company
       emissions (EPA GHGRP 2023 + SEC revenue, n=163), shrunk toward the
       literature prior, with a pooled revenue-size term
     • other sectors: literature prior table (not yet validated)

Measured accuracy (5-fold held-out, 163 US heavy-industry companies)
  Revenue × sector estimates are typically ~3.5× off for an individual company
  (median |log10 error| ≈ 0.55; ~30 % of companies within 2×). The spread is
  intrinsic — two utilities with the same revenue can run coal or nuclear
  fleets — so confidence intervals are set from this measured spread, not
  assumed. Accuracy comes from step 2: asset-level observed emissions.

History: versions before 2.0 trained an XGBoost model on 3,000 synthetic rows
sampled from the prior table, and described themselves as "trained on CDP
disclosures". That model could not beat the table it was sampled from and
xgboost was not installed in production, so it was removed.

Scope 2 and Scope 3 remain parametric (grid factor × Scope 1 share; sector
supply-chain multipliers) and are flagged as estimates.

Data build + evaluation: scripts/build_emissions_dataset.py,
scripts/evaluate_emissions_models.py, scripts/export_emissions_runtime.py
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Optional

_DATA = Path(__file__).resolve().parent.parent / "data"
_Z80 = 1.2816          # two-sided 80 % interval, normal in log10 space


# ── Result type ───────────────────────────────────────────────────────────────

@dataclass
class EmissionsEstimate:
    scope1_mt_co2e: float
    scope1_lo:      float
    scope1_hi:      float
    scope2_mt_co2e: float
    scope2_lo:      float
    scope2_hi:      float
    scope3_mt_co2e: float
    scope3_lo:      float
    scope3_hi:      float
    data_quality:   str       # "ai_estimated" | "hybrid" | "verified"
    imputed_fields: list[str]
    model_confidence: float   # 0–1
    model_version:    str
    scope1_source:    str = ""          # where the Scope 1 number came from
    scope1_coverage:  str = ""          # "complete" | "lower_bound" | "estimate"
    notes:            list[str] = field(default_factory=list)


# Literature PRIOR — median t CO2e per $M revenue (USD 2020), with p10/p90.
# Used directly only for sectors without observed calibration data, and as the
# shrinkage prior for sectors that have it (see emissions_calibration.json).

_SECTOR_INTENSITY: dict[str, tuple[float, float, float]] = {
    # (median, p10, p90)  — tCO2e per $M revenue
    "coal":           (2800.0, 1400.0, 6000.0),
    "oil_gas":        ( 320.0,  120.0,  850.0),
    "steel":          ( 520.0,  280.0, 1100.0),
    "cement":         ( 480.0,  260.0,  950.0),
    "chemicals":      ( 210.0,   90.0,  540.0),
    "utilities":      ( 380.0,  140.0,  900.0),
    "mining":         ( 175.0,   70.0,  420.0),
    "shipping":       ( 145.0,   65.0,  340.0),
    "aviation":       ( 210.0,  110.0,  450.0),
    "automotive":     (  62.0,   28.0,  160.0),
    "real_estate":    (  18.0,    6.0,   55.0),
    "agriculture":    ( 130.0,   55.0,  310.0),
    "food_beverage":  (  95.0,   38.0,  230.0),
    "retail":         (  14.0,    5.0,   42.0),
    "financials":     (   8.0,    2.5,   22.0),
    "technology":     (  12.0,    4.0,   35.0),
    "healthcare":     (  22.0,    8.0,   58.0),
    "media":          (   8.5,    3.0,   24.0),
    "telecom":        (  18.0,    7.0,   50.0),
    "construction":   (  55.0,   22.0,  140.0),
}

# Grid carbon intensity by jurisdiction (tCO2e per MWh) — IEA 2023 data
_GRID_INTENSITY: dict[str, float] = {
    "US": 0.386, "CN": 0.581, "IN": 0.708, "DE": 0.350, "GB": 0.191,
    "FR": 0.058, "JP": 0.451, "AU": 0.510, "BR": 0.074, "CA": 0.140,
    "ZA": 0.928, "SA": 0.680, "KR": 0.415, "MX": 0.455, "NL": 0.295,
    "PL": 0.740, "ID": 0.760, "NG": 0.430, "EG": 0.500, "TR": 0.440,
    "IT": 0.288, "ES": 0.175, "SE": 0.028, "NO": 0.018, "FI": 0.092,
}

# Scope 3 multiplier relative to Scope 1+2 (sector-specific) — GHG Protocol
_SCOPE3_MULT: dict[str, float] = {
    "coal": 11.0, "oil_gas": 8.5, "steel": 0.8, "cement": 0.7,
    "chemicals": 2.5, "utilities": 0.6, "mining": 1.2, "shipping": 0.9,
    "aviation": 1.1, "automotive": 5.5, "real_estate": 1.8,
    "agriculture": 2.2, "food_beverage": 6.0, "retail": 4.5,
    "financials": 700.0,   # financed emissions — typically huge
    "technology": 8.0, "healthcare": 3.5, "media": 3.0,
    "telecom": 2.8, "construction": 2.0,
}



def _sector_lookup(sector: str) -> tuple[float, float, float]:
    s = sector.lower().replace(" ", "_").replace("-", "_")
    if s in _SECTOR_INTENSITY:
        return _SECTOR_INTENSITY[s]
    for key in _SECTOR_INTENSITY:
        if key in s or s in key:
            return _SECTOR_INTENSITY[key]
    return (95.0, 35.0, 250.0)   # generic manufacturing


def _canon_sector(sector: str) -> str:
    s = sector.lower().replace(" ", "_").replace("-", "_")
    if s in _SECTOR_INTENSITY:
        return s
    for key in _SECTOR_INTENSITY:
        if key in s or s in key:
            return key
    return s


def _grid_intensity(jurisdiction: str) -> float:
    return _GRID_INTENSITY.get((jurisdiction or "US").upper()[:2], 0.40)


# ── Runtime data (shipped in cri/data) ────────────────────────────────────────

@lru_cache(maxsize=1)
def _calibration() -> dict:
    try:
        return json.loads((_DATA / "emissions_calibration.json").read_text())
    except Exception:           # noqa: BLE001 — fall back to prior table
        return {}


_SUFFIX = re.compile(
    r"\b(the|inc|incorporated|corp|corporation|co|company|companies|llc|l\.l\.c|lp|l\.p|ltd|limited|plc|"
    r"holdings?|group|sa|nv|ag|se|na|usa|us|trust|partners)\b")


def _norm(name: str) -> str:
    s = str(name).lower().replace("&", " and ")
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    s = _SUFFIX.sub(" ", s)
    return re.sub(r"\s+", " ", s).strip()


@lru_cache(maxsize=1)
def _observed_index() -> tuple[dict, dict, dict]:
    try:
        obs = json.loads((_DATA / "observed_emissions_us.json").read_text())
    except Exception:           # noqa: BLE001
        return {}, {}, {}
    by_ticker, by_name = {}, {}
    for r in obs.get("companies", []):
        by_ticker[r["ticker"].upper()] = r
        for n in r.get("names_norm", []):
            by_name.setdefault(n, r)
    return by_ticker, by_name, obs


def observed_scope1(company_name: Optional[str] = None, ticker: Optional[str] = None) -> Optional[dict]:
    """Observed EPA GHGRP Scope 1 record for a company, or None."""
    by_ticker, by_name, meta = _observed_index()
    rec = None
    if ticker:
        rec = by_ticker.get(ticker.upper().strip())
    if rec is None and company_name:
        rec = by_name.get(_norm(company_name))
    if rec is None:
        return None
    return {**rec, "year": meta.get("year"), "source": meta.get("source")}


def _default_sd() -> float:
    """Log10 spread for sectors without calibration: from measured held-out error."""
    acc = (_calibration().get("heldout_accuracy") or {}).get("A_parametric_table") or {}
    mae = acc.get("median_abs_log10_err", 0.55)
    return mae / 0.6745        # median |N(0, sd)| = 0.6745 sd


def _scope1_estimate(revenue_usd_m: float, sector: str, year: int) -> tuple[float, float, float, str, float]:
    """Return (point, lo80, hi80, method, sd_log10) in Mt CO2e."""
    cal = _calibration()
    sec = _canon_sector(sector)
    trend = 0.97 ** (year - (cal.get("year") or 2020))
    s = (cal.get("sectors") or {}).get(sec)
    rev = max(revenue_usd_m, 1e-6)
    if s:
        li = s["log10_intensity"] + cal.get("size_slope_log10", 0.0) * (math.log10(rev) - cal.get("log_rev_center", 3.5))
        sd = max(float(s.get("resid_sd", _default_sd())), 0.15)
        method = f"calibrated on observed emissions (EPA GHGRP {cal.get('year')}, n={s.get('n')})"
    else:
        med, _, _ = _sector_lookup(sector)
        li = math.log10(med * 0.97 ** (year - 2020)) - math.log10(trend or 1)   # table has own trend
        sd = _default_sd()
        method = "literature prior table (not validated on observed data for this sector)"
    point = 10 ** li * trend * rev / 1e6
    f = 10 ** (_Z80 * sd)
    return point, point / f, point * f, method, sd


def _table_estimate(revenue_usd_m, sector, jurisdiction, employee_count, year, scope1_known, scope2_known):
    """Pre-2.0 literature-table method, kept for benchmarking in evaluate_emissions_models.py."""
    med, p10, p90 = _sector_lookup(sector)
    t = 0.97 ** (year - 2020)
    s1 = med * t * revenue_usd_m / 1e6
    return EmissionsEstimate(s1, p10 * t * revenue_usd_m / 1e6, p90 * t * revenue_usd_m / 1e6,
                             0, 0, 0, 0, 0, 0, "ai_estimated", ["scope1"], 0.5, "table-1.x")


def _analytical_estimate(
    revenue_usd_m: float,
    sector: str,
    jurisdiction: str,
    employee_count: Optional[int],
    year: int,
    scope1_known: Optional[float],
    scope2_known: Optional[float],
    observed: Optional[dict] = None,
) -> EmissionsEstimate:
    notes: list[str] = []
    imputed: list[str] = []
    est, lo, hi, method, sd = _scope1_estimate(revenue_usd_m, sector, year)

    if scope1_known is not None:
        s1, s1_lo, s1_hi = scope1_known, scope1_known * 0.9, scope1_known * 1.1
        src, cov = "caller-supplied", "complete"
    elif observed is not None:
        obs = observed["scope1_mt"]
        if observed["coverage"] == "complete":
            s1, s1_lo, s1_hi = obs, obs * 0.9, obs * 1.1
            cov = "complete"
        else:
            # Observed = large US facilities only → a floor for the whole company.
            # Point = the observed floor (transparent); the upper range stays open.
            # (max(observed, sector estimate) was tried and overshot — e.g. GM 4.8 Mt
            # vs ~1.5–2 Mt disclosed — because uncalibrated sector priors run high.)
            s1 = obs
            s1_lo, s1_hi = obs, obs * 3.0
            cov = "lower_bound"
            notes.append("Scope 1 shown is EPA-reported emissions from this company's large US "
                         "facilities only; total company Scope 1 is higher.")
        src = f"EPA GHGRP {observed.get('year')} ({observed['n_facilities']} US facilities, {observed['name']})"
    else:
        s1, s1_lo, s1_hi = est, lo, hi
        src, cov = method, "estimate"
        imputed.append("scope1")

    grid = _grid_intensity(jurisdiction)
    if scope2_known is not None:
        s2, s2_lo, s2_hi = scope2_known, scope2_known * 0.85, scope2_known * 1.2
    else:
        s2 = s1 * 0.18 * (grid / 0.40)
        s2_lo, s2_hi = s2 / 3, s2 * 3
        imputed.append("scope2")

    s3_mult = _SCOPE3_MULT.get(_canon_sector(sector), 2.0)
    s3 = (s1 + s2) * s3_mult
    imputed.append("scope3")

    n_est = len(imputed)
    quality = "verified" if n_est == 1 and "scope1" not in imputed else ("hybrid" if n_est < 3 else "ai_estimated")
    if "scope1" not in imputed and cov == "complete":
        conf = 0.9
    else:
        conf = round(max(0.15, min(0.8, 1.0 - sd)), 2)   # sd≈0.6 log10 → 0.4

    return EmissionsEstimate(
        scope1_mt_co2e=round(s1, 4), scope1_lo=round(s1_lo, 4), scope1_hi=round(s1_hi, 4),
        scope2_mt_co2e=round(s2, 4), scope2_lo=round(s2_lo, 4), scope2_hi=round(s2_hi, 4),
        scope3_mt_co2e=round(s3, 4), scope3_lo=round(s3 * 0.5, 4), scope3_hi=round(s3 * 2.0, 4),
        data_quality=quality, imputed_fields=imputed, model_confidence=conf,
        model_version=EmissionsEstimator.MODEL_VERSION,
        scope1_source=src, scope1_coverage=cov, notes=notes,
    )


class EmissionsEstimator:
    """
    Usage
    ─────
    EmissionsEstimator().estimate(revenue_usd_m=12_500, sector="utilities",
                                  jurisdiction="US", company_name="Duke Energy")
    """

    MODEL_VERSION = "2.0.0-ghgrp-calibrated"

    def __init__(self, seed: int = 42):   # seed kept for API compatibility
        self._seed = seed

    def estimate(
        self,
        revenue_usd_m: float,
        sector: str,
        jurisdiction: str = "US",
        year: int = 2025,
        scope1_known: Optional[float] = None,
        scope2_known: Optional[float] = None,
        employee_count: Optional[int] = None,
        company_name: Optional[str] = None,
        ticker: Optional[str] = None,
    ) -> EmissionsEstimate:
        observed = None
        if scope1_known is None and (company_name or ticker):
            observed = observed_scope1(company_name, ticker)
        return _analytical_estimate(revenue_usd_m, sector, jurisdiction, employee_count,
                                    year, scope1_known, scope2_known, observed)
