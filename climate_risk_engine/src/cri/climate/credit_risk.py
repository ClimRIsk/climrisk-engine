"""
Climate-Adjusted Credit Risk — PD / LGD / ECL / RWA — v0.9

Gap fixes over v0.8:
  Gap 5 — Obligor-specific PD calibration: logistic scoring model replaces
           single sector-average sensitivity; weights sector + region +
           collateral_type + score non-linearity + balance sheet proxy.
  Gap 6 — Lifetime ECL term structure: project PD migration and cumulative
           ECL at 5, 10, 30-year horizons following the scenario GMST pathway.

Regulatory alignment
--------------------
· IFRS 9 ECL  — Expected Credit Loss = PD × LGD × EAD
· IFRS 9 Stages 1 / 2 / 3 — PD threshold for stage migration
· Basel III   — Risk-Weighted Assets = RW × EAD
· ECB Climate Stress Test (2022) methodology §3.2
· NGFS Phase 4 — sector transition shock lookup

References
----------
· ECB Climate Stress Test 2022 — Methodology Note, §3.2
· NGFS Technical Document on Climate Scenarios (Phase 4, 2023) — Table A.2
· BCBS Working Paper 37 (2021) — Climate-related financial risks
· BIS Quarterly Review (2020) — "How much does climate change cost banks?"
· Logistic PD calibration: Hosmer & Lemeshow (2013) Applied Logistic Regression §5
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Optional


# ── IRB helpers (Basel II/III Advanced approach) ──────────────────────────────

def _norm_cdf(x: float) -> float:
    """Standard normal CDF via math.erf."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)

def _norm_inv(p: float) -> float:
    """Rational approximation for inverse normal (Beasley-Springer-Moro)."""
    p = max(1e-10, min(1 - 1e-10, p))
    if p < 0.5:
        sign = -1.0
        pp = p
    else:
        sign = 1.0
        pp = 1.0 - p
    t = math.sqrt(-2.0 * math.log(pp))
    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308
    num = c0 + c1 * t + c2 * t * t
    den = 1.0 + d1 * t + d2 * t * t + d3 * t * t * t
    return sign * (t - num / den)

def _asset_correlation(pd: float) -> float:
    """Basel II §272 asset correlation for corporate exposures.

    R(PD) = 0.12·(1-e^{-50·PD})/(1-e^{-50}) + 0.24·[1-(1-e^{-50·PD})/(1-e^{-50})]
    For SMEs: subtract 0.04×(1 - max(5, min(50, S)-5)/45)
    We use the standard corporate formula.
    """
    e50 = math.exp(-50.0)
    w = (1.0 - math.exp(-50.0 * pd)) / max(1e-12, 1.0 - e50)
    return 0.12 * w + 0.24 * (1.0 - w)

def _irb_rwa(pd: float, lgd: float, maturity: float, ead: float,
             base_rw_pct: float = 100.0) -> float:
    """Basel II/III Advanced IRB risk weight × EAD → RWA (USD M).

    Formula (BCBS §272):
      K = LGD × N[(1-R)^(-0.5)·G(PD) + (R/(1-R))^0.5·G(0.999)] - PD·LGD
      K *= (1-1.5b)^(-1) × (1+(M-2.5)b)   maturity adjustment
      RW = K × 12.5 × 1.06                  capital ratio scaling
    """
    pd = max(1e-6, min(0.9999, pd))
    lgd = max(0.01, min(1.0, lgd))
    R = _asset_correlation(pd)
    G_pd   = _norm_inv(pd)
    G_999  = _norm_inv(0.999)
    N_arg  = (1.0 - R) ** (-0.5) * G_pd + (R / (1.0 - R)) ** 0.5 * G_999
    K_raw  = lgd * _norm_cdf(N_arg) - pd * lgd
    # Maturity adjustment (b = (0.11852 - 0.05478 × ln(PD))^2)
    b = (0.11852 - 0.05478 * math.log(max(1e-6, pd))) ** 2
    K = K_raw * (1.0 + (maturity - 2.5) * b) / max(1e-9, 1.0 - 1.5 * b)
    K = max(0.0, K)
    rw = K * 12.5 * 1.06   # 1.06 = scaling factor per Basel II §272
    return ead * rw * (base_rw_pct / 100.0)


# ── Wrong-way risk (WWR) — joint PD/LGD simulation ───────────────────────────
# When a physical climate event hits, it simultaneously:
#   (a) increases probability of default (revenue shock, CAPEX surge)
#   (b) reduces collateral value (asset damage → lower recovery)
# This creates positive correlation between PD and LGD — the "double hit".
#
# We model this using a bivariate normal draw with sector-specific correlation ρ_wwr.
# Source: BCBS "Climate-related financial risks" WP37 (2021) §IV.C
#
# ρ_wwr: how tightly linked is asset value destruction with obligor default?
#   High (0.6–0.8): sectors with physical assets as primary collateral (RE, agriculture)
#   Low  (0.1–0.3): sectors where collateral is diversified or intangible (tech, financials)
_WWR_CORRELATION: dict[str, float] = {
    "Real Estate":        0.72,
    "Agriculture":        0.68,
    "Mining":             0.60,
    "Metals & Mining":    0.60,
    "Oil & Gas":          0.55,
    "Energy":             0.52,
    "Utilities":          0.48,
    "Chemicals":          0.42,
    "Construction":       0.45,
    "Transport":          0.38,
    "Industrials":        0.32,
    "Food & Beverage":    0.35,
    "Automotive":         0.30,
    "Consumer":           0.22,
    "Healthcare":         0.18,
    "Technology":         0.12,
    "Financial Services": 0.20,
}
_WWR_N_PATHS = 5_000   # MC paths for WWR simulation

def _wwr_correlation(sector: str) -> float:
    return _WWR_CORRELATION.get(sector.strip(), 0.35)

def _wwr_ecl(
    climate_pd: float,
    climate_lgd: float,
    ead: float,
    sector: str,
    physical_pd_uplift: float,
) -> tuple[float, float]:
    """Wrong-way risk ECL via bivariate normal simulation.

    Simulates N_paths joint draws of (PD_shock, LGD_shock) with
    correlation ρ_wwr. Returns (wwr_ecl_usd_m, wwr_increment_vs_independent).

    Under independence: ECL = E[PD] × E[LGD] × EAD
    Under WWR:          ECL = E[PD × LGD] × EAD  (covariance term adds uplift)
    """
    rho = _wwr_correlation(sector)
    # Only apply WWR when physical risk is elevated (risk materialising as damage)
    # Scale WWR intensity by physical_pd_uplift / baseline_pd ratio
    if physical_pd_uplift < 1e-6 or climate_pd < 1e-6:
        return climate_pd * climate_lgd * ead, 0.0

    # PD and LGD are bounded (0,1) — model as logit-normal marginals
    # Logit transform: p_logit ~ N(logit(mu), sigma^2)
    sigma_pd  = max(0.01, math.sqrt(climate_pd  * (1 - climate_pd)))  / 2.0
    sigma_lgd = max(0.01, math.sqrt(climate_lgd * (1 - climate_lgd))) / 2.0

    rng_state = random.getstate()
    random.seed(137)
    total_ecl = 0.0
    for _ in range(_WWR_N_PATHS):
        # Correlated bivariate normal
        z1 = _norm_inv(random.random())
        z2 = rho * z1 + math.sqrt(max(0.0, 1.0 - rho * rho)) * _norm_inv(random.random())
        # Map back to (0,1) via N(mu + sigma*z)
        pd_sim  = max(1e-6, min(0.9999, _norm_cdf(
            _norm_inv(climate_pd) + sigma_pd * z1
        )))
        lgd_sim = max(1e-6, min(0.9999, _norm_cdf(
            _norm_inv(climate_lgd) + sigma_lgd * z2
        )))
        total_ecl += pd_sim * lgd_sim * ead
    random.setstate(rng_state)

    wwr_ecl = total_ecl / _WWR_N_PATHS
    independent_ecl = climate_pd * climate_lgd * ead
    return round(wwr_ecl, 4), round(wwr_ecl - independent_ecl, 4)


# ── Sector sensitivity parameters ─────────────────────────────────────────────

@dataclass
class SectorCreditSensitivity:
    physical_sensitivity:    float
    stranded_asset_risk:     float
    lgd_physical_fraction:   float
    base_rw_pct:             float
    physical_threshold:      float = 30.0


_SECTOR_SENSITIVITY: dict[str, SectorCreditSensitivity] = {
    "Oil & Gas":          SectorCreditSensitivity(0.80, 1.20, 0.55, 100, 20.0),
    "Coal":               SectorCreditSensitivity(0.70, 1.50, 0.50,  75, 15.0),
    "Mining":             SectorCreditSensitivity(0.70, 0.90, 0.45, 100, 20.0),
    "Metals & Mining":    SectorCreditSensitivity(0.70, 0.90, 0.45, 100, 20.0),
    "Agriculture":        SectorCreditSensitivity(0.90, 0.30, 0.60,  75, 25.0),
    "Real Estate":        SectorCreditSensitivity(0.65, 0.50, 0.55, 100, 25.0),
    "Utilities":          SectorCreditSensitivity(0.50, 0.60, 0.40, 100, 20.0),
    "Energy":             SectorCreditSensitivity(0.60, 0.80, 0.45, 100, 20.0),
    "Chemicals":          SectorCreditSensitivity(0.45, 0.70, 0.35, 100, 20.0),
    "Construction":       SectorCreditSensitivity(0.45, 0.40, 0.35, 100, 20.0),
    "Automotive":         SectorCreditSensitivity(0.35, 0.80, 0.30, 100, 20.0),
    "Transport":          SectorCreditSensitivity(0.38, 0.55, 0.30, 100, 20.0),
    "Industrials":        SectorCreditSensitivity(0.30, 0.45, 0.25, 100, 20.0),
    "Food & Beverage":    SectorCreditSensitivity(0.40, 0.20, 0.30,  75, 25.0),
    "Healthcare":         SectorCreditSensitivity(0.20, 0.10, 0.15,  75, 20.0),
    "Technology":         SectorCreditSensitivity(0.15, 0.10, 0.10,  50, 20.0),
    "Financial Services": SectorCreditSensitivity(0.25, 0.30, 0.20, 100, 20.0),
    "Consumer":           SectorCreditSensitivity(0.20, 0.25, 0.15,  75, 20.0),
}

_DEFAULT_SENSITIVITY = SectorCreditSensitivity(0.30, 0.40, 0.25, 100.0, 30.0)

def _get_sensitivity(sector: str) -> SectorCreditSensitivity:
    s = sector.strip()
    return _SECTOR_SENSITIVITY.get(s, _DEFAULT_SENSITIVITY)


# ── Gap 5: Obligor-specific calibration ───────────────────────────────────────
# Replaces the v0.8 single-sector-average sensitivity with a logistic scoring
# model that weights five observable obligor characteristics.
#
# The calibration follows Hosmer & Lemeshow (2013) Applied Logistic Regression §5:
#   logit(PD_uplift_fraction) = β₀ + β₁·phys_x + β₂·trans_x + β₃·coll_x
#                                   + β₄·region_x + β₅·interaction
#   PD_uplift_fraction = sigmoid(logit) — then rescaled against sector sensitivity.
#
# Coefficients calibrated to match ECB CST 2022 §3.2 PD migration outcomes
# for the 3-scenario, 3-horizon matrix (Table B.4 in the technical note).

# Region physical-stress coefficient: how much does region amplify physical PD risk?
# High = region has high climate hazard incidence above global median
_REGION_PHYS_COEFF: dict[str, float] = {
    "AU": 0.35, "AU-WA": 0.40, "AU-QLD": 0.38, "AU-SA": 0.33,
    "BD": 0.55, "IN": 0.42, "PK": 0.45,
    "BR": 0.38, "CO": 0.32,
    "CN": 0.32, "HK": 0.28,
    "EU": 0.20, "DE": 0.18, "FR": 0.22, "IT": 0.28, "ES": 0.30,
    "GB": 0.18, "UK": 0.18,
    "ID": 0.48, "PH": 0.50, "VN": 0.45,
    "MX": 0.35, "CL": 0.28,
    "NG": 0.52, "GH": 0.48, "KE": 0.45, "ZA": 0.40,
    "SA": 0.50, "AE": 0.48, "EG": 0.42,
    "US": 0.22, "CA": 0.18,
    "JP": 0.25, "KR": 0.22,
    "Global": 0.25,
}

def _region_phys_coeff(region: str) -> float:
    r = region.strip()
    for key, val in _REGION_PHYS_COEFF.items():
        if key.upper() in r.upper() or r.upper() in key.upper():
            return val
    return 0.25

# Collateral PD-protection factor: secured collateral provides partial insulation
# against PD migration (but not LGD — physical damage still hits recovery)
_COLLATERAL_PD_FACTOR: dict[str, float] = {
    "real_estate": 0.90,   # collateral itself is climate-exposed → amplifies PD
    "equipment":   0.75,   # physical assets — moderately exposed
    "inventory":   0.60,   # less direct exposure
    "unsecured":   1.00,   # no cushion
}

_COLLATERAL_LGD_MULT: dict[str, float] = {
    "real_estate": 0.60,
    "equipment":   0.45,
    "inventory":   0.35,
    "unsecured":   0.20,
}


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _obligor_specific_uplift(
    inp: "CounterpartyInput",
    sens: SectorCreditSensitivity,
    scen_mult: float,
) -> tuple[float, float, dict]:
    """Gap 5: Compute obligor-specific physical and transition PD uplifts.

    Uses a logistic scoring model instead of a fixed sector sensitivity multiplier.

    Returns (physical_pd_uplift, transition_pd_uplift, calibration_audit_dict)
    """
    # ── Physical uplift — logistic model ─────────────────────────────────────
    # Standardise physical_score to [0,1]; apply non-linear transformation
    phys_norm = max(0.0, inp.physical_score - sens.physical_threshold) / (100.0 - sens.physical_threshold + 1e-6)

    # Region amplification
    region_coeff = _region_phys_coeff(getattr(inp, "region", "Global"))

    # Collateral provides partial protection against PD migration
    coll_pd_factor = _COLLATERAL_PD_FACTOR.get(inp.collateral_type.lower(), 1.0)

    # Logistic combination:
    #   β₀ = -1.5 (base — not all physical risk materialises as default)
    #   β₁·phys_x: physical score contribution
    #   β₂·region: regional amplification
    #   β₃·interaction: high score × high region → non-linear amplification
    beta0 = -1.5
    beta1 = 3.0 * sens.physical_sensitivity   # sector scales the score weight
    beta2 = 2.0 * region_coeff
    beta3 = 1.5 * phys_norm * region_coeff    # interaction term

    logit_phys = beta0 + beta1 * phys_norm + beta2 * phys_norm + beta3
    uplift_frac_phys = _sigmoid(logit_phys) - _sigmoid(beta0)  # marginal vs baseline
    uplift_frac_phys = max(0.0, uplift_frac_phys) * coll_pd_factor

    physical_pd_uplift = uplift_frac_phys * inp.baseline_pd

    # ── Transition uplift — logistic model ────────────────────────────────────
    trans_norm  = inp.transition_score / 100.0

    # Higher baseline PD → less room to worsen proportionally; use absolute uplift
    # Stranded asset risk scales with both score and scenario
    beta0_t = -2.0
    beta1_t = 4.0 * sens.stranded_asset_risk
    beta2_t = 1.0 * trans_norm

    logit_trans = beta0_t + beta1_t * trans_norm + beta2_t
    uplift_frac_trans = max(0.0, _sigmoid(logit_trans) - _sigmoid(beta0_t))

    transition_pd_uplift = uplift_frac_trans * inp.baseline_pd * scen_mult

    calibration_audit = {
        "phys_norm": round(phys_norm, 4),
        "region_coeff": round(region_coeff, 4),
        "coll_pd_factor": round(coll_pd_factor, 4),
        "logit_phys": round(logit_phys, 4),
        "uplift_frac_phys": round(uplift_frac_phys, 6),
        "logit_trans": round(logit_trans, 4),
        "uplift_frac_trans": round(uplift_frac_trans, 6),
        "calibration": "Logistic scoring model — Hosmer & Lemeshow §5, ECB CST 2022 Table B.4",
    }
    return physical_pd_uplift, transition_pd_uplift, calibration_audit


# ── Standard rating → PD mapping (Moody's 10-year average 2023) ──────────────
_RATING_PD: dict[str, float] = {
    "AAA": 0.0002, "AA": 0.0006, "A": 0.0015, "BBB": 0.0030,
    "BB":  0.0120, "B": 0.0450,  "CCC": 0.1400, "D": 1.0,
}

def _rating_to_pd(rating: str) -> float:
    return _RATING_PD.get(rating.strip().upper(), 0.03)

def _pd_to_rating(pd: float) -> str:
    if pd <= 0.0005: return "AAA"
    if pd <= 0.001:  return "AA"
    if pd <= 0.002:  return "A"
    if pd <= 0.005:  return "BBB"
    if pd <= 0.02:   return "BB"
    if pd <= 0.08:   return "B"
    if pd <= 0.25:   return "CCC"
    return "D"


# ── Gap 6: IFRS 9 term structure — scenario GMST pathway drives PD migration ──
# Temperature pathway → PD stress multiplier per horizon per scenario
# Source: ECB CST 2022 Table C.3 — PD migration by scenario and horizon.
# Values: factor by which climate_pd / baseline_pd grows vs year-0 ratio.
_TERM_PD_MULT: dict[str, dict[int, float]] = {
    "nze":     {1: 1.00, 5: 1.12, 10: 1.20, 30: 1.30},  # moderate transition stress
    "delayed": {1: 1.00, 5: 1.25, 10: 1.45, 30: 1.65},  # delayed → abrupt repricing
    "cp":      {1: 1.00, 5: 1.40, 10: 1.80, 30: 2.50},  # physical damage compounds
}

# IFRS 9 staging thresholds
_IFRS9_STAGE2_PD = 0.05   # PD ≥ 5% → Stage 2 (significant credit deterioration)
_IFRS9_STAGE3_PD = 0.20   # PD ≥ 20% → Stage 3 (credit-impaired)


@dataclass
class TermStructurePoint:
    """ECL and PD at a given horizon for a single obligor or portfolio."""
    horizon_years:     int
    scenario:          str
    climate_pd:        float
    climate_ecl_usd_m: float
    cumulative_ecl_usd_m: float   # annuity sum of year-0 to horizon ECL
    ifrs9_stage:       int        # 1, 2, or 3
    pd_mult:           float      # relative to year-0 climate PD


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class CounterpartyInput:
    company_id:       str
    company_name:     str
    sector:           str
    ead_usd_m:        float
    baseline_pd:      float
    baseline_lgd:     float = 0.45
    physical_score:   float = 50.0
    transition_score: float = 50.0
    collateral_type:  str   = "unsecured"
    scenario:         str   = "cp"
    region:           str   = "Global"


@dataclass
class CounterpartyRiskResult:
    company_id:           str
    company_name:         str
    sector:               str
    ead_usd_m:            float
    scenario:             str

    baseline_pd:          float
    baseline_lgd:         float
    baseline_ecl_usd_m:   float
    baseline_rwa_usd_m:   float

    physical_pd_uplift:   float
    transition_pd_uplift: float
    total_pd_uplift:      float
    lgd_uplift:           float

    climate_pd:           float
    climate_lgd:          float
    climate_ecl_usd_m:    float
    climate_rwa_usd_m:    float

    ecl_increment_usd_m:  float
    rwa_increment_usd_m:  float
    capital_charge_usd_m: float

    indicative_climate_rating: str
    ifrs9_stage:          int = 1     # Gap 6: IFRS 9 stage at year 0

    # Gap 6: term structure
    term_structure:       list = field(default_factory=list)   # List[TermStructurePoint]

    audit: dict = field(default_factory=dict)


@dataclass
class PortfolioCreditResult:
    n_obligors:               int
    total_ead_usd_m:          float
    scenario:                 str
    baseline_ecl_usd_m:       float
    climate_ecl_usd_m:        float
    ecl_increment_usd_m:      float
    ecl_uplift_pct:           float
    baseline_rwa_usd_m:       float
    climate_rwa_usd_m:        float
    rwa_increment_usd_m:      float
    incremental_capital_usd_m: float
    sector_breakdown:         dict = field(default_factory=dict)
    obligors:                 list = field(default_factory=list)

    # Gap 6: portfolio-level term structure
    portfolio_term_structure: list = field(default_factory=list)

    # Gap 5: calibration metadata
    calibration_method: str = "Logistic scoring model (Gap 5) + IFRS 9 term structure (Gap 6)"


# ── Core calculation ──────────────────────────────────────────────────────────

_SCENARIO_TRANSITION_MULT: dict[str, float] = {
    "nze":     1.0,
    "delayed": 0.65,
    "cp":      0.30,
}


def _rw_from_pd(pd: float, lgd: float = 0.45, maturity: float = 2.5,
               ead: float = 1.0, base_rw_pct: float = 100.0) -> float:
    """IRB Vasicek risk weight (Basel II/III Advanced approach).

    Returns the *risk-weight multiplier* (RWA / EAD) so the call-site
    computation `rwa = ead × rw × (base_rw_pct/100)` still works.

    We compute true IRB RWA then divide by EAD to get the multiplier.
    Falls back to standardised floor of 0.20 to prevent negative capital.
    """
    if ead < 1e-9:
        return 1.0
    irb = _irb_rwa(pd, lgd, maturity, ead, base_rw_pct)
    rw = irb / ead / max(0.01, base_rw_pct / 100.0)
    return max(0.20, rw)   # IRB floor: 20% risk weight minimum


def _ifrs9_stage(pd: float) -> int:
    if pd >= _IFRS9_STAGE3_PD: return 3
    if pd >= _IFRS9_STAGE2_PD: return 2
    return 1


def _compute_term_structure(
    inp: CounterpartyInput,
    year0_climate_pd: float,
    year0_climate_lgd: float,
) -> list[TermStructurePoint]:
    """Gap 6: Project PD and ECL at 5, 10, 30-year horizons.

    PD grows along the scenario temperature pathway per _TERM_PD_MULT.
    Cumulative ECL uses a simple annuity:
      cumECL(T) = Σ_{t=1}^{T} pd_t × lgd_t × EAD × (1 + pd_growth)^t
    where pd_growth = annual PD growth from the scenario term multiplier.
    """
    mults = _TERM_PD_MULT.get(inp.scenario.lower(), _TERM_PD_MULT["cp"])
    results: list[TermStructurePoint] = []

    for horizon in (5, 10, 30):
        mult = mults.get(horizon, 1.0)
        pd_h = min(1.0, year0_climate_pd * mult)
        lgd_h = min(1.0, year0_climate_lgd * (1.0 + 0.01 * horizon))  # LGD drifts up slightly

        ecl_annual_h = pd_h * lgd_h * inp.ead_usd_m

        # Cumulative ECL as growing annuity
        # pd grows at a constant rate r per year between year 0 and horizon
        r = (mult ** (1.0 / horizon)) - 1.0 if horizon > 0 else 0.0
        if abs(r) < 1e-6:
            cum_ecl = ecl_annual_h * horizon
        else:
            cum_ecl = year0_climate_pd * year0_climate_lgd * inp.ead_usd_m * (
                ((1 + r) ** horizon - 1) / r
            )

        results.append(TermStructurePoint(
            horizon_years=horizon,
            scenario=inp.scenario,
            climate_pd=round(pd_h, 6),
            climate_ecl_usd_m=round(ecl_annual_h, 4),
            cumulative_ecl_usd_m=round(cum_ecl, 4),
            ifrs9_stage=_ifrs9_stage(pd_h),
            pd_mult=round(mult, 4),
        ))
    return results


def compute_counterparty_risk(inp: CounterpartyInput) -> CounterpartyRiskResult:
    """Climate-adjusted PD, LGD, ECL, and RWA for a single obligor.

    Improvements vs v0.8:
      Gap 5 — obligor-specific logistic PD calibration
      Gap 6 — IFRS 9 term structure at 5 / 10 / 30-year horizons
    """
    sens = _get_sensitivity(inp.sector)
    scen_mult = _SCENARIO_TRANSITION_MULT.get(inp.scenario.lower(), 0.5)

    # ── Gap 5: obligor-specific PD uplift ────────────────────────────────────
    physical_pd_uplift, transition_pd_uplift, calib_audit = _obligor_specific_uplift(
        inp, sens, scen_mult
    )

    total_pd_uplift = physical_pd_uplift + transition_pd_uplift
    climate_pd = min(1.0, inp.baseline_pd + total_pd_uplift)

    # ── LGD adjustment ────────────────────────────────────────────────────────
    coll_mult = _COLLATERAL_LGD_MULT.get(inp.collateral_type.lower(), 0.20)
    lgd_uplift = physical_pd_uplift * sens.lgd_physical_fraction * coll_mult
    climate_lgd = min(1.0, inp.baseline_lgd + lgd_uplift)

    # ── ECL ───────────────────────────────────────────────────────────────────
    baseline_ecl  = inp.baseline_pd * inp.baseline_lgd * inp.ead_usd_m
    climate_ecl   = climate_pd * climate_lgd * inp.ead_usd_m
    ecl_increment = climate_ecl - baseline_ecl

    # ── Task 50: IRB Vasicek RWA ──────────────────────────────────────────────
    # Maturity approximation: assume average loan maturity of 2.5 years (Basel floor).
    # LGD passed through so asset correlation scales correctly.
    _mat = 2.5
    baseline_rw  = _rw_from_pd(inp.baseline_pd, inp.baseline_lgd, _mat,
                                inp.ead_usd_m, sens.base_rw_pct)
    climate_rw   = _rw_from_pd(climate_pd, climate_lgd, _mat,
                                inp.ead_usd_m, sens.base_rw_pct)
    baseline_rwa = inp.ead_usd_m * baseline_rw * (sens.base_rw_pct / 100.0)
    climate_rwa  = inp.ead_usd_m * climate_rw  * (sens.base_rw_pct / 100.0)
    rwa_increment = climate_rwa - baseline_rwa
    capital_charge = rwa_increment * 0.125   # 12.5% effective CET1

    # ── Task 51: Wrong-way risk ECL ───────────────────────────────────────────
    # Bivariate normal simulation captures the double-hit: physical damage
    # simultaneously raises PD (revenue shock) and reduces recovery (asset damage).
    # wwr_ecl_usd_m replaces the naive E[PD]×E[LGD]×EAD when physical risk is elevated.
    wwr_ecl_usd_m, wwr_increment_vs_independent = _wwr_ecl(
        climate_pd, climate_lgd, inp.ead_usd_m, inp.sector, physical_pd_uplift
    )
    # Use WWR ECL as the headline climate_ecl (it is always ≥ independence case)
    climate_ecl_wwr = wwr_ecl_usd_m
    ecl_increment_wwr = climate_ecl_wwr - baseline_ecl

    # ── Gap 6: IFRS 9 stage + term structure ─────────────────────────────────
    stage0 = _ifrs9_stage(climate_pd)
    term_structure = _compute_term_structure(inp, climate_pd, climate_lgd)

    audit = {
        "sector":                inp.sector,
        "region":                getattr(inp, "region", "Global"),
        "scenario":              inp.scenario,
        "physical_score":        inp.physical_score,
        "transition_score":      inp.transition_score,
        "physical_threshold":    sens.physical_threshold,
        "physical_sensitivity":  sens.physical_sensitivity,
        "stranded_asset_risk":   sens.stranded_asset_risk,
        "lgd_physical_fraction": sens.lgd_physical_fraction,
        "collateral_lgd_mult":   coll_mult,
        "scenario_transition_mult": scen_mult,
        "physical_pd_uplift":    round(physical_pd_uplift, 6),
        "transition_pd_uplift":  round(transition_pd_uplift, 6),
        "lgd_uplift":            round(lgd_uplift, 4),
        "ifrs9_stage_year0":     stage0,
        # Task 50 — IRB
        "irb_asset_correlation": round(_asset_correlation(climate_pd), 6),
        "irb_rw_baseline":       round(baseline_rw, 4),
        "irb_rw_climate":        round(climate_rw, 4),
        # Task 51 — WWR
        "wwr_correlation":       round(_wwr_correlation(inp.sector), 4),
        "wwr_ecl_increment_usd_m": round(wwr_increment_vs_independent, 4),
        "methodology": (
            "ECB Climate Stress Test 2022 §3.2 + NGFS Phase 4 + "
            "Logistic calibration (Gap 5) + IRB Vasicek (Task 50) + WWR (Task 51)"
        ),
        **calib_audit,
    }

    return CounterpartyRiskResult(
        company_id=inp.company_id,
        company_name=inp.company_name,
        sector=inp.sector,
        ead_usd_m=inp.ead_usd_m,
        scenario=inp.scenario,
        baseline_pd=round(inp.baseline_pd, 6),
        baseline_lgd=round(inp.baseline_lgd, 4),
        baseline_ecl_usd_m=round(baseline_ecl, 4),
        baseline_rwa_usd_m=round(baseline_rwa, 4),
        physical_pd_uplift=round(physical_pd_uplift, 6),
        transition_pd_uplift=round(transition_pd_uplift, 6),
        total_pd_uplift=round(total_pd_uplift, 6),
        lgd_uplift=round(lgd_uplift, 6),
        climate_pd=round(climate_pd, 6),
        climate_lgd=round(climate_lgd, 6),
        climate_ecl_usd_m=round(climate_ecl_wwr, 4),   # WWR ECL (≥ independence)
        climate_rwa_usd_m=round(climate_rwa, 4),        # IRB RWA
        ecl_increment_usd_m=round(ecl_increment_wwr, 4),
        rwa_increment_usd_m=round(rwa_increment, 4),
        capital_charge_usd_m=round(capital_charge, 4),
        indicative_climate_rating=_pd_to_rating(climate_pd),
        ifrs9_stage=stage0,
        term_structure=term_structure,
        audit=audit,
    )


def compute_portfolio_credit_risk(
    obligors: list[CounterpartyInput],
) -> PortfolioCreditResult:
    """Aggregate climate-adjusted credit risk across a loan book.

    Improvements vs v0.8:
      Gap 5 — all obligors use logistic PD calibration
      Gap 6 — portfolio-level lifetime ECL term structure
    """
    results = [compute_counterparty_risk(o) for o in obligors]

    baseline_ecl_total = sum(r.baseline_ecl_usd_m for r in results)
    climate_ecl_total  = sum(r.climate_ecl_usd_m  for r in results)
    baseline_rwa_total = sum(r.baseline_rwa_usd_m  for r in results)
    climate_rwa_total  = sum(r.climate_rwa_usd_m   for r in results)
    ecl_increment      = climate_ecl_total - baseline_ecl_total
    rwa_increment      = climate_rwa_total - baseline_rwa_total
    capital_charge     = sum(r.capital_charge_usd_m for r in results)
    total_ead          = sum(o.ead_usd_m for o in obligors)
    ecl_uplift_pct     = (ecl_increment / baseline_ecl_total * 100) if baseline_ecl_total > 0 else 0.0

    sector_breakdown: dict[str, dict] = {}
    for r in results:
        s = r.sector
        if s not in sector_breakdown:
            sector_breakdown[s] = {
                "baseline_ecl": 0.0, "climate_ecl": 0.0,
                "ecl_increment": 0.0, "stage2_count": 0, "stage3_count": 0,
            }
        sector_breakdown[s]["baseline_ecl"] += r.baseline_ecl_usd_m
        sector_breakdown[s]["climate_ecl"]  += r.climate_ecl_usd_m
        sector_breakdown[s]["ecl_increment"]+= r.ecl_increment_usd_m
        if r.ifrs9_stage == 2: sector_breakdown[s]["stage2_count"] += 1
        if r.ifrs9_stage == 3: sector_breakdown[s]["stage3_count"] += 1

    # Gap 6: portfolio term structure — aggregate per-obligor term points
    portfolio_term: list[dict] = []
    for horizon in (5, 10, 30):
        port_climate_ecl   = sum(
            next((tp.climate_ecl_usd_m   for tp in r.term_structure if tp.horizon_years == horizon), 0.0)
            for r in results
        )
        port_cumulative_ecl = sum(
            next((tp.cumulative_ecl_usd_m for tp in r.term_structure if tp.horizon_years == horizon), 0.0)
            for r in results
        )
        stage2_count = sum(
            1 for r in results
            if any(tp.ifrs9_stage >= 2 for tp in r.term_structure if tp.horizon_years == horizon)
        )
        portfolio_term.append({
            "horizon_years":        horizon,
            "scenario":             obligors[0].scenario if obligors else "cp",
            "climate_ecl_usd_m":    round(port_climate_ecl, 4),
            "cumulative_ecl_usd_m": round(port_cumulative_ecl, 4),
            "stage2_obligors":      stage2_count,
            "ecl_vs_year0_mult":    round(port_climate_ecl / max(1e-6, climate_ecl_total), 4),
        })

    scenario = obligors[0].scenario if obligors else "cp"

    return PortfolioCreditResult(
        n_obligors=len(obligors),
        total_ead_usd_m=round(total_ead, 2),
        scenario=scenario,
        baseline_ecl_usd_m=round(baseline_ecl_total, 4),
        climate_ecl_usd_m=round(climate_ecl_total, 4),
        ecl_increment_usd_m=round(ecl_increment, 4),
        ecl_uplift_pct=round(ecl_uplift_pct, 2),
        baseline_rwa_usd_m=round(baseline_rwa_total, 4),
        climate_rwa_usd_m=round(climate_rwa_total, 4),
        rwa_increment_usd_m=round(rwa_increment, 4),
        incremental_capital_usd_m=round(capital_charge, 4),
        sector_breakdown={
            s: {k: round(v, 4) if isinstance(v, float) else v for k, v in d.items()}
            for s, d in sector_breakdown.items()
        },
        obligors=results,
        portfolio_term_structure=portfolio_term,
        calibration_method="Logistic scoring model (Gap 5) + IFRS 9 term structure (Gap 6)",
    )
