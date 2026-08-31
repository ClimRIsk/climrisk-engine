"""
Climate-Adjusted Credit Risk — PD / LGD / ECL / RWA.

Implements the counterparty risk layer of an Aladdin-style platform:
maps physical and transition climate risk scores to adjustments on
standard credit risk parameters used in bank loan books and bond portfolios.

Regulatory alignment
--------------------
· IFRS 9 ECL  — Expected Credit Loss = PD × LGD × EAD
· Basel III   — Risk-Weighted Assets = RW × EAD
· ECB Climate Stress Test (2022) methodology for PD uplift
· NGFS Climate Scenarios Phase 4 — sector-level transition shock lookup

Method
------
1. Physical PD uplift
   Climate-adjusted PD = baseline_PD × (1 + physical_uplift)
   physical_uplift = max(0, physical_score − threshold) / 100
                     × sector_physical_sensitivity

   The threshold (default 30) means scores below 30/100 produce no uplift —
   these are low-exposure sectors (tech, financials) where physical hazard is
   remote or diversified away at portfolio level.

2. Transition PD uplift
   transition_uplift = transition_score / 100 × sector_stranded_asset_risk

   Reflects the risk that high-carbon assets become economically stranded
   under rapid decarbonisation (NGFS NZE 2050 / delayed transition paths).

3. LGD adjustment
   Climate risk can degrade collateral value (physical damage to real assets)
   and reduce recovery rates in distressed sales (lower buyer appetite for
   climate-exposed assets). LGD is uplifted by a fraction of physical_uplift.

4. ECL = climate_adjusted_PD × climate_adjusted_LGD × EAD

5. Basel III RWA impact
   Under the Standardised Approach, RW is PD-sensitive via rating migration.
   We model this as the incremental capital charge vs a stable-climate baseline.

References
----------
· ECB Climate Stress Test 2022 — Methodology Note, Section 3.2
· NGFS Technical Document on Climate Scenarios (Phase 4, 2023) — Table A.2
· BCBS Working Paper 37 (2021) — Climate-related financial risks
· BIS Quarterly Review (2020) — "How much does climate change cost banks?"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# ── Sector sensitivity parameters ─────────────────────────────────────────────

@dataclass
class SectorCreditSensitivity:
    """Calibration constants for one sector."""
    # Physical: how much does a 1-unit increase in physical_score drive PD?
    physical_sensitivity:    float   # 0–1 scalar (multiplied by score/100)
    # Transition: how much does transition score drive PD via stranded assets?
    stranded_asset_risk:     float   # 0–1.5 scalar
    # Physical: fraction of physical_uplift that flows into LGD uplift
    lgd_physical_fraction:   float   # typically 0.3–0.6
    # Basel III Standardised Approach base risk weight (%) for a BBB-equivalent obligor
    base_rw_pct:             float   # e.g. 100 = 100%
    # Threshold physical score below which no uplift applies
    physical_threshold:      float = 30.0


# Sector sensitivity table — calibrated against ECB Climate Stress Test 2022
# and NGFS Phase 4 sector-level scenario analysis.
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

_DEFAULT_SENSITIVITY = SectorCreditSensitivity(
    physical_sensitivity=0.30,
    stranded_asset_risk=0.40,
    lgd_physical_fraction=0.25,
    base_rw_pct=100.0,
    physical_threshold=30.0,
)


def _get_sensitivity(sector: str) -> SectorCreditSensitivity:
    s = sector.strip()
    return _SECTOR_SENSITIVITY.get(s, _DEFAULT_SENSITIVITY)


# ── Standard rating → PD mapping (Moody's 10-year average, 2023) ─────────────
_RATING_PD: dict[str, float] = {
    "AAA": 0.0002, "AA": 0.0006, "A": 0.0015, "BBB": 0.0030,
    "BB":  0.0120, "B": 0.0450,  "CCC": 0.1400, "D": 1.0,
}

def _rating_to_pd(rating: str) -> float:
    r = rating.strip().upper()
    return _RATING_PD.get(r, 0.03)   # default 3% if not recognised


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class CounterpartyInput:
    """Single borrower / bond issuer input.

    Parameters
    ----------
    company_id       : Identifier string (must match COMPANY_REGISTRY or user-supplied).
    company_name     : Display name.
    sector           : Sector string — used for sensitivity lookup.
    ead_usd_m        : Exposure at default (loan balance / bond notional, USD M).
    baseline_pd      : Baseline annual probability of default (0–1). Use
                       _rating_to_pd() to convert a credit rating string.
    baseline_lgd     : Baseline loss given default (0–1). Typically 0.40–0.60.
    physical_score   : 0–100 physical risk score (from CRI engine or user input).
    transition_score : 0–100 transition risk score.
    collateral_type  : "real_estate" | "equipment" | "inventory" | "unsecured"
    scenario         : "nze" | "delayed" | "cp"  — NGFS scenario for uplift.
    """
    company_id:       str
    company_name:     str
    sector:           str
    ead_usd_m:        float
    baseline_pd:      float              # annual PD, e.g. 0.03 for 3%
    baseline_lgd:     float = 0.45       # typical corporate LGD
    physical_score:   float = 50.0
    transition_score: float = 50.0
    collateral_type:  str   = "unsecured"
    scenario:         str   = "cp"       # worst-case for stress reporting


@dataclass
class CounterpartyRiskResult:
    """Climate-adjusted credit risk output per obligor."""
    company_id:           str
    company_name:         str
    sector:               str
    ead_usd_m:            float
    scenario:             str

    # Baseline (pre-climate adjustment)
    baseline_pd:          float
    baseline_lgd:         float
    baseline_ecl_usd_m:   float
    baseline_rwa_usd_m:   float

    # Climate adjustments
    physical_pd_uplift:   float   # additive uplift to PD (e.g. 0.012 = +1.2pp)
    transition_pd_uplift: float
    total_pd_uplift:      float
    lgd_uplift:           float

    # Climate-adjusted figures
    climate_pd:           float
    climate_lgd:          float
    climate_ecl_usd_m:    float
    climate_rwa_usd_m:    float

    # Incremental impact
    ecl_increment_usd_m:  float   # climate_ecl − baseline_ecl
    rwa_increment_usd_m:  float   # incremental RWA from climate PD migration
    capital_charge_usd_m: float   # incremental capital at 12.5% CET1 ratio

    # Rating migration (indicative letter rating based on climate-adjusted PD)
    indicative_climate_rating: str

    # Audit
    audit: dict = field(default_factory=dict)


@dataclass
class PortfolioCreditResult:
    """Aggregate credit risk result for a portfolio of obligors."""
    n_obligors:               int
    total_ead_usd_m:          float
    scenario:                 str
    # Aggregate ECL
    baseline_ecl_usd_m:       float
    climate_ecl_usd_m:        float
    ecl_increment_usd_m:      float
    ecl_uplift_pct:           float      # % increase vs baseline
    # Aggregate RWA
    baseline_rwa_usd_m:       float
    climate_rwa_usd_m:        float
    rwa_increment_usd_m:      float
    # Capital
    incremental_capital_usd_m: float
    # By sector
    sector_breakdown:         dict = field(default_factory=dict)
    # Per-obligor
    obligors:                 list = field(default_factory=list)  # List[CounterpartyRiskResult]


# ── Core calculation ──────────────────────────────────────────────────────────

# Collateral LGD stress multiplier — physical risk degrades collateral faster
# for real assets than for intangibles.
_COLLATERAL_LGD_MULT: dict[str, float] = {
    "real_estate": 0.60,   # coastal/flood-exposed property values drop
    "equipment":   0.45,   # physical damage to machinery
    "inventory":   0.35,   # stock damaged by climate events
    "unsecured":   0.20,   # general corporate (no specific collateral)
}

# Transition scenario multiplier on transition_pd_uplift
_SCENARIO_TRANSITION_MULT: dict[str, float] = {
    "nze":     1.0,    # aggressive decarbonisation — highest transition pressure
    "delayed": 0.65,   # slower, gives firms time to adapt
    "cp":      0.30,   # current policies — minimal transition pressure (but high physical)
}

# Indicative rating from annual PD
def _pd_to_rating(pd: float) -> str:
    if pd <= 0.0005: return "AAA"
    if pd <= 0.001:  return "AA"
    if pd <= 0.002:  return "A"
    if pd <= 0.005:  return "BBB"
    if pd <= 0.02:   return "BB"
    if pd <= 0.08:   return "B"
    if pd <= 0.25:   return "CCC"
    return "D"


def compute_counterparty_risk(inp: CounterpartyInput) -> CounterpartyRiskResult:
    """Compute climate-adjusted PD, LGD, ECL, and RWA for a single obligor.

    Parameters
    ----------
    inp : CounterpartyInput — see dataclass definition above.

    Returns
    -------
    CounterpartyRiskResult with baseline and climate-adjusted figures.
    """
    sens = _get_sensitivity(inp.sector)

    # ── Physical PD uplift ────────────────────────────────────────────────────
    phys_excess = max(0.0, inp.physical_score - sens.physical_threshold)
    physical_pd_uplift = (phys_excess / 100.0) * sens.physical_sensitivity * inp.baseline_pd

    # ── Transition PD uplift ──────────────────────────────────────────────────
    scen_mult = _SCENARIO_TRANSITION_MULT.get(inp.scenario.lower(), 0.5)
    transition_pd_uplift = (
        (inp.transition_score / 100.0)
        * sens.stranded_asset_risk
        * inp.baseline_pd
        * scen_mult
    )

    total_pd_uplift = physical_pd_uplift + transition_pd_uplift
    climate_pd = min(1.0, inp.baseline_pd + total_pd_uplift)

    # ── LGD adjustment ────────────────────────────────────────────────────────
    coll_mult = _COLLATERAL_LGD_MULT.get(inp.collateral_type.lower(), 0.20)
    lgd_uplift = physical_pd_uplift * sens.lgd_physical_fraction * coll_mult
    climate_lgd = min(1.0, inp.baseline_lgd + lgd_uplift)

    # ── ECL ───────────────────────────────────────────────────────────────────
    baseline_ecl  = inp.baseline_pd  * inp.baseline_lgd  * inp.ead_usd_m
    climate_ecl   = climate_pd       * climate_lgd        * inp.ead_usd_m
    ecl_increment = climate_ecl - baseline_ecl

    # ── RWA (Standardised Approach approximation) ─────────────────────────────
    # RW under SA: BBB (100%), B (150%), CCC (250%) — simplified PD → RW lookup
    def _rw_from_pd(pd: float) -> float:
        if pd <= 0.005: return 1.00   # IG
        if pd <= 0.02:  return 1.00   # BB
        if pd <= 0.08:  return 1.50   # B
        if pd <= 0.25:  return 2.50   # CCC
        return 3.00                   # D-territory

    baseline_rw = _rw_from_pd(inp.baseline_pd)
    climate_rw  = _rw_from_pd(climate_pd)
    baseline_rwa  = inp.ead_usd_m * baseline_rw * (sens.base_rw_pct / 100.0)
    climate_rwa   = inp.ead_usd_m * climate_rw  * (sens.base_rw_pct / 100.0)
    rwa_increment = climate_rwa - baseline_rwa
    # CET1 capital at 8% minimum (Basel III Pillar 1) + 4.5% buffer = 12.5% effective
    capital_charge = rwa_increment * 0.125

    audit = {
        "sector":                inp.sector,
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
        "methodology": "ECB Climate Stress Test 2022 §3.2 + NGFS Phase 4 sector shocks",
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
        climate_ecl_usd_m=round(climate_ecl, 4),
        climate_rwa_usd_m=round(climate_rwa, 4),
        ecl_increment_usd_m=round(ecl_increment, 4),
        rwa_increment_usd_m=round(rwa_increment, 4),
        capital_charge_usd_m=round(capital_charge, 4),
        indicative_climate_rating=_pd_to_rating(climate_pd),
        audit=audit,
    )


def compute_portfolio_credit_risk(
    obligors: list[CounterpartyInput],
) -> PortfolioCreditResult:
    """Aggregate climate-adjusted credit risk across a loan book.

    Parameters
    ----------
    obligors : List of CounterpartyInput — one per borrower / bond position.

    Returns
    -------
    PortfolioCreditResult with aggregate ECL uplift, RWA, capital charge.
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

    # Sector breakdown of ECL increment
    sector_breakdown: dict[str, dict] = {}
    for r in results:
        s = r.sector
        if s not in sector_breakdown:
            sector_breakdown[s] = {"baseline_ecl": 0.0, "climate_ecl": 0.0, "ecl_increment": 0.0}
        sector_breakdown[s]["baseline_ecl"] += r.baseline_ecl_usd_m
        sector_breakdown[s]["climate_ecl"]  += r.climate_ecl_usd_m
        sector_breakdown[s]["ecl_increment"]+= r.ecl_increment_usd_m

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
            s: {k: round(v, 4) for k, v in d.items()}
            for s, d in sector_breakdown.items()
        },
        obligors=results,
    )
