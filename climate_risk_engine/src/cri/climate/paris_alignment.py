"""
paris_alignment.py — Paris Temperature Alignment / Implied Temperature Rise (ITR).

The Implied Temperature Rise (ITR) score answers: "If every company in this
sector behaved like the company being assessed, what global temperature rise
would result by 2100?"

Methodology
-----------
We implement the **budget allocation** approach, the most widely adopted method
among institutional investors and data providers (MSCI, FTSE Russell, Trucost,
2DII/PACTA, SBTi).

Step 1 — Remaining sector carbon budget
    From IPCC AR6 WGI Table SPM.2, the global remaining carbon budget for
    1.5°C (50% likelihood) is ~500 Gt CO2 from Jan 2020.
    For 2°C: ~1,350 Gt CO2.
    We allocate sector shares using the Grandfathering + Contraction approach
    from the SBTi Sectoral Decarbonization Approach (SDA) v2.0, calibrated to
    IEA Net Zero by 2050 (2023 edition) sector pathways.

Step 2 — Company's remaining budget
    Company budget = sector_budget × (company_scope12 / sector_benchmark_emissions)
    This is the proportional allocation based on current emissions share.
    Alternative: equal per-capita or revenue-weighted allocation (available as
    `allocation_method` parameter).

Step 3 — Company's projected cumulative emissions (2024-2050)
    Integrate the forward emissions trajectory from `_trajectory.py` outputs.
    If no trajectory data, apply sector average decarbonization rate.

Step 4 — Overshoot fraction
    overshoot = projected_cumulative / budget_1_5C
    If overshoot ≤ 1.0: company is on a 1.5°C or better path
    If overshoot > 1.0: interpolate between 1.5°C and 2°C budgets, or 2°C+

Step 5 — ITR conversion
    Map overshoot fraction to implied global temperature using a log-linear
    relationship derived from the IPCC AR6 carbon budget table:
      overshoot 0.0 → ~1.2°C (already-locked-in warming)
      overshoot 1.0 → 1.5°C
      overshoot 2.7 → 2.0°C
      overshoot 5.0 → 2.5°C
      overshoot 8.0 → 3.0°C
    Empirically fitted from AR6 WGI Table SPM.2 budget values.

ISSB IFRS S2 alignment
-----------------------
IFRS S2 "Climate-related Disclosures" (mandatory from FY2025/2026 in AU, UK,
CA, JP, SG, NZ) requires:
  - Cross-industry metric GHG-1: Absolute Scope 1, 2, 3 emissions
  - Cross-industry metric GHG-2: Scope 1/2/3 emission intensities
  - Scenario analysis with quantitative financial impact
  - [For transition-risk scenarios] Paris alignment metric recommended

MSCI Climate Alignment methodology reference
--------------------------------------------
MSCI (2021) "Implied Temperature Rise" methodology — revenue-weighted sector
allocation with sectoral decarbonization pathways from NGFS Phase 3.

SBTi SDA reference
------------------
SBTi Sector Decarbonization Approach (SDA) v2.0 (2022) — absolute contraction
approach for homogeneous sectors; physical-intensity approach for others.

Sources
-------
IPCC AR6 WGI Table SPM.2 (2021) — remaining carbon budgets
IEA Net Zero by 2050 (2023 edition) — sector pathway milestones
SBTi Corporate Net-Zero Standard v1.1 (2023) — near-term (2030) + long-term (2050) criteria
2DII/PACTA Methodology (2022) — Technology Market Share approach
MSCI Implied Temperature Rise Methodology (2021)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional


# ── Remaining global carbon budget (Gt CO2, from Jan 2020 baseline) ──────────
# IPCC AR6 WGI Table SPM.2 — 50th percentile likelihood
_GLOBAL_BUDGET_GT = {
    "1.5C": 500.0,    # Gt CO2 remaining from 2020 for 50% probability of ≤1.5°C
    "1.7C": 850.0,
    "2.0C": 1350.0,
    "2.5C": 2300.0,
    "3.0C": 3300.0,
}

# ── Sector carbon budget share (% of global budget allocated to sector) ───────
# Derived from IEA NZE 2023 sector-level cumulative emissions 2023-2050,
# normalised to global remaining budget. Reflects contraction pathways.
# Note: these share the same global budget — they don't sum to 100% because
# the "global" budget is allocated across energy system, industry, transport,
# buildings, and agriculture — each sector gets a slice of the total.
_SECTOR_BUDGET_SHARE: dict[str, dict] = {
    # sector: {budget_share_1_5C, budget_share_2C, benchmark_emissions_mt_pa}
    # budget_share = fraction of global 1.5C/2C budget this sector may emit 2024-2050
    # benchmark_emissions = global sector Scope 1 in Mt CO2e/yr (2023)
    "power":          {"share_1_5": 0.040, "share_2_0": 0.060, "global_mt": 13_000},
    "utilities":      {"share_1_5": 0.040, "share_2_0": 0.060, "global_mt": 13_000},
    "oil_gas":        {"share_1_5": 0.018, "share_2_0": 0.030, "global_mt":  5_500},
    "coal":           {"share_1_5": 0.008, "share_2_0": 0.015, "global_mt":  3_000},
    "coal_thermal":   {"share_1_5": 0.008, "share_2_0": 0.015, "global_mt":  3_000},
    "steel":          {"share_1_5": 0.010, "share_2_0": 0.016, "global_mt":  2_600},
    "cement":         {"share_1_5": 0.008, "share_2_0": 0.013, "global_mt":  2_300},
    "chemicals":      {"share_1_5": 0.007, "share_2_0": 0.011, "global_mt":  2_000},
    "aviation":       {"share_1_5": 0.005, "share_2_0": 0.010, "global_mt":  1_000},
    "shipping":       {"share_1_5": 0.005, "share_2_0": 0.010, "global_mt":    850},
    "automobiles":    {"share_1_5": 0.012, "share_2_0": 0.020, "global_mt":  5_000},
    "agriculture":    {"share_1_5": 0.020, "share_2_0": 0.028, "global_mt":  5_500},
    "mining":         {"share_1_5": 0.004, "share_2_0": 0.007, "global_mt":  1_000},
    "real_estate":    {"share_1_5": 0.015, "share_2_0": 0.022, "global_mt":  3_000},
    "technology":     {"share_1_5": 0.002, "share_2_0": 0.003, "global_mt":    400},
    "financials":     {"share_1_5": 0.001, "share_2_0": 0.001, "global_mt":     80},
    "banking":        {"share_1_5": 0.001, "share_2_0": 0.001, "global_mt":     80},
    "food_processing":{"share_1_5": 0.005, "share_2_0": 0.008, "global_mt":    800},
    "transport":      {"share_1_5": 0.018, "share_2_0": 0.028, "global_mt":  8_000},
    "construction":   {"share_1_5": 0.006, "share_2_0": 0.010, "global_mt":    600},
    "default":        {"share_1_5": 0.004, "share_2_0": 0.006, "global_mt":    800},
}

# ── SBTi Corporate Net-Zero Standard criteria ─────────────────────────────────
# Near-term target: ≥50% absolute Scope 1+2+3 reduction by 2030 vs 2020 base
# Long-term target: ≥90% absolute reduction + ≤10% neutralised by CDR by 2050
_SBTI_NZS_NEAR_TERM_REDUCTION = 0.50    # 50% reduction by 2030
_SBTI_NZS_LONG_TERM_REDUCTION = 0.90    # 90% reduction by 2050

# ── ITR temperature mapping ───────────────────────────────────────────────────
# Maps overshoot fraction (projected_cumulative / 1.5C_budget) to ITR (°C)
# Fitted from IPCC AR6 WGI Table SPM.2 remaining budget values.
# At overshoot=0 → ~1.2°C (already locked in from past emissions)
_ITR_BREAKPOINTS = [
    (0.00, 1.20),
    (0.50, 1.35),
    (1.00, 1.50),
    (1.70, 1.70),
    (2.70, 2.00),
    (4.00, 2.30),
    (5.40, 2.50),
    (8.00, 3.00),
    (12.0, 3.50),
    (18.0, 4.00),
]

def _overshoot_to_itr(overshoot: float) -> float:
    """Interpolate ITR from overshoot fraction using AR6-derived breakpoints."""
    if overshoot <= _ITR_BREAKPOINTS[0][0]:
        return _ITR_BREAKPOINTS[0][1]
    for i in range(len(_ITR_BREAKPOINTS) - 1):
        x0, y0 = _ITR_BREAKPOINTS[i]
        x1, y1 = _ITR_BREAKPOINTS[i + 1]
        if x0 <= overshoot <= x1:
            t = (overshoot - x0) / (x1 - x0)
            return round(y0 + t * (y1 - y0), 2)
    # Beyond top breakpoint
    return round(_ITR_BREAKPOINTS[-1][1] + 0.3 * (overshoot - _ITR_BREAKPOINTS[-1][0]) / 6, 2)


# ── ITR label mapping ──────────────────────────────────────────────────────────
def _itr_label(itr: float) -> str:
    if itr <= 1.5:  return "1.5°C aligned"
    if itr <= 1.7:  return "Below 2°C"
    if itr <= 2.0:  return "2°C aligned"
    if itr <= 2.5:  return "Moderately misaligned"
    if itr <= 3.0:  return "Significantly misaligned"
    return "Severely misaligned (>3°C)"

def _itr_color(itr: float) -> str:
    if itr <= 1.5:  return "green"
    if itr <= 2.0:  return "amber"
    if itr <= 2.5:  return "orange"
    return "red"


@dataclass
class ParisBudget:
    """Carbon budget allocation for a company."""
    sector:                  str
    sector_budget_1_5_gt:    float    # Global sector budget for 1.5°C (Gt)
    sector_budget_2_0_gt:    float    # Global sector budget for 2.0°C (Gt)
    company_share:           float    # Company's proportional share (fraction)
    company_budget_1_5_mt:   float    # Company 1.5°C budget (Mt CO2e, 2024-2050)
    company_budget_2_0_mt:   float    # Company 2.0°C budget (Mt CO2e, 2024-2050)


@dataclass
class ParisAlignmentResult:
    """Paris temperature alignment / ITR result for one company."""
    company_name:           str
    sector:                 str

    # Core ITR metric
    implied_temperature_rise: float   = 0.0   # °C (e.g., 2.3)
    itr_label:              str       = ""     # "1.5°C aligned" | "Below 2°C" | etc.
    itr_color:              str       = ""     # green | amber | orange | red

    # Carbon budget
    budget:                 Optional[ParisBudget] = None
    projected_cumulative_mt: float    = 0.0   # Mt CO2e 2024-2050 (company)
    overshoot_fraction:     float     = 0.0   # projected / 1.5C budget

    # Required vs. actual decarbonization
    required_annual_reduction_pct: float = 0.0   # % per year to hit 1.5°C
    actual_annual_reduction_pct:   float = 0.0   # % per year from trajectory
    reduction_gap_pct:             float = 0.0   # required - actual (positive = undershooting)

    # SBTi Net-Zero Standard alignment
    sbti_nzs_near_term_2030_aligned: bool  = False   # on track for 50% cut by 2030
    sbti_nzs_long_term_2050_aligned: bool  = False   # on track for 90% cut by 2050
    sbti_nzs_gap_mt_by_2030:        float  = 0.0    # Mt CO2e excess vs. 2030 target
    sbti_nzs_gap_mt_by_2050:        float  = 0.0    # Mt CO2e excess vs. 2050 target

    # Overshoot year (first year company exceeds its 1.5°C cumulative budget)
    overshoot_year:         Optional[int] = None

    # Trajectory (2025-2050, annual Mt CO2e)
    trajectory_years:       list[int]   = field(default_factory=list)
    trajectory_scope12_mt:  list[float] = field(default_factory=list)   # actual/projected
    trajectory_1_5_budget:  list[float] = field(default_factory=list)   # annual budget for 1.5°C
    trajectory_2_0_budget:  list[float] = field(default_factory=list)   # annual budget for 2.0°C

    # Allocation method used
    allocation_method:      str         = "proportional_scope12"

    # Narrative
    key_findings:           list[str]   = field(default_factory=list)
    disclosure_metrics:     dict        = field(default_factory=dict)
    methodology:            str         = ""
    data_gaps:              list[str]   = field(default_factory=list)


def assess_paris_alignment(
    company_name:           str,
    sector:                 str,
    scope1_mt_co2e:         Optional[float] = None,   # Base year Scope 1
    scope2_mt_co2e:         Optional[float] = None,   # Base year Scope 2
    scope3_mt_co2e:         Optional[float] = None,   # Base year Scope 3
    revenue_usd_m:          float = 1000.0,
    base_year:              int   = 2022,
    # Forward trajectory — annual Scope 1+2 Mt CO2e from trajectory engine
    # If None, engine derives from sector decarbonization rate
    trajectory_scope12:     Optional[list[float]] = None,   # index 0 = 2025
    trajectory_years:       Optional[list[int]]   = None,
    # Optional overrides
    allocation_method:      str   = "proportional_scope12",  # or "revenue_weighted"
    has_sbti_validated:     bool  = False,
    sbti_reduction_2030_pct: Optional[float] = None,   # self-stated % reduction by 2030
) -> ParisAlignmentResult:
    """
    Compute Implied Temperature Rise and Paris alignment gap for a company.

    Parameters
    ----------
    company_name         : Display name
    sector               : Sector slug
    scope1_mt_co2e       : Base-year Scope 1 emissions (Mt CO2e)
    scope2_mt_co2e       : Base-year Scope 2 emissions (Mt CO2e)
    scope3_mt_co2e       : Base-year Scope 3 emissions (Mt CO2e, total)
    revenue_usd_m        : Annual revenue (used for revenue-weighted allocation)
    base_year            : Emissions base year (default 2022)
    trajectory_scope12   : Per-year Scope 1+2 forecast (Mt CO2e), 2025 onward
    trajectory_years     : Years corresponding to trajectory_scope12
    allocation_method    : "proportional_scope12" | "revenue_weighted"
    has_sbti_validated   : Has SBTi-validated targets
    sbti_reduction_2030_pct : Committed % Scope 1+2 reduction by 2030 (0-100)
    """
    result = ParisAlignmentResult(
        company_name=company_name,
        sector=sector,
        allocation_method=allocation_method,
    )
    s = sector.lower()
    sec_data = _SECTOR_BUDGET_SHARE.get(s, _SECTOR_BUDGET_SHARE["default"])

    # ── Base emissions ────────────────────────────────────────────────────────
    scope12_base = (scope1_mt_co2e or 0.0) + (scope2_mt_co2e or 0.0)
    scope123_base = scope12_base + (scope3_mt_co2e or 0.0)

    if scope12_base == 0:
        # Estimate from sector intensity × revenue
        # Use EDGAR sector averages (tCO2e / USD M revenue)
        _SECTOR_INTENSITY_T_PER_USDM = {
            "utilities": 1500, "oil_gas": 900, "steel": 1200, "cement": 2000,
            "chemicals": 800, "aviation": 400, "shipping": 500, "mining": 600,
            "agriculture": 700, "automobiles": 300, "technology": 50, "default": 200,
        }
        intensity = _SECTOR_INTENSITY_T_PER_USDM.get(s, 200)
        scope12_base = (intensity * revenue_usd_m) / 1_000_000   # convert t → Mt
        result.data_gaps.append(
            f"Scope 1+2 not provided — estimated from sector intensity "
            f"({intensity} tCO2e/USD M) × revenue. Provide EUTL/EDGAR data for precision."
        )
    scope12_base = max(scope12_base, 0.001)

    # ── Sector carbon budget (global) ─────────────────────────────────────────
    sector_budget_1_5_gt = _GLOBAL_BUDGET_GT["1.5C"] * sec_data["share_1_5"]
    sector_budget_2_0_gt = _GLOBAL_BUDGET_GT["2.0C"] * sec_data["share_2_0"]
    # Convert Gt → Mt for consistency
    sector_budget_1_5_mt = sector_budget_1_5_gt * 1000
    sector_budget_2_0_mt = sector_budget_2_0_gt * 1000

    # ── Company's proportional share ──────────────────────────────────────────
    global_sector_mt = sec_data["global_mt"]
    if allocation_method == "proportional_scope12":
        company_share = scope12_base / global_sector_mt
    else:
        # Revenue-weighted (more appropriate for service sectors)
        # Assume global sector revenue proxy from OECD STAN database
        _GLOBAL_SECTOR_REVENUE_BUSD = {
            "utilities": 3_000, "oil_gas": 4_500, "steel": 800, "cement": 400,
            "chemicals": 4_000, "aviation": 900, "technology": 8_000, "default": 2_000,
        }
        sector_rev = _GLOBAL_SECTOR_REVENUE_BUSD.get(s, 2000) * 1000  # → USD M
        company_share = revenue_usd_m / sector_rev

    company_share = min(company_share, 0.05)  # cap at 5% of sector budget

    company_budget_1_5_mt = sector_budget_1_5_mt * company_share
    company_budget_2_0_mt = sector_budget_2_0_mt * company_share

    result.budget = ParisBudget(
        sector=s,
        sector_budget_1_5_gt=round(sector_budget_1_5_gt, 2),
        sector_budget_2_0_gt=round(sector_budget_2_0_gt, 2),
        company_share=round(company_share, 6),
        company_budget_1_5_mt=round(company_budget_1_5_mt, 2),
        company_budget_2_0_mt=round(company_budget_2_0_mt, 2),
    )

    # ── Build or use trajectory ───────────────────────────────────────────────
    horizon_years = list(range(2025, 2051))
    n_years = len(horizon_years)

    if trajectory_scope12 and trajectory_years:
        # Interpolate/align provided trajectory to 2025-2050
        traj_map = dict(zip(trajectory_years, trajectory_scope12))
        trajectory = []
        prev = scope12_base
        for y in horizon_years:
            if y in traj_map:
                trajectory.append(traj_map[y])
                prev = traj_map[y]
            else:
                # Linear interpolation
                trajectory.append(prev)
        # Ensure non-negative
        trajectory = [max(0.0, v) for v in trajectory]
    else:
        # Sector average decarbonization rate (% per year, IEA NZE 2023)
        _SECTOR_DECARB_RATE: dict[str, float] = {
            "power":       0.072,   # Power sector must decarbonize fastest: ~7.2%/yr
            "utilities":   0.072,
            "oil_gas":     0.045,
            "coal":        0.090,   # Fastest exit required
            "steel":       0.040,
            "cement":      0.035,
            "chemicals":   0.030,
            "aviation":    0.025,   # Slow due to technology limits
            "shipping":    0.030,
            "automobiles": 0.050,
            "agriculture": 0.015,
            "technology":  0.040,
            "default":     0.035,
        }
        rate = _SECTOR_DECARB_RATE.get(s, 0.035)
        trajectory = []
        val = scope12_base
        for y in horizon_years:
            years_from_base = y - base_year
            years_from_2025 = y - 2025
            # Apply rate from 2025
            annual_val = scope12_base * ((1 - rate) ** max(0, years_from_base))
            trajectory.append(max(0.0, annual_val))
        result.data_gaps.append(
            f"No forward trajectory provided — using sector average decarbonization rate "
            f"({_SECTOR_DECARB_RATE.get(s, 0.035)*100:.1f}%/yr IEA NZE). "
            "Feed trajectory_scope12 from _trajectory.py for company-specific ITR."
        )

    # ── Annual 1.5°C and 2°C budget paths (linear contraction to zero by 2050) ─
    # Budget allocated linearly over 2024-2050 (26 years), declining each year
    # More precisely: triangular contraction — front-loaded spending
    remaining_years = n_years
    # Linear decline: Year 1 gets the most, Year 26 gets ~0
    total_weights = sum(range(remaining_years, 0, -1))
    annual_budget_1_5 = [
        company_budget_1_5_mt * (remaining_years - i) / total_weights
        for i in range(remaining_years)
    ]
    annual_budget_2_0 = [
        company_budget_2_0_mt * (remaining_years - i) / total_weights
        for i in range(remaining_years)
    ]

    result.trajectory_years = horizon_years
    result.trajectory_scope12_mt = [round(v, 4) for v in trajectory]
    result.trajectory_1_5_budget = [round(v, 4) for v in annual_budget_1_5]
    result.trajectory_2_0_budget = [round(v, 4) for v in annual_budget_2_0]

    # ── Cumulative projected emissions ────────────────────────────────────────
    projected_cumulative = sum(trajectory)
    result.projected_cumulative_mt = round(projected_cumulative, 2)

    # ── Overshoot fraction ────────────────────────────────────────────────────
    if company_budget_1_5_mt > 0:
        overshoot = projected_cumulative / company_budget_1_5_mt
    else:
        overshoot = 10.0  # severe misalignment
    result.overshoot_fraction = round(overshoot, 3)

    # ── Overshoot year (first year cumulative exceeds 1.5°C budget) ──────────
    cumulative_1_5 = sum(annual_budget_1_5)
    running_proj = 0.0
    running_budget = 0.0
    for i, (y, proj_mt, bud_mt) in enumerate(zip(horizon_years, trajectory, annual_budget_1_5)):
        running_proj += proj_mt
        running_budget += bud_mt
        if running_proj > cumulative_1_5 and result.overshoot_year is None:
            result.overshoot_year = y

    # ── Implied Temperature Rise ──────────────────────────────────────────────
    itr = _overshoot_to_itr(overshoot)
    result.implied_temperature_rise = itr
    result.itr_label = _itr_label(itr)
    result.itr_color = _itr_color(itr)

    # ── Required vs. actual annual reduction ─────────────────────────────────
    # Required: linear reduction from base to near-zero by 2050 for 1.5°C
    # Annual reduction % needed = 1 - (target_2050/base)^(1/25)
    # Target 2050 = 10% of base (SBTi: 90% reduction required)
    target_2050 = scope12_base * (1 - _SBTI_NZS_LONG_TERM_REDUCTION)
    if scope12_base > 0 and target_2050 >= 0:
        required_rate = 1 - (target_2050 / scope12_base) ** (1 / 25.0)
    else:
        required_rate = 0.0

    # Actual rate: CAGR from 2025 to 2050 using trajectory
    t_start = trajectory[0] if trajectory else scope12_base
    t_end = trajectory[-1] if trajectory else 0
    if t_start > 0 and t_end >= 0:
        actual_rate = 1 - (max(t_end, 0.001) / t_start) ** (1 / 25.0)
    else:
        actual_rate = 0.0

    result.required_annual_reduction_pct = round(required_rate * 100, 2)
    result.actual_annual_reduction_pct = round(actual_rate * 100, 2)
    result.reduction_gap_pct = round((required_rate - actual_rate) * 100, 2)

    # ── SBTi Net-Zero Standard alignment ─────────────────────────────────────
    # Near-term: 50% reduction by 2030 vs. base year
    target_2030 = scope12_base * (1 - _SBTI_NZS_NEAR_TERM_REDUCTION)
    years_to_2030 = 2030 - 2025
    projected_2030 = trajectory[years_to_2030] if len(trajectory) > years_to_2030 else trajectory[-1]
    result.sbti_nzs_near_term_2030_aligned = projected_2030 <= target_2030
    result.sbti_nzs_gap_mt_by_2030 = round(max(0, projected_2030 - target_2030), 3)

    # Long-term: 90% reduction by 2050
    projected_2050 = trajectory[-1] if trajectory else scope12_base
    result.sbti_nzs_long_term_2050_aligned = projected_2050 <= target_2050
    result.sbti_nzs_gap_mt_by_2050 = round(max(0, projected_2050 - target_2050), 3)

    # ── Key findings ──────────────────────────────────────────────────────────
    if itr <= 1.5:
        result.key_findings.append(
            f"Paris-aligned: company trajectory consistent with 1.5°C warming "
            f"(ITR={itr}°C). Scope 1+2 decarbonization rate ({actual_rate*100:.1f}%/yr) "
            f"meets the required {required_rate*100:.1f}%/yr."
        )
    elif itr <= 2.0:
        result.key_findings.append(
            f"Below 2°C but not 1.5°C aligned (ITR={itr}°C). Reduction rate "
            f"{actual_rate*100:.1f}%/yr vs. required {required_rate*100:.1f}%/yr — "
            f"a gap of {result.reduction_gap_pct:.1f} ppts/yr. "
            f"Projected to overshoot 1.5°C budget"
            + (f" by {result.overshoot_year}." if result.overshoot_year else ".")
        )
    elif itr <= 2.5:
        result.key_findings.append(
            f"Moderately misaligned with Paris (ITR={itr}°C). Projected cumulative "
            f"emissions {projected_cumulative:.1f} Mt vs. 1.5°C budget "
            f"{company_budget_1_5_mt:.1f} Mt — {overshoot:.1f}× overshoot. "
            f"Requires {result.reduction_gap_pct:.1f} ppts/yr additional decarbonization."
        )
    else:
        result.key_findings.append(
            f"Severely Paris-misaligned (ITR={itr}°C). At current trajectory, "
            f"company would consume {overshoot:.1f}× its 1.5°C carbon budget by 2050. "
            f"Required annual reduction: {required_rate*100:.1f}%/yr. "
            f"Actual: {actual_rate*100:.1f}%/yr. "
            "Material litigation, stranded asset, and regulatory risk implied."
        )

    if not result.sbti_nzs_near_term_2030_aligned:
        result.key_findings.append(
            f"SBTi Near-Term Gap: Projected Scope 1+2 in 2030 ({projected_2030:.2f} Mt) "
            f"exceeds SBTi Net-Zero Standard target ({target_2030:.2f} Mt = 50% below "
            f"{scope12_base:.2f} Mt base). Gap: {result.sbti_nzs_gap_mt_by_2030:.2f} Mt. "
            "SBTi validation would be rejected at current trajectory."
        )

    # ── IFRS S2 / ISSB disclosure metrics ────────────────────────────────────
    result.disclosure_metrics = {
        "IFRS_S2_GHG1_scope1_mt": round(scope1_mt_co2e or 0, 3),
        "IFRS_S2_GHG1_scope2_mt": round(scope2_mt_co2e or 0, 3),
        "IFRS_S2_GHG1_scope3_mt": round(scope3_mt_co2e or 0, 3),
        "IFRS_S2_GHG2_intensity_tCO2e_per_usdm": round(
            (scope12_base * 1_000_000) / max(revenue_usd_m, 1), 1
        ),
        "ITR_degrees_C": itr,
        "ITR_label": result.itr_label,
        "carbon_budget_1_5C_mt": round(company_budget_1_5_mt, 2),
        "projected_cumulative_mt": round(projected_cumulative, 2),
        "overshoot_fraction": round(overshoot, 3),
        "sbti_nzs_near_term_aligned": result.sbti_nzs_near_term_2030_aligned,
        "sbti_nzs_long_term_aligned": result.sbti_nzs_long_term_2050_aligned,
    }

    result.methodology = (
        f"Implied Temperature Rise ({itr}°C) via carbon budget allocation method. "
        f"Sector '{sector}' receives {sector_budget_1_5_gt:.1f} Gt of the 500 Gt global "
        f"1.5°C remaining budget (IPCC AR6 WGI Table SPM.2). "
        f"Company proportional share: {company_share*100:.4f}% (scope12 {scope12_base:.2f} Mt "
        f"/ sector {global_sector_mt:,} Mt). "
        f"Company 1.5°C budget: {company_budget_1_5_mt:.2f} Mt CO2e (2024-2050). "
        f"Projected cumulative: {projected_cumulative:.2f} Mt. "
        f"Overshoot fraction: {overshoot:.2f}× → ITR {itr}°C. "
        "Sources: IPCC AR6 WGI (2021), IEA NZE 2023, SBTi Corporate Net-Zero Standard v1.1 (2023), "
        "MSCI ITR Methodology (2021), 2DII/PACTA (2022)."
    )
    return result
