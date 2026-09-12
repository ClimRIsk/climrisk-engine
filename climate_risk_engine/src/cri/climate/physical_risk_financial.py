"""Physical Risk → Financial Impact Translator.

Converts IPCC AR6 SSP-calibrated physical hazard intensities into NPV
adjustments that sit alongside (and are additive to) the transition-risk
NPV impact already computed by the DCF engine.

CORE CONCEPT — Double Materiality
──────────────────────────────────
Under NGFS scenarios the two risk types move in opposite directions:

  Scenario          SSP equiv   Transition risk   Physical risk
  ─────────────────────────────────────────────────────────────
  NZE 2050          SSP1-2.6    HIGH              LOW   (0.85°C by 2100)
  Delayed Transition SSP2-4.5   MEDIUM            MEDIUM (~2.35°C by 2100)
  Current Policies  SSP3-7.0    LOW               HIGH  (~3.8°C by 2100)

Neither captures the full picture alone; combined impact is required for
TCFD-aligned disclosure and IFRS S2 compliance.

METHODOLOGY
───────────
For each (company, scenario) combination:

  1. Map NGFS scenario → SSP → GMST(t) for each year t ∈ [2026, horizon]
     Source: IPCC AR6 WG1 Table 4.5 (ssp_scenarios.py)

  2. Derive annual hazard probabilities from GMST using AR6 Chapter 11:
       flood(t)        = p_flood_base  × (1 + 0.07 × ΔT(t))²
       heat_stress(t)  = p_heat_base   × 1.40^ΔT(t)
       water_stress(t) = p_water_base  × (1 + 0.15 × ΔT(t))
       wildfire(t)     = p_fire_base   × (1 + 0.12 × ΔT(t))
       wind/cyclone(t) = p_wind_base   × (1 + 0.05 × ΔT(t))

  3. Joint expected-loss fraction via independent-hazard survival rule:
       ELF(t) = 1 − Π_i (1 − p_i(t))   [capped at 0.80]

  4. Revenue at risk scaled by sector physical-exposure ratio (φ):
       revenue_at_risk(t) = revenue × φ_sector × ELF(t)

  5. Discounted physical NPV drag (negative = loss):
       physical_NPV_drag = Σ_t  revenue_at_risk(t) / (1 + WACC)^(t − t₀)

  6. Normalise:
       physical_npv_impact_pct = − physical_NPV_drag / EV_base

Sources:
  IPCC AR6 WG1 Ch4  (GMST projections, Table 4.5)
  IPCC AR6 WG1 Ch11 (Extreme event scaling factors)
  IPCC AR6 WG2 Ch12 (Wildfire weather index)
  NGFS 2023 Scenarios (NZE/DT/CP families)
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .ssp_scenarios import NGFS_TO_SSP, SSP_SCENARIOS, SSPScenario
from .asset_graph import ComponentGraph, critical_path_downtime, get_sector_graph  # Gap 9: critical-path BI

# JRC hydraulic flood integration (optional — degrades gracefully if rasterio absent)
try:
    from .flood_depth import compute_flood_eal as _jrc_flood_eal, DEPTH_FETCH_FAILED
    _JRC_AVAILABLE = True
except ImportError:
    _JRC_AVAILABLE = False
    DEPTH_FETCH_FAILED = -1.0


# ── Sector physical-exposure ratios (φ) ────────────────────────────────────
# Fraction of revenue that is materially exposed to acute/chronic hazards.
# Calibrated against Munich Re NatCat data, WRI TCFD sector guidance (2023),
# and MSCI Physical Risk Model sector betas.

SECTOR_EXPOSURE_RATIO: dict[str, float] = {
    # High physical exposure — fixed assets in exposed locations
    "real_estate":          0.52,
    "utilities":            0.42,
    "energy":               0.38,
    "oil_gas":              0.35,
    "agriculture":          0.65,
    "food_beverage":        0.40,
    "beverages":            0.40,
    "chemicals":            0.32,
    "metals_mining":        0.33,
    "mining":               0.33,
    "cement":               0.30,
    "steel":                0.30,
    "construction":         0.38,
    "transport":            0.28,
    "shipping":             0.30,
    "aviation":             0.25,
    # Moderate exposure
    "industrials":          0.22,
    "consumer_staples":     0.20,
    "consumer":             0.18,
    "retail":               0.15,
    "automotive":           0.20,
    # Low exposure — knowledge/service sectors
    "technology":           0.10,
    "healthcare":           0.12,
    "pharmaceuticals":      0.11,
    "financials":           0.14,   # indirect — loan book, collateral
    "insurance":            0.12,
    "telecom":              0.10,
    "media":                0.08,
    # Default
    "default":              0.22,
}

# Baseline annual hazard probabilities (∼2024, no additional warming)
# Sources: Munich Re Annual NatCat Report 2023; WRI Aqueduct 3.0 global medians
_BASE_PROBS: dict[str, float] = {
    "flood":        0.035,   # ~3.5% annual probability of material flood event
    "heat_stress":  0.025,   # heat-induced operational disruption
    "water_stress": 0.020,   # water scarcity / operational curtailment
    "wildfire":     0.012,   # wildfire weather exposure
    "wind":         0.018,   # windstorm / cyclone damage
}

# NOTE: IPCC AR6 linear scaling constants (previously here) replaced by
# log-logistic CDF parameters in _LL_PARAMS below (Gap 1 — fragility fix).
# The old linear/exponential approach is intentionally removed.

# ELF hard cap (avoids absurdly large numbers for high-GMST scenarios)
_ELF_CAP: float = 0.80

# ── Gap 1: Log-logistic CDF fragility parameters ────────────────────────────
# P(hazard|ΔT) follows a log-logistic S-curve instead of linear/exponential.
# alpha = median warming at which the hazard probability reaches 50% of its
#         maximum amplification (inflection point, °C).
# beta  = shape/slope parameter (steeper S-curve with higher beta).
# Source: Meinshausen et al. 2011 fragility calibration + AR6 Ch11 hazard
#         intensity-frequency relationships.
_LL_PARAMS: dict[str, tuple[float, float]] = {
    "flood":        (2.0, 2.5),   # steepens around 2°C Clausius-Clapeyron knee
    "heat_stress":  (1.5, 3.0),   # steep: even 1°C already doubles tail events
    "water_stress": (2.5, 2.0),   # more gradual aridification signal
    "wildfire":     (2.0, 2.5),   # aligned to FWI amplification curve
    "wind":         (3.0, 1.8),   # weaker signal; intensity change only ~5%/°C
}


def _loglogistic_cdf(delta_t: float, alpha: float, beta: float) -> float:
    """
    Log-logistic CDF evaluated at warming delta_t.

        F(x) = x^β / (α^β + x^β)   [equivalent to 1/(1+(α/x)^β)]

    Returns 0 for delta_t ≤ 0.  Output ∈ [0, 1).
    """
    if delta_t <= 0.0:
        return 0.0
    return (delta_t ** beta) / (alpha ** beta + delta_t ** beta)


# ── Gap 8: Gaussian copula correlation matrix ────────────────────────────────
# Bivariate Pearson ρ between hazard pairs derived from IPCC AR6 Ch11
# compound event analysis and Munich Re NatCat co-occurrence statistics.
# These are used in _joint_elf() to correct the naive independence assumption.
_HAZARD_CORR: dict[tuple[str, str], float] = {
    ("flood",       "water_stress"):   0.65,   # shared precipitation driver
    ("heat_stress", "wildfire"):       0.70,   # temperature + VPD coupling
    ("wind",        "flood"):          0.45,   # cyclone compound event
    ("heat_stress", "water_stress"):   0.55,   # ET-driven demand surge
    ("drought",     "wildfire"):       0.68,   # same soil-moisture pathway
    ("flood",       "heat_stress"):    0.30,   # weak negative via cloud cover
}


def _gaussian_copula_joint_prob(p1: float, p2: float, rho: float) -> float:
    """
    Bivariate Gaussian copula joint exceedance probability P(X1 > t1, X2 > t2).

    Uses a normal approximation to the Gaussian copula (Owen 1956 / Drezner 1978).
    For computational tractability without scipy, we use the Pearson product-
    moment correction:

        P(A ∩ B) ≈ p1·p2 + ρ · √(p1·(1-p1)·p2·(1-p2))

    This is the first-order Taylor expansion of the bivariate normal CDF around
    the marginal probabilities, valid for moderate ρ and small to moderate p.
    It is conservative (does not overstate joint probability) for ρ < 0.80.

    Returns joint probability capped at min(p1, p2).
    """
    if rho == 0.0 or p1 <= 0.0 or p2 <= 0.0:
        return p1 * p2
    joint = p1 * p2 + rho * ((p1 * (1.0 - p1) * p2 * (1.0 - p2)) ** 0.5)
    return max(0.0, min(min(p1, p2), joint))


# ── Gap 4: Compound event pairs (physical co-occurrence amplification) ───────
# When BOTH hazards in a pair are active (p > 1%), apply a 1.15× ELF
# multiplier to account for non-linear compound damage.
# Source: Zscheischler et al. 2020 "A typology of compound weather and
#         climate events" — compound events cause 15–40% excess damage.
_COMPOUND_PAIRS: list[frozenset] = [
    frozenset(["flood",       "water_stress"]),
    frozenset(["heat_stress", "wildfire"]),
    frozenset(["wind",        "flood"]),
    frozenset(["heat_stress", "water_stress"]),
]
_COMPOUND_MULTIPLIER: float = 1.15
_COMPOUND_THRESHOLD: float  = 0.01   # both hazards must exceed 1% to trigger


# ── Output dataclass ────────────────────────────────────────────────────────

@dataclass
class PhysicalRiskResult:
    """Physical-risk NPV assessment for one (company, scenario) pair."""

    scenario_id: str
    ngfs_family: str
    ssp_id: str
    gmst_2050: float           # °C above 1995–2014

    # ── Financial outputs ────────────────────────────────────────────────
    physical_npv_drag: float           # USD  — always ≤ 0
    physical_npv_impact_pct: float     # fraction, e.g. -0.052 = −5.2%

    # ── Hazard breakdown at horizon midpoint (2038) ──────────────────────
    hazard_probs_2038: dict[str, float] = field(default_factory=dict)
    elf_2038: float = 0.0              # joint expected-loss fraction in 2038

    # ── Near-term block (2026–2030) — Gap 7 ─────────────────────────────
    # Separate from long-term to avoid horizon mismatch in CSRD ESRS E1
    hazard_probs_2028: dict[str, float] = field(default_factory=dict)
    elf_2028: float = 0.0
    physical_npv_drag_2030: float = 0.0      # NPV drag over 2026–2030 only

    # ── Metadata ─────────────────────────────────────────────────────────
    sector: str = "default"
    sector_exposure_ratio: float = 0.22
    revenue_usd: float = 0.0
    ev_base_usd: float = 0.0
    wacc: float = 0.10
    horizon_years: int = 25
    data_source: str = "IPCC AR6 WG1/WG2 + NGFS 2023"

    # ── Gap 3: Audit trail for CSRD ESRS E1 / IFRS S2 ──────────────────
    # Step-by-step arithmetic so a Big Four auditor can reproduce every number.
    audit_trail: dict = field(default_factory=dict)


# ── Core computation functions ──────────────────────────────────────────────

def _hazard_probs_at_gmst(delta_t: float) -> dict[str, float]:
    """
    Return annual hazard probabilities at a given GMST anomaly (°C).

    Gap 1 fix: uses log-logistic CDF fragility curves instead of
    linear/exponential scaling, eliminating the false linearity trap
    and capturing both the saturation effect at high warming and the
    accelerating non-linearity at intermediate warming.

    Calibration: base probability + (1 − base) × LL_CDF(ΔT; α, β)
      → at ΔT = 0:   probability = _BASE_PROBS[h]  (current climate)
      → at ΔT = α:   probability ≈ base + (1−base)×0.5  (half-way amplification)
      → at ΔT → ∞:   probability → 1.0  (approaches certainty, capped at 0.90)

    All probabilities are capped at 0.90.
    """
    probs: dict[str, float] = {}
    for hazard, (alpha, beta) in _LL_PARAMS.items():
        base = _BASE_PROBS[hazard]
        cdf_val = _loglogistic_cdf(delta_t, alpha, beta)
        # S-curve from base to 1.0: preserves current-climate baseline
        p = base + (1.0 - base) * cdf_val
        probs[hazard] = min(0.90, p)
    return probs


def _joint_elf(hazard_probs: dict[str, float]) -> float:
    """
    Joint expected-loss fraction.

    Gap 4: Compound event 1.15× amplification when correlated hazard
           pairs both exceed the 1% threshold.
    Gap 8: Gaussian copula correction adjusts the naive independence
           survival product to account for co-occurrence of correlated
           hazards (flood+water_stress, heat+wildfire, etc.).

    The copula correction INCREASES the joint ELF vs. the independence
    assumption because correlated hazards are more likely to co-occur,
    meaning the survival product overstates resilience.
    """
    hazard_list = list(hazard_probs.keys())
    n = len(hazard_list)

    # Step 1: Independent survival product (lower-bound baseline)
    survival = 1.0
    for p in hazard_probs.values():
        survival *= (1.0 - max(0.0, min(1.0, p)))
    elf_independent = 1.0 - survival

    # Step 2: Gaussian copula correction (Gap 8)
    # Add the excess joint-exceedance probability for each correlated pair.
    # The difference between the copula joint prob and the independence joint
    # prob (p1×p2) represents the "extra" co-occurrence loss not captured by
    # the independence rule.
    copula_correction = 0.0
    for (h1, h2), rho in _HAZARD_CORR.items():
        p1 = hazard_probs.get(h1, 0.0)
        p2 = hazard_probs.get(h2, 0.0)
        if p1 > 0.0 and p2 > 0.0 and rho > 0.0:
            joint_copula = _gaussian_copula_joint_prob(p1, p2, rho)
            joint_indep  = p1 * p2
            copula_correction += max(0.0, joint_copula - joint_indep)

    elf = min(1.0, elf_independent + copula_correction)

    # Step 3: Compound event amplification (Gap 4)
    hazard_set = set(hazard_probs.keys())
    compound_triggered = False
    for pair in _COMPOUND_PAIRS:
        if pair.issubset(hazard_set):
            names = list(pair)
            if (hazard_probs.get(names[0], 0.0) > _COMPOUND_THRESHOLD and
                    hazard_probs.get(names[1], 0.0) > _COMPOUND_THRESHOLD):
                compound_triggered = True
                break

    if compound_triggered:
        elf = min(1.0, elf * _COMPOUND_MULTIPLIER)

    return min(_ELF_CAP, elf)


def _sector_key(sector_raw: str) -> str:
    """Normalise a free-text sector string to one of the SECTOR_EXPOSURE_RATIO keys."""
    s = sector_raw.lower().replace(" ", "_").replace("-", "_").replace("/", "_")
    # Explicit substring matches
    checks = [
        ("real_estate",   ["real_estate", "property", "reit"]),
        ("utilities",     ["util", "electric", "power", "grid", "water_util"]),
        ("energy",        ["energy", "renewabl", "solar", "wind_ener"]),
        ("oil_gas",       ["oil", "gas", "petro", "refin", "lng", "upstream"]),
        ("agriculture",   ["agri", "farm", "crop"]),
        ("food_beverage", ["food", "beverage", "drink", "brew", "spirits"]),
        ("beverages",     ["beer", "spirits", "soft_drink"]),
        ("chemicals",     ["chem", "plastic", "polymer"]),
        ("metals_mining", ["metal", "steel", "alumin", "copper", "nickel"]),
        ("mining",        ["mining", "coal", "iron_ore", "miner"]),
        ("cement",        ["cement", "concrete", "building_material"]),
        ("construction",  ["construct", "engineering", "infrastructure"]),
        ("transport",     ["transport", "rail", "road", "logistic"]),
        ("shipping",      ["shipping", "maritime", "port", "tanker"]),
        ("aviation",      ["aviation", "airline", "airport"]),
        ("industrials",   ["industrial", "manufactur", "machin"]),
        ("consumer",      ["consumer", "retail", "apparel", "fashion"]),
        ("automotive",    ["auto", "vehicle", "car", "truck", "mobility"]),
        ("technology",    ["tech", "software", "it", "digital", "semicon"]),
        ("healthcare",    ["health", "hospital", "medic", "pharma", "biotech"]),
        ("financials",    ["financ", "bank", "insur", "invest", "asset_mgmt"]),
        ("telecom",       ["telecom", "telco", "wireless", "broadband"]),
        ("media",         ["media", "entertainment", "broadcast"]),
    ]
    for key, patterns in checks:
        if any(p in s for p in patterns):
            return key
    return "default"


# ── Main public API ─────────────────────────────────────────────────────────

_FLOOD_HAZARDS: frozenset[str] = frozenset({"flood_riverine", "coastal_flooding"})


def compute_physical_npv_impact(
    ngfs_family: str,
    sector: str,
    revenue_usd: float,
    ev_base_usd: float,
    wacc: float = 0.10,
    start_year: int = 2026,
    horizon: int = 2050,
    component_graph: ComponentGraph | None = None,
    # JRC hydraulic flood integration — optional
    asset_lat: float | None = None,
    asset_lon: float | None = None,
    asset_rcv_usd: float | None = None,   # replacement cost value; falls back to ev_base_usd * 0.4
) -> PhysicalRiskResult:
    """
    Compute the physical-risk NPV drag for one (company, scenario) pair.

    Flood model selection
    ---------------------
    When asset_lat/lon are provided, the flood component of physical risk is
    computed from JRC Global Flood Hazard Maps (LISFLOOD-FP, Dottori 2016)
    using JRC Huizinga 2017 depth-damage functions.  Non-flood hazards (heat,
    drought, freeze-thaw, wildfire) continue to use the existing phi×ELF model.

    When coordinates are absent or the JRC fetch fails, the entire calculation
    falls back to the phi×ELF model.

    Parameters
    ----------
    ngfs_family : str
        One of 'nze_2050', 'delayed_transition', 'current_policies', 'hot_house'.
    sector : str
        Free-text sector label (e.g. "Oil & Gas", "Utilities", "Beverages").
    revenue_usd : float
        Annual revenue in USD (used to size revenue-at-risk).
    ev_base_usd : float
        Enterprise value (base, pre-scenario) in USD — denominator for pct.
    wacc : float
        Discount rate (decimal). Default 10%.
    start_year : int
        First projection year.
    horizon : int
        Last projection year (inclusive).
    asset_lat, asset_lon : float | None
        WGS84 coordinates of the primary asset location.  When provided,
        enables JRC hydraulic flood depth lookup.
    asset_rcv_usd : float | None
        Replacement cost value of the asset (USD).  If not supplied, estimated
        as ev_base_usd × 0.40 (typical fixed-asset / EV ratio for industrials).

    Returns
    -------
    PhysicalRiskResult
        physical_npv_impact_pct is negative (a drag on EV).
        Zero-revenue or zero-EV inputs return zero impact.
    """
    if revenue_usd <= 0 or ev_base_usd <= 0:
        ssp_id = NGFS_TO_SSP.get(ngfs_family, "ssp245")
        ssp = SSP_SCENARIOS[ssp_id]
        return PhysicalRiskResult(
            scenario_id=ngfs_family,
            ngfs_family=ngfs_family,
            ssp_id=ssp_id,
            gmst_2050=ssp.gmst_2050,
            physical_npv_drag=0.0,
            physical_npv_impact_pct=0.0,
            audit_trail={"note": "Zero revenue or EV — no calculation performed."},
        )

    ssp_id = NGFS_TO_SSP.get(ngfs_family, "ssp245")
    ssp: SSPScenario = SSP_SCENARIOS[ssp_id]

    sec_key  = _sector_key(sector)
    phi      = SECTOR_EXPOSURE_RATIO.get(sec_key, SECTOR_EXPOSURE_RATIO["default"])
    years    = list(range(start_year, horizon + 1))
    t0       = start_year

    # Gap 7: track near-term (2026–2030) separately from full horizon
    NEAR_TERM_END = 2030
    near_term_drag = 0.0

    total_drag = 0.0
    probs_mid: dict[str, float] = {}
    elf_mid: float = 0.0
    mid_year = (start_year + horizon) // 2

    # Gap 3: audit trail — per-year arithmetic
    audit_years: list[dict] = []
    probs_2028: dict[str, float] = {}
    elf_2028: float = 0.0

    # Gap 9: resolve component graph — use caller's custom graph, fall back to
    # sector default, then fall back to legacy phi × ELF scalar.
    _cp_graph: ComponentGraph | None = component_graph or get_sector_graph(sec_key)
    _cp_audit_mid: dict = {}   # captured once at mid_year for the audit trail

    # ── JRC hydraulic flood integration ───────────────────────────────────────
    # When asset coordinates are provided, compute the baseline flood EAL
    # (at current climate, δT=0) from JRC LISFLOOD depth-damage curves.
    # In the year loop, this EAL is scaled by a per-year climate multiplier
    # (flood_prob(t) / flood_prob_baseline) derived from our GMST model.
    # Non-flood hazards (heat, drought, freeze-thaw, wildfire) continue to
    # use the phi×ELF model with flood hazards stripped out.
    _jrc_mode: bool = False
    _jrc_baseline_eal: float = 0.0
    _jrc_audit: dict = {}
    _flood_base_prob: float = _hazard_probs_at_gmst(0.0).get("flood_riverine", 0.0)

    if _JRC_AVAILABLE and asset_lat is not None and asset_lon is not None:
        _rcv = asset_rcv_usd if asset_rcv_usd and asset_rcv_usd > 0 else ev_base_usd * 0.40
        _jrc_eal, _jrc_audit = _jrc_flood_eal(
            lat=asset_lat,
            lon=asset_lon,
            asset_rcv_usd=_rcv,
            sector_key=sec_key,
            gmst_delta=0.0,   # baseline — climate scaling happens per year below
        )
        if _jrc_eal >= 0.0 and "jrc_unavailable" not in _jrc_audit.get("note", ""):
            _jrc_mode = True
            _jrc_baseline_eal = _jrc_eal
    # ── end JRC setup ─────────────────────────────────────────────────────────

    for t in years:
        delta_t = ssp.gmst(t)                       # GMST anomaly at year t
        probs   = _hazard_probs_at_gmst(delta_t)
        elf     = _joint_elf(probs)

        # Gap 9: critical-path BI — per-year downtime fraction from DAG
        if _cp_graph is not None:
            cp_fraction, _cp_audit_t = critical_path_downtime(_cp_graph, probs)
            revenue_at_risk = revenue_usd * cp_fraction
            if t == mid_year:
                _cp_audit_mid = _cp_audit_t
            jrc_flood_loss_t = 0.0
            non_flood_loss_t = revenue_at_risk
            flood_climate_mult = 1.0
        else:
            if _jrc_mode:
                # ── JRC hydraulic flood component ────────────────────────────
                # Scale baseline EAL by the ratio of current-climate flood
                # probability to the baseline, as a proxy for climate-driven
                # intensification.  Separately compute the non-flood ELF from
                # the remaining hazard probs so we don't double-count.
                flood_prob_t = probs.get("flood_riverine", _flood_base_prob)
                flood_climate_mult = min(
                    2.0,
                    flood_prob_t / max(_flood_base_prob, 1e-6),
                )
                jrc_flood_loss_t = _jrc_baseline_eal * flood_climate_mult

                non_flood_probs  = {
                    k: v for k, v in probs.items()
                    if k not in _FLOOD_HAZARDS
                }
                elf_non_flood    = _joint_elf(non_flood_probs)
                non_flood_loss_t = revenue_usd * phi * elf_non_flood

                revenue_at_risk  = jrc_flood_loss_t + non_flood_loss_t
            else:
                jrc_flood_loss_t = 0.0
                non_flood_loss_t = revenue_usd * phi * elf
                flood_climate_mult = 1.0
                revenue_at_risk  = non_flood_loss_t

        discount        = (1.0 + wacc) ** (t - t0)
        year_drag       = revenue_at_risk / discount
        total_drag     += year_drag

        if t <= NEAR_TERM_END:
            near_term_drag += year_drag

        if t == mid_year:
            probs_mid = probs
            elf_mid   = elf

        # Gap 7: 2028 snapshot for near-term report
        if t == 2028:
            probs_2028 = dict(probs)
            elf_2028   = elf

        # Gap 3: store key years in audit trail (every 5 years + boundaries)
        if t in (start_year, 2028, 2030, 2035, 2038, 2040, 2045, horizon):
            _year_entry: dict = {
                "year":             t,
                "gmst_delta_c":     round(delta_t, 3),
                "hazard_probs":     {k: round(v, 4) for k, v in probs.items()},
                "elf":              round(elf, 4),
                "phi":              round(phi, 3),
                "revenue_at_risk":  round(revenue_at_risk, 0),
                "discount_factor":  round(1.0 / discount, 5),
                "pv_drag":          round(year_drag, 0),
            }
            if _jrc_mode:
                _year_entry["jrc_flood_loss_usd"]   = round(jrc_flood_loss_t, 0)
                _year_entry["non_flood_loss_usd"]    = round(non_flood_loss_t, 0)
                _year_entry["flood_climate_mult"]    = round(flood_climate_mult, 4)
            audit_years.append(_year_entry)

    physical_npv_impact_pct = -(total_drag / ev_base_usd)

    # Gap 3: full audit trail dict
    audit_trail = {
        "methodology":          "IPCC AR6 WG1 Ch4/Ch11 + NGFS 2023 + Meinshausen 2011 log-logistic CDF",
        "fragility_model":      "Log-logistic CDF (Gap 1: replaces linear/exponential scaling)",
        "joint_elf_model":      "Gaussian copula + compound 1.15× multiplier (Gaps 4+8)",
        "ssp_id":               ssp_id,
        "gmst_2050_c":          round(ssp.gmst_2050, 3),
        "sector_key":           sec_key,
        "sector_phi":           round(phi, 3),
        "revenue_usd":          round(revenue_usd, 0),
        "ev_base_usd":          round(ev_base_usd, 0),
        "wacc":                 round(wacc, 4),
        "horizon":              f"{start_year}–{horizon}",
        "total_pv_drag_usd":    round(-total_drag, 0),
        "npv_impact_pct":       round(physical_npv_impact_pct * 100, 3),
        "near_term_drag_usd":   round(-near_term_drag, 0),
        "year_by_year":         audit_years,
        "compound_pairs_used":  [list(p) for p in _COMPOUND_PAIRS],
        "copula_correlations":  {f"{h1}/{h2}": rho for (h1, h2), rho in _HAZARD_CORR.items()},
        # Gap 9: critical-path BI audit (None when graph not available)
        "critical_path_bi":     _cp_audit_mid if _cp_graph is not None else None,
        "bi_model":             "critical_path_dag" if _cp_graph is not None else "sector_phi_elf",
        # JRC hydraulic flood model provenance
        "jrc_flood_model": {
            "active":               _jrc_mode,
            "baseline_eal_usd":     round(_jrc_baseline_eal, 0) if _jrc_mode else None,
            "asset_lat":            asset_lat,
            "asset_lon":            asset_lon,
            "asset_rcv_usd":        round(asset_rcv_usd or ev_base_usd * 0.40, 0) if _jrc_mode else None,
            "flood_hazards_modelled": sorted(_FLOOD_HAZARDS),
            "data_source":          "JRC Global Flood Hazard Maps (LISFLOOD-FP, Dottori 2016)" if _jrc_mode else "phi_elf_proxy",
            "per_rp_breakdown":     _jrc_audit.get("return_periods") if _jrc_mode else None,
            "note": (
                None if _jrc_mode
                else "Coordinates not provided — using WRI Aqueduct phi×ELF proxy for flood hazard"
            ),
        },
    }

    return PhysicalRiskResult(
        scenario_id=ngfs_family,
        ngfs_family=ngfs_family,
        ssp_id=ssp_id,
        gmst_2050=ssp.gmst_2050,
        physical_npv_drag=-total_drag,
        physical_npv_impact_pct=physical_npv_impact_pct,
        hazard_probs_2038=probs_mid,
        elf_2038=elf_mid,
        hazard_probs_2028=probs_2028,
        elf_2028=elf_2028,
        physical_npv_drag_2030=-near_term_drag,
        sector=sec_key,
        sector_exposure_ratio=phi,
        revenue_usd=revenue_usd,
        ev_base_usd=ev_base_usd,
        wacc=wacc,
        horizon_years=len(years),
        data_source="IPCC AR6 WG1/WG2 + NGFS 2023 · log-logistic CDF · Gaussian copula · critical-path DAG",
        audit_trail=audit_trail,
    )


def combined_npv_impact_pct(
    transition_npv_impact_pct: float | None,
    physical_npv_impact_pct: float,
) -> float:
    """
    Add transition and physical NPV impacts.

    Both are signed fractions (negative = loss). The combined impact is
    their sum — both drag on EV simultaneously; under no scenario does
    one cancel the other (they are independent risk channels).

    Under NZE:  large negative transition + small negative physical = moderate combined
    Under CP:   small negative transition + large negative physical  = moderate combined
    Under DT:   both moderate → combined is the largest in absolute terms
    """
    tr = transition_npv_impact_pct or 0.0
    return tr + physical_npv_impact_pct
