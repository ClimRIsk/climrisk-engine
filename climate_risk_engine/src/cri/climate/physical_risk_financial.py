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

# IPCC AR6 scaling constants (per °C GMST)
_FLOOD_PRECIP_PCT_PER_DEG   = 0.07    # Ch11.4 Clausius-Clapeyron 7%/°C → ²
_HEAT_FREQ_PER_DEG          = 1.40    # Ch11.3 — 40% more frequent/°C
_WATER_STRESS_PCT_PER_DEG   = 0.15    # Ch4 — aridification 15%/°C
_WILDFIRE_IDX_PCT_PER_DEG   = 0.12    # Ch12 / Abatzoglou 2019
_CYCLONE_INTENSITY_PCT_PER_DEG = 0.05 # Ch11.7 Knutson et al. 2020

# ELF hard cap (avoids absurdly large numbers for high-GMST scenarios)
_ELF_CAP: float = 0.80


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

    # ── Metadata ─────────────────────────────────────────────────────────
    sector: str = "default"
    sector_exposure_ratio: float = 0.22
    revenue_usd: float = 0.0
    ev_base_usd: float = 0.0
    wacc: float = 0.10
    horizon_years: int = 25
    data_source: str = "IPCC AR6 WG1/WG2 + NGFS 2023"


# ── Core computation functions ──────────────────────────────────────────────

def _hazard_probs_at_gmst(delta_t: float) -> dict[str, float]:
    """
    Return annual hazard probabilities at a given GMST anomaly (°C).

    All probabilities are capped at 0.90 to avoid nonsensical values in
    extreme warming scenarios.
    """
    p_flood = _BASE_PROBS["flood"] * (1 + _FLOOD_PRECIP_PCT_PER_DEG * delta_t) ** 2
    p_heat  = _BASE_PROBS["heat_stress"] * (_HEAT_FREQ_PER_DEG ** delta_t)
    p_water = _BASE_PROBS["water_stress"] * (1 + _WATER_STRESS_PCT_PER_DEG * delta_t)
    p_fire  = _BASE_PROBS["wildfire"] * (1 + _WILDFIRE_IDX_PCT_PER_DEG * delta_t)
    p_wind  = _BASE_PROBS["wind"] * (1 + _CYCLONE_INTENSITY_PCT_PER_DEG * delta_t)

    return {
        "flood":        min(0.90, p_flood),
        "heat_stress":  min(0.90, p_heat),
        "water_stress": min(0.90, p_water),
        "wildfire":     min(0.90, p_fire),
        "wind":         min(0.90, p_wind),
    }


def _joint_elf(hazard_probs: dict[str, float]) -> float:
    """Joint expected-loss fraction via independent-hazard survival rule."""
    survival = 1.0
    for p in hazard_probs.values():
        survival *= (1.0 - p)
    return min(_ELF_CAP, 1.0 - survival)


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

def compute_physical_npv_impact(
    ngfs_family: str,
    sector: str,
    revenue_usd: float,
    ev_base_usd: float,
    wacc: float = 0.10,
    start_year: int = 2026,
    horizon: int = 2050,
) -> PhysicalRiskResult:
    """
    Compute the physical-risk NPV drag for one (company, scenario) pair.

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
        )

    ssp_id = NGFS_TO_SSP.get(ngfs_family, "ssp245")
    ssp: SSPScenario = SSP_SCENARIOS[ssp_id]

    sec_key  = _sector_key(sector)
    phi      = SECTOR_EXPOSURE_RATIO.get(sec_key, SECTOR_EXPOSURE_RATIO["default"])
    years    = list(range(start_year, horizon + 1))
    t0       = start_year

    total_drag = 0.0
    probs_mid: dict[str, float] = {}
    elf_mid: float = 0.0
    mid_year = (start_year + horizon) // 2

    for t in years:
        delta_t = ssp.gmst(t)                       # GMST anomaly at year t
        probs   = _hazard_probs_at_gmst(delta_t)
        elf     = _joint_elf(probs)

        revenue_at_risk = revenue_usd * phi * elf
        discount        = (1.0 + wacc) ** (t - t0)
        total_drag     += revenue_at_risk / discount

        if t == mid_year:
            probs_mid = probs
            elf_mid   = elf

    physical_npv_impact_pct = -(total_drag / ev_base_usd)

    return PhysicalRiskResult(
        scenario_id=ngfs_family,
        ngfs_family=ngfs_family,
        ssp_id=ssp_id,
        gmst_2050=ssp.gmst_2050,
        physical_npv_drag=-total_drag,
        physical_npv_impact_pct=physical_npv_impact_pct,
        hazard_probs_2038=probs_mid,
        elf_2038=elf_mid,
        sector=sec_key,
        sector_exposure_ratio=phi,
        revenue_usd=revenue_usd,
        ev_base_usd=ev_base_usd,
        wacc=wacc,
        horizon_years=len(years),
        data_source="IPCC AR6 WG1/WG2 + NGFS 2023",
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
