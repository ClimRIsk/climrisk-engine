"""
IPCC AR6 SSP Scenario Parameters.

Four Shared Socioeconomic Pathways (SSPs) from IPCC AR6 WG1 (2021):

  SSP1-2.6  — Sustainability / low emissions (~1.5–2.0°C by 2100)
  SSP2-4.5  — Middle of the road (~2.1–3.5°C by 2100)
  SSP3-7.0  — Regional rivalry / high emissions (~3.3–5.7°C by 2100)
  SSP5-8.5  — Fossil-fuelled development (~3.3–5.7°C+ by 2100)

Each SSP drives:
  - Global mean surface temperature (GMST) anomaly vs 1995–2014 baseline
  - Regional warming multipliers (polar amplification, etc.)
  - Sea level rise projections (IPCC AR6 Ch9, likely range medians)
  - Precipitation change (% per °C of GMST)
  - Extreme event frequency multipliers
  - Cyclone intensity change
  - Wildfire weather index change

Cross-reference with NGFS scenarios:
  NZE 2050          ≈  SSP1-2.6 (1.5°C compatible)
  Delayed Transition ≈  SSP2-4.5 (2°C+ by 2050, ~2.5°C by 2100)
  Current Policies   ≈  SSP3-7.0 / SSP5-8.5 (3–4°C by 2100)

Data sources:
  - IPCC AR6 WG1 Chapter 4 (GMST projections)
  - IPCC AR6 WG1 Chapter 9 (Sea level rise)
  - IPCC AR6 WG1 Chapter 11 (Extreme events)
  - IPCC AR6 WG2 Chapter 4 (Water cycle)
  - NASA NEX-GDDP-CMIP6 ensemble means
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict


# ---------------------------------------------------------------------------
# Global mean surface temperature anomaly (°C vs 1995–2014)
# Source: IPCC AR6 WG1 Table 4.5 — likely range medians
# ---------------------------------------------------------------------------

_GMST_ANOMALY: dict[str, dict[int, float]] = {
    "ssp126": {  # SSP1-2.6
        2026: 0.40, 2030: 0.55, 2035: 0.65, 2040: 0.75,
        2045: 0.80, 2050: 0.85, 2060: 0.90, 2070: 0.90,
        2080: 0.85, 2090: 0.80, 2100: 0.75,
    },
    "ssp245": {  # SSP2-4.5
        2026: 0.42, 2030: 0.60, 2035: 0.80, 2040: 1.00,
        2045: 1.20, 2050: 1.35, 2060: 1.65, 2070: 1.95,
        2080: 2.10, 2090: 2.25, 2100: 2.35,
    },
    "ssp370": {  # SSP3-7.0
        2026: 0.43, 2030: 0.65, 2035: 0.90, 2040: 1.15,
        2045: 1.45, 2050: 1.70, 2060: 2.20, 2070: 2.70,
        2080: 3.10, 2090: 3.50, 2100: 3.80,
    },
    "ssp585": {  # SSP5-8.5
        2026: 0.45, 2030: 0.70, 2035: 1.00, 2040: 1.30,
        2045: 1.65, 2050: 2.00, 2060: 2.65, 2070: 3.35,
        2080: 3.90, 2090: 4.40, 2100: 4.80,
    },
}

# NGFS → SSP mapping
NGFS_TO_SSP: dict[str, str] = {
    "nze_2050":            "ssp126",
    "below_2c":            "ssp126",
    "delayed_transition":  "ssp245",
    "current_policies":    "ssp370",
    "hot_house":           "ssp585",
}


# ---------------------------------------------------------------------------
# Regional warming amplification factors vs global mean
# Source: IPCC AR6 WG1 Chapter 4, Atlas
# Higher values = region warms faster than global mean
# ---------------------------------------------------------------------------

REGIONAL_AMPLIFICATION: dict[str, float] = {
    # Arctic / boreal — polar amplification
    "CA-QC": 1.8, "CA-AB": 1.6,
    # Arid / semi-arid zones
    "AU-WA": 1.15, "AU-SA": 1.20, "AU-QLD": 1.10,
    "ZA": 1.25, "CL-02": 1.30, "MN-01": 1.55,
    "CN-NM": 1.50, "IN-MH": 1.10,
    # Tropical
    "ID-KI": 0.85, "BR-PA": 0.90,
    # Temperate
    "GB-ENG": 0.90, "NL-NH": 0.95,
    "US-TX": 1.10, "US-WY": 1.20, "US-OK": 1.10,
    "AU-NSW": 1.05, "AU-VIC": 1.00,
    "PE-01": 1.05,
    "global": 1.00,
}


# ---------------------------------------------------------------------------
# Sea level rise projections — median (m above 1995–2014 baseline)
# Source: IPCC AR6 WG1 Chapter 9, Table 9.9
# Note: these are GLOBAL MEAN; local values differ due to subsidence,
#       ocean dynamics — we apply a coastal amplification factor separately.
# ---------------------------------------------------------------------------

_SLR_GLOBAL: dict[str, dict[int, float]] = {
    "ssp126": {
        2026: 0.06, 2030: 0.09, 2035: 0.12, 2040: 0.16,
        2045: 0.20, 2050: 0.24, 2060: 0.32, 2070: 0.40,
        2080: 0.47, 2090: 0.53, 2100: 0.56,
    },
    "ssp245": {
        2026: 0.07, 2030: 0.10, 2035: 0.14, 2040: 0.19,
        2045: 0.24, 2050: 0.29, 2060: 0.40, 2070: 0.52,
        2080: 0.62, 2090: 0.70, 2100: 0.76,
    },
    "ssp370": {
        2026: 0.07, 2030: 0.11, 2035: 0.15, 2040: 0.21,
        2045: 0.27, 2050: 0.33, 2060: 0.47, 2070: 0.62,
        2080: 0.76, 2090: 0.87, 2100: 0.95,
    },
    "ssp585": {
        2026: 0.08, 2030: 0.12, 2035: 0.17, 2040: 0.23,
        2045: 0.30, 2050: 0.38, 2060: 0.55, 2070: 0.73,
        2080: 0.91, 2090: 1.07, 2100: 1.20,
    },
}

# Coastal subsidence amplification by region (multiplier on global SLR)
COASTAL_SLR_AMPLIFICATION: dict[str, float] = {
    # Deltas and low-lying coasts with high subsidence
    "ID-KI": 2.5,   # Jakarta area — severe subsidence
    "IN-MH": 1.8,   # Mumbai — moderate subsidence
    "NL-NH": 1.5,   # Netherlands — below sea level
    "BR-PA": 1.3,   # Amazon delta
    # Normal coasts
    "AU-WA": 1.0, "AU-QLD": 1.0, "AU-NSW": 1.0,
    "GB-ENG": 1.1,
    "US-TX": 1.4,   # Gulf Coast subsidence
    "ZA": 1.0, "global": 1.0,
}


# ---------------------------------------------------------------------------
# Extreme heat frequency multipliers
# Source: IPCC AR6 WG1 Chapter 11.3
# = how many times MORE FREQUENT a "1-in-50-year" heat extreme becomes
#   per 1°C of global warming
# ---------------------------------------------------------------------------

# At 1°C GMST: factor × baseline frequency of 1-in-50yr extreme heat
HEAT_FREQ_PER_DEG_C: float = 1.40   # 40% more frequent per °C


# ---------------------------------------------------------------------------
# Heavy precipitation change
# Source: IPCC AR6 WG1 Chapter 11.4 — Clausius-Clapeyron scaling
# ---------------------------------------------------------------------------

PRECIP_CHANGE_PCT_PER_DEG_C: float = 7.0   # ~7% increase in heavy precip intensity per °C


# ---------------------------------------------------------------------------
# Wildfire weather index change
# Source: IPCC AR6 WG2 Chapter 12; Abatzoglou et al. 2019
# ---------------------------------------------------------------------------

WILDFIRE_INDEX_PCT_PER_DEG_C: float = 12.0  # ~12% increase in fire weather index per °C


# ---------------------------------------------------------------------------
# Tropical cyclone intensity change
# Source: IPCC AR6 WG1 Chapter 11.7; Knutson et al. 2020
# ---------------------------------------------------------------------------

CYCLONE_INTENSITY_PCT_PER_DEG_C: float = 5.0   # +5% peak wind speed per °C
CYCLONE_FREQ_CHANGE_PCT_PER_DEG_C: float = -5.0 # fewer but more intense (net risk up)


# ---------------------------------------------------------------------------
# SSP scenario metadata
# ---------------------------------------------------------------------------

@dataclass
class SSPScenario:
    id: str
    name: str
    description: str
    gmst_2050: float        # °C above 1995–2014 median
    gmst_2100: float
    slr_2050_m: float       # global mean SLR median
    slr_2100_m: float
    ngfs_equivalent: str
    radiative_forcing: str  # W/m² by 2100

    def gmst(self, year: int) -> float:
        """Linear interpolation of GMST anomaly for a given year."""
        tbl = _GMST_ANOMALY[self.id]
        yrs = sorted(tbl.keys())
        if year <= yrs[0]: return tbl[yrs[0]]
        if year >= yrs[-1]: return tbl[yrs[-1]]
        for i in range(len(yrs) - 1):
            if yrs[i] <= year <= yrs[i+1]:
                t = (year - yrs[i]) / (yrs[i+1] - yrs[i])
                return tbl[yrs[i]] + t * (tbl[yrs[i+1]] - tbl[yrs[i]])
        return tbl[yrs[-1]]

    def slr(self, year: int, region: str = "global") -> float:
        """Sea level rise in metres for a given year and region."""
        tbl = _SLR_GLOBAL[self.id]
        yrs = sorted(tbl.keys())
        if year <= yrs[0]: base = tbl[yrs[0]]
        elif year >= yrs[-1]: base = tbl[yrs[-1]]
        else:
            base = 0.0
            for i in range(len(yrs) - 1):
                if yrs[i] <= year <= yrs[i+1]:
                    t = (year - yrs[i]) / (yrs[i+1] - yrs[i])
                    base = tbl[yrs[i]] + t * (tbl[yrs[i+1]] - tbl[yrs[i]])
                    break
        amp = COASTAL_SLR_AMPLIFICATION.get(region, 1.0)
        return base * amp

    def regional_warming(self, year: int, region: str) -> float:
        """Local warming (°C) for a region at a given year."""
        amp = REGIONAL_AMPLIFICATION.get(region, 1.0)
        return self.gmst(year) * amp


SSP_SCENARIOS: dict[str, SSPScenario] = {
    "ssp126": SSPScenario(
        id="ssp126",
        name="SSP1-2.6",
        description="Sustainability — low challenges to mitigation and adaptation. "
                    "Strong decarbonisation; likely below 2°C by 2100.",
        gmst_2050=0.85,
        gmst_2100=0.75,
        slr_2050_m=0.24,
        slr_2100_m=0.56,
        ngfs_equivalent="NZE 2050",
        radiative_forcing="~2.6 W/m²",
    ),
    "ssp245": SSPScenario(
        id="ssp245",
        name="SSP2-4.5",
        description="Middle of the road — intermediate challenges to mitigation. "
                    "Broadly current trajectory if some NDCs are met; ~2.7°C by 2100.",
        gmst_2050=1.35,
        gmst_2100=2.35,
        slr_2050_m=0.29,
        slr_2100_m=0.76,
        ngfs_equivalent="Delayed Transition",
        radiative_forcing="~4.5 W/m²",
    ),
    "ssp370": SSPScenario(
        id="ssp370",
        name="SSP3-7.0",
        description="Regional rivalry — high challenges to mitigation. "
                    "Fragmented world; limited climate policy; ~3.8°C by 2100.",
        gmst_2050=1.70,
        gmst_2100=3.80,
        slr_2050_m=0.33,
        slr_2100_m=0.95,
        ngfs_equivalent="Current Policies",
        radiative_forcing="~7.0 W/m²",
    ),
    "ssp585": SSPScenario(
        id="ssp585",
        name="SSP5-8.5",
        description="Fossil-fuelled development — high challenges, worst-case pathway. "
                    "Very high emissions; ~4.8°C by 2100 (upper tail).",
        gmst_2050=2.00,
        gmst_2100=4.80,
        slr_2050_m=0.38,
        slr_2100_m=1.20,
        ngfs_equivalent="Hot House World",
        radiative_forcing="~8.5 W/m²",
    ),
}


def get_ssp(ssp_id: str) -> SSPScenario:
    """Return an SSP scenario by id. Raises KeyError if not found."""
    if ssp_id not in SSP_SCENARIOS:
        raise KeyError(f"Unknown SSP: {ssp_id!r}. Valid: {list(SSP_SCENARIOS)}")
    return SSP_SCENARIOS[ssp_id]


def ngfs_to_ssp(ngfs_family: str) -> SSPScenario:
    """Map an NGFS scenario family string to its closest SSP."""
    ssp_id = NGFS_TO_SSP.get(ngfs_family, "ssp245")
    return SSP_SCENARIOS[ssp_id]
