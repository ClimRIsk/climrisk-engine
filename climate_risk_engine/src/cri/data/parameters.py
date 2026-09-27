"""CRI Parametric Data Library — v2.0.0
═══════════════════════════════════════════════════════════════════════════════
Self-contained, zero-dependency parameter tables for physical and transition
climate risk scoring.  All data is embedded in-module; no external API calls
or network access required at runtime.

Data provenance
───────────────
Physical hazard profiles:
  • IPCC AR6 Working Group I, Chapter 12 (Climate Change Information for
    Regional Impact and for Risk Assessment), Table 12.1 and Figure 12.4.
    DOI: 10.1017/9781009157896.014

  • IPCC AR6 Synthesis Report (2023): Summary for Policymakers.
    Table SPM.1 — Reasons for Concern, regional hazard confidence levels.

  • WRI Aqueduct 4.0 Baseline (2023): Water stress, riverine flood risk.
    https://www.wri.org/data/aqueduct-global-maps-40-data

Transition risk parameters:
  • NGFS Phase 4 Climate Scenarios (2023): Carbon price pathways,
    Net Zero 2050 / Below 2°C Orderly / Delayed Transition / Current Policies.
    https://www.ngfs.net/ngfs-scenarios-portal

  • IEA World Energy Outlook 2023: Sector emission intensities,
    technology roadmaps.  https://www.iea.org/reports/world-energy-outlook-2023

  • European Systemic Risk Board (ESRB) climate risk taxonomy (2021):
    Sector-level stranded asset risk classifications.

Financial translation:
  • Network for Greening the Financial System (NGFS) Reference Guide (2023):
    Climate-to-financial risk transmission pathways.

Usage
─────
    from cri.data.parameters import (
        country_to_climate_zone,
        get_hazard_profile,
        SECTOR_PHYSICAL_SENSITIVITY,
        SECTOR_TRANSITION_PARAMS,
        SECTOR_EMISSION_INTENSITIES,
        SECTOR_REPRESENTATIVE_ASSETS,
        generate_hazard_paths,
    )
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

# ════════════════════════════════════════════════════════════════════════════
# 1.  CLIMATE ZONE TAXONOMY
#     Seven canonical zones drawn from IPCC AR6 regional typology.
# ════════════════════════════════════════════════════════════════════════════

#: Zone identifiers used throughout this module and in Scenario.hazards
CLIMATE_ZONES = [
    "tropical_humid",      # South/SE Asia, West Africa, Central America, Amazon
    "arid_hot",            # Middle East, N Africa, Atacama, arid Sub-Saharan
    "temperate_maritime",  # NW Europe, Pacific NW, NZ, SE Australia
    "temperate_continental",  # Central/Eastern Europe, Japan, Korea, Eastern US
    "continental_cold",    # Russia, Canada, NW China, Northern US/Canada
    "mediterranean",       # S Europe, Turkey, California, N Africa coast, SW AU
    "coastal_delta",       # Bangladesh, Mekong, Rhine, Nile delta, Pacific atolls
]

# ── Country → primary climate zone ──────────────────────────────────────────
# Source: Beck et al. (2018) Köppen–Geiger classification aggregated by IPCC
# AR6 regional clusters.  Where a country spans multiple zones the primary
# economic zone is chosen (e.g. AU → temperate_maritime for major cities,
# not outback which is arid).
COUNTRY_CLIMATE_ZONE: Dict[str, str] = {
    # ── tropical_humid ───────────────────────────────────────────────────────
    "IN": "tropical_humid",   "BD": "coastal_delta",   "TH": "tropical_humid",
    "PH": "tropical_humid",   "MY": "tropical_humid",  "ID": "tropical_humid",
    "VN": "coastal_delta",    "LK": "tropical_humid",  "MM": "tropical_humid",
    "KH": "tropical_humid",   "LA": "tropical_humid",  "SG": "tropical_humid",
    "NG": "tropical_humid",   "GH": "tropical_humid",  "CI": "tropical_humid",
    "CM": "tropical_humid",   "ET": "tropical_humid",  "KE": "tropical_humid",
    "TZ": "tropical_humid",   "MZ": "tropical_humid",  "UG": "tropical_humid",
    "CO": "tropical_humid",   "VE": "tropical_humid",  "EC": "tropical_humid",
    "PE": "tropical_humid",   "GT": "tropical_humid",  "HN": "tropical_humid",
    "NI": "tropical_humid",   "CR": "tropical_humid",  "PA": "tropical_humid",
    "BR": "tropical_humid",   "MX": "tropical_humid",  "CU": "tropical_humid",
    "DO": "tropical_humid",   "HT": "tropical_humid",  "JM": "tropical_humid",
    "TW": "tropical_humid",   "HK": "tropical_humid",

    # ── arid_hot ─────────────────────────────────────────────────────────────
    "SA": "arid_hot",   "AE": "arid_hot",   "QA": "arid_hot",   "KW": "arid_hot",
    "OM": "arid_hot",   "BH": "arid_hot",   "YE": "arid_hot",   "DZ": "arid_hot",
    "LY": "arid_hot",   "EG": "arid_hot",   "SD": "arid_hot",   "ML": "arid_hot",
    "NE": "arid_hot",   "TD": "arid_hot",   "SO": "arid_hot",   "IQ": "arid_hot",
    "IR": "arid_hot",   "PK": "arid_hot",   "AF": "arid_hot",   "SY": "arid_hot",
    "JO": "arid_hot",   "LB": "arid_hot",   "MA": "arid_hot",   "TN": "arid_hot",
    "NA": "arid_hot",   "BW": "arid_hot",   "ZW": "arid_hot",   "ZM": "arid_hot",
    "CL": "arid_hot",   "UZ": "arid_hot",   "TM": "arid_hot",   "MN": "arid_hot",

    # ── temperate_maritime ───────────────────────────────────────────────────
    "GB": "temperate_maritime", "IE": "temperate_maritime",
    "NL": "coastal_delta",      "BE": "temperate_maritime",
    "DK": "temperate_maritime", "NO": "temperate_maritime",
    "SE": "temperate_maritime", "FI": "continental_cold",
    "AU": "temperate_maritime", "NZ": "temperate_maritime",
    "CL_south": "temperate_maritime", "AR": "temperate_maritime",

    # ── temperate_continental ────────────────────────────────────────────────
    "DE": "temperate_continental",  "FR": "temperate_continental",
    "AT": "temperate_continental",  "CH": "temperate_continental",
    "US": "temperate_continental",  "JP": "temperate_continental",
    "KR": "temperate_continental",  "CZ": "temperate_continental",
    "SK": "temperate_continental",  "HR": "temperate_continental",
    "SI": "temperate_continental",  "LU": "temperate_continental",
    "HU": "temperate_continental",  "CN": "temperate_continental",

    # ── continental_cold ─────────────────────────────────────────────────────
    "RU": "continental_cold",  "CA": "continental_cold",
    "PL": "continental_cold",  "UA": "continental_cold",
    "BY": "continental_cold",  "RO": "continental_cold",
    "BG": "continental_cold",  "KZ": "continental_cold",
    "KG": "continental_cold",  "TJ": "continental_cold",
    "MN_north": "continental_cold",

    # ── mediterranean ────────────────────────────────────────────────────────
    "ES": "mediterranean",  "IT": "mediterranean",  "PT": "mediterranean",
    "GR": "mediterranean",  "TR": "mediterranean",  "IL": "mediterranean",
    "ZA": "mediterranean",  "CY": "mediterranean",  "MT": "mediterranean",

    # ── coastal_delta ────────────────────────────────────────────────────────
    "MV": "coastal_delta",  "FJ": "coastal_delta",  "MH": "coastal_delta",
    "KI": "coastal_delta",  "TV": "coastal_delta",  "PW": "coastal_delta",
    "GY": "coastal_delta",  "SR": "coastal_delta",
}


def country_to_climate_zone(country_iso2: str) -> str:
    """Return the primary climate zone for a 2-letter ISO country code.

    Falls back to 'temperate_continental' for unmapped codes.
    Also handles region codes like 'AU-WA', 'US-CA', etc. by extracting
    the country prefix.
    """
    code = country_iso2.upper().strip()
    if "-" in code:
        code = code.split("-")[0]
    return COUNTRY_CLIMATE_ZONE.get(code, "temperate_continental")


def jurisdiction_to_region_code(jurisdiction: str) -> str:
    """Map a jurisdiction (ISO 2-letter or region code like 'IN-MH') to its
    canonical climate-zone region code for use in HazardPath.region.

    This is the region string that will be used in both the Asset.region field
    and the HazardPath.region field of the generated scenarios, ensuring the
    ScenarioHazardProvider can match them.
    """
    return country_to_climate_zone(jurisdiction)


# ════════════════════════════════════════════════════════════════════════════
# 2.  PHYSICAL HAZARD PROFILES BY CLIMATE ZONE & SCENARIO
#     Anchor-point severity values calibrated to IPCC AR6 Chapter 12.
#     severity ∈ [0, 1] where 0 = no hazard, 1 = catastrophic.
#
#     Five hazard types: heat_stress, water_stress, flood, cyclone, wildfire
#
#     Anchor years: 2025, 2030, 2035, 2040, 2045, 2050
#     Three scenarios: nze (Net Zero 2050), delayed (Delayed Transition),
#                       cp (Current Policies)
#
#     Reference: IPCC AR6 WGI Ch12 Table 12.1; SYR Table SPM.1
# ════════════════════════════════════════════════════════════════════════════

# Format: zone → scenario → hazard → [sev_2025, sev_2030, sev_2035, sev_2040, sev_2045, sev_2050]
_HAZARD_ANCHORS: Dict[str, Dict[str, Dict[str, List[float]]]] = {

    "tropical_humid": {
        # IPCC AR6: Tropical regions face high-confidence heat stress increases
        # even under low-emission pathways.  Monsoon intensification adds to flood risk.
        "nze": {
            "heat_stress":   [0.08, 0.10, 0.12, 0.13, 0.14, 0.14],
            "water_stress":  [0.05, 0.06, 0.07, 0.07, 0.08, 0.08],
            "flood":         [0.10, 0.12, 0.13, 0.14, 0.14, 0.15],
            "cyclone":       [0.06, 0.07, 0.08, 0.08, 0.09, 0.09],
            "wildfire":      [0.02, 0.03, 0.03, 0.03, 0.04, 0.04],
        },
        "delayed": {
            "heat_stress":   [0.08, 0.12, 0.17, 0.22, 0.27, 0.30],
            "water_stress":  [0.05, 0.07, 0.10, 0.13, 0.16, 0.18],
            "flood":         [0.10, 0.13, 0.17, 0.21, 0.24, 0.26],
            "cyclone":       [0.06, 0.08, 0.11, 0.14, 0.17, 0.19],
            "wildfire":      [0.02, 0.03, 0.04, 0.05, 0.06, 0.07],
        },
        "cp": {
            "heat_stress":   [0.08, 0.14, 0.22, 0.31, 0.41, 0.52],
            "water_stress":  [0.05, 0.08, 0.13, 0.19, 0.25, 0.31],
            "flood":         [0.10, 0.15, 0.22, 0.30, 0.38, 0.46],
            "cyclone":       [0.06, 0.09, 0.14, 0.20, 0.27, 0.33],
            "wildfire":      [0.02, 0.04, 0.06, 0.09, 0.12, 0.15],
        },
    },

    "arid_hot": {
        # IPCC AR6: Arid regions face "compound" risk of heat + water scarcity.
        # Very high confidence for both hazard types under all scenarios (Ch12, p.1832).
        # Extreme heat events already occurring near physiological limits.
        "nze": {
            "heat_stress":   [0.15, 0.18, 0.20, 0.22, 0.23, 0.23],
            "water_stress":  [0.20, 0.23, 0.25, 0.26, 0.27, 0.27],
            "flood":         [0.03, 0.04, 0.04, 0.05, 0.05, 0.05],
            "cyclone":       [0.01, 0.01, 0.02, 0.02, 0.02, 0.02],
            "wildfire":      [0.04, 0.05, 0.06, 0.06, 0.07, 0.07],
        },
        "delayed": {
            "heat_stress":   [0.15, 0.21, 0.29, 0.36, 0.43, 0.48],
            "water_stress":  [0.20, 0.26, 0.33, 0.40, 0.46, 0.51],
            "flood":         [0.03, 0.04, 0.06, 0.07, 0.08, 0.09],
            "cyclone":       [0.01, 0.02, 0.02, 0.03, 0.03, 0.04],
            "wildfire":      [0.04, 0.06, 0.08, 0.10, 0.12, 0.14],
        },
        "cp": {
            "heat_stress":   [0.15, 0.24, 0.36, 0.48, 0.58, 0.67],
            "water_stress":  [0.20, 0.29, 0.40, 0.51, 0.61, 0.70],
            "flood":         [0.03, 0.05, 0.07, 0.10, 0.13, 0.16],
            "cyclone":       [0.01, 0.02, 0.03, 0.04, 0.05, 0.06],
            "wildfire":      [0.04, 0.07, 0.11, 0.15, 0.19, 0.23],
        },
    },

    "temperate_maritime": {
        # IPCC AR6: Temperate maritime zones see moderate warming,
        # increased winter rainfall, and more extreme precipitation events.
        # Coastal flooding risk rising with sea level.
        "nze": {
            "heat_stress":   [0.04, 0.05, 0.06, 0.07, 0.07, 0.08],
            "water_stress":  [0.03, 0.03, 0.04, 0.04, 0.04, 0.04],
            "flood":         [0.07, 0.08, 0.09, 0.10, 0.10, 0.11],
            "cyclone":       [0.03, 0.03, 0.04, 0.04, 0.04, 0.05],
            "wildfire":      [0.02, 0.02, 0.03, 0.03, 0.03, 0.03],
        },
        "delayed": {
            "heat_stress":   [0.04, 0.06, 0.09, 0.12, 0.14, 0.16],
            "water_stress":  [0.03, 0.04, 0.06, 0.07, 0.09, 0.10],
            "flood":         [0.07, 0.09, 0.12, 0.14, 0.17, 0.19],
            "cyclone":       [0.03, 0.04, 0.06, 0.07, 0.09, 0.10],
            "wildfire":      [0.02, 0.03, 0.04, 0.05, 0.06, 0.07],
        },
        "cp": {
            "heat_stress":   [0.04, 0.07, 0.12, 0.17, 0.22, 0.28],
            "water_stress":  [0.03, 0.05, 0.08, 0.11, 0.14, 0.18],
            "flood":         [0.07, 0.10, 0.14, 0.19, 0.24, 0.29],
            "cyclone":       [0.03, 0.05, 0.07, 0.10, 0.13, 0.16],
            "wildfire":      [0.02, 0.03, 0.05, 0.08, 0.11, 0.14],
        },
    },

    "temperate_continental": {
        # IPCC AR6: Continental areas show stronger warming signal than maritime.
        # Summer heat waves increasing in frequency and intensity.
        # Reduced summer soil moisture (high confidence).
        "nze": {
            "heat_stress":   [0.06, 0.08, 0.09, 0.10, 0.11, 0.11],
            "water_stress":  [0.05, 0.06, 0.07, 0.07, 0.08, 0.08],
            "flood":         [0.06, 0.07, 0.08, 0.09, 0.09, 0.10],
            "cyclone":       [0.01, 0.01, 0.01, 0.01, 0.02, 0.02],
            "wildfire":      [0.03, 0.04, 0.04, 0.05, 0.05, 0.05],
        },
        "delayed": {
            "heat_stress":   [0.06, 0.09, 0.13, 0.18, 0.22, 0.25],
            "water_stress":  [0.05, 0.07, 0.10, 0.13, 0.16, 0.18],
            "flood":         [0.06, 0.08, 0.11, 0.14, 0.17, 0.19],
            "cyclone":       [0.01, 0.01, 0.02, 0.02, 0.03, 0.03],
            "wildfire":      [0.03, 0.05, 0.07, 0.09, 0.11, 0.13],
        },
        "cp": {
            "heat_stress":   [0.06, 0.11, 0.17, 0.25, 0.33, 0.41],
            "water_stress":  [0.05, 0.08, 0.13, 0.19, 0.25, 0.31],
            "flood":         [0.06, 0.09, 0.13, 0.18, 0.24, 0.30],
            "cyclone":       [0.01, 0.02, 0.02, 0.03, 0.04, 0.05],
            "wildfire":      [0.03, 0.06, 0.10, 0.14, 0.19, 0.24],
        },
    },

    "continental_cold": {
        # IPCC AR6: Arctic amplification — these regions warm 2–3× global average.
        # Permafrost thaw, reduced snow cover, but lower heat/cyclone risk.
        # Wildfire risk rising sharply (Canada, Siberia — high confidence).
        "nze": {
            "heat_stress":   [0.03, 0.05, 0.06, 0.07, 0.07, 0.08],
            "water_stress":  [0.02, 0.02, 0.03, 0.03, 0.03, 0.04],
            "flood":         [0.04, 0.05, 0.06, 0.07, 0.07, 0.08],
            "cyclone":       [0.00, 0.00, 0.00, 0.01, 0.01, 0.01],
            "wildfire":      [0.06, 0.08, 0.10, 0.11, 0.12, 0.12],
        },
        "delayed": {
            "heat_stress":   [0.03, 0.06, 0.10, 0.14, 0.17, 0.20],
            "water_stress":  [0.02, 0.03, 0.04, 0.05, 0.06, 0.07],
            "flood":         [0.04, 0.06, 0.09, 0.12, 0.14, 0.16],
            "cyclone":       [0.00, 0.00, 0.01, 0.01, 0.01, 0.02],
            "wildfire":      [0.06, 0.09, 0.14, 0.19, 0.23, 0.26],
        },
        "cp": {
            "heat_stress":   [0.03, 0.07, 0.13, 0.20, 0.28, 0.36],
            "water_stress":  [0.02, 0.04, 0.06, 0.09, 0.12, 0.16],
            "flood":         [0.04, 0.07, 0.11, 0.16, 0.21, 0.27],
            "cyclone":       [0.00, 0.01, 0.01, 0.02, 0.02, 0.03],
            "wildfire":      [0.06, 0.11, 0.18, 0.26, 0.33, 0.40],
        },
    },

    "mediterranean": {
        # IPCC AR6: Mediterranean is a "climate change hot spot" —
        # drying trend, heat waves, and wildfire all high-confidence increasing.
        # Water scarcity already critical in Southern Europe, Turkey, N Africa.
        "nze": {
            "heat_stress":   [0.12, 0.14, 0.16, 0.17, 0.18, 0.18],
            "water_stress":  [0.15, 0.17, 0.18, 0.20, 0.21, 0.21],
            "flood":         [0.04, 0.05, 0.05, 0.06, 0.06, 0.06],
            "cyclone":       [0.01, 0.01, 0.01, 0.01, 0.02, 0.02],
            "wildfire":      [0.10, 0.12, 0.13, 0.14, 0.15, 0.15],
        },
        "delayed": {
            "heat_stress":   [0.12, 0.17, 0.23, 0.29, 0.35, 0.39],
            "water_stress":  [0.15, 0.20, 0.27, 0.33, 0.39, 0.43],
            "flood":         [0.04, 0.05, 0.07, 0.09, 0.10, 0.11],
            "cyclone":       [0.01, 0.01, 0.02, 0.02, 0.03, 0.03],
            "wildfire":      [0.10, 0.14, 0.19, 0.24, 0.29, 0.32],
        },
        "cp": {
            "heat_stress":   [0.12, 0.19, 0.29, 0.39, 0.50, 0.60],
            "water_stress":  [0.15, 0.23, 0.33, 0.44, 0.54, 0.63],
            "flood":         [0.04, 0.06, 0.09, 0.12, 0.15, 0.19],
            "cyclone":       [0.01, 0.02, 0.02, 0.03, 0.04, 0.05],
            "wildfire":      [0.10, 0.16, 0.25, 0.33, 0.42, 0.50],
        },
    },

    "coastal_delta": {
        # IPCC AR6: Coastal low-elevation zones face compound risks:
        # sea level rise + storm surge + riverine flooding.
        # Very high confidence for increasing coastal flooding (Ch12, p.1796).
        # Cyclone intensity increasing; heat and humidity compounding.
        "nze": {
            "heat_stress":   [0.10, 0.12, 0.13, 0.14, 0.15, 0.15],
            "water_stress":  [0.05, 0.06, 0.06, 0.07, 0.07, 0.07],
            "flood":         [0.20, 0.23, 0.26, 0.28, 0.30, 0.31],  # sea level + riverine
            "cyclone":       [0.10, 0.12, 0.13, 0.14, 0.15, 0.15],
            "wildfire":      [0.01, 0.01, 0.01, 0.02, 0.02, 0.02],
        },
        "delayed": {
            "heat_stress":   [0.10, 0.14, 0.19, 0.24, 0.29, 0.33],
            "water_stress":  [0.05, 0.07, 0.09, 0.12, 0.14, 0.16],
            "flood":         [0.20, 0.26, 0.33, 0.40, 0.47, 0.52],
            "cyclone":       [0.10, 0.13, 0.18, 0.23, 0.28, 0.31],
            "wildfire":      [0.01, 0.02, 0.02, 0.03, 0.04, 0.04],
        },
        "cp": {
            "heat_stress":   [0.10, 0.16, 0.24, 0.33, 0.43, 0.53],
            "water_stress":  [0.05, 0.08, 0.12, 0.17, 0.22, 0.28],
            "flood":         [0.20, 0.29, 0.40, 0.52, 0.63, 0.73],
            "cyclone":       [0.10, 0.15, 0.23, 0.31, 0.40, 0.49],
            "wildfire":      [0.01, 0.02, 0.03, 0.04, 0.05, 0.07],
        },
    },
}

_ANCHOR_YEARS = [2025, 2030, 2035, 2040, 2045, 2050]


def _interp_anchors(anchors: List[float], year: int) -> float:
    """Linear interpolation between 6 anchor points for 2025–2050."""
    y_ref = _ANCHOR_YEARS
    if year <= y_ref[0]:
        return anchors[0]
    if year >= y_ref[-1]:
        return anchors[-1]
    for i in range(len(y_ref) - 1):
        if y_ref[i] <= year <= y_ref[i + 1]:
            t = (year - y_ref[i]) / (y_ref[i + 1] - y_ref[i])
            return anchors[i] + t * (anchors[i + 1] - anchors[i])
    return anchors[-1]


def get_hazard_profile(
    climate_zone: str,
    scenario: str,
    hazard: str,
    years: Optional[List[int]] = None,
) -> Dict[int, float]:
    """Return a {year: severity} dict for the given zone/scenario/hazard.

    Args:
        climate_zone: One of the CLIMATE_ZONES identifiers.
        scenario:     'nze' | 'delayed' | 'cp'
        hazard:       'heat_stress' | 'water_stress' | 'flood' | 'cyclone' | 'wildfire'
        years:        Years to include. Defaults to 2026–2050.

    Returns:
        Dict mapping year → severity (0–1).
    """
    if years is None:
        years = list(range(2026, 2051))
    zone_data = _HAZARD_ANCHORS.get(climate_zone, _HAZARD_ANCHORS["temperate_continental"])
    scen_data = zone_data.get(scenario, zone_data.get("delayed", {}))
    anchors = scen_data.get(hazard, [0.05] * 6)
    return {yr: round(_interp_anchors(anchors, yr), 4) for yr in years}


def generate_hazard_paths(
    climate_zone: str,
    region_code: str,
    scenario: str,
) -> List[Tuple[str, str, Dict[int, float]]]:
    """Generate all hazard paths for a region under a given scenario.

    Returns a list of (hazard_name, region_code, path_dict) tuples,
    ready for conversion to HazardPath objects.
    """
    hazards = ["heat_stress", "water_stress", "flood", "cyclone", "wildfire"]
    result = []
    for h in hazards:
        path = get_hazard_profile(climate_zone, scenario, h)
        result.append((h, region_code, path))
    return result


# ════════════════════════════════════════════════════════════════════════════
# 3.  SECTOR PHYSICAL SENSITIVITY
#     How much each sector's operations are disrupted by each hazard type.
#     Calibrated from CLIMADA sectoral impact studies and NGFS financial
#     sector vulnerability analysis (NGFS Phase 4, 2023, Annex 3).
#     sensitivity ∈ [0, 1]; higher = more disrupted per unit of hazard severity.
# ════════════════════════════════════════════════════════════════════════════

SECTOR_PHYSICAL_SENSITIVITY: Dict[str, Dict[str, float]] = {
    # Energy / Resources
    "oil_gas":        {"heat_stress": 0.12, "water_stress": 0.18, "flood": 0.20, "cyclone": 0.25, "wildfire": 0.08},
    "coal":           {"heat_stress": 0.10, "water_stress": 0.22, "flood": 0.18, "cyclone": 0.15, "wildfire": 0.12},
    "utilities":      {"heat_stress": 0.15, "water_stress": 0.25, "flood": 0.22, "cyclone": 0.20, "wildfire": 0.14},
    "renewable_energy":{"heat_stress": 0.06, "water_stress": 0.12, "flood": 0.16, "cyclone": 0.18, "wildfire": 0.08},

    # Heavy Industry
    "steel":          {"heat_stress": 0.14, "water_stress": 0.24, "flood": 0.16, "cyclone": 0.12, "wildfire": 0.06},
    "cement":         {"heat_stress": 0.12, "water_stress": 0.20, "flood": 0.14, "cyclone": 0.10, "wildfire": 0.05},
    "chemicals":      {"heat_stress": 0.13, "water_stress": 0.22, "flood": 0.20, "cyclone": 0.18, "wildfire": 0.07},
    "mining":         {"heat_stress": 0.18, "water_stress": 0.28, "flood": 0.16, "cyclone": 0.14, "wildfire": 0.12},
    "aluminium":      {"heat_stress": 0.12, "water_stress": 0.20, "flood": 0.14, "cyclone": 0.10, "wildfire": 0.06},

    # Transport
    "aviation":       {"heat_stress": 0.08, "water_stress": 0.02, "flood": 0.12, "cyclone": 0.20, "wildfire": 0.06},
    "shipping":       {"heat_stress": 0.05, "water_stress": 0.03, "flood": 0.15, "cyclone": 0.22, "wildfire": 0.04},
    "automotive":     {"heat_stress": 0.10, "water_stress": 0.08, "flood": 0.14, "cyclone": 0.10, "wildfire": 0.06},
    "logistics":      {"heat_stress": 0.09, "water_stress": 0.05, "flood": 0.16, "cyclone": 0.14, "wildfire": 0.08},

    # Agriculture / Food
    "agriculture":    {"heat_stress": 0.30, "water_stress": 0.35, "flood": 0.25, "cyclone": 0.20, "wildfire": 0.10},
    "food_beverage":  {"heat_stress": 0.18, "water_stress": 0.25, "flood": 0.16, "cyclone": 0.12, "wildfire": 0.07},
    "fishing":        {"heat_stress": 0.20, "water_stress": 0.15, "flood": 0.10, "cyclone": 0.18, "wildfire": 0.03},

    # Real Estate / Infrastructure
    "real_estate":    {"heat_stress": 0.16, "water_stress": 0.08, "flood": 0.28, "cyclone": 0.24, "wildfire": 0.14},
    "construction":   {"heat_stress": 0.14, "water_stress": 0.06, "flood": 0.20, "cyclone": 0.16, "wildfire": 0.08},
    "infrastructure": {"heat_stress": 0.12, "water_stress": 0.10, "flood": 0.24, "cyclone": 0.22, "wildfire": 0.10},

    # Services / Knowledge
    "financials":     {"heat_stress": 0.04, "water_stress": 0.02, "flood": 0.08, "cyclone": 0.06, "wildfire": 0.03},
    "insurance":      {"heat_stress": 0.05, "water_stress": 0.02, "flood": 0.10, "cyclone": 0.08, "wildfire": 0.04},
    "technology":     {"heat_stress": 0.06, "water_stress": 0.04, "flood": 0.08, "cyclone": 0.06, "wildfire": 0.04},
    "healthcare":     {"heat_stress": 0.08, "water_stress": 0.05, "flood": 0.10, "cyclone": 0.08, "wildfire": 0.04},
    "consumer":       {"heat_stress": 0.07, "water_stress": 0.04, "flood": 0.09, "cyclone": 0.07, "wildfire": 0.04},
    "retail":         {"heat_stress": 0.07, "water_stress": 0.03, "flood": 0.10, "cyclone": 0.08, "wildfire": 0.05},
    "telecom":        {"heat_stress": 0.05, "water_stress": 0.03, "flood": 0.09, "cyclone": 0.10, "wildfire": 0.04},
    "pharma":         {"heat_stress": 0.06, "water_stress": 0.05, "flood": 0.08, "cyclone": 0.06, "wildfire": 0.03},
    "media":          {"heat_stress": 0.03, "water_stress": 0.01, "flood": 0.06, "cyclone": 0.05, "wildfire": 0.02},
}

# Sector name normalisation — map common synonyms to canonical names above
SECTOR_ALIASES: Dict[str, str] = {
    "oil": "oil_gas", "gas": "oil_gas", "petroleum": "oil_gas", "energy": "oil_gas",
    "fossil fuel": "oil_gas", "upstream oil": "oil_gas",
    "power": "utilities", "electricity": "utilities", "electric": "utilities",
    "power generation": "utilities", "utility": "utilities",
    "renewable": "renewable_energy", "solar": "renewable_energy", "wind": "renewable_energy",
    "iron": "steel", "iron and steel": "steel", "integrated steel": "steel",
    "building materials": "cement", "lime": "cement",
    "chemical": "chemicals", "specialty chemicals": "chemicals",
    "base metals": "mining", "gold": "mining", "copper": "mining",
    "iron ore": "mining", "mineral": "mining", "extractives": "mining",
    "air": "aviation", "airline": "aviation", "airport": "aviation",
    "marine": "shipping", "ocean": "shipping", "freight": "shipping",
    "car": "automotive", "vehicle": "automotive", "auto": "automotive",
    "truck": "automotive", "transport": "logistics",
    "food": "food_beverage", "beverage": "food_beverage", "consumer staples": "food_beverage",
    "farm": "agriculture", "crop": "agriculture", "agri": "agriculture",
    "bank": "financials", "banking": "financials", "finance": "financials",
    "investment": "financials", "asset management": "financials",
    "tech": "technology", "software": "technology", "semiconductor": "technology",
    "it": "technology", "data": "technology",
    "pharma": "pharma", "drugs": "pharma", "biotech": "pharma",
    "property": "real_estate", "reit": "real_estate", "housing": "real_estate",
    "infra": "infrastructure", "water": "infrastructure",
    "telecom": "telecom", "telecommunications": "telecom", "telecoms": "telecom",
    "consumer discretionary": "consumer", "luxury": "consumer",
}


def normalise_sector(sector_str: str) -> str:
    """Return the canonical sector key from a free-text sector description."""
    s = sector_str.lower().strip()
    if s in SECTOR_PHYSICAL_SENSITIVITY:
        return s
    # Try alias lookup
    for alias, canonical in SECTOR_ALIASES.items():
        if alias in s:
            return canonical
    # Partial match on canonical keys
    for key in SECTOR_PHYSICAL_SENSITIVITY:
        if key in s:
            return key
    return "technology"   # safe low-risk default for unknown sectors


# ════════════════════════════════════════════════════════════════════════════
# 4.  SECTOR TRANSITION RISK PARAMETERS
#     Source: NGFS Phase 4 (2023) financial sector impact studies +
#     IEA Net Zero by 2050 (2023) sector roadmaps + ESRB (2021) taxonomy.
#
#     Fields per sector:
#       carbon_intensity_t_per_m_rev: Scope 1+2 emission intensity
#           (tCO2e per $M revenue) — basis for carbon cost exposure.
#           Source: IEA/CDP sector averages.
#       carbon_price_coverage: Share of emissions under carbon pricing
#           or ETS exposure at 2025. Grows with policy tightening.
#       policy_risk_score: Likelihood of disruptive regulatory tightening
#           (0–1). Higher = more exposed to abrupt policy shifts.
#       stranded_asset_risk: Fraction of physical assets at risk of early
#           retirement under NZE pathway (0–1).
#           Source: IEA NZE 2050 stranded asset estimates.
#       demand_erosion_nze: Revenue loss (fraction) under NZE by 2050
#           relative to CP baseline. Negative for sectors that grow.
#           Source: NGFS Phase 4 sectoral demand pathways.
#       green_revenue_potential: Fraction of revenue that could transition
#           to low-carbon products under NZE by 2040.
# ════════════════════════════════════════════════════════════════════════════

SECTOR_TRANSITION_PARAMS: Dict[str, Dict[str, float]] = {
    "oil_gas": {
        "carbon_intensity_t_per_m_rev": 450.0,
        "carbon_price_coverage": 0.55,
        "policy_risk_score": 0.85,
        "stranded_asset_risk": 0.45,
        "demand_erosion_nze": -0.55,
        "green_revenue_potential": 0.25,
    },
    "coal": {
        "carbon_intensity_t_per_m_rev": 1_200.0,
        "carbon_price_coverage": 0.60,
        "policy_risk_score": 0.95,
        "stranded_asset_risk": 0.80,
        "demand_erosion_nze": -0.85,
        "green_revenue_potential": 0.05,
    },
    "utilities": {
        "carbon_intensity_t_per_m_rev": 320.0,
        "carbon_price_coverage": 0.75,
        "policy_risk_score": 0.70,
        "stranded_asset_risk": 0.30,
        "demand_erosion_nze": -0.15,    # electricity demand grows under NZE
        "green_revenue_potential": 0.60,
    },
    "renewable_energy": {
        "carbon_intensity_t_per_m_rev": 8.0,
        "carbon_price_coverage": 0.05,
        "policy_risk_score": 0.10,
        "stranded_asset_risk": 0.02,
        "demand_erosion_nze": 0.80,     # grows under NZE
        "green_revenue_potential": 0.95,
    },
    "steel": {
        "carbon_intensity_t_per_m_rev": 350.0,
        "carbon_price_coverage": 0.65,
        "policy_risk_score": 0.75,
        "stranded_asset_risk": 0.35,
        "demand_erosion_nze": -0.20,
        "green_revenue_potential": 0.30,
    },
    "cement": {
        "carbon_intensity_t_per_m_rev": 500.0,
        "carbon_price_coverage": 0.60,
        "policy_risk_score": 0.72,
        "stranded_asset_risk": 0.30,
        "demand_erosion_nze": -0.22,
        "green_revenue_potential": 0.25,
    },
    "chemicals": {
        "carbon_intensity_t_per_m_rev": 180.0,
        "carbon_price_coverage": 0.55,
        "policy_risk_score": 0.60,
        "stranded_asset_risk": 0.20,
        "demand_erosion_nze": -0.10,
        "green_revenue_potential": 0.35,
    },
    "mining": {
        "carbon_intensity_t_per_m_rev": 120.0,
        "carbon_price_coverage": 0.45,
        "policy_risk_score": 0.50,
        "stranded_asset_risk": 0.25,
        "demand_erosion_nze": 0.15,     # critical minerals grow
        "green_revenue_potential": 0.20,
    },
    "aluminium": {
        "carbon_intensity_t_per_m_rev": 280.0,
        "carbon_price_coverage": 0.58,
        "policy_risk_score": 0.65,
        "stranded_asset_risk": 0.22,
        "demand_erosion_nze": 0.08,     # EV demand supports aluminium
        "green_revenue_potential": 0.40,
    },
    "aviation": {
        "carbon_intensity_t_per_m_rev": 280.0,
        "carbon_price_coverage": 0.40,
        "policy_risk_score": 0.72,
        "stranded_asset_risk": 0.15,
        "demand_erosion_nze": -0.20,
        "green_revenue_potential": 0.20,
    },
    "shipping": {
        "carbon_intensity_t_per_m_rev": 160.0,
        "carbon_price_coverage": 0.30,
        "policy_risk_score": 0.62,
        "stranded_asset_risk": 0.20,
        "demand_erosion_nze": -0.18,
        "green_revenue_potential": 0.25,
    },
    "automotive": {
        "carbon_intensity_t_per_m_rev": 80.0,
        "carbon_price_coverage": 0.35,
        "policy_risk_score": 0.68,
        "stranded_asset_risk": 0.30,
        "demand_erosion_nze": -0.05,
        "green_revenue_potential": 0.65,
    },
    "logistics": {
        "carbon_intensity_t_per_m_rev": 90.0,
        "carbon_price_coverage": 0.35,
        "policy_risk_score": 0.50,
        "stranded_asset_risk": 0.12,
        "demand_erosion_nze": -0.08,
        "green_revenue_potential": 0.40,
    },
    "agriculture": {
        "carbon_intensity_t_per_m_rev": 90.0,
        "carbon_price_coverage": 0.15,
        "policy_risk_score": 0.35,
        "stranded_asset_risk": 0.08,
        "demand_erosion_nze": 0.05,
        "green_revenue_potential": 0.30,
    },
    "food_beverage": {
        "carbon_intensity_t_per_m_rev": 45.0,
        "carbon_price_coverage": 0.20,
        "policy_risk_score": 0.30,
        "stranded_asset_risk": 0.05,
        "demand_erosion_nze": 0.03,
        "green_revenue_potential": 0.25,
    },
    "real_estate": {
        "carbon_intensity_t_per_m_rev": 35.0,
        "carbon_price_coverage": 0.25,
        "policy_risk_score": 0.55,
        "stranded_asset_risk": 0.18,
        "demand_erosion_nze": -0.05,
        "green_revenue_potential": 0.45,
    },
    "construction": {
        "carbon_intensity_t_per_m_rev": 60.0,
        "carbon_price_coverage": 0.30,
        "policy_risk_score": 0.45,
        "stranded_asset_risk": 0.10,
        "demand_erosion_nze": 0.08,
        "green_revenue_potential": 0.35,
    },
    "infrastructure": {
        "carbon_intensity_t_per_m_rev": 55.0,
        "carbon_price_coverage": 0.35,
        "policy_risk_score": 0.48,
        "stranded_asset_risk": 0.12,
        "demand_erosion_nze": 0.05,
        "green_revenue_potential": 0.40,
    },
    "financials": {
        "carbon_intensity_t_per_m_rev": 8.0,
        "carbon_price_coverage": 0.10,
        "policy_risk_score": 0.35,
        "stranded_asset_risk": 0.05,
        "demand_erosion_nze": -0.03,
        "green_revenue_potential": 0.30,
    },
    "insurance": {
        "carbon_intensity_t_per_m_rev": 6.0,
        "carbon_price_coverage": 0.08,
        "policy_risk_score": 0.40,
        "stranded_asset_risk": 0.04,
        "demand_erosion_nze": 0.10,    # climate risk management demand grows
        "green_revenue_potential": 0.25,
    },
    "technology": {
        "carbon_intensity_t_per_m_rev": 15.0,
        "carbon_price_coverage": 0.10,
        "policy_risk_score": 0.20,
        "stranded_asset_risk": 0.03,
        "demand_erosion_nze": 0.15,
        "green_revenue_potential": 0.55,
    },
    "healthcare": {
        "carbon_intensity_t_per_m_rev": 18.0,
        "carbon_price_coverage": 0.12,
        "policy_risk_score": 0.20,
        "stranded_asset_risk": 0.02,
        "demand_erosion_nze": 0.10,
        "green_revenue_potential": 0.25,
    },
    "consumer": {
        "carbon_intensity_t_per_m_rev": 25.0,
        "carbon_price_coverage": 0.15,
        "policy_risk_score": 0.28,
        "stranded_asset_risk": 0.04,
        "demand_erosion_nze": 0.02,
        "green_revenue_potential": 0.30,
    },
    "retail": {
        "carbon_intensity_t_per_m_rev": 20.0,
        "carbon_price_coverage": 0.12,
        "policy_risk_score": 0.25,
        "stranded_asset_risk": 0.03,
        "demand_erosion_nze": 0.01,
        "green_revenue_potential": 0.28,
    },
    "telecom": {
        "carbon_intensity_t_per_m_rev": 12.0,
        "carbon_price_coverage": 0.10,
        "policy_risk_score": 0.18,
        "stranded_asset_risk": 0.02,
        "demand_erosion_nze": 0.12,
        "green_revenue_potential": 0.45,
    },
    "pharma": {
        "carbon_intensity_t_per_m_rev": 22.0,
        "carbon_price_coverage": 0.12,
        "policy_risk_score": 0.18,
        "stranded_asset_risk": 0.02,
        "demand_erosion_nze": 0.08,
        "green_revenue_potential": 0.30,
    },
    "fishing": {
        "carbon_intensity_t_per_m_rev": 85.0,
        "carbon_price_coverage": 0.18,
        "policy_risk_score": 0.40,
        "stranded_asset_risk": 0.10,
        "demand_erosion_nze": -0.05,
        "green_revenue_potential": 0.20,
    },
    "media": {
        "carbon_intensity_t_per_m_rev": 5.0,
        "carbon_price_coverage": 0.05,
        "policy_risk_score": 0.12,
        "stranded_asset_risk": 0.01,
        "demand_erosion_nze": 0.05,
        "green_revenue_potential": 0.35,
    },
}


# ════════════════════════════════════════════════════════════════════════════
# 5.  SECTOR FINANCIAL PARAMETERS
#     Used by CompanyProfileBuilder to construct realistic Company objects.
#     Source: Bloomberg Sector financial ratios (median) + Damodaran online
#     (Jan 2025) + NGFS financial impact studies for climate-adjusted WACC.
# ════════════════════════════════════════════════════════════════════════════

SECTOR_FINANCIAL_PARAMS: Dict[str, Dict[str, float]] = {
    # sector → {ebitda_margin, wacc_base, capex_pct_rev, maintenance_capex_share,
    #            tax_rate, net_debt_to_ebitda, emission_scope1_t_per_m_rev}
    "oil_gas":     {"ebitda_margin": 0.28, "wacc_base": 0.090, "capex_pct_rev": 0.22, "maint_capex": 0.45, "tax_rate": 0.28, "nd_ebitda": 1.8},
    "coal":        {"ebitda_margin": 0.24, "wacc_base": 0.110, "capex_pct_rev": 0.18, "maint_capex": 0.50, "tax_rate": 0.30, "nd_ebitda": 1.5},
    "utilities":   {"ebitda_margin": 0.35, "wacc_base": 0.065, "capex_pct_rev": 0.28, "maint_capex": 0.55, "tax_rate": 0.22, "nd_ebitda": 4.5},
    "renewable_energy": {"ebitda_margin": 0.55, "wacc_base": 0.060, "capex_pct_rev": 0.45, "maint_capex": 0.30, "tax_rate": 0.20, "nd_ebitda": 5.0},
    "steel":       {"ebitda_margin": 0.12, "wacc_base": 0.098, "capex_pct_rev": 0.10, "maint_capex": 0.55, "tax_rate": 0.25, "nd_ebitda": 2.2},
    "cement":      {"ebitda_margin": 0.22, "wacc_base": 0.090, "capex_pct_rev": 0.12, "maint_capex": 0.50, "tax_rate": 0.25, "nd_ebitda": 1.8},
    "chemicals":   {"ebitda_margin": 0.18, "wacc_base": 0.085, "capex_pct_rev": 0.10, "maint_capex": 0.45, "tax_rate": 0.24, "nd_ebitda": 2.0},
    "mining":      {"ebitda_margin": 0.35, "wacc_base": 0.095, "capex_pct_rev": 0.20, "maint_capex": 0.40, "tax_rate": 0.28, "nd_ebitda": 0.8},
    "aluminium":   {"ebitda_margin": 0.14, "wacc_base": 0.090, "capex_pct_rev": 0.12, "maint_capex": 0.50, "tax_rate": 0.26, "nd_ebitda": 1.6},
    "aviation":    {"ebitda_margin": 0.16, "wacc_base": 0.100, "capex_pct_rev": 0.14, "maint_capex": 0.60, "tax_rate": 0.22, "nd_ebitda": 3.5},
    "shipping":    {"ebitda_margin": 0.30, "wacc_base": 0.090, "capex_pct_rev": 0.18, "maint_capex": 0.45, "tax_rate": 0.18, "nd_ebitda": 2.8},
    "automotive":  {"ebitda_margin": 0.10, "wacc_base": 0.080, "capex_pct_rev": 0.06, "maint_capex": 0.55, "tax_rate": 0.25, "nd_ebitda": 1.2},
    "logistics":   {"ebitda_margin": 0.12, "wacc_base": 0.078, "capex_pct_rev": 0.08, "maint_capex": 0.55, "tax_rate": 0.23, "nd_ebitda": 1.5},
    "agriculture": {"ebitda_margin": 0.14, "wacc_base": 0.075, "capex_pct_rev": 0.08, "maint_capex": 0.60, "tax_rate": 0.20, "nd_ebitda": 1.0},
    "food_beverage":{"ebitda_margin": 0.16, "wacc_base": 0.070, "capex_pct_rev": 0.06, "maint_capex": 0.55, "tax_rate": 0.23, "nd_ebitda": 1.5},
    "real_estate": {"ebitda_margin": 0.55, "wacc_base": 0.065, "capex_pct_rev": 0.15, "maint_capex": 0.70, "tax_rate": 0.18, "nd_ebitda": 6.0},
    "construction":{"ebitda_margin": 0.10, "wacc_base": 0.082, "capex_pct_rev": 0.06, "maint_capex": 0.50, "tax_rate": 0.24, "nd_ebitda": 0.8},
    "infrastructure":{"ebitda_margin": 0.42, "wacc_base": 0.065, "capex_pct_rev": 0.22, "maint_capex": 0.65, "tax_rate": 0.20, "nd_ebitda": 5.0},
    "financials":  {"ebitda_margin": 0.40, "wacc_base": 0.095, "capex_pct_rev": 0.02, "maint_capex": 0.80, "tax_rate": 0.23, "nd_ebitda": 0.5},
    "insurance":   {"ebitda_margin": 0.15, "wacc_base": 0.088, "capex_pct_rev": 0.02, "maint_capex": 0.80, "tax_rate": 0.22, "nd_ebitda": 0.3},
    "technology":  {"ebitda_margin": 0.28, "wacc_base": 0.090, "capex_pct_rev": 0.05, "maint_capex": 0.60, "tax_rate": 0.18, "nd_ebitda": 0.2},
    "healthcare":  {"ebitda_margin": 0.22, "wacc_base": 0.075, "capex_pct_rev": 0.04, "maint_capex": 0.65, "tax_rate": 0.20, "nd_ebitda": 0.8},
    "consumer":    {"ebitda_margin": 0.16, "wacc_base": 0.078, "capex_pct_rev": 0.04, "maint_capex": 0.60, "tax_rate": 0.22, "nd_ebitda": 1.2},
    "retail":      {"ebitda_margin": 0.10, "wacc_base": 0.080, "capex_pct_rev": 0.03, "maint_capex": 0.65, "tax_rate": 0.22, "nd_ebitda": 1.0},
    "telecom":     {"ebitda_margin": 0.32, "wacc_base": 0.072, "capex_pct_rev": 0.18, "maint_capex": 0.65, "tax_rate": 0.22, "nd_ebitda": 3.5},
    "pharma":      {"ebitda_margin": 0.30, "wacc_base": 0.078, "capex_pct_rev": 0.04, "maint_capex": 0.60, "tax_rate": 0.18, "nd_ebitda": 0.5},
    "fishing":     {"ebitda_margin": 0.18, "wacc_base": 0.085, "capex_pct_rev": 0.08, "maint_capex": 0.55, "tax_rate": 0.22, "nd_ebitda": 1.4},
    "media":       {"ebitda_margin": 0.22, "wacc_base": 0.085, "capex_pct_rev": 0.04, "maint_capex": 0.70, "tax_rate": 0.21, "nd_ebitda": 2.0},
}


def get_sector_financial_params(sector: str) -> Dict[str, float]:
    """Return financial parameters for a given sector (canonical or alias)."""
    canonical = normalise_sector(sector)
    return SECTOR_FINANCIAL_PARAMS.get(canonical, SECTOR_FINANCIAL_PARAMS["technology"])


def get_sector_transition_params(sector: str) -> Dict[str, float]:
    """Return transition risk parameters for a given sector."""
    canonical = normalise_sector(sector)
    return SECTOR_TRANSITION_PARAMS.get(canonical, SECTOR_TRANSITION_PARAMS["technology"])


def get_sector_physical_sensitivity(sector: str) -> Dict[str, float]:
    """Return physical hazard sensitivity for a given sector."""
    canonical = normalise_sector(sector)
    return SECTOR_PHYSICAL_SENSITIVITY.get(canonical, SECTOR_PHYSICAL_SENSITIVITY["technology"])


# ════════════════════════════════════════════════════════════════════════════
# 6.  COMPOSITE RISK SCORE COMPUTATION
#     A deterministic, parameter-driven composite CRI score (0–100).
#     Higher = higher climate-financial risk.
#
#     Sub-score weights:
#       Physical Risk (40%): Hazard severity × sector sensitivity, EAL-adjusted
#       Transition Risk (40%): Carbon cost × price coverage × stranded assets
#       Adaptive Capacity (20%): Proxy from sector green revenue potential
#
#     All sub-scores are normalised to 0–100 before weighting.
# ════════════════════════════════════════════════════════════════════════════

def compute_composite_score(
    sector: str,
    climate_zone: str,
    scenario: str,
    carbon_price_usd: float = 50.0,
    year: int = 2030,
) -> Dict[str, float]:
    """Compute a parametric composite CRI score without needing a Company object.

    Args:
        sector:           Sector string (will be normalised).
        climate_zone:     One of the CLIMATE_ZONES identifiers.
        scenario:         'nze' | 'delayed' | 'cp'
        carbon_price_usd: Carbon price in USD/tCO2e at the given year.
        year:             Assessment year (2025–2050).

    Returns:
        Dict with keys: physical_score, transition_score, adaptive_score,
        composite_score (all 0–100).
    """
    sec = normalise_sector(sector)
    phys_sens = get_sector_physical_sensitivity(sec)
    trans_p   = get_sector_transition_params(sec)

    # Physical: weighted sum of hazard severity × sector sensitivity
    hazard_weights = {
        "heat_stress": 0.25, "water_stress": 0.25,
        "flood": 0.25, "cyclone": 0.15, "wildfire": 0.10,
    }
    phys_raw = 0.0
    for hazard, weight in hazard_weights.items():
        sev = get_hazard_profile(climate_zone, scenario, hazard, [year])[year]
        phys_raw += weight * sev * phys_sens.get(hazard, 0.05)

    # Normalise physical score to 0–100 (max possible ≈ 0.7 × 0.35 × 1.0 = 0.245)
    physical_score = min(100.0, phys_raw / 0.25 * 100.0)

    # Transition: carbon cost exposure + policy risk + stranded assets
    carbon_cost_pct = (
        trans_p["carbon_intensity_t_per_m_rev"]
        * trans_p["carbon_price_coverage"]
        * carbon_price_usd
        / 1_000_000.0  # cost as fraction of $1M revenue
        * 100.0          # to % of revenue
    )
    # Normalise: 10% of revenue = score 50; 20% = score 100
    carbon_score = min(100.0, carbon_cost_pct * 5.0)
    policy_score = trans_p["policy_risk_score"] * 100.0
    stranded_score = trans_p["stranded_asset_risk"] * 100.0
    transition_score = 0.5 * carbon_score + 0.35 * policy_score + 0.15 * stranded_score

    # Adaptive capacity (inverse of green potential → lower green potential = worse)
    adaptive_score = (1.0 - trans_p["green_revenue_potential"]) * 100.0

    # Composite (TCFD-aligned weightings)
    composite = (
        0.40 * physical_score
        + 0.40 * transition_score
        + 0.20 * adaptive_score
    )

    return {
        "physical_score":    round(physical_score, 1),
        "transition_score":  round(transition_score, 1),
        "adaptive_score":    round(adaptive_score, 1),
        "composite_score":   round(composite, 1),
    }
