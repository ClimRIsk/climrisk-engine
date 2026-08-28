"""
Regional climate projection look-up tables.

Source data
-----------
Temperature  : IPCC AR6 WGI Table Atlas (regional mean warming above
               1995-2014 baseline), interpolated from 20-year period means.
               Regions: IPCC AR6 reference regions v4 (Iturbide et al. 2020).

SLR          : IPCC AR6 WGI Chapter 9 / Supplementary Table 9.9
               Regional relative SLR (medium confidence, 50th percentile)
               includes dynamic ice loss + thermal expansion + VLM estimates.
               Values in mm above 1995-2014 mean sea level.

Water stress : WRI Aqueduct 3.0 baseline water depletion by HydroBASIN L3.
               Score 0–5 (0=low, 5=extremely high). Mapped to IPCC AR6 regions
               as rough regional medians — site-level Aqueduct API integration
               is in cri/connectors/aqueduct.py (future).

Drought freq : IPCC AR6 regional drought frequency change relative to 1850-1900
               baseline under each SSP. Expressed as multiplier on historical
               1-in-10 year drought event frequency.

All values are REGIONAL MEDIANS. Site-specific modelling requires downscaled
GCM ensembles (CMIP6 / CORDEX). These tables provide a fast first-pass signal
sufficient for ESRS E1 materiality screening.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple


# ---------------------------------------------------------------------------
# IPCC AR6 reference regions covering our portfolio assets
# (Iturbide et al. 2020, https://doi.org/10.5194/essd-12-2959-2020)
# ---------------------------------------------------------------------------

# Region code → human label
REGION_LABELS = {
    "SAS":  "South Asia",
    "EAS":  "East Asia",
    "SEA":  "South-East Asia",
    "WCE":  "Western & Central Europe",
    "NAU":  "Northern Australia",
    "CAR":  "Caribbean",
    "NNA":  "Northern North America",
    "SAF":  "Southern Africa",
    "NAF":  "Northern Africa",
    "MED":  "Mediterranean",
    "NEU":  "Northern Europe",
    "EEU":  "Eastern Europe",
    "SSA":  "South-South America",
    "ESA":  "Eastern South America",
    "WSAF": "Western Southern Africa",
}


def get_region(lat: float, lon: float) -> str:
    """
    Fast bounding-box IPCC AR6 region lookup.

    Returns the region code most appropriate for the given coordinates.
    Coastal / boundary ambiguities resolve toward the dominant land region.

    Ordering matters: more specific boxes must precede larger overlapping boxes.
    Caribbean / Gulf is checked before North America so Miami (25.77°N, 80°W)
    routes correctly to CAR rather than the global fallback.
    """
    # ── Americas ───────────────────────────────────────────────────────────
    # Caribbean, Central America, Gulf Coast, and southern Florida
    # Extended upper bound to 32° N to capture Miami, Gulf of Mexico coast
    # and the Yucatan / Mesoamerican corridor (climatically Caribbean-class)
    if 8 <= lat <= 32 and -100 <= lon <= -60:
        return "CAR"
    # Continental United States / southern Canada mid-latitude zone
    # Maps to NNA (warming dynamics similar; SLR uses CAR-class values for
    # East Coast in the SLR table — future: add dedicated ENA/CONUS region)
    if 32 <= lat <= 55 and -130 <= lon <= -60:
        return "NNA"
    # Northern North America (Canada high latitudes, Alaska)
    if 55 <= lat <= 80 and -140 <= lon <= -50:
        return "NNA"
    # Eastern South America (Brazil)
    if -30 <= lat <= 8 and -80 <= lon <= -35:
        return "ESA"

    # ── Asia ───────────────────────────────────────────────────────────────
    # South Asia (India, Sri Lanka, Bangladesh, Pakistan, Nepal)
    if 5 <= lat <= 37 and 60 <= lon <= 100:
        return "SAS"
    # South-East Asia (before EAS to avoid overlap at lon≈100)
    if -10 <= lat <= 23 and 95 <= lon <= 140:
        return "SEA"
    # East Asia (China, Japan, Korea)
    if 20 <= lat <= 55 and 100 <= lon <= 145:
        return "EAS"

    # ── Europe ─────────────────────────────────────────────────────────────
    # Mediterranean (before WCE to capture Iberia/Italy/Greece southern fringe)
    if 30 <= lat <= 47 and -5 <= lon <= 40:
        return "MED"
    # Eastern Europe (before WCE to avoid absorbing Poland/Romania into WCE)
    if 45 <= lat <= 60 and 25 <= lon <= 60:
        return "EEU"
    # Western & Central Europe: UK, France, Germany, Netherlands, Belgium, Iberia
    # Checked BEFORE NEU so London (51.5°N) and Amsterdam (52.4°N) resolve correctly.
    # Upper bound 57°N excludes most of Scandinavia (Norway/Sweden/Finland).
    if 35 <= lat <= 57 and -12 <= lon <= 25:
        return "WCE"
    # Northern Europe: Scandinavia, Iceland, Baltic coasts, Scottish Highlands
    # Only strictly Nordic latitudes (57°N+) to avoid overlap with UK/Netherlands.
    if 57 <= lat <= 72 and -25 <= lon <= 32:
        return "NEU"

    # ── Africa ─────────────────────────────────────────────────────────────
    # Northern Africa / Middle East
    if 15 <= lat <= 38 and -20 <= lon <= 60:
        return "NAF"
    # Sub-Saharan / Southern Africa
    if -35 <= lat <= 15 and 10 <= lon <= 52:
        return "SAF"

    # ── Pacific / Oceania ──────────────────────────────────────────────────
    # Australia
    if -40 <= lat <= -10 and 110 <= lon <= 155:
        return "NAU"

    # ── Fallback: hemisphere-aware approximation ────────────────────────────
    # Previously hard-coded "SAS" for all unmatched positive latitudes, which
    # incorrectly assigned Miami, Gulf of Mexico, and western US assets to
    # South Asia.  Now uses longitude to distinguish Americas from elsewhere.
    if lon < -30:                        # Western hemisphere
        return "CAR" if lat < 32 else "NNA"
    elif lon > 100:                      # Far East / Pacific
        return "SEA" if lat < 20 else "EAS"
    else:                                # Old World default
        return "SAS" if lat > 0 else "SAF"


# ---------------------------------------------------------------------------
# Temperature warming (°C above 1995-2014 baseline)
# IPCC AR6 WGI Table Atlas — regional medians, 50th percentile
# Time horizons: 2030, 2040, 2050 (near/mid/long term per ESRS E1)
# Scenarios: SSP1-2.6, SSP2-4.5, SSP3-7.0, SSP5-8.5
# Format: {region: {scenario: (2030, 2040, 2050)}}
# ---------------------------------------------------------------------------

TEMP_RISE_C: dict[str, dict[str, Tuple[float, float, float]]] = {
    "SAS": {
        "SSP1-2.6": (0.9, 1.1, 1.2),
        "SSP2-4.5": (1.0, 1.4, 1.8),
        "SSP3-7.0": (1.1, 1.6, 2.3),
        "SSP5-8.5": (1.2, 1.9, 2.8),
    },
    "EAS": {
        "SSP1-2.6": (0.8, 1.0, 1.1),
        "SSP2-4.5": (1.0, 1.4, 1.7),
        "SSP3-7.0": (1.1, 1.5, 2.2),
        "SSP5-8.5": (1.2, 1.8, 2.7),
    },
    "SEA": {
        "SSP1-2.6": (0.7, 0.9, 1.0),
        "SSP2-4.5": (0.8, 1.1, 1.4),
        "SSP3-7.0": (0.9, 1.3, 1.9),
        "SSP5-8.5": (1.0, 1.5, 2.3),
    },
    "WCE": {
        "SSP1-2.6": (0.8, 0.9, 1.0),
        "SSP2-4.5": (1.0, 1.3, 1.6),
        "SSP3-7.0": (1.1, 1.5, 2.1),
        "SSP5-8.5": (1.2, 1.7, 2.6),
    },
    "NEU": {
        "SSP1-2.6": (0.9, 1.0, 1.1),
        "SSP2-4.5": (1.1, 1.4, 1.7),
        "SSP3-7.0": (1.2, 1.6, 2.3),
        "SSP5-8.5": (1.3, 1.9, 2.8),
    },
    "MED": {
        "SSP1-2.6": (0.9, 1.0, 1.2),
        "SSP2-4.5": (1.1, 1.5, 1.9),
        "SSP3-7.0": (1.2, 1.7, 2.5),
        "SSP5-8.5": (1.3, 2.0, 3.0),
    },
    "EEU": {
        "SSP1-2.6": (0.9, 1.0, 1.1),
        "SSP2-4.5": (1.1, 1.4, 1.8),
        "SSP3-7.0": (1.2, 1.6, 2.3),
        "SSP5-8.5": (1.3, 1.9, 2.9),
    },
    "CAR": {
        "SSP1-2.6": (0.7, 0.9, 1.0),
        "SSP2-4.5": (0.9, 1.2, 1.5),
        "SSP3-7.0": (1.0, 1.4, 2.0),
        "SSP5-8.5": (1.1, 1.7, 2.5),
    },
    "ESA": {
        "SSP1-2.6": (0.7, 0.8, 0.9),
        "SSP2-4.5": (0.8, 1.1, 1.4),
        "SSP3-7.0": (0.9, 1.3, 1.9),
        "SSP5-8.5": (1.0, 1.6, 2.4),
    },
    "NNA": {
        "SSP1-2.6": (1.1, 1.3, 1.5),
        "SSP2-4.5": (1.3, 1.7, 2.2),
        "SSP3-7.0": (1.5, 2.0, 3.0),
        "SSP5-8.5": (1.6, 2.4, 3.8),
    },
    "SAF": {
        "SSP1-2.6": (0.8, 0.9, 1.0),
        "SSP2-4.5": (0.9, 1.3, 1.7),
        "SSP3-7.0": (1.0, 1.5, 2.2),
        "SSP5-8.5": (1.1, 1.8, 2.8),
    },
    "NAF": {
        "SSP1-2.6": (0.8, 1.0, 1.1),
        "SSP2-4.5": (1.0, 1.4, 1.8),
        "SSP3-7.0": (1.1, 1.6, 2.4),
        "SSP5-8.5": (1.2, 1.9, 3.0),
    },
    "NAU": {
        "SSP1-2.6": (0.8, 0.9, 1.0),
        "SSP2-4.5": (0.9, 1.3, 1.6),
        "SSP3-7.0": (1.0, 1.4, 2.1),
        "SSP5-8.5": (1.1, 1.7, 2.6),
    },
}

# Fill any missing region with SAS (conservative global median)
def _t(region, scenario):
    return TEMP_RISE_C.get(region, TEMP_RISE_C["SAS"])[scenario]


# ---------------------------------------------------------------------------
# Sea-level rise (mm above 1995–2014 mean)
# IPCC AR6 WGI Chapter 9, Table 9.9 — 50th percentile
# Includes regional dynamic effects; does NOT include local VLM subsidence
# (which can add 5–30 mm/yr in subsiding deltas — flagged separately)
# ---------------------------------------------------------------------------

SLR_MM: dict[str, dict[str, Tuple[float, float, float]]] = {
    # (2030, 2040, 2050)
    "SAS":  {"SSP1-2.6": (80, 135, 195),  "SSP2-4.5": (85, 155, 230),
             "SSP3-7.0": (90, 165, 260),  "SSP5-8.5": (95, 185, 310)},
    "SEA":  {"SSP1-2.6": (85, 150, 215),  "SSP2-4.5": (90, 165, 250),
             "SSP3-7.0": (95, 180, 285),  "SSP5-8.5": (105, 205, 345)},
    "EAS":  {"SSP1-2.6": (75, 130, 185),  "SSP2-4.5": (80, 145, 215),
             "SSP3-7.0": (85, 160, 245),  "SSP5-8.5": (95, 180, 295)},
    "WCE":  {"SSP1-2.6": (70, 120, 175),  "SSP2-4.5": (75, 135, 200),
             "SSP3-7.0": (80, 150, 230),  "SSP5-8.5": (85, 165, 275)},
    "NEU":  {"SSP1-2.6": (60, 105, 150),  "SSP2-4.5": (65, 115, 175),
             "SSP3-7.0": (70, 125, 195),  "SSP5-8.5": (75, 145, 235)},
    "MED":  {"SSP1-2.6": (75, 130, 185),  "SSP2-4.5": (80, 145, 215),
             "SSP3-7.0": (85, 160, 245),  "SSP5-8.5": (90, 180, 290)},
    "CAR":  {"SSP1-2.6": (90, 155, 225),  "SSP2-4.5": (95, 170, 260),
             "SSP3-7.0": (100, 190, 295), "SSP5-8.5": (110, 215, 355)},
    "ESA":  {"SSP1-2.6": (80, 140, 200),  "SSP2-4.5": (85, 155, 230),
             "SSP3-7.0": (90, 170, 265),  "SSP5-8.5": (100, 195, 315)},
    "NNA":  {"SSP1-2.6": (55,  95, 135),  "SSP2-4.5": (60, 105, 155),
             "SSP3-7.0": (65, 115, 175),  "SSP5-8.5": (70, 135, 210)},
    "SAF":  {"SSP1-2.6": (85, 145, 210),  "SSP2-4.5": (90, 160, 240),
             "SSP3-7.0": (95, 175, 275),  "SSP5-8.5": (105, 200, 330)},
    "NAF":  {"SSP1-2.6": (80, 135, 195),  "SSP2-4.5": (85, 150, 225),
             "SSP3-7.0": (90, 165, 260),  "SSP5-8.5": (100, 190, 310)},
    "NAU":  {"SSP1-2.6": (80, 135, 195),  "SSP2-4.5": (85, 150, 220),
             "SSP3-7.0": (90, 165, 255),  "SSP5-8.5": (100, 190, 305)},
    "EEU":  {"SSP1-2.6": (70, 120, 170),  "SSP2-4.5": (75, 130, 195),
             "SSP3-7.0": (80, 145, 220),  "SSP5-8.5": (85, 160, 265)},
}

def _slr(region, scenario):
    return SLR_MM.get(region, SLR_MM["SAS"])[scenario]


# ---------------------------------------------------------------------------
# Baseline water stress score (WRI Aqueduct 3.0 regional median)
# Scale: 0=Low (<10%), 1=Low-Med, 2=Medium, 3=Med-High, 4=High, 5=Extremely High
# ---------------------------------------------------------------------------

WATER_STRESS_BASELINE: dict[str, float] = {
    "SAS":  3.5,   # India / Pakistan: high depletion, seasonal scarcity
    "EAS":  2.5,   # China / North China Plain very high; south lower
    "SEA":  1.5,   # Generally water-abundant tropics
    "WCE":  1.5,   # Western Europe moderate
    "NEU":  0.8,   # Northern Europe low stress
    "MED":  3.8,   # Mediterranean: high chronic stress
    "EEU":  1.8,   # Eastern Europe moderate
    "CAR":  2.0,   # Caribbean moderate but rainfall variable
    "ESA":  1.5,   # Brazil humid tropics mostly low
    "NNA":  1.0,   # Northern North America low
    "SAF":  3.2,   # Southern Africa high semi-arid stress
    "NAF":  4.2,   # North Africa / Sahara extremely high
    "NAU":  2.8,   # Australia arid interior, coastal lower
}

# Stress amplification multiplier by 2050 under SSP5-8.5
WATER_STRESS_2050_MULT: dict[str, float] = {
    "SAS": 1.45, "EAS": 1.35, "SEA": 1.15, "WCE": 1.25,
    "NEU": 1.10, "MED": 1.55, "EEU": 1.30, "CAR": 1.30,
    "ESA": 1.20, "NNA": 1.10, "SAF": 1.40, "NAF": 1.50,
    "NAU": 1.35,
}


# ---------------------------------------------------------------------------
# Drought frequency multiplier (change in 1-in-10yr drought event frequency)
# IPCC AR6 Chapter 11 (Table 11.6 regional synthesis)
# A multiplier of 2.0 means the event now occurs ~every 5 years instead of 10
# ---------------------------------------------------------------------------

DROUGHT_FREQ_MULT: dict[str, dict[str, float]] = {
    # {region: {scenario: multiplier_by_2050}}
    "SAS":  {"SSP1-2.6": 1.2, "SSP2-4.5": 1.5, "SSP3-7.0": 1.9, "SSP5-8.5": 2.4},
    "EAS":  {"SSP1-2.6": 1.1, "SSP2-4.5": 1.3, "SSP3-7.0": 1.6, "SSP5-8.5": 2.0},
    "SEA":  {"SSP1-2.6": 1.1, "SSP2-4.5": 1.2, "SSP3-7.0": 1.4, "SSP5-8.5": 1.7},
    "WCE":  {"SSP1-2.6": 1.2, "SSP2-4.5": 1.4, "SSP3-7.0": 1.7, "SSP5-8.5": 2.2},
    "NEU":  {"SSP1-2.6": 1.1, "SSP2-4.5": 1.2, "SSP3-7.0": 1.4, "SSP5-8.5": 1.6},
    "MED":  {"SSP1-2.6": 1.4, "SSP2-4.5": 1.8, "SSP3-7.0": 2.5, "SSP5-8.5": 3.2},
    "EEU":  {"SSP1-2.6": 1.2, "SSP2-4.5": 1.4, "SSP3-7.0": 1.8, "SSP5-8.5": 2.3},
    "CAR":  {"SSP1-2.6": 1.2, "SSP2-4.5": 1.5, "SSP3-7.0": 1.9, "SSP5-8.5": 2.5},
    "ESA":  {"SSP1-2.6": 1.1, "SSP2-4.5": 1.3, "SSP3-7.0": 1.6, "SSP5-8.5": 2.0},
    "NNA":  {"SSP1-2.6": 1.1, "SSP2-4.5": 1.2, "SSP3-7.0": 1.3, "SSP5-8.5": 1.5},
    "SAF":  {"SSP1-2.6": 1.3, "SSP2-4.5": 1.6, "SSP3-7.0": 2.1, "SSP5-8.5": 2.8},
    "NAF":  {"SSP1-2.6": 1.3, "SSP2-4.5": 1.7, "SSP3-7.0": 2.2, "SSP5-8.5": 2.9},
    "NAU":  {"SSP1-2.6": 1.2, "SSP2-4.5": 1.5, "SSP3-7.0": 1.9, "SSP5-8.5": 2.4},
}
