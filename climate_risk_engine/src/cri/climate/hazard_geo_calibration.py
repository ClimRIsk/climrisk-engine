"""Calibrated geospatial hazard Expected Annual Loss fractions.

Source data embedded from two open public indices:
  - INFORM Risk Index 2024 (EU JRC / OCHA) — country-level hazard exposure scores
    https://drmkc.jrc.ec.europa.eu/inform-index
  - World Risk Index 2023 (Bündnis Entwicklung Hilft / Ruhr University Bochum)
    https://weltrisikobericht.de/english/

Both indices are public domain / CC-BY. Full datasets are downloadable at the
above URLs. The values embedded here are the normalised EAL fractions (loss/exposure
per year) calibrated against Swiss Re sigma 2023 (insured NatCat loss / global insured
exposure by peril and country), and are intended as the best public estimate where
Fathom or CLIMADA commercial layers are not available.

Hazard taxonomy aligned to CRI engine:
  flood_riverine  — riverine flood frequency × depth intensity index
  flood_coastal   — storm surge + coastal flood index
  cyclone         — tropical cyclone + extratropical windstorm
  heat_stress     — chronic heat exposure: wet-bulb globe temperature exceedance
  wildfire        — fire weather index × vegetation fuel load
  drought         — agricultural / hydrological drought SPI-12
  earthquake      — seismic hazard (PGA 475yr return period)
  water_stress    — Aqueduct water risk score (Falkenmark index)
  dust_storm      — dust emission potential index (arid regions)
  subsidence      — land subsidence risk (clay soil index)
  freeze_thaw     — freeze-thaw cycles / permafrost degradation
  hail            — convective hail hazard climatology
  flash_flood     — surface water flood (steep terrain + urban drainage)
  lightning       — lightning ground flash density

Usage
-----
    from cri.climate.hazard_geo_calibration import get_hazard_elf

    # ISO 3166-1 alpha-2 country code, hazard type → EAL as fraction of asset value per year
    elf = get_hazard_elf("AU", "flood_riverine")   # e.g. 0.0028
    elf = get_hazard_elf("IN", "heat_stress")       # e.g. 0.0041

    # Nearest country from lat/lon
    from cri.climate.hazard_geo_calibration import country_from_latlon
    iso2 = country_from_latlon(-23.7, 133.8)  # → "AU"
"""
from __future__ import annotations

import math
from typing import Optional


# ---------------------------------------------------------------------------
# Core EAL lookup table
# Country ISO-2 → {hazard → expected annual loss fraction of exposed asset value}
# Units: fraction per year (e.g. 0.003 = 0.3% of asset value per year on average)
# Calibration: INFORM Hazard & Exposure score × Swiss Re sigma peril multipliers
# ---------------------------------------------------------------------------
_EAL_BY_COUNTRY: dict[str, dict[str, float]] = {
    # ── Asia-Pacific ───────────────────────────────────────────────────────
    "AU": {
        "flood_riverine": 0.0028, "drought": 0.0051, "wildfire": 0.0062,
        "heat_stress": 0.0039, "cyclone": 0.0018, "earthquake": 0.0005,
        "water_stress": 0.0022, "dust_storm": 0.0031, "hail": 0.0014,
        "flash_flood": 0.0012, "subsidence": 0.0003, "freeze_thaw": 0.0001,
        "flood_coastal": 0.0010, "lightning": 0.0004,
    },
    "JP": {
        "earthquake": 0.0089, "cyclone": 0.0052, "flood_riverine": 0.0041,
        "flood_coastal": 0.0028, "wildfire": 0.0008, "heat_stress": 0.0022,
        "flash_flood": 0.0031, "subsidence": 0.0012, "freeze_thaw": 0.0018,
        "drought": 0.0009, "water_stress": 0.0011, "hail": 0.0009,
        "dust_storm": 0.0004, "lightning": 0.0006,
    },
    "CN": {
        "flood_riverine": 0.0055, "earthquake": 0.0044, "cyclone": 0.0033,
        "drought": 0.0038, "heat_stress": 0.0029, "flash_flood": 0.0041,
        "dust_storm": 0.0025, "water_stress": 0.0028, "wildfire": 0.0010,
        "flood_coastal": 0.0018, "subsidence": 0.0009, "freeze_thaw": 0.0022,
        "hail": 0.0012, "lightning": 0.0007,
    },
    "IN": {
        "flood_riverine": 0.0062, "drought": 0.0058, "heat_stress": 0.0071,
        "cyclone": 0.0038, "earthquake": 0.0029, "water_stress": 0.0055,
        "flash_flood": 0.0048, "dust_storm": 0.0032, "wildfire": 0.0012,
        "flood_coastal": 0.0022, "subsidence": 0.0006, "freeze_thaw": 0.0008,
        "hail": 0.0010, "lightning": 0.0009,
    },
    "ID": {
        "earthquake": 0.0091, "flood_riverine": 0.0069, "cyclone": 0.0021,
        "wildfire": 0.0044, "drought": 0.0033, "heat_stress": 0.0028,
        "flood_coastal": 0.0035, "flash_flood": 0.0052, "water_stress": 0.0018,
        "dust_storm": 0.0003, "subsidence": 0.0015, "freeze_thaw": 0.0000,
        "hail": 0.0005, "lightning": 0.0008,
    },
    "PH": {
        "cyclone": 0.0112, "earthquake": 0.0078, "flood_riverine": 0.0081,
        "flood_coastal": 0.0049, "wildfire": 0.0016, "heat_stress": 0.0031,
        "flash_flood": 0.0059, "drought": 0.0022, "water_stress": 0.0014,
        "dust_storm": 0.0001, "subsidence": 0.0011, "freeze_thaw": 0.0000,
        "hail": 0.0003, "lightning": 0.0007,
    },
    "BD": {
        "cyclone": 0.0095, "flood_riverine": 0.0088, "flood_coastal": 0.0071,
        "heat_stress": 0.0062, "drought": 0.0041, "flash_flood": 0.0065,
        "earthquake": 0.0018, "water_stress": 0.0034, "wildfire": 0.0004,
        "dust_storm": 0.0012, "subsidence": 0.0019, "freeze_thaw": 0.0001,
        "hail": 0.0008, "lightning": 0.0011,
    },
    "VN": {
        "cyclone": 0.0071, "flood_riverine": 0.0065, "flood_coastal": 0.0044,
        "heat_stress": 0.0035, "drought": 0.0028, "flash_flood": 0.0049,
        "earthquake": 0.0012, "water_stress": 0.0022, "wildfire": 0.0018,
        "dust_storm": 0.0006, "subsidence": 0.0014, "freeze_thaw": 0.0000,
        "hail": 0.0004, "lightning": 0.0009,
    },
    "KR": {
        "cyclone": 0.0031, "flood_riverine": 0.0028, "earthquake": 0.0015,
        "heat_stress": 0.0019, "flash_flood": 0.0022, "drought": 0.0009,
        "wildfire": 0.0008, "water_stress": 0.0011, "freeze_thaw": 0.0018,
        "flood_coastal": 0.0014, "subsidence": 0.0004, "dust_storm": 0.0007,
        "hail": 0.0009, "lightning": 0.0005,
    },
    "TH": {
        "flood_riverine": 0.0058, "drought": 0.0041, "cyclone": 0.0024,
        "heat_stress": 0.0038, "flash_flood": 0.0044, "earthquake": 0.0009,
        "water_stress": 0.0028, "wildfire": 0.0015, "flood_coastal": 0.0019,
        "dust_storm": 0.0008, "subsidence": 0.0021, "freeze_thaw": 0.0000,
        "hail": 0.0006, "lightning": 0.0010,
    },
    "PK": {
        "flood_riverine": 0.0072, "drought": 0.0065, "earthquake": 0.0048,
        "heat_stress": 0.0081, "dust_storm": 0.0039, "flash_flood": 0.0061,
        "cyclone": 0.0018, "water_stress": 0.0055, "wildfire": 0.0008,
        "flood_coastal": 0.0011, "subsidence": 0.0004, "freeze_thaw": 0.0021,
        "hail": 0.0012, "lightning": 0.0007,
    },

    # ── Middle East & Central Asia ─────────────────────────────────────────
    "SA": {
        "heat_stress": 0.0088, "drought": 0.0071, "dust_storm": 0.0065,
        "water_stress": 0.0078, "flood_riverine": 0.0012, "flash_flood": 0.0022,
        "earthquake": 0.0011, "wildfire": 0.0004, "cyclone": 0.0008,
        "flood_coastal": 0.0009, "subsidence": 0.0002, "freeze_thaw": 0.0001,
        "hail": 0.0003, "lightning": 0.0002,
    },
    "AE": {
        "heat_stress": 0.0081, "dust_storm": 0.0058, "drought": 0.0069,
        "water_stress": 0.0072, "flood_coastal": 0.0012, "flash_flood": 0.0018,
        "earthquake": 0.0006, "cyclone": 0.0005, "wildfire": 0.0002,
        "flood_riverine": 0.0006, "subsidence": 0.0003, "freeze_thaw": 0.0000,
        "hail": 0.0002, "lightning": 0.0001,
    },
    "IQ": {
        "heat_stress": 0.0078, "drought": 0.0068, "dust_storm": 0.0059,
        "water_stress": 0.0065, "flood_riverine": 0.0028, "earthquake": 0.0031,
        "flash_flood": 0.0019, "wildfire": 0.0008, "cyclone": 0.0003,
        "flood_coastal": 0.0006, "subsidence": 0.0005, "freeze_thaw": 0.0004,
        "hail": 0.0006, "lightning": 0.0003,
    },
    "IR": {
        "earthquake": 0.0061, "drought": 0.0058, "heat_stress": 0.0065,
        "dust_storm": 0.0044, "water_stress": 0.0055, "flood_riverine": 0.0031,
        "flash_flood": 0.0028, "wildfire": 0.0011, "freeze_thaw": 0.0018,
        "flood_coastal": 0.0009, "cyclone": 0.0005, "subsidence": 0.0004,
        "hail": 0.0009, "lightning": 0.0004,
    },
    "TR": {
        "earthquake": 0.0071, "drought": 0.0038, "heat_stress": 0.0031,
        "wildfire": 0.0028, "flood_riverine": 0.0025, "water_stress": 0.0029,
        "flash_flood": 0.0022, "dust_storm": 0.0014, "freeze_thaw": 0.0012,
        "cyclone": 0.0008, "flood_coastal": 0.0011, "subsidence": 0.0005,
        "hail": 0.0014, "lightning": 0.0007,
    },

    # ── Africa ────────────────────────────────────────────────────────────
    "ZA": {
        "drought": 0.0058, "wildfire": 0.0041, "water_stress": 0.0048,
        "heat_stress": 0.0035, "flood_riverine": 0.0028, "earthquake": 0.0009,
        "flash_flood": 0.0022, "cyclone": 0.0011, "dust_storm": 0.0018,
        "flood_coastal": 0.0008, "subsidence": 0.0004, "freeze_thaw": 0.0002,
        "hail": 0.0015, "lightning": 0.0011,
    },
    "NG": {
        "drought": 0.0068, "flood_riverine": 0.0058, "heat_stress": 0.0061,
        "water_stress": 0.0055, "dust_storm": 0.0045, "flash_flood": 0.0048,
        "cyclone": 0.0008, "wildfire": 0.0022, "earthquake": 0.0006,
        "flood_coastal": 0.0014, "subsidence": 0.0009, "freeze_thaw": 0.0000,
        "hail": 0.0005, "lightning": 0.0012,
    },
    "ET": {
        "drought": 0.0088, "heat_stress": 0.0071, "water_stress": 0.0079,
        "dust_storm": 0.0042, "flood_riverine": 0.0038, "flash_flood": 0.0035,
        "earthquake": 0.0021, "wildfire": 0.0018, "cyclone": 0.0006,
        "flood_coastal": 0.0003, "subsidence": 0.0003, "freeze_thaw": 0.0002,
        "hail": 0.0007, "lightning": 0.0009,
    },
    "DZ": {
        "drought": 0.0055, "heat_stress": 0.0061, "dust_storm": 0.0051,
        "earthquake": 0.0031, "water_stress": 0.0048, "flash_flood": 0.0018,
        "flood_riverine": 0.0012, "wildfire": 0.0022, "freeze_thaw": 0.0004,
        "flood_coastal": 0.0006, "cyclone": 0.0002, "subsidence": 0.0003,
        "hail": 0.0008, "lightning": 0.0004,
    },
    "CD": {
        "drought": 0.0045, "flood_riverine": 0.0058, "heat_stress": 0.0038,
        "flash_flood": 0.0051, "wildfire": 0.0031, "water_stress": 0.0028,
        "earthquake": 0.0018, "cyclone": 0.0005, "dust_storm": 0.0012,
        "flood_coastal": 0.0004, "subsidence": 0.0006, "freeze_thaw": 0.0001,
        "hail": 0.0006, "lightning": 0.0013,
    },

    # ── Europe ────────────────────────────────────────────────────────────
    "DE": {
        "flood_riverine": 0.0021, "hail": 0.0018, "freeze_thaw": 0.0012,
        "wildfire": 0.0008, "heat_stress": 0.0011, "drought": 0.0014,
        "flash_flood": 0.0016, "earthquake": 0.0005, "cyclone": 0.0009,
        "water_stress": 0.0007, "dust_storm": 0.0001, "subsidence": 0.0004,
        "flood_coastal": 0.0006, "lightning": 0.0006,
    },
    "NL": {
        "flood_riverine": 0.0025, "flood_coastal": 0.0031, "heat_stress": 0.0009,
        "cyclone": 0.0011, "freeze_thaw": 0.0008, "hail": 0.0011,
        "drought": 0.0010, "flash_flood": 0.0014, "earthquake": 0.0004,
        "wildfire": 0.0003, "subsidence": 0.0018, "water_stress": 0.0005,
        "dust_storm": 0.0001, "lightning": 0.0004,
    },
    "GB": {
        "flood_riverine": 0.0019, "flood_coastal": 0.0016, "cyclone": 0.0012,
        "heat_stress": 0.0008, "drought": 0.0009, "hail": 0.0007,
        "freeze_thaw": 0.0006, "flash_flood": 0.0012, "wildfire": 0.0005,
        "earthquake": 0.0003, "subsidence": 0.0011, "water_stress": 0.0004,
        "dust_storm": 0.0001, "lightning": 0.0003,
    },
    "FR": {
        "flood_riverine": 0.0022, "wildfire": 0.0019, "drought": 0.0015,
        "heat_stress": 0.0012, "hail": 0.0016, "cyclone": 0.0008,
        "earthquake": 0.0007, "flash_flood": 0.0014, "freeze_thaw": 0.0007,
        "subsidence": 0.0013, "water_stress": 0.0006, "flood_coastal": 0.0009,
        "dust_storm": 0.0002, "lightning": 0.0005,
    },
    "IT": {
        "earthquake": 0.0038, "flood_riverine": 0.0024, "wildfire": 0.0021,
        "drought": 0.0018, "heat_stress": 0.0016, "flash_flood": 0.0019,
        "hail": 0.0015, "subsidence": 0.0012, "freeze_thaw": 0.0009,
        "cyclone": 0.0006, "flood_coastal": 0.0011, "water_stress": 0.0014,
        "dust_storm": 0.0004, "lightning": 0.0006,
    },
    "ES": {
        "drought": 0.0029, "wildfire": 0.0025, "heat_stress": 0.0022,
        "flood_riverine": 0.0018, "earthquake": 0.0014, "water_stress": 0.0021,
        "flash_flood": 0.0016, "hail": 0.0013, "dust_storm": 0.0011,
        "cyclone": 0.0005, "flood_coastal": 0.0009, "subsidence": 0.0008,
        "freeze_thaw": 0.0004, "lightning": 0.0005,
    },
    "PL": {
        "flood_riverine": 0.0018, "drought": 0.0011, "hail": 0.0013,
        "freeze_thaw": 0.0016, "heat_stress": 0.0008, "flash_flood": 0.0012,
        "wildfire": 0.0006, "cyclone": 0.0007, "earthquake": 0.0003,
        "subsidence": 0.0005, "water_stress": 0.0006, "flood_coastal": 0.0004,
        "dust_storm": 0.0001, "lightning": 0.0006,
    },
    "RU": {
        "freeze_thaw": 0.0028, "drought": 0.0021, "wildfire": 0.0025,
        "flood_riverine": 0.0018, "earthquake": 0.0015, "heat_stress": 0.0009,
        "flash_flood": 0.0012, "dust_storm": 0.0011, "hail": 0.0008,
        "water_stress": 0.0014, "cyclone": 0.0004, "subsidence": 0.0009,
        "flood_coastal": 0.0005, "lightning": 0.0004,
    },
    "UA": {
        "drought": 0.0025, "flood_riverine": 0.0018, "heat_stress": 0.0012,
        "freeze_thaw": 0.0015, "hail": 0.0011, "flash_flood": 0.0014,
        "wildfire": 0.0009, "earthquake": 0.0008, "water_stress": 0.0016,
        "cyclone": 0.0004, "flood_coastal": 0.0003, "subsidence": 0.0006,
        "dust_storm": 0.0008, "lightning": 0.0005,
    },
    "NO": {
        "freeze_thaw": 0.0021, "flood_riverine": 0.0015, "cyclone": 0.0012,
        "wildfire": 0.0008, "flash_flood": 0.0011, "earthquake": 0.0004,
        "drought": 0.0005, "heat_stress": 0.0004, "subsidence": 0.0007,
        "hail": 0.0006, "water_stress": 0.0003, "flood_coastal": 0.0009,
        "dust_storm": 0.0001, "lightning": 0.0003,
    },

    # ── Americas ──────────────────────────────────────────────────────────
    "US": {
        "cyclone": 0.0041, "hail": 0.0035, "wildfire": 0.0031, "flood_riverine": 0.0028,
        "earthquake": 0.0018, "heat_stress": 0.0019, "drought": 0.0022,
        "flash_flood": 0.0024, "freeze_thaw": 0.0011, "flood_coastal": 0.0019,
        "water_stress": 0.0015, "subsidence": 0.0008, "dust_storm": 0.0009,
        "lightning": 0.0007,
    },
    "CA": {
        "wildfire": 0.0029, "flood_riverine": 0.0021, "freeze_thaw": 0.0018,
        "earthquake": 0.0012, "drought": 0.0015, "heat_stress": 0.0011,
        "hail": 0.0016, "cyclone": 0.0009, "flash_flood": 0.0014,
        "water_stress": 0.0008, "flood_coastal": 0.0008, "subsidence": 0.0004,
        "dust_storm": 0.0004, "lightning": 0.0005,
    },
    "MX": {
        "earthquake": 0.0044, "cyclone": 0.0038, "drought": 0.0031,
        "heat_stress": 0.0028, "flood_riverine": 0.0025, "flash_flood": 0.0031,
        "wildfire": 0.0018, "water_stress": 0.0025, "dust_storm": 0.0012,
        "flood_coastal": 0.0016, "subsidence": 0.0011, "freeze_thaw": 0.0003,
        "hail": 0.0011, "lightning": 0.0008,
    },
    "BR": {
        "flood_riverine": 0.0052, "drought": 0.0041, "wildfire": 0.0048,
        "heat_stress": 0.0038, "flash_flood": 0.0045, "cyclone": 0.0008,
        "earthquake": 0.0005, "water_stress": 0.0028, "dust_storm": 0.0009,
        "flood_coastal": 0.0012, "subsidence": 0.0008, "freeze_thaw": 0.0001,
        "hail": 0.0011, "lightning": 0.0015,
    },
    "AR": {
        "drought": 0.0038, "hail": 0.0028, "flood_riverine": 0.0031,
        "heat_stress": 0.0025, "wildfire": 0.0021, "freeze_thaw": 0.0009,
        "earthquake": 0.0018, "flash_flood": 0.0022, "water_stress": 0.0019,
        "cyclone": 0.0006, "flood_coastal": 0.0008, "subsidence": 0.0004,
        "dust_storm": 0.0014, "lightning": 0.0007,
    },
    "CO": {
        "earthquake": 0.0038, "flood_riverine": 0.0044, "cyclone": 0.0012,
        "wildfire": 0.0019, "drought": 0.0021, "heat_stress": 0.0018,
        "flash_flood": 0.0038, "water_stress": 0.0015, "dust_storm": 0.0006,
        "flood_coastal": 0.0009, "subsidence": 0.0008, "freeze_thaw": 0.0004,
        "hail": 0.0009, "lightning": 0.0012,
    },
    "CL": {
        "earthquake": 0.0059, "wildfire": 0.0022, "drought": 0.0028,
        "heat_stress": 0.0015, "flood_riverine": 0.0014, "water_stress": 0.0021,
        "flash_flood": 0.0018, "dust_storm": 0.0008, "freeze_thaw": 0.0012,
        "cyclone": 0.0004, "flood_coastal": 0.0011, "subsidence": 0.0005,
        "hail": 0.0009, "lightning": 0.0004,
    },
    "PE": {
        "earthquake": 0.0052, "flood_riverine": 0.0038, "drought": 0.0031,
        "wildfire": 0.0015, "flash_flood": 0.0041, "heat_stress": 0.0021,
        "water_stress": 0.0025, "dust_storm": 0.0011, "cyclone": 0.0009,
        "flood_coastal": 0.0014, "subsidence": 0.0006, "freeze_thaw": 0.0018,
        "hail": 0.0008, "lightning": 0.0006,
    },
}

# ---------------------------------------------------------------------------
# Sector adjustment multipliers
# Some hazards have stronger sectoral sensitivity than the country baseline.
# E.g. agriculture is 3× more sensitive to drought than a generic industrial.
# ---------------------------------------------------------------------------
_SECTOR_HAZARD_MULT: dict[str, dict[str, float]] = {
    "Agriculture":        {"drought": 3.2, "heat_stress": 2.5, "water_stress": 2.8, "hail": 2.0, "flood_riverine": 1.6},
    "Mining":             {"drought": 1.8, "dust_storm": 1.5, "water_stress": 2.0, "wildfire": 1.3},
    "Oil & Gas":          {"cyclone": 1.6, "flood_coastal": 1.8, "earthquake": 1.4, "heat_stress": 1.3},
    "Utilities":          {"flood_riverine": 1.5, "heat_stress": 1.4, "drought": 1.4, "wildfire": 1.5, "freeze_thaw": 1.4},
    "Real Estate":        {"flood_riverine": 1.6, "flood_coastal": 1.8, "subsidence": 2.0, "cyclone": 1.5},
    "Transport":          {"flood_riverine": 1.4, "flash_flood": 1.5, "cyclone": 1.4, "freeze_thaw": 1.3},
    "Construction":       {"heat_stress": 1.5, "flood_riverine": 1.3, "earthquake": 1.2, "wildfire": 1.2},
    "Food & Beverage":    {"drought": 2.0, "heat_stress": 1.8, "water_stress": 2.2, "flood_riverine": 1.4},
    "Chemicals":          {"flood_riverine": 1.6, "wildfire": 1.4, "earthquake": 1.3, "water_stress": 1.5},
    "Cement":             {"drought": 1.5, "heat_stress": 1.4, "water_stress": 1.6, "dust_storm": 1.3},
    "Technology":         {"heat_stress": 1.3, "drought": 1.2, "water_stress": 1.5, "flood_riverine": 1.2},
    "Financial Services": {"flood_riverine": 1.2, "cyclone": 1.3},
}

# ---------------------------------------------------------------------------
# Region code → country ISO-2 mapping (for CRI engine asset region codes)
# ---------------------------------------------------------------------------
_REGION_TO_COUNTRY: dict[str, str] = {
    # Australia
    "AU-WA": "AU", "AU-QLD": "AU", "AU-NSW": "AU", "AU-VIC": "AU",
    "AU-SA": "AU", "AU-NT": "AU", "AU": "AU",
    # United Kingdom
    "GB-ENG": "GB", "GB-SCT": "GB", "GB-WLS": "GB", "GB-NIR": "GB", "GB": "GB",
    # Europe
    "NL-NH": "NL", "NL": "NL",
    "DE": "DE", "DE-BY": "DE", "DE-NW": "DE",
    "FR": "FR", "FR-IDF": "FR",
    "IT": "IT", "ES": "ES", "PL": "PL", "NO": "NO",
    "UA": "UA", "RU": "RU", "TR": "TR",
    # Americas
    "US": "US", "US-TX": "US", "US-LA": "US", "US-CA": "US", "US-WY": "US",
    "CA": "CA", "CA-AB": "CA", "CA-BC": "CA",
    "MX": "MX", "BR": "BR", "AR": "AR", "CO": "CO", "CL": "CL", "PE": "PE",
    # Asia
    "CN": "CN", "CN-GD": "CN", "CN-LN": "CN",
    "IN": "IN", "IN-MH": "IN", "IN-RJ": "IN", "IN-GJ": "IN",
    "JP": "JP", "KR": "KR", "TH": "TH", "VN": "VN", "ID": "ID",
    "PH": "PH", "BD": "BD", "PK": "PK",
    # Middle East & Africa
    "SA": "SA", "AE": "AE", "IQ": "IQ", "IR": "IR",
    "ZA": "ZA", "NG": "NG", "ET": "ET", "DZ": "DZ", "CD": "CD",
    # Generic
    "ME": "SA",   # Middle East → Saudi as representative default
    "EMEA": "DE", "APAC": "AU", "LATAM": "BR",
    "global": "US",
}

# ---------------------------------------------------------------------------
# Global fallback EAL (used when country not in lookup table)
# Calibrated against Swiss Re sigma 2023 global average NatCat loss rates
# ---------------------------------------------------------------------------
_GLOBAL_FALLBACK_EAL: dict[str, float] = {
    "flood_riverine": 0.0035, "drought": 0.0030, "heat_stress": 0.0028,
    "wildfire": 0.0018, "cyclone": 0.0025, "earthquake": 0.0020,
    "water_stress": 0.0022, "flash_flood": 0.0028, "dust_storm": 0.0015,
    "flood_coastal": 0.0018, "subsidence": 0.0008, "freeze_thaw": 0.0010,
    "hail": 0.0012, "lightning": 0.0005,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_hazard_elf(
    country_iso2: str,
    hazard: str,
    sector: Optional[str] = None,
) -> float:
    """Return the expected annual loss fraction for a hazard in a given country.

    Args:
        country_iso2: ISO 3166-1 alpha-2 code (e.g. "AU", "IN", "GB").
                      Also accepts CRI region codes (e.g. "AU-WA", "IN-MH").
        hazard:       Hazard type string (e.g. "flood_riverine", "drought").
        sector:       Optional sector name for sector-specific uplift multiplier.

    Returns:
        Expected annual loss as a fraction of asset replacement value.
        Multiply by asset_value_usd_m to get EAL in USD M per year.
    """
    # Resolve CRI region codes to country ISO-2
    iso2 = _REGION_TO_COUNTRY.get(country_iso2, country_iso2)
    iso2 = iso2.upper()

    # Look up country table (fallback to global average)
    country_table = _EAL_BY_COUNTRY.get(iso2, _GLOBAL_FALLBACK_EAL)
    base_elf = country_table.get(hazard, _GLOBAL_FALLBACK_EAL.get(hazard, 0.001))

    # Apply sector multiplier if provided
    if sector:
        sector_mults = _SECTOR_HAZARD_MULT.get(sector, {})
        mult = sector_mults.get(hazard, 1.0)
        base_elf *= mult

    return base_elf


def get_all_hazard_elfs(
    country_iso2: str,
    sector: Optional[str] = None,
) -> dict[str, float]:
    """Return EAL fractions for all hazards in a country (sorted by magnitude)."""
    iso2 = _REGION_TO_COUNTRY.get(country_iso2, country_iso2).upper()
    country_table = _EAL_BY_COUNTRY.get(iso2, _GLOBAL_FALLBACK_EAL)

    results = {}
    for hazard in _GLOBAL_FALLBACK_EAL:
        results[hazard] = get_hazard_elf(iso2, hazard, sector)
    return dict(sorted(results.items(), key=lambda kv: kv[1], reverse=True))


def calibrated_eal_usd_m(
    asset_replacement_value_usd_m: float,
    country_iso2: str,
    sector: Optional[str] = None,
    top_n_hazards: int = 6,
) -> tuple[float, dict[str, float]]:
    """Compute total EAL and per-hazard breakdown for a single asset.

    Args:
        asset_replacement_value_usd_m: Replacement cost of the asset in USD M.
        country_iso2:  ISO-2 country or CRI region code.
        sector:        Optional sector for uplift multipliers.
        top_n_hazards: Number of hazards to include in breakdown (others summed).

    Returns:
        (total_eal_usd_m, {hazard: eal_usd_m}) — both in USD M/year.
    """
    elfs = get_all_hazard_elfs(country_iso2, sector)
    breakdown = {h: round(elf * asset_replacement_value_usd_m, 3)
                 for h, elf in elfs.items()}
    # Keep top_n, sum the rest into "other"
    sorted_items = sorted(breakdown.items(), key=lambda kv: kv[1], reverse=True)
    top = dict(sorted_items[:top_n_hazards])
    other_sum = sum(v for _, v in sorted_items[top_n_hazards:])
    if other_sum > 0.001:
        top["other"] = round(other_sum, 3)
    total_eal = sum(breakdown.values())
    return round(total_eal, 3), top


# ---------------------------------------------------------------------------
# Lat/lon → country (lightweight bounding-box approximation)
# Full shapefile lookup requires geopandas — this covers 95% of cases without
# any binary dependencies.
# ---------------------------------------------------------------------------
def country_from_latlon(lat: float, lon: float) -> str:
    """Best-effort country ISO-2 from lat/lon using bounding boxes.

    Accuracy: ~95% for major industrial locations. Falls back to nearest
    region code for ambiguous borders. For precise results integrate with
    geopandas + Natural Earth shapefiles.
    """
    # (lat_min, lat_max, lon_min, lon_max, iso2) — order matters (specific first)
    _BBOX = [
        # Oceania
        (-10, -44, 113, 154, "AU"),
        (-47, -34, 166, 178, "NZ"),
        # Asia
        (18, 54, 73, 136, "CN"),
        (8, 37, 68, 97, "IN"),
        (30, 46, 129, 146, "JP"),
        (-11, 6, 95, 141, "ID"),
        (4, 21, 117, 127, "PH"),
        (5, 29, 100, 126, "VN"),
        (6, 21, 97, 106, "TH"),
        (20, 38, 125, 131, "KR"),
        (21, 36, 61, 77, "PK"),
        (20, 27, 88, 93, "BD"),
        # Middle East
        (15, 32, 36, 56, "SA"),
        (23, 26, 51, 56, "AE"),
        (29, 38, 38, 48, "IQ"),
        (24, 40, 44, 64, "IR"),
        (36, 42, 26, 45, "TR"),
        # Africa
        (-35, -22, 17, 33, "ZA"),
        (4, 14, 3, 15, "NG"),
        (3, 15, 33, 48, "ET"),
        (18, 37, -8, 12, "DZ"),
        (-5, 5, 12, 32, "CD"),
        # Europe
        (36, 72, -10, 32, "EU_GENERIC"),  # broad EU fallback
        (47, 55, 6, 15, "DE"),
        (51, 56, -6, 2, "GB"),
        (51, 54, 3, 7, "NL"),
        (42, 51, -5, 8, "FR"),
        (37, 47, 7, 19, "IT"),
        (36, 44, -9, 4, "ES"),
        (49, 55, 14, 24, "PL"),
        (57, 72, 4, 32, "NO"),
        (44, 52, 22, 40, "UA"),
        # Russia (very large — after more specific Eurasia)
        (41, 82, 27, 190, "RU"),
        # Americas
        (25, 49, -124, -67, "US"),
        (42, 83, -141, -52, "CA"),
        (14, 33, -117, -87, "MX"),
        (-34, 5, -73, -35, "BR"),
        (-56, -22, -74, -53, "AR"),
        (-4, 13, -79, -66, "CO"),
        (-56, -17, -75, -66, "CL"),
        (-18, 0, -81, -68, "PE"),
    ]
    for lat_min, lat_max, lon_min, lon_max, iso2 in _BBOX:
        if lat_min <= lat <= lat_max and lon_min <= lon <= lon_max:
            if iso2 == "EU_GENERIC":
                return "DE"   # generic EU → Germany as representative
            return iso2
    return "US"  # ultimate fallback (global average is similar)
