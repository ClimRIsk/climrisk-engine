"""
CRI Multi-Hazard Physical Risk Engine — v0.4 (25 hazards).

Derives 25 physical climate hazard scores for an asset location using:
  1. NASA NEX-GDDP-CMIP6 (heat stress, precipitation)
  2. WRI Aqueduct 4.0 (water stress, riverine/coastal flood, drought)
  3. IPCC AR6 / NOAA (sea level rise, coastal flood)
  4. SRTM/Copernicus DEM proxies (elevation → landslide, SLR exposure)
  5. IBTRACS + IPCC AR6 Ch11 (tropical cyclone)
  6. NASA FIRMS + IPCC AR6 (wildfire weather index)
  7. IPCC AR6 + FAO AQUASTAT (saltwater intrusion)
  8. LULC context from NASA MODIS/HLS proxy (land cover type)
  9. IEA Wind Task 19 + Rissanen 2022 (blade icing)
 10. IPCC AR6 Ch11 + NOAA (cold extremes, extratropical cyclone, hail, tornado)
 11. UNDRR DesInventar + IPCC (landslide, GLOF, avalanche, subsidence)
 12. NOAA / Copernicus MCS (flash flood, compound flood, lightning)
 13. Permafrost IPCC + AMAP (permafrost thaw, freeze-thaw cycle)
 14. WMO / ESA CCI (dust storm, marine heatwave)

All 25 hazards
--------------
 Core 10 (original):
  1. HEAT_STRESS         — extreme heat days, WBGT index, livestock/labour impact
  2. FLOOD_RIVERINE      — fluvial flooding (heavy precip + river systems)
  3. FLOOD_COASTAL       — storm surge + sea level rise + cyclone compound events
  4. SEA_LEVEL_RISE      — chronic inundation risk for coastal/low-elevation assets
  5. SALTWATER_INTRUSION — saline water table / aquifer intrusion (coastal + SLR)
  6. LANDSLIDE           — slope instability from extreme precip + terrain
  7. WILDFIRE            — fire weather index, vegetation fuel load
  8. CYCLONE             — tropical cyclone wind + surge (latitude-gated)
  9. DROUGHT             — meteorological + hydrological drought intensity
 10. WATER_STRESS        — chronic freshwater scarcity for operations

 New 15 (v0.4):
 11. EXTREME_COLD        — cold waves, frost damage, sub-zero operational disruption
 12. BLADE_ICING         — wind turbine / transmission line icing (freeze-thaw band)
 13. EXTRATROPICAL_CYCLONE — mid-latitude storm systems (wind, storm surge, flooding)
 14. FLASH_FLOOD         — rapid-onset flooding from intense convective precipitation
 15. PERMAFROST_THAW     — ground stability loss in Arctic/subarctic (infrastructure)
 16. DUST_STORM          — haboob / aeolian transport (equipment, visibility, health)
 17. HAIL                — large hail impact (crop damage, equipment, PV panels)
 18. LIGHTNING           — direct strikes + wildfire ignition from lightning
 19. SUBSIDENCE          — ground sinking from extraction, clay shrinkage, groundwater
 20. FREEZE_THAW_CYCLE   — repeated freeze-thaw: infrastructure cracking, pipe burst
 21. COMPOUND_FLOOD      — simultaneous riverine + coastal flooding (non-additive)
 22. AVALANCHE           — snow avalanche in mountain terrain
 23. MARINE_HEATWAVE     — anomalously warm ocean: marine operations, aquaculture
 24. GLOF                — glacial lake outburst flood (high-energy debris flows)
 25. TORNADO             — rotating convective winds (US/AU tornado-prone zones)

Output per hazard per year
--------------------------
  - annual_probability  (0–1): chance of a damaging event in that year
  - severity_index      (0–5): WRI-aligned severity scale
  - production_loss_pct (0–1): expected % of production lost due to this hazard
  - data_source         (str): which dataset drove this estimate

SSP scenarios
-------------
  SSP1-2.6 → NZE 2050
  SSP2-4.5 → Delayed Transition
  SSP3-7.0 → Current Policies
  SSP5-8.5 → Hot House (stress test)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

from .ssp_scenarios import (
    SSPScenario, SSP_SCENARIOS, NGFS_TO_SSP,
    HEAT_FREQ_PER_DEG_C, PRECIP_CHANGE_PCT_PER_DEG_C,
    WILDFIRE_INDEX_PCT_PER_DEG_C, CYCLONE_INTENSITY_PCT_PER_DEG_C,
    ngfs_to_ssp,
)
from .met_data import (
    get_met_baseline,
    heat_stress_baseline_multiplier,
    precip_variability_index,
    observed_cyclone_prob,
    observed_drought_return,
)

# Lazy imports for live API connectors — only loaded when coordinates provided
def _get_nasa_power():
    try:
        from ..connectors.nasa_power import get_baseline
        return get_baseline
    except Exception:
        return None

def _get_open_meteo():
    try:
        from ..connectors.open_meteo_climate import get_projection, compute_warming_delta
        return get_projection, compute_warming_delta
    except Exception:
        return None, None


# ---------------------------------------------------------------------------
# WRI Aqueduct 4.0 baseline scores by region (0–5 scale)
# Source: WRI Aqueduct 4.0 (2023) — aggregated to CRI region codes
# 0 = Low / 5 = Extremely High
# ---------------------------------------------------------------------------

WRI_BASELINE: dict[str, dict[str, float]] = {
    "AU-WA":  {"water_stress": 2.2, "riverine_flood": 1.6, "coastal_flood": 1.4,
               "drought": 3.3, "heat_baseline": 2.8},
    "AU-QLD": {"water_stress": 1.8, "riverine_flood": 2.4, "coastal_flood": 2.1,
               "drought": 2.1, "heat_baseline": 3.2},
    "AU-SA":  {"water_stress": 3.5, "riverine_flood": 0.8, "coastal_flood": 0.7,
               "drought": 4.1, "heat_baseline": 3.5},
    "AU-NSW": {"water_stress": 2.0, "riverine_flood": 2.1, "coastal_flood": 1.8,
               "drought": 2.5, "heat_baseline": 2.5},
    "AU-VIC": {"water_stress": 1.5, "riverine_flood": 1.4, "coastal_flood": 1.2,
               "drought": 1.8, "heat_baseline": 2.0},
    "AU-NT":  {"water_stress": 1.2, "riverine_flood": 2.8, "coastal_flood": 1.5,
               "drought": 1.5, "heat_baseline": 4.0},
    "US-TX":  {"water_stress": 3.2, "riverine_flood": 2.8, "coastal_flood": 2.5,
               "drought": 2.9, "heat_baseline": 2.6},
    "US-WY":  {"water_stress": 2.8, "riverine_flood": 1.2, "coastal_flood": 0.0,
               "drought": 3.5, "heat_baseline": 2.0},
    "US-OK":  {"water_stress": 3.0, "riverine_flood": 2.6, "coastal_flood": 0.5,
               "drought": 3.2, "heat_baseline": 2.5},
    "CA-AB":  {"water_stress": 1.2, "riverine_flood": 1.0, "coastal_flood": 0.0,
               "drought": 1.5, "heat_baseline": 1.5},
    "CA-QC":  {"water_stress": 0.5, "riverine_flood": 1.2, "coastal_flood": 0.8,
               "drought": 0.8, "heat_baseline": 1.0},
    "GB-ENG": {"water_stress": 1.0, "riverine_flood": 2.0, "coastal_flood": 1.8,
               "drought": 1.2, "heat_baseline": 1.2},
    "NL-NH":  {"water_stress": 0.8, "riverine_flood": 2.5, "coastal_flood": 3.5,
               "drought": 0.9, "heat_baseline": 1.0},
    "ZA":     {"water_stress": 3.8, "riverine_flood": 1.4, "coastal_flood": 1.0,
               "drought": 4.2, "heat_baseline": 3.0},
    "CL-02":  {"water_stress": 4.2, "riverine_flood": 0.6, "coastal_flood": 0.8,
               "drought": 4.5, "heat_baseline": 2.2},
    "PE-01":  {"water_stress": 3.5, "riverine_flood": 1.8, "coastal_flood": 1.5,
               "drought": 3.0, "heat_baseline": 2.0},
    "BR-PA":  {"water_stress": 0.8, "riverine_flood": 3.5, "coastal_flood": 1.5,
               "drought": 1.5, "heat_baseline": 3.2},
    "ID-KI":  {"water_stress": 1.5, "riverine_flood": 3.2, "coastal_flood": 3.0,
               "drought": 1.2, "heat_baseline": 3.5},
    "CN-NM":  {"water_stress": 3.8, "riverine_flood": 1.0, "coastal_flood": 0.0,
               "drought": 3.5, "heat_baseline": 2.8},
    "IN-MH":  {"water_stress": 3.5, "riverine_flood": 2.8, "coastal_flood": 2.5,
               "drought": 3.0, "heat_baseline": 3.8},
    "MN-01":  {"water_stress": 3.5, "riverine_flood": 0.8, "coastal_flood": 0.0,
               "drought": 4.0, "heat_baseline": 2.8},
    # ── India states — WRI Aqueduct 4.0 (2023) sub-national India layer ──────
    "IN-JH":  {"water_stress": 2.5, "riverine_flood": 3.5, "coastal_flood": 0.0,
               "drought": 1.8, "heat_baseline": 3.5},   # Jharkhand (Jamshedpur)
    "IN-OD":  {"water_stress": 2.0, "riverine_flood": 3.8, "coastal_flood": 2.5,
               "drought": 2.0, "heat_baseline": 3.8},   # Odisha (Kalinganagar)
    "IN-RJ":  {"water_stress": 4.5, "riverine_flood": 0.8, "coastal_flood": 0.0,
               "drought": 4.2, "heat_baseline": 4.2},   # Rajasthan (Aditya Works)
    "IN-GJ":  {"water_stress": 3.8, "riverine_flood": 2.0, "coastal_flood": 2.5,
               "drought": 3.5, "heat_baseline": 4.0},   # Gujarat
    "IN-AP":  {"water_stress": 3.0, "riverine_flood": 2.5, "coastal_flood": 2.0,
               "drought": 3.2, "heat_baseline": 3.8},   # Andhra Pradesh
    "IN-KA":  {"water_stress": 3.2, "riverine_flood": 1.8, "coastal_flood": 0.5,
               "drought": 3.0, "heat_baseline": 3.5},   # Karnataka
    "IN-MP":  {"water_stress": 2.8, "riverine_flood": 2.0, "coastal_flood": 0.0,
               "drought": 3.0, "heat_baseline": 3.8},   # Madhya Pradesh
    # ── United Kingdom ────────────────────────────────────────────────────────
    "GB-WLS": {"water_stress": 0.6, "riverine_flood": 2.2, "coastal_flood": 2.0,
               "drought": 0.8, "heat_baseline": 0.8},   # Wales (Port Talbot)
    "GB-SCT": {"water_stress": 0.4, "riverine_flood": 2.0, "coastal_flood": 1.8,
               "drought": 0.5, "heat_baseline": 0.6},   # Scotland
    # ── Continental Europe ────────────────────────────────────────────────────
    "DE-HH":  {"water_stress": 0.8, "riverine_flood": 2.3, "coastal_flood": 2.5,
               "drought": 0.8, "heat_baseline": 0.9},   # Hamburg (tidal Elbe, North Sea)
    "ES-CT":  {"water_stress": 3.2, "riverine_flood": 1.0, "coastal_flood": 1.5,
               "drought": 3.5, "heat_baseline": 2.5},   # Catalonia / Barcelona
    "FR-IDF": {"water_stress": 1.2, "riverine_flood": 1.8, "coastal_flood": 0.5,
               "drought": 1.5, "heat_baseline": 1.5},   # Île-de-France
    # ── United States (additional states) ────────────────────────────────────
    "US-FL":  {"water_stress": 1.5, "riverine_flood": 2.5, "coastal_flood": 3.5,
               "drought": 1.8, "heat_baseline": 2.8},   # Florida (Miami; hurricane/SLR)
    "US-GA":  {"water_stress": 1.8, "riverine_flood": 2.2, "coastal_flood": 1.5,
               "drought": 2.0, "heat_baseline": 2.2},   # Georgia (Atlanta)
    "US-NY":  {"water_stress": 1.0, "riverine_flood": 2.0, "coastal_flood": 2.5,
               "drought": 1.0, "heat_baseline": 1.5},   # New York (JFK/LGA; coastal flood)
    "US-CA":  {"water_stress": 3.8, "riverine_flood": 1.2, "coastal_flood": 1.5,
               "drought": 3.8, "heat_baseline": 2.5},   # California (LAX; water/wildfire)
    "US-MN":  {"water_stress": 1.2, "riverine_flood": 1.8, "coastal_flood": 0.0,
               "drought": 1.5, "heat_baseline": 1.5},   # Minnesota (MSP)
    "US-WA":  {"water_stress": 1.0, "riverine_flood": 1.8, "coastal_flood": 1.2,
               "drought": 1.5, "heat_baseline": 1.5},   # Washington state (SEA)
    "global": {"water_stress": 2.0, "riverine_flood": 1.5, "coastal_flood": 1.0,
               "drought": 2.0, "heat_baseline": 2.0},
}


# ---------------------------------------------------------------------------
# Coastal proximity lookup (regions with significant coastal assets)
# True = coastal; drives SLR, saltwater intrusion, cyclone exposure
# ---------------------------------------------------------------------------

COASTAL_REGIONS: set[str] = {
    "AU-WA", "AU-QLD", "AU-NSW", "AU-VIC", "AU-NT",
    "GB-ENG", "NL-NH", "US-TX", "US-OK",
    "IN-MH", "ID-KI", "BR-PA", "PE-01", "ZA",
    "CL-02",    # Atacama coast
    # Added for demo company coverage
    "GB-WLS",               # Port Talbot — Bristol Channel / Irish Sea coast
    "DE-HH",                # Hamburg — tidal Elbe estuary, North Sea
    "US-FL",                # Miami — Atlantic/Gulf coast, hurricane exposure
    "US-NY",                # JFK/LGA — Long Island Sound / Atlantic coast
    "US-CA",                # LAX — Pacific coast
    "IN-OD",                # Odisha (Kalinganagar) — Bay of Bengal coast
    "IN-GJ",                # Gujarat — Arabian Sea coast
    "IN-AP",                # Andhra Pradesh — Bay of Bengal coast
    "ES-CT",                # Barcelona — Mediterranean coast
}

# Cyclone-susceptible latitude bands (approx 5°–25° N/S)
CYCLONE_REGIONS: set[str] = {
    "AU-QLD", "AU-WA", "AU-NT",
    "IN-MH", "ID-KI", "BR-PA",
    "US-TX",   # Gulf hurricanes
    "ZA",      # Southern Indian Ocean
    "US-FL",   # Florida / Atlantic + Gulf hurricane corridor
    "IN-OD",   # Odisha — Bay of Bengal cyclone landfall (e.g. Fani 2019, Phailin 2013)
    "IN-AP",   # Andhra Pradesh — Bay of Bengal cyclone exposure
}

# Landslide-prone regions (steep terrain, high precip)
LANDSLIDE_REGIONS: set[str] = {
    "CL-02",   # Andes
    "PE-01",   # Andes
    "ID-KI",   # Kalimantan — tropical slopes
    "BR-PA",   # Amazon escarpment
    "IN-MH",   # Western Ghats
    "AU-NSW",  # Blue Mountains escarpment
    "MN-01",   # Mongolian highlands
}

# Wildfire-high-risk (IPCC AR6 — fire weather index, elevated significantly by 2050)
WILDFIRE_REGIONS: set[str] = {
    "AU-WA", "AU-SA", "AU-QLD", "AU-NSW", "AU-VIC",
    "US-TX", "US-WY", "US-OK",
    "ZA", "CL-02",
    "CN-NM",   # Inner Mongolia grassland fires
    "MN-01",   # Mongolian steppe fires
    "US-CA",   # California — among highest wildfire weather index globally (IPCC AR6 Ch12)
    "IN-RJ",   # Rajasthan — dryland fire and Thar Desert fringe events
}

# ---------------------------------------------------------------------------
# Additional region sets for v0.4 expanded hazard library
# ---------------------------------------------------------------------------

# Extreme cold / severe frost (cold winters → operational disruption)
COLD_REGIONS: set[str] = {
    "CA-AB", "CA-QC",   # Canadian prairies / Quebec
    "MN-01",             # Mongolia (continental extreme cold)
    "CN-NM",             # Inner Mongolia
    "US-WY",             # Wyoming highlands
}

# Permafrost ground coverage (continuous / discontinuous permafrost zones)
# Source: IPA Global Permafrost Zonation Map; AMAP 2021
PERMAFROST_REGIONS: set[str] = {
    "CA-QC",   # Northern Quebec — widespread discontinuous permafrost
    "CA-AB",   # Northern Alberta fringe — sporadic permafrost
    "MN-01",   # Siberian-adjacent Mongolia — widespread permafrost
}

# Dust-storm-prone regions (semi-arid, arid with high wind erosion potential)
# Source: WMO SDS-WAS; Ginoux et al 2012 MODIS dust source inventory
DUST_STORM_REGIONS: set[str] = {
    "AU-SA", "AU-WA", "AU-NT",   # Australian interior red-dust events
    "MN-01",                      # Mongolian / Gobi dust — dzud corollary
    "CN-NM",                      # Inner Mongolia Gobi dust transport
    "ZA",                         # Karoo / Namib dust events
    "CL-02",                      # Atacama aeolian transport
    "US-TX", "US-OK",             # Texas/Oklahoma Dust Bowl analog
    "IN-MH",                      # Thar desert fringe dust
    "IN-RJ",                      # Rajasthan — core Thar Desert dust source region
}

# Hail-prone regions (large convective hail; supercell storms)
# Source: IPCC AR6 Ch11.7; Prein & Holland 2018; AONB / NOAA SPC
HAIL_REGIONS: set[str] = {
    "AU-QLD", "AU-NSW", "AU-VIC",   # Australian Hail Belt (SE Australia)
    "US-TX", "US-OK", "US-WY",      # Great Plains Hail Alley
    "ZA",                            # South Africa highveld (summer convection)
    "IN-MH",                         # Indian subcontinent pre-monsoon hail
    "BR-PA",                         # Amazonian MCS convective hail
}

# Extratropical cyclone (mid-latitude storm) exposure
# Source: IPCC AR6 Ch11.7; Zappa et al 2013; Mölter et al 2016
EXTRATROPICAL_CYCLONE_REGIONS: set[str] = {
    "GB-ENG", "NL-NH",         # North-Atlantic storm track
    "CA-QC",                    # Canadian Maritime / Nor'easter
    "AU-VIC", "AU-NSW",         # Southern Ocean extra-tropical storm track
    "ZA",                       # Cape Town extratropical cyclones
    "US-WY",                    # Northern Plains blizzard/extratropical systems
    "CL-02",                    # Chilean extratropical low-pressure systems
    "GB-WLS",                   # Wales — Bristol Channel, Irish Sea storm track
    "DE-HH",                    # Hamburg — North Sea extratropical cyclones
    "US-NY",                    # New York — Nor'easters / Atlantic storm track
    "US-MN",                    # Minnesota — mid-latitude winter cyclones / blizzards
}

# Tornado-prone regions (rotating convective mesocyclones)
# Source: NOAA SPC; Allen & Tippett 2015; Geerts 2018
TORNADO_REGIONS: set[str] = {
    "US-TX", "US-OK",    # US Tornado Alley — world's highest tornado frequency
    "AU-QLD",             # Australian tornado events (significantly lower freq than US)
}

# Avalanche-prone (steep terrain + seasonal snowpack)
# Source: UNDRR DesInventar; SLF / WSL Avalanche Atlas
AVALANCHE_REGIONS: set[str] = {
    "CL-02",   # Andes (operational mining routes)
    "PE-01",   # Tropical Andes (glaciated terrain)
    "CA-AB",   # Canadian Rockies
    "US-WY",   # Wyoming / Rocky Mountain cordillera
    "MN-01",   # Mongolian Altai mountains
}

# GLOF (Glacial Lake Outburst Flood) — glacier retreat accelerates risk
# Source: IPCC SROCC Ch2; Emmer et al 2022; Veh et al 2020
GLOF_REGIONS: set[str] = {
    "CL-02",   # Andean glaciers (rapidly retreating)
    "PE-01",   # Tropical Andes — GLOF events documented (Huaraz 1941 etc.)
    "CA-AB",   # Rocky Mountain glaciers (Columbia Icefield drainage)
    "IN-MH",   # Himalayan fringes (downstream flooding)
    "MN-01",   # Altai mountains (Mongolian glacial lakes)
}

# Marine heatwave (anomalously warm sea surface temperature)
# Source: Oliver et al 2021; Hobday et al 2018; NOAA CoralWatch
MARINE_HEATWAVE_REGIONS: set[str] = {
    "AU-WA",    # SE Indian Ocean — most intense MHW hotspot globally
    "AU-QLD",   # Coral Sea / Great Barrier Reef bleaching
    "AU-SA",    # Southern Ocean / Spencer Gulf
    "ZA",       # SE Atlantic / Agulhas current
    "ID-KI",    # Coral Triangle — thermally sensitive
    "IN-MH",    # Arabian Sea MHW events
    "GB-ENG",   # North Sea anomalous warming
}

# Subsidence-prone (ground sinking: extraction, peat/clay, groundwater depletion)
# Source: Cigna & Tapete 2021; WRI; InSAR monitoring datasets
SUBSIDENCE_REGIONS: set[str] = {
    "NL-NH",    # Peat soils + below sea-level (worst in world: -10 cm/yr)
    "ID-KI",    # Borneo peat + Jakarta groundwater depletion
    "CN-NM",    # Coal extraction subsidence (Inner Mongolia coalfield)
    "AU-QLD",   # Queensland coal mine subsidence
    "IN-MH",    # Groundwater extraction depletion (Marathwada)
    "MN-01",    # Mongolian mining operations
    "US-TX",    # Texas oil/gas extraction (Houston subsidence)
    "US-OK",    # Oklahoma induced seismicity + subsidence
}


# ---------------------------------------------------------------------------
# Elevation proxy (metres above sea level — used when no DEM available)
# Source: SRTM/Copernicus region-average elevation
# ---------------------------------------------------------------------------

REGION_ELEVATION_M: dict[str, float] = {
    "AU-WA": 400, "AU-QLD": 200, "AU-SA": 180, "AU-NSW": 350,
    "AU-VIC": 240, "AU-NT": 200,
    "US-TX": 150, "US-WY": 2000, "US-OK": 380,
    "CA-AB": 700, "CA-QC": 300,
    "GB-ENG": 90, "NL-NH": 2,    # Below sea level in parts!
    "ZA": 1200, "CL-02": 2500,
    "PE-01": 3000, "BR-PA": 50,
    "ID-KI": 80, "CN-NM": 1000, "IN-MH": 600,
    "MN-01": 1500,
    # India states
    "IN-JH": 180,   # Jamshedpur / Jharkhand
    "IN-OD": 100,   # Kalinganagar / Odisha coastal
    "IN-RJ": 400,   # Rajasthan plateau
    "IN-GJ": 50,    # Gujarat coastal plain
    "IN-AP": 100,   # Andhra Pradesh coastal
    "IN-KA": 800,   # Karnataka Deccan Plateau
    "IN-MP": 450,   # Madhya Pradesh plateau
    # UK
    "GB-WLS": 80,   # Port Talbot / South Wales coast
    "GB-SCT": 150,  # Scotland
    # Continental Europe
    "DE-HH": 10,    # Hamburg (tidal Elbe, near sea level)
    "ES-CT": 50,    # Barcelona / Catalonia coast
    "FR-IDF": 100,  # Île-de-France / Paris basin
    # US additional states
    "US-FL": 5,     # Miami (essentially at sea level)
    "US-GA": 320,   # Atlanta
    "US-NY": 30,    # New York / JFK area
    "US-CA": 90,    # LA basin
    "US-MN": 290,   # Minneapolis
    "US-WA": 120,   # Seattle
    "global": 300,
}


# ---------------------------------------------------------------------------
# LULC land cover context (NASA MODIS/HLS proxy by region)
# Drives wildfire fuel load, flood runoff, heat island effect
# ---------------------------------------------------------------------------

class LULCType:
    FOREST        = "forest"
    GRASSLAND     = "grassland"
    SHRUBLAND     = "shrubland"
    CROPLAND      = "cropland"
    WETLAND       = "wetland"
    URBAN         = "urban"
    BARE_ROCK     = "bare_rock"
    WATER         = "water"
    SNOW_ICE      = "snow_ice"


REGION_LULC: dict[str, str] = {
    "AU-WA": LULCType.SHRUBLAND,
    "AU-QLD": LULCType.GRASSLAND,
    "AU-SA": LULCType.BARE_ROCK,
    "AU-NSW": LULCType.FOREST,
    "AU-VIC": LULCType.FOREST,
    "AU-NT": LULCType.GRASSLAND,
    "US-TX": LULCType.GRASSLAND,
    "US-WY": LULCType.FOREST,
    "US-OK": LULCType.CROPLAND,
    "CA-AB": LULCType.FOREST,
    "CA-QC": LULCType.FOREST,
    "GB-ENG": LULCType.CROPLAND,
    "NL-NH": LULCType.WETLAND,
    "ZA": LULCType.SHRUBLAND,
    "CL-02": LULCType.BARE_ROCK,
    "PE-01": LULCType.FOREST,
    "BR-PA": LULCType.FOREST,
    "ID-KI": LULCType.FOREST,
    "CN-NM": LULCType.GRASSLAND,
    "IN-MH": LULCType.CROPLAND,
    "MN-01": LULCType.GRASSLAND,
    # India states
    "IN-JH": LULCType.FOREST,     # Jharkhand — significant forest cover
    "IN-OD": LULCType.FOREST,     # Odisha — forest + cropland
    "IN-RJ": LULCType.SHRUBLAND,  # Rajasthan — semi-arid / Thar desert scrub
    "IN-GJ": LULCType.CROPLAND,   # Gujarat — predominantly agricultural
    "IN-AP": LULCType.CROPLAND,   # Andhra Pradesh — agricultural
    "IN-KA": LULCType.CROPLAND,   # Karnataka — Deccan cropland
    "IN-MP": LULCType.FOREST,     # Madhya Pradesh — central Indian forests
    # UK / Europe
    "GB-WLS": LULCType.GRASSLAND, # Wales — upland grassland / moorland
    "GB-SCT": LULCType.FOREST,    # Scotland — boreal / upland
    "DE-HH": LULCType.URBAN,      # Hamburg — major port city
    "ES-CT": LULCType.CROPLAND,   # Catalonia — Mediterranean agriculture
    "FR-IDF": LULCType.URBAN,     # Île-de-France — Paris metropolitan
    # US states
    "US-FL": LULCType.WETLAND,    # Florida — mangroves, wetlands, coastal
    "US-GA": LULCType.FOREST,     # Georgia — SE pine forest
    "US-NY": LULCType.URBAN,      # New York — metropolitan
    "US-CA": LULCType.SHRUBLAND,  # California — chaparral / mixed
    "US-MN": LULCType.FOREST,     # Minnesota — boreal / deciduous forest
    "US-WA": LULCType.FOREST,     # Washington — Pacific Northwest forest
    "global": LULCType.GRASSLAND,
}

# Wildfire fuel load multiplier by LULC
WILDFIRE_FUEL_BY_LULC: dict[str, float] = {
    LULCType.FOREST:    1.8,
    LULCType.GRASSLAND: 1.2,
    LULCType.SHRUBLAND: 1.5,
    LULCType.CROPLAND:  0.9,
    LULCType.WETLAND:   0.4,
    LULCType.URBAN:     0.6,
    LULCType.BARE_ROCK: 0.1,
    LULCType.WATER:     0.0,
    LULCType.SNOW_ICE:  0.0,
}

# Surface runoff multiplier (affects flood risk)
RUNOFF_BY_LULC: dict[str, float] = {
    LULCType.BARE_ROCK: 2.0,
    LULCType.URBAN:     1.8,
    LULCType.CROPLAND:  1.3,
    LULCType.GRASSLAND: 1.0,
    LULCType.SHRUBLAND: 0.9,
    LULCType.FOREST:    0.7,
    LULCType.WETLAND:   0.5,
    LULCType.WATER:     0.0,
    LULCType.SNOW_ICE:  1.5,
}


# ---------------------------------------------------------------------------
# Downscaling: spatial context resolution from lat/lon
# ---------------------------------------------------------------------------

# Reference coastal points: (lat, lon) for major coastlines worldwide.
# Used to compute distance-to-coast for continuous coastal exposure scoring.
# Source: Natural Earth coastline centroids (simplified, ~200km spacing).
_COASTAL_REFERENCE_POINTS: list[Tuple[float, float]] = [
    # North Sea / Atlantic Europe
    (51.9, 4.5), (53.5, 8.0), (55.7, 12.5), (58.0, 7.5),
    # British Isles
    (51.5, 1.0), (53.0, -4.5), (57.5, -2.5),
    # Bay of Biscay / Iberia
    (43.5, -2.0), (38.5, -9.0), (36.5, -6.0),
    # Mediterranean
    (43.3, 5.3), (41.0, 14.0), (37.9, 23.7), (35.5, 33.0),
    # West Africa
    (5.3, -4.0), (14.7, -17.5), (-8.8, 13.2),
    # Southern Africa
    (-33.9, 18.4), (-29.9, 31.0),
    # East Africa / Indian Ocean
    (-4.0, 39.7), (11.6, 43.1),
    # Arabian Peninsula / Persian Gulf
    (24.5, 56.4), (22.6, 59.5), (21.5, 39.2),
    # South Asia
    (8.5, 77.0), (13.1, 80.3), (18.9, 72.8), (22.6, 88.4), (23.7, 90.4),
    # Southeast Asia
    (1.3, 103.8), (3.1, 101.7), (-6.2, 106.8), (13.7, 100.5),
    # East Asia
    (22.5, 114.2), (30.7, 121.5), (35.7, 139.7), (34.7, 135.2),
    # Australia
    (-31.9, 115.9), (-27.5, 153.0), (-37.8, 144.9), (-33.9, 151.2),
    (-12.5, 130.8), (-34.9, 138.6),
    # North America East
    (25.8, -80.2), (29.9, -90.1), (37.8, -75.5), (42.4, -70.9),
    (44.6, -63.6), (46.8, -71.2),
    # North America West
    (34.0, -118.5), (37.8, -122.4), (47.6, -122.3), (58.3, -134.4),
    # Gulf of Mexico / Caribbean
    (23.1, -82.3), (17.9, -76.8), (10.5, -61.4), (19.4, -99.1),
    # South America
    (-23.0, -43.2), (-34.6, -58.4), (-33.0, -71.6), (-8.1, -34.9),
    (-3.7, -38.5), (10.5, -66.9),
    # Arctic / Nordic coasts
    (70.0, 25.0), (69.0, 18.0), (64.1, -21.9), (59.9, 10.7),
]


def _fetch_live_climate(
    lat: float,
    lon: float,
    year: int,
    ssp_id: str,
) -> dict:
    """
    Fetch real climate data from NASA POWER (baseline) and Open-Meteo (projection).

    Delta-downscaling approach — three data points, two APIs:
      1. NASA POWER MERRA-2 (2001-2020): observed climatological baseline at coordinates
         (T2M_MAX, precip) — what the asset experiences today
      2. Open-Meteo CMIP6 historical (year 2010): same model, historical run
         — establishes the model's own baseline (removes GCM systematic bias)
      3. Open-Meteo CMIP6 future (target year): same model, SSP5-8.5 run
         — future projection

    Warming delta = CMIP6_future_T2M_MAX − CMIP6_historical_T2M_MAX (model-consistent)
    This delta is then added to the NASA POWER observed baseline for the asset-level
    projected temperature. SSP scaling applied via IPCC AR6 WG1 Table 4.5.

    Returns a dict with:
      - warming_c: CMIP6 model-consistent warming delta, SSP-scaled (°C)
      - precip_delta_pct: CMIP6 model-consistent precipitation change, SSP-scaled (%)
      - baseline_t2m_max_c: observed T2M_MAX from NASA POWER (2001-2020)
      - baseline_precip_mm_day: observed precipitation from NASA POWER
      - data_source: audit trail description
      - live: True if full pipeline succeeded, False if fallback used

    Falls back gracefully — calling code uses lookup-table path on any failure.
    """
    result = {
        "warming_c": None,
        "precip_delta_pct": None,
        "baseline_t2m_max_c": None,
        "baseline_precip_mm_day": None,
        "data_source": "fallback (API unavailable)",
        "live": False,
    }

    # ── Step 1: NASA POWER observed baseline ─────────────────────────────────
    get_baseline = _get_nasa_power()
    if get_baseline is None:
        return result
    baseline = get_baseline(lat, lon)
    if baseline is None:
        return result

    result["baseline_t2m_max_c"] = baseline.t2m_max_c
    result["baseline_precip_mm_day"] = baseline.precip_mm_day

    # ── Step 2 + 3: Open-Meteo CMIP6 historical + future ─────────────────────
    get_projection, compute_warming_delta_fn = _get_open_meteo()
    if get_projection is None:
        result["data_source"] = "NASA POWER baseline only (Open-Meteo unavailable)"
        return result

    # Historical reference year: centre of 2001-2020 NASA POWER baseline period.
    # Using year 2010 gives the best-available CMIP6 historical run overlap.
    _CMIP6_HIST_YEAR = 2010
    cmip6_hist = get_projection(lat, lon, _CMIP6_HIST_YEAR)
    if cmip6_hist is None:
        result["data_source"] = f"NASA POWER baseline only (CMIP6 historical run unavailable for {_CMIP6_HIST_YEAR})"
        return result

    cmip6_future = get_projection(lat, lon, year)
    if cmip6_future is None:
        result["data_source"] = f"NASA POWER baseline only (CMIP6 future projection unavailable for {year})"
        return result

    # ── Step 4: Model-consistent delta + SSP scaling ──────────────────────────
    delta = compute_warming_delta_fn(
        cmip6_hist,
        cmip6_future,
        ssp_id,
        year,
        observed_precip_mm_day=baseline.precip_mm_day,
    )
    result["warming_c"] = delta["warming_c"]
    result["precip_delta_pct"] = delta.get("precip_delta_pct")
    result["data_source"] = delta["source"]
    result["live"] = True

    return result


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two lat/lon points in km."""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    return R * 2 * math.asin(math.sqrt(a))


def _subgrid_elevation_correction(asset_elev_m: float, region_mean_elev_m: float) -> dict:
    """
    Sub-grid correction factors derived from asset elevation relative to the
    regional (25km grid cell) mean elevation.

    The WRI Aqueduct 4.0 and NASA NEX-GDDP-CMIP6 baselines represent 0.25°
    (~25km) grid cell averages. Within a cell, hazard exposure varies sharply
    with terrain. This function computes multiplicative correction factors that
    bring the grid-cell average down (or up) to the specific asset elevation.

    Corrections applied:

    FLOOD (riverine):
      The 25km grid value integrates floodplain and hillslope. Assets above the
      regional mean sit on higher terrain and have lower flood exposure.
      Source: LISFLOOD sub-grid DEM analysis; WRI Aqueduct 4.0 validation report.
        elev_delta < -50m  (below regional mean — floodplain): 1.25×
        elev_delta 0–100m : 1.00× (at or near regional mean)
        elev_delta 100–300m: linear decay 1.00 → 0.65
        elev_delta 300–600m: linear decay 0.65 → 0.35
        elev_delta > 600m  : 0.20× (highland — negligible riverine flood)

    HEAT (surface temperature):
      Environmental lapse rate: ~0.65°C per 100m elevation gain reduces the
      effective surface temperature and heat stress experienced at the asset.
      Source: IPCC AR6 WG1 Ch2; WMO standard atmosphere lapse rate.
        Correction = exp(-0.0025 × elev_delta) where elev_delta = asset - regional mean
        Equivalent to −0.25% heat severity per 10m above regional mean.

    WATER STRESS:
      High-elevation assets are often above river catchment stress zones and
      may have access to fresher highland water. Reduction of 15% per 500m above
      regional mean (capped at 40% reduction).
      Source: FAO AQUASTAT; WRI Aqueduct 4.0 sub-basin analysis.

    Returns:
        dict with keys: flood_factor, heat_factor, water_stress_factor
    """
    elev_delta = asset_elev_m - region_mean_elev_m

    # ── Flood correction ──────────────────────────────────────────────────────
    if elev_delta < -50:
        flood_f = 1.25   # below regional mean — in floodplain
    elif elev_delta < 100:
        flood_f = 1.00
    elif elev_delta < 300:
        flood_f = 1.00 - 0.35 * (elev_delta - 100) / 200   # 1.00 → 0.65
    elif elev_delta < 600:
        flood_f = 0.65 - 0.30 * (elev_delta - 300) / 300   # 0.65 → 0.35
    else:
        flood_f = 0.20

    # ── Heat correction (lapse rate) ──────────────────────────────────────────
    # Every 100m above regional mean ≈ −0.65°C → reduces heat stress
    heat_f = math.exp(-0.0025 * max(0, elev_delta))   # no uplift for below-mean

    # ── Water stress correction ───────────────────────────────────────────────
    ws_reduction = min(0.40, max(0.0, elev_delta / 500 * 0.15))
    water_stress_f = 1.0 - ws_reduction

    return {
        "flood_factor":        round(max(0.05, flood_f), 4),
        "heat_factor":         round(max(0.30, heat_f), 4),
        "water_stress_factor": round(water_stress_f, 4),
        "elev_delta_m":        round(elev_delta, 1),
    }


def _coastal_proximity_factor(lat: float, lon: float) -> float:
    """
    Continuous coastal proximity factor (0–1) based on distance to nearest
    reference coastal point.

    Decay function:
      < 10 km  → 1.00  (directly coastal)
      10–50 km → linear decay 1.0 → 0.85
      50–150km → linear decay 0.85 → 0.40
      150–300km→ linear decay 0.40 → 0.10
      > 300 km → 0.05  (minimal coastal influence)

    Source: Analogous to CLIMADA coastal exposure decay; consistent with
    WRI Aqueduct coastal flood exposure gradient methodology.
    """
    if lat is None or lon is None:
        return 0.5   # unknown — use moderate default

    min_dist = min(
        _haversine_km(lat, lon, cp[0], cp[1])
        for cp in _COASTAL_REFERENCE_POINTS
    )

    if min_dist < 10:
        return 1.00
    elif min_dist < 50:
        return 1.00 - 0.15 * (min_dist - 10) / 40
    elif min_dist < 150:
        return 0.85 - 0.45 * (min_dist - 50) / 100
    elif min_dist < 300:
        return 0.40 - 0.30 * (min_dist - 150) / 150
    else:
        return 0.05


def _resolve_spatial_context(
    lat: Optional[float],
    lon: Optional[float],
    region: str,
    elevation_override: Optional[float],
) -> dict:
    """
    Derive asset-level spatial context from coordinates.

    Returns a dict with:
      - elevation_m: refined elevation estimate
      - is_coastal: boolean (True if coastal_factor >= 0.40)
      - coastal_factor: continuous 0–1 proximity score
      - lat_effective: lat used (falls back to region centroid if None)
      - lon_effective: lon used
      - spatial_resolution: human-readable resolution string for audit trail
      - downscaling_method: methodology description
    """
    # Region centroids for fallback when no coordinates given
    _REGION_CENTROIDS: dict[str, Tuple[float, float]] = {
        "AU-WA": (-25.0, 122.0), "AU-QLD": (-22.0, 144.0), "AU-SA": (-30.0, 135.0),
        "AU-NSW": (-33.0, 146.0), "AU-VIC": (-37.0, 144.0), "AU-NT": (-19.0, 133.0),
        "US-TX": (31.0, -99.0), "US-WY": (43.0, -107.0), "US-OK": (35.5, -97.5),
        "CA-AB": (54.0, -115.0), "CA-QC": (52.0, -72.0),
        "GB-ENG": (52.5, -1.5), "NL-NH": (52.4, 4.9),
        "ZA": (-29.0, 25.0), "CL-02": (-23.0, -68.0),
        "PE-01": (-12.0, -77.0), "BR-PA": (-5.0, -55.0),
        "ID-KI": (-1.0, 114.0), "CN-NM": (44.0, 113.0),
        "IN-MH": (19.0, 75.0), "MN-01": (47.0, 103.0),
        # India states (added for demo company coverage)
        "IN-JH": (23.6, 85.5),   # Jharkhand (Jamshedpur ~23.8°N, 86.2°E)
        "IN-OD": (20.9, 85.1),   # Odisha (Kalinganagar ~20.9°N, 85.0°E)
        "IN-RJ": (27.0, 74.0),   # Rajasthan (Aditya Works, Shambhupura)
        "IN-GJ": (22.3, 71.2),   # Gujarat (industrial coast)
        "IN-AP": (15.9, 79.7),   # Andhra Pradesh
        "IN-KA": (15.3, 75.7),   # Karnataka
        "IN-MP": (23.5, 77.0),   # Madhya Pradesh
        # UK
        "GB-WLS": (51.7, -3.5),  # South Wales (Port Talbot ~51.6°N, -3.8°E)
        "GB-SCT": (57.0, -4.0),  # Scotland
        # Continental Europe
        "DE-HH": (53.5, 10.0),   # Hamburg
        "ES-CT": (41.4, 2.2),    # Barcelona / Catalonia
        "FR-IDF": (48.9, 2.4),   # Île-de-France / Paris
        # US additional states
        "US-FL": (27.7, -81.5),  # Florida (Miami ~25.8°N, -80.2°E)
        "US-GA": (32.7, -83.5),  # Georgia (Atlanta ~33.7°N, -84.4°E)
        "US-NY": (40.7, -74.0),  # New York (JFK ~40.6°N, -73.8°E)
        "US-CA": (36.8, -119.4), # California (LAX ~33.9°N, -118.4°E)
        "US-MN": (46.4, -93.3),  # Minnesota (MSP ~44.9°N, -93.2°E)
        "US-WA": (47.4, -120.5), # Washington state (Seattle ~47.6°N, -122.3°E)
        "global": (20.0, 0.0),
    }

    centroid = _REGION_CENTROIDS.get(region, (20.0, 0.0))
    lat_eff = lat if lat is not None else centroid[0]
    lon_eff = lon if lon is not None else centroid[1]

    # Coastal factor
    cf = _coastal_proximity_factor(lat_eff, lon_eff)
    is_coastal = cf >= 0.40

    # Elevation: use override, else region default
    # Coastal refinement: if asset is directly coastal (factor > 0.85) but the
    # region average elevation is > 50m, cap at 12m. Ports and coastal industrial
    # facilities are by definition near sea level; a region average like Maharashtra
    # (600m, Western Ghats influence) would otherwise incorrectly suppress coastal
    # hazards for Mumbai Port. Source: SRTM DEM coastal zone analysis.
    region_elev = REGION_ELEVATION_M.get(region, 300)
    if elevation_override is not None:
        elev = elevation_override
    elif cf >= 0.85 and region_elev > 50:
        elev = 12.0   # coastal asset assumed near sea level if no override given
    else:
        elev = region_elev

    # Sub-grid elevation corrections relative to regional (25km cell) mean
    subgrid = _subgrid_elevation_correction(elev, region_elev)

    # Resolution description for audit trail — describes exactly what the engine does,
    # not what the upstream data sources are capable of.
    if lat is not None and lon is not None:
        spatial_res = (
            f"Asset coordinates ({lat:.4f}°N, {lon:.4f}°E); "
            f"regional baseline (WRI Aqueduct 4.0, state/province level) refined by "
            f"sub-grid elevation correction (asset {elev:.0f}m vs regional mean {region_elev:.0f}m, "
            f"delta {subgrid['elev_delta_m']:+.0f}m)"
        )
        method = (
            "Step 1 — Baseline: WRI Aqueduct 4.0 state/province-level hazard scores "
            "(water stress, riverine flood, coastal flood, drought) anchored to observed "
            "1991-2020 climatology. NASA NEX-GDDP-CMIP6 and IPCC AR6 Ch11 used for "
            "extreme-event frequency calibration at regional level. "
            "Step 2 — Climate delta: IPCC AR6 WG1 Ch4 GMST trajectories (SSP1-2.6 through "
            "SSP5-8.5) with latitude-band warming amplification (0.80x equatorial to 2.20x Arctic, "
            "IPCC AR6 WG1 Fig 4.19). "
            "Step 3 — Sub-grid elevation correction: asset elevation relative to regional mean "
            "applies terrain-based adjustments: flood factor {:.2f}x (LISFLOOD DEM methodology), "
            "heat factor {:.2f}x (WMO lapse rate 0.65 deg C per 100m), "
            "water stress factor {:.2f}x (FAO AQUASTAT upstream catchment). "
            "Step 4 — Coastal proximity: Haversine distance-decay to Natural Earth coastline "
            "reference points replaces binary coastal classification; factor {:.2f} at this asset."
        ).format(
            subgrid["flood_factor"], subgrid["heat_factor"],
            subgrid["water_stress_factor"], round(cf, 2)
        )
    else:
        spatial_res = (
            f"Region centroid ({lat_eff:.2f}°N, {lon_eff:.2f}°E); "
            f"WRI Aqueduct 4.0 regional baseline applied without sub-grid correction. "
            f"Provide asset coordinates for terrain-based downscaling."
        )
        method = (
            "Step 1 — Baseline: WRI Aqueduct 4.0 state/province-level hazard scores. "
            "Step 2 — Climate delta: IPCC AR6 WG1 Ch4 GMST trajectories with regional "
            "warming amplification factors (IPCC AR6 Atlas). "
            "Sub-grid correction not applied — no asset coordinates provided. "
            "Results represent the regional average exposure for this hazard profile. "
            "Accuracy improves when asset-level latitude, longitude, and elevation are supplied."
        )

    return {
        "elevation_m": elev,
        "region_mean_elev_m": region_elev,
        "is_coastal": is_coastal,
        "coastal_factor": round(cf, 3),
        "lat_effective": lat_eff,
        "lon_effective": lon_eff,
        "subgrid": subgrid,
        "spatial_resolution": spatial_res,
        "downscaling_method": method,
    }


# ---------------------------------------------------------------------------
# Hazard result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class HazardScore:
    hazard: str
    annual_probability: float       # 0–1: chance of a damaging event this year
    severity_index: float           # 0–5: WRI-aligned severity
    production_loss_pct: float      # 0–1: expected fraction of production disrupted
    trend_2030: float               # severity index in 2030
    trend_2050: float               # severity index in 2050
    data_source: str
    applicable: bool = True         # False if hazard not relevant (e.g., SLR inland)
    notes: str = ""


@dataclass
class AssetHazardProfile:
    """Full 25-hazard profile for one asset under one SSP scenario (v0.4)."""

    asset_id: str
    asset_name: str
    region: str
    lat: Optional[float]
    lon: Optional[float]
    elevation_m: float
    is_coastal: bool
    coastal_factor: float   # continuous 0–1 proximity score (replaces binary is_coastal)
    lulc_type: str
    ssp: str

    hazards: dict[str, HazardScore] = field(default_factory=dict)

    # Composite scores
    physical_risk_score: float = 0.0      # 0–100
    annual_loss_pct: float = 0.0          # expected annual production loss %
    peak_loss_2050_pct: float = 0.0

    # Key findings for narrative
    top_hazards: list[str] = field(default_factory=list)
    critical_year: Optional[int] = None  # year when loss exceeds 5%

    # Spatial resolution metadata — required for audit trail
    spatial_resolution: str = ""
    downscaling_method: str = ""

    # Live API data fetched at assessment time (None if API unavailable)
    live_baseline: Optional[dict] = None      # NASA POWER observed baseline
    live_projection: Optional[dict] = None    # Open-Meteo CMIP6 projection delta

    # Satellite / real-time hazard observations (populated when _depth==0, lat/lon provided)
    # Keys match hazard names (e.g. "wildfire", "flood_riverine", "cyclone")
    # Values: True = active observation confirmed; False = connector returned no event
    observed_alerts: dict = field(default_factory=dict)
    # Human-readable list of observation data sources consulted
    observation_sources: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Core hazard computation functions
# ---------------------------------------------------------------------------

def _heat_stress(region: str, ssp: SSPScenario, year: int,
                 heat_factor: float = 1.0,
                 live_warming_c: Optional[float] = None,
                 live_baseline_t2m_max_c: Optional[float] = None) -> HazardScore:
    """
    Heat stress: WBGT exceedance, extreme heat days.

    Downscaling path (when coordinates provided):
      - live_baseline_t2m_max_c: actual observed T2M_MAX from NASA POWER (2001-2020)
      - live_warming_c: actual CMIP6 warming delta from Open-Meteo at SSP/year
      These replace the WRI regional heat_baseline and IPCC GMST lookup.

    Fallback path (no coordinates / API unavailable):
      - WRI_BASELINE regional heat_baseline
      - IPCC AR6 Ch11 GMST warming trajectory
      - Sub-grid heat_factor (lapse rate correction)

    Source: NASA POWER MERRA-2 baseline; Open-Meteo MRI-AGCM3.2-S CMIP6;
            IPCC AR6 Ch11.3; WMO standard atmosphere lapse rate.
    """
    # Choose warming and baseline from live data if available
    if live_warming_c is not None and live_baseline_t2m_max_c is not None:
        # Map actual T2M_MAX to a 0-5 severity scale anchored at 25°C
        # >35°C baseline → severity 4-5; 25°C → severity 2; <20°C → severity 1
        live_severity_base = max(0.5, min(5.0, (live_baseline_t2m_max_c - 20.0) / 5.0))
        warming = live_warming_c
        baseline_src = f"NASA POWER T2M_MAX {live_baseline_t2m_max_c:.1f}°C"
        data_src = "NASA POWER 2001-2020 baseline + Open-Meteo CMIP6 warming delta"
    else:
        live_severity_base = WRI_BASELINE.get(region, WRI_BASELINE["global"])["heat_baseline"]
        warming = ssp.regional_warming(year, region)
        baseline_src = f"WRI regional heat_baseline {live_severity_base:.2f}"
        data_src = "WRI Aqueduct 4.0 regional baseline + IPCC AR6 Ch11.3; Hasan et al 2022 (PMC) empirical loss calibration"

    freq_mult = HEAT_FREQ_PER_DEG_C ** warming
    met_mult = heat_stress_baseline_multiplier(region)
    base_prob = min(0.60, 0.06 * met_mult)
    adjusted_baseline = live_severity_base * heat_factor
    severity = min(5.0, adjusted_baseline * met_mult * (1 + warming * 0.35))
    prob = min(0.95, base_prob * freq_mult * heat_factor)
    # Empirical calibration (v0.4): PMC scoping review (Hasan et al 2022) finds
    # 0.3–10% productivity loss per 1°C WBGT increase across industrial sectors.
    # Outdoor/heavy-industry workers (mining, construction) show up to 60% loss
    # at WBGT >24°C. Coefficient 0.0010 anchors to the lower end (~1% at severity 3)
    # consistent with hardened industrial assets; equipment sensitivity multiplier
    # (GIS resolver) scales outdoor/open-pit assets to the upper empirical range.
    # Cap 0.012 ≈ 1.2% annual EL for extreme heat zones (WBGT 35°C+, SSP3-7.0).
    loss = min(0.012, max(0.0, (severity - 2.0) * 0.0010))  # empirical recal v0.4

    w30 = ssp.regional_warming(2030, region)
    w50 = ssp.regional_warming(2050, region)
    met_b = get_met_baseline(region)

    return HazardScore(
        hazard="heat_stress",
        annual_probability=round(prob, 4),
        severity_index=round(severity, 2),
        production_loss_pct=round(loss, 4),
        trend_2030=round(min(5.0, adjusted_baseline * met_mult * (1 + w30 * 0.35)), 2),
        trend_2050=round(min(5.0, adjusted_baseline * met_mult * (1 + w50 * 0.35)), 2),
        data_source=data_src + "; WMO lapse-rate sub-grid correction",
        notes=(
            f"{baseline_src}; lapse correction {heat_factor:.3f}; "
            f"warming {warming:.2f}°C; freq mult {freq_mult:.2f}×; "
            f"obs hot days {met_b.hot_days_above_35c:.0f}/yr; met calib {met_mult:.2f}×"
        ),
    )


def _flood_riverine(region: str, ssp: SSPScenario, year: int, lulc: str,
                    flood_factor: float = 1.0,
                    live_precip_delta_pct: Optional[float] = None) -> HazardScore:
    """
    Riverine (fluvial) flood: heavy precip + river system.
    Sources: WRI Aqueduct + IPCC AR6 Ch11.4 Clausius-Clapeyron + WMO precipitation normals.

    Downscaling: flood_factor is the sub-grid elevation correction derived from
    _subgrid_elevation_correction(). An asset 400m above the regional mean gets
    flood_factor ~0.50 — the 25km grid cell average is halved to reflect the
    asset sitting above the regional floodplain. Assets below the regional mean
    (in floodplain position) get flood_factor 1.25.
    Source: LISFLOOD sub-grid DEM analysis; WRI Aqueduct 4.0 validation.

    Live downscaling path (when coordinates provided):
      live_precip_delta_pct: actual projected precipitation change (%) from Open-Meteo
      CMIP6 at asset coordinates, SSP-scaled. Replaces IPCC GMST × Clausius-Clapeyron
      global average with a coordinate-level CMIP6 precipitation signal.
    """
    baseline = WRI_BASELINE.get(region, WRI_BASELINE["global"])["riverine_flood"]
    # Apply sub-grid elevation correction to the baseline before any SSP scaling
    adjusted_baseline = baseline * flood_factor
    warming = ssp.regional_warming(year, region)

    # Precipitation change factor: prefer live CMIP6 value, fall back to IPCC CC
    if live_precip_delta_pct is not None:
        precip_change = 1 + (live_precip_delta_pct / 100)
        precip_src = f"Open-Meteo CMIP6 {live_precip_delta_pct:+.1f}%"
    else:
        precip_change = 1 + (PRECIP_CHANGE_PCT_PER_DEG_C * warming / 100)
        precip_src = f"IPCC CC +{(precip_change - 1) * 100:.1f}%"

    # Guard against negative precip_change (unusual dry signal reducing precip below 0)
    precip_change = max(0.10, precip_change)

    runoff_mult = RUNOFF_BY_LULC.get(lulc, 1.0)
    pv_idx = precip_variability_index(region)
    severity = min(5.0, adjusted_baseline * precip_change * runoff_mult * (0.7 + 0.3 * pv_idx))
    prob = min(0.90, adjusted_baseline / 5.0 * precip_change * 0.25)
    # Empirical calibration (v0.4): Frontiers Water (2023) industrial vulnerability
    # study finds 2–8% annual production loss for high-exposure processing facilities.
    # Zurich Re (2012) 550-company survey: 49% of firms reported lost productivity
    # from flooding. Bangkok Metropolitan flood survey (2011/2017/2022): depth-damage
    # functions show significant sectoral variation. EM-DAT global data confirms
    # 1-4% crop yield loss at >10y return flood events.
    # Coefficient 0.0018 gives ~1.5% at severity 3.3 (high exposure, high runoff);
    # cap 0.020 = 2.0% EAL for worst-case (floodplain, urban runoff, SSP3-7.0).
    loss = min(0.020, max(0.0, (severity - 1.5) * 0.0018))  # empirical recal v0.4

    # Trend lines always use IPCC trajectory (live data is single-year point estimate)
    w30 = ssp.regional_warming(2030, region); w50 = ssp.regional_warming(2050, region)
    pc30 = 1 + PRECIP_CHANGE_PCT_PER_DEG_C * w30 / 100
    pc50 = 1 + PRECIP_CHANGE_PCT_PER_DEG_C * w50 / 100

    data_src = (
        "Open-Meteo CMIP6 precipitation delta + WRI Aqueduct 4.0 + sub-grid elevation; Frontiers Water 2023; Zurich Re 2012 flood survey"
        if live_precip_delta_pct is not None
        else "WRI Aqueduct 4.0 (0.25° grid) + IPCC AR6 Clausius-Clapeyron + sub-grid elevation; Frontiers Water 2023; Zurich Re 2012 flood survey"
    )

    return HazardScore(
        hazard="flood_riverine",
        annual_probability=round(prob, 4),
        severity_index=round(severity, 2),
        production_loss_pct=round(loss, 4),
        trend_2030=round(min(5.0, adjusted_baseline * pc30 * runoff_mult * (0.7 + 0.3 * pv_idx)), 2),
        trend_2050=round(min(5.0, adjusted_baseline * pc50 * runoff_mult * (0.7 + 0.3 * pv_idx)), 2),
        data_source=data_src,
        notes=(
            f"Grid baseline {baseline:.2f} × sub-grid correction {flood_factor:.2f} = adjusted {adjusted_baseline:.2f}; "
            f"precip change {precip_src}; runoff {runoff_mult:.1f}× (LULC: {lulc}); variability {pv_idx:.2f}"
        ),
    )


def _flood_coastal(region: str, ssp: SSPScenario, year: int,
                   is_coastal: bool, elevation_m: float,
                   coastal_factor: float = 1.0) -> HazardScore:
    """
    Coastal flood: storm surge + sea level rise compound event.
    Source: WRI Aqueduct (coastal) + IPCC AR6 Ch9.

    Downscaling improvement: coastal_factor (0–1) replaces the binary is_coastal
    flag. A factory 5km from the coast gets full exposure; one 200km inland gets
    near-zero even if nominally in a 'coastal' region. Consistent with CLIMADA
    coastal exposure decay and WRI Aqueduct 4.0 coastal gradient methodology.

    Elevation gate: assets above 80m are not materially exposed to storm surge.
    """
    if elevation_m > 80 or coastal_factor < 0.10:
        reason = (
            f"Asset elevation {elevation_m:.0f}m > 80m — storm surge not material"
            if elevation_m > 80
            else f"Asset coastal proximity {coastal_factor:.2f} < 0.10 — coastal flood not material"
        )
        return HazardScore(
            hazard="flood_coastal", annual_probability=0.0, severity_index=0.0,
            production_loss_pct=0.0, trend_2030=0.0, trend_2050=0.0,
            data_source="N/A", applicable=False,
            notes=reason,
        )
    baseline = WRI_BASELINE.get(region, WRI_BASELINE["global"])["coastal_flood"]
    slr = ssp.slr(year, region)
    # SLR amplifies coastal flood frequency (IPCC AR6 Ch9 Fig 9.28)
    slr_amplifier = 1 + slr * 3.5     # +3.5× risk per metre of SLR
    # Apply coastal proximity as a continuous exposure weight
    severity = min(5.0, baseline * slr_amplifier * coastal_factor)
    prob = min(0.80, baseline / 5.0 * slr_amplifier * coastal_factor * 0.20)
    loss = min(0.020, max(0.0, (severity - 1.0) * 0.0020))  # recal v0.3

    slr30 = ssp.slr(2030, region); slr50 = ssp.slr(2050, region)

    return HazardScore(
        hazard="flood_coastal",
        annual_probability=round(prob, 4),
        severity_index=round(severity, 2),
        production_loss_pct=round(loss, 4),
        trend_2030=round(min(5.0, baseline * (1 + slr30 * 3.5) * coastal_factor), 2),
        trend_2050=round(min(5.0, baseline * (1 + slr50 * 3.5) * coastal_factor), 2),
        data_source="WRI Aqueduct 4.0 (coastal) + IPCC AR6 Ch9 SLR; Haversine coastal proximity downscaling",
        notes=f"SLR {slr:.3f}m by {year}; coastal proximity factor {coastal_factor:.2f}; compound storm-surge amplification",
    )


def _sea_level_rise(region: str, ssp: SSPScenario, year: int,
                    is_coastal: bool, elevation_m: float,
                    coastal_factor: float = 1.0) -> HazardScore:
    """
    Chronic SLR inundation risk for low-lying coastal assets.
    Source: IPCC AR6 Ch9 + NOAA SLR viewer proxy.

    Downscaling: coastal_factor applied to SLR severity — an asset 250km inland
    has negligible SLR exposure even if the region is broadly 'coastal'.
    """
    if elevation_m > 20 or coastal_factor < 0.10:
        return HazardScore(
            hazard="sea_level_rise", annual_probability=0.0, severity_index=0.0,
            production_loss_pct=0.0, trend_2030=0.0, trend_2050=0.0,
            data_source="N/A", applicable=False,
            notes=f"{'Elevation ' + str(int(elevation_m)) + 'm > 20m threshold' if elevation_m > 20 else f'Coastal proximity {coastal_factor:.2f} < 0.10'} — SLR not material",
        )
    slr = ssp.slr(year, region)
    # Low-elevation assets (<5m) face higher chronic inundation risk
    elev_factor = max(0.1, 1 - elevation_m / 20.0)   # 0.75 at 5m, 0.25 at 15m
    severity = min(5.0, slr * 5 * elev_factor * 2.0 * coastal_factor)
    prob = min(0.70, severity / 5.0 * 0.30)
    loss = min(0.015, severity * 0.0018 * elev_factor)  # recal v0.3

    slr30 = ssp.slr(2030, region); slr50 = ssp.slr(2050, region)

    return HazardScore(
        hazard="sea_level_rise",
        annual_probability=round(prob, 4),
        severity_index=round(severity, 2),
        production_loss_pct=round(loss, 4),
        trend_2030=round(min(5.0, slr30 * 5 * elev_factor * 2.0 * coastal_factor), 2),
        trend_2050=round(min(5.0, slr50 * 5 * elev_factor * 2.0 * coastal_factor), 2),
        data_source="IPCC AR6 Ch9 + NOAA SLR Viewer proxy; Copernicus DEM elevation; coastal proximity downscaling",
        notes=f"SLR {slr:.3f}m by {year}; elevation ~{elevation_m:.0f}m; elev factor {elev_factor:.2f}; coastal proximity {coastal_factor:.2f}",
    )


def _saltwater_intrusion(region: str, ssp: SSPScenario, year: int,
                         is_coastal: bool, elevation_m: float,
                         coastal_factor: float = 1.0) -> HazardScore:
    """
    Saline water table / aquifer intrusion.
    Relevant for coastal assets relying on groundwater (mining dewatering,
    irrigation, drinking water for workforce).
    Source: IPCC AR6 WG2 Ch4 (freshwater security); regional SLR.

    Downscaling: coastal_factor gates applicability continuously.
    """
    if coastal_factor < 0.15:
        return HazardScore(
            hazard="saltwater_intrusion", annual_probability=0.0, severity_index=0.0,
            production_loss_pct=0.0, trend_2030=0.0, trend_2050=0.0,
            data_source="N/A", applicable=False,
            notes=f"Asset coastal proximity {coastal_factor:.2f} < 0.15 — saltwater intrusion not applicable",
        )
    slr = ssp.slr(year, region)
    water_stress = WRI_BASELINE.get(region, WRI_BASELINE["global"])["water_stress"]
    # Intrusion depends on SLR + groundwater draw-down + elevation + coastal proximity
    elev_factor = max(0.0, 1 - elevation_m / 15.0)
    severity = min(5.0, (slr * 4 + water_stress * 0.5) * elev_factor * coastal_factor)
    prob = min(0.60, severity / 5.0 * 0.20)
    loss = min(0.008, severity * 0.0008)  # recal v0.3

    slr30 = ssp.slr(2030, region); slr50 = ssp.slr(2050, region)

    return HazardScore(
        hazard="saltwater_intrusion",
        annual_probability=round(prob, 4),
        severity_index=round(severity, 2),
        production_loss_pct=round(loss, 4),
        trend_2030=round(min(5.0, (slr30 * 4 + water_stress * 0.5) * elev_factor * coastal_factor), 2),
        trend_2050=round(min(5.0, (slr50 * 4 + water_stress * 0.5) * elev_factor * coastal_factor), 2),
        data_source="IPCC AR6 WG2 Ch4 + WRI Aqueduct water stress; SLR-driven intrusion; coastal proximity",
        notes=f"SLR {slr:.3f}m; elevation {elevation_m:.0f}m; water stress {water_stress:.1f}/5; coastal factor {coastal_factor:.2f}",
    )


def _landslide(region: str, ssp: SSPScenario, year: int,
               elevation_m: float, lulc: str) -> HazardScore:
    """
    Rainfall-triggered landslide and slope instability.
    Source: IPCC AR6 WG2 Ch5 (food/land); slope proxy from elevation; LULC cover.
    """
    is_prone = region in LANDSLIDE_REGIONS
    base_severity = 2.5 if is_prone else 0.8
    warming = ssp.regional_warming(year, region)
    precip_change = 1 + PRECIP_CHANGE_PCT_PER_DEG_C * warming / 100
    # Forest cover reduces landslide (root cohesion); bare rock increases
    cover_factor = {"forest": 0.7, "grassland": 0.9, "shrubland": 0.85,
                    "cropland": 1.1, "bare_rock": 1.5, "urban": 1.3,
                    "wetland": 0.8, "snow_ice": 1.2}.get(lulc, 1.0)
    # High elevation = steeper slopes
    elev_factor = min(2.0, 1 + elevation_m / 3000)
    severity = min(5.0, base_severity * precip_change * cover_factor * elev_factor * 0.85)
    prob = min(0.40, severity / 5.0 * 0.15)
    loss = min(0.010, max(0.0, (severity - 1.5) * 0.0010))  # recal v0.3

    w30 = ssp.regional_warming(2030, region); w50 = ssp.regional_warming(2050, region)
    pc30 = 1 + PRECIP_CHANGE_PCT_PER_DEG_C * w30 / 100
    pc50 = 1 + PRECIP_CHANGE_PCT_PER_DEG_C * w50 / 100

    return HazardScore(
        hazard="landslide",
        annual_probability=round(prob, 4),
        severity_index=round(severity, 2),
        production_loss_pct=round(loss, 4),
        trend_2030=round(min(5.0, base_severity * pc30 * cover_factor * elev_factor * 0.85), 2),
        trend_2050=round(min(5.0, base_severity * pc50 * cover_factor * elev_factor * 0.85), 2),
        data_source="IPCC AR6 WG2 Ch5; SRTM/Copernicus DEM elevation proxy; LULC cover",
        notes=f"Elevation {elevation_m:.0f}m; LULC {lulc}; landslide-prone region: {is_prone}",
    )


def _wildfire(region: str, ssp: SSPScenario, year: int, lulc: str) -> HazardScore:
    """
    Wildfire weather index (FWI) and fire season length.
    Source: NASA FIRMS historical + IPCC AR6 WG2 Ch12.3; Abatzoglou et al. 2019.
    """
    is_prone = region in WILDFIRE_REGIONS
    base_severity = 2.8 if is_prone else 1.0
    warming = ssp.regional_warming(year, region)
    fwi_change = 1 + WILDFIRE_INDEX_PCT_PER_DEG_C * warming / 100
    fuel_load = WILDFIRE_FUEL_BY_LULC.get(lulc, 1.0)
    severity = min(5.0, base_severity * fwi_change * fuel_load)
    prob = min(0.85, severity / 5.0 * (0.35 if is_prone else 0.10))
    # Empirical calibration (v0.4): California wildfire data (CalFire 2018–2022):
    # - Thomas Fire: $70M agricultural losses; citrus/avocado 10–20% annual loss
    # - Camp Fire (Paradise): 100% facility loss for directly affected assets;
    #   supply chain disruption 2–6 weeks for nearby food/ag processors.
    # NIST SP 1215 (Costs and Losses of Wildfires): median direct cost per
    # acre-equivalence ~$32k for industrial assets.
    # Industrial assets with firebreaks/hardening: 1–4% EAL in fire-prone zones.
    # Coefficient 0.0012; cap 0.018 (1.8% EAL) reflects worst-case fire-season
    # disruption (agriculture/F&B in CA chaparral or AU eucalyptus belt).
    # Equipment sensitivity multiplier (processing_plant ×1.2) handles F&B uplift.
    loss = min(0.018, max(0.0, (severity - 2.0) * 0.0012))  # empirical recal v0.4: CalFire/NIST

    w30 = ssp.regional_warming(2030, region); w50 = ssp.regional_warming(2050, region)
    fwi30 = 1 + WILDFIRE_INDEX_PCT_PER_DEG_C * w30 / 100
    fwi50 = 1 + WILDFIRE_INDEX_PCT_PER_DEG_C * w50 / 100

    return HazardScore(
        hazard="wildfire",
        annual_probability=round(prob, 4),
        severity_index=round(severity, 2),
        production_loss_pct=round(loss, 4),
        trend_2030=round(min(5.0, base_severity * fwi30 * fuel_load), 2),
        trend_2050=round(min(5.0, base_severity * fwi50 * fuel_load), 2),
        data_source="NASA FIRMS (historical fire hotspots) + IPCC AR6 WG2 Ch12.3; CalFire 2018-2022; NIST SP 1215 wildfire cost analysis; empirical recal v0.4",
        notes=f"FWI change +{(fwi_change-1)*100:.1f}%; fuel load {fuel_load:.1f}× (LULC: {lulc})",
    )


def _cyclone(region: str, ssp: SSPScenario, year: int, is_coastal: bool,
             is_cyclone_belt_override: bool | None = None) -> HazardScore:
    """
    Tropical cyclone / hurricane intensity and track frequency.
    Source: IBTRACS historical + IPCC AR6 Ch11.7; Knutson et al. 2020.

    is_cyclone_belt_override: when provided from GIS resolver (IBTrACS track density),
    allows the engine to enable cyclone risk even for region codes not in CYCLONE_REGIONS.
    """
    # GIS override: real lat/lon cyclone belt detection takes precedence
    in_belt = is_cyclone_belt_override if is_cyclone_belt_override is not None \
        else (region in CYCLONE_REGIONS)
    if not (in_belt and is_coastal):
        return HazardScore(
            hazard="cyclone", annual_probability=0.0, severity_index=0.0,
            production_loss_pct=0.0, trend_2030=0.0, trend_2050=0.0,
            data_source="N/A", applicable=False,
            notes="Asset outside tropical cyclone belt or inland — not applicable",
        )
    # Base cyclone frequency: blend IBTrACS historical lookup with observed met baseline
    _ibtracs_base = {
        "AU-QLD": 0.12, "AU-WA": 0.08, "AU-NT": 0.10,
        "IN-MH": 0.06, "ID-KI": 0.04, "BR-PA": 0.03,
        "US-TX": 0.10, "ZA": 0.04,
    }.get(region, 0.05)
    # Met baseline provides additional IBTRACS-derived probability for cross-check
    met_cyclone_p = observed_cyclone_prob(region)
    # Blend: 60% IBTrACS table + 40% met baseline (convergence as met data quality improves)
    base_prob = 0.6 * _ibtracs_base + 0.4 * (met_cyclone_p if met_cyclone_p > 0 else _ibtracs_base)
    warming = ssp.regional_warming(year, region)
    intensity_change = 1 + CYCLONE_INTENSITY_PCT_PER_DEG_C * warming / 100
    # Fewer but more intense (IPCC AR6 Ch11.7 — frequency -5% to -10% per °C,
    # intensity +5% per °C for Category 4–5 events).
    prob = min(0.40, base_prob * (1 - 0.03 * warming))
    severity = min(5.0, 3.0 * intensity_change)
    # Empirical calibration (v0.4): BSEE Gulf of Mexico data (2005–2024) shows
    # Hurricanes Katrina/Rita caused 30% and 21% annual production loss from
    # offshore oil assets respectively (9 months post-storm).
    # SPE/JPT empirical model (historical 1950-2003): per-storm loss ranges
    # 3.35%–84.3% of daily production, ~$850M BOE equivalent per major storm.
    # Peak shut-in: up to 100% oil, 80% gas for Category 4–5 landfall.
    # Expected annual loss = hit probability × ~25% per-event loss fraction
    # (reflects Category 3–5 average, 2–4 week shutdown at a cyclone-exposed asset).
    # Knutson et al. 2020 confirms intensity escalation for Cat4-5 storms.
    # Cap 0.08 = 8% EAL (e.g., AU-QLD oil terminal: p=0.10 × 0.80 event loss).
    loss = min(0.08, prob * intensity_change * 0.25)  # empirical recal v0.4: BSEE/SPE data

    w30 = ssp.regional_warming(2030, region); w50 = ssp.regional_warming(2050, region)
    ic30 = 1 + CYCLONE_INTENSITY_PCT_PER_DEG_C * w30 / 100
    ic50 = 1 + CYCLONE_INTENSITY_PCT_PER_DEG_C * w50 / 100

    return HazardScore(
        hazard="cyclone",
        annual_probability=round(prob, 4),
        severity_index=round(severity, 2),
        production_loss_pct=round(loss, 4),
        trend_2030=round(min(5.0, 3.0 * ic30), 2),
        trend_2050=round(min(5.0, 3.0 * ic50), 2),
        data_source="IBTrACS v4 + WMO 1991-2020 observed landfall p + IPCC AR6 Ch11.7; Knutson et al. 2020; BSEE GoM shut-in data 2005-2024; SPE/JPT empirical storm production model",
        notes=(
            f"Blended base prob {base_prob:.3f} (IBTrACS {_ibtracs_base:.2f} + met obs {met_cyclone_p:.2f}); "
            f"intensity +{(intensity_change-1)*100:.1f}% vs baseline"
        ),
    )


def _drought(region: str, ssp: SSPScenario, year: int) -> HazardScore:
    """
    Meteorological + hydrological drought intensity and duration.
    Sources: WRI Aqueduct (baseline) + IPCC AR6 Ch11.6 (PDSI) + WMO observed return periods.

    Calibration: the annual drought probability is anchored to the observed
    drought return period from met baselines (e.g. CL-02 returns every 3yr
    vs CA-QC every 12yr), giving meaningfully different event probabilities.
    """
    baseline = WRI_BASELINE.get(region, WRI_BASELINE["global"])["drought"]
    warming = ssp.regional_warming(year, region)
    # Drought intensifies non-linearly with warming (IPCC AR6 — PDSI scaling)
    drought_mult = 1 + warming * 0.22 + warming ** 2 * 0.04
    severity = min(5.0, baseline * drought_mult)
    # Met-calibrated probability: inverse of observed return period
    obs_return = observed_drought_return(region)                # years between events
    base_drought_prob = min(0.40, 1.0 / max(obs_return, 1.0))  # e.g. 1/3 = 0.33 for CL-02
    prob = min(0.80, base_drought_prob * drought_mult)
    # Empirical calibration (v0.4): CRU Group water-mining study + Antofagasta/Anglo
    # American operational data from Chilean Atacama drought events:
    #   - Los Pelambres copper mine: 24% production drop in Q1 2022 (drought-driven)
    #   - Los Bronces: 28% decline in Q4 2019, 16% year-on-year due to water shortage
    #   - Sustained drought scenario: 15% annual production loss for high water-use mines
    # Annual EL = event probability × per-event loss fraction.
    # At CL-02 return period 3yr (p≈0.33): EL = 0.33 × 0.08 = 2.6% — anchors coefficient.
    # Cap 0.025 = 2.5% EAL reflects extreme-drought, high water-stress mining/processing.
    # Equipment sensitivity multiplier applies additional sector scaling (open_pit_mine ×1.5).
    loss = min(0.025, max(0.0, (severity - 2.0) * 0.0018))  # empirical recal v0.4: CRU/Antofagasta

    w30 = ssp.regional_warming(2030, region); w50 = ssp.regional_warming(2050, region)
    dm30 = 1 + w30 * 0.22 + w30**2 * 0.04
    dm50 = 1 + w50 * 0.22 + w50**2 * 0.04

    return HazardScore(
        hazard="drought",
        annual_probability=round(prob, 4),
        severity_index=round(severity, 2),
        production_loss_pct=round(loss, 4),
        trend_2030=round(min(5.0, baseline * dm30), 2),
        trend_2050=round(min(5.0, baseline * dm50), 2),
        data_source="WRI Aqueduct 4.0 + IPCC AR6 Ch11.6 PDSI + WMO return periods; CRU Group/Antofagasta operational data (Los Bronces/Los Pelambres 2019-2022); empirical recal v0.4",
        notes=(
            f"Drought mult {drought_mult:.2f}× at {warming:.2f}°C; "
            f"observed return period {obs_return:.0f}yr; base annual prob {base_drought_prob:.3f}"
        ),
    )


def _water_stress(region: str, ssp: SSPScenario, year: int,
                  water_stress_factor: float = 1.0) -> HazardScore:
    """
    Chronic freshwater scarcity — water withdrawal vs availability.
    Source: WRI Aqueduct 4.0 (primary); FAO AQUASTAT; IPCC AR6 WG2 Ch4.

    Downscaling: water_stress_factor reduces the grid-cell average for assets
    at higher elevation, which are typically upstream of water stress zones and
    have access to fresher highland catchments. −15% per 500m above regional mean,
    capped at 40% reduction. Source: FAO AQUASTAT; WRI Aqueduct 4.0 sub-basin.
    """
    baseline = WRI_BASELINE.get(region, WRI_BASELINE["global"])["water_stress"]
    adjusted_baseline = baseline * water_stress_factor
    warming = ssp.regional_warming(year, region)
    stress_mult = 1 + warming * 0.15
    severity = min(5.0, adjusted_baseline * stress_mult)
    prob = min(0.95, severity / 5.0 * 0.55)
    loss = min(0.012, max(0.0, (severity - 2.5) * 0.0012))  # recal v0.3

    w30 = ssp.regional_warming(2030, region); w50 = ssp.regional_warming(2050, region)

    return HazardScore(
        hazard="water_stress",
        annual_probability=round(prob, 4),
        severity_index=round(severity, 2),
        production_loss_pct=round(loss, 4),
        trend_2030=round(min(5.0, adjusted_baseline * (1 + w30 * 0.15)), 2),
        trend_2050=round(min(5.0, adjusted_baseline * (1 + w50 * 0.15)), 2),
        data_source="WRI Aqueduct 4.0 (0.25° grid) + IPCC AR6 WG2 Ch4; sub-grid elevation correction",
        notes=(
            f"Grid baseline {baseline:.2f} × elevation factor {water_stress_factor:.2f} = {adjusted_baseline:.2f}; "
            f"stress mult {stress_mult:.2f}×"
        ),
    )


# ===========================================================================
# New v0.4 Hazard Functions — 15 additional physical climate hazards
# ===========================================================================

def _extreme_cold(region: str, ssp: SSPScenario, year: int,
                  elevation_m: float = 300) -> HazardScore:
    """
    Extreme cold waves and severe frost events disrupting outdoor operations.

    Warming globally reduces cold-snap frequency, but the effect is non-linear:
    Arctic amplification can paradoxically increase cold-air-outbreak frequency
    at mid-latitudes via weakened polar vortex (Overland et al 2021). High-
    elevation assets remain exposed even as global mean warms.

    Sources: IPCC AR6 Ch11.3; WMO Cold Wave Guidance; Overland et al 2021.
    """
    _COLD_BASELINE: dict[str, float] = {
        "CA-AB": 0.65, "CA-QC": 0.60, "MN-01": 0.70,
        "CN-NM": 0.55, "US-WY": 0.45,
    }
    # Not applicable for warm regions below 1500m elevation
    if region not in COLD_REGIONS and elevation_m < 1500:
        return HazardScore(
            hazard="extreme_cold", annual_probability=0.0, severity_index=0.0,
            production_loss_pct=0.0, trend_2030=0.0, trend_2050=0.0,
            data_source="IPCC AR6 Ch11.3", applicable=False,
            notes="Region/elevation outside cold-hazard zone",
        )

    warming = ssp.regional_warming(year, region)
    # Warming reduces cold-snap frequency ~7% per °C of mean warming
    cold_reduction = max(0.20, 1.0 - warming * 0.07)
    base_prob = _COLD_BASELINE.get(region, 0.30)
    prob = min(0.80, base_prob * cold_reduction)

    # Elevation amplifies severity (lapse rate ~0.65°C/100m)
    elev_factor = min(1.5, 1.0 + max(0.0, elevation_m - 500) / 5000)
    sev_base = base_prob / 0.70 * 3.0  # normalise to 0–5 scale
    severity = min(5.0, sev_base * elev_factor * max(0.5, 1.0 - warming * 0.10))
    loss = min(0.010, max(0.0, (severity - 1.5) * 0.0012))

    w30 = ssp.regional_warming(2030, region)
    w50 = ssp.regional_warming(2050, region)
    return HazardScore(
        hazard="extreme_cold",
        annual_probability=round(prob, 4),
        severity_index=round(severity, 2),
        production_loss_pct=round(loss, 4),
        trend_2030=round(min(5.0, sev_base * elev_factor * max(0.5, 1.0 - w30 * 0.10)), 2),
        trend_2050=round(min(5.0, sev_base * elev_factor * max(0.5, 1.0 - w50 * 0.10)), 2),
        data_source="IPCC AR6 Ch11.3 (cold extremes); Overland et al 2021 (Arctic amplification); WMO",
        notes=f"Cold reduction {cold_reduction:.2f}× at +{warming:.2f}°C; elevation factor {elev_factor:.2f}×",
    )


def _blade_icing(region: str, ssp: SSPScenario, year: int,
                 elevation_m: float = 300,
                 mean_winter_temp_override: float | None = None,
                 equipment_type: str | None = None) -> HazardScore:
    """
    Wind turbine blade and overhead-line icing.

    *** APPLICABILITY: wind turbines and overhead power lines ONLY. ***
    This hazard is NOT applicable to breweries, mines, refineries, smelters,
    or any asset that does not have rotating wind-energy blades or long-span
    overhead conductors. For those assets, the relevant cold-weather risks are
    modelled by _freeze_thaw_cycle (pipe burst, concrete cracking) and
    _extreme_cold (cold-wave operational disruption).

    Icing is maximum in the 'glaze-ice band': mean winter temperature −10°C to 0°C
    with liquid water present (freezing rain, rime, fog). This creates an asymmetric
    climate response: warming INCREASES icing risk in very cold continental regions
    (Siberia, Mongolia) by shifting temperatures into the danger band, while
    DECREASING it in milder cold regions (Scandinavia, UK) already near 0°C.
    Modelled as a Gaussian exposure function centred on −5°C mean winter temperature.

    mean_winter_temp_override: when provided from GIS resolver (ERA5 2°×2° grid),
    replaces the region-code lookup table for more accurate temperature input.
    equipment_type: if provided and NOT a blade/line asset, returns not-applicable.

    Sources: IEA Wind Task 19 (2021); Rissanen et al 2022; Dobesch et al 2003;
             IPCC AR6 Ch11.3; RenewableUK (2020) icing incident registry.
    """
    # ── Equipment applicability gate ───────────────────────────────────────
    # Blade icing is ONLY meaningful for assets with rotating aerodynamic surfaces
    # or long-span overhead conductors. For all other equipment types the hazard
    # is not applicable — freeze-thaw and extreme-cold cover the relevant risks.
    _BLADE_EQUIPMENT = {
        "wind_farm", "wind_turbine", "transmission_line",
        "overhead_line", "power_line",
    }
    # If equipment_type is explicitly set to something non-blade, skip.
    # If equipment_type is None (unknown), we keep the hazard to avoid false negatives
    # for cases where the user hasn't specified (e.g. custom imports without type).
    if equipment_type is not None and equipment_type not in _BLADE_EQUIPMENT:
        return HazardScore(
            hazard="blade_icing", annual_probability=0.0, severity_index=0.0,
            production_loss_pct=0.0, trend_2030=0.0, trend_2050=0.0,
            data_source="IEA Wind Task 19", applicable=False,
            notes=f"Not applicable — {equipment_type} has no rotating wind-energy blades. "
                  "See freeze_thaw_cycle and extreme_cold for cold-weather infrastructure risks.",
        )

    # Mean winter 2-m temperature proxy (°C) — historical normal, pre-warming
    # Sources: ERA5 1991-2020 seasonal climatology; NOAA Global Climate Normals
    _MEAN_WINTER_T: dict[str, float] = {
        "CA-AB": -12.0, "CA-QC": -14.0, "MN-01": -20.0,
        "CN-NM": -15.0, "US-WY": -8.0, "GB-ENG": 4.0,
        "NL-NH": 3.0, "AU-WA": 14.0, "AU-QLD": 20.0,
        # India states — subtropical/tropical, no meaningful winter freeze
        "IN-JH": 15.0, "IN-OD": 18.0, "IN-RJ": 10.0, "IN-GJ": 16.0,
        "IN-AP": 22.0, "IN-KA": 22.0, "IN-MP": 13.0,
        # UK / Europe
        "GB-WLS": 5.0,   # Port Talbot maritime mild
        "GB-SCT": 3.0,   # Scotland cool maritime
        "DE-HH": 1.0,    # Hamburg (near 0°C but icing exposure moderate)
        "ES-CT": 9.0,    # Barcelona — too warm for meaningful icing
        "FR-IDF": 4.0,   # Paris — marginal icing
        # US states
        "US-FL": 20.0,   # Miami — tropical
        "US-GA": 6.0,    # Atlanta — mild winters
        "US-NY": 0.0,    # New York — near freezing (icing possible)
        "US-CA": 14.0,   # LA — mild, no freeze
        "US-MN": -12.0,  # Minneapolis — very cold (outside glaze-ice band)
        "US-WA": 5.0,    # Seattle — maritime mild
    }
    _ICING_BASE_PROB: dict[str, float] = {
        "CA-AB": 0.55, "CA-QC": 0.50, "MN-01": 0.40,
        "CN-NM": 0.35, "US-WY": 0.45, "GB-ENG": 0.20, "NL-NH": 0.15,
        "DE-HH": 0.12, "GB-WLS": 0.08, "GB-SCT": 0.10,
        "US-NY": 0.10, "US-MN": 0.15,
    }
    # Regions too warm for blade icing (mean winter T reliably > 8°C)
    _NO_ICING = {
        "AU-WA", "AU-QLD", "AU-SA", "AU-NSW", "AU-NT",
        "IN-MH", "ID-KI", "BR-PA", "PE-01", "ZA", "CL-02",
        # India states — all subtropical/tropical
        "IN-JH", "IN-OD", "IN-RJ", "IN-GJ", "IN-AP", "IN-KA", "IN-MP",
        # US/Europe warm regions
        "US-FL", "US-GA", "US-CA",
        "ES-CT",   # Barcelona — winter T ~9°C (marginal; no industrial wind assets)
    }
    # Only skip based on region code when no GIS override provides a cold MWT
    if region in _NO_ICING and mean_winter_temp_override is None:
        return HazardScore(
            hazard="blade_icing", annual_probability=0.0, severity_index=0.0,
            production_loss_pct=0.0, trend_2030=0.0, trend_2050=0.0,
            data_source="IEA Wind Task 19", applicable=False,
            notes="Region too warm for blade-icing hazard",
        )
    # GIS resolver provides ERA5 mean winter T → use if available
    baseline_mwt = (
        mean_winter_temp_override
        if mean_winter_temp_override is not None
        else _MEAN_WINTER_T.get(region, -5.0)
    )
    # Warm-region shortcut: if GIS confirms warm baseline, skip immediately
    if baseline_mwt > 8.0:
        return HazardScore(
            hazard="blade_icing", annual_probability=0.0, severity_index=0.0,
            production_loss_pct=0.0, trend_2030=0.0, trend_2050=0.0,
            data_source="IEA Wind Task 19 / GIS ERA5", applicable=False,
            notes=f"GIS mean winter T {baseline_mwt:.1f}°C — too warm for icing",
        )

    warming = ssp.regional_warming(year, region)
    mean_winter_t_now = baseline_mwt + warming
    danger_center, danger_std = -5.0, 8.0

    def _icing_exposure(t: float) -> float:
        return math.exp(-0.5 * ((t - danger_center) / danger_std) ** 2)

    ie = _icing_exposure(mean_winter_t_now)
    elev_factor = min(1.6, 1.0 + max(0.0, elevation_m - 400) / 2000)
    base_prob = _ICING_BASE_PROB.get(region, 0.25)
    prob = min(0.85, base_prob * ie * elev_factor)
    severity = min(5.0, ie * 4.0 * elev_factor)
    loss = min(0.015, max(0.0, prob * 0.022))

    w30 = ssp.regional_warming(2030, region)
    w50 = ssp.regional_warming(2050, region)
    mwt0 = _MEAN_WINTER_T.get(region, -5.0)
    ie30 = _icing_exposure(mwt0 + w30) * elev_factor
    ie50 = _icing_exposure(mwt0 + w50) * elev_factor
    return HazardScore(
        hazard="blade_icing",
        annual_probability=round(prob, 4),
        severity_index=round(severity, 2),
        production_loss_pct=round(loss, 4),
        trend_2030=round(min(5.0, ie30 * 4.0), 2),
        trend_2050=round(min(5.0, ie50 * 4.0), 2),
        data_source="IEA Wind Task 19 (2021); Rissanen et al 2022; IPCC AR6 Ch11.3",
        notes=(
            f"Winter T {mwt0:.1f}°C + {warming:.2f}°C warming = {mean_winter_t_now:.1f}°C; "
            f"icing exposure {ie:.3f} (danger band {danger_center}°C±{danger_std}°C); "
            f"elevation factor {elev_factor:.2f}×"
        ),
    )


def _extratropical_cyclone(region: str, ssp: SSPScenario, year: int,
                           elevation_m: float = 300) -> HazardScore:
    """
    Mid-latitude (extratropical) cyclone — wind, storm surge, extreme precipitation.

    Unlike tropical cyclones, extratropical cyclones are not latitude-gated.
    IPCC AR6 projects: poleward shift of storm tracks (~0.5°/°C), potential
    intensification of the strongest events. Net risk change is region-dependent.
    Coastal assets face compound flood risk from extratropical surge.

    Sources: IPCC AR6 Ch11.7; Zappa et al 2013; Mölter et al 2016;
             ERA5 reanalysis (Copernicus C3S).
    """
    _ETC_BASELINE: dict[str, float] = {
        "GB-ENG": 3.2, "NL-NH": 3.0, "CA-QC": 2.5,
        "AU-VIC": 2.0, "AU-NSW": 1.8, "ZA": 1.5,
        "US-WY": 1.8, "CL-02": 2.2,
    }
    if region not in EXTRATROPICAL_CYCLONE_REGIONS:
        return HazardScore(
            hazard="extratropical_cyclone", annual_probability=0.0, severity_index=0.0,
            production_loss_pct=0.0, trend_2030=0.0, trend_2050=0.0,
            data_source="IPCC AR6 Ch11.7", applicable=False,
            notes="Region outside extratropical storm-track exposure zone",
        )

    baseline = _ETC_BASELINE.get(region, 1.5)
    warming = ssp.regional_warming(year, region)
    # Poleward storm track shift may modestly intensify storms at mid-latitudes
    # Net effect: ~5% intensity increase per °C for strongest events (IPCC AR6)
    intensity_mult = 1.0 + warming * 0.05
    severity = min(5.0, baseline * intensity_mult)
    prob = min(0.90, baseline / 5.0 * 0.60 * intensity_mult)
    loss = min(0.040, max(0.0, (severity - 1.5) * 0.010))

    w30 = ssp.regional_warming(2030, region)
    w50 = ssp.regional_warming(2050, region)
    return HazardScore(
        hazard="extratropical_cyclone",
        annual_probability=round(prob, 4),
        severity_index=round(severity, 2),
        production_loss_pct=round(loss, 4),
        trend_2030=round(min(5.0, baseline * (1 + w30 * 0.05)), 2),
        trend_2050=round(min(5.0, baseline * (1 + w50 * 0.05)), 2),
        data_source="IPCC AR6 Ch11.7; ERA5/Copernicus C3S; Zappa et al 2013",
        notes=f"ETC baseline {baseline:.2f}; intensity mult {intensity_mult:.3f}× at +{warming:.2f}°C",
    )


def _flash_flood(region: str, ssp: SSPScenario, year: int, lulc: str,
                 elevation_m: float = 300) -> HazardScore:
    """
    Flash flood: rapid-onset flooding from intense convective precipitation.

    Distinct from riverine flood — driven by short-duration extreme precipitation
    (minutes to hours) overwhelming drainage capacity. Clausius-Clapeyron scaling
    predicts ~7% increase in extreme hourly precipitation per °C of warming.
    High-imperviousness LULC (urban, bare rock) amplifies runoff.

    Sources: IPCC AR6 Ch11.4; Westra et al 2014; NOAA Atlas 14.
    """
    # Flash flood probability derived from WRI riverine baseline (proxy)
    riverine_base = WRI_BASELINE.get(region, WRI_BASELINE["global"])["riverine_flood"]
    warming = ssp.regional_warming(year, region)

    # Extreme precipitation scaling: +7% per °C (Clausius-Clapeyron; IPCC AR6 Box 11.1)
    extreme_precip_mult = (1 + warming * 0.07)
    # High-imperviousness LULC increases runoff velocity
    runoff_mult = RUNOFF_BY_LULC.get(lulc, 1.0)
    # Low-elevation terrain is more susceptible (valley accumulation)
    elev_discount = max(0.4, 1.0 - elevation_m / 3000)

    severity = min(5.0, riverine_base * extreme_precip_mult * runoff_mult * elev_discount)
    prob = min(0.80, riverine_base / 5.0 * extreme_precip_mult * 0.30)
    # Empirical calibration (v0.4): Frontiers Water (2023) industrial vulnerability
    # study (petrochemical/food processing): flash floods cause 0.5–8% production
    # loss per event. For 10–30% annual probability in urban high-runoff zones:
    # EAL = 0.15 × 0.12 (per-event avg) = 1.8%. EM-DAT food-sector events show
    # rapid recovery (<3 weeks) typical; severe cases reach 6-week shutdown.
    # Cap 0.018; coefficient 0.0017 gives ~1.5% at severity 2.9 (high-runoff urban).
    loss = min(0.018, max(0.0, (severity - 1.0) * 0.0017))  # empirical recal v0.4: Frontiers 2023

    w30 = ssp.regional_warming(2030, region)
    w50 = ssp.regional_warming(2050, region)
    return HazardScore(
        hazard="flash_flood",
        annual_probability=round(prob, 4),
        severity_index=round(severity, 2),
        production_loss_pct=round(loss, 4),
        trend_2030=round(min(5.0, riverine_base * (1 + w30 * 0.07) * runoff_mult * elev_discount), 2),
        trend_2050=round(min(5.0, riverine_base * (1 + w50 * 0.07) * runoff_mult * elev_discount), 2),
        data_source="IPCC AR6 Ch11.4 Box 11.1 (Clausius-Clapeyron); WRI Aqueduct; Westra et al 2014",
        notes=(
            f"Riverine proxy {riverine_base:.2f}; CC extreme precip mult {extreme_precip_mult:.3f}×; "
            f"runoff {runoff_mult:.1f}× (LULC: {lulc}); elevation discount {elev_discount:.2f}×"
        ),
    )


def _permafrost_thaw(region: str, ssp: SSPScenario, year: int,
                    is_permafrost_override: bool | None = None) -> HazardScore:
    """
    Permafrost thaw — ground instability for infrastructure in Arctic/subarctic.

    Thawing permafrost causes subsidence, bearing-capacity loss, and pipeline/
    foundation failure. Arctic warming is 2–4× global mean rate (Arctic amplification).
    IPCC AR6 projects 25% reduction in near-surface permafrost area by 2100 under SSP5-8.5.
    Active-layer deepening causes progressive infrastructure damage.

    is_permafrost_override: when provided from GIS resolver (NSIDC 2020 permafrost extent),
    allows the engine to enable permafrost risk for assets whose region code is not in
    PERMAFROST_REGIONS but whose lat/lon falls within mapped permafrost boundaries.

    Sources: IPCC AR6 Ch9.5.2; AMAP 2021 (Arctic Monitoring); Hjort et al 2022.
    """
    in_permafrost = is_permafrost_override if is_permafrost_override is not None \
        else (region in PERMAFROST_REGIONS)
    if not in_permafrost:
        return HazardScore(
            hazard="permafrost_thaw", annual_probability=0.0, severity_index=0.0,
            production_loss_pct=0.0, trend_2030=0.0, trend_2050=0.0,
            data_source="IPCC AR6 Ch9.5.2; AMAP 2021", applicable=False,
            notes="Region not within mapped permafrost zone",
        )

    _PERM_BASELINE: dict[str, float] = {
        "CA-QC": 2.5,    # Widespread discontinuous permafrost
        "CA-AB": 1.0,    # Sporadic permafrost (northern fringes only)
        "MN-01": 3.0,    # Continental Mongolia — extensive permafrost
    }
    baseline = _PERM_BASELINE.get(region, 1.5)
    # Arctic warming amplification factor ~3×
    warming = ssp.regional_warming(year, region) * 2.5
    severity = min(5.0, baseline * (1 + warming * 0.30))
    prob = min(0.90, baseline / 5.0 * (1 + warming * 0.20))
    # Empirical calibration (v0.4): Hjort et al 2022 (Nature Reviews Earth)
    # projects 30–50% of current Arctic infrastructure at high hazard risk by 2050.
    # IOPscience permafrost infrastructure cost study (Russia): annual repair costs
    # growing from ₽14–28B/yr by 2030, reaching $276B cumulative by 2050.
    # AMAP 2021: 49% of Russian roads, 17% Canadian infrastructure at risk; active-
    # layer deepening causes progressive bearing-capacity loss for pipelines, roads.
    # Annual EAL: ~1% at current exposure, growing to ~3% by 2050 at high baseline.
    # Coefficient 0.0045; cap 0.030 (3%) reflects worst-case discontinuous permafrost.
    loss = min(0.030, max(0.0, (severity - 1.0) * 0.0045))  # empirical recal v0.4: Hjort/AMAP

    w30 = ssp.regional_warming(2030, region) * 2.5
    w50 = ssp.regional_warming(2050, region) * 2.5
    return HazardScore(
        hazard="permafrost_thaw",
        annual_probability=round(prob, 4),
        severity_index=round(severity, 2),
        production_loss_pct=round(loss, 4),
        trend_2030=round(min(5.0, baseline * (1 + w30 * 0.30)), 2),
        trend_2050=round(min(5.0, baseline * (1 + w50 * 0.30)), 2),
        data_source="IPCC AR6 Ch9.5.2; AMAP 2021 Permafrost Atlas; Hjort et al 2022 (Nature Reviews Earth); IOPscience Russia infra cost study; empirical recal v0.4",
        notes=(
            f"Baseline coverage {baseline:.2f}/5; Arctic-amplified warming "
            f"{warming:.2f}°C (2.5× global mean); progressive active-layer deepening"
        ),
    )


def _dust_storm(region: str, ssp: SSPScenario, year: int,
               is_arid_override: bool | None = None) -> HazardScore:
    """
    Dust storm / haboob / aeolian transport hazard.

    Impacts: equipment abrasion, PV panel soiling (−30 to −50% output), visibility
    reduction, respiratory health, supply-chain disruption. Dryland expansion
    under warming (IPCC AR6 Ch8) amplifies dust generation. Some regions show
    frequency increases (MENA, Mongolia), others mixed signals.

    is_arid_override: when provided from GIS resolver (Köppen zone + WRI Aqueduct
    dryland classification), enables dust-storm risk even for region codes outside
    the DUST_STORM_REGIONS set (e.g. an asset in a dryland enclave within a humid region).

    Sources: WMO SDS-WAS; Ginoux et al 2012; IPCC AR6 Ch8.2; Middleton 2017.
    """
    in_arid = is_arid_override if is_arid_override is not None \
        else (region in DUST_STORM_REGIONS)
    if not in_arid:
        return HazardScore(
            hazard="dust_storm", annual_probability=0.0, severity_index=0.0,
            production_loss_pct=0.0, trend_2030=0.0, trend_2050=0.0,
            data_source="WMO SDS-WAS", applicable=False,
            notes="Region outside dust-storm-prone arid/semi-arid zone",
        )

    _DUST_BASELINE: dict[str, float] = {
        "AU-SA": 3.0, "AU-WA": 2.2, "AU-NT": 2.5,
        "MN-01": 3.5, "CN-NM": 3.2,
        "ZA": 1.5, "CL-02": 1.8,
        "US-TX": 2.0, "US-OK": 1.8,
        "IN-MH": 1.5,
    }
    baseline = _DUST_BASELINE.get(region, 1.5)
    warming = ssp.regional_warming(year, region)
    # Dryland expansion ~ +4% per °C warming (IPCC AR6 Ch8)
    drought_amplifier = 1.0 + warming * 0.08
    severity = min(5.0, baseline * drought_amplifier)
    prob = min(0.85, baseline / 5.0 * drought_amplifier * 0.55)
    loss = min(0.008, max(0.0, (severity - 1.5) * 0.0010))

    w30 = ssp.regional_warming(2030, region)
    w50 = ssp.regional_warming(2050, region)
    return HazardScore(
        hazard="dust_storm",
        annual_probability=round(prob, 4),
        severity_index=round(severity, 2),
        production_loss_pct=round(loss, 4),
        trend_2030=round(min(5.0, baseline * (1 + w30 * 0.08)), 2),
        trend_2050=round(min(5.0, baseline * (1 + w50 * 0.08)), 2),
        data_source="WMO SDS-WAS; Ginoux et al 2012 MODIS dust inventory; IPCC AR6 Ch8.2",
        notes=f"Dust baseline {baseline:.2f}; dryland expansion amplifier {drought_amplifier:.3f}× at +{warming:.2f}°C",
    )


def _hail(region: str, ssp: SSPScenario, year: int) -> HazardScore:
    """
    Large hail impact (crop damage, equipment, solar panels, vehicles).

    Hail frequency-intensity relationship with warming is uncertain. IPCC AR6:
    overall number of hail days may decrease, but frequency of large hail (>2cm)
    may increase due to stronger convective instability. Net risk: stable to slightly
    elevated for assets vulnerable to large hail (PV arrays, crops, structures).

    Sources: IPCC AR6 Ch11.7; Prein & Holland 2018; Allen et al 2020.
    """
    if region not in HAIL_REGIONS:
        return HazardScore(
            hazard="hail", annual_probability=0.0, severity_index=0.0,
            production_loss_pct=0.0, trend_2030=0.0, trend_2050=0.0,
            data_source="IPCC AR6 Ch11.7", applicable=False,
            notes="Region not in mapped hail-prone zone",
        )

    _HAIL_BASELINE: dict[str, float] = {
        "AU-QLD": 2.8, "AU-NSW": 2.5, "AU-VIC": 2.0,
        "US-TX": 3.5, "US-OK": 3.2, "US-WY": 2.5,
        "ZA": 2.2, "IN-MH": 1.8, "BR-PA": 1.5,
    }
    baseline = _HAIL_BASELINE.get(region, 1.5)
    warming = ssp.regional_warming(year, region)
    # Large hail frequency increases ~+5% per °C (convective instability)
    # but total hail days slightly decrease; net for large hail: +3%/°C
    large_hail_mult = 1.0 + warming * 0.03
    severity = min(5.0, baseline * large_hail_mult)
    prob = min(0.70, baseline / 5.0 * 0.45 * large_hail_mult)
    loss = min(0.010, max(0.0, (severity - 1.5) * 0.0010))

    w30 = ssp.regional_warming(2030, region)
    w50 = ssp.regional_warming(2050, region)
    return HazardScore(
        hazard="hail",
        annual_probability=round(prob, 4),
        severity_index=round(severity, 2),
        production_loss_pct=round(loss, 4),
        trend_2030=round(min(5.0, baseline * (1 + w30 * 0.03)), 2),
        trend_2050=round(min(5.0, baseline * (1 + w50 * 0.03)), 2),
        data_source="IPCC AR6 Ch11.7; Prein & Holland 2018; Allen et al 2020 (HAIL-CAESAR)",
        notes=f"Hail baseline {baseline:.2f}; large-hail mult {large_hail_mult:.3f}× at +{warming:.2f}°C",
    )


def _lightning(region: str, ssp: SSPScenario, year: int) -> HazardScore:
    """
    Lightning strike hazard — direct equipment damage and wildfire ignition.

    Lightning flash rate increases ~12% per °C of warming (Romps et al 2014).
    Direct damage to structures, electrical equipment, and control systems.
    Lightning is also a primary wildfire ignition source; in drier conditions
    under warming, a larger fraction of ignitions result in fires.

    Sources: Romps et al 2014 (Science); IPCC AR6 Ch11.7; Thornton et al 2017.
    """
    # Lightning density baseline by region (flashes/km²/yr — TRMM/LIS proxy)
    _LIGHTNING_DENSITY: dict[str, float] = {
        "BR-PA": 4.5,   # Amazon — world's highest lightning density
        "ID-KI": 3.5,   # Maritime Continent
        "IN-MH": 3.0,   # Indian monsoon belt
        "AU-NT": 2.5,   # Darwin — high thunderstorm frequency
        "AU-QLD": 2.0,  # Queensland storm belt
        "ZA": 2.0,      # Southern Africa highveld
        "US-TX": 1.8,   # Gulf Coast convection
        "US-OK": 1.6,   # Great Plains
        "AU-WA": 1.0,
        "CA-AB": 0.8,
        "GB-ENG": 0.4,
        "NL-NH": 0.3,
        "MN-01": 0.6,
        "global": 1.0,
    }
    baseline_density = _LIGHTNING_DENSITY.get(region, _LIGHTNING_DENSITY["global"])
    warming = ssp.regional_warming(year, region)
    # +12% per °C (Romps et al 2014)
    flash_mult = 1.0 + warming * 0.12
    severity = min(5.0, baseline_density / 4.5 * 4.0 * flash_mult)
    prob = min(0.95, baseline_density / 4.5 * 0.60 * flash_mult)
    loss = min(0.005, max(0.0, (severity - 1.0) * 0.0005))

    w30 = ssp.regional_warming(2030, region)
    w50 = ssp.regional_warming(2050, region)
    return HazardScore(
        hazard="lightning",
        annual_probability=round(prob, 4),
        severity_index=round(severity, 2),
        production_loss_pct=round(loss, 4),
        trend_2030=round(min(5.0, baseline_density / 4.5 * 4.0 * (1 + w30 * 0.12)), 2),
        trend_2050=round(min(5.0, baseline_density / 4.5 * 4.0 * (1 + w50 * 0.12)), 2),
        data_source="Romps et al 2014 (Science +12%/°C); TRMM/LIS lightning density; IPCC AR6 Ch11.7",
        notes=f"Flash density {baseline_density:.1f} fl/km²/yr; flash mult {flash_mult:.3f}× at +{warming:.2f}°C",
    )


def _subsidence(region: str, ssp: SSPScenario, year: int) -> HazardScore:
    """
    Ground subsidence — sinking from groundwater extraction, peat oxidation,
    coal/mineral extraction, and clay shrinkage from drought drying.

    Climate link: drought intensification drives increased groundwater extraction
    (demand) and accelerates peat oxidation and clay desiccation.
    Extraction subsidence is largely anthropogenic but climate-amplified.

    Sources: Cigna & Tapete 2021; WRI; InSAR Sentinel-1; IPCC AR6 Ch4.
    """
    if region not in SUBSIDENCE_REGIONS:
        return HazardScore(
            hazard="subsidence", annual_probability=0.0, severity_index=0.0,
            production_loss_pct=0.0, trend_2030=0.0, trend_2050=0.0,
            data_source="Cigna & Tapete 2021; WRI", applicable=False,
            notes="Region not flagged for significant subsidence risk",
        )

    _SUBS_BASELINE: dict[str, float] = {
        "NL-NH": 4.5,    # Peat soils; −10 cm/yr observed (Cigna)
        "ID-KI": 3.5,    # Jakarta groundwater depletion + peat
        "CN-NM": 2.5,    # Coal mine subsidence
        "AU-QLD": 1.5,   # Mine subsidence
        "IN-MH": 2.0,    # Groundwater extraction
        "MN-01": 2.0,    # Mining + permafrost-adjacent
        "US-TX": 1.8,    # Houston subsidence (oil/gas)
        "US-OK": 1.5,    # Induced subsidence
    }
    baseline = _SUBS_BASELINE.get(region, 1.5)
    warming = ssp.regional_warming(year, region)
    # Drought-driven groundwater demand increases ~5% per °C
    drought_amp = 1.0 + warming * 0.05
    severity = min(5.0, baseline * drought_amp)
    prob = min(0.85, baseline / 5.0 * 0.70 * drought_amp)
    loss = min(0.010, max(0.0, (severity - 1.0) * 0.0014))

    w30 = ssp.regional_warming(2030, region)
    w50 = ssp.regional_warming(2050, region)
    return HazardScore(
        hazard="subsidence",
        annual_probability=round(prob, 4),
        severity_index=round(severity, 2),
        production_loss_pct=round(loss, 4),
        trend_2030=round(min(5.0, baseline * (1 + w30 * 0.05)), 2),
        trend_2050=round(min(5.0, baseline * (1 + w50 * 0.05)), 2),
        data_source="Cigna & Tapete 2021 InSAR; WRI; Sentinel-1 ground motion; IPCC AR6 Ch4 (groundwater)",
        notes=f"Subsidence baseline {baseline:.2f}/5; drought amplifier {drought_amp:.3f}× at +{warming:.2f}°C",
    )


def _freeze_thaw_cycle(region: str, ssp: SSPScenario, year: int,
                       elevation_m: float = 300,
                       mean_winter_temp_override: float | None = None) -> HazardScore:
    """
    Freeze-thaw cycle infrastructure damage — pipe bursting, concrete cracking,
    road pavement heaving, bridge-deck damage.

    Warming may INCREASE freeze-thaw cycles at currently cold sites by pushing
    temperatures from "always frozen" into the cycling zone. At currently mild
    cold sites (near 0°C mean winter T), warming reduces cycling frequency.
    This is the same 'danger band' logic as blade icing but applied to infrastructure.

    mean_winter_temp_override: when provided from GIS resolver (ERA5 2°×2° grid),
    replaces the region-code lookup table for more accurate temperature input.

    Sources: IPCC AR6 Ch12.4; ASCE infrastructure resilience; Groisman et al 2006.
    """
    # Sources: ERA5 1991-2020 DJF mean T2M; NOAA Global Climate Normals
    _MEAN_WINTER_T_FT: dict[str, float] = {
        "CA-AB": -12.0, "CA-QC": -14.0, "MN-01": -20.0,
        "CN-NM": -15.0, "US-WY": -8.0, "GB-ENG": 4.0,
        "NL-NH": 3.0, "AU-WA": 14.0, "AU-QLD": 20.0,
        "AU-SA": 8.0, "ZA": 8.0, "CL-02": -2.0,
        "PE-01": -5.0,  # High Andes
        # India states — subtropical/tropical (no freeze-thaw cycling)
        "IN-JH": 15.0, "IN-OD": 18.0, "IN-RJ": 10.0, "IN-GJ": 16.0,
        "IN-AP": 22.0, "IN-KA": 22.0, "IN-MP": 13.0,
        # UK / Europe
        "GB-WLS": 5.0,   # Port Talbot — maritime mild, occasional near-zero nights
        "GB-SCT": 3.0,   # Scotland — cool maritime, mild FT events
        "DE-HH": 1.0,    # Hamburg — near-zero winters, genuine FT cycling
        "ES-CT": 9.0,    # Barcelona — mild Mediterranean, minimal FT
        "FR-IDF": 4.0,   # Paris — some FT events most winters
        # US states
        "US-FL": 20.0,   # Miami — tropical, no FT
        "US-GA": 6.0,    # Atlanta — some near-freeze events
        "US-NY": 0.0,    # New York — genuine FT cycling, significant infrastructure risk
        "US-CA": 14.0,   # LA — mild, no FT
        "US-MN": -12.0,  # Minneapolis — very cold (less cycling; stays frozen)
        "US-WA": 5.0,    # Seattle — maritime mild, occasional FT
    }
    # Regions too hot for freeze-thaw cycling (mean winter T reliably > 12°C)
    _NO_FT = {
        "AU-QLD", "AU-NT", "IN-MH", "ID-KI", "BR-PA",
        # India states (subtropical; no sustained below-0°C temperatures)
        "IN-JH", "IN-OD", "IN-GJ", "IN-AP", "IN-KA", "IN-MP",
        # US/tropical
        "US-FL", "US-CA",
    }
    # Use GIS mean winter T override when available
    baseline_mwt = (
        mean_winter_temp_override
        if mean_winter_temp_override is not None
        else _MEAN_WINTER_T_FT.get(region, -3.0)
    )
    # Skip hot regions (mean winter T > 12°C) unless GIS reveals otherwise
    if region in _NO_FT and elevation_m < 2000 and baseline_mwt > 10.0:
        return HazardScore(
            hazard="freeze_thaw_cycle", annual_probability=0.0, severity_index=0.0,
            production_loss_pct=0.0, trend_2030=0.0, trend_2050=0.0,
            data_source="IPCC AR6 Ch12.4", applicable=False,
            notes="Region too warm for freeze-thaw cycling",
        )

    warming = ssp.regional_warming(year, region)
    mwt = baseline_mwt + warming
    # Freeze-thaw cycles peak near 0°C mean winter T
    danger_center, danger_std = 0.0, 6.0
    ft_exposure = math.exp(-0.5 * ((mwt - danger_center) / danger_std) ** 2)
    severity = min(5.0, ft_exposure * 4.5)
    prob = min(0.90, ft_exposure * 0.75)
    loss = min(0.008, max(0.0, ft_exposure * 0.0080))

    w30 = ssp.regional_warming(2030, region)
    w50 = ssp.regional_warming(2050, region)
    mwt0 = baseline_mwt   # already resolved (GIS or region table)
    ft30 = math.exp(-0.5 * ((mwt0 + w30 - danger_center) / danger_std) ** 2)
    ft50 = math.exp(-0.5 * ((mwt0 + w50 - danger_center) / danger_std) ** 2)
    return HazardScore(
        hazard="freeze_thaw_cycle",
        annual_probability=round(prob, 4),
        severity_index=round(severity, 2),
        production_loss_pct=round(loss, 4),
        trend_2030=round(min(5.0, ft30 * 4.5), 2),
        trend_2050=round(min(5.0, ft50 * 4.5), 2),
        data_source="IPCC AR6 Ch12.4; ASCE; Groisman et al 2006; WMO lapse-rate",
        notes=(
            f"Mean winter T {mwt0:.1f}°C + {warming:.2f}°C warming = {mwt:.1f}°C; "
            f"FT exposure {ft_exposure:.3f} (danger band {danger_center}°C±{danger_std}°C)"
        ),
    )


def _compound_flood(region: str, ssp: SSPScenario, year: int,
                    is_coastal: bool, elevation_m: float,
                    coastal_factor: float = 1.0) -> HazardScore:
    """
    Compound flood — simultaneous or sequential riverine + coastal storm surge,
    amplified by sea level rise.

    Compound events are non-additive: joint probability is lower than the product
    of individual probabilities, but damage severity is higher than either alone.
    IPCC AR6: compound flooding frequency will increase significantly in low-lying
    coastal delta regions as both SLR and extreme precipitation intensify.

    Sources: IPCC AR6 Ch11.4+9; Zscheischler et al 2020 (compound events);
             Ward et al 2018 (global compound flood risk).
    """
    if elevation_m > 60 or coastal_factor < 0.15:
        return HazardScore(
            hazard="compound_flood", annual_probability=0.0, severity_index=0.0,
            production_loss_pct=0.0, trend_2030=0.0, trend_2050=0.0,
            data_source="IPCC AR6 Ch11.4; Ward et al 2018", applicable=False,
            notes=f"Elevation {elevation_m:.0f}m or coastal_factor {coastal_factor:.2f} too low for compound flood",
        )

    wri = WRI_BASELINE.get(region, WRI_BASELINE["global"])
    riverine_base = wri["riverine_flood"]
    coastal_base = wri["coastal_flood"]
    warming = ssp.regional_warming(year, region)

    # Compound severity > either individual event; model as geometric mean × amplifier
    compound_amplifier = 1.30  # 30% severity uplift for co-occurrence
    base_severity = math.sqrt(riverine_base * max(coastal_base, 0.1)) * compound_amplifier
    slr_cm = max(0, 20 + warming * 8)   # SLR proxy (cm) — raises compound frequency
    slr_freq_mult = 1 + slr_cm / 100 * 0.8
    severity = min(5.0, base_severity * slr_freq_mult * coastal_factor)
    prob = min(0.80, severity / 5.0 * 0.35 * coastal_factor)
    loss = min(0.025, max(0.0, (severity - 1.5) * 0.0045))

    w30 = ssp.regional_warming(2030, region)
    w50 = ssp.regional_warming(2050, region)
    slr30 = max(0, 20 + w30 * 8)
    slr50 = max(0, 20 + w50 * 8)
    return HazardScore(
        hazard="compound_flood",
        annual_probability=round(prob, 4),
        severity_index=round(severity, 2),
        production_loss_pct=round(loss, 4),
        trend_2030=round(min(5.0, base_severity * (1 + slr30 / 100 * 0.8) * coastal_factor), 2),
        trend_2050=round(min(5.0, base_severity * (1 + slr50 / 100 * 0.8) * coastal_factor), 2),
        data_source="IPCC AR6 Ch11.4+9; Zscheischler et al 2020; Ward et al 2018 compound flood",
        notes=(
            f"Riverine {riverine_base:.2f} + Coastal {coastal_base:.2f} base; "
            f"compound amplifier {compound_amplifier}×; SLR proxy {slr_cm:.0f}cm; "
            f"coastal factor {coastal_factor:.2f}"
        ),
    )


def _avalanche(region: str, ssp: SSPScenario, year: int,
               elevation_m: float = 300) -> HazardScore:
    """
    Snow avalanche risk for mountain-terrain assets and access infrastructure.

    Climate link: warming alters snowpack characteristics (wetter, heavier snow
    → increased wet-slab avalanche risk) and accelerates snow-season shortening.
    Some research suggests warming increases wet avalanche frequency while
    decreasing dry-slab frequency. Net risk for operational assets: stable to
    slightly elevated in near-term, declining long-term as snowpack diminishes.

    Sources: Hock et al 2019 (IPCC SROCC Ch2); Sinickas et al 2016;
             SLF/WSL Avalanche Atlas; Techel et al 2020.
    """
    if region not in AVALANCHE_REGIONS or elevation_m < 800:
        return HazardScore(
            hazard="avalanche", annual_probability=0.0, severity_index=0.0,
            production_loss_pct=0.0, trend_2030=0.0, trend_2050=0.0,
            data_source="IPCC SROCC Ch2; SLF/WSL", applicable=False,
            notes="Region not in avalanche zone or elevation < 800m",
        )

    _AVL_BASELINE: dict[str, float] = {
        "CL-02": 3.2,  # High Andes mining routes
        "PE-01": 2.8,  # Tropical Andes (rapidly losing glacial bonding)
        "CA-AB": 2.5,  # Columbia/Rockies
        "US-WY": 2.2,  # Wyoming ranges
        "MN-01": 2.0,  # Altai mountains
    }
    baseline = _AVL_BASELINE.get(region, 1.5)
    warming = ssp.regional_warming(year, region)
    # Near-term: wet avalanche frequency +10%/°C; long-term snowpack loss −15%/°C
    # Net near-2050 effect: stable to slight decline
    elev_factor = min(1.5, elevation_m / 1500)
    wet_avl_mult = 1.0 + warming * 0.05   # net of wet-avl increase and snowpack loss
    severity = min(5.0, baseline * elev_factor * wet_avl_mult)
    prob = min(0.70, baseline / 5.0 * elev_factor * 0.40 * wet_avl_mult)
    loss = min(0.015, max(0.0, (severity - 1.0) * 0.0018))

    w30 = ssp.regional_warming(2030, region)
    w50 = ssp.regional_warming(2050, region)
    return HazardScore(
        hazard="avalanche",
        annual_probability=round(prob, 4),
        severity_index=round(severity, 2),
        production_loss_pct=round(loss, 4),
        trend_2030=round(min(5.0, baseline * elev_factor * (1 + w30 * 0.05)), 2),
        trend_2050=round(min(5.0, baseline * elev_factor * (1 + w50 * 0.05)), 2),
        data_source="IPCC SROCC Ch2; Hock et al 2019; SLF/WSL Avalanche Atlas; Sinickas et al 2016",
        notes=f"Avalanche baseline {baseline:.2f}; elevation factor {elev_factor:.2f}×; wet-avl mult {wet_avl_mult:.3f}×",
    )


def _marine_heatwave(region: str, ssp: SSPScenario, year: int,
                     is_coastal: bool, coastal_factor: float = 1.0) -> HazardScore:
    """
    Marine heatwave (MHW) — anomalously warm sea surface temperature events.

    Impacts marine operations, aquaculture, port infrastructure cooling, desalination
    intake temperatures, and coral bleaching (affecting fisheries/biodiversity).
    MHW frequency has doubled since 1982; under SSP3-7.0, events that are now
    'extreme' will become 'permanent conditions' by 2100.

    Sources: Oliver et al 2021 (MHW review); Hobday et al 2018; IPCC SROCC Ch6;
             Frölicher et al 2018 (Nature).
    """
    if region not in MARINE_HEATWAVE_REGIONS or (not is_coastal and coastal_factor < 0.20):
        return HazardScore(
            hazard="marine_heatwave", annual_probability=0.0, severity_index=0.0,
            production_loss_pct=0.0, trend_2030=0.0, trend_2050=0.0,
            data_source="Oliver et al 2021; IPCC SROCC", applicable=False,
            notes="Region not coastal or not in MHW-elevated zone",
        )

    _MHW_BASELINE: dict[str, float] = {
        "AU-WA": 3.5,   # SE Indian Ocean — most intense MHW hotspot
        "AU-QLD": 3.0,  # GBR bleaching events
        "AU-SA": 2.0,   # Southern Ocean / Spencer Gulf
        "ZA": 2.0,      # Agulhas current MHW
        "ID-KI": 2.5,   # Coral Triangle thermal sensitivity
        "IN-MH": 2.2,   # Arabian Sea warming
        "GB-ENG": 1.5,  # North Sea anomalous warming
    }
    baseline = _MHW_BASELINE.get(region, 1.5)
    warming = ssp.regional_warming(year, region)
    # MHW frequency scales non-linearly: frequency doubles per 0.5°C of ocean warming
    # Ocean warms at ~0.65× land rate
    ocean_warming = warming * 0.65
    mhw_freq_mult = 2.0 ** (ocean_warming / 0.5)   # doubling per 0.5°C ocean warming
    severity = min(5.0, baseline * min(mhw_freq_mult, 3.0) * coastal_factor)
    prob = min(0.95, baseline / 5.0 * min(mhw_freq_mult, 3.0) * 0.50 * coastal_factor)
    loss = min(0.010, max(0.0, (severity - 1.5) * 0.0014))

    w30 = ssp.regional_warming(2030, region)
    w50 = ssp.regional_warming(2050, region)
    fm30 = 2.0 ** (w30 * 0.65 / 0.5)
    fm50 = 2.0 ** (w50 * 0.65 / 0.5)
    return HazardScore(
        hazard="marine_heatwave",
        annual_probability=round(prob, 4),
        severity_index=round(severity, 2),
        production_loss_pct=round(loss, 4),
        trend_2030=round(min(5.0, baseline * min(fm30, 3.0) * coastal_factor), 2),
        trend_2050=round(min(5.0, baseline * min(fm50, 3.0) * coastal_factor), 2),
        data_source="Oliver et al 2021; Frölicher et al 2018 Nature; IPCC SROCC Ch6; Hobday et al 2018",
        notes=(
            f"MHW baseline {baseline:.2f}; ocean warming {ocean_warming:.2f}°C; "
            f"MHW freq mult {mhw_freq_mult:.2f}× (doubles per 0.5°C ocean warming)"
        ),
    )


def _glof(region: str, ssp: SSPScenario, year: int) -> HazardScore:
    """
    Glacial Lake Outburst Flood (GLOF) — high-energy debris flows from sudden
    moraine dam failure or ice-dam release of glacial lakes.

    GLOF risk INCREASES with warming as glacier retreat creates new and larger
    glacial lakes. Pro-glacial lake area has increased 51% since 1990 globally
    (IPCC SROCC). Downstream assets in valley corridors face catastrophic,
    low-probability, extreme-impact events.

    Sources: IPCC SROCC Ch2.3; Emmer et al 2022; Veh et al 2020;
             Richardson & Reynolds 2000; Carrivick & Tweed 2016.
    """
    if region not in GLOF_REGIONS:
        return HazardScore(
            hazard="glof", annual_probability=0.0, severity_index=0.0,
            production_loss_pct=0.0, trend_2030=0.0, trend_2050=0.0,
            data_source="IPCC SROCC Ch2.3; Emmer et al 2022", applicable=False,
            notes="Region not in mapped GLOF-susceptible zone",
        )

    _GLOF_BASELINE: dict[str, float] = {
        "PE-01": 3.5,   # Tropical Andes — highest historical GLOF frequency
        "CL-02": 2.8,   # Southern Andes — active glacier retreat
        "CA-AB": 2.0,   # Rocky Mountain glacier systems
        "IN-MH": 2.5,   # Himalayan fringes (downstream)
        "MN-01": 1.8,   # Altai glacial lakes (growing)
    }
    baseline = _GLOF_BASELINE.get(region, 1.5)
    warming = ssp.regional_warming(year, region)
    # GLOF risk increases with glacier retreat (~+15%/°C as new lakes form)
    glacial_lake_growth = 1.0 + warming * 0.15
    severity = min(5.0, baseline * glacial_lake_growth)
    # GLOF probability remains low but rising (low-frequency, high-consequence)
    prob = min(0.15, baseline / 5.0 * 0.08 * glacial_lake_growth)
    loss = min(0.018, max(0.0, (severity - 1.5) * 0.0035))

    w30 = ssp.regional_warming(2030, region)
    w50 = ssp.regional_warming(2050, region)
    return HazardScore(
        hazard="glof",
        annual_probability=round(prob, 4),
        severity_index=round(severity, 2),
        production_loss_pct=round(loss, 4),
        trend_2030=round(min(5.0, baseline * (1 + w30 * 0.15)), 2),
        trend_2050=round(min(5.0, baseline * (1 + w50 * 0.15)), 2),
        data_source="IPCC SROCC Ch2.3; Emmer et al 2022; Veh et al 2020; Carrivick & Tweed 2016",
        notes=(
            f"GLOF baseline {baseline:.2f}; glacial lake growth factor {glacial_lake_growth:.3f}× "
            f"at +{warming:.2f}°C warming (IPCC SROCC +51% lake area since 1990)"
        ),
    )


def _tornado(region: str, ssp: SSPScenario, year: int) -> HazardScore:
    """
    Tornado — rotating convective winds (EF-scale events).

    Strong constraint: tornadoes require specific atmospheric instability conditions
    (CAPE + wind shear). IPCC AR6: modest frequency changes uncertain, but poleward
    expansion of tornado-prone zones possible. Some studies project US tornado alley
    shift eastward. Impacts are highly localised but extremely destructive.

    Sources: IPCC AR6 Ch11.7; Tippett et al 2016; Allen & Tippett 2015;
             NOAA SPC 1950-2024 climatology.
    """
    if region not in TORNADO_REGIONS:
        return HazardScore(
            hazard="tornado", annual_probability=0.0, severity_index=0.0,
            production_loss_pct=0.0, trend_2030=0.0, trend_2050=0.0,
            data_source="IPCC AR6 Ch11.7; NOAA SPC", applicable=False,
            notes="Region not in mapped tornado-susceptible zone",
        )

    _TORNADO_BASELINE: dict[str, float] = {
        "US-TX": 4.0,    # 150+ tornadoes/yr in Texas historically
        "US-OK": 4.5,    # Oklahoma — highest tornado density globally
        "AU-QLD": 0.8,   # Rare but documented Australian tornadoes
    }
    baseline = _TORNADO_BASELINE.get(region, 1.0)
    warming = ssp.regional_warming(year, region)
    # CAPE increases with warming (~+7%/°C); wind shear changes uncertain
    # Net tornado risk signal uncertain; assume modest +2%/°C
    cape_mult = 1.0 + warming * 0.02
    severity = min(5.0, baseline * cape_mult)
    prob = min(0.80, baseline / 5.0 * 0.55 * cape_mult)
    loss = min(0.012, max(0.0, (severity - 1.0) * 0.0018))

    w30 = ssp.regional_warming(2030, region)
    w50 = ssp.regional_warming(2050, region)
    return HazardScore(
        hazard="tornado",
        annual_probability=round(prob, 4),
        severity_index=round(severity, 2),
        production_loss_pct=round(loss, 4),
        trend_2030=round(min(5.0, baseline * (1 + w30 * 0.02)), 2),
        trend_2050=round(min(5.0, baseline * (1 + w50 * 0.02)), 2),
        data_source="IPCC AR6 Ch11.7; Tippett et al 2016; NOAA SPC 1950-2024 climatology",
        notes=f"Tornado baseline {baseline:.2f}; CAPE mult {cape_mult:.3f}× at +{warming:.2f}°C",
    )


# ---------------------------------------------------------------------------
# Real-time satellite / disaster observation overlay
# ---------------------------------------------------------------------------

def _fetch_current_observations(
    lat: float,
    lon: float,
) -> tuple[dict, list[str]]:
    """Query NASA FIRMS (active fire) and GDACS (active flood/cyclone/tsunami)
    for current hazard observations near (lat, lon).

    Called only at ``_depth == 0`` and only for near-term years (≤ 2028) so
    that real-time observations don't pollute long-horizon scenario projections.

    Returns
    -------
    observed_alerts : dict[str, bool]
        Hazard name → True if an active event was detected.
    observation_sources : list[str]
        Data source strings for provenance record.

    Both collections are empty dicts/lists if all connectors fail.
    Failures are silenced — observations are supplementary, never blocking.
    """
    alerts: dict = {}
    sources: list = []

    # ── NASA FIRMS active fire (wildfire hazard) ─────────────────────────────
    try:
        from ..connectors.nasa_firms import get_active_fire
        fire_obs = get_active_fire(lat, lon)
        if fire_obs is not None:
            alerts["wildfire"] = fire_obs.active_fire
            sources.append(fire_obs.source)
    except Exception:
        pass

    # ── GDACS active flood / cyclone / tsunami ────────────────────────────────
    try:
        from ..connectors.gdacs import get_active_disasters
        disaster_obs = get_active_disasters(lat, lon)
        if disaster_obs is not None:
            if disaster_obs.active_flood:
                alerts["flood_riverine"] = True
            if disaster_obs.active_cyclone:
                alerts["cyclone"] = True
            if disaster_obs.active_tsunami:
                alerts["coastal_flooding"] = True
            sources.append(disaster_obs.source)
    except Exception:
        pass

    return alerts, sources


# ---------------------------------------------------------------------------
# Main assessment engine
# ---------------------------------------------------------------------------

class PhysicalHazardEngine:
    """
    Compute full 25-hazard physical risk profile for an asset (v0.4).

    Usage:
        engine = PhysicalHazardEngine()
        profile = engine.assess(
            asset_id="pilbara_001",
            asset_name="Pilbara Iron Ore",
            region="AU-WA",
            lat=-22.5, lon=118.8,
            ssp="ssp370",    # or use ngfs_family="current_policies"
            year=2030,
        )
    """

    def assess(
        self,
        asset_id: str,
        asset_name: str,
        region: str,
        year: int,
        ssp: str | None = None,
        ngfs_family: str | None = None,
        lat: float | None = None,
        lon: float | None = None,
        elevation_override: float | None = None,
        equipment_type: str | None = None,
        _depth: int = 0,   # internal recursion guard — do NOT pass externally
    ) -> AssetHazardProfile:
        """
        Assess all hazards for one asset at one point in time.

        Args:
            asset_id: Unique identifier for the asset.
            asset_name: Human-readable name.
            region: CRI region code (e.g., "AU-WA").
            year: Assessment year (2026–2050).
            ssp: SSP scenario id (e.g., "ssp370"). Overrides ngfs_family.
            ngfs_family: NGFS scenario family — translated to SSP if ssp not given.
            lat, lon: Asset coordinates (decimal degrees). Enables GIS-based
                      spatial resolution (elevation, coastal distance, permafrost
                      zone, cyclone belt, Köppen zone, mean winter temperature).
            elevation_override: Override the region default elevation (metres).
            equipment_type: Asset equipment class (e.g. "open_pit_mine", "wind_farm").
                            Activates per-hazard sensitivity multipliers from the
                            GIS equipment sensitivity matrix.
        """
        # Resolve SSP
        if ssp:
            scenario = SSP_SCENARIOS.get(ssp, SSP_SCENARIOS["ssp245"])
        elif ngfs_family:
            scenario = ngfs_to_ssp(ngfs_family)
        else:
            scenario = SSP_SCENARIOS["ssp245"]

        # ── GIS resolution (lat/lon → spatial attributes) ─────────────────────
        # When lat/lon are provided, the GIS resolver augments/overrides the
        # region-code lookup tables with real spatial data from embedded grids.
        gis_attrs = None
        gis_mean_winter_temp: float | None = None
        gis_is_permafrost: bool | None = None
        gis_is_cyclone_belt: bool | None = None
        gis_is_arid: bool | None = None
        gis_equipment_sensitivity: dict[str, float] = {}

        if lat is not None and lon is not None and _depth == 0:
            try:
                from .gis import resolve as _gis_resolve
                gis_attrs = _gis_resolve(lat, lon, equipment_type)
                gis_mean_winter_temp = gis_attrs.mean_winter_temp
                gis_is_permafrost = gis_attrs.is_permafrost
                gis_is_cyclone_belt = gis_attrs.is_cyclone_belt
                gis_is_arid = gis_attrs.is_arid
                gis_equipment_sensitivity = gis_attrs.equipment_sensitivity
                # GIS elevation overrides region table when provided
                if elevation_override is None:
                    elevation_override = float(gis_attrs.elevation_m)
            except Exception:
                # GIS resolver failure is non-fatal — fall back to region tables
                gis_attrs = None

        # Resolve spatial context — uses lat/lon for asset-level downscaling
        spatial = _resolve_spatial_context(lat, lon, region, elevation_override)
        elevation = spatial["elevation_m"]
        is_coastal = spatial["is_coastal"]
        coastal_factor = spatial["coastal_factor"]
        lulc = REGION_LULC.get(region, LULCType.GRASSLAND)

        # Override is_coastal from GIS coastal distance when GIS available
        if gis_attrs is not None:
            if gis_attrs.coastal_km < 50:
                is_coastal = True
                coastal_factor = max(coastal_factor, 1.0 + (50 - gis_attrs.coastal_km) / 50.0 * 0.4)
            elif gis_attrs.coastal_km > 300:
                is_coastal = False

        # Sub-grid correction factors (elevation relative to 25km cell mean)
        sg = spatial["subgrid"]

        # ── Live coordinate-level climate data (NASA POWER + Open-Meteo CMIP6) ──────
        # Only fetched at depth=0 (user-requested year) to avoid duplicate API calls
        # in recursive trend/critical-year sub-assessments.  Gracefully degrades to
        # lookup-table path if either API is unavailable.
        live_data: Optional[dict] = None
        live_warming_c: Optional[float] = None
        live_baseline_t2m_max_c: Optional[float] = None
        live_precip_delta_pct: Optional[float] = None
        if lat is not None and lon is not None and _depth == 0:
            live_data = _fetch_live_climate(lat, lon, year, scenario.id)
            if live_data["live"]:
                live_warming_c = live_data["warming_c"]
                live_baseline_t2m_max_c = live_data["baseline_t2m_max_c"]
                live_precip_delta_pct = live_data.get("precip_delta_pct")

        # ── Assemble live baseline / projection metadata for audit trail ─────────────
        live_baseline_meta: Optional[dict] = None
        live_projection_meta: Optional[dict] = None
        if live_data is not None:
            if live_data.get("baseline_t2m_max_c") is not None:
                live_baseline_meta = {
                    "t2m_max_c": live_data["baseline_t2m_max_c"],
                    "precip_mm_day": live_data["baseline_precip_mm_day"],
                    "source": "NASA POWER MERRA-2 2001-2020 climatological normal",
                }
            live_projection_meta = {
                "warming_c": live_data["warming_c"],
                "precip_delta_pct": live_data.get("precip_delta_pct"),
                "data_source": live_data["data_source"],
                "live": live_data["live"],
            }

        # ── Extend downscaling_method description when live API succeeded ────────────
        downscaling_str = spatial["downscaling_method"]
        if live_data is not None and live_data["live"]:
            downscaling_str += (
                " Step 5 — Live coordinate-level API downscaling: "
                f"NASA POWER MERRA-2 (2001-2020) observed T2M_MAX "
                f"{live_data['baseline_t2m_max_c']:.1f} deg C and precipitation "
                f"{live_data['baseline_precip_mm_day']:.3f} mm/day fetched at asset "
                f"coordinates ({lat:.4f} N, {lon:.4f} E). "
                f"Open-Meteo CMIP6 ({live_data['data_source'].split(' ')[2] if ' ' in live_data['data_source'] else 'MRI-AGCM3.2-S'}) "
                f"future projection scaled to {scenario.id} using IPCC AR6 WG1 Table 4.5 GMST ratios; "
                f"warming delta {live_data['warming_c']:.2f} deg C above observed baseline. "
                "Heat stress anchored to coordinate-level observed climatology; "
                "riverine flood driven by CMIP6 precipitation signal at asset grid cell."
            )
        elif live_data is not None and not live_data["live"]:
            downscaling_str += (
                f" Step 5 — Live API attempted but unavailable ({live_data['data_source']}); "
                "steps 1-4 applied using lookup-table baseline."
            )

        # Compute all 25 hazards with sub-grid downscaling + GIS overrides applied
        hazards: dict[str, HazardScore] = {
            # ── Core 10 (original) ────────────────────────────────────────────
            "heat_stress":            _heat_stress(region, scenario, year,
                                          heat_factor=sg["heat_factor"],
                                          live_warming_c=live_warming_c,
                                          live_baseline_t2m_max_c=live_baseline_t2m_max_c),
            "flood_riverine":         _flood_riverine(region, scenario, year, lulc,
                                          flood_factor=sg["flood_factor"],
                                          live_precip_delta_pct=live_precip_delta_pct),
            "flood_coastal":          _flood_coastal(region, scenario, year, is_coastal, elevation, coastal_factor),
            "sea_level_rise":         _sea_level_rise(region, scenario, year, is_coastal, elevation, coastal_factor),
            "saltwater_intrusion":    _saltwater_intrusion(region, scenario, year, is_coastal, elevation, coastal_factor),
            "landslide":              _landslide(region, scenario, year, elevation, lulc),
            "wildfire":               _wildfire(region, scenario, year, lulc),
            # GIS: is_cyclone_belt_override enables risk when IBTrACS confirms belt
            "cyclone":                _cyclone(region, scenario, year, is_coastal,
                                          is_cyclone_belt_override=gis_is_cyclone_belt),
            "drought":                _drought(region, scenario, year),
            "water_stress":           _water_stress(region, scenario, year,
                                          water_stress_factor=sg["water_stress_factor"]),
            # ── New 15 (v0.4) — GIS overrides passed where relevant ───────────
            "extreme_cold":           _extreme_cold(region, scenario, year, elevation_m=elevation),
            # GIS: ERA5 mean winter temp overrides region lookup for icing Gaussian
            # equipment_type gate: only applicable to wind_farm / transmission_line
            "blade_icing":            _blade_icing(region, scenario, year, elevation_m=elevation,
                                          mean_winter_temp_override=gis_mean_winter_temp,
                                          equipment_type=equipment_type),
            "extratropical_cyclone":  _extratropical_cyclone(region, scenario, year, elevation_m=elevation),
            "flash_flood":            _flash_flood(region, scenario, year, lulc, elevation_m=elevation),
            # GIS: NSIDC permafrost extent enables risk beyond PERMAFROST_REGIONS set
            "permafrost_thaw":        _permafrost_thaw(region, scenario, year,
                                          is_permafrost_override=gis_is_permafrost),
            # GIS: Köppen + WRI Aqueduct dryland flag extends dust-storm applicability
            "dust_storm":             _dust_storm(region, scenario, year,
                                          is_arid_override=gis_is_arid),
            "hail":                   _hail(region, scenario, year),
            "lightning":              _lightning(region, scenario, year),
            "subsidence":             _subsidence(region, scenario, year),
            # GIS: ERA5 mean winter temp overrides region lookup for FT Gaussian
            "freeze_thaw_cycle":      _freeze_thaw_cycle(region, scenario, year,
                                          elevation_m=elevation,
                                          mean_winter_temp_override=gis_mean_winter_temp),
            "compound_flood":         _compound_flood(region, scenario, year, is_coastal, elevation, coastal_factor),
            "avalanche":              _avalanche(region, scenario, year, elevation_m=elevation),
            "marine_heatwave":        _marine_heatwave(region, scenario, year, is_coastal, coastal_factor),
            "glof":                   _glof(region, scenario, year),
            "tornado":                _tornado(region, scenario, year),
        }

        # ── Equipment sensitivity multipliers (GIS resolver) ─────────────────
        # Apply per-hazard sensitivity multipliers from the equipment type matrix.
        # Multipliers > 1 amplify production_loss_pct; < 1 attenuate it.
        # This reflects that an open-pit mine has different vulnerability to heat
        # than a wind farm, even at the same location.
        if gis_equipment_sensitivity:
            for hazard_key, score in hazards.items():
                mult = gis_equipment_sensitivity.get(hazard_key, 1.0)
                if mult != 1.0 and score.applicable:
                    # Clamp: multiplier can at most double the base loss (prevents runaway)
                    new_loss = round(
                        min(score.production_loss_pct * mult,
                            score.production_loss_pct * 2.0),
                        5,
                    )
                    score.production_loss_pct = new_loss

        # Joint annual production loss (independent events approximation)
        # P(loss) = 1 - ∏(1 - p_i) per hazard
        survival = 1.0
        for h in hazards.values():
            if h.applicable:
                survival *= (1.0 - h.production_loss_pct)
        total_loss = 1.0 - survival

        # Physical risk composite score 0–100
        applicable = [h for h in hazards.values() if h.applicable]
        if applicable:
            avg_severity = sum(h.severity_index for h in applicable) / len(applicable)
            score = min(100.0, avg_severity * 20)
        else:
            score = 0.0

        # Top hazards by severity (always computed)
        top = sorted(
            [(k, v.severity_index) for k, v in hazards.items() if v.applicable and v.severity_index > 1.0],
            key=lambda x: x[1], reverse=True
        )[:3]
        top_names = [t[0].replace("_", " ").title() for t in top]

        # Peak 2050 loss — only computed at top-level call to avoid recursion
        if _depth == 0 and year != 2050:
            peak_profile = self.assess(
                asset_id, asset_name, region, 2050,
                ssp=scenario.id, lat=lat, lon=lon,
                elevation_override=elevation_override,
                equipment_type=equipment_type,
                _depth=_depth + 1,
            )
            peak_loss = peak_profile.annual_loss_pct
        else:
            peak_loss = total_loss

        # Critical year (when cumulative loss first exceeds 5%)
        # Only computed at top-level to prevent recursive self-calls
        critical_year = None
        if _depth == 0:
            for yr in range(year, 2051, 5):
                p = self.assess(
                    asset_id, asset_name, region, yr,
                    ssp=scenario.id, lat=lat, lon=lon,
                    elevation_override=elevation_override,
                    equipment_type=equipment_type,
                    _depth=_depth + 1,
                )
                if p.annual_loss_pct >= 0.05:
                    critical_year = yr
                    break

        # Real-time satellite / disaster observations
        # Only consulted at _depth==0 (full resolution) for near-term years
        # (≤ 2028) — beyond that, scenario projections are authoritative.
        observed_alerts: dict = {}
        observation_sources: list = []
        if _depth == 0 and lat is not None and lon is not None and year <= 2028:
            observed_alerts, observation_sources = _fetch_current_observations(
                lat, lon
            )
            # Escalate loss fractions for hazards with confirmed active events.
            # Conservative 50% escalation capped at 2× the base value to prevent
            # runaway scores from transient events (e.g., a fire 40 km away).
            for haz_name, is_active in observed_alerts.items():
                if is_active and haz_name in hazards and hazards[haz_name].applicable:
                    h = hazards[haz_name]
                    escalated = min(
                        h.production_loss_pct * 1.5,
                        h.production_loss_pct * 2.0,
                    )
                    h.production_loss_pct = round(escalated, 5)
                    h.notes = (h.notes or "") + " [satellite: active event confirmed]"

            # Recompute total loss and score after escalation
            if observed_alerts:
                survival = 1.0
                for h in hazards.values():
                    if h.applicable:
                        survival *= (1.0 - h.production_loss_pct)
                total_loss = 1.0 - survival

                applicable = [h for h in hazards.values() if h.applicable]
                if applicable:
                    avg_severity = sum(h.severity_index for h in applicable) / len(applicable)
                    score = min(100.0, avg_severity * 20)

        return AssetHazardProfile(
            asset_id=asset_id,
            asset_name=asset_name,
            region=region,
            lat=lat,
            lon=lon,
            elevation_m=elevation,
            is_coastal=is_coastal,
            coastal_factor=coastal_factor,
            lulc_type=lulc,
            ssp=scenario.id,
            hazards=hazards,
            physical_risk_score=round(score, 1),
            annual_loss_pct=round(total_loss, 4),
            peak_loss_2050_pct=round(peak_loss, 4),
            top_hazards=top_names,
            critical_year=critical_year,
            spatial_resolution=spatial["spatial_resolution"],
            downscaling_method=downscaling_str,
            live_baseline=live_baseline_meta,
            live_projection=live_projection_meta,
            observed_alerts=observed_alerts,
            observation_sources=observation_sources,
        )

    def assess_trajectory(
        self,
        asset_id: str,
        asset_name: str,
        region: str,
        ssp: str,
        years: list[int],
        lat: float | None = None,
        lon: float | None = None,
        compute_critical_year: bool = False,
    ) -> dict[int, AssetHazardProfile]:
        """
        Assess hazard profiles across a list of years.

        Args:
            compute_critical_year: If True, compute critical_year for each year
                (expensive — triggers extra sub-calls per year). Usually False
                for trajectory use; call assess() directly for a single-year
                result with full critical_year computation.
        """
        depth = 0 if compute_critical_year else 1
        return {
            yr: self.assess(
                asset_id, asset_name, region, yr,
                ssp=ssp, lat=lat, lon=lon,
                _depth=depth,
            )
            for yr in years
        }
