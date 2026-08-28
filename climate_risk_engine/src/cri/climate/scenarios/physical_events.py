"""
Physical Climate Event Library.

Defines a canonical set of 20 physical climate events that the scenario
cascade engine can apply to any company and asset portfolio.  Each event
is a compound climate signal — not a single hazard, but a coherent physical
driver that simultaneously intensifies certain hazards and suppresses others.

DESIGN PRINCIPLES
─────────────────
• Events model REAL climate phenomena, not abstract "worst cases."
  Every event has historical analogs drawn from the observational record.
• hazard_multipliers scale the baseline HazardScore.severity_index from
  PhysicalHazardEngine.  A value of 2.5 means a hazard that normally sits
  at severity 2.0 will reach 5.0 (ceiling at 5.0).
• hazard_floors set minimum severity regardless of the asset's baseline.
  Useful for hazards that are essentially guaranteed during this event
  even in regions with low chronic exposure (e.g., cyclone flooding in
  a historically low-flood zone).
• duration_months drives the temporal extent of production loss and
  emergency response cost calculations.
• acute=True events are episodic (days–weeks); acute=False are seasonal
  or multi-year conditions.

SOURCES
───────
• NOAA ENSO historical data (www.psl.noaa.gov)
• WMO State of the Global Climate Reports 2015–2023
• EM-DAT International Disaster Database (emdat.be)
• Swiss Re sigma: Natural catastrophes 2022/2023
• Munich Re NatCatSERVICE database
• IPCC AR6 WG1 Ch11 (Weather and Climate Extremes)
• Diffenbaugh et al. (2017): Quantifying the influence of global warming
  on unprecedented extreme climate events. PNAS.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class EventDriver(str, Enum):
    """Primary atmospheric / oceanic driver of the event."""
    ENSO_EL_NINO = "enso_el_nino"
    ENSO_LA_NINA = "enso_la_nina"
    HEAT_EXTREME = "heat_extreme"
    PRECIPITATION_EXCESS = "precipitation_excess"
    PRECIPITATION_DEFICIT = "precipitation_deficit"
    TROPICAL_CYCLONE = "tropical_cyclone"
    COMPOUND = "compound"           # Multiple drivers simultaneously
    COASTAL = "coastal"
    CRYOSPHERE = "cryosphere"
    ANTHROPOGENIC_TREND = "anthropogenic_trend"


@dataclass(frozen=True)
class PhysicalEvent:
    """A compound physical climate event with hazard intensity profile.

    Attributes
    ----------
    id                  : Unique slug, used as the API key.
    name                : Human-readable event name.
    driver              : Primary climate driver category.
    hazard_multipliers  : dict[hazard_name, multiplier] — scales the asset's
                          baseline severity_index.  Missing hazards unchanged.
    hazard_floors       : dict[hazard_name, min_severity] — minimum severity
                          floor applied regardless of the asset's baseline.
    duration_months     : Expected duration of elevated conditions.
    acute               : True = episodic (hours–weeks); False = seasonal/chronic.
    context             : Scientific background explaining the event mechanism.
    historical_analogs  : Documented real events used for calibration.
    affected_regions    : Indicative regions most exposed (informational).
    """
    id: str
    name: str
    driver: EventDriver
    hazard_multipliers: dict = field(default_factory=dict)
    hazard_floors: dict = field(default_factory=dict)
    duration_months: int = 6
    acute: bool = False
    context: str = ""
    historical_analogs: tuple = ()
    affected_regions: tuple = ()


# ---------------------------------------------------------------------------
# ENSO — El Niño events
# ---------------------------------------------------------------------------

EL_NINO_SUPER_DROUGHT = PhysicalEvent(
    id="el_nino_super_drought",
    name="Super El Niño — Severe Drought",
    driver=EventDriver.ENSO_EL_NINO,
    hazard_multipliers={
        "water_stress":        2.8,
        "drought":             2.6,
        "heat_stress":         2.0,
        "wildfire":            2.4,
        "flood_riverine":      0.4,   # suppressed — drier conditions
        "water_contamination": 1.8,
        "permafrost_thaw":     1.3,
    },
    hazard_floors={
        "water_stress": 3.5,
        "drought":      3.2,
    },
    duration_months=18,
    acute=False,
    context=(
        "A strong El Niño (ONI > +1.5°C) disrupts the Walker Circulation, "
        "driving severe drought across tropical Australia, southern Africa, "
        "South and Southeast Asia, and parts of South America.  Concurrent "
        "heat amplification raises wildfire risk and collapses surface-water "
        "availability.  Industrial water allocations are often curtailed 30–60%."
    ),
    historical_analogs=(
        "2015–16 Super El Niño (global economic loss ~USD 70 bn)",
        "1997–98 El Niño (USD 45 bn loss, 23,000 deaths)",
        "1982–83 El Niño",
    ),
    affected_regions=(
        "AU-WA", "AU-QLD", "AU-SA", "ZA", "IN", "ID", "PH", "CL", "PE", "BR-CE",
    ),
)

EL_NINO_MODERATE_DROUGHT = PhysicalEvent(
    id="el_nino_moderate_drought",
    name="Moderate El Niño — Agricultural Drought",
    driver=EventDriver.ENSO_EL_NINO,
    hazard_multipliers={
        "water_stress": 1.8,
        "drought":      1.7,
        "heat_stress":  1.4,
        "wildfire":     1.5,
        "flood_riverine": 0.7,
    },
    hazard_floors={
        "water_stress": 2.0,
        "drought":      1.8,
    },
    duration_months=12,
    acute=False,
    context=(
        "A moderate El Niño (ONI +0.5 to +1.5°C) reduces seasonal rainfall "
        "across affected regions by 20–40%, stressing rain-fed agriculture "
        "and reservoir storage while elevating wildfire risk."
    ),
    historical_analogs=(
        "2009–10 El Niño",
        "2018–19 El Niño",
        "2023 El Niño emergence",
    ),
    affected_regions=("AU-QLD", "AU-NSW", "ZA", "IN-MH", "ID", "CL"),
)

# ---------------------------------------------------------------------------
# ENSO — La Niña events
# ---------------------------------------------------------------------------

LA_NINA_HEAVY_RAIN_FLOOD = PhysicalEvent(
    id="la_nina_heavy_rain_flood",
    name="La Niña — Heavy Rain and Supply Chain Flooding",
    driver=EventDriver.ENSO_LA_NINA,
    hazard_multipliers={
        "flood_riverine":  2.6,
        "flood_coastal":   1.6,
        "landslide":       2.2,
        "cyclone":         1.8,   # La Niña increases TC activity in W. Pacific/AU
        "water_stress":    0.5,   # suppressed
        "drought":         0.4,   # suppressed
    },
    hazard_floors={
        "flood_riverine": 3.0,
    },
    duration_months=12,
    acute=False,
    context=(
        "La Niña drives above-average rainfall and flooding across eastern "
        "Australia, Southeast Asia, and parts of South America.  The 2010–12 "
        "triple-dip La Niña caused USD 30 bn in Queensland alone, devastating "
        "coal, sugar, and beef supply chains.  Supply chain disruption extends "
        "well beyond directly flooded assets through logistics network collapse."
    ),
    historical_analogs=(
        "2010–12 triple-dip La Niña (Queensland floods: USD 30 bn)",
        "2021–22 double-dip La Niña",
        "2011 Thai floods (electronics supply chain: USD 45 bn)",
    ),
    affected_regions=("AU-QLD", "AU-NSW", "TH", "ID", "PH", "CO", "PE", "MX"),
)

# ---------------------------------------------------------------------------
# Tropical cyclone events
# ---------------------------------------------------------------------------

TROPICAL_CYCLONE_CAT4 = PhysicalEvent(
    id="tropical_cyclone_cat4",
    name="Category 4 Tropical Cyclone — Direct Landfall",
    driver=EventDriver.TROPICAL_CYCLONE,
    hazard_multipliers={
        "cyclone":          4.0,
        "flood_coastal":    3.0,
        "flood_riverine":   2.0,
        "landslide":        2.5,
        "saltwater_intrusion": 2.8,
        "wind_damage":      4.5,   # even if not in the 25-hazard model, captures general damage
    },
    hazard_floors={
        "cyclone":       4.5,
        "flood_coastal": 3.5,
    },
    duration_months=2,   # acute event + recovery period
    acute=True,
    context=(
        "A Category 4 cyclone (sustained winds 130–156 mph) causes catastrophic "
        "structural damage within 50–100 km of the eye track.  Storm surge of "
        "3–6 m inundates coastal assets.  Power outages of 2–8 weeks are typical. "
        "Supply chains through affected ports collapse for 4–12 weeks."
    ),
    historical_analogs=(
        "Cyclone Yasi 2011 (QLD, USD 3.5 bn)",
        "Typhoon Haiyan 2013 (Philippines, USD 13 bn)",
        "Hurricane Harvey 2017 (Texas, USD 130 bn)",
        "Cyclone Idai 2019 (Mozambique/Zimbabwe, USD 2 bn)",
    ),
    affected_regions=("AU-QLD", "PH", "MX", "IN-OR", "MZ", "ZW"),
)

TROPICAL_CYCLONE_CAT5 = PhysicalEvent(
    id="tropical_cyclone_cat5",
    name="Category 5 Tropical Cyclone — Catastrophic Landfall",
    driver=EventDriver.TROPICAL_CYCLONE,
    hazard_multipliers={
        "cyclone":          5.0,
        "flood_coastal":    4.5,
        "flood_riverine":   3.0,
        "landslide":        3.5,
        "saltwater_intrusion": 4.0,
    },
    hazard_floors={
        "cyclone":       5.0,
        "flood_coastal": 4.0,
    },
    duration_months=3,
    acute=True,
    context=(
        "Category 5 cyclones (winds > 157 mph) cause near-total destruction "
        "within 30–60 km of the track.  Economic losses typically exceed USD 10 bn "
        "for direct landfall events.  Assets in the path face potential total loss "
        "of surface structures.  Port and rail logistics can be offline 3–6 months."
    ),
    historical_analogs=(
        "Hurricane Maria 2017 (Puerto Rico, USD 90 bn)",
        "Cyclone Winston 2016 (Fiji, USD 1.4 bn — 8% of GDP)",
        "Hurricane Dorian 2019 (Bahamas)",
        "Typhoon Tip 1979 (most intense TC on record)",
    ),
    affected_regions=("PH", "JP", "AU-QLD", "MX", "US-FL", "US-TX"),
)

# ---------------------------------------------------------------------------
# Heat extreme events
# ---------------------------------------------------------------------------

HEAT_DOME_ACUTE = PhysicalEvent(
    id="heat_dome_acute",
    name="Heat Dome — Acute 3-Week Extreme Heat Event",
    driver=EventDriver.HEAT_EXTREME,
    hazard_multipliers={
        "heat_stress":      3.5,
        "water_stress":     2.0,
        "wildfire":         2.8,
        "drought":          1.5,
        "permafrost_thaw":  2.5,
    },
    hazard_floors={
        "heat_stress": 4.0,
    },
    duration_months=1,
    acute=True,
    context=(
        "A blocking high-pressure system traps hot air for 2–4 weeks, driving "
        "temperatures 10–20°C above seasonal norms.  The 2021 Pacific Northwest "
        "heat dome reached 49.6°C in Lytton, BC.  Industrial labour productivity "
        "falls 25–50% under wet-bulb temperatures >28°C.  Cooling demand spikes "
        "cause grid stress.  Outdoor operations (mining, agriculture) are curtailed."
    ),
    historical_analogs=(
        "2021 Pacific Northwest heat dome (USD 10 bn crop loss in Canada)",
        "2003 European heat wave (70,000 deaths, USD 13 bn crop loss)",
        "2010 Russian heat wave (USD 15 bn, 55,000 deaths)",
        "2019 European heat wave (new temperature records in 7 countries)",
    ),
    affected_regions=("US-OR", "US-WA", "CA-BC", "FR", "DE", "ES", "RU"),
)

COMPOUND_DROUGHT_HEAT = PhysicalEvent(
    id="compound_drought_heat",
    name="Compound Drought + Heat Stress (Summer Season)",
    driver=EventDriver.COMPOUND,
    hazard_multipliers={
        "drought":          2.5,
        "heat_stress":      2.8,
        "water_stress":     2.4,
        "wildfire":         3.2,
        "flood_riverine":   0.5,
    },
    hazard_floors={
        "drought":     2.5,
        "heat_stress": 2.8,
        "wildfire":    2.0,
    },
    duration_months=5,
    acute=False,
    context=(
        "Compound drought-heat events are increasing in frequency 7× faster than "
        "drought or heat alone (Zscheischler et al. 2020, Nature Climate Change). "
        "Low soil moisture from drought reduces evaporative cooling, amplifying "
        "surface temperatures by 3–8°C above what the synoptic system would "
        "produce alone.  Critical for agriculture (compound crop failure), "
        "beverages (water sourcing + input crop failure simultaneously), and "
        "any water-intensive manufacturing."
    ),
    historical_analogs=(
        "2012 US Midwest drought + heat (USD 30 bn crop loss)",
        "2018 Europe compound event (grain harvest −20%)",
        "2019–20 Australia fire-drought (Black Summer)",
        "2022 South Asia heat wave + drought",
    ),
    affected_regions=("US-KS", "US-IA", "AU-NSW", "AU-VIC", "IN", "FR", "ES"),
)

# ---------------------------------------------------------------------------
# Precipitation extremes — excess
# ---------------------------------------------------------------------------

ATMOSPHERIC_RIVER_FLOOD = PhysicalEvent(
    id="atmospheric_river_flood",
    name="Atmospheric River — Extreme Precipitation and Flash Flooding",
    driver=EventDriver.PRECIPITATION_EXCESS,
    hazard_multipliers={
        "flood_riverine": 3.5,
        "landslide":      3.0,
        "flood_coastal":  1.5,
    },
    hazard_floors={
        "flood_riverine": 4.0,
        "landslide":      2.5,
    },
    duration_months=2,
    acute=True,
    context=(
        "Atmospheric rivers (ARs) carry moisture fluxes comparable to 15× the "
        "average flow of the Amazon River.  Landfall events deliver 200–500% of "
        "monthly normal rainfall in 24–72 hours.  The 2022–23 California AR sequence "
        "caused USD 4 bn in damage over 9 consecutive AR events.  Supply chains "
        "through affected road/rail corridors collapse for weeks."
    ),
    historical_analogs=(
        "2022–23 California AR sequence (USD 4 bn)",
        "2021 BC floods (trans-Canada rail/highway: USD 7.5 bn)",
        "2017 Oroville Dam crisis",
        "2009 UK flooding from AR (Cumbria: GDP-1.0% regional)",
    ),
    affected_regions=("US-CA", "CA-BC", "GB-ENG", "NO", "PT"),
)

MONSOON_FAILURE_AND_FLOOD = PhysicalEvent(
    id="monsoon_failure_and_flood",
    name="Monsoon Failure then Rebound Flash Flood",
    driver=EventDriver.COMPOUND,
    hazard_multipliers={
        "drought":        2.0,
        "water_stress":   2.2,
        "flood_riverine": 2.5,   # delayed heavy rain on dry/cracked soil
        "landslide":      2.0,
        "heat_stress":    1.8,
    },
    duration_months=6,
    acute=False,
    context=(
        "A failed monsoon onset (delayed ≥ 3 weeks) causes acute drought stress "
        "on standing crops, followed by intense 'catch-up' rainfall on hardened "
        "soil with low infiltration capacity.  Run-off rates 2–3× normal cause "
        "severe flash flooding and landslides.  The combined event hits agriculture "
        "twice: first drought-induced yield loss, then flood damage to harvested "
        "and stored crops."
    ),
    historical_analogs=(
        "2002 Indian monsoon failure (crop loss USD 9 bn)",
        "2017 Pakistan monsoon + flooding (1,200 deaths)",
        "2021 China Henan floods (USD 17 bn after dry spring)",
    ),
    affected_regions=("IN", "PK", "BD", "CN-HN", "TH", "VN"),
)

RIVER_FLOOD_MAJOR = PhysicalEvent(
    id="river_flood_major",
    name="Major Riverine Flood — Extended Inundation (6–12 weeks)",
    driver=EventDriver.PRECIPITATION_EXCESS,
    hazard_multipliers={
        "flood_riverine":      3.2,
        "water_contamination": 2.0,
        "landslide":           1.5,
    },
    hazard_floors={
        "flood_riverine": 3.5,
    },
    duration_months=4,
    acute=False,
    context=(
        "Prolonged fluvial flooding driven by sustained above-normal rainfall "
        "or snowmelt.  Inundation persists 6–12 weeks, causing progressive "
        "structural damage (foundations, electrical, HVAC) beyond what acute "
        "flash flooding causes.  Supply chain disruption is extended — logistics "
        "networks require drying-out and structural inspection before reopening."
    ),
    historical_analogs=(
        "2011 Thai floods (logistics: USD 45 bn, 7M impacted workers)",
        "2013 European floods (Danube/Elbe: USD 15 bn)",
        "2016 US Mississippi River floods",
        "2020 China Yangtze floods (USD 25 bn)",
    ),
    affected_regions=("TH", "DE", "AT", "CN-HB", "US-MS", "BD"),
)

# ---------------------------------------------------------------------------
# Wildfire events
# ---------------------------------------------------------------------------

WILDFIRE_SEASON_SEVERE = PhysicalEvent(
    id="wildfire_season_severe",
    name="Severe Wildfire Season — Multi-Month Perimeter Expansion",
    driver=EventDriver.HEAT_EXTREME,
    hazard_multipliers={
        "wildfire":         4.0,
        "heat_stress":      2.0,
        "drought":          1.8,
        "water_stress":     1.6,
        "air_quality":      4.5,   # smoke impact even far from fire perimeter
    },
    hazard_floors={
        "wildfire": 3.5,
    },
    duration_months=4,
    acute=False,
    context=(
        "A severe wildfire season driven by compound drought-heat conditions.  "
        "Direct perimeter assets face partial or total loss.  Assets within "
        "100 km face smoke-related operational constraints (air quality limits "
        "on outdoor workers), supply chain disruption from closed highways, "
        "and power outage from damaged transmission infrastructure.  The 2019–20 "
        "Australian Black Summer burned 18.6 Mha, with USD 103 bn total cost."
    ),
    historical_analogs=(
        "2019–20 Australia Black Summer (USD 103 bn total, 3 bn animals)",
        "2018 California Camp Fire (USD 16.5 bn)",
        "2021 Bootleg Fire Oregon + BC record fires",
        "2022 European wildfires (record 700k ha burned)",
    ),
    affected_regions=("AU-NSW", "AU-VIC", "US-CA", "ES", "GR", "PT", "CA-BC"),
)

WILDFIRE_ACUTE_ASSET = PhysicalEvent(
    id="wildfire_acute_asset",
    name="Acute Wildfire — Perimeter Within 10 km of Asset",
    driver=EventDriver.HEAT_EXTREME,
    hazard_multipliers={
        "wildfire":   5.0,
        "heat_stress": 2.5,
    },
    hazard_floors={
        "wildfire": 4.8,
    },
    duration_months=1,
    acute=True,
    context=(
        "A wildfire perimeter approaching within 10 km of an asset triggers "
        "mandatory evacuation and production halt.  Embers can travel >20 km, "
        "creating spot fire risk.  Power transmission lines in the area are "
        "de-energised for safety (PSPS), cutting power to the asset.  Even if "
        "the asset survives, smoke damage to equipment and product requires "
        "extensive remediation."
    ),
    historical_analogs=(
        "2021 Dixie Fire (California industrial facilities evacuated)",
        "2020 Glass Fire (Napa Valley wineries: total loss of several estates)",
        "2019 Black Summer Australian mines evacuated",
    ),
    affected_regions=("AU-NSW", "AU-VIC", "US-CA", "ES", "PT"),
)

# ---------------------------------------------------------------------------
# Coastal events
# ---------------------------------------------------------------------------

COASTAL_STORM_SURGE_COMPOUND = PhysicalEvent(
    id="coastal_storm_surge_compound",
    name="Compound Coastal Flood — Storm Surge + High Tide + Rainfall",
    driver=EventDriver.COASTAL,
    hazard_multipliers={
        "flood_coastal":       4.0,
        "saltwater_intrusion": 3.5,
        "sea_level_rise":      2.0,   # amplified by temporary SLR
        "flood_riverine":      1.8,
    },
    hazard_floors={
        "flood_coastal":       4.0,
        "saltwater_intrusion": 2.5,
    },
    duration_months=1,
    acute=True,
    context=(
        "Compound coastal flooding from the simultaneous occurrence of storm surge, "
        "spring tidal peak, and heavy precipitation.  Each component alone would "
        "be manageable; combined they produce inundation depths 40–100% greater "
        "than the worst individual component.  Frequency doubles under 1°C SLR "
        "(Bevacqua et al. 2019, Science Advances).  Critical for port infrastructure, "
        "coastal manufacturing, aquaculture, and coastal real estate."
    ),
    historical_analogs=(
        "2012 Hurricane Sandy (NYC: USD 65 bn)",
        "2013 North Sea storm surge (UK/NL: EUR 1 bn)",
        "2021 European floods (Belgium/Germany: EUR 40 bn)",
        "2022 Pakistan floods (30% of land submerged)",
    ),
    affected_regions=("US-NY", "US-NJ", "NL", "DE", "GB-ENG", "BD", "VN"),
)

# ---------------------------------------------------------------------------
# Precipitation extremes — deficit / drought
# ---------------------------------------------------------------------------

PROLONGED_MULTI_YEAR_DROUGHT = PhysicalEvent(
    id="prolonged_multi_year_drought",
    name="Multi-Year Drought — Chronic Water Security Crisis (3+ years)",
    driver=EventDriver.PRECIPITATION_DEFICIT,
    hazard_multipliers={
        "water_stress":    3.5,
        "drought":         3.2,
        "heat_stress":     1.8,
        "wildfire":        2.5,
        "water_contamination": 2.0,
    },
    hazard_floors={
        "water_stress": 4.0,
        "drought":      3.5,
    },
    duration_months=36,
    acute=False,
    context=(
        "A multi-year precipitation deficit depletes surface water storage "
        "(reservoirs, wetlands) and groundwater aquifers.  Industrial water "
        "allocations are cut 50–90% under Stage 4/5 restrictions.  Urban "
        "water stress cascades into supply-chain disruption for water-intensive "
        "industries (beverages, food processing, semiconductors, chemicals). "
        "The Cape Town 'Day Zero' crisis (2018) saw industrial water costs rise "
        "40× and several production facilities relocate."
    ),
    historical_analogs=(
        "2015–17 Cape Town drought (South Africa: industrial water 40× cost)",
        "2012–17 California drought (agricultural loss: USD 3 bn/year)",
        "2017–20 Chile megadrought (Atacama copper operations restricted)",
        "2019–20 Murray–Darling drought (Australian agriculture: AUD 5.7 bn)",
    ),
    affected_regions=("ZA-WC", "US-CA", "CL", "AU-NSW", "AU-VIC", "MX"),
)

WATER_CRISIS_MUNICIPAL = PhysicalEvent(
    id="water_crisis_municipal",
    name="Municipal Water Supply Failure — Industrial Allocation Suspended",
    driver=EventDriver.PRECIPITATION_DEFICIT,
    hazard_multipliers={
        "water_stress": 4.5,
        "drought":      2.0,
    },
    hazard_floors={
        "water_stress": 4.5,
    },
    duration_months=6,
    acute=False,
    context=(
        "Acute municipal water supply failure triggers immediate industrial "
        "allocation suspension.  Typically driven by reservoir storage dropping "
        "below 20% combined capacity, triggering emergency water management "
        "orders.  Industrial users (highest consumers) face 60–100% cuts before "
        "residential users are impacted.  Operations must either switch to "
        "emergency water trucking, invest in on-site storage, or halt production."
    ),
    historical_analogs=(
        "2018 Cape Town Day Zero (industries pre-emptively cut 45%)",
        "2021 Taiwan drought (TSMC/semiconductor plants: groundwater emergency)",
        "2022 Chennai India water crisis",
    ),
    affected_regions=("ZA-WC", "TW", "IN-TN", "MX-NL"),
)

# ---------------------------------------------------------------------------
# Cryosphere events
# ---------------------------------------------------------------------------

GLACIAL_OUTBURST_FLOOD = PhysicalEvent(
    id="glacial_outburst_flood",
    name="Glacial Lake Outburst Flood (GLOF)",
    driver=EventDriver.CRYOSPHERE,
    hazard_multipliers={
        "flood_riverine": 4.5,
        "landslide":      4.0,
        "flood_coastal":  1.2,
    },
    hazard_floors={
        "flood_riverine": 4.5,
        "landslide":      3.5,
    },
    duration_months=1,
    acute=True,
    context=(
        "Rapid drainage of a glacial lake (natural dam failure or ice dam breach) "
        "releases enormous water volumes in hours.  GLOFs are increasing as "
        "glacier mass loss accelerates.  Peak discharges can exceed 100× normal "
        "river flow.  Particularly relevant for Andean mining operations, "
        "Himalayan hydropower, and high-altitude agriculture supply chains."
    ),
    historical_analogs=(
        "2013 Uttarakhand GLOF (India: 5,700 dead, USD 3 bn)",
        "2020 Chamoli disaster (India: hydropower plants destroyed)",
        "Multiple Andean GLOFs affecting Chilean copper operations",
        "Bhutan GLOF events (hydropower and downstream agriculture)",
    ),
    affected_regions=("IN-UK", "CL", "PE", "NP", "BT", "PK-GB"),
)

PERMAFROST_THAW_CHRONIC = PhysicalEvent(
    id="permafrost_thaw_chronic",
    name="Accelerated Permafrost Thaw — Infrastructure Destabilisation",
    driver=EventDriver.CRYOSPHERE,
    hazard_multipliers={
        "permafrost_thaw":  4.0,
        "landslide":        2.5,
        "flood_riverine":   1.5,
        "heat_stress":      1.4,
    },
    hazard_floors={
        "permafrost_thaw": 4.0,
    },
    duration_months=60,   # chronic
    acute=False,
    context=(
        "Accelerated permafrost thaw (active layer deepening > 3× historical rate) "
        "destabilises foundations of buildings, pipelines, roads, and mine "
        "infrastructure across Arctic and sub-Arctic regions.  Russian energy "
        "sector estimated USD 1.3 bn/year in permafrost-related infrastructure "
        "damage.  Particularly relevant for oil/gas pipelines, mining haul roads, "
        "and LNG facilities in northern latitudes."
    ),
    historical_analogs=(
        "2020 Norilsk Nickel fuel spill (permafrost tank failure: USD 2 bn fine)",
        "Alaska North Slope pipeline degradation",
        "Siberian sinkholes — accelerating thaw rate",
        "Russian infrastructure losses: RUB 150 bn/year estimated by 2050",
    ),
    affected_regions=("RU-YAN", "US-AK", "CA-YT", "NO-SV", "GL"),
)

# ---------------------------------------------------------------------------
# Late-season frost (agriculture specific)
# ---------------------------------------------------------------------------

LATE_SEASON_FROST = PhysicalEvent(
    id="late_season_frost",
    name="Late-Season Frost — Crop Damage at Flowering Stage",
    driver=EventDriver.HEAT_EXTREME,   # anti-cyclonic intrusion
    hazard_multipliers={
        "frost_crop_damage": 5.0,
        "heat_stress":       0.3,      # suppressed
    },
    hazard_floors={
        "frost_crop_damage": 4.5,
    },
    duration_months=1,
    acute=True,
    context=(
        "An unexpected late-season frost during crop flowering or bud-burst "
        "can cause 40–100% yield loss in affected orchards and vineyards.  "
        "More common as climate variability increases — earlier warm springs "
        "trigger early bud-burst, increasing frost exposure window.  "
        "Fruit, viticulture, and specialty crops are most exposed."
    ),
    historical_analogs=(
        "2021 European viticulture frost (France: EUR 2 bn, 40–80% loss in Bordeaux)",
        "2020 US apple/cherry frost losses (Pacific Northwest: USD 800 M)",
        "2017 European late frost (cereal loss: EUR 1.2 bn)",
    ),
    affected_regions=("FR", "IT", "ES", "DE", "US-WA", "US-MI", "AU-SA"),
)

# ---------------------------------------------------------------------------
# Event library index
# ---------------------------------------------------------------------------

EVENT_LIBRARY: dict[str, PhysicalEvent] = {
    e.id: e
    for e in [
        EL_NINO_SUPER_DROUGHT,
        EL_NINO_MODERATE_DROUGHT,
        LA_NINA_HEAVY_RAIN_FLOOD,
        TROPICAL_CYCLONE_CAT4,
        TROPICAL_CYCLONE_CAT5,
        HEAT_DOME_ACUTE,
        COMPOUND_DROUGHT_HEAT,
        ATMOSPHERIC_RIVER_FLOOD,
        MONSOON_FAILURE_AND_FLOOD,
        RIVER_FLOOD_MAJOR,
        WILDFIRE_SEASON_SEVERE,
        WILDFIRE_ACUTE_ASSET,
        COASTAL_STORM_SURGE_COMPOUND,
        PROLONGED_MULTI_YEAR_DROUGHT,
        WATER_CRISIS_MUNICIPAL,
        GLACIAL_OUTBURST_FLOOD,
        PERMAFROST_THAW_CHRONIC,
        LATE_SEASON_FROST,
    ]
}


def get_event(event_id: str) -> PhysicalEvent:
    """Return a PhysicalEvent by ID, raising KeyError if not found."""
    try:
        return EVENT_LIBRARY[event_id]
    except KeyError:
        available = ", ".join(sorted(EVENT_LIBRARY))
        raise KeyError(
            f"Unknown physical event '{event_id}'. "
            f"Available: {available}"
        ) from None


def list_events() -> list[dict]:
    """Return a summary list of all events (id, name, driver, duration_months)."""
    return [
        {
            "id": e.id,
            "name": e.name,
            "driver": e.driver.value,
            "duration_months": e.duration_months,
            "acute": e.acute,
            "historical_analogs": list(e.historical_analogs),
            "affected_regions": list(e.affected_regions),
        }
        for e in EVENT_LIBRARY.values()
    ]
