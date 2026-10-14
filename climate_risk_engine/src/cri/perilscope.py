"""
Peril scope: which perils matter for an asset class at a place, and what this engine can and cannot say about each.

Why it exists. A price or a risk screen is only honest if it states what it leaves out. Public insurer and reinsurer sources are clear that the perils that drive
loss depend on the sector and the region (cables for offshore wind, hail for solar, freeze for Gulf Coast refineries, flash flood and heat for the Gulf, river flood
and supply chains for Southeast Asian factories), and that "secondary" perils are now most of insured nat-cat loss (Swiss Re sigma 1/2026: 92 % in 2025). The engine
models a handful of them. This module makes that gap explicit for every site, so a premium or screen is always shown next to the relevant perils it does not contain.

Two tables, both reviewable:
  ENGINE   per canonical peril: what the engine has (loss model, hazard only, screen, or nothing), where, and how far it was validated. Checked against the hazard
           registry by tests, so it cannot drift from the code.
  SECTOR   per asset class: the perils that matter, whether each is primary or secondary, and the evidence (a public source or a labelled domain judgement).
  REGION   per region: the perils its climate or tectonics make relevant, from coordinates.

Nothing here computes a loss. It says what is in the number and what is not. `not_assessed` never means zero.
"""
from __future__ import annotations

import re
from typing import Optional

# ── what the engine has ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
# status: loss_model  hazard plus an engine loss path (insurance.py) at least where maps or a record exist
#         hazard_only a hazard value or return level exists, but nothing converts it to loss for an asset
#         screen      a proxy or susceptibility screen, not a hazard model
#         none        nothing
ENGINE = {
    "tropical_cyclone": {"status": "loss_model", "keys": ["cyclone"], "note": "Holland/Emanuel wind on IBTrACS, damage curve; surge separate", "validated": "held-out seasons (hindcast)"},
    "extratropical_windstorm": {"status": "hazard_only", "keys": ["wind"], "note": "ERA5 gust return levels only; 25 to 50 km gusts read low; no onshore damage curve; offshore wind has its own storm path", "validated": "ERA5 hindcast only"},
    "severe_convective_hail": {"status": "none", "keys": [], "note": "no hail hazard (needs radar or reports; reanalysis cannot resolve hail)", "validated": None},
    "tornado": {"status": "none", "keys": [], "note": "no tornado hazard", "validated": None},
    "lightning": {"status": "none", "keys": [], "note": "no lightning flash-density hazard", "validated": None},
    "river_flood": {"status": "loss_model", "keys": ["river_flood", "river", "flood_terrain_radar"], "note": "flood maps plus depth-damage; null where no map exists", "validated": "NFIP claims and flood events (US)"},
    "pluvial_flash_flood": {"status": "hazard_only", "keys": ["rain", "flood", "flood_terrain_radar"], "note": "rainfall return levels and terrain screen; no flash-flood loss path", "validated": "ERA5 hindcast only"},
    "coastal_flood_surge": {"status": "loss_model", "keys": ["coastal_flood", "surge"], "note": "surge and extreme sea level with depth-damage where coastal maps exist", "validated": "US tide gauges only"},
    "tsunami": {"status": "none", "keys": [], "note": "no tsunami hazard", "validated": None},
    "sea_level_rise": {"status": "hazard_only", "keys": ["sealevel"], "note": "IPCC AR6 scenarios and Bruun retreat", "validated": "US tide gauges only"},
    "earthquake": {"status": "hazard_only", "keys": ["earthquake"], "note": "shaking (Gutenberg-Richter, Joyner-Boore) without building or plant vulnerability", "validated": "USGS catalogue hindcast"},
    "landslide_debris_flow": {"status": "screen", "keys": ["landslide", "landslide_slope", "debris_flow"], "note": "slope susceptibility screen", "validated": None},
    "wildfire": {"status": "hazard_only", "keys": ["fire", "wildfire_fuel"], "note": "fire weather index plus fuel screen; no spread or damage model", "validated": "Canada only: AUC 0.75, no better than temperature alone"},
    "wildfire_smoke_air": {"status": "hazard_only", "keys": ["air", "air_local"], "note": "PM2.5 box model and local screen", "validated": "ground monitors: median daily correlation 0.58, polluted sites read low"},
    "heat_workers": {"status": "hazard_only", "keys": ["heat", "thermal_stress", "urban_heat_landcover"], "note": "WBGT and UTCI hours; no labour-productivity or loss conversion", "validated": "airport stations, bias about -0.1 to -1.7 C by region"},
    "heat_asset_derating": {"status": "none", "keys": [], "note": "no equipment or cooling derating model", "validated": None},
    "extreme_cold_freeze": {"status": "hazard_only", "keys": ["cold_extreme", "thermal_stress"], "note": "return levels of the coldest day and exceedance of a design minimum; no freeze-failure (pipes, instruments, power loss) or loss model", "validated": "one event: Uri recovered at about 1-in-30 years at Houston and Dallas"},
    "ice_storm_freezing_rain": {"status": "none", "keys": [], "note": "no freezing-rain or ice-accretion hazard on land", "validated": None},
    "snow_load": {"status": "none", "keys": [], "note": "no snow-load hazard", "validated": None},
    "permafrost_thaw": {"status": "none", "keys": [], "note": "no permafrost or ground-ice hazard", "validated": None},
    "sea_ice_icing": {"status": "screen", "keys": [], "note": "offshore cold-season screen only (freezing degree-days, icing-prone days)", "validated": None},
    "drought": {"status": "hazard_only", "keys": ["drought"], "note": "soil water balance, no crop, hydropower or cooling-water loss path", "validated": "US Drought Monitor: ROC area 0.70, detection about 0.2"},
    "water_stress": {"status": "hazard_only", "keys": ["water_stress"], "note": "baseline water stress; no operational cost path", "validated": None},
    "dust_sand_storm": {"status": "none", "keys": [], "note": "no dust or soiling hazard", "validated": None},
    "volcanic": {"status": "none", "keys": [], "note": "no volcanic ash or lava hazard", "validated": None},
    "subsidence": {"status": "none", "keys": [], "note": "no subsidence or clay-shrink hazard", "validated": None},
    "coastal_erosion": {"status": "hazard_only", "keys": ["sealevel"], "note": "Bruun shoreline retreat only", "validated": None},
    "glacial_lake_outburst": {"status": "none", "keys": [], "note": "no glacier or lake-outburst hazard", "validated": None},
    "offshore_storm_waves": {"status": "loss_model", "keys": [], "note": "ERA5 wind and wave return levels with a placeholder damage curve (a bracket, not calibrated)", "validated": "not yet checked against buoys"},
    "supply_chain_cbi": {"status": "none", "keys": [], "note": "no supplier network or contingent business interruption model; Thailand 2011 and Tohoku 2011 losses were dominated by it", "validated": None},
    "grid_outage_dependency": {"status": "none", "keys": [], "note": "no power-grid dependency or outage model", "validated": None},
    "cable_and_component_failure": {"status": "loss_model", "keys": [], "note": "offshore wind only, and only from the insurer's own claims (credibility rating); not a hazard", "validated": "tests on invented claims only"},
}

# ── sector tables ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
# (peril, role, evidence)  role: primary or secondary; evidence starts with 'src:' (a public source named in docs) or 'judgement:' (to be reviewed by an engineer)
S = "src:"
J = "judgement:"
SECTOR = {
    "offshore_wind": [("offshore_storm_waves", "primary", S + "DNV, Allianz Commercial: waves and wind set loads and repair waiting time"), ("cable_and_component_failure", "primary", S + "Allianz 2014-2020: cable damage 53 % of claims by value, turbine failure 20 %; GCube: cabling 55 %, foundations 35 %"),
                      ("lightning", "primary", S + "DNV statement relayed in press: most common claim for wind farm owners; loss adjuster: 60 % of operational blade losses"), ("sea_ice_icing", "secondary", S + "salt depresses freezing point of spray, sea ice at foundations in cold winters; no public offshore claims found"),
                      ("tropical_cyclone", "secondary", S + "GCube: nat-cat exposure rising in Taiwan and the US"), ("earthquake", "secondary", J + "low in the North Sea, high off Taiwan and Japan")],
    "onshore_wind": [("extratropical_windstorm", "primary", J + "design-load exceedance in storms"), ("lightning", "primary", S + "same blade-loss evidence as offshore"), ("sea_ice_icing", "secondary", S + "blade icing derates and stops turbines; 20 to 50 % annual power loss at harsh sites"),
                     ("tropical_cyclone", "secondary", J + "typhoon-exposed Asian and US sites"), ("extreme_cold_freeze", "secondary", J + "low-temperature operating limits")],
    "solar": [("severe_convective_hail", "primary", S + "Midway, Texas 2019: 58 % of 685,000 modules damaged, claim about 70 to 75 million USD; DNV: Texas hail losses on PV above 600 million USD since 2018"), ("extratropical_windstorm", "secondary", J + "racking and tracker damage"),
              ("tornado", "secondary", J + "US plains"), ("river_flood", "secondary", J + "inverters and cabling at grade"), ("wildfire", "secondary", J + "western US, Mediterranean, Australia"), ("heat_asset_derating", "secondary", J + "output and inverter derating"),
              ("dust_sand_storm", "secondary", J + "soiling and abrasion in the Gulf and Sahel"), ("snow_load", "secondary", J + "northern sites"), ("lightning", "secondary", J + "")],
    "hydropower": [("drought", "primary", J + "generation falls with inflow"), ("glacial_lake_outburst", "secondary", J + "Himalaya and Andes"), ("river_flood", "primary", J + "spillway capacity, sediment"), ("landslide_debris_flow", "secondary", J + "reservoir slopes"), ("earthquake", "secondary", J + "dam safety"),
                  ("extreme_cold_freeze", "secondary", J + "river ice")],
    "oil_gas_refining": [("tropical_cyclone", "primary", S + "AIChE/FM Global: coastal chemical plants exposed to hurricanes and floods"), ("river_flood", "primary", S + "same source"), ("coastal_flood_surge", "primary", S + "same source"),
                         ("extreme_cold_freeze", "primary", S + "Winter Storm Uri 2021: Dow downstream insured loss 600 to 650 million USD; Marsh: site damage below 189 million USD per site but large cumulative BI"),
                         ("supply_chain_cbi", "primary", S + "Uri and Thailand 2011: business interruption dominated"), ("earthquake", "secondary", J + ""), ("heat_asset_derating", "secondary", J + "cooling water and air-cooler limits"),
                         ("water_stress", "secondary", J + "process and cooling water"), ("lightning", "secondary", J + "tank fires"), ("wildfire", "secondary", S + "Fort McMurray 2016, 3.58 billion CAD insured"), ("permafrost_thaw", "secondary", J + "northern Canada")],
    "mining": [("pluvial_flash_flood", "primary", J + "pit flooding and tailings dam overtopping"), ("drought", "primary", J + "process water"), ("water_stress", "primary", J + ""), ("landslide_debris_flow", "primary", J + "pit walls and haul roads"), ("earthquake", "secondary", J + "tailings dams"),
               ("permafrost_thaw", "secondary", S + "thermosyphons used to keep arsenic frozen at Canadian mines"), ("heat_workers", "secondary", J + ""), ("wildfire", "secondary", J + ""), ("extreme_cold_freeze", "secondary", J + ""), ("dust_sand_storm", "secondary", J + "")],
    "heavy_industry": [("heat_workers", "primary", S + "Aon: working hours lost to heat in Asia more than doubled in three decades, worst in manufacturing"), ("heat_asset_derating", "primary", J + "cooling"), ("water_stress", "primary", J + ""), ("river_flood", "primary", J + ""),
                       ("coastal_flood_surge", "secondary", J + ""), ("tropical_cyclone", "secondary", J + ""), ("earthquake", "secondary", J + ""), ("drought", "secondary", J + "hydropower-dependent smelters"), ("supply_chain_cbi", "secondary", J + "")],
    "electronics_manufacturing": [("river_flood", "primary", S + "Thailand 2011: 40 to 45 % of world hard-disk output flooded; insured loss 10 to 15 billion USD, mostly manufacturing and contingent BI"), ("supply_chain_cbi", "primary", S + "same; RMS: contingent BI the largest uncertainty after Tohoku"),
                                  ("earthquake", "primary", S + "Tohoku 2011"), ("tsunami", "secondary", S + "Tohoku 2011"), ("tropical_cyclone", "secondary", J + ""), ("heat_workers", "secondary", J + ""), ("water_stress", "secondary", J + "fabs")],
    "textile_garment": [("river_flood", "primary", J + "Bangladesh, Cambodia, Vietnam"), ("pluvial_flash_flood", "primary", J + ""), ("tropical_cyclone", "primary", J + ""), ("heat_workers", "primary", S + "Aon: heat-hours lost, manufacturing"), ("supply_chain_cbi", "secondary", J + "")],
    "data_centre": [("heat_asset_derating", "primary", J + "cooling capacity"), ("water_stress", "primary", S + "FM Resilience Index 2026: low water stress essential for cooling and fire suppression"), ("river_flood", "secondary", J + ""),
                    ("wildfire", "secondary", S + "FM Resilience Index: water stress and fire risk linked"), ("grid_outage_dependency", "primary", J + ""), ("extreme_cold_freeze", "secondary", J + "")],
    "port_logistics": [("coastal_flood_surge", "primary", S + "Swiss Re Asia 2025: ports concentrate value in typhoon and surge zones"), ("tropical_cyclone", "primary", S + "same"), ("sea_level_rise", "secondary", J + ""), ("tsunami", "secondary", J + ""),
                       ("earthquake", "secondary", J + ""), ("extratropical_windstorm", "secondary", J + ""), ("heat_workers", "secondary", J + "")],
    "power_grid": [("extratropical_windstorm", "primary", J + "line and pole loss"), ("ice_storm_freezing_rain", "primary", S + "Eastern Canada ice storm 1998: about 1.5 to 2.2 billion CAD insured, nearly a million without power"), ("wildfire", "primary", S + "LA fires 2025: about 40 billion USD insured; utility ignition"),
                   ("heat_asset_derating", "secondary", J + ""), ("extreme_cold_freeze", "primary", S + "Uri: about two-thirds of ERCOT generation offline"), ("tropical_cyclone", "secondary", J + "")],
    "agriculture_food": [("drought", "primary", J + ""), ("severe_convective_hail", "primary", S + "northern Italy July 2023: 6 billion EUR hail damage across all property"), ("heat_workers", "secondary", J + ""), ("river_flood", "secondary", J + ""), ("wildfire", "secondary", J + "")],
    "real_estate": [("severe_convective_hail", "primary", S + "Calgary 2020: almost 1.2 billion CAD insured, costliest Canadian hailstorm"), ("wildfire", "primary", S + "Fort McMurray 2016 and LA 2025"), ("river_flood", "primary", J + ""), ("extratropical_windstorm", "primary", S + "Europe: Lothar 5.8 billion EUR, Kyrill 3.7 billion EUR (PERILS)"),
                    ("tropical_cyclone", "secondary", J + ""), ("earthquake", "secondary", J + ""), ("subsidence", "secondary", J + "")],
    "pipeline": [("landslide_debris_flow", "primary", J + ""), ("permafrost_thaw", "primary", S + "9,500 km of Arctic pipelines at risk (satellite estimate)"), ("river_flood", "primary", J + "scour and exposure"), ("earthquake", "secondary", J + ""), ("extreme_cold_freeze", "secondary", J + "")],
    "manufacturing_general": [("river_flood", "primary", J + "plants sit on floodplains and coasts"), ("pluvial_flash_flood", "primary", J + ""), ("tropical_cyclone", "primary", J + "where the basin has cyclones"), ("earthquake", "primary", J + ""),
                              ("extratropical_windstorm", "secondary", J + ""), ("heat_workers", "secondary", S + "Aon: heat-hours lost in manufacturing"), ("supply_chain_cbi", "primary", S + "Thailand 2011 and Tohoku 2011: contingent BI dominated manufacturing losses"),
                              ("severe_convective_hail", "secondary", J + ""), ("water_stress", "secondary", J + "")],
    "generic": [("tropical_cyclone", "secondary", J + "no sector given"), ("river_flood", "secondary", J + ""), ("coastal_flood_surge", "secondary", J + ""), ("earthquake", "secondary", J + ""), ("extratropical_windstorm", "secondary", J + "")],
}

_SECTOR_RX = [("offshore_wind", r"offshore\s+wind|wind\s*farm.*offshore|north sea.*wind"), ("onshore_wind", r"wind\b|windmill|turbine"), ("solar", r"solar|photovoltaic|\bpv\b"), ("hydropower", r"hydro|dam\b|reservoir"),
              ("oil_gas_refining", r"refiner|petrochem|chemical|oil|gas|lng|ammonia|fertili[sz]er|cracker"), ("heavy_industry", r"cement|steel|alumin|smelter|glass|paper|pulp|foundry|heavy industr"),
              ("mining", r"\bmin(?:e|ing)\b|quarry|tailings"), ("electronics_manufacturing", r"electronic|semiconductor|chip|fab\b|hard.?disk|automotive|auto parts"),
              ("textile_garment", r"textile|garment|apparel|clothing"), ("data_centre", r"data\s*cent|datacent|server farm"), ("port_logistics", r"\bport\b|terminal|harbou?r|logistic|warehouse|shipping"),
              ("power_grid", r"grid|transmission|substation|utility|power\s*plant|power station"), ("agriculture_food", r"farm|agri|crop|food|plantation|dairy"), ("real_estate", r"real estate|building|office|retail|residential|property|hotel|mall"),
              ("pipeline", r"pipeline"), ("manufacturing_general", r"manufactur|factory|plant|assembly|works\b|industrial")]


def sector_key(text: Optional[str]) -> str:
    t = (text or "").lower()
    for k, rx in _SECTOR_RX:
        if re.search(rx, t):
            return k
    return "generic"


# ── regions from coordinates ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
_ISO = {
    "us": "USA", "canada": "CAN",
    "south_southeast_asia": "IND PAK BGD LKA NPL BTN MMR THA VNM KHM LAO MYS SGP IDN PHL BRN TLS MDV AFG",
    "gulf_middle_east": "ARE SAU QAT BHR KWT OMN IRQ IRN YEM JOR ISR SYR LBN TUR EGY",
    "east_asia": "JPN KOR CHN TWN PRK MNG HKG MAC",
    "south_america": "ARG BOL BRA CHL COL ECU GUY PRY PER SUR URY VEN",
    "australasia": "AUS NZL PNG FJI",
}


def _iso3(lat: float, lon: float) -> Optional[str]:
    try:
        from .finance import geo
        return geo.admin_at(lat, lon).get("iso3")
    except Exception:                                             # noqa: BLE001
        return None


def regions(lat: float, lon: float, iso3: Optional[str] = None) -> list:
    """Regions the point belongs to. Country first (offline boundary file), so Ottawa is Canada and not the United States; boxes only for open water and as a fallback."""
    iso3 = iso3 or _iso3(lat, lon)
    r = []
    inside = lambda a, b, c, d: a <= lat <= b and c <= lon <= d   # noqa: E731
    if iso3:
        for k, v in _ISO.items():
            if iso3 in v.split():
                r.append(k)
        try:
            from .finance import geo
            c = geo.continent(iso3)
        except Exception:                                         # noqa: BLE001
            c = None
        if c == "europe" and iso3 not in ("TUR",):
            r.append("europe")
        if c == "africa" and "gulf_middle_east" not in r:
            r.append("africa")
        return r or ["other"]
    # open water or no country: boxes
    if inside(34, 72, -25, 45):
        r.append("europe")
    if inside(51, 62, -4, 9):
        r.append("north_sea")
    if inside(24, 50, -100, -60) and lon > -98:
        r.append("us")
    if inside(5, 38, 60, 150) or inside(-12, 5, 95, 150):
        r.append("south_southeast_asia")
    if inside(20, 45, 120, 150):
        r.append("east_asia")
    if inside(12, 32, 48, 60):
        r.append("gulf_middle_east")
    return r or ["other"]


REGION = {
    "europe": ["extratropical_windstorm", "severe_convective_hail", "river_flood", "pluvial_flash_flood", "heat_workers"],
    "north_sea": ["extratropical_windstorm", "coastal_flood_surge", "sea_ice_icing"],
    "us": ["severe_convective_hail", "tornado", "river_flood", "extreme_cold_freeze", "wildfire", "earthquake"],
    "canada": ["wildfire", "ice_storm_freezing_rain", "snow_load", "severe_convective_hail", "river_flood", "extreme_cold_freeze"],
    "south_southeast_asia": ["tropical_cyclone", "river_flood", "pluvial_flash_flood", "heat_workers", "earthquake", "landslide_debris_flow", "tsunami", "sea_level_rise", "supply_chain_cbi"],
    "gulf_middle_east": ["heat_workers", "heat_asset_derating", "dust_sand_storm", "pluvial_flash_flood", "water_stress", "sea_level_rise"],
    "africa": ["drought", "river_flood", "heat_workers", "dust_sand_storm"],
    "south_america": ["river_flood", "drought", "landslide_debris_flow", "earthquake", "wildfire", "glacial_lake_outburst"],
    "australasia": ["wildfire", "tropical_cyclone", "river_flood", "severe_convective_hail", "earthquake"],
    "east_asia": ["tropical_cyclone", "earthquake", "tsunami", "river_flood"],
}


def _hazard_status(peril: str, profile: Optional[dict]) -> Optional[dict]:
    """If a site profile has been run, report what it returned for the peril's hazard keys (modelled, constructed, none_found or null) so 'engine has it' is not confused with 'engine produced it here'."""
    if not profile:
        return None
    hz = profile.get("hazards") or {}
    got = {k: hz[k]["status"] for k in ENGINE.get(peril, {}).get("keys", []) if k in hz}
    return got or None


def scope(sector: Optional[str], lat: float, lon: float, covered_in_price: Optional[list] = None, profile: Optional[dict] = None, latitude_permafrost: float = 56.0) -> dict:
    """Perils that matter at this place for this asset class, each with what the engine has. `covered_in_price`: canonical perils the quoted number actually contains."""
    sk = sector if sector in SECTOR else sector_key(sector)
    inc = set(covered_in_price or [])
    reg = regions(lat, lon)
    if sk == "offshore_wind" and 51 <= lat <= 62 and -4 <= lon <= 9 and "north_sea" not in reg:
        reg = reg + ["north_sea"]
    items: dict = {}
    for p, role, ev in SECTOR[sk]:
        items[p] = {"peril": p, "role": role, "why": ev, "from": ["sector"]}
    for rg in reg:
        for p in REGION.get(rg, []):
            if p in items:
                items[p]["from"].append("region:" + rg)
            else:
                items[p] = {"peril": p, "role": "secondary", "why": J + f"regional climate or tectonics ({rg})", "from": ["region:" + rg]}
    if "canada" in reg and lat >= latitude_permafrost and "permafrost_thaw" not in items:
        items["permafrost_thaw"] = {"peril": "permafrost_thaw", "role": "secondary", "why": J + "northern Canada lies on discontinuous or continuous permafrost; confirm with a ground-ice map", "from": ["region:canada"]}
    if "tropical_cyclone" in items and not _tc_possible(lat, lon):
        items.pop("tropical_cyclone")                                   # a cyclone is not expected at 52 N; do not list it as a gap
    out = []
    for p, it in items.items():
        e = ENGINE.get(p, {"status": "none", "note": "peril not in the engine's taxonomy", "validated": None, "keys": []})
        out.append({**it, "engine_status": e["status"], "engine_note": e["note"], "validated": e["validated"], "in_price": p in inc, "profile_status": _hazard_status(p, profile)})
    order = {"primary": 0, "secondary": 1}
    out.sort(key=lambda x: (order[x["role"]], {"none": 0, "screen": 1, "hazard_only": 2, "loss_model": 3}[x["engine_status"]], x["peril"]))
    prim = [x for x in out if x["role"] == "primary"]
    summ = {"sector": sk, "regions": reg, "perils_listed": len(out), "primary": len(prim),
            "primary_with_loss_model": sum(1 for x in prim if x["engine_status"] == "loss_model"),
            "primary_hazard_or_screen_only": sum(1 for x in prim if x["engine_status"] in ("hazard_only", "screen")),
            "primary_not_modelled": [x["peril"] for x in prim if x["engine_status"] == "none"],
            "primary_in_price": sum(1 for x in prim if x["in_price"]),
            "primary_not_in_price": [x["peril"] for x in prim if not x["in_price"]]}
    summ["statement"] = (f"{summ['primary_in_price']} of {summ['primary']} primary perils for this asset class and place are inside the number; "
                         f"{len(summ['primary_not_in_price'])} are not ({', '.join(summ['primary_not_in_price']) or 'none'}), and {len(summ['primary_not_modelled'])} of the primary perils have no engine module at all.")
    return {"summary": summ, "perils": out, "rule": "A peril that is listed but not in the price is outside the number, not zero. Sector and region lists are public-source or labelled judgement and should be reviewed by an engineer."}


def _tc_possible(lat: float, lon: float) -> bool:
    """Tropical cyclones form and track in the tropics and subtropics, not poleward of about 40 degrees and not in the South Atlantic or south-east Pacific."""
    return abs(lat) <= 40 and not (lat < -5 and -75 < lon < 20)
