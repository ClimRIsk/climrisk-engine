"""
cri/physical_report.py — the Physical Risk Assessment report (no financial figures).

For each site the report says, in this order:
  1. how the infrastructure sits    elevation, terrain, slope, land cover, distance to water, catchment
  2. what zone it is in             river / coastal / surface-water flood, cyclone, slope, wildfire, heat, water stress
  3. what its age means             construction year → age now and in 2050, design life, ageing mechanisms
  4. what will go wrong             sector-specific consequences, tied to the measured evidence
  5. what to do about it            physical measures, prioritised by rating

It is built from the engine's site evidence (AWS terrain, Sentinel-2, Sentinel-1, ESA WorldCover, IBTrACS,
JRC river-flood and Deltares coastal-flood maps, FLOPROS defences, WRI Aqueduct). It contains no monetary
loss, value, EBITDA or scenario cash-flow numbers by design.

Ratings are hazard-intensity ratings with stated thresholds, not damage percentages.
"""
from __future__ import annotations

import base64
import html as _html
import io
import math
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from typing import Optional

LEVELS = ["NEGLIGIBLE", "LOW", "MODERATE", "HIGH", "CRITICAL"]
REF_YEAR = 2050


def _lv(x: str) -> int:
    return LEVELS.index(x) if x in LEVELS else 0


def _max_level(levels: list) -> str:
    return LEVELS[max([_lv(x) for x in levels if x] or [0])]


# ── sector profiles: what each hazard does to this kind of infrastructure ─────

PROFILES = {
    "cement": {
        "label": "Cement plant",
        "critical": "kiln and preheater tower, raw and cement mills, clinker and cement silos, conveyors, electrical "
                    "substations and motor control rooms, quarry and haul roads, dispatch by road, rail or jetty",
        "flood": "Water reaching ground-level motor control rooms, mill drives and substations trips the plant. Wet clinker "
                 "and cement in open stockpiles and at silo bases sets or is lost. The kiln has to be held hot or cooled "
                 "under control, and an uncontrolled stop can crack the refractory lining. Flooded access roads stop "
                 "raw-material and dispatch trucks.",
        "wind": "The preheater tower, stacks and silos carry the highest wind loads. Roof sheeting, cladding and conveyor "
                "galleries fail first and loose items become missiles. Loss of grid power during a storm trips the kiln.",
        "slope": "Quarry benches, haul roads and spoil heaps are the slope-sensitive parts: heavy rain destabilises faces "
                 "and washes out roads, and debris reaching the plant platform blocks drainage and access.",
        "fire": "Coal, petcoke and alternative-fuel stockpiles, dust deposits and conveyors are the ignition and spread paths; "
                "embers landing on fuel stores are the main concern.",
        "heat": "High ambient temperature derates mills, fans and electrical equipment, reduces clinker-cooler capacity and "
                "raises heat stress for workers around the kiln.",
        "water": "The plant needs water for cooling, dust suppression and concrete production; high stress raises the "
                 "chance of abstraction limits in dry years.",
    },
    "steel": {
        "label": "Steel plant",
        "critical": "blast furnace or electric arc furnace and converter shops, hot-metal and ladle routes, casters and "
                    "rolling mills, substations, cooling-water circuits, raw-material yards and port or rail links",
        "flood": "Water meeting molten metal is an explosion hazard, so flooding of casting pits, hot-metal routes or cooling "
                 "circuits forces emergency shutdowns. Flooded substations and rolling-mill drives stop production, and "
                 "stockyards and access roads are cut.",
        "wind": "Large-span mill buildings, overhead cranes and conveyor galleries carry high wind loads; cladding failure "
                "and crane damage are typical, and stockyard and port equipment is exposed.",
        "slope": "Raw-material yards, slag heaps and cut slopes around the works can fail or wash out in heavy rain, "
                 "blocking drainage and internal roads.",
        "fire": "Coal, coke and by-product stores, oil-filled equipment and dust deposits are ignition and spread paths.",
        "heat": "Radiant heat adds to high ambient temperature: worker heat stress, warmer cooling water and derated "
                "electrical equipment.",
        "water": "Steelmaking uses large volumes of cooling water; high stress raises the chance of abstraction limits.",
    },
    "power": {
        "label": "Power station",
        "critical": "boilers and turbines, switchyard and transformers, cooling-water intake and cooling towers, fuel "
                    "handling (coal, ash, gas or oil supply), control rooms and grid connection",
        "flood": "A switchyard, transformer bay or control room under water trips the unit and can take weeks to restore. "
                 "Flooded coal yards, ash ponds and intake works disrupt fuel supply and cooling, and ash-pond overtopping "
                 "is an environmental incident.",
        "wind": "Transmission lines, cooling towers, stacks and switchyard gantries are wind-sensitive; losing the grid "
                "connection during a storm trips the plant, and damaged lines delay restart.",
        "slope": "Fuel and ash storage on slopes and the cut slopes around the site can fail in heavy rain and block "
                 "access or drainage.",
        "fire": "Fuel stores, transformer oil and cable galleries are the ignition and spread paths; the grid corridor "
                "is exposed to fire-driven line outages.",
        "heat": "Hot weather and warm intake water reduce thermal efficiency and can force output limits when cooling-water "
                "discharge limits are reached.",
        "water": "Cooling-water availability and temperature set achievable output; high stress raises the chance of "
                 "restrictions in dry years.",
    },
    "oil_gas": {
        "label": "Oil, gas and refining site",
        "critical": "process units and flares, tank farms, jetties and pipelines, utilities (power, steam, cooling water), "
                    "control rooms and loading terminals",
        "flood": "Floating tanks, damaged pipework and flooded utilities cause releases and unit trips; flooded control and "
                 "power systems force a controlled shutdown, and terminals and roads are cut.",
        "wind": "Floating-roof tanks, flare stacks, columns and pipe racks are wind-sensitive. Storm surge combined with wind "
                "threatens jetties and marine loading, and power loss forces an emergency shutdown.",
        "slope": "Pipeline corridors and sites on slopes are exposed to washouts and ground movement in heavy rain.",
        "fire": "Hydrocarbon inventories make any wildfire approach a shutdown trigger; vegetation clearance around "
                "tank farms and pipelines is the main control.",
        "heat": "High ambient temperature reduces air-cooler and compressor performance and raises heat stress for operators.",
        "water": "Cooling and process water supply is critical; high stress raises the chance of abstraction limits.",
    },
    "chemicals": {
        "label": "Chemical plant",
        "critical": "reactors and process units, storage tanks and bunds, utilities (power, steam, cooling water), "
                    "effluent treatment, control rooms and loading areas",
        "flood": "Flooding can float tanks, overtop bunds and mix incompatible materials, and takes out power and control "
                 "systems; effluent systems overflow, turning a flood into a release event.",
        "wind": "Tall columns, storage tanks, pipe racks and cooling towers are wind-sensitive, and loss of power or "
                "cooling during a storm forces an emergency shutdown.",
        "slope": "Sites on slopes are exposed to washouts and drainage failure in heavy rain.",
        "fire": "Flammable inventories make wildfire approach a shutdown trigger; vegetation clearance and fire-water "
                "storage are the main controls.",
        "heat": "High ambient temperature reduces cooling performance and can push exothermic processes towards limits.",
        "water": "Process and cooling water supply is critical; high stress raises the chance of abstraction limits.",
    },
    "mining": {
        "label": "Mine",
        "critical": "pits or underground workings, haul roads, tailings and water-management structures, processing "
                    "plant, power supply and stockpiles",
        "flood": "Flooded pits and underground workings stop mining and need pumping before restart; haul roads wash out, "
                 "and overtopped tailings or water-storage dams are the most serious consequence.",
        "wind": "Processing buildings, conveyors and power lines are wind-sensitive; dust and loose material spread in "
                "strong winds.",
        "slope": "Pit walls, waste dumps and haul-road cuttings are the slope-sensitive parts: heavy rain is the main "
                 "trigger for failures.",
        "fire": "Vegetation around the site, coal stockpiles and conveyors are ignition and spread paths.",
        "heat": "High ambient temperature raises heat stress for operators and derates haul trucks and electrical equipment.",
        "water": "Mineral processing depends on water; high stress raises the chance of abstraction limits.",
    },
    "aluminium": {
        "label": "Aluminium smelter or refinery",
        "critical": "pot lines, rectifiers and substations, anode plant, casthouse, alumina handling, port or rail links",
        "flood": "Flooded rectifiers or substations stop the potlines, and water in contact with molten metal is an "
                 "explosion hazard in the casthouse.",
        "wind": "Large potroom and casthouse buildings, stacks and alumina storage are wind-loaded; loss of grid power "
                "during a storm is the main operational threat.",
        "slope": "Cut slopes and stockpile areas are exposed to washouts in heavy rain.",
        "fire": "Anode and carbon materials, coal tar pitch stores and conveyors are ignition and spread paths.",
        "heat": "Hot weather limits potroom ventilation and raises heat stress; electrical equipment is derated.",
        "water": "Refineries and smelters use cooling and process water; high stress raises the chance of abstraction limits.",
    },
    "pulp": {
        "label": "Pulp and paper mill",
        "critical": "digesters and recovery boilers, paper machines, wood yard and chip piles, substations, effluent "
                    "treatment, water intake",
        "flood": "Flooded wood yards, substations and effluent treatment stop production; wet fibre and finished product "
                 "are lost and effluent systems overflow.",
        "wind": "Large machine halls, stacks and wood-yard cranes are wind-loaded; fallen timber and loose debris damage "
                "equipment.",
        "slope": "Wood yards and cut slopes around the mill are exposed to washouts in heavy rain.",
        "fire": "Wood, chips and fibre dust are highly combustible, and the mill usually sits within or beside forest.",
        "heat": "High ambient temperature reduces cooling performance and raises heat stress.",
        "water": "Pulp and paper is one of the most water-intensive processes; high stress raises the chance of "
                 "abstraction limits.",
    },
    "real_estate": {
        "label": "Building",
        "critical": "basements and ground-floor units, plant rooms, lifts and electrical intake, roof coverings, cladding "
                    "and glazing, access roads and car parks",
        "flood": "Water in basements, plant rooms and ground-floor units damages finishes, electrics and lifts, displaces "
                 "occupants and can take months to dry out and re-commission.",
        "wind": "Roof coverings, cladding, windows and rooftop plant fail first; the resulting water ingress damages the "
                "interior, and debris endangers people outside.",
        "slope": "Sites at the foot of slopes are exposed to debris and surface runoff; retaining walls and drainage "
                 "are the points of failure.",
        "fire": "Combustible cladding, vegetation within 30 m and exposed roof voids are the main ember-ingress paths.",
        "heat": "Higher cooling load and overheating of occupied space; poorly ventilated or heavily glazed buildings "
                "become unsafe in heatwaves.",
        "water": "Supply restrictions in dry years affect cooling towers, irrigation and fire-fighting reserves.",
    },
    "data_centre": {
        "label": "Data centre",
        "critical": "UPS and battery rooms, generators and fuel, chillers and cooling plant, switchgear, network "
                    "entry points and access roads",
        "flood": "Water in electrical and battery rooms or the generator yard stops the facility; flooded access roads "
                 "block fuel deliveries, and a long outage means failover to another site.",
        "wind": "Rooftop chillers, generator exhausts and power lines are wind-sensitive; losing both grid feeds leaves "
                "the site on generators and fuel stock.",
        "slope": "Sites on slopes are exposed to runoff and washouts that can block access.",
        "fire": "Vegetation near the site and cable routes are ember and smoke exposures; air intakes are sensitive to smoke.",
        "heat": "High ambient temperature reduces chiller capacity and can exceed the design temperature of cooling "
                "plant, forcing load shedding.",
        "water": "Evaporative cooling needs water; high stress raises the chance of supply restrictions.",
    },
    "port_logistics": {
        "label": "Port or logistics site",
        "critical": "quays and yard drainage, cranes, electrical supply, rail and road connections, warehouses and "
                    "fuel stores",
        "flood": "Quays, yards and approach roads under water stop handling; flooded electrics and low warehouse floors "
                 "damage stock, and the site is cut off even if it stays dry.",
        "wind": "Cranes, container stacks, warehouse roofs and cladding are wind-loaded; cranes are normally "
                "taken out of service above a set wind speed, halting operations.",
        "slope": "Approach roads on slopes are exposed to washouts and landslips that cut access.",
        "fire": "Vegetation beside the site and combustible stock are the exposures.",
        "heat": "High ambient temperature restricts outdoor working and derates electrical equipment.",
        "water": "Less critical to operations; relevant for cooling and fire-fighting reserves.",
    },
    "agriculture": {
        "label": "Agricultural or food site",
        "critical": "fields and irrigation, storage and processing buildings, cold chain, access roads, water source",
        "flood": "Flooded fields lose crops and soil; processing buildings and cold stores lose stock, and roads cut "
                 "market access.",
        "wind": "Greenhouses, storage sheds and tree crops are damaged by strong wind.",
        "slope": "Fields and access roads on slopes are exposed to erosion and washouts.",
        "fire": "Dry vegetation and stored crop are the ignition and spread paths.",
        "heat": "Heat stress cuts yields and animal welfare and raises cold-chain load.",
        "water": "Irrigation depends on water availability; high stress raises the chance of supply restrictions.",
    },
    "generic": {
        "label": "Industrial site",
        "critical": "main buildings and process equipment, electrical supply and switchgear, drainage, access roads, "
                    "storage areas",
        "flood": "Water reaching electrical rooms, equipment and stock stops operations; flooded access roads cut "
                 "the site off even if it stays dry.",
        "wind": "Roofs, cladding, doors and rooftop equipment fail first; loss of power lines and blocked roads follow.",
        "slope": "Sites near slopes are exposed to runoff, debris and washouts that damage drainage and block access.",
        "fire": "Vegetation near the site and combustible stores are the ember-ingress and spread paths.",
        "heat": "High ambient temperature derates equipment and raises worker heat stress.",
        "water": "High stress raises the chance of supply restrictions in dry years.",
    },
}

# typical ageing mechanisms (engineering judgement, not a condition survey)
AGE_NOTES = {
    "flood": "Floor levels, bunds and drainage were set to the rainfall and flood records available at construction; "
             "drains silt up and pumps wear with age, so today's capacity may be below the original design.",
    "wind": "Roof fixings, cladding, sheeting and door seals loosen and corrode with age, and older structures were "
            "designed to the wind records and code of their time.",
    "coast": "Coastal exposure accelerates corrosion of reinforcement and steel with age, which reduces both strength "
             "and flood resilience.",
    "slope": "Retaining structures, slope drainage and cut-slope protection deteriorate with age.",
    "fire": "Older cladding, roof voids and ventilation openings are more likely to let embers in.",
    "heat": "Cooling and ventilation plant is sized to historical design temperatures and loses capacity with age.",
    "water": "Older plant tends to be less water-efficient and harder to retrofit.",
}


def profile_key(sector: Optional[str], atype: Optional[str]) -> str:
    s, a = (sector or "").lower(), (atype or "").lower()
    t = f"{s} {a}"
    for key, words in (("cement", ("cement",)), ("steel", ("steel", "iron-and")), ("aluminium", ("alumin", "bauxite")),
                       ("power", ("electricity", "power")), ("oil_gas", ("refin", "oil-and-gas", "oil_gas", "oil and gas")),
                       ("chemicals", ("chem",)), ("pulp", ("pulp", "paper")),
                       ("mining", ("mining", "mine")), ("data_centre", ("data",)),
                       ("port_logistics", ("port", "logistic", "warehouse")),
                       ("agriculture", ("agri", "farm", "food")),
                       ("real_estate", ("real", "estate", "building", "commercial", "residential", "office", "hospital",
                                        "retail"))):
        if any(w in t for w in words):
            return key
    return "generic"


# ── small helpers ─────────────────────────────────────────────────────────────

def _f(label, value, source=None, unit=""):
    return {"label": label, "value": value if not unit else f"{value} {unit}", "source": source}


def _ev_facts(finding: dict) -> list:
    out = []
    for e in finding.get("evidence", []):
        v = e.get("value")
        out.append(_f(e.get("metric"), f"{v} {e.get('unit') or ''}".strip(), e.get("source")))
    return out


def _zone_from_finding(f: dict, what: str) -> str:
    ev = [f"{e_.get('metric')}: {e_.get('value')} {e_.get('unit') or ''}".strip() for e_ in f.get("evidence", [])[:3]]
    lead = (f"No significant {what} exposure found" if _lv(f["severity"]) == 0 else
            f"{f['severity'].capitalize()} {what} exposure")
    return lead + (" (" + "; ".join(ev) + ")." if ev else ".")


def _depth_stats(depths: list) -> dict:
    if not depths:
        return {"wet_pct": 0.0, "mean_m": 0.0, "max_m": 0.0}
    n = len(depths)
    return {"wet_pct": round(100 * sum(1 for d in depths if d > 0.05) / n, 0),
            "mean_m": round(sum(depths) / n, 2), "max_m": round(max(depths), 2)}


def _depth_rating(mean_depth: float, protected: bool) -> str:
    """Thresholds: 0.05 m = wet; 0.3 m ≈ plinth/floor height; 1 m ≈ switchgear height; 2 m ≈ ground-floor ceiling."""
    r = 0 if mean_depth < 0.05 else 1 if mean_depth < 0.3 else 2 if mean_depth < 1.0 else 3 if mean_depth < 2.0 else 4
    return LEVELS[max(0, r - 1) if protected and r else r]


def _nearest(rps: list, target: int) -> int:
    return min(range(len(rps)), key=lambda i: abs(rps[i] - target))


def _first_wet(rows: list) -> Optional[int]:
    for r in rows:
        if r["wet_pct"] >= 5:
            return r["rp"]
    return None


def _sentence_site(a: dict, terr: dict, lc: dict, sat: dict) -> str:
    bits = []
    if terr.get("elevation_m") is not None:
        pos = terr.get("terrain_position") or "terrain"
        bits.append(f"sits at {terr['elevation_m']:.0f} m above sea level on {pos} ground "
                    f"(slope {terr.get('slope_at_asset_deg', 0):.1f}°)")
    cl = (lc.get("class_at_asset") or "").replace("_", "-") if (lc.get("class_at_asset") or "") == "built_up" else (lc.get("class_at_asset") or "").replace("_", " ")
    if cl:
        b250 = lc.get("built_up_within_250m")
        bits.append(f"in {cl} land cover" + (f" ({100 * b250:.0f}% built-up within 250 m)" if b250 is not None else ""))
    if sat.get("nearest_water_m") is not None:
        bits.append(f"the nearest surface water is about {sat['nearest_water_m']:.0f} m away")
    if terr.get("upslope_area_km2") is not None and terr.get("elevation_head_m") is not None:
        bits.append(f"{terr['upslope_area_km2']:.1f} km² of land drains towards it from up to "
                    f"{terr['elevation_head_m']:.0f} m higher")
    name = a.get("name") or "The site"
    return f"{name} " + ", ".join(bits[:2]) + ("; " + "; ".join(bits[2:]) if bits[2:] else "") + "." if bits else ""


# ── building one asset report ────────────────────────────────────────────────

def build_asset_report(lat: float, lon: float, name: Optional[str] = None, sector: Optional[str] = None,
                       asset_type: Optional[str] = None, commissioned_year: Optional[int] = None,
                       include_image: bool = True, radius_km: float = 2.5, _flood: Optional[dict] = None,
                       _assess: Optional[dict] = None) -> dict:
    from .analysis.asset_intelligence import assess_asset
    from .finance import analyst, geo
    from .finance.flood_hazard import site_flood
    t0 = time.time()
    reg = None
    if not sector:
        i = analyst.nearest_register_asset(lat, lon)
        if i is not None:
            reg = analyst.register_asset(i)
            name = name or reg["name"]
            sector = reg["sector"]
            asset_type = asset_type if asset_type is not None else reg["atype"]
    pk = profile_key(sector, asset_type)
    prof = PROFILES[pk]
    adm = geo.admin_at(lat, lon)
    iso3 = adm.get("iso3")
    ai_type = {"data_centre": "data_centre", "port_logistics": "port", "agriculture": "agriculture"}.get(pk, "industrial")
    ai = _assess if _assess is not None else assess_asset(lat, lon, name or "Site", ai_type, radius_km=radius_km,
                                                           include_visuals=include_image)
    fl = _flood if _flood is not None else site_flood(lat, lon, 250.0, elev_m=(ai.get("terrain") or {}).get("elevation_m"))
    terr, sat, lc, sar = ai.get("terrain") or {}, ai.get("satellite") or {}, ai.get("land_cover") or {}, ai.get("radar") or {}
    findings = {f["hazard"]: f for f in ai.get("findings", [])}
    # Open water / unmapped: the terrain model also covers the sea floor (and reads 0 m for much of the open sea), so an offshore
    # pin used to be rated as a "valley" 5 km down with CRITICAL surface-water flooding, and a pole (no layers at all) as
    # NEGLIGIBLE. Neither is a finding. Every real island nation has an admin unit in the boundary layer; open water has none.
    _elev = terr.get("elevation_m")
    if (not adm.get("found")) or (not adm.get("name") and not iso3 and _elev is not None and _elev < -15):
        raise ValueError(
            f"({lat:.4f}, {lon:.4f}) is not on assessable land: no administrative unit in the engine's boundary layer covers it "
            "(open water, or land the layer does not map such as the polar caps)"
            + (f", and the terrain model puts it {abs(_elev):.0f} m below sea level" if _elev is not None and _elev < -15 else "")
            + ". A land-based physical-risk rating is not meaningful there; move the pin onto the site.")
    asset = {"name": name or "Site", "lat": lat, "lon": lon, "country": iso3, "admin1": adm.get("name"),
             "sector": sector, "asset_type": asset_type, "profile": pk, "profile_label": prof["label"],
             "commissioned_year": commissioned_year}

    # ── age and design ───────────────────────────────────────────────────────
    this_year = date.today().year
    age = {"provided": bool(commissioned_year)}
    if commissioned_year:
        y = int(commissioned_year)
        if not 1800 <= y <= this_year:
            raise ValueError(f"commissioned_year must be between 1800 and {this_year}")
        age.update(commissioned=y, age_now=this_year - y, age_2050=REF_YEAR - y)
        from .finance.distribution import DESIGN_LIFE
        from .finance.transition import activity_key
        from .finance.asset_value import cost_key
        ak = activity_key(sector or "", asset_type or "", cost_key(sector or "", asset_type or ""))
        dl = DESIGN_LIFE.get(ak or "")
        if dl:
            age["design_life_years"] = {"low": dl[0], "typical": dl[1], "high": dl[2]}
            age["design_life_source"] = "sector design-life range used by the engine (analyst judgement, not a condition survey)"
            age["typical_end_of_design_life"] = y + dl[1]
            if age["age_2050"] > dl[2]:
                age["statement"] = (f"Built {y}; by 2050 it will be {age['age_2050']} years old, beyond the upper end of the "
                                    f"typical design life ({dl[2]} years). Major refurbishment or replacement is likely before "
                                    "then, which is the cheapest moment to build in the protection below.")
            elif age["age_2050"] > dl[1]:
                age["statement"] = (f"Built {y}; by 2050 it will be {age['age_2050']} years old, past the typical design life "
                                    f"({dl[1]} years, range {dl[0]}–{dl[2]}). Expect life-extension work within the period.")
            else:
                age["statement"] = (f"Built {y}; {age['age_now']} years old now and {age['age_2050']} in 2050, within the "
                                    f"typical design life ({dl[1]} years, range {dl[0]}–{dl[2]}).")
        else:
            age["statement"] = f"Built {y}; {age['age_now']} years old now and {age['age_2050']} in 2050."
    else:
        age["statement"] = ("Construction year not provided. Add it to see age, remaining design life and the "
                            "ageing mechanisms that apply to each hazard.")
    old = bool(commissioned_year) and (this_year - int(commissioned_year)) >= 15

    # ── how it sits ──────────────────────────────────────────────────────────
    site_facts = []
    if terr:
        site_facts += [_f("Elevation", f"{terr.get('elevation_m'):.0f}", "AWS Terrain Tiles ~10 m", "m a.s.l."),
                       _f("Terrain position", terr.get("terrain_position"), "AWS Terrain Tiles"),
                       _f("Slope at the site", f"{terr.get('slope_at_asset_deg'):.1f}", "AWS Terrain Tiles", "°"),
                       _f("Catchment draining to the site", f"{terr.get('upslope_area_km2'):.2f}", "pysheds D8 flow routing", "km²"),
                       _f("Height of upslope land above the site", f"{terr.get('elevation_head_m'):.0f}", "AWS Terrain Tiles", "m")]
    if lc.get("available"):
        site_facts += [_f("Land cover at the site", (lc.get("class_at_asset") or "").replace("_", " "), "ESA WorldCover 2021"),
                       _f("Built-up land within 250 m", f"{100 * (lc.get('built_up_within_250m') or 0):.0f}", "ESA WorldCover 2021", "%"),
                       _f("Tree cover within 1 km", f"{100 * (lc.get('tree_cover_within_1km') or 0):.0f}", "ESA WorldCover 2021", "%")]
    if sat.get("nearest_water_m") is not None:
        site_facts.append(_f("Nearest surface water", f"{sat['nearest_water_m']:.0f}", f"Sentinel-2 ({sat.get('scene_date')})", "m"))
    if sar.get("available") and sar.get("ever_wet_within_500m") is not None:
        site_facts.append(_f("Land within 500 m that has been under water in the radar record",
                             f"{100 * sar['ever_wet_within_500m']:.0f}", f"Sentinel-1 ({sar.get('date_range')})", "%"))
    site_facts.append(_f("Location", f"{lat:.4f}, {lon:.4f}" + (f" — {adm.get('name')}, {iso3}" if adm.get("name") else ""), None))
    site = {"summary": _sentence_site(asset, terr, lc, sat), "facts": site_facts}

    # ── hazard zones ─────────────────────────────────────────────────────────
    zones = []
    riv = fl.get("river") or {}
    cst = fl.get("coastal") or {}
    prot_r, prot_c = float(adm.get("prot_river_rp") or 0), float(adm.get("prot_coast_rp") or 0)

    # river
    if riv.get("available") and not riv.get("footprint_on_water"):
        rows = [dict(rp=rp, **_depth_stats(riv["depths"].get(str(rp), []))) for rp in riv["rps"]]
        i100 = _nearest(riv["rps"], 100)
        rating = _depth_rating(rows[i100]["mean_m"], prot_r >= 100)
        fw = _first_wet(rows)
        if fw is None:
            zone = f"Outside the modelled river floodplain up to a 1-in-{riv['rps'][-1]}-year flood."
        else:
            zone = (f"Inside the modelled river floodplain: the site first floods at a 1-in-{fw}-year event; at 1-in-100 "
                    f"{rows[i100]['wet_pct']:.0f}% of the site footprint is under a mean {rows[i100]['mean_m']:.1f} m "
                    f"(deepest {rows[i100]['max_m']:.1f} m).")
        zones.append({"key": "river_flood", "title": "River flooding", "rating": rating, "zone": zone,
                      "table": None if fw is None else {"header": ["Flood return period (years)", "Site footprint under water", "Mean depth (m)",
                                           "Deepest (m)"],
                                "rows": [[f"1-in-{r['rp']}", f"{r['wet_pct']:.0f}%", f"{r['mean_m']:.2f}", f"{r['max_m']:.2f}"]
                                         for r in rows]},
                      "facts": [_f("Local flood-defence standard", f"about 1-in-{prot_r:.0f} years" if prot_r else "none recorded",
                                   "FLOPROS (regional estimate)"),
                                _f("Flood maps", "undefended; floods rarer than the defence standard overtop it" if prot_r else
                                   "undefended", "JRC CEMS-GloFAS v2.1")],
                      "trend": "Frequencies are historical; no climate-change uplift is applied to river floods.",
                      "_depth100": rows[i100]["mean_m"], "_first_wet": fw, "_rows": rows})
    elif riv.get("footprint_on_water"):
        zones.append({"key": "river_flood", "title": "River flooding", "rating": None,
                      "zone": "Not assessed: the site footprint lies on a permanent water body. Move the pin onto the site.",
                      "facts": []})
    else:
        zones.append({"key": "river_flood", "title": "River flooding", "rating": None,
                      "zone": "Not assessed: " + (riv.get("reason") or "no river flood map covers this location") + ".",
                      "facts": []})

    # coastal
    if cst.get("available"):
        rps = cst["rps"]
        rows = []
        for rp in rps:
            a18 = _depth_stats(cst["depths"].get(f"2018_{rp}") or [])
            a50 = _depth_stats(cst["depths"].get(f"2050_{rp}") or [])
            rows.append({"rp": rp, "a18": a18, "a50": a50})
        i100 = _nearest(rps, 100)
        r50 = rows[i100]["a50"]["mean_m"]
        rating = _depth_rating(r50, prot_c >= 100)
        fw50 = next((r["rp"] for r in rows if r["a50"]["wet_pct"] >= 5), None)
        fw18 = next((r["rp"] for r in rows if r["a18"]["wet_pct"] >= 5), None)
        if fw50 is None:
            zone = f"Outside the modelled storm-surge zone up to a 1-in-{rps[-1]}-year event, now and at 2050 sea level."
        else:
            zone = (f"Inside the modelled storm-surge zone. At 2050 sea level the site first floods at a 1-in-{fw50}-year "
                    f"event" + (f" (today: 1-in-{fw18})" if fw18 else " (today: not within the modelled range)") +
                    f"; at 1-in-100, {rows[i100]['a50']['wet_pct']:.0f}% of the footprint is under a mean "
                    f"{r50:.1f} m (today {rows[i100]['a18']['mean_m']:.1f} m).")
        zones.append({"key": "coastal_flood", "title": "Coastal flooding and sea-level rise", "rating": rating, "zone": zone,
                      "table": None if fw50 is None else {"header": ["Return period (years)", "Under water, today", "Mean depth today (m)",
                                           "Under water, 2050", "Mean depth 2050 (m)"],
                                "rows": [[f"1-in-{r['rp']}", f"{r['a18']['wet_pct']:.0f}%", f"{r['a18']['mean_m']:.2f}",
                                          f"{r['a50']['wet_pct']:.0f}%", f"{r['a50']['mean_m']:.2f}"] for r in rows]},
                      "facts": [_f("Coastal defence standard", f"about 1-in-{prot_c:.0f} years" if prot_c else "none recorded",
                                   "FLOPROS (regional estimate)")],
                      "trend": "Depths include sea-level rise to 2050.", "_depth100": r50, "_first_wet": fw50})
    elif "above 30 m" in str(cst.get("reason") or ""):
        zones.append({"key": "coastal_flood", "title": "Coastal flooding and sea-level rise", "rating": "NEGLIGIBLE",
                      "zone": "Site is above 30 m; coastal flooding is not a hazard here.", "facts": []})
    else:
        zones.append({"key": "coastal_flood", "title": "Coastal flooding and sea-level rise", "rating": None,
                      "zone": "Not assessed: " + (str(cst.get("reason") or "no coastal flood map could be read for this location")[:200]) + ".",
                      "facts": []})

    # surface-water / observed flooding (terrain + satellite + radar)
    ff = findings.get("flood")
    if ff:
        zones.append({"key": "surface_water", "title": "Surface-water and observed flooding", "rating": ff["severity"],
                      "zone": ff.get("summary") or _zone_from_finding(ff, "surface-water flooding"),
                      "facts": _ev_facts(ff), "trend": None, "_finding": ff})

    # cyclone
    tcf = findings.get("tropical_cyclone")
    if tcf:
        try:
            tcp = analyst._tc_params(lat, lon, reg)
            zone = (f"Inside a tropical-cyclone region: the 1-in-100-year wind is about {tcp['rp100_ms']:.0f} m/s "
                    f"({tcp['category_100yr']}).")
        except Exception:                                   # noqa: BLE001
            zone = "Inside a tropical-cyclone region."
        zones.append({"key": "cyclone", "title": "Tropical cyclone wind", "rating": tcf["severity"], "zone": zone,
                      "facts": _ev_facts(tcf), "trend": "Frequencies are historical (1980–2025); no climate-change uplift is applied.",
                      "_finding": tcf})
    elif "tropical_cyclone" in (ai.get("errors") or {}):
        zones.append({"key": "cyclone", "title": "Tropical cyclone wind", "rating": None,
                      "zone": "Not assessed (data unavailable).", "facts": []})

    # slope
    sl = [findings[k] for k in ("landslide", "debris_flow") if k in findings]
    if sl:
        rating = _max_level([f["severity"] for f in sl])
        facts = []
        for f in sl:
            facts += _ev_facts(f)
        zones.append({"key": "slope", "title": "Landslide and debris flow", "rating": rating,
                      "zone": ("Terrain upslope can deliver material or water to the site." if _lv(rating) >= 2 else
                               "Terrain around the site gives low susceptibility to slope failure or debris flow."),
                      "facts": facts, "trend": None, "_finding": sl[0]})
    wf = findings.get("wildfire")
    if wf:
        zones.append({"key": "wildfire", "title": "Wildfire", "rating": wf["severity"],
                      "zone": wf.get("summary") or _zone_from_finding(wf, "wildfire"),
                      "facts": _ev_facts(wf), "trend": None, "_finding": wf})
    hf = findings.get("urban_heat")
    if hf:
        zones.append({"key": "heat", "title": "Heat (land-cover indicator)", "rating": hf["severity"],
                      "zone": _zone_from_finding(hf, "heat") + " Indicator from land cover and latitude, not a temperature projection.",
                      "facts": _ev_facts(hf), "trend": None, "_finding": hf})

    # water stress
    try:
        w = analyst._water(lat, lon, sector)
    except Exception:                                       # noqa: BLE001
        w = None
    if w:
        ov = w.get("overall") or {}
        # Same quantity and thresholds as the Asset Intelligence panel: the larger of baseline water stress and
        # drought risk, so the two outputs give the same rating for the same place.
        _q = [s for s in ((w.get("water_stress") or {}).get("score"), (w.get("drought") or {}).get("score")) if s is not None]
        sc = max(_q) if _q else ov.get("score")
        rating = None if sc is None else LEVELS[0 if sc < 1 else 1 if sc < 2 else 2 if sc < 3 else 3 if sc < 4 else 4]
        path = w.get("water_stress_path_bau") or {}
        zones.append({"key": "water", "title": "Water stress (context)", "rating": rating,
                      "zone": f"Catchment water risk is {ov.get('label') or (w.get('water_stress') or {}).get('label') or 'not scored'} "
                              f"(WRI Aqueduct 4.0, {w.get('admin1')}).",
                      "facts": [_f("Baseline water stress", (w.get("water_stress") or {}).get("label"), "WRI Aqueduct 4.0"),
                                _f("Drought risk", (w.get("drought") or {}).get("label"), "WRI Aqueduct 4.0"),
                                _f("Groundwater decline", (w.get("groundwater_decline") or {}).get("label"), "WRI Aqueduct 4.0")],
                      "trend": ("Water stress under business-as-usual: " + ", ".join(f"{k} {v}" for k, v in list(path.items())[:4])
                                if isinstance(path, dict) and path else None)})

    # ── issues: what will go wrong ───────────────────────────────────────────
    group = {"river_flood": "flood", "coastal_flood": "flood", "surface_water": "flood", "cyclone": "wind", "slope": "slope",
             "wildfire": "fire", "heat": "heat", "water": "water"}
    agekey = {"river_flood": "flood", "coastal_flood": "coast", "surface_water": "flood", "cyclone": "wind", "slope": "slope",
              "wildfire": "fire", "heat": "heat", "water": "water"}
    issues = []
    for z in zones:
        if z.get("rating") is None or _lv(z["rating"]) < 1:
            continue
        g = group[z["key"]]
        what = [prof[g]]
        if z["key"] == "river_flood" and z.get("_depth100"):
            d = z["_depth100"]
            what.insert(0, f"A 1-in-100 flood puts about {d:.1f} m of water over the site; anything below that level "
                           "(ground-floor equipment, electrical rooms, stock) is inundated.")
        if z["key"] == "coastal_flood" and z.get("_depth100"):
            what.insert(0, f"At 2050 sea level a 1-in-100 surge puts about {z['_depth100']:.1f} m of water over the site.")
        issue = {"hazard": z["key"], "title": z["title"], "rating": z["rating"], "where": z["zone"], "what_will_happen": what,
                 "components_at_risk": prof["critical"],
                 "age_note": AGE_NOTES[agekey[z["key"]]] if old and _lv(z["rating"]) >= 1 else None}
        issues.append(issue)
    issues.sort(key=lambda i: (i["hazard"] == "water", -_lv(i["rating"])))

    # ── measures ─────────────────────────────────────────────────────────────
    measures = []
    for i in issues:
        z = next(zz for zz in zones if zz["key"] == i["hazard"])
        acts = []
        f = z.get("_finding")
        if f:
            acts += f.get("actions", [])
        if z["key"] == "river_flood" and _lv(z["rating"]) >= 2:
            d = z.get("_depth100") or 0
            acts += [f"Raise or wet-proof critical equipment (switchgear, motor control rooms, control rooms) above "
                     f"{d + 0.5:.1f} m, the 1-in-100 depth plus 0.5 m freeboard.",
                     "Fit demountable barriers at openings and non-return valves on drainage outfalls; write a flood "
                     "emergency plan with water-level triggers.",
                     "Confirm the actual defence standard with the local water authority."]
        if z["key"] == "coastal_flood" and _lv(z["rating"]) >= 2:
            acts += ["Design any new or refurbished low-lying plant to the 2050 1-in-100 surge level.",
                     "Check the design level and condition of seawalls and levees; relocate critical stock above surge level."]
        if z["key"] == "water" and _lv(z["rating"]) >= 3:
            acts += ["Map the site's water sources and consumption; secure a second source or on-site storage for dry years."]
        if i.get("age_note") and _lv(z["rating"]) >= 2:
            acts.append("Commission a condition survey of the relevant structures and systems (age-related deterioration).")
        acts = list(dict.fromkeys(acts))[:6]
        if acts:
            measures.append({"hazard": z["key"], "title": z["title"], "rating": z["rating"],
                             "when": "Immediately" if _lv(z["rating"]) >= 3 else "Within 12 months" if _lv(z["rating"]) == 2
                             else "Monitor and review annually", "actions": acts})

    rated = [z["rating"] for z in zones if z.get("rating") and z["key"] != "water"]
    overall = _max_level(rated)
    top = [z for z in sorted(zones, key=lambda z: -_lv(z.get("rating") or "NEGLIGIBLE"))
           if _lv(z.get("rating") or "NEGLIGIBLE") >= 2 and z["key"] != "water"]
    wz = next((z for z in zones if z["key"] == "water" and _lv(z.get("rating") or "NEGLIGIBLE") >= 3), None)
    headline = (f"{asset['name']} ({prof['label'].lower()}): overall physical exposure {overall}. "
                + (("Main exposures: " + "; ".join(f"{z['title'].lower()} ({z['rating']})" for z in top[:3]) + ".") if top else
                   "No hazard reaches moderate exposure at this location on the available evidence.")
                + (f" Water supply context: {wz['rating'].lower()} water stress." if wz else ""))

    image = None
    vis = ai.get("visual") or {}
    if include_image and vis.get("texture_jpeg_b64") and vis.get("bounds"):
        try:
            image = _annotate(vis["texture_jpeg_b64"], vis["bounds"], lat, lon, name or "Site", vis.get("scene_date"))
        except Exception:                                   # noqa: BLE001
            image = None

    cautions = []
    elev = terr.get("elevation_m")
    cz = next((z for z in zones if z["key"] == "coastal_flood"), None)
    if (elev is not None and elev < 5 and sat.get("nearest_water_m") is not None and sat["nearest_water_m"] < 500
            and cz and cz.get("rating") == "NEGLIGIBLE" and cst.get("available")):
        cautions.append(f"The site is only {elev:.0f} m above sea level and about {sat['nearest_water_m']:.0f} m from water, yet the "
                        "coastal flood map shows no flooding. The map uses a 90 m surface model that can overstate ground "
                        "height in dense urban areas, so treat this as unconfirmed and check against a local flood study.")
    sz = next((z for z in zones if z["key"] == "slope"), None)
    if elev is not None and elev < 10 and cz and cz.get("rating") in (None, "NEGLIGIBLE", "LOW") and \
            (sat.get("nearest_water_m") is None or sat["nearest_water_m"] < 1500):
        cautions.append(f"The elevation model puts this site at {elev:.0f} m above sea level, close to the sea. Global terrain (about 30 m) and "
                        "coastal-flood maps (about 90 m) cannot resolve atolls, reclaimed land, dikes or narrow coastal strips, and routinely read "
                        "low islands several metres too high (Tuvalu, the Maldives and Kiribati are the usual examples), so a low or negligible "
                        "coastal rating here is unconfirmed. Use a surveyed ground level and a local inundation study.")
    if elev is not None and elev < 5 and sz and sz.get("rating") in ("HIGH", "CRITICAL"):
        cautions.append("The slope rating comes from a terrain model that is at or near sea level here; steep cells next to the shore are usually "
                        "sea-floor or noise rather than land that can slide. Treat the landslide rating as unconfirmed.")
    if "water" in str(lc.get("class_at_asset") or "").lower():
        cautions.append("The pin sits on permanent water in the land-cover map (a lake, sea or river). Every rating below describes "
                        "that location as if it were land; if the asset is on the shore, move the pin onto it.")
    gaps = [f"{k}: {v}" for k, v in (ai.get("errors") or {}).items()]
    for z in zones:
        z.pop("_finding", None)
        z.pop("_rows", None)
    return {"asset": asset, "headline": headline, "overall_rating": overall, "site": site, "age": age, "zones": zones,
            "issues": issues, "measures": measures, "image": image, "data_gaps": gaps, "cautions": cautions,
            "sources": ai.get("sources", []) + ["JRC CEMS-GloFAS river flood hazard maps v2.1",
                                                  "Deltares Global Flood Maps (Planetary Computer)",
                                                  "FLOPROS v1 (Scussolini et al. 2016)", "WRI Aqueduct 4.0"],
            "basis": ["Ratings describe hazard intensity at the site against stated thresholds; they are not damage estimates.",
                      "Flood depth rating (mean over a 250 m footprint at a 1-in-100-year event): below 0.05 m negligible, "
                      "below 0.3 m low, below 1 m moderate, below 2 m high, above that critical; one level lower if the "
                      "local defence standard is 1-in-100 or better.",
                      "Screening-grade: it prioritises sites for survey and does not replace a hydraulic flood model, a "
                      "structural condition survey or a geotechnical investigation.",
                      "Terrain comes from a surface model: buildings and trees can raise apparent ground height."],
            "elapsed_s": round(time.time() - t0, 1)}


def _annotate(jpeg_b64: str, bounds: list, lat: float, lon: float, name: str, scene_date: Optional[str]) -> str:
    """Satellite image with a 250 m ring on the site and a 500 m scale bar; returns a data-URI JPEG."""
    from PIL import Image, ImageDraw
    im = Image.open(io.BytesIO(base64.b64decode(jpeg_b64))).convert("RGB")
    w, h = im.size
    lon0, lat0, lon1, lat1 = bounds
    m_w = (lon1 - lon0) * 111320.0 * math.cos(math.radians(lat))
    ppm = w / m_w
    cx = (lon - lon0) / (lon1 - lon0) * w
    cy = (lat1 - lat) / (lat1 - lat0) * h
    d = ImageDraw.Draw(im)
    r = 250 * ppm
    d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(255, 255, 255), width=3)
    d.ellipse([cx - r - 1, cy - r - 1, cx + r + 1, cy + r + 1], outline=(220, 38, 38), width=1)
    d.line([cx - 12, cy, cx + 12, cy], fill=(220, 38, 38), width=2)
    d.line([cx, cy - 12, cx, cy + 12], fill=(220, 38, 38), width=2)
    bar = 500 * ppm
    x0, y0 = 16, h - 22
    d.rectangle([x0 - 4, y0 - 14, x0 + bar + 70, y0 + 10], fill=(0, 0, 0))
    d.line([x0, y0, x0 + bar, y0], fill=(255, 255, 255), width=3)
    d.text((x0 + bar + 8, y0 - 6), "500 m", fill=(255, 255, 255))
    d.rectangle([0, 0, w, 20], fill=(0, 0, 0))
    d.text((8, 4), f"{name} · Sentinel-2 true colour {scene_date or ''} · ring = 250 m", fill=(255, 255, 255))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


# ── portfolio: grouped by sector ─────────────────────────────────────────────

def build_portfolio_report(assets: list, title: Optional[str] = None, client: Optional[str] = None,
                           include_image: bool = True, workers: int = 3) -> dict:
    def one(a):
        try:
            return build_asset_report(float(a["lat"]), float(a["lon"]), a.get("name"), a.get("sector"), a.get("asset_type"),
                                      a.get("commissioned_year"), include_image=include_image)
        except Exception as exc:                            # noqa: BLE001
            return {"error": f"{a.get('name') or a.get('lat')}: {exc}", "asset": {"name": a.get("name")}}
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        reps = list(pool.map(one, assets))
    ok = [r for r in reps if "error" not in r]
    failed = [r["error"] for r in reps if "error" in r]
    sectors: dict = {}
    for r in ok:
        sectors.setdefault(r["asset"]["profile"], []).append(r)
    blocks = []
    for pk, rs in sectors.items():
        rs.sort(key=lambda r: -_lv(r["overall_rating"]))
        hz: dict = {}
        for r in rs:
            for z in r["zones"]:
                if z.get("rating") in ("HIGH", "CRITICAL"):
                    hz.setdefault(z["title"], []).append(r["asset"]["name"])
        counts = {lvl: sum(1 for r in rs if r["overall_rating"] == lvl) for lvl in LEVELS}
        label = PROFILES[pk]["label"]
        msg = (f"{len(rs)} {label.lower()}{'s' if len(rs) != 1 else ''}: "
               + ", ".join(f"{n} {l.lower()}" for l, n in counts.items() if n) + " overall exposure."
               + ("".join(f" {t}: HIGH or CRITICAL at {len(v)} site{'s' if len(v) != 1 else ''}." for t, v in hz.items())))
        blocks.append({"sector": pk, "label": label, "count": len(rs), "rating_counts": counts, "high_hazards": hz,
                       "message": msg, "table": [{"name": r["asset"]["name"], "country": r["asset"]["country"],
                                                  "overall": r["overall_rating"],
                                                  "main": [z["title"] for z in sorted(
                                                      r["zones"], key=lambda z: -_lv(z.get("rating") or "NEGLIGIBLE"))
                                                      if _lv(z.get("rating") or "NEGLIGIBLE") >= 2 and z["key"] != "water"][:3],
                                                  "age_2050": (r["age"] or {}).get("age_2050")} for r in rs]})
    blocks.sort(key=lambda b: -max(_lv(k) for k, v in b["rating_counts"].items() if v))
    overall = {lvl: sum(1 for r in ok if r["overall_rating"] == lvl) for lvl in LEVELS}
    return {"title": title or "Physical Risk Assessment", "client": client, "generated": date.today().isoformat(),
            "n_assets": len(ok), "overall_counts": overall, "sectors": blocks, "assets": ok, "failed": failed}


# ── HTML ──────────────────────────────────────────────────────────────────────

_COL = {"NEGLIGIBLE": "#6b7280", "LOW": "#16a34a", "MODERATE": "#d97706", "HIGH": "#dc2626", "CRITICAL": "#7f1d1d"}
e = lambda s: _html.escape(str(s if s is not None else ""))     # noqa: E731


def _chip(r):
    return (f'<span class="chip" style="background:{_COL.get(r, "#374151")}">{e(r or "NOT ASSESSED")}</span>')


def _asset_html(r: dict) -> str:
    a = r["asset"]
    h = [f'<section class="asset"><h2>{e(a["name"])} <small>{e(a["profile_label"])} · {e(a.get("admin1") or "")} {e(a.get("country") or "")}</small></h2>',
         f'<div class="head">{_chip(r["overall_rating"])} <span>{e(r["headline"])}</span></div>']
    for c in r.get("cautions", []):
        h.append(f'<div class="caution"><b>Verify:</b> {e(c)}</div>')
    if r.get("image"):
        h.append(f'<img class="sat" src="{r["image"]}" alt="Satellite view of the site">')
    h.append(f'<h3>1 · How the site sits</h3><p>{e(r["site"]["summary"])}</p><table class="kv">' +
             "".join(f'<tr><td>{e(f["label"])}</td><td>{e(f["value"])}</td><td class="src">{e(f["source"])}</td></tr>' for f in r["site"]["facts"]) + "</table>")
    h.append('<h3>2 · Hazard zones</h3><table class="zt"><tr><th>Hazard</th><th>Rating</th><th>Where the site sits</th></tr>' +
             "".join(f'<tr><td>{e(z["title"])}</td><td>{_chip(z.get("rating"))}</td><td>{e(z["zone"])}'
                     + (f'<br><span class="src">{e(z["trend"])}</span>' if z.get("trend") else "") + "</td></tr>" for z in r["zones"]) + "</table>")
    for z in r["zones"]:
        if z.get("table"):
            t = z["table"]
            h.append(f'<h4>{e(z["title"])}: depths by return period</h4><table class="num"><tr>' + "".join(f"<th>{e(x)}</th>" for x in t["header"]) +
                     "</tr>" + "".join("<tr>" + "".join(f"<td>{e(c)}</td>" for c in row) + "</tr>" for row in t["rows"]) + "</table>")
    ag = r["age"]
    h.append(f'<h3>3 · Construction year and age</h3><p>{e(ag["statement"])}</p>')
    h.append('<h3>4 · What will go wrong</h3>')
    if r["issues"]:
        h.append(f'<p class="src"><b>Parts of this kind of site that matter most:</b> {e(r["issues"][0]["components_at_risk"])}.</p>')
    if not r["issues"]:
        h.append("<p>No hazard reaches a rating that needs action on the available evidence.</p>")
    for i in r["issues"]:
        h.append(f'<div class="issue"><h4>{e(i["title"])} {_chip(i["rating"])}</h4><p class="where">{e(i["where"])}</p>' +
                 "".join(f"<p>{e(t)}</p>" for t in i["what_will_happen"]) +
                 (f'<p class="age"><b>Because of its age:</b> {e(i["age_note"])} <span class="src">(typical ageing mechanism; confirm with a condition survey)</span></p>' if i.get("age_note") else "") + "</div>")
    h.append('<h3>5 · What to do</h3>')
    for m in r["measures"]:
        h.append(f'<div class="issue"><h4>{e(m["title"])} {_chip(m["rating"])} <small>{e(m["when"])}</small></h4><ul>' +
                 "".join(f"<li>{e(x)}</li>" for x in m["actions"]) + "</ul></div>")
    if r.get("data_gaps"):
        h.append('<p class="src">Data gaps in this run: ' + e("; ".join(r["data_gaps"])) + "</p>")
    h.append("</section>")
    return "".join(h)


_CSS = """body{font-family:-apple-system,Segoe UI,Helvetica,Arial,sans-serif;color:#111827;margin:0;background:#fff;line-height:1.45}
.wrap{max-width:900px;margin:0 auto;padding:28px}h1{font-size:26px;margin:0 0 4px}h2{font-size:20px;margin:30px 0 6px;border-top:3px solid #0f172a;padding-top:14px}
h2 small{font-weight:400;color:#6b7280;font-size:13px}h3{font-size:15px;margin:18px 0 6px;color:#0f172a}h4{font-size:13.5px;margin:12px 0 4px}
.chip{color:#fff;font-size:10.5px;padding:2px 8px;border-radius:10px;font-weight:600;letter-spacing:.3px;white-space:nowrap}
.head{display:flex;gap:10px;align-items:flex-start;background:#f3f4f6;padding:10px 12px;border-radius:6px;margin:8px 0}
table{border-collapse:collapse;width:100%;font-size:12.5px;margin:6px 0}td,th{border-bottom:1px solid #e5e7eb;padding:5px 8px;text-align:left;vertical-align:top}
th{background:#f9fafb;font-size:11.5px}.num td,.num th{text-align:right}.num td:first-child,.num th:first-child{text-align:left}
.kv td:first-child{width:36%;color:#374151}.src{color:#6b7280;font-size:11px}.sat{width:100%;border-radius:6px;margin:8px 0}
.issue{border-left:3px solid #d1d5db;padding:2px 12px;margin:10px 0}.age{background:#fffbeb;padding:6px 8px;border-radius:4px}
.where{color:#374151;font-style:italic}.caution{background:#fffbeb;border:1px solid #f59e0b;padding:8px 10px;border-radius:5px;font-size:12px;margin:6px 0}.foot{margin-top:30px;font-size:11px;color:#6b7280}.sector{margin:14px 0}
@media print{.wrap{padding:0}section.asset{page-break-before:always}h2{border-top:none}}"""


def render_html(rep: dict) -> str:
    if "assets" not in rep:
        rep = {"title": "Physical Risk Assessment", "client": None, "generated": date.today().isoformat(), "n_assets": 1,
               "overall_counts": {rep["overall_rating"]: 1}, "sectors": [], "assets": [rep], "failed": []}
    h = [f'<!doctype html><html><head><meta charset="utf-8"><title>{e(rep["title"])}</title><style>{_CSS}</style></head><body><div class="wrap">',
         f'<h1>{e(rep["title"])}</h1><div class="src">{e(rep.get("client") or "")} · {e(rep["generated"])} · {rep["n_assets"]} site(s)</div>']
    if len(rep["assets"]) > 1:
        h.append("<h3>Summary by sector</h3>")
        for b in rep["sectors"]:
            h.append(f'<div class="sector"><b>{e(b["label"])}</b> — {e(b["message"])}<table><tr><th>Site</th><th>Overall</th><th>Main exposures</th><th>Age in 2050</th></tr>' +
                     "".join(f'<tr><td>{e(t["name"])} <span class="src">{e(t["country"])}</span></td><td>{_chip(t["overall"])}</td><td>{e(", ".join(t["main"]) or "—")}</td><td>{e(t["age_2050"] if t["age_2050"] is not None else "—")}</td></tr>' for t in b["table"]) +
                     "</table></div>")
    for r in rep["assets"]:
        h.append(_asset_html(r))
    if rep.get("failed"):
        h.append('<p class="src">Not assessed: ' + e("; ".join(rep["failed"])) + "</p>")
    h.append('<div class="foot"><b>Basis.</b> ' + e(" ".join(rep["assets"][0]["basis"])) + "<br><b>Sources.</b> " +
             e("; ".join(dict.fromkeys(s for r in rep["assets"] for s in r["sources"]))) + "</div></div></body></html>")
    return "".join(h)


# ── PDF ───────────────────────────────────────────────────────────────────────

def render_pdf(rep: dict) -> bytes:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import Image as RLImage, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    if "assets" not in rep:
        rep = {"title": "Physical Risk Assessment", "client": None, "generated": date.today().isoformat(), "n_assets": 1,
               "overall_counts": {rep["overall_rating"]: 1}, "sectors": [], "assets": [rep], "failed": []}
    ss = getSampleStyleSheet()
    B = ParagraphStyle("b", parent=ss["Normal"], fontSize=8.8, leading=12)
    S = ParagraphStyle("s", parent=B, fontSize=7.5, textColor=colors.HexColor("#6b7280"), leading=10)
    H1 = ParagraphStyle("h1", parent=ss["Title"], fontSize=20, alignment=0, spaceAfter=2)
    H2 = ParagraphStyle("h2", parent=ss["Heading2"], fontSize=14, spaceBefore=4, spaceAfter=4)
    H3 = ParagraphStyle("h3", parent=ss["Heading3"], fontSize=10.5, spaceBefore=9, spaceAfter=3, keepWithNext=1,
                        textColor=colors.HexColor("#0f172a"))
    P = lambda t, st=B: Paragraph(_html.escape(str(t)).replace("\n", "<br/>"), st)    # noqa: E731
    W = A4[0] - 30 * mm

    def chip(r):
        c = _COL.get(r, "#374151")
        return Paragraph(f'<font color="{c}"><b>{_html.escape(r or "NOT ASSESSED")}</b></font>', B)

    def tbl(data, widths, head=True):
        t = Table(data, colWidths=widths, repeatRows=1 if head else 0)
        st = [("FONTSIZE", (0, 0), (-1, -1), 8), ("VALIGN", (0, 0), (-1, -1), "TOP"),
              ("LINEBELOW", (0, 0), (-1, -1), 0.3, colors.HexColor("#e5e7eb")),
              ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3)]
        if head:
            st.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f3f4f6")))
        t.setStyle(TableStyle(st))
        return t

    def footer(c, d):
        c.saveState()
        c.setFont("Helvetica", 7)
        c.setFillColor(colors.HexColor("#6b7280"))
        c.drawString(15 * mm, 9 * mm, f"{rep['title']} · {rep['generated']} · screening-grade physical risk assessment")
        c.drawRightString(A4[0] - 15 * mm, 9 * mm, f"Page {d.page}")
        c.restoreState()

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=15 * mm, rightMargin=15 * mm, topMargin=14 * mm, bottomMargin=16 * mm,
                            title=rep["title"])
    el = [P(rep["title"], H1), P(" · ".join(x for x in (rep.get("client"), rep["generated"], f"{rep['n_assets']} site(s)") if x), S), Spacer(1, 6)]
    if len(rep["assets"]) > 1:
        el.append(P("Summary by sector", H2))
        for b in rep["sectors"]:
            el += [P(f"{b['label']}: {b['message']}"), Spacer(1, 3),
                   tbl([[P("Site", S), P("Overall", S), P("Main exposures", S), P("Age 2050", S)]] +
                       [[P(f"{t['name']} ({t['country'] or ''})"), chip(t["overall"]), P(", ".join(t["main"]) or "—"),
                         P(t["age_2050"] if t["age_2050"] is not None else "—")] for t in b["table"]],
                       [W * .34, W * .14, W * .42, W * .10]), Spacer(1, 8)]
    for r in rep["assets"]:
        a = r["asset"]
        if len(rep["assets"]) > 1:
            el.append(PageBreak())
        el += [P(a["name"], H2), P(f"{a['profile_label']} · {a.get('admin1') or ''} {a.get('country') or ''} · "
                                   f"{a['lat']:.4f}, {a['lon']:.4f}", S), Spacer(1, 3),
               tbl([[chip(r["overall_rating"]), P(r["headline"])]], [W * .16, W * .84], head=False)]
        for c in r.get("cautions", []):
            el += [Spacer(1, 3), P("Verify: " + c, S)]
        if r.get("image"):
            try:
                raw = base64.b64decode(r["image"].split(",", 1)[1])
                el += [Spacer(1, 4), RLImage(io.BytesIO(raw), width=W, height=W * 0.62, kind="proportional")]
            except Exception:                               # noqa: BLE001
                pass
        el += [P("1 · How the site sits", H3), P(r["site"]["summary"]),
               tbl([[P(f["label"], S), P(f["value"]), P(f["source"] or "", S)] for f in r["site"]["facts"]],
                   [W * .36, W * .34, W * .30], head=False),
               P("2 · Hazard zones", H3),
               tbl([[P("Hazard", S), P("Rating", S), P("Where the site sits", S)]] +
                   [[P(z["title"]), chip(z.get("rating")), P(z["zone"] + (f"\n{z['trend']}" if z.get("trend") else ""))]
                    for z in r["zones"]], [W * .22, W * .13, W * .65])]
        for z in r["zones"]:
            if z.get("table"):
                t = z["table"]
                el += [Spacer(1, 4), P(f"{z['title']}: depths by return period", S),
                       tbl([[P(x, S) for x in t["header"]]] + [[P(c) for c in row] for row in t["rows"]],
                           [W / len(t["header"])] * len(t["header"]))]
        el += [P("3 · Construction year and age", H3), P(r["age"]["statement"]), P("4 · What will go wrong", H3)]
        if r["issues"]:
            el.append(P("Parts of this kind of site that matter most: " + r["issues"][0]["components_at_risk"] + ".", S))
        if not r["issues"]:
            el.append(P("No hazard reaches a rating that needs action on the available evidence."))
        for i in r["issues"]:
            blk = [Spacer(1, 3), Paragraph(f"<b>{_html.escape(i['title'])}</b> — "
                                           f"<font color='{_COL.get(i['rating'], '#374151')}'><b>{i['rating']}</b></font>", B),
                   P(i["where"], S)] + [P(t) for t in i["what_will_happen"]] + \
                  []
            if i.get("age_note"):
                blk.append(P("Because of its age: " + i["age_note"] + " (typical ageing mechanism; confirm with a condition survey)", S))
            el.append(KeepTogether(blk))
        el.append(P("5 · What to do", H3))
        for m in r["measures"]:
            el.append(KeepTogether([Paragraph(f"<b>{_html.escape(m['title'])}</b> — {m['rating']} · {m['when']}", B)] +
                                   [P("• " + x) for x in m["actions"]] + [Spacer(1, 4)]))
        if r.get("data_gaps"):
            el.append(P("Data gaps in this run: " + "; ".join(r["data_gaps"]), S))
    el += [Spacer(1, 10), P("Basis: " + " ".join(rep["assets"][0]["basis"]), S),
           P("Sources: " + "; ".join(dict.fromkeys(s for r in rep["assets"] for s in r["sources"])), S)]
    doc.build(el, onFirstPage=footer, onLaterPages=footer)
    return buf.getvalue()
