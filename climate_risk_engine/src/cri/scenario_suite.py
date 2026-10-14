"""
Scenario suite: realistic client questions run end to end through the engine, with checks that do not depend on a number being right.

A scenario is a client in Europe, the US or Canada asking about assets somewhere in the world (the pattern the engine's users have). Each one carries the sector,
the sites, the question, and where a public reference exists, the real event that the answer should be consistent with.

Two ways to run it:
  offline   `check_offline(s)`   deterministic; hazards are supplied as 'no map, no cyclone record' (what the engine meets in most of the world) or as a stated cyclone rate where
                                 one is documented. It tests the CONTRACT: the number never silently omits a peril, nothing relevant is dropped, nulls are explained, scope is stated.
  live      scripts/run_scenarios.py  uses real data sources (ERA5 and others), records what each hazard returned at each site, and writes docs/SCENARIO_REPORT.md.

A check that fails is a finding, not a test to loosen. The report lists them as gaps.
"""
from __future__ import annotations

from typing import Optional

from . import perilscope as P

# (id, client, title, sector, sites[(name, lat, lon, tiv_usd_m)], question, reference)
# reference: public event the engine's answer must be consistent with, or None. 'src' says where the figure comes from (named in docs/SCENARIO_RESEARCH.md).
_RAW = [
    ("ow_northsea", "Danish/Dutch wind developer", "North Sea offshore wind premium", "offshore wind farm", [("Horns Rev (DK)", 55.68, 7.63, 1200), ("Borssele (NL)", 51.70, 3.00, 1500)],
     "Set the insurance premium for these offshore wind farms and say what the main issues are.", {"event": "Offshore wind claims by value: cables 53 %, turbines 20 %", "src": "Allianz Commercial 2014-2020"}),
    ("ow_baltic", "German wind developer", "Baltic offshore wind: sea ice and icing", "offshore wind farm", [("Arkona (DE)", 54.78, 14.12, 1100)],
     "What are the ice risks to an offshore wind farm in the Baltic?", {"event": "Baltic sea ice affects foundations and vessel access", "src": "Donnelly et al. 2025; Baltic design-ice studies"}),
    ("ow_taiwan", "European offshore developer", "Taiwan Strait offshore wind: typhoon", "offshore wind farm", [("Changhua (TW)", 24.05, 120.15, 2000)],
     "What would a typhoon do to this wind farm and what premium follows?", {"event": "GCube: nat-cat exposure rising in Taiwan offshore wind", "src": "GCube 2019 and 2023"}),
    ("solar_texas", "US solar owner", "Texas solar hail", "solar farm", [("Midland solar (TX)", 31.95, -102.10, 450)],
     "What is the hail risk to this solar farm and what do we reserve?", {"event": "Midway, Texas 2019: 58 % of 685,000 modules damaged, claim about 70 to 75 million USD", "src": "Renewable Energy World, Business Insurance, GCube"}),
    ("refinery_gulf", "US refiner", "Gulf Coast refinery: hurricane, flood, freeze", "refinery", [("Houston Ship Channel", 29.72, -95.00, 3500), ("Beaumont", 30.00, -94.07, 2800)],
     "Price property and BI for these refineries.", {"event": "Winter Storm Uri 2021: Dow Chemical downstream insured loss 600 to 650 million USD; site damage under 189 million USD but cumulative BI large", "src": "Insurance Insider; Marsh"}),
    ("oilsands_wildfire", "Canadian operator", "Oil sands: wildfire and permafrost", "oil sands facility", [("Fort McMurray", 56.73, -111.38, 4000)],
     "Wildfire and ground risks for this site.", {"event": "Fort McMurray wildfire 2016: 3.58 billion CAD insured", "src": "Insurance Bureau of Canada via Global News"}),
    ("mine_arctic", "Canadian miner", "Arctic mine: permafrost, cold, flood", "diamond mine", [("Diavik (NT)", 64.50, -110.30, 1500)],
     "What physical risks does this northern mine face?", {"event": "Thermosyphons keep arsenic frozen at Canadian mines; thaw threatens up to half of Arctic infrastructure by 2050", "src": "Arctic Today, Reuters 2022"}),
    ("realestate_calgary", "Canadian REIT", "Calgary commercial property: hail", "commercial real estate", [("Calgary office", 51.05, -114.07, 300)],
     "Which perils drive loss for this Calgary property?", {"event": "Calgary hail 13 June 2020: almost 1.2 billion CAD insured", "src": "CatIQ via Canadian Underwriter"}),
    ("grid_ice_ontario", "Canadian utility", "Eastern Canada power grid: ice storm", "electricity transmission grid", [("Ottawa corridor", 45.40, -75.70, 800)],
     "What is the ice storm exposure of this transmission corridor?", {"event": "Ice storm Jan 1998: about 1.5 to 2.2 billion CAD insured, nearly a million without power", "src": "IBC; Global News"}),
    ("re_cologne", "German insurer", "Cologne residential: windstorm", "residential property", [("Cologne portfolio", 50.94, 6.96, 500)],
     "Windstorm risk for this residential book.", {"event": "Kyrill 2007: 3.65 billion EUR; Lothar 1999: 5.77 billion EUR (PERILS, original insured property loss)", "src": "PERILS via Strategic Risk"}),
    ("agri_lombardy", "Italian agri insurer", "Lombardy crops: hail", "agriculture crop farm", [("Lombardy farms", 45.50, 9.90, 200)],
     "Hail risk to these crops.", {"event": "Northern Italy July 2023 hail: about 6 billion EUR damage across all property", "src": "2026 European hail review (pv magazine summary)"}),
    ("garment_dhaka", "European retailer", "Bangladesh garment supplier: flood, heat", "garment factory", [("Dhaka supplier", 23.81, 90.41, 60), ("Chittagong supplier", 22.36, 91.78, 45)],
     "What physical risks threaten our suppliers in Bangladesh?", {"event": "Aon: heat-hours lost in Asia more than doubled in three decades, worst in manufacturing", "src": "Aon 2026 Climate and Catastrophe Insight"}),
    ("electronics_thailand", "US electronics buyer", "Thailand electronics cluster: flood and supply chain", "electronics manufacturing", [("Ayutthaya plant", 14.35, 100.57, 400)],
     "Flood and contingent BI exposure at this cluster.", {"event": "Thailand floods 2011: insured loss 10 to 15 billion USD, mostly manufacturing and contingent BI; 40 to 45 % of world hard-disk output", "src": "Canadian Underwriter; AM Best; SCOR"}),
    ("smelter_uae", "European metals group", "UAE aluminium smelter", "aluminium smelter", [("Taweelah (AE)", 24.78, 54.73, 3000)],
     "Which physical risks matter for this smelter?", {"event": "UAE April 2024 floods: 254.8 mm in 24 h, insured loss estimated 1.5 to 2.8 billion USD", "src": "Guy Carpenter; S&P; MEIR"}),
    ("dubai_flood", "European insurer", "Dubai commercial property: flash flood", "commercial real estate", [("Dubai Marina", 25.08, 55.14, 250)],
     "How likely is a repeat of April 2024 at this property?", {"event": "Dubai 142 mm in one day on 15 to 16 April 2024, heaviest since records began in 1949", "src": "Reuters via press; Guy Carpenter"}),
    ("refinery_jubail", "US operator", "Saudi refinery: heat and dust", "petrochemical plant", [("Jubail", 27.00, 49.65, 2500)],
     "Heat and dust risk at this petrochemical site.", None),
    ("port_mumbai", "European shipper", "Mumbai port: cyclone and surge", "port terminal", [("JNPT", 18.95, 72.95, 900)],
     "Cyclone and surge exposure for this port terminal.", {"event": "Swiss Re Asia 2025: ports concentrate value in typhoon and surge zones", "src": "Swiss Re Institute"}),
    ("mine_sulawesi", "European battery maker", "Indonesian nickel mine: landslide, rain, earthquake", "nickel mine", [("Sulawesi", -2.60, 121.40, 700)],
     "Physical risks for this nickel supplier.", None),
    ("mine_chile", "Canadian miner", "Atacama copper mine: earthquake and water", "copper mine", [("Atacama", -23.50, -68.90, 2200)],
     "Earthquake and water risk for this copper mine.", None),
    ("hydro_peru", "Canadian utility", "Andean hydropower: glacier lake outburst and drought", "hydropower dam", [("Canon del Pato", -8.80, -77.80, 600)],
     "Risks to this hydro plant.", None),
    ("hydro_brazil", "European utility", "Brazil hydropower: drought", "hydropower dam", [("Itaipu", -25.41, -54.59, 5000)],
     "Drought risk to generation.", None),
    ("dc_virginia", "US cloud operator", "Northern Virginia data centre: heat, water, power", "data centre", [("Ashburn", 39.04, -77.49, 1200)],
     "Physical risks to this data centre campus.", {"event": "FM Resilience Index 2026: water stress and fire risk link for data-centre cooling and suppression", "src": "FM"}),
    ("dc_singapore", "US cloud operator", "Singapore data centre: heat and humidity", "data centre", [("Tuas", 1.30, 103.64, 900)],
     "Heat and flood risk for a data centre in Singapore.", None),
    ("port_rotterdam", "Dutch port operator", "Rotterdam port: storm surge and sea level", "port terminal", [("Maasvlakte", 51.96, 4.04, 1500)],
     "Surge and sea-level exposure.", None),
    ("mfg_cebu", "Japanese-owned US supplier", "Philippines factory: typhoon", "electronics manufacturing", [("Cebu plant", 10.32, 123.90, 150)],
     "Typhoon and flood loss at this plant.", None),
    ("fab_taiwan", "US chip designer", "Taiwan semiconductor fab: earthquake and water", "semiconductor fab", [("Hsinchu", 24.78, 121.00, 3000)],
     "Earthquake and drought risk at the fab.", {"event": "Tohoku 2011: contingent BI the largest uncertainty in model loss", "src": "RMS; Swiss Re presentation"}),
    ("solar_queensland", "European fund", "Queensland solar: hail, cyclone, flood", "solar farm", [("Emerald solar", -23.40, 150.50, 300)],
     "Hail and cyclone exposure for this solar farm.", None),
    ("pipeline_north", "Canadian pipeline operator", "Northern pipeline: permafrost and slope", "pipeline", [("Norman Wells corridor", 65.28, -126.83, 700)],
     "Ground risks along this pipeline.", {"event": "9,500 km of Arctic pipelines at risk from thaw (satellite estimate)", "src": "Arctic Today, Reuters 2022"}),
    ("re_la", "US insurer", "Los Angeles residential: wildfire", "residential property", [("Pacific Palisades", 34.05, -118.55, 600)],
     "Wildfire loss exposure for this book.", {"event": "Palisades and Eaton fires January 2025: about 40 billion USD insured", "src": "Swiss Re sigma 1/2026; Munich Re"}),
    ("group_multi", "European industrial group", "Global group portfolio across six regions", "manufacturing plant", [("Rotterdam", 51.92, 4.48, 400), ("Houston", 29.76, -95.37, 500), ("Calgary", 51.05, -114.07, 300),
                                                                                             ("Dhaka", 23.81, 90.41, 120), ("Dubai", 25.20, 55.27, 350), ("Sao Paulo", -23.55, -46.63, 250)],
     "Price the portfolio and tell me what the number leaves out.", None),
]


def _canned_hazard(lat: float, lon: float) -> dict:
    """What the engine meets away from mapped places: a cyclone record where the basin has one, and no flood map. Used for offline contract checks, not for magnitudes."""
    tc = P._tc_possible(lat, lon) and abs(lat) < 35
    wind = {"lam": 0.5, "sigma": 8.0, "rp100_ms": 55.0} if tc else {"lam": 0.0, "sigma": None, "rp100_ms": None}
    return {"wind": wind, "river": None, "coast": None, "cont": "asia" if 0 < lon < 180 and lat > -10 else "europe", "tc_region": "ROW", "iso3": None, "notes": []}


def scenarios() -> list:
    out = []
    for sid, client, title, sector, sites, q, ref in _RAW:
        out.append({"id": sid, "client": client, "title": title, "sector": sector, "sites": [{"name": n, "lat": la, "lon": lo, "tiv_usd": m * 1e6} for n, la, lo, m in sites],
                    "question": q, "reference": ref})
    return out


def get(sid: str) -> dict:
    return next(s for s in scenarios() if s["id"] == sid)


def _site_inputs(s: dict, with_hazard: bool = True) -> list:
    out = []
    for x in s["sites"]:
        d = {**x, "sector": s["sector"]}
        if s["sector"].startswith("offshore"):
            d["offshore"] = {"turbines": 60, "hub_height_m": 105}
        if with_hazard:
            d["hazard"] = _canned_hazard(x["lat"], x["lon"]) if not s["sector"].startswith("offshore") else {**_canned_hazard(x["lat"], x["lon"]), "wind": {"lam": 0.0, "sigma": None, "rp100_ms": None}}
        out.append(d)
    return out


def check_offline(s: dict, n_years: int = 400) -> dict:
    """Run the insurer module on the scenario with canned hazards and test the contract. Returns {'passed': bool, 'findings': [...], 'result': run-result}."""
    from . import insurance as I
    res = I.run(_site_inputs(s), n_years=n_years)
    f: list = []
    prem = res["premium"]
    for r in res["sites"]:
        sc = r.get("peril_scope")
        if not sc:
            f.append({"check": "scope_present", "site": r["name"], "finding": "no peril_scope on the site row"})
            continue
        sm = sc["summary"]
        want = P.SECTOR[sm["sector"]]
        if sm["sector"] == "generic":
            f.append({"check": "sector_recognised", "site": r["name"], "finding": f"sector '{s['sector']}' fell back to the generic list: its own perils are not known to the engine"})
        listed = {p["peril"] for p in sc["perils"]}
        missing = [p for p, role, _ in want if role == "primary" and p not in listed and not (p == "tropical_cyclone" and not P._tc_possible(r["lat"], r["lon"]))]
        if missing:
            f.append({"check": "primary_perils_not_dropped", "site": r["name"], "finding": f"primary perils missing from scope: {missing}"})
        # a number is never presented with no statement of what it leaves out
        if r["technical_premium"] is not None and sm["primary_not_in_price"] and not prem.get("scope_note"):
            f.append({"check": "scope_note", "site": r["name"], "finding": "premium given without a scope note although primary perils are outside it"})
        # a peril that has no engine module must be flagged, not absent
        for p in sc["perils"]:
            if p["engine_status"] == "none" and p["in_price"]:
                f.append({"check": "no_module_but_in_price", "site": r["name"], "finding": f"{p['peril']} claims to be in the price but the engine has no module"})
        # no zero presented as a result where nothing was assessed
        if r["not_assessed"] and len(r["not_assessed"]) >= len(r["covers"]) and (r["technical_premium"] is not None or r["aal"] is not None):
            f.append({"check": "false_zero", "site": r["name"], "finding": "numbers shown although no covered peril was assessed"})
    # tropical-cyclone sites priced as offshore wind must withhold: a North Sea curve is not a typhoon curve
    if s["sector"].startswith("offshore"):
        for r in res["sites"]:
            if P._tc_possible(r["lat"], r["lon"]) and abs(r["lat"]) < 35 and r["technical_premium"] is not None:
                f.append({"check": "offshore_typhoon_site_priced", "site": r["name"], "finding": "a typhoon-exposed offshore site was priced with the North Sea storm path; typhoon loss for turbines is not modelled"})
    return {"id": s["id"], "passed": not f, "findings": f, "result": res}
