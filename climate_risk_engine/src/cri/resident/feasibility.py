"""
cri/resident/feasibility.py — "can I build a N-storey building here, and what would it cost, in broad terms?"

What the engine can say: which site hazards bite a building of that height, how each one changes the design and therefore the cost,
and what the screening could not assess. What it cannot say: a price. It holds no construction-cost, land, planning or soil data,
and it does not invent any. Every row is traced to the hazard rating in the physical report for that exact point (null stays null).
"""
from __future__ import annotations

import re
from typing import Optional

_WORDS = {"two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "fifteen": 15,
          "twenty": 20, "thirty": 30}
_NUM = r"(\d{1,3}|" + "|".join(_WORDS) + r")"
_STOREYS = re.compile(rf"\b{_NUM}[\s\-]*(?:stor(?:e)?y|stor(?:e)?ys|stories|storeys|floors?|levels?)\b", re.I)
_USES = [("residential", r"apartments?|flats?|condo\w*|residential|dwellings?|student (?:hostel|housing)|hostel|housing|block of flats"), ("hotel", r"hotels?|resort"), ("office", r"offices?|office tower|commercial"),
         ("warehouse", r"warehouses?|distribution cent(?:re|er)|logistics"), ("factory", r"factor(?:y|ies)|plant|mill|manufacturing"), ("data centre", r"data ?cent(?:re|er)s?"),
         ("hospital", r"hospitals?|clinic"), ("school", r"schools?|universit\w+|college"), ("mall", r"malls?|shopping (?:centre|center)|retail"), ("car park", r"car ?park|parking"),
         ("house", r"houses?|homes?|villa|bungalow|cottage")]
_BNOUN = re.compile(r"\b(building|buildings|tower|high[\s\-]?rise|skyscraper|apartments?|hotel|block|warehouse|factory|plant|data ?cent(?:re|er)|hospital|school|mall|car ?park|house|home|hostel|office|development|distribution cent(?:re|er)|villa|bungalow|clinic|university|resort|flats?|shopping cent(?:re|er)|condo\w*)\b", re.I)
_BVERB = re.compile(r"\b(build|construct\w*|develop\w*|put up|erect|set up)\b", re.I)
_ASK = re.compile(r"\b(cost|costs|price|budget|expens\w*|feasib\w*|can i|can we|i can|we can|could i|could we|i could|may i|should i|do you think|is it ok|would it be|is it (?:possible|allowed|safe|viable|ok|okay)|allowed|permit\w*|viable|how much|what would|what will|is it safe|risky\s+(?:is it\s+)?to\s+(?:build|construct|put up|develop|erect)|safe\s+to\s+(?:build|construct|put up|develop|erect))\b", re.I)
_NOT = re.compile(r"\b(3d|twin|viewer|protect\w*|scenario|simulate|flood model|report)\b", re.I)
_HEIGHT = re.compile(r"(\d{2,3}(?:\.\d+)?)\s*(?:m|metres?|meters?)\s*(?:high|tall|height)?\b(?!\s*(?:2|²|sq|square))", re.I)
_AREA_M2 = re.compile(r"(\d[\d,\.]*)\s*(?:sq\.?\s?m|square\s+met(?:re|er)s?|m2|m²)\b", re.I)
_AREA_FT2 = re.compile(r"(\d[\d,\.]*)\s*(?:sq\.?\s?ft|square\s+feet|square\s+foot|ft2|ft²)\b", re.I)
STOREY_M = 3.2   # typical floor-to-floor height used ONLY to turn a storey count into an approximate height; it is a rule of thumb, not a code value

LEVEL = {"NEGLIGIBLE": 0, "LOW": 1, "MODERATE": 2, "HIGH": 3, "CRITICAL": 4, "VERY HIGH": 4}

# hazard key -> (what it changes in a building of this kind, cost direction when the rating is MODERATE or above)
DRIVERS = {
    "river_flood": ("Raise the floor level or accept flood-resilient ground floors; no basement or plant below the flood level; flood-proof entrances and electrics", "adds cost; may also limit use of the lowest floor"),
    "coastal_flood": ("Raised platform and resilient ground floor; surge and wave loading on the lowest floors; sea-level allowance over the design life", "adds cost; strongest effect on foundations and ground floor"),
    "surface_water": ("Site drainage, attenuation and floor-level control", "adds modest cost"),
    "cyclone": ("Design wind speed for the structure and cladding; wind load grows with height, so a 10-storey frame is governed by wind sooner than a low building", "adds cost to frame, façade and roof fixings"),
    "tornado": ("Envelope and roof resilience; usually managed through shelter areas and robust façade rather than full design for tornado loads", "adds modest cost"),
    "hail": ("Roof and façade material choice, glazing and solar-panel resilience", "adds modest cost; also raises insurance cost"),
    "earthquake": ("Seismic design category: structural system, ductile detailing, deeper foundations; the effect grows with height", "adds cost to structure and foundations"),
    "slope": ("Geotechnical investigation, retaining structures, slope stabilisation, foundation type", "adds cost, can be large and uncertain"),
    "wildfire": ("Non-combustible façade and roof, defensible space, ember-resistant vents", "adds modest cost"),
    "heat": ("Higher cooling plant capacity and shading; overheating control for occupants", "adds operating cost over the life of the building"),
    "water": ("Water-efficient fixtures, storage or alternative supply for a building this size", "adds modest cost; operating risk"),
    "permafrost": ("Thermal foundations and ground-ice survey", "adds large cost"),
    "freeze_thaw": ("Frost-resistant materials and foundation depth", "adds modest cost"),
}

ALWAYS = [
    "Ground conditions: the engine has no soil or groundwater data for this point. A multi-storey building is normally on piles or a raft, and the foundation type is often the largest single swing in cost. A site investigation is needed before any budget is credible.",
    "Planning and zoning: height limits, permitted use, density, airport or heritage restrictions and flood-planning rules are local and are not in the engine. Ask the local planning authority before spending on design.",
    "Fire and access rules: buildings above roughly 18–25 m (about 6–8 storeys) are commonly treated as high-rise for fire safety, with extra stairs, lifts and sprinklers. The threshold is set locally, so check it for this jurisdiction.",
    "Land, labour and material prices: these vary by country and by year. The engine does not hold them, so it does not quote a figure per square metre.",
]
PRICE_ROUTE = ("For a price, get a cost per square metre for your city from a quantity surveyor, or from a published construction cost survey for the market "
               "(Turner & Townsend, Rider Levett Bucknall and Cushman & Wakefield each publish city-level surveys), and add the site-specific items above.")


def storeys(text: str) -> Optional[int]:
    m = _STOREYS.search(text or "")
    if not m:
        return None
    g = m.group(1).lower()
    return int(g) if g.isdigit() else _WORDS.get(g)


def use_of(text: str) -> Optional[str]:
    for k, rx in _USES:
        if re.search(rf"\b(?:{rx})\b", text or "", re.I):
            return k
    return None


def parse(text: str) -> dict:
    """What the question says about the building: storeys, height (m), floor area (m2, with ft2 converted), use. Only what is stated."""
    t = text or ""
    out: dict = {}
    n = storeys(t)
    if n:
        out["storeys"] = n
    m = _AREA_M2.search(t)
    if m:
        out["area_m2"] = float(m.group(1).replace(",", ""))
        t = t[:m.start()] + " ; " + t[m.end():]
    else:
        m = _AREA_FT2.search(t)
        if m:
            out["area_ft2"] = float(m.group(1).replace(",", ""))
            out["area_m2"] = round(out["area_ft2"] * 0.092903, 1)
            t = t[:m.start()] + " ; " + t[m.end():]
    m = _HEIGHT.search(t)
    if m and not n:
        out["height_m"] = float(m.group(1))
    u = use_of(text)
    if u:
        out["use"] = u
    return out


def matches(text: str) -> bool:
    """A development-feasibility question: something to build (with a height, an area or a use) plus a can-I / cost / safe question. Tool words (twin, protect, report) keep their own route."""
    t = text or ""
    _CAN = re.search(r"\b(can|could|should|may)\s+(?:i|we)\s+(?:build|construct|develop|put up|erect)\b|\bis it (?:feasible|possible|allowed|viable|safe)\s+to\s+(?:build|construct|develop|put up|erect)\b", t, re.I)
    if not (((_BNOUN.search(t) and _ASK.search(t)) or _CAN)) or _NOT.search(t):
        return False
    if re.search(r"\b(ship|vessel|tanker|voyage)\b", t, re.I):
        return False
    p = parse(t)
    return bool(p) or bool(_BVERB.search(t) and not re.search(r"\b(software|app|model|website|code|team|business|brand)\b", t, re.I)) or bool(re.search(r"\b(tower|high[\s\-]?rise|skyscraper|apartment|hotel)\b", t, re.I))


def analyse(report: dict, n: Optional[int], spec: Optional[dict] = None) -> dict:
    """Turn a physical report (zones with ratings) into the hazard-driven cost and design list for an n-storey building."""
    rows, notrated = [], []
    for z in report.get("zones", []):
        key, r = z.get("key"), z.get("rating")
        if key not in DRIVERS:
            continue
        if r is None:
            notrated.append(z.get("title") or key)
            continue
        lv = LEVEL.get(str(r).upper(), 0)
        what, cost = DRIVERS[key]
        rows.append({"hazard": z.get("title") or key, "rating": r, "level": lv, "design": what if lv >= 2 else "No hazard-specific design item at this rating",
                     "cost": cost if lv >= 2 else "no extra cost expected from this hazard", "basis": z.get("rating_basis") or ""})
    rows.sort(key=lambda x: -x["level"])
    drivers = [x for x in rows if x["level"] >= 2]
    spec = spec or {}
    height = f"about {round(n * STOREY_M)} m at {STOREY_M} m per floor, a rule of thumb" if n else (f"{spec['height_m']:g} m" if spec.get("height_m") else None)
    return {"spec": spec, "use_notes": USE_NOTES.get(spec.get("use"), []), "storeys": n, "height": height, "rows": rows, "drivers": drivers, "not_assessed": notrated,
            "overall": report.get("overall_rating"), "cautions": report.get("cautions") or []}


USE_NOTES = {  # general design practice by building type; qualitative, not engine data
    "hospital": ["A hospital is critical infrastructure: it has to keep working in the event that causes damage, so flood level, wind, seismic category and backup power are set by the most severe event it must survive, not the typical one."],
    "data centre": ["A data centre depends on power, cooling water and connectivity: check grid reliability, heat and water-stress exposure, and that plant is not placed below the flood level."],
    "warehouse": ["A warehouse has a large roof and footprint: roof uplift in high wind and surface-water drainage over a large paved area usually matter more than for a compact building. Stock held at floor level is exposed to flooding."],
    "factory": ["A factory carries process plant and stock: the cost of flooding or heat is often in downtime and equipment, not only the structure."],
    "house": ["A low-rise house is governed by flood level, roof wind resistance and ground conditions rather than frame design; insurance availability can matter as much as build cost."],
    "car park": ["A car park with basement levels is the building type most exposed to flooding of the lowest level."],
    "hotel": ["A hotel is often on the coast or at a destination site: coastal flood, cyclone and sea-level allowance are usually the first things to check."],
    "school": ["A school is a public building with occupants who cannot self-evacuate: shelter, access routes and heat in classrooms are design drivers."],
}


def reply_text(a: dict, place: str) -> str:
    n, sp = a["storeys"], a.get("spec") or {}
    use = sp.get("use")
    size = (f", {sp['area_m2']:,.0f} m2" if sp.get("area_m2") else "")
    if n:
        head = f"A {n}-storey {use or 'building'}{size} ({a['height']}) at {place}: "
    elif sp.get("height_m"):
        head = f"A {sp['height_m']:g} m {use or 'building'}{size} at {place}: "
    elif use or sp.get("area_m2"):
        head = f"A {use or 'building'}{size} at {place}: "
    else:
        head = f"A tall building at {place}: "
    if a["drivers"]:
        names = ", ".join(f"{d['hazard'].lower()} ({d['rating'].lower()})" for d in a["drivers"][:4])
        body = f"the site screening flags {names}. Those are the hazard-driven items that would add to design and cost; the rest rate low or negligible."
    else:
        body = "the site screening shows nothing at moderate or above, so hazards add little to the cost. Ground conditions, planning and local prices will dominate, and the engine does not hold those."
    return head + body + " I can't put a price on it: the engine holds no construction-cost, land or soil data, and I won't invent a figure. What follows is what would move the cost, and what to get before you commit."
