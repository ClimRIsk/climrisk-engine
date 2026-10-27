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
_BUILD = re.compile(r"\b(build|building|construct\w*|develop\w*|put up|erect|tower|high[\s\-]?rise|apartment|hotel|block)\b", re.I)
_ASK = re.compile(r"\b(cost|costs|price|budget|expens\w*|feasib\w*|can i|could i|may i|is it (?:possible|allowed|safe)|allowed|permit\w*|viable)\b", re.I)
_NOT = re.compile(r"\b(3d|twin|viewer|protect\w*|scenario|simulate|flood model)\b", re.I)

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
    "Ground conditions: the engine has no soil or groundwater data for this point. A 10-storey building is normally on piles or a raft, and the foundation type is often the largest single swing in cost. A site investigation is needed before any budget is credible.",
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


def matches(text: str) -> bool:
    """A development-feasibility question: a building of some height or size, plus a can-I / cost question. Tool words (twin, protect) keep their own route."""
    t = text or ""
    return bool(_BUILD.search(t) and _ASK.search(t) and not _NOT.search(t) and (storeys(t) or re.search(r"\b(tower|high[\s\-]?rise|apartment|hotel)\b", t, re.I)))


def analyse(report: dict, n: Optional[int]) -> dict:
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
    height = f"about {round(n * 3.2)} m" if n else None
    return {"storeys": n, "height": height, "rows": rows, "drivers": drivers, "not_assessed": notrated,
            "overall": report.get("overall_rating"), "cautions": report.get("cautions") or []}


def reply_text(a: dict, place: str) -> str:
    n = a["storeys"]
    head = f"A {n}-storey building ({a['height']} tall) at {place}: " if n else f"A tall building at {place}: "
    if a["drivers"]:
        names = ", ".join(f"{d['hazard'].lower()} ({d['rating'].lower()})" for d in a["drivers"][:4])
        body = f"the site screening flags {names}. Those are the hazard-driven items that would add to design and cost; the rest rate low or negligible."
    else:
        body = "the site screening shows nothing at moderate or above, so hazards add little to the cost. Ground conditions, planning and local prices will dominate, and the engine does not hold those."
    return head + body + " I can't put a price on it: the engine holds no construction-cost, land or soil data, and I won't invent a figure. What follows is what would move the cost, and what to get before you commit."
