"""District landslide-exposure rank for an Indian site, from NRSC's Landslide Atlas of India (Feb 2023), Table 3.  (17 Nov 2026)

NRSC mapped about 80,000 landslides (1998-2022) and ranked 147 districts in 17 states and 2 UTs by their exposure to landslides in terms of socio-economic parameters
(population, households, livestock, roads), normalised by district area. 1 = most exposed.

What it is: an independent, observation-based district ranking from India's national space agency, useful as a check on our slope screening.
What it is not: a hazard measure or probability (it mixes where landslides are with what is there to be hit); district scale; and only 147 mountain districts were ranked. A district
that is not in the table was NOT assessed, which is not the same as safe. It is never shown as zero or as a low rank.
"""
from __future__ import annotations

import json
import re
import unicodedata
from functools import lru_cache
from pathlib import Path

from . import india_flood_atlas as _fa

_D = Path(__file__).resolve().parents[1] / "data"


def _norm(s: str) -> str:
    s = "".join(c for c in unicodedata.normalize("NFKD", s or "") if not unicodedata.combining(c))
    return re.sub(r"[^a-z]", "", re.sub(r"\(.*?\)", "", s.lower().replace("&", "and")))


@lru_cache(maxsize=1)
def _load():
    d = json.loads((_D / "india_landslide_exposure.json").read_text())
    by = {(_norm(r["matched_state"]), _norm(r["matched"])): r for r in d["districts"] if r["matched"]}
    # the geo file spells a matched district with its own name (e.g. "Leh(Ladakh)"), so key on that exact string as well
    by.update({(_norm(r["matched_state"]), r["matched"]): r for r in d["districts"] if r["matched"]})
    return d, by


def lookup(lat: float, lon: float) -> dict:
    """Atlas exposure rank for the district containing the point, or an explicit reason it has none. Never returns a rank or zero for 'not ranked'."""
    try:
        d, by = _load()
        hit = _fa.district_at(lat, lon)
    except Exception as e:                                           # noqa: BLE001
        return {"available": False, "reason": f"landslide table not readable ({type(e).__name__})"}
    base = {"available": True, "source": d["source"], "url": d["url"], "licence": d["licence"], "caveats": d["limits"]}
    if hit is None:
        return {**base, "in_india": False, "reason": "the point is not inside an Indian district polygon (2021 boundaries)"}
    out = {**base, "in_india": True, "district": hit["name"], "state": hit["state"]}
    r = by.get((_norm(hit["state"]), _norm(hit["name"]))) or by.get((_norm(hit["state"]), hit["name"]))
    if r is None:
        return {**out, "rank": None, "status": "not_ranked",
                "reason": "this district is not among the 147 NRSC ranked (or it was split from one that was); that means NOT ASSESSED by this table, not low and not safe"}
    n = len(d["districts"])
    return {**out, "status": "ranked", "atlas_district": r["district"], "rank": r["rank"], "of": n,
            "reading": f"NRSC ranks {r['district']} district ({r['state']}) {r['rank']} of {n} for exposure to landslides (1 = most exposed); this is exposure of people and assets, not a hazard probability."}
