"""Observed flood history for an Indian site from the NRSC/NDMA Flood Affected Area Atlas (1998-2022).  (17 Nov 2026)

The atlas is built by ISRO's National Remote Sensing Centre from multi-date optical and SAR satellite flood maps plus Central Water Commission gauge levels. It gives, per
district, the cumulative area that was ever mapped as flooded in 25 years. That is OBSERVED, not modelled, and it is independent of our terrain and rain models, which is why it is
used as a check on them and not as a score of the site.

What it is not: district scale (a site can be dry in a flooded district), cumulative extent only (no depth, no frequency, no return period), and it includes rain-water pooling in low
land. A district that is missing from a covered state means no flood was mapped there; it is reported as "not listed", never as zero. Atlas districts are older than the 2021
boundaries we use to find the district; some have been split since, and those stay unmatched (see `scripts/build_india_flood_atlas.py`).

The percentile is the share of the atlas's 439 districts with flood-affected area less than or equal to this one: a ranking of the data, not an invented severity scale.
"""
from __future__ import annotations

import gzip
import json
import re
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Optional

_D = Path(__file__).resolve().parents[1] / "data"
_CAVEATS = ["District scale only: a site can be dry in a flooded district and flooded in a district with little mapped flood.",
            "Cumulative extent over 1998-2022: no depth, frequency or return period.",
            "Includes rain-water accumulation in low land, not only river flooding.",
            "Only events NRSC mapped are in it. Its Tamil Nadu event list is 2007-2010, 2012 and 2020; the 2015 Chennai flood is not in it, so Chennai district reads low. A low value never clears a site."]


def _norm(s: str) -> str:
    s = "".join(c for c in unicodedata.normalize("NFKD", s or "") if not unicodedata.combining(c))
    return re.sub(r"[^a-z]", "", re.sub(r"\(.*?\)", "", s.lower()))


@lru_cache(maxsize=1)
def _load():
    atlas = json.loads((_D / "india_flood_atlas.json").read_text())
    geo = json.loads(gzip.open(_D / "india_districts_adm2.json.gz", "rt").read())["districts"]
    for g in geo:
        polys = g["geom"]["coordinates"] if g["geom"]["type"] == "MultiPolygon" else [g["geom"]["coordinates"]]
        xs = [p[0] for poly in polys for p in poly[0]]
        ys = [p[1] for poly in polys for p in poly[0]]
        g["_polys"], g["_bbox"] = polys, (min(xs), min(ys), max(xs), max(ys))
    rows = atlas["districts"]
    by = {(_norm(r["state"]) if r["state"] != "Jammu & Kashmir" else _norm("Jammu and Kashmir"), _norm(r["matched"])): r for r in rows if r["matched"]}
    states = {(_norm(r["state"]) if r["state"] != "Jammu & Kashmir" else _norm("Jammu and Kashmir")) for r in rows}
    has = sorted(r["ha"] for r in rows)
    return atlas, geo, by, states, has


def _in_ring(x: float, y: float, ring) -> bool:
    inside, j = False, len(ring) - 1
    for i in range(len(ring)):
        xi, yi, xj, yj = ring[i][0], ring[i][1], ring[j][0], ring[j][1]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi + 1e-300) + xi:
            inside = not inside
        j = i
    return inside


def _in_poly(x: float, y: float, poly) -> bool:
    return _in_ring(x, y, poly[0]) and not any(_in_ring(x, y, h) for h in poly[1:])


def lookup(lat: float, lon: float) -> dict:
    """Atlas record for the district containing the point, or an explicit reason it has none. Never returns zero for 'no record'."""
    try:
        atlas, geo, by, states, has = _load()
    except Exception as e:                                           # noqa: BLE001
        return {"available": False, "reason": f"atlas files not readable ({type(e).__name__})"}
    hit = next((g for g in geo if g["_bbox"][0] <= lon <= g["_bbox"][2] and g["_bbox"][1] <= lat <= g["_bbox"][3] and any(_in_poly(lon, lat, p) for p in g["_polys"])), None)
    base = {"available": True, "source": atlas["source"], "url": atlas["url"], "caveats": _CAVEATS + atlas.get("limits", []), "licence": atlas["licence"]}
    if hit is None:
        return {**base, "in_india": False, "reason": "the point is not inside an Indian district polygon (2021 boundaries)"}
    sn = _norm(hit["state"])
    out = {**base, "in_india": True, "district": hit["name"], "state": hit["state"]}
    if sn not in states:
        return {**out, "atlas_ha": None, "status": "state_not_covered", "reason": "this state is not in the atlas; no observed-flood statistic is available (this is not zero)"}
    r = by.get((sn, _norm(hit["name"])))
    if r is None:
        return {**out, "atlas_ha": None, "status": "not_matched",
                "reason": "no atlas row matches this district: it is either not listed (no mapped flood) or it was created after the atlas districts were drawn; not shown as zero"}
    ha = r["ha"]
    pct = sum(1 for v in has if v <= ha) / len(has)
    same = sorted((x["ha"] for x in atlas["districts"] if x["state"] == r["state"]), reverse=True)
    return {**out, "status": "listed", "atlas_district": r["district"], "atlas_ha": ha, "national_percentile": round(pct, 3), "state_rank": same.index(ha) + 1, "state_districts_listed": len(same),
            "reading": (f"Satellite flood maps 1998-2022 mapped {ha:,} ha as flooded in {r['district']} district ({r['state']}): "
                        f"rank {same.index(ha) + 1} of {len(same)} listed districts in the state, and above {round(100 * pct)} % of the {len(has)} districts in the atlas.")}
