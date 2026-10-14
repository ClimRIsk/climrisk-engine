"""Recorded flood events near a coordinate, from the compact index built by scripts/flood_event_benchmark.py --stage index.

The index holds bounding boxes and dates only: England Environment Agency fluvial outlines (1700s onward, dense from the 1940s) and
Copernicus EMS mapped events (2023–2026). It is NOT a complete flood history of anywhere. `coverage` says which sources can speak for
the place, so that "no events listed" is never read as "never flooded".
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Optional

_P = Path(__file__).resolve().parents[1] / "data" / "flood_event_index.json"


@lru_cache(maxsize=1)
def _load() -> Optional[dict]:
    try:
        return json.loads(_P.read_text())
    except Exception:                                               # noqa: BLE001
        return None


def _in_england(lat: float, lon: float) -> bool:
    return 49.8 <= lat <= 55.9 and -6.5 <= lon <= 1.9


def near(lat: float, lon: float, km: float = 2.0, limit: int = 25) -> dict:
    d = _load()
    if not d:
        return {"available": False, "reason": "flood event index not shipped in this build", "events": []}
    pad_lat = km / 111.0
    pad_lon = km / (111.0 * max(0.2, abs(__import__("math").cos(__import__("math").radians(lat)))))
    hits = []
    for e in d["events"]:
        b = e["b"]
        if b[0] - pad_lon <= lon <= b[2] + pad_lon and b[1] - pad_lat <= lat <= b[3] + pad_lat:
            inside = b[0] <= lon <= b[2] and b[1] <= lat <= b[3]
            hits.append({"source": e["s"], "id": e["id"], "name": e["n"], "date": e["d"], "bbox_contains_point": inside})
    hits.sort(key=lambda h: h["date"], reverse=True)
    cov = []
    if _in_england(lat, lon):
        cov.append("England fluvial outlines (Environment Agency recorded flood outlines)")
    cov.append("Copernicus EMS mapped events 2023–2026 (only events that triggered a mapping activation)")
    return {"available": True, "radius_km": km, "n": len(hits), "events": hits[:limit], "coverage": cov,
            "note": "Bounding-box match against recorded outlines. No events listed does not mean no flooding: events outside these sources are not in the index.",
            "outside_dense_coverage": not _in_england(lat, lon)}
