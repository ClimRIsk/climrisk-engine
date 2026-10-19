"""
cri/geo_io.py — read a study area from KML, KMZ, GeoJSON or a bare list of coordinates. Standard library only.

  parse_study_area(data: bytes, filename: str) -> {"polygons": [[(lon, lat), ...]], "bbox": (w, s, e, n), "centroid": (lat, lon), "area_km2": float, "source": str}

Only polygon outer rings are kept (holes are ignored and said so in "notes"). Lines and points are used only if no polygon exists: a point becomes a centre
with no boundary. Coordinates are WGS84 longitude, latitude as KML and GeoJSON define them. Nothing is guessed: a file with no usable geometry raises ValueError.
"""
from __future__ import annotations

import io
import json
import math
import re
import zipfile
import xml.etree.ElementTree as ET


def _ring_area_km2(ring) -> float:
    lat0 = sum(p[1] for p in ring) / len(ring)
    kx, ky = 111.320 * math.cos(math.radians(lat0)), 110.574
    a = 0.0
    for (x1, y1), (x2, y2) in zip(ring, ring[1:] + ring[:1]):
        a += (x1 * kx) * (y2 * ky) - (x2 * kx) * (y1 * ky)
    return abs(a) / 2.0


def _coords(text: str) -> list:
    out = []
    for tok in re.split(r"\s+", (text or "").strip()):
        parts = tok.split(",")
        if len(parts) >= 2:
            try:
                lon, lat = float(parts[0]), float(parts[1])
            except ValueError:
                continue
            if -180 <= lon <= 180 and -90 <= lat <= 90:
                out.append((lon, lat))
    return out


def _from_kml(root) -> tuple:
    polys, pts, lines, notes = [], [], [], []
    for el in root.iter():
        tag = el.tag.split("}")[-1]
        if tag == "Polygon":
            outer = next((c for c in el.iter() if c.tag.split("}")[-1] == "outerBoundaryIs"), None)
            if outer is not None:
                ring = next((_coords(c.text) for c in outer.iter() if c.tag.split("}")[-1] == "coordinates"), [])
                if len(ring) >= 4:
                    polys.append(ring[:-1] if ring[0] == ring[-1] else ring)
            if any(c.tag.split("}")[-1] == "innerBoundaryIs" for c in el.iter()):
                notes.append("holes in the polygon were ignored")
        elif tag == "Point":
            c = next((_coords(c.text) for c in el.iter() if c.tag.split("}")[-1] == "coordinates"), [])
            if c:
                pts.append(c[0])
        elif tag == "LineString":
            c = next((_coords(c.text) for c in el.iter() if c.tag.split("}")[-1] == "coordinates"), [])
            if len(c) >= 2:
                lines.append(c)
    return polys, pts, lines, notes


def _from_geojson(obj) -> tuple:
    polys, pts, lines, notes = [], [], [], []

    def geom(g):
        if not g:
            return
        t, c = g.get("type"), g.get("coordinates")
        if t == "Polygon" and c:
            ring = [tuple(p[:2]) for p in c[0]]
            if len(ring) >= 4:
                polys.append(ring[:-1] if ring[0] == ring[-1] else ring)
            if len(c) > 1:
                notes.append("holes in the polygon were ignored")
        elif t == "MultiPolygon" and c:
            for poly in c:
                geom({"type": "Polygon", "coordinates": poly})
        elif t == "Point" and c:
            pts.append(tuple(c[:2]))
        elif t == "LineString" and c:
            lines.append([tuple(p[:2]) for p in c])
        elif t == "GeometryCollection":
            for gg in g.get("geometries", []):
                geom(gg)

    t = obj.get("type")
    if t == "FeatureCollection":
        for f in obj.get("features", []):
            geom(f.get("geometry"))
    elif t == "Feature":
        geom(obj.get("geometry"))
    else:
        geom(obj)
    return polys, pts, lines, notes


def parse_study_area(data: bytes, filename: str = "") -> dict:
    name = (filename or "").lower()
    notes = []
    if name.endswith(".kmz") or data[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            kml = next((n for n in z.namelist() if n.lower().endswith(".kml")), None)
            if not kml:
                raise ValueError("the KMZ archive holds no .kml file")
            data = z.read(kml)
        name = "x.kml"
    text = data.decode("utf-8", errors="replace").lstrip("﻿").strip()
    if name.endswith((".json", ".geojson")) or text.startswith("{"):
        try:
            polys, pts, lines, notes = _from_geojson(json.loads(text))
        except json.JSONDecodeError as e:
            raise ValueError(f"not valid GeoJSON: {e}") from e
        src = "GeoJSON"
    else:
        try:
            root = ET.fromstring(text.encode("utf-8"))
        except ET.ParseError as e:
            raise ValueError(f"not valid KML: {e}") from e
        polys, pts, lines, notes = _from_kml(root)
        src = "KML"
    if not polys and lines:
        notes.append("no polygon found: the first line was closed into a boundary")
        ln = lines[0]
        polys = [ln if len(ln) >= 3 else ln]
        if len(polys[0]) < 3:
            polys = []
    if not polys and pts:
        lon, lat = pts[0]
        return {"polygons": [], "bbox": (lon, lat, lon, lat), "centroid": (lat, lon), "area_km2": 0.0, "source": src, "notes": notes + ["only a point was found: no boundary"]}
    if not polys:
        raise ValueError("no polygon, line or point with valid coordinates was found in the file")
    allp = [p for r in polys for p in r]
    w, e = min(p[0] for p in allp), max(p[0] for p in allp)
    s, n = min(p[1] for p in allp), max(p[1] for p in allp)
    big = max(polys, key=_ring_area_km2)
    cx = sum(p[0] for p in big) / len(big)
    cy = sum(p[1] for p in big) / len(big)
    return {"polygons": [[list(p) for p in r] for r in polys], "bbox": (w, s, e, n), "centroid": (cy, cx), "area_km2": round(sum(_ring_area_km2(r) for r in polys), 4), "source": src, "notes": notes}


def points_in_polygons(lons, lats, polygons):
    """Boolean mask: True where (lon, lat) falls inside any polygon."""
    import numpy as np
    from matplotlib.path import Path
    pts = np.column_stack([np.ravel(lons), np.ravel(lats)])
    m = np.zeros(len(pts), bool)
    for r in polygons:
        m |= Path(np.asarray(r)).contains_points(pts)
    return m.reshape(np.shape(lons))
