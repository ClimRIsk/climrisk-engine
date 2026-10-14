"""
cri/analysis/site_sources.py — find what emits near a place WITHOUT being told.

A user who types a latitude and longitude should not have to say "there is a cement works next door". The engine looks, in
this order, and records where every source came from and how sure it is:

  1. REGISTER     the engine's own facility register (Climate TRACE-derived: cement, power, steel, chemicals, refining, mining …)
                  with name, process type and annual CO2e. Offline, global, instant. Most reliable for large plants.
  2. OPENSTREETMAP  industrial emitters tagged by mappers: non-renewable power plants/generators, works, chimneys, quarries,
                  industrial land. Incomplete; absence here is not absence on the ground.
  3. PLACE KNOWLEDGE  Wikipedia text of settlements within ~25 km (via Wikidata) searched for industries the town is known for
                  (fireworks and match making, cement, thermal power, brick kilns …). This finds clusters that no facility
                  database lists. It says "the town is known for X", never "X emits Y".
  4. SATELLITE HEAT  persistent thermal anomalies (NASA FIRMS VIIRS, several years) at industrial sites, when a FIRMS key is
                  configured. Not available otherwise, and the result says so.
  5. INFERENCE    rules from what was found, always labelled as inference and given low confidence: an integrated cement plant
                  usually runs a captive power plant on site.

The user can still declare or correct sources (they are used and labelled as declared), but nothing here needs them.
Each source: kind, name, lat, lon, distance_km, bearing_deg, provenance, confidence (high / medium / low / inferred), and
whatever size information exists (annual CO2e, capacity). Nothing is invented: if a size is unknown it is left empty and the
screen sweeps it.
"""
from __future__ import annotations

import json
import math
import os
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Callable, Optional

_UA = {"User-Agent": "climrisk-research/0.1 (site source discovery)"}
CO2_T_PER_T_CLINKER = 0.87        # integrated dry kiln: ~0.53 process + ~0.3 fuel + small electricity share; assumption, stated in output
KNOWN_FOR = [                      # place-knowledge patterns are deliberately strict: a word in passing ("cement roads") must not create a source
    ("fireworks", r"firework|firecracker|fire\s?cracker|match factor|matchbox|pyrotech"),
    ("cement", r"cement (?:industr|plant|factor|works|manufactur|cluster|hub)|(?:industr|plant|factor|works|manufactur)\w* of cement"),
    ("thermal", r"thermal power (?:plant|station)|coal[- ]fired|power station"),
    ("brick", r"brick kilns?|brick[- ]making|brickworks"),
    ("steel", r"steel (?:plant|mill)|foundr(?:y|ies)|smelter"),
    ("chemicals", r"chemical (?:plant|industr)|fertili[sz]er (?:plant|factor)|refiner(?:y|ies)"),
    ("textile_dyeing", r"dyeing units?|tanneries|tannery"),
]


def _hav(lat0: float, lon0: float, lat1: float, lon1: float) -> tuple:
    dy = (lat1 - lat0) * 111320.0
    dx = (lon1 - lon0) * 111320.0 * math.cos(math.radians(lat0))
    return math.hypot(dx, dy) / 1000.0, (math.degrees(math.atan2(dx, dy)) + 360.0) % 360.0


def _get(url: str, data: Optional[bytes] = None, timeout: float = 40.0):
    return json.loads(urllib.request.urlopen(urllib.request.Request(url, data=data, headers=_UA), timeout=timeout).read())


def _cache(name: str) -> Path:
    d = Path(os.environ.get("CRI_CACHE_DIR", Path.home() / ".climrisk" / "cache")) / "site_sources"
    d.mkdir(parents=True, exist_ok=True)
    return d / name


# ── 1. register ──────────────────────────────────────────────────────────────────

_REG_KIND = {"cement": "cement", "electricity-generation": "thermal", "iron-and-steel": "steel", "chemicals": "chemicals", "oil-and-gas-refining": "chemicals",
             "petrochemical-steam-cracking": "chemicals", "aluminum": "steel", "pulp-and-paper": "industrial", "coal-mining": "quarry", "iron-mining": "quarry",
             "copper-mining": "quarry", "bauxite-mining": "quarry"}


def from_register(lat: float, lon: float, radius_km: float = 25.0, near: Optional[Callable] = None) -> list:
    try:
        if near is None:
            from ..resident.research import assets_near as near
        r = near(lat, lon, radius_km, limit=200)
    except Exception:                                                  # noqa: BLE001
        return []
    out = []
    for a in r.get("top", []):
        kind = _REG_KIND.get(a["sector"], "industrial" if not str(a["sector"]).startswith("ghgrp") else "industrial")
        if kind == "thermal" and str(a.get("type")) in ("solar", "wind", "hydro", "nuclear", "geothermal"):
            continue
        d, b = _hav(lat, lon, a["lat"], a["lon"])
        out.append({"kind": kind, "name": a["name"], "lat": a["lat"], "lon": a["lon"], "distance_km": round(d, 2), "bearing_deg": round(b),
                    "provenance": "engine facility register (Climate TRACE-derived)", "confidence": "high", "process": a.get("type"),
                    "co2e_t_per_year": a.get("emissions_t")})
    return out


# ── 2. OpenStreetMap ─────────────────────────────────────────────────────────────

def from_osm(lat: float, lon: float, radius_km: float = 12.0, post: Optional[Callable] = None) -> tuple:
    q = (f'[out:json][timeout:25];(nwr(around:{int(radius_km * 1000)},{lat},{lon})["power"~"^(plant|generator)$"];'
         f'nwr(around:{int(radius_km * 1000)},{lat},{lon})["man_made"~"^(works|chimney)$"];nwr(around:{int(radius_km * 1000)},{lat},{lon})["industrial"];'
         f'nwr(around:{int(radius_km * 1000)},{lat},{lon})["landuse"="quarry"];);out center tags 300;')
    els = None
    for ep in ("https://overpass-api.de/api/interpreter", "https://overpass.kumi.systems/api/interpreter"):
        try:
            els = post(ep, q) if post else _get(ep, urllib.parse.urlencode({"data": q}).encode(), 35.0)["elements"]
            break
        except Exception:                                              # noqa: BLE001
            continue
    if els is None:
        return [], "OpenStreetMap could not be queried just now."
    out, skipped = [], 0
    for e in els:
        t = e.get("tags", {})
        src = (t.get("plant:source") or t.get("generator:source") or "").lower()
        if t.get("power") and src in ("solar", "wind", "hydro", "water", "battery"):
            skipped += 1
            continue
        c = e.get("center") or e
        if "lat" not in c:
            continue
        txt = " ".join(str(v) for v in t.values()).lower()
        kind = ("cement" if "cement" in txt else "fireworks" if re.search(r"firework|cracker|pyro|match", txt) else "quarry" if t.get("landuse") == "quarry"
                else "thermal" if t.get("power") else "industrial")
        d, b = _hav(lat, lon, c["lat"], c["lon"])
        out.append({"kind": kind, "name": t.get("name"), "lat": c["lat"], "lon": c["lon"], "distance_km": round(d, 2), "bearing_deg": round(b), "provenance": "OpenStreetMap",
                    "confidence": "medium", "tags": {k: v for k, v in t.items() if k in ("power", "plant:source", "plant:output:electricity", "industrial", "man_made", "operator", "product")}})
    return out, f"{len(out)} industrial emitters tagged in OpenStreetMap within {radius_km:.0f} km; {skipped} solar/wind/hydro generators ignored."


# ── 3. place knowledge ──────────────────────────────────────────────────────────

def from_places(lat: float, lon: float, radius_km: float = 25.0, get: Callable = _get) -> tuple:
    """Settlements with an English Wikipedia page within radius, whose intro text names an industry they are known for."""
    key = _cache("places_%.2f_%.2f.json" % (lat, lon))
    if key.exists() and time.time() - key.stat().st_mtime < 90 * 86400:
        try:
            return json.loads(key.read_text())
        except Exception:                                              # noqa: BLE001
            pass
    q = ('SELECT ?i ?iLabel ?c ?a WHERE { SERVICE wikibase:around { ?i wdt:P625 ?c . bd:serviceParam wikibase:center "Point(%s %s)"^^geo:wktLiteral . '
         'bd:serviceParam wikibase:radius "%s" . } ?a schema:about ?i ; schema:isPartOf <https://en.wikipedia.org/> . '
         'SERVICE wikibase:label { bd:serviceParam wikibase:language "en". } } LIMIT 150') % (lon, lat, radius_km)
    try:
        rows = get("https://query.wikidata.org/sparql?" + urllib.parse.urlencode({"format": "json", "query": q}), timeout=60)["results"]["bindings"]
    except Exception as e:                                             # noqa: BLE001
        return [], f"Place knowledge (Wikidata) could not be read: {type(e).__name__}."
    items = []
    for b in rows:
        m = re.match(r"Point\(([-\d.]+) ([-\d.]+)\)", b["c"]["value"])
        if m:
            items.append((b["a"]["value"].rsplit("/", 1)[-1], b["iLabel"]["value"], float(m.group(2)), float(m.group(1))))
    out = []
    for i in range(0, len(items), 20):
        chunk = items[i:i + 20]
        try:
            pg = get("https://en.wikipedia.org/w/api.php?" + urllib.parse.urlencode(
                {"action": "query", "prop": "extracts", "exintro": 1, "explaintext": 1, "exlimit": 20, "titles": "|".join(urllib.parse.unquote(c[0]) for c in chunk), "redirects": 1, "format": "json"}))["query"]["pages"]
        except Exception:                                              # noqa: BLE001
            continue
        by_title = {urllib.parse.unquote(c[0]).replace("_", " "): c for c in chunk}
        for v in pg.values():
            tx = v.get("extract") or ""
            c = by_title.get(v.get("title", ""))
            if not c:
                continue
            for kind, rx in KNOWN_FOR:
                sent = next((s.strip() for s in re.split(r"(?<=[.!?])\s+", tx) if re.search(rx, s, re.I) and re.search(r"known|famous|hub|centre|center|produce|manufactur|industr|cluster|home to|major", s, re.I)), None)
                if sent:
                    d, bg = _hav(lat, lon, c[2], c[3])
                    out.append({"kind": "fireworks" if kind == "fireworks" else kind, "name": c[1] + " (settlement)", "lat": c[2], "lon": c[3], "distance_km": round(d, 2), "bearing_deg": round(bg),
                                "provenance": "Wikipedia: " + sent[:220], "confidence": "low", "cluster": True})
    note = f"{len(items)} settlements with a Wikipedia page within {radius_km:.0f} km read for the industries they are known for; {len(out)} industry mentions found."
    try:
        key.write_text(json.dumps([out, note]))
    except Exception:                                                  # noqa: BLE001
        pass
    return out, note


# ── 4. satellite heat ───────────────────────────────────────────────────────────

def from_firms(lat: float, lon: float, radius_km: float = 10.0) -> tuple:
    key = os.environ.get("NASA_FIRMS_MAP_KEY")
    if not key:
        return [], "Satellite thermal anomalies (NASA FIRMS) not used: no FIRMS key configured on this machine."
    try:
        d = radius_km / 111.0
        dl = d / math.cos(math.radians(lat))
        url = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{key}/VIIRS_SNPP_SP/{lon - dl:.3f},{lat - d:.3f},{lon + dl:.3f},{lat + d:.3f}/10"
        txt = urllib.request.urlopen(urllib.request.Request(url, headers=_UA), timeout=60).read().decode()
    except Exception as e:                                             # noqa: BLE001
        return [], f"NASA FIRMS could not be read: {type(e).__name__}."
    rows = [r.split(",") for r in txt.strip().splitlines()]
    if len(rows) < 2:
        return [], "NASA FIRMS returned no detections."
    h = rows[0]
    li, lo, dt_i, frp = h.index("latitude"), h.index("longitude"), h.index("acq_date"), h.index("frp")
    cells: dict = {}
    for r in rows[1:]:
        k = (round(float(r[li]), 3), round(float(r[lo]), 3))
        cells.setdefault(k, set()).add(r[dt_i][:7])
    out = []
    for (la, lo_), months in cells.items():
        if len(months) >= 18:                                          # same ~375 m pixel hot in 18+ distinct months: a fixed industrial heat source, not a fire
            dkm, b = _hav(lat, lon, la, lo_)
            out.append({"kind": "industrial_heat", "name": "persistent satellite heat source", "lat": la, "lon": lo_, "distance_km": round(dkm, 2), "bearing_deg": round(b),
                        "provenance": f"NASA FIRMS VIIRS: hot in {len(months)} distinct months", "confidence": "medium"})
    return out, f"NASA FIRMS: {len(out)} persistent heat sources (hot in 18+ months) within {radius_km:.0f} km."


# ── 5. inference ────────────────────────────────────────────────────────────────

def infer(found: list) -> list:
    out = []
    has_power = any(s["kind"] == "thermal" for s in found)
    for s in found:
        if s["kind"] == "cement" and not has_power and s.get("process") in ("integrated dry", "integrated wet", None) and s["distance_km"] <= 8:
            out.append({"kind": "thermal", "name": f"captive power plant at {s['name']} (inferred)", "lat": s["lat"], "lon": s["lon"], "distance_km": s["distance_km"], "bearing_deg": s["bearing_deg"],
                        "provenance": "inference: integrated cement plants commonly run a captive power plant on site; capacity unknown", "confidence": "inferred", "assumptions": {"mw": 15.0}})
            break
    return out


# ── cascade ──────────────────────────────────────────────────────────────────────

def _dedupe(srcs: list) -> list:
    rank = {"high": 0, "medium": 1, "inferred": 2, "low": 3}
    out: list = []
    for s in sorted(srcs, key=lambda s: (rank.get(s["confidence"], 4), s["distance_km"])):
        if any(o["kind"] == s["kind"] and _hav(o["lat"], o["lon"], s["lat"], s["lon"])[0] < 2.0 and not s.get("cluster") and not o.get("cluster") for o in out):
            continue
        out.append(s)
    return sorted(out, key=lambda s: s["distance_km"])


def discover(lat: float, lon: float, radius_km: float = 25.0, osm: bool = True, places: bool = True, register: Optional[Callable] = None,
             osm_post: Optional[Callable] = None, get: Callable = _get) -> dict:
    log: list = []
    reg = from_register(lat, lon, radius_km, near=register)
    log.append(f"Facility register: {len(reg)} facilities within {radius_km:.0f} km" + (": " + "; ".join(f"{s['name']} ({s['kind']}, {s['distance_km']} km)" for s in reg[:4]) if reg else "."))
    found = list(reg)
    if osm:
        o, note = from_osm(lat, lon, min(radius_km, 12.0), osm_post)
        log.append(note)
        found += o
    if places:
        p, note = from_places(lat, lon, radius_km, get)
        log.append(note)
        found += p
    f, note = from_firms(lat, lon)
    log.append(note)
    found += f
    found = _dedupe(found)
    inf = infer(found)
    if inf:
        log.append("Inference: " + inf[0]["provenance"] + ".")
    return {"sources": _dedupe(found + inf), "log": log, "radius_km": radius_km}
