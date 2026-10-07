"""
cri/asset_twin.py — the Asset Twin: a 3D model of a real site, with a hazard-replay agent.

What it builds
  1. the real layout     OpenStreetMap building / tank / silo / chimney / substation footprints inside the site radius
                         (Overpass API), with heights from OSM tags where present and from the sector kind otherwise.
                         Where OSM has no usable footprints, an ILLUSTRATIVE parametric layout for the sector is used
                         and flagged as such. Every component carries its source: osm | parametric | user.
  2. the ground          AWS terrain tiles (~9 m) at site scale.
  3. the hazards         river and coastal flood depths at the site (JRC / Deltares maps, by return period and for 2050),
                         cyclone design winds (IBTrACS + Holland model) and every storm since 1980 that gave tropical-storm
                         force wind at the site, with approach direction. Radar-observed flooding (Sentinel-1) and
                         user-entered incident records are added to the timeline.
  4. the physics         flood: water level = site ground reference + mapped site-mean depth, projected over the 10 m
                         terrain (level-pool; NOT a hydraulic simulation). Wind: 0.5·ρ·v² with a power-law height profile
                         on the 10 m open-terrain wind. Component states are threshold bands against stated heights and
                         design winds, not fragility curves.
  5. protection          raise equipment, raise plinth, perimeter flood barrier, wind retrofit to a design speed. Each is
                         applied by re-running the same evaluation, so the "after" picture is computed, not drawn.
  6. the agent           a deterministic pipeline (observe → quantify → diagnose → propose → verify) that writes a step
                         transcript and findings of the form problem → cause → protection → verified residual.

No money figures anywhere. Component heights, equipment levels and design wind speeds that were not read from OSM or
supplied by the user are assumptions and are listed as such in the output.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
import urllib.request
from pathlib import Path
from typing import Optional

import numpy as np

RHO = 1.225                 # air density, kg/m3
ALPHA = 0.16                # power-law exponent, open terrain
GUST = 1.25                 # 1-min sustained → 3-s gust (engineering assumption, open exposure)
FREEBOARD_M = 0.30          # barrier freeboard above the water level
M_PER_DEG = 111320.0
OVERPASS_ENDPOINTS = ["https://overpass-api.de/api/interpreter", "https://overpass.kumi.systems/api/interpreter",
                      "https://overpass.private.coffee/api/interpreter"]
UA = "ClimRisk/1.0 (climate risk research; contact via project owner)"
CACHE = Path(os.environ.get("CRI_CACHE_DIR", Path.home() / ".climrisk" / "cache")) / "osm"

# ── component kinds ───────────────────────────────────────────────────────────
# equip_h  = height above finished floor of the lowest flood-critical equipment (m); None = no flood-critical equipment
# dw       = assumed 3-s gust design wind speed (m/s) — an ASSUMPTION unless the user supplies the as-built value
KINDS = {
    "process":      {"label": "Process hall / plant building", "h": 22.0, "equip_h": 0.30, "dw": 50.0,
                     "flood": "motor control rooms, drives and instrumentation at ground level",
                     "wind": "roof sheeting, cladding and openings"},
    "tower":        {"label": "Tall process tower", "h": 60.0, "equip_h": 0.50, "dw": 55.0,
                     "flood": "ground-floor drives, fans and electrical rooms",
                     "wind": "slender structure — the highest wind loads on site"},
    "silo":         {"label": "Silo / bin", "h": 30.0, "equip_h": 1.00, "dw": 55.0,
                     "flood": "extraction gear and discharge equipment at the base; stored product wetted from below",
                     "wind": "large exposed shell; empty silos are the most vulnerable"},
    "tank":         {"label": "Storage tank", "h": 12.0, "equip_h": 0.00, "dw": 50.0,
                     "flood": "empty or part-full tanks can float or buckle; connecting pipework shears",
                     "wind": "empty tanks buckle or slide; roofs and floating-roof seals are damaged"},
    "stack":        {"label": "Stack / chimney", "h": 60.0, "equip_h": None, "dw": 60.0,
                     "flood": "foundation scour only",
                     "wind": "tall slender structure — across-wind loading and fatigue"},
    "substation":   {"label": "Substation / electrical room", "h": 7.0, "equip_h": 0.50, "dw": 45.0,
                     "flood": "switchgear, transformers and control panels — loss of supply to the whole site",
                     "wind": "overhead lines, insulators and outdoor equipment"},
    "warehouse":    {"label": "Warehouse / store", "h": 9.0, "equip_h": 0.05, "dw": 40.0,
                     "flood": "stock stored on the floor is wetted at the first centimetres of water",
                     "wind": "light roof sheeting and large doors — the first to fail in a storm"},
    "office":       {"label": "Office / ancillary", "h": 7.0, "equip_h": 0.20, "dw": 45.0,
                     "flood": "IT, plant control and records at floor level",
                     "wind": "glazing, roof fixings"},
    "building":     {"label": "Building", "h": 8.0, "equip_h": 0.20, "dw": 45.0,
                     "flood": "floor-level services and contents",
                     "wind": "roof and openings"},
    "jetty":        {"label": "Jetty / berth", "h": 6.0, "equip_h": 1.00, "dw": 50.0,
                     "flood": "loading arms and conveyors at deck level",
                     "wind": "cranes and conveyors exposed to wind and surge"},
}

# Parametric sector layouts (used only when OSM gives no usable footprints). (kind, name, width, depth, height or None)
LAYOUTS = {
    "cement": [("tower", "Preheater tower and kiln", 22, 22, 90), ("process", "Raw / cement mill hall", 60, 28, 28),
               ("silo", "Clinker silo", 24, 24, 38), ("silo", "Cement silo A", 18, 18, 42), ("silo", "Cement silo B", 18, 18, 42),
               ("stack", "Main stack", 6, 6, 80), ("substation", "Main substation / MCC", 30, 18, 7),
               ("warehouse", "Packing and dispatch", 50, 30, 11), ("office", "Administration", 28, 16, 8)],
    "steel": [("process", "Melt shop", 80, 36, 30), ("process", "Rolling mill", 120, 30, 18), ("tank", "Water / quench tank", 20, 20, 10),
              ("stack", "Fume stack", 6, 6, 60), ("substation", "Main substation", 36, 20, 8), ("warehouse", "Product store", 70, 32, 12),
              ("office", "Administration", 26, 16, 8)],
    "power": [("process", "Turbine hall", 70, 30, 26), ("tower", "Boiler house", 40, 36, 55), ("stack", "Stack", 8, 8, 100),
              ("tank", "Fuel / water tank", 22, 22, 14), ("substation", "Switchyard", 50, 40, 12), ("office", "Control building", 28, 18, 9)],
    "oil_gas": [("tank", "Storage tank 1", 28, 28, 14), ("tank", "Storage tank 2", 28, 28, 14), ("process", "Process unit", 50, 30, 20),
                ("stack", "Flare / stack", 4, 4, 60), ("substation", "Substation", 30, 18, 7), ("office", "Control room", 24, 16, 6)],
    "chemicals": [("process", "Reactor / process hall", 50, 30, 24), ("tank", "Chemical tank 1", 20, 20, 12), ("tank", "Chemical tank 2", 20, 20, 12),
                  ("stack", "Scrubber stack", 5, 5, 45), ("substation", "Substation / MCC", 28, 16, 7), ("warehouse", "Drum store", 40, 24, 9)],
    "mining": [("process", "Processing plant", 60, 30, 25), ("silo", "Ore bin", 22, 22, 24), ("substation", "Substation", 28, 18, 7),
               ("warehouse", "Workshop and stores", 44, 24, 10), ("office", "Administration", 24, 14, 7)],
    "aluminium": [("process", "Potline hall", 160, 30, 20), ("process", "Casthouse", 60, 30, 16), ("substation", "Rectifier substation", 40, 24, 10),
                  ("silo", "Alumina silo", 22, 22, 36), ("stack", "Fume treatment stack", 6, 6, 60)],
    "pulp": [("tower", "Recovery boiler", 30, 30, 60), ("process", "Pulp mill hall", 80, 36, 26), ("tank", "Liquor tank", 22, 22, 16),
             ("stack", "Stack", 6, 6, 70), ("substation", "Substation", 30, 18, 7), ("warehouse", "Wood yard store", 60, 30, 10)],
    "real_estate": [("building", "Main block", 60, 24, 24), ("building", "Annex", 30, 20, 14), ("substation", "Plant / electrical room", 16, 12, 5),
                    ("warehouse", "Basement / car park (shown at ground)", 50, 30, 4)],
    "data_centre": [("building", "Data hall", 80, 40, 12), ("substation", "Electrical plant", 40, 20, 8), ("tank", "Fuel tank", 12, 12, 6),
                    ("office", "Operations block", 24, 16, 8)],
    "port_logistics": [("warehouse", "Warehouse 1", 80, 36, 12), ("warehouse", "Warehouse 2", 80, 36, 12), ("jetty", "Berth / jetty", 120, 16, 4),
                       ("substation", "Substation", 24, 16, 6), ("office", "Terminal office", 24, 14, 8)],
    "agriculture": [("warehouse", "Storage barn", 50, 24, 9), ("silo", "Grain silo", 14, 14, 20), ("process", "Processing shed", 40, 22, 9),
                    ("substation", "Pump / electrical room", 14, 10, 4)],
    "generic": [("process", "Main building", 50, 28, 14), ("warehouse", "Store", 40, 24, 9), ("substation", "Electrical room", 18, 12, 6),
                ("office", "Office", 22, 14, 7)],
}

_KEYWORDS = [
    (r"kiln|preheater|cyclone|boiler|tower|column", "tower"),
    (r"silo|hopper|\bbin\b|bunker", "silo"),
    (r"tank|reservoir|holder", "tank"),
    (r"chimney|stack|flare", "stack"),
    (r"substation|transformer|switch|mcc|electrical|power house|powerhouse|generator", "substation"),
    (r"warehouse|godown|store|storage|shed|packing|dispatch|depot", "warehouse"),
    (r"office|admin|canteen|clinic|guest|mosque|school", "office"),
    (r"mill|crusher|plant|factory|process|workshop|hall|works|furnace|turbine|reactor|unit", "process"),
    (r"jetty|pier|wharf|berth|quay", "jetty"),
]


# ── geometry ─────────────────────────────────────────────────────────────────

def _proj(lat0: float, lon0: float):
    k = M_PER_DEG * math.cos(math.radians(lat0))
    return (lambda lat, lon: ((lon - lon0) * k, (lat - lat0) * M_PER_DEG))


def _centroid(poly):
    a = cx = cy = 0.0
    for (x0, y0), (x1, y1) in zip(poly, poly[1:] + poly[:1]):
        c = x0 * y1 - x1 * y0
        a += c
        cx += (x0 + x1) * c
        cy += (y0 + y1) * c
    if abs(a) < 1e-9:
        xs, ys = zip(*poly)
        return sum(xs) / len(xs), sum(ys) / len(ys), 0.0
    return cx / (3 * a), cy / (3 * a), abs(a) / 2


# ── Overpass ──────────────────────────────────────────────────────────────────

def fetch_osm(lat: float, lon: float, radius_m: float, timeout: float = 40.0) -> dict:
    """OSM elements around the site (cached on disk). Returns the raw Overpass JSON."""
    dlat = radius_m / M_PER_DEG
    dlon = radius_m / (M_PER_DEG * math.cos(math.radians(lat)))
    s, w, n, e = lat - dlat, lon - dlon, lat + dlat, lon + dlon
    q = f"""[out:json][timeout:30];
(
 way["building"]({s},{w},{n},{e});
 way["man_made"~"^(chimney|storage_tank|silo|tower|works|water_tower|pier|petroleum_well|flare)$"]({s},{w},{n},{e});
 way["power"~"^(substation|generator|plant)$"]({s},{w},{n},{e});
 way["highway"~"^(motorway|trunk|primary|secondary|tertiary|unclassified|residential|service)$"]({s},{w},{n},{e});
 way["railway"="rail"]({s},{w},{n},{e});
 way["waterway"~"^(river|canal|stream|drain)$"]({s},{w},{n},{e});
 way["natural"="water"]({s},{w},{n},{e});
 node["man_made"~"^(chimney|storage_tank|silo|tower|flare)$"]({s},{w},{n},{e});
);
out body geom;"""
    key = CACHE / (hashlib.md5(f"{s:.4f}{w:.4f}{n:.4f}{e:.4f}".encode()).hexdigest() + ".json")
    if key.exists():
        try:
            return json.loads(key.read_text())
        except Exception:                                      # noqa: BLE001
            pass
    out, last = None, None
    for ep in OVERPASS_ENDPOINTS:                               # public servers are often busy: try each in turn
        req = urllib.request.Request(ep, data=("data=" + urllib.request.quote(q)).encode(),
                                     headers={"User-Agent": UA, "Content-Type": "application/x-www-form-urlencoded"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                out = json.loads(r.read())
            break
        except Exception as exc:                                # noqa: BLE001
            last = exc
            time.sleep(1.0)
    if out is None:
        raise last if last else RuntimeError("Overpass unavailable")
    try:
        CACHE.mkdir(parents=True, exist_ok=True)
        key.write_text(json.dumps(out, separators=(",", ":")))
    except Exception:                                          # noqa: BLE001
        pass
    return out


def _num(v) -> Optional[float]:
    if v is None:
        return None
    m = re.search(r"[-+]?\d+(?:[.,]\d+)?", str(v))
    return float(m.group(0).replace(",", ".")) if m else None


def classify(tags: dict) -> str:
    mm, pw, bd = tags.get("man_made", ""), tags.get("power", ""), tags.get("building", "")
    if pw == "substation":
        return "substation"
    if pw in ("generator", "plant"):
        return "process"
    if mm in ("chimney", "flare"):
        return "stack"
    if mm == "storage_tank" or bd == "storage_tank":
        return "tank"
    if mm == "silo" or bd == "silo":
        return "silo"
    if mm in ("tower", "water_tower"):
        return "tower"
    if mm == "pier":
        return "jetty"
    name = " ".join(str(tags.get(k, "")) for k in ("name", "name:en", "description", "operator", "industrial")).lower()
    for pat, kind in _KEYWORDS:
        if re.search(pat, name):
            return kind
    if bd in ("industrial", "factory", "manufacture"):
        return "process"
    if bd in ("warehouse", "storage", "hangar", "barn", "farm_auxiliary"):
        return "warehouse"
    if bd in ("office", "commercial", "retail", "school", "hospital", "public", "civic"):
        return "office"
    return "building"


def components_from_osm(osm: dict, lat0: float, lon0: float, focus_radius_m: float, max_components: int = 220) -> tuple:
    """(components, linear features, notes). Local metres: x east, y north of the site pin."""
    P = _proj(lat0, lon0)
    comps, lin, n_assumed_h = [], [], 0
    for el in osm.get("elements", []):
        tags = el.get("tags") or {}
        geom = el.get("geometry")
        if el["type"] == "node":
            if not tags.get("man_made"):
                continue
            x, y = P(el["lat"], el["lon"])
            kind = classify(tags)
            r = 3.0 if kind in ("stack", "tower") else 5.0
            poly = [(x - r, y - r), (x + r, y - r), (x + r, y + r), (x - r, y + r)]
            area = (2 * r) ** 2
        elif geom and len(geom) >= 2:
            pts = [P(g["lat"], g["lon"]) for g in geom]
            hw, rw, ww = tags.get("highway"), tags.get("railway"), tags.get("waterway")
            if hw or rw or ww:
                lin.append({"id": f"l{el['id']}", "type": "road" if hw else ("rail" if rw else "water"),
                            "name": tags.get("name") or (hw or rw or ww),
                            "pts": [[round(a, 1), round(b, 1)] for a, b in pts]})
                continue
            if tags.get("natural") == "water":
                lin.append({"id": f"l{el['id']}", "type": "waterbody", "name": tags.get("name") or "water body",
                            "pts": [[round(a, 1), round(b, 1)] for a, b in pts]})
                continue
            if len(pts) < 4 or pts[0] != pts[-1] and math.dist(pts[0], pts[-1]) > 1.0:
                continue
            poly = pts[:-1] if pts[0] == pts[-1] else pts
            _, _, area = _centroid(poly)
            kind = classify(tags)
        else:
            continue
        cx, cy, a = _centroid(poly)
        if math.hypot(cx, cy) > focus_radius_m * 1.6:
            continue
        kd = KINDS[kind]
        h = _num(tags.get("height")) or _num(tags.get("building:height"))
        hsrc = "osm height tag"
        if not h:
            lv = _num(tags.get("building:levels"))
            if lv:
                h, hsrc = lv * 3.2 + 1.0, "osm levels × 3.2 m"
        if not h:
            h, hsrc = kd["h"], "assumed by kind"
            n_assumed_h += 1
        comps.append({
            "id": f"c{el['id']}", "name": tags.get("name") or tags.get("name:en") or kd["label"],
            "kind": kind, "footprint": [[round(px - cx, 1), round(py - cy, 1)] for px, py in poly],
            "x": round(cx, 1), "y": round(cy, 1), "area_m2": round(a, 0),
            "height_m": round(float(h), 1), "height_source": hsrc,
            "equip_height_m": kd["equip_h"], "design_wind_ms": kd["dw"], "design_wind_source": "assumed by kind",
            "source": "osm", "osm_id": el["id"], "on_site": math.hypot(cx, cy) <= focus_radius_m,
        })
    # nearest first, capped; keep every industrial kind
    comps.sort(key=lambda c: (c["kind"] in ("building",), math.hypot(c["x"], c["y"])))
    comps = comps[:max_components]
    comps = [c for c in comps if c["area_m2"] >= 12 or c["kind"] in ("stack", "tank", "silo", "tower")]
    return comps, lin, {"assumed_height_count": n_assumed_h}


def parametric_components(sector_key: str) -> list:
    spec = LAYOUTS.get(sector_key) or LAYOUTS["generic"]
    out, x, row_y, row_h, i = [], -90.0, 40.0, 0.0, 0
    for kind, name, w, d, h in spec:
        if x + w > 110:
            x, row_y, row_h = -90.0, row_y - row_h - 30.0, 0.0
        cx, cy = x + w / 2, row_y - d / 2
        kd = KINDS[kind]
        out.append({"id": f"p{i}", "name": name, "kind": kind,
                    "footprint": [[-w / 2, -d / 2], [w / 2, -d / 2], [w / 2, d / 2], [-w / 2, d / 2]],
                    "x": round(cx, 1), "y": round(cy, 1), "area_m2": round(w * d, 0),
                    "height_m": float(h or kd["h"]), "height_source": "parametric sector template",
                    "equip_height_m": kd["equip_h"], "design_wind_ms": kd["dw"], "design_wind_source": "assumed by kind",
                    "source": "parametric", "on_site": True})
        x += w + 18.0
        row_h = max(row_h, d)
        i += 1
    return out


# ── terrain ──────────────────────────────────────────────────────────────────

class Ground:
    """Smoothed terrain with bilinear sampling in local metres (x east, y north of the pin)."""

    def __init__(self, grid):
        from scipy.ndimage import uniform_filter
        z = np.asarray(grid.elevations, dtype=float)
        z = np.where(np.isfinite(z), z, np.nanmedian(z))
        self.raw = z
        self.z = uniform_filter(z, size=3, mode="nearest")
        self.rows, self.cols = z.shape
        half = float(grid.radius_km) * 1000.0
        self.dx = 2 * half / self.cols
        self.dy = 2 * half / self.rows
        # pixel (r,c) centre → local metres: pin sits at (asset_col, asset_row)
        self.ac, self.ar = float(grid.asset_col), float(grid.asset_row)

    @classmethod
    def from_payload(cls, p: dict) -> "Ground":
        g = object.__new__(cls)
        g.z = g.raw = np.asarray(p["z"], dtype=float)
        g.rows, g.cols = g.z.shape
        g.dx, g.dy = float(p["dx_m"]), float(p["dy_m"])
        g.ar, g.ac = float(p["pin_row"]), float(p["pin_col"])
        return g

    def xy(self, r, c):
        return (c - self.ac) * self.dx, (self.ar - r) * self.dy

    def at(self, x: float, y: float) -> float:
        c = self.ac + x / self.dx
        r = self.ar - y / self.dy
        c = min(max(c, 0.0), self.cols - 1.001)
        r = min(max(r, 0.0), self.rows - 1.001)
        r0, c0 = int(r), int(c)
        fr, fc = r - r0, c - c0
        z = self.z
        return float((z[r0, c0] * (1 - fc) + z[r0, c0 + 1] * fc) * (1 - fr) + (z[r0 + 1, c0] * (1 - fc) + z[r0 + 1, c0 + 1] * fc) * fr)

    def mean_within(self, radius_m: float) -> float:
        rr, cc = np.mgrid[0:self.rows, 0:self.cols]
        d = np.hypot((cc - self.ac) * self.dx, (self.ar - rr) * self.dy)
        m = d <= radius_m
        return float(self.z[m].mean()) if m.any() else float(self.z.mean())

    def ground_pct(self, radius_m: float = 250.0, n: int = 101) -> list:
        rr, cc = np.mgrid[0:self.rows, 0:self.cols]
        d = np.hypot((cc - self.ac) * self.dx, (self.ar - rr) * self.dy)
        v = self.z[d <= radius_m]
        v = v if v.size else self.z.ravel()
        return [round(float(x), 2) for x in np.percentile(v, np.linspace(0, 100, n))]

    def payload(self) -> dict:
        return {"rows": self.rows, "cols": self.cols, "dx_m": round(self.dx, 2), "dy_m": round(self.dy, 2),
                "pin_row": self.ar, "pin_col": self.ac,
                "x0_m": round(-self.ac * self.dx, 1), "y0_m": round(self.ar * self.dy, 1),
                "z": [[round(float(v), 1) for v in row] for row in self.z]}


# ── hazards ──────────────────────────────────────────────────────────────────

def flood_scenarios(flood: dict) -> list:
    out = []
    riv = (flood or {}).get("river") or {}
    if riv.get("available") and riv.get("depths") and not riv.get("footprint_on_water"):
        for rp in sorted(int(k) for k in riv["depths"]):
            d = np.asarray(riv["depths"][str(rp)], dtype=float)
            if d.size == 0:
                continue
            out.append({"id": f"river_{rp}", "kind": "river", "label": f"River flood, 1-in-{rp}", "rp": rp, "year": None,
                        "depth_m": round(float(d.mean()), 2), "depth_p90_m": round(float(np.percentile(d, 90)), 2),
                        "wet_share": round(float((d > 0.02).mean()), 3), "source": riv.get("source"),
                        "q": [round(float(x), 2) for x in np.percentile(d, np.linspace(0, 100, 21))]})
    co = (flood or {}).get("coastal") or {}
    if co.get("available") and co.get("depths"):
        for key in sorted(co["depths"]):
            try:
                y, rp = (int(t) for t in key.split("_"))
            except ValueError:
                continue
            d = np.asarray(co["depths"][key], dtype=float)
            if d.size == 0 or rp < 10:
                continue
            out.append({"id": f"coastal_{y}_{rp}", "kind": "coastal",
                        "label": f"Coastal surge + tide, 1-in-{rp}, {'today' if y <= 2020 else 'with sea-level rise to ' + str(y)}",
                        "rp": rp, "year": y, "depth_m": round(float(d.mean()), 2),
                        "depth_p90_m": round(float(np.percentile(d, 90)), 2), "wet_share": round(float((d > 0.02).mean()), 3),
                        "source": co.get("source"), "q": [round(float(x), 2) for x in np.percentile(d, np.linspace(0, 100, 21))]})
    return out


def _bearing(lat1, lon1, lat2, lon2) -> float:
    p1, p2, dl = math.radians(lat1), math.radians(lat2), math.radians(lon2 - lon1)
    y = math.sin(dl) * math.cos(p2)
    x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return (math.degrees(math.atan2(y, x)) + 360) % 360


def wind_heading(bearing_to_centre: float, lat: float) -> float:
    """Direction the wind blows TOWARD at the site, from the bearing of the storm centre (≈20° inflow)."""
    return (bearing_to_centre + (70.0 if lat >= 0 else -70.0)) % 360


def storm_history(lat: float, lon: float, min_ms: float = 17.5, limit: int = 40) -> list:
    """Every IBTrACS storm since 1980 giving ≥ min_ms at the site, with approach bearing at closest approach."""
    from .climate import tropical_cyclone as tc
    T = tc.tracks()
    d = tc._haversine_km(T["lat"], T["lon"], lat, lon)
    out = []
    for si in np.unique(T["storm"][d <= tc.SEARCH_KM]):
        ix = np.where(T["storm"] == si)[0]
        v, rmin, _ = tc.storm_peak_at(lat, lon, ix, T)
        if v < min_ms:
            continue
        s = tc._interp_storm(ix, T)
        rr = tc._haversine_km(s["lat"], s["lon"], lat, lon)
        j = int(np.nanargmin(rr))
        brg = _bearing(lat, lon, float(s["lat"][j]), float(s["lon"][j]))
        nm = str(T["name"][si]).title()
        if nm.upper() in ("NOT_NAMED", "UNNAMED", "NONAME", ""):
            nm = "Unnamed storm"
        out.append({"id": f"storm_{int(si)}", "kind": "cyclone", "name": nm, "year": int(T["season"][si]),
                    "peak_wind_ms": round(v, 1), "category_at_site": tc.category(v), "closest_km": round(rmin, 0),
                    "bearing_to_centre_deg": round(brg, 0), "wind_to_deg": round(wind_heading(brg, lat), 0)})
    out.sort(key=lambda e: -e["peak_wind_ms"])
    out = out[:limit]
    out.sort(key=lambda e: e["year"])
    return out


def wind_at(v10_ms: float, z_m: float) -> dict:
    """3-s gust at height z and dynamic pressure, from a 10 m open-terrain 1-min sustained wind."""
    vg = v10_ms * GUST * (max(z_m, 10.0) / 10.0) ** ALPHA
    return {"gust_ms": vg, "q_kpa": 0.5 * RHO * vg * vg / 1000.0}


# ── evaluation ───────────────────────────────────────────────────────────────

FLOOD_LABEL = {"dry": "Dry", "wet_floor": "Water on floor, equipment clear", "equipment_inundated": "Equipment inundated",
               "submerged": "Submerged", "protected": "Protected by barrier"}
WIND_BANDS = [(0.6, "none", "No damage expected"), (0.8, "minor", "Minor: loose items, cladding fixings begin to fail"),
              (1.0, "moderate", "Moderate: approaching design wind, roof sheeting and cladding loss likely at weak fixings"),
              (1.2, "severe", "Severe: design wind exceeded"), (9e9, "failure", "Well beyond design: structural damage possible")]


def _barrier(measures, comp_id) -> Optional[float]:
    h = None
    for m in measures or []:
        if m.get("type") == "flood_barrier" and (m.get("components") in (None, "all") or comp_id in (m.get("components") or [])):
            h = max(h or 0.0, float(m.get("height_m", 0)))
    return h


def _sum(measures, typ, comp_id) -> float:
    t = 0.0
    for m in measures or []:
        if m.get("type") == typ and (m.get("components") in (None, "all") or comp_id in (m.get("components") or [])):
            t += float(m.get("delta_m", 0))
    return t


def _design_wind(measures, comp) -> float:
    dw = float(comp.get("design_wind_ms") or 0)
    for m in measures or []:
        if m.get("type") == "wind_retrofit" and (m.get("components") in (None, "all") or comp["id"] in (m.get("components") or [])):
            dw = max(dw, float(m.get("design_wind_ms", 0)))
    return dw


PCT = np.linspace(0.0, 1.0, 21)


def _rank(site: dict, z: float) -> float:
    """Share of the site's ground (within 250 m) that lies at or below elevation z: 0 = lowest ground, 1 = highest."""
    gp = site.get("ground_pct") or []
    if len(gp) < 2:
        return 0.5
    return float(np.interp(z, gp, np.linspace(0.0, 1.0, len(gp))))


def depth_at(site: dict, fl: dict, ground_z: float, add_m: float = 0.0) -> float:
    """Flood depth above ground at elevation ground_z.

    Mapped scenario (fl has "q"): the hazard-map depth distribution is allocated lowest ground first, so the lowest
    land gets the deepest water and the share of land that is wet matches the map. A what-if rise (add_m) deepens
    wet land and floods dry land in order of its height above the wet/dry boundary.
    Custom level (fl has "depth_m" or "water_level_m"): a flat water level over the terrain.
    """
    q = fl.get("q")
    if q:
        r = _rank(site, ground_z)
        d0 = float(np.interp(1.0 - r, np.linspace(0.0, 1.0, len(q)), q))
        if d0 > 0.0:
            return d0 + add_m
        gp = site.get("ground_pct") or []
        zb = float(np.interp(float(fl.get("wet_share", 0.0)), np.linspace(0.0, 1.0, len(gp)), gp)) if len(gp) > 1 else ground_z
        return max(0.0, add_m - max(ground_z - zb, 0.0))
    if fl.get("water_level_m") is not None:
        level = float(fl["water_level_m"])
    else:
        level = float(site["ground_ref_m"]) + float(fl.get("depth_m") or 0.0)
    return max(0.0, level + add_m - ground_z)


def _site_depth(fl: dict, add_m: float) -> float:
    if fl.get("comp") is not None:
        d = sorted(v["depth_at_ground_m"] for v in fl["comp"].values() if v["depth_at_ground_m"] > 0)
        return (d[int(0.9 * (len(d) - 1))] + add_m) if d else 0.0
    q = fl.get("q")
    base = float(q[18]) if q else float(fl.get("depth_m") or 0.0)           # 90th percentile of the mapped depths
    return max(0.0, base + add_m)


def evaluate(components: list, site: dict, hazard: dict, measures: Optional[list] = None,
             linear: Optional[list] = None) -> dict:
    """Per-component state for one flood scenario and / or one wind, with optional protection measures applied.

    hazard = {"flood": {"q": [21 depths], "wet_share": s} | {"depth_m": x} | {"water_level_m": z},
              "wind": {"v10_ms": v}, "what_if": {"flood_add_m": 0, "wind_scale": 1}}
    """
    fl, wd, wi = hazard.get("flood"), hazard.get("wind"), hazard.get("what_if") or {}
    add = float(wi.get("flood_add_m", 0))
    v10 = float(wd["v10_ms"]) * float(wi.get("wind_scale", 1)) if wd and wd.get("v10_ms") is not None else None
    res, tallies = {}, {"flood": {"dry": 0, "wet_floor": 0, "equipment_inundated": 0, "submerged": 0, "protected": 0},
                        "wind": {"none": 0, "minor": 0, "moderate": 0, "severe": 0, "failure": 0}}
    site_depth = _site_depth(fl, add) if fl else 0.0
    for c in components:
        if c.get("removed"):
            continue
        st = {}
        if fl:
            sc_ = (fl.get("comp") or {}).get(c["id"]) if fl.get("comp") is not None else None
            if fl.get("comp") is not None:
                d0 = sc_["depth_at_ground_m"] if sc_ else 0.0
                d_ground = d0 + add if d0 > 0 else 0.0           # solver result; a what-if rise is re-simulated by the viewer
            else:
                d_ground = depth_at(site, fl, float(c["ground_m"]), add)
            above = float(c["floor_elev_m"]) + _sum(measures, "raise_plinth", c["id"]) - float(c["ground_m"])
            depth = max(0.0, d_ground - above)                    # water above the finished floor
            eq = c.get("equip_height_m")
            eqh = None if eq is None else float(eq) + _sum(measures, "raise_equipment", c["id"])
            bar = None if fl.get("barrier_in_sim") else _barrier(measures, c["id"])
            protected = bool(bar) and depth > 0 and bar >= site_depth + FREEBOARD_M
            if protected:
                depth_eff, state = 0.0, "protected"
            else:
                depth_eff = depth
                if depth <= 0:
                    state = "dry"
                elif eqh is None or depth < eqh:
                    state = "wet_floor"
                elif depth < float(c["height_m"]):
                    state = "equipment_inundated"
                else:
                    state = "submerged"
            st["flood"] = {"state": state, "label": FLOOD_LABEL[state], "depth_m": round(depth_eff, 2),
                           "depth_at_ground_m": round(d_ground, 2),
                           "margin_m": None if eqh is None else round(eqh - depth, 2),
                           "barrier_overtopped": bool(bar) and depth > 0 and not protected,
                           "velocity_ms": (sc_ or {}).get("velocity_ms"), "arrival_min": (sc_ or {}).get("arrival_min"),
                           "depth_velocity_m2s": (sc_ or {}).get("depth_velocity_m2s")}
            tallies["flood"][state] += 1
        if v10 is not None:
            w = wind_at(v10, float(c["height_m"]))
            dw = _design_wind(measures, c)
            ratio = w["gust_ms"] / dw if dw else 9.9
            band = next((b for b in WIND_BANDS if ratio < b[0]), WIND_BANDS[-1])
            st["wind"] = {"state": band[1], "label": band[2], "gust_ms": round(w["gust_ms"], 1), "q_kpa": round(w["q_kpa"], 2),
                          "design_wind_ms": round(dw, 1), "ratio": round(ratio, 2)}
            tallies["wind"][band[1]] += 1
        res[c["id"]] = st

    access = []
    if fl and linear:
        for l in linear:
            if l["type"] not in ("road", "rail") or not l.get("z"):
                continue
            d = max(depth_at(site, fl, z, add) for z in l["z"]) if fl.get("comp") is None else fl.get("road_depth", {}).get(l["id"], 0.0)
            state = "passable" if d < 0.15 else ("restricted" if d < 0.30 else "cut")
            access.append({"id": l["id"], "name": l["name"], "type": l["type"], "max_depth_m": round(d, 2), "state": state})
    return {"site_depth_m": round(site_depth, 2), "wind_v10_ms": None if v10 is None else round(v10, 1),
            "components": res, "tally": tallies, "access": access}


# ── build ────────────────────────────────────────────────────────────────────

def _sector_key(sector: Optional[str], atype: Optional[str] = None) -> str:
    try:
        from .physical_report import profile_key
        return profile_key(sector, atype)
    except Exception:                                          # noqa: BLE001
        return "generic"


def apply_edits(components: list, edits: Optional[list]) -> list:
    """User corrections: patch height / kind / equipment level / design wind / name / removal; add equipment by x,y,w,d,h."""
    by = {c["id"]: c for c in components}
    for e in edits or []:
        cid = e.get("id")
        if cid in by:
            c = by[cid]
            for k in ("name", "height_m", "equip_height_m", "design_wind_ms", "floor_offset_m", "removed"):
                if k in e:
                    c[k] = e[k]
                    if k == "height_m":
                        c["height_source"] = "user"
                    if k == "design_wind_ms":
                        c["design_wind_source"] = "user"
            if "kind" in e and e["kind"] in KINDS:
                c["kind"] = e["kind"]
            c["source"] = c["source"] if c["source"] == "user" else c["source"] + "+user"
        else:
            kind = e.get("kind") if e.get("kind") in KINDS else "building"
            kd = KINDS[kind]
            w, d = float(e.get("w", 10)), float(e.get("d", 10))
            n = len([k for k in by if k.startswith("u")])
            c = {"id": f"u{n}", "name": e.get("name") or kd["label"], "kind": kind,
                 "footprint": [[-w / 2, -d / 2], [w / 2, -d / 2], [w / 2, d / 2], [-w / 2, d / 2]],
                 "x": float(e.get("x", 0)), "y": float(e.get("y", 0)), "area_m2": round(w * d, 0),
                 "height_m": float(e.get("height_m", kd["h"])), "height_source": "user",
                 "equip_height_m": e.get("equip_height_m", kd["equip_h"]),
                 "design_wind_ms": float(e.get("design_wind_ms", kd["dw"])),
                 "design_wind_source": "user" if "design_wind_ms" in e else "assumed by kind",
                 "source": "user", "on_site": True}
            by[c["id"]] = c
            components.append(c)
    return components


def place_on_ground(components: list, ground: Ground) -> None:
    for c in components:
        pts = [(c["x"] + fx, c["y"] + fy) for fx, fy in c["footprint"]] + [(c["x"], c["y"])]
        c["ground_m"] = round(float(np.mean([ground.at(x, y) for x, y in pts])), 2)
        c["floor_elev_m"] = round(c["ground_m"] + float(c.get("floor_offset_m", 0.15)), 2)   # 0.15 m default plinth


def build_twin(lat: float, lon: float, name: str = "Asset", sector: Optional[str] = None, asset_type: Optional[str] = None,
               commissioned_year: Optional[int] = None, radius_m: float = 300.0, edits: Optional[list] = None,
               incidents: Optional[list] = None, include_observed: bool = True,
               run_hydraulics: bool = True, _osm=None, _grid=None, _flood=None, _storms=None, _tc=None, _sar=None, _era=None) -> dict:
    t0 = time.time()
    key = _sector_key(sector, asset_type)
    notes, gaps = [], []
    # terrain
    if _grid is None:
        from .connectors.terrain_tiles import get_elevation_grid
        _grid = get_elevation_grid(lat, lon, radius_km=(radius_m + 250) / 1000.0, zoom=14)
    ground = Ground(_grid)
    ground_ref = ground.mean_within(250.0)

    # layout
    linear, comps, layout_source = [], [], "osm"
    try:
        osm = _osm if _osm is not None else fetch_osm(lat, lon, radius_m + 60)
        comps, linear, meta = components_from_osm(osm, lat, lon, radius_m)
    except Exception as exc:                                    # noqa: BLE001
        gaps.append(f"OpenStreetMap footprints could not be read ({type(exc).__name__}).")
    industrial = [c for c in comps if c["kind"] != "building" and c["on_site"]]
    if len([c for c in comps if c["on_site"]]) < 2:
        comps = parametric_components(key) + [c for c in comps if not c["on_site"]]
        layout_source = "parametric"
        notes.append("OpenStreetMap holds no usable building footprints at this location, so an ILLUSTRATIVE layout for the sector is "
                     "shown. It is not the real plant — place and size the real structures with the editor.")
    elif not industrial:
        notes.append("OpenStreetMap footprints here are general buildings; process equipment is not mapped. "
                     "Add the key equipment (silos, tanks, substations, stacks) with the editor so it can be assessed.")
    n_ass = sum(1 for c in comps if c["source"] == "osm" and c["height_source"] == "assumed by kind")
    if n_ass:
        notes.append(f"{n_ass} of {len([c for c in comps if c['source'] == 'osm'])} OpenStreetMap footprints have no height tag; "
                     "their heights are assumed from the kind of structure — correct them in the viewer.")
    comps = apply_edits(comps, edits)
    place_on_ground(comps, ground)
    for l in linear:
        l["z"] = [round(ground.at(x, y), 2) for x, y in l["pts"]]

    # hazards
    if _flood is None:
        from .finance.flood_hazard import site_flood
        try:
            _flood = site_flood(lat, lon, 250.0, elev_m=ground_ref)
        except Exception as exc:                                # noqa: BLE001
            _flood, _ = {}, gaps.append(f"Flood maps unavailable ({type(exc).__name__}).")
    floods = flood_scenarios(_flood)
    if not floods:
        gaps.append("No river or coastal flood depth is mapped for this site (outside map coverage, or the site is on water).")
    try:
        if _tc is None:
            from .climate.tropical_cyclone import site_hazard
            _tc = site_hazard(lat, lon).to_dict()
        winds = [{"id": f"rp{rp}", "kind": "design", "label": f"Cyclone wind, 1-in-{rp} years", "rp": int(rp),
                  "v10_ms": float(v), "category": _tc["rp_category"].get(rp, _tc["rp_category"].get(str(rp)))}
                 for rp, v in sorted(_tc.get("rp_wind_ms", {}).items(), key=lambda kv: int(kv[0])) if float(v) > 0]
        storms = _storms if _storms is not None else storm_history(lat, lon)
    except Exception as exc:                                    # noqa: BLE001
        winds, storms, _tc = [], [], {}
        gaps.append(f"Cyclone model unavailable ({type(exc).__name__}).")
    if not winds:
        notes.append("Tropical-cyclone hazard is negligible here; no cyclone wind scenario is offered (other wind types are not modelled).")

    # all-cause wind (ERA5 reanalysis): ordinary storms, monsoon squalls, cyclones as the reanalysis sees them
    from .wind_extreme import site_extreme_wind
    era = _era if _era is not None else (site_extreme_wind(lat, lon) if include_observed else {"available": False})
    windstorms = []
    if era.get("available"):
        for rp in (10, 50, 100):
            g = float(era["rp_gust_ms"][rp] if rp in era["rp_gust_ms"] else era["rp_gust_ms"][str(rp)])
            winds.append({"id": f"gust{rp}", "kind": "allcause", "label": f"All-cause wind (ERA5), 1-in-{rp} years", "rp": rp, "gust10_ms": g,
                          "v10_ms": round(g / GUST, 1), "source": era["source"],
                          "uncertain": rp in era.get("extrapolated", [])})
        for e in era["top_events"][:6]:
            if e.get("wind_to_deg") is None:
                continue
            windstorms.append({"id": f"era5_{e['date']}", "kind": "windstorm", "name": "Windstorm", "year": e["year"], "date": e["date"],
                               "peak_wind_ms": round(e["gust_ms"] / GUST, 1), "gust_ms": e["gust_ms"], "wind_to_deg": e["wind_to_deg"],
                               "category_at_site": f"gust {e['gust_ms']} m/s", "closest_km": None})
    elif include_observed or _era is not None:
        gaps.append("ERA5 all-cause wind could not be read for this site" + (f" ({era.get('reason')})" if era.get("reason") else "") + ".")
    observed = []
    if include_observed:
        try:
            sar = _sar if _sar is not None else __import__("cri.connectors.sentinel1_sar", fromlist=["x"]).get_sar_water_history(
                lat, lon, radius_km=0.5, days_back=1460, max_scenes=14, include_layer=False)
            if sar.get("available") and (sar.get("peak_wet_within_500m") or 0) >= 0.01 and sar.get("peak_wet_date"):
                observed.append({"kind": "observed_flood", "date": sar["peak_wet_date"], "year": int(sar["peak_wet_date"][:4]),
                                 "label": f"Radar saw {100 * sar['peak_wet_within_500m']:.0f}% of land within 500 m under water",
                                 "source": f"Sentinel-1 ({sar.get('date_range')}, {sar.get('scenes_used')} dates sampled)"})
        except Exception:                                       # noqa: BLE001
            gaps.append("Sentinel-1 flood history could not be read for this site.")
    user_inc = [{"kind": "incident", "date": i.get("date"), "year": int(str(i.get("date", "0"))[:4] or 0), "hazard": i.get("hazard"),
                 "label": i.get("note") or "Reported incident", "depth_m": i.get("depth_m"), "source": "user-reported"}
                for i in (incidents or [])]

    twin = {
        "asset": {"name": name, "lat": lat, "lon": lon, "sector": key, "commissioned_year": commissioned_year},
        "site": {"radius_m": radius_m, "ground_ref_m": round(ground_ref, 2), "relief_m": round(float(ground.z.max() - ground.z.min()), 1),
                 "pin_ground_m": round(ground.at(0, 0), 2), "cell_m": round(ground.dx, 1), "ground_pct": ground.ground_pct(),
                 "layout_source": layout_source, "hydraulics": bool(run_hydraulics), "gust_factor": GUST, "wind_exponent": ALPHA, "freeboard_m": FREEBOARD_M},
        "terrain": ground.payload(),
        "components": comps,
        "linear": linear,
        "scenarios": {"flood": floods, "wind": winds},
        "history": {"storms": storms, "windstorms": windstorms, "observed": observed, "incidents": user_inc,
                    "record_years": (_tc or {}).get("years")},
        "kinds": {k: {"label": v["label"], "equip_h": v["equip_h"], "dw": v["dw"], "h": v["h"]} for k, v in KINDS.items()},
        "data_gaps": gaps, "notes": notes,
        "basis": {
            "layout": "OpenStreetMap contributors via Overpass API (ODbL). Heights from OSM tags where present, otherwise assumed by kind.",
            "terrain": "AWS Terrain Tiles (Terrarium), ~9 m, 3×3 smoothed; includes some building/vegetation height in urban cover.",
            "flood": "JRC CEMS-GloFAS river-flood maps (1 in 10–500 yr) and Deltares coastal maps (2018 and 2050); undefended, 90 m. "
                     "Water is routed over the 9 m terrain by a 2D local-inertial shallow-water solver (Bates et al. 2010) with buildings as "
                     "solid obstructions and Manning friction; the map's water-surface level is imposed at the model edge, at sea and in deep wet "
                     "cells, rising over 2 h and held to 4 h. Arrival time, velocity and depth×velocity are computed. No rainfall, drains, pumps "
                     "or waves; event duration is an assumption; the map level is the ceiling.",
            "wind": "Tropical cyclones: IBTrACS best tracks + Holland (1980) wind profile, 1-min sustained 10 m open-terrain wind. "
                    "All-cause wind: ERA5 daily maximum gust since 1950, Gumbel return levels (0.25° grid: smooths local peaks and "
                    "under-represents cyclones, so it is a floor). Gust factor "
                    f"{GUST}, power-law profile exponent {ALPHA}, dynamic pressure ½ρv².",
            "states": "Flood state compares water level with the finished-floor level and the height of flood-critical equipment. Wind state "
                      "is the ratio of 3-s gust at roof height to the design wind. Both are threshold bands, not damage functions. "
                      "Equipment levels and design winds not supplied by the user are assumptions by kind.",
            "limits": "Wind: tropical-cyclone model plus ERA5 all-cause gusts; tornadoes and thunderstorm downbursts are not modelled. "
                      "No rainfall-runoff, drains, pumps or wave action. Debris impact and structural detail are not modelled.",
        },
        "elapsed_s": round(time.time() - t0, 1),
    }
    return twin


# ── the agent ────────────────────────────────────────────────────────────────

def _key_flood(twin, solver: bool = False) -> list:
    fs = twin["scenarios"]["flood"]
    pick = []
    rivers = [s for s in fs if s["kind"] == "river"]
    coast_now = [s for s in fs if s["kind"] == "coastal" and (s["year"] or 0) <= 2020]
    coast_f = [s for s in fs if s["kind"] == "coastal" and (s["year"] or 0) > 2020]
    for grp, rps in ((rivers, (10, 100)), (coast_now, (100,)), (coast_f, (100,))):
        for rp in rps:
            m = [s for s in grp if s["rp"] == rp]
            if m:
                pick.append(m[0])
    return pick if solver else [s for s in pick if s["depth_m"] > 0]


_DEM = {}


def _dem_coarse(lat: float, lon: float, half_m: float):
    k = (round(lat, 3), round(lon, 3), int(half_m))
    if k not in _DEM:
        from .connectors.terrain_tiles import get_elevation_grid
        _DEM.clear()
        _DEM[k] = Ground(get_elevation_grid(lat, lon, radius_km=(half_m + 300) / 1000.0, zoom=12))
    return _DEM[k]


def simulate_flood(twin: dict, scenario_id: str, add_m: float = 0.0, barrier_m: float = 0.0, n_frames: int = 6, _dem=None) -> dict:
    """Run the 2D solver for one mapped scenario on the twin's own terrain and structures."""
    from . import hydraulics as H
    sc = next((x for x in twin["scenarios"]["flood"] if x["id"] == scenario_id), None)
    if sc is None:
        raise KeyError(scenario_id)
    g = Ground.from_payload(twin["terrain"])
    half = max(g.rows, g.cols) * g.dx / 2.0 + 1200.0
    lat, lon = twin["asset"]["lat"], twin["asset"]["lon"]
    comps = [c for c in twin["components"] if not c.get("removed")]
    on = [c for c in comps if c.get("on_site", True) and (c["kind"] != "building" or not any(x["kind"] != "building" for x in comps if x.get("on_site", True)))]
    ring = max([math.hypot(c["x"], c["y"]) for c in on] or [40.0]) + 25.0
    dem = _dem if _dem is not None else _dem_coarse(lat, lon, half)
    return H.simulate(lat, lon, g, comps, sc, dem, add_m=add_m, barrier_m=barrier_m, barrier_radius_m=ring, n_frames=n_frames)


def ensure_sims(twin: dict) -> None:
    """Run the solver for the key flood scenarios that have no result yet (in parallel); stores twin['sims']."""
    twin.setdefault("sims", {})
    if not twin["site"].get("hydraulics", True):
        return
    from concurrent.futures import ThreadPoolExecutor
    todo = [sc["id"] for sc in _key_flood(twin, solver=True) if sc["id"] not in twin["sims"]]
    if not todo:
        return

    def run(sid):
        try:
            return sid, simulate_flood(twin, sid)
        except Exception as exc:                                # noqa: BLE001
            return sid, {"scenario": sid, "failed": f"{type(exc).__name__}: {str(exc)[:160]}", "dry": True, "comp": {}}
    with ThreadPoolExecutor(max_workers=min(4, len(todo))) as pool:
        for sid, sim in pool.map(run, todo):
            twin["sims"][sid] = sim
    blk = next((v.pop("block", None) for v in twin["sims"].values() if v.get("block")), None)
    for v in twin["sims"].values():
        v.pop("block", None)
    if blk is not None:
        twin["terrain"]["block"] = blk


def _hz_flood(s, twin=None):
    sim = ((twin or {}).get("sims") or {}).get(s["id"])
    if sim and sim.get("comp") and not sim.get("failed"):
        return {"flood": {"comp": sim["comp"], "q": s["q"], "wet_share": s["wet_share"]}, "_sim": True}
    return {"flood": {"q": s["q"], "wet_share": s["wet_share"]}}


def _propose_flood(twin, sc, base, comps_by) -> list:
    """Measures that solve the flooding at this scenario, from the numbers (not from a table)."""
    hit = [cid for cid, st in base["components"].items() if st["flood"]["state"] in ("equipment_inundated", "submerged")]
    wet = [cid for cid, st in base["components"].items() if st["flood"]["state"] == "wet_floor"]
    out = []
    depth_site = base["site_depth_m"]
    if hit or wet:
        # raise equipment: need equipment level ≥ water level + freeboard for every inundated component
        need = 0.0
        for cid in hit:
            st = base["components"][cid]["flood"]
            need = max(need, -(st["margin_m"] if st["margin_m"] is not None else 0) + FREEBOARD_M)
        if hit and need > 0:
            out.append({"type": "raise_equipment", "delta_m": round(math.ceil(need * 10) / 10, 1), "components": hit,
                        "why": "Lift the flood-critical equipment above the water level plus freeboard."})
        out.append({"type": "flood_barrier", "height_m": round(math.ceil((depth_site + FREEBOARD_M) * 10) / 10, 1), "components": "all",
                    "why": "Perimeter flood wall or demountable barrier to the water level plus freeboard (needs sump pumping for rain)."})
    return out


def _propose_wind(twin, sc, base) -> list:
    bad = [cid for cid, st in base["components"].items() if st["wind"]["state"] in ("severe", "failure")]
    if not bad:
        return []
    need = max(base["components"][c]["wind"]["gust_ms"] for c in bad)
    return [{"type": "wind_retrofit", "design_wind_ms": float(math.ceil(need * 1.1 / 5.0) * 5), "components": bad,
             "why": "Strengthen roof fixings, cladding and bracing to a design gust of this speed."}]


def analyse(twin: dict, extra_scenarios: Optional[list] = None) -> dict:
    """Deterministic agent: observe → quantify → diagnose → propose protection → verify by re-evaluation."""
    comps = [c for c in twin["components"] if not c.get("removed")]
    onsite = [c for c in comps if c.get("on_site", True)]
    if any(c["kind"] != "building" for c in onsite):             # plant mapped: judge the plant, treat plain houses as context
        onsite = [c for c in onsite if c["kind"] != "building"]
    by = {c["id"]: c for c in comps}
    gref = twin["site"]
    sector_text = _sector_text(twin["asset"]["sector"])
    steps, findings = [], []

    steps.append({"step": "Observe", "tool": "OpenStreetMap + terrain",
                  "result": f"{len(onsite)} components on site ({twin['site']['layout_source']} layout), ground {twin['site']['ground_ref_m']} m, "
                            f"relief {twin['site']['relief_m']} m across the model."})
    hazards = twin["scenarios"]
    steps.append({"step": "Hazards", "tool": "JRC / Deltares / IBTrACS",
                  "result": f"{len(hazards['flood'])} flood scenarios, {len(hazards['wind'])} design winds, "
                            f"{len(twin['history']['storms'])} historical storms ≥ tropical-storm force."})

    # flood diagnosis — 2D solver where it ran, mapped-depth allocation otherwise
    ensure_sims(twin)
    notes = []
    for sc in _key_flood(twin, solver=True):
        hz = _hz_flood(sc, twin)
        sim = twin.get("sims", {}).get(sc["id"])
        solver_on = bool(hz.get("_sim"))
        hz.pop("_sim", None)
        base = evaluate(onsite, gref, hz, None, twin.get("linear"))
        t = base["tally"]["flood"]
        bad = [cid for cid, st in base["components"].items() if st["flood"]["state"] in ("wet_floor", "equipment_inundated", "submerged")]
        if solver_on:
            v = sim.get("validation") or {}
            steps.append({"step": "Quantify flood", "tool": "2D shallow-water solver (9 m, buildings as obstructions)",
                          "result": f"{sc['label']}: {len(bad)} of {len(onsite)} components reached by water, "
                                    f"{t['equipment_inundated'] + t['submerged']} with equipment inundated; deepest {v.get('max_depth_m', 0)} m; "
                                    f"mass residual {v.get('mass_residual_pct', 0)}%; solver wet land {round(100 * v.get('solver_wet_share_of_open_land', 0))}% vs map "
                                    f"{round(100 * v.get('map_wet_share_of_open_land', 0))}% ({v.get('consistency', '')})."})
            if abs(v.get("solver_wet_share_of_open_land", 0) - v.get("map_wet_share_of_open_land", 0)) > 0.15 and v.get("solver_wet_share_of_open_land", 0) > v.get("map_wet_share_of_open_land", 0):
                notes.append(f"{sc['label']}: " + v["consistency"])
        elif sim and sim.get("dry") and sim.get("note"):
            b = sim.get("boundary") or {}
            steps.append({"step": "Quantify flood", "tool": "2D shallow-water solver",
                          "result": f"{sc['label']}: {sim['note']}"})
            if sc["kind"] == "coastal" and b.get("level_m_p10_p50_p90"):
                notes.append(f"{sc['label']}: the map puts the water surface at about {b['level_m_p10_p50_p90'][1]} m above datum at the model "
                             f"boundary; the site stands at {twin['site']['ground_ref_m']} m.")
        else:
            why = (sim or {}).get("failed")
            steps.append({"step": "Quantify flood", "tool": "mapped-depth allocation (solver unavailable)",
                          "result": f"{sc['label']}: mapped site depth {sc['depth_m']} m → {len(bad)} of {len(onsite)} components wet"
                                    + (f" (solver failed: {why})" if why else "") + "."})
        if solver_on and sim.get("dry") is False and not bad and sc["kind"] == "coastal" and twin["site"]["ground_ref_m"] < 5.0:
            notes.append(f"Verify: {sc['label']} reaches the model but not the site, which stands only {twin['site']['ground_ref_m']} m above datum.")
        if not bad:
            continue
        sd = base["site_depth_m"]
        props = _propose_flood(twin, sc, base, by)
        verified = []
        for m in props:
            if m["type"] == "flood_barrier" and solver_on:
                h, resid = m["height_m"], None
                for _ in range(2):                                  # the solver decides whether the wall holds
                    sb = simulate_flood(twin, sc["id"], barrier_m=h)
                    hb = {"flood": {"comp": sb["comp"], "barrier_in_sim": True}}
                    r = evaluate(onsite, gref, hb, [], twin.get("linear"))
                    left = [cid for cid, st in r["components"].items() if st["flood"]["state"] in ("equipment_inundated", "submerged")]
                    left_wet = [cid for cid, st in r["components"].items() if st["flood"]["state"] == "wet_floor"]
                    resid = (left, left_wet)
                    if not left and not left_wet:
                        break
                    h = round(h + 0.5, 1)
                m = dict(m, height_m=h if (not resid[0] and not resid[1]) else h - 0.5 if False else h)
                verified.append(dict(m, residual_equipment_inundated=len(resid[0]), residual_wet_floor=len(resid[1]),
                                     resolves_equipment=len(resid[0]) == 0, verified_by="2D solver with the wall in the model"))
            else:
                r = evaluate(onsite, gref, hz, [m], twin.get("linear"))
                left = [cid for cid, st in r["components"].items() if st["flood"]["state"] in ("equipment_inundated", "submerged")]
                left_wet = [cid for cid, st in r["components"].items() if st["flood"]["state"] == "wet_floor"]
                verified.append(dict(m, residual_equipment_inundated=len(left), residual_wet_floor=len(left_wet),
                                     resolves_equipment=len(left) == 0, verified_by="re-evaluation of component levels against the solver depths"))
        worst = sorted(bad, key=lambda c: -base["components"][c]["flood"]["depth_m"])[:12]
        fast = [c for c in bad if (base["components"][c]["flood"].get("velocity_ms") or 0) >= 1.0]
        findings.append({
            "id": f"f_{sc['id']}", "hazard": "flood", "scenario": sc["id"], "title": f"{sc['label']}: {len(bad)} component(s) reached by water",
            "components": bad, "worst": worst,
            "problem": _group_flood(worst, by, base),
            "cause": (_flood_cause(sc, sim, solver_on, bad, by, twin)),
            "consequence": sector_text["flood"] + (f" Flow speeds of 1 m/s or more reach {len(fast)} structure(s): at that speed foundations scour and light structures and "
                                                    "stored items are moved." if fast else ""),
            "protection": verified,
            "after": {"flood_tally": base["tally"]["flood"], "combined_applied": bool(props)},
            "access": sorted([a for a in base["access"] if a["state"] != "passable"], key=lambda a: -a["max_depth_m"])[:6],
        })
        steps.append({"step": "Protect and verify", "tool": "solver / re-evaluation",
                      "result": "; ".join(f"{m['type'].replace('_', ' ')} {m.get('delta_m', m.get('height_m'))} m → "
                                          f"{m['residual_equipment_inundated']} inundated, {m['residual_wet_floor']} wet ({m['verified_by']})" for m in verified) or "no measure needed"})
    twin["agent_notes"] = notes

    # wind diagnosis
    wsc = [s for s in hazards["wind"] if (s["kind"] == "design" and s.get("rp") in (100, 250)) or (s["kind"] == "allcause" and s.get("rp") in (50, 100))] + list(extra_scenarios or [])
    st = twin["history"]["storms"]
    if st:
        w = max(st, key=lambda e: e["peak_wind_ms"])
        wsc.append({"id": w["id"], "label": f"{w['name']} {w['year']} (strongest on record at the site)", "v10_ms": w["peak_wind_ms"], "rp": None})
    for sc in wsc:
        base = evaluate(onsite, gref, {"wind": {"v10_ms": sc["v10_ms"]}})
        bad = [cid for cid, st in base["components"].items() if st["wind"]["state"] in ("severe", "failure")]
        steps.append({"step": "Quantify wind", "tool": "power-law + ½ρv²",
                      "result": f"{sc['label']}: {sc['v10_ms']} m/s sustained → {len(bad)} of {len(onsite)} components beyond design wind."})
        if not bad:
            continue
        props = _propose_wind(twin, sc, base)
        verified = []
        for m in props:
            r = evaluate(onsite, gref, {"wind": {"v10_ms": sc["v10_ms"]}}, [m])
            left = [c for c, st in r["components"].items() if st["wind"]["state"] in ("severe", "failure")]
            verified.append(dict(m, residual_beyond_design=len(left)))
        worst = sorted(bad, key=lambda c: -base["components"][c]["wind"]["ratio"])[:12]
        findings.append({
            "id": f"f_{sc['id']}", "hazard": "wind", "scenario": sc["id"], "title": f"{sc['label']}: {len(bad)} component(s) beyond design wind",
            "components": bad, "worst": worst,
            "problem": _group_wind(worst, by, base),
            "cause": (f"A {sc['v10_ms']} m/s sustained wind at 10 m becomes a gust of up to "
                      f"{max(base['components'][c]['wind']['gust_ms'] for c in bad)} m/s at the top of the tallest affected structure "
                      "(wind rises with height), and load grows with the square of speed."),
            "consequence": sector_text["wind"],
            "protection": verified, "after": {"wind_tally": evaluate(onsite, gref, {"wind": {"v10_ms": sc['v10_ms']}}, props)["tally"]["wind"] if props else base["tally"]["wind"]},
        })
        steps.append({"step": "Protect and verify", "tool": "re-evaluation",
                      "result": "; ".join(f"retrofit to {m['design_wind_ms']} m/s → {m['residual_beyond_design']} still beyond design" for m in verified)})

    steps.append({"step": "Residual risk", "tool": "limits",
                  "result": "Events rarer than the scenarios shown, wave loading, debris impact, rainfall drainage and structural detail are not covered; "
                            "protection measures here are sized to the modelled water level or gust and stated freeboard."})
    return {"findings": findings, "transcript": steps, "assumptions": _assumptions(twin), "notes": twin.get("agent_notes", [])}


def _flood_cause(sc, sim, solver_on, bad, by, twin) -> str:
    if solver_on:
        b = sim.get("boundary") or {}
        lv = b.get("level_m_p10_p50_p90")
        arr = [by[c]["id"] for c in bad]
        ams = [sim["comp"][c]["arrival_min"] for c in bad if sim["comp"].get(c, {}).get("arrival_min") is not None]
        gz = [by[c]["ground_m"] for c in bad]
        return (f"The {sc['kind']} flood map puts the water surface at about {lv[1] if lv else '–'} m above datum at the edge of the site model "
                f"(range {lv[0]}–{lv[2]} m). The solver routes that water over the terrain around the buildings"
                + (f"; it reaches the first affected structure after about {int(min(ams))} min of the rise" if ams else "")
                + f". The affected structures stand at {min(gz):.1f}–{max(gz):.1f} m above datum with floors 0.15 m above ground unless edited, "
                  "so water gets in before the equipment inside is lifted clear.")
    return (f"The {sc['kind']} flood map gives a mean depth of {sc['depth_m']} m (90th percentile {sc['depth_p90_m']} m) over the site, with "
            f"{int(round(100 * sc['wet_share']))}% of the ground wet. The affected components stand on the lowest ground "
            f"({min(by[c]['ground_m'] for c in bad)}–{max(by[c]['ground_m'] for c in bad)} m above datum) with floors 0.15 m above it unless edited.")


def _rng(vals, unit="") -> str:
    lo, hi = min(vals), max(vals)
    return f"{lo:g}{unit}" if abs(hi - lo) < 0.05 * max(abs(hi), 1) else f"{lo:g}–{hi:g}{unit}"


def _group_flood(ids, by, base) -> list:
    g = {}
    for c in ids:
        f = base["components"][c]["flood"]
        g.setdefault((by[c]["name"], f["state"]), []).append(f)
    out = []
    for (name, state), fs in g.items():
        eq = "equipment under water" if state != "wet_floor" else "equipment clear by " + _rng([x["margin_m"] for x in fs if x["margin_m"] is not None] or [0], " m")
        out.append(f"{name}: water {_rng([x['depth_m'] for x in fs], ' m')} over the floor, {eq}" + (f" (×{len(fs)})" if len(fs) > 1 else ""))
    return out


def _group_wind(ids, by, base) -> list:
    g = {}
    for c in ids:
        g.setdefault((by[c]["name"], by[c]["design_wind_source"]), []).append((base["components"][c]["wind"], by[c]["height_m"]))
    out = []
    for (name, src), ws in g.items():
        out.append(f"{name}: gust {_rng([w['gust_ms'] for w, _ in ws], ' m/s')} at {_rng([h for _, h in ws], ' m')} height, "
                   f"{_rng([w['q_kpa'] for w, _ in ws], ' kPa')}, {int(round(100 * min(w['ratio'] for w, _ in ws)))}–"
                   f"{int(round(100 * max(w['ratio'] for w, _ in ws)))}% of design ({_rng([w['design_wind_ms'] for w, _ in ws], ' m/s')}, {src})"
                   + (f" (×{len(ws)})" if len(ws) > 1 else ""))
    return out


def _group(lines: list) -> list:
    """Collapse identical lines into 'text (×N)'."""
    seen, out = {}, []
    for t in lines:
        if t in seen:
            seen[t] += 1
        else:
            seen[t] = 1
            out.append(t)
    return [f"{t} (×{seen[t]})" if seen[t] > 1 else t for t in out]


def _sector_text(key: str) -> dict:
    try:
        from .physical_report import PROFILES
        p = PROFILES.get(key) or PROFILES["generic"]
        return {"flood": p["flood"], "wind": p["wind"]}
    except Exception:                                          # noqa: BLE001
        return {"flood": "Water reaching floor-level electrical and process equipment stops production.",
                "wind": "Roof sheeting, cladding and tall structures take the highest wind loads."}


def _assumptions(twin) -> list:
    out = []
    comps = [c for c in twin["components"] if not c.get("removed") and c.get("on_site", True)]
    h = sum(1 for c in comps if c["height_source"] == "assumed by kind")
    dw = sum(1 for c in comps if c["design_wind_source"] == "assumed by kind")
    if h:
        out.append(f"{h} component heights are assumed by kind (no OSM height tag).")
    out.append(f"All equipment flood-critical levels are assumed by kind unless edited ({', '.join(sorted({c['kind'] for c in comps}))}).")
    if dw:
        out.append(f"{dw} components use an assumed design gust by kind; enter the as-built design wind speed to replace it.")
    out.append(f"Finished floor = ground + 0.15 m unless edited; gust factor {GUST}; wind profile exponent {ALPHA}.")
    return out
