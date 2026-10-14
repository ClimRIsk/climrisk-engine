"""
cri/analysis/local_air.py — air quality at a place, from what is MEASURED there and what is KNOWN to be nearby.

Why this exists
---------------
The Model Lab air study is a city-scale box model scored against the CAMS analysis. At a site beside a cement works, a
captive power plant or fireworks units it shows "no skill", correctly: a regional model cannot see a point source.
This module answers the right question in three evidence layers, each labelled for what it is:

 1. RECORD     CAMS PM2.5 / PM10 / SO2 / NO2 (Open-Meteo air-quality API, hourly since Aug 2022, ~10 km cell) and ERA5 wind:
               annual means, days above the India NAAQS (60 µg/m3 PM2.5 per 24 h) and the WHO 2021 guideline (15), the
               monthly cycle, and a Diwali-day test. This is regional background. It cannot resolve a source next door.
 2. SOURCES    nearby emitters: (a) found in OpenStreetMap, (b) DECLARED by the user. Each carries its provenance.
               A source the engine cannot find in any database is never silently dropped: the user's word is used and said so.
 3. SCREEN     a Gaussian-plume screen (Briggs urban dispersion, hourly ERA5 wind, Pasquill stability) of each source's
               ground-level increment at the site. Emission rates are NOT measured: they are built from capacity and
               documented factors, in three scenarios (low / central / high concentration). Unknown bearing or distance is
               swept, not guessed. Output is a range with its assumptions beside it.

Not modelled (stated in every result): fugitive dust (quarry, crusher, stockpile, road), secondary PM from SO2 and NOx,
plume rise (so near-field values are conservative), building downwash, fireworks burning events, and any emission above
what the stated scenarios cover. Without a ground monitor at the site the result is a screening range, never a measurement.

Thresholds: WHO 2021 annual PM2.5 guideline 5 µg/m3 (interim targets 35/25/15/10); India NAAQS (2009) annual 40, 24-h 60.
"""
from __future__ import annotations

import datetime as dt
import json
import math
import os
import re
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Callable, Optional

import numpy as np

WHO_ANNUAL, NAAQS_ANNUAL, NAAQS_24H, WHO_24H = 5.0, 40.0, 60.0, 15.0
NAAQS_SO2_ANNUAL, NAAQS_SO2_24H = 50.0, 80.0
DIWALI = {2022: "2022-10-24", 2023: "2023-11-12", 2024: "2024-11-01", 2025: "2025-10-20", 2026: "2026-11-08"}
_UA = {"User-Agent": "climrisk-research/0.1 (local air quality)"}
_AQ_START = "2022-08-01"

COMPASS = {"n": 0, "north": 0, "ne": 45, "north-east": 45, "northeast": 45, "e": 90, "east": 90, "se": 135, "south-east": 135, "southeast": 135,
           "s": 180, "south": 180, "sw": 225, "south-west": 225, "southwest": 225, "w": 270, "west": 270, "nw": 315, "north-west": 315, "northwest": 315}

# Documented defaults. Every one is an ASSUMPTION the user can override on the source (assumptions are echoed in the result).
DEFAULTS = {
    "thermal": {"label": "Captive thermal (coal) power plant", "mw": 13.0, "coal_kg_per_kwh": 0.7, "fluegas_nm3_per_kg_coal": 5.5, "stack_h_m": 50.0,
                "coal_sulphur_pct": 0.5, "pm25_fraction": 0.5, "pm_mg_nm3": {"low": 30.0, "central": 100.0, "high": 500.0}},
    "cement": {"label": "Cement works (kiln stack only)", "clinker_mtpa": 1.0, "fluegas_nm3_per_t_clinker": 2000.0, "stack_h_m": 60.0,
               "pm25_fraction": 0.5, "pm_mg_nm3": {"low": 30.0, "central": 60.0, "high": 300.0}, "operating_days": 330},
}
ASSUMPTION_NOTE = ("Emission rates are built from capacity and typical factors, not measured. The legal limit that applies to a given plant depends on its "
                   "age and consent conditions and was not verified. The cement kiln limit of 30 mg/Nm3 is reported in trade coverage of the 2016 notification; "
                   "the gazette text was not checked. Treat low/central/high as scenarios, and replace them with the plant's stack monitoring data when available.")
NOT_MODELLED = ["fugitive dust (quarry, crusher, stockpiles, haul roads), often larger than the stack for coarse particles",
                "secondary fine particles formed from SO2 and NOx", "plume rise (ignored, so near-field values are on the high side)",
                "building downwash and terrain effects", "fireworks manufacture and burning events (no emission model; see the Diwali test)"]


# ── fetch and cache ──────────────────────────────────────────────────────────────

def _get(url: str, timeout: float = 150.0) -> dict:
    return json.loads(urllib.request.urlopen(urllib.request.Request(url, headers=_UA), timeout=timeout).read())


def _cache_dir() -> Path:
    d = Path(os.environ.get("CRI_CACHE_DIR", Path.home() / ".climrisk" / "cache")) / "local_air"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _fetch_series(lat: float, lon: float, end: dt.date, fetch: Callable = _get) -> dict:
    """Hourly CAMS composition and ERA5 meteorology on a common Asia-neutral UTC clock, cached per place and month."""
    key = _cache_dir() / f"rec_{lat:.2f}_{lon:.2f}_{end:%Y%m}.json"
    if key.exists():
        try:
            return json.loads(key.read_text())
        except Exception:                                          # noqa: BLE001
            pass
    aq: dict = {}
    y = dt.date.fromisoformat(_AQ_START)
    while y <= end:
        y1 = min(dt.date(y.year, 12, 31), end)
        d = fetch("https://air-quality-api.open-meteo.com/v1/air-quality?" + urllib.parse.urlencode(
            {"latitude": round(lat, 4), "longitude": round(lon, 4), "start_date": y.isoformat(), "end_date": y1.isoformat(),
             "hourly": "pm2_5,pm10,sulphur_dioxide,nitrogen_dioxide", "timezone": "UTC"}))
        h = d["hourly"]
        for k in h:
            aq.setdefault(k, []).extend(h[k])
        cell = (d.get("latitude"), d.get("longitude"))
        y = y1 + dt.timedelta(days=1)
    wend = min(end, dt.date.today() - dt.timedelta(days=6))        # ERA5 lags real time by several days
    w = fetch("https://archive-api.open-meteo.com/v1/archive?" + urllib.parse.urlencode(
        {"latitude": round(lat, 4), "longitude": round(lon, 4), "start_date": _AQ_START, "end_date": wend.isoformat(),
         "hourly": "wind_speed_10m,wind_direction_10m,cloud_cover,is_day", "wind_speed_unit": "ms", "timezone": "UTC"}))["hourly"]
    out = {"aq": aq, "met": w, "cell": cell, "fetched": dt.date.today().isoformat()}
    try:
        key.write_text(json.dumps(out))
    except Exception:                                              # noqa: BLE001
        pass
    return out


def _arr(v) -> np.ndarray:
    return np.array([np.nan if x is None else float(x) for x in v], float)


# ── layer 1: the measured record ─────────────────────────────────────────────────

def record_stats(raw: dict) -> dict:
    aq = raw["aq"]
    t = [dt.datetime.fromisoformat(x) for x in aq["time"]]
    pm25, pm10, so2, no2 = (_arr(aq[k]) for k in ("pm2_5", "pm10", "sulphur_dioxide", "nitrogen_dioxide"))
    yrs = np.array([x.year for x in t])
    mon = np.array([x.month for x in t])
    days: dict = {}
    for x, v in zip(t, pm25):
        days.setdefault(x.date(), []).append(v)
    dm = {k: float(np.nanmean(v)) for k, v in days.items() if np.isfinite(v).sum() >= 18}
    full_years = [int(y) for y in sorted(set(yrs)) if (yrs == y).sum() >= 8000]
    by_year = {int(y): {"pm25": round(float(np.nanmean(pm25[yrs == y])), 1), "pm10": round(float(np.nanmean(pm10[yrs == y])), 1),
                        "so2": round(float(np.nanmean(so2[yrs == y])), 1), "no2": round(float(np.nanmean(no2[yrs == y])), 1),
                        "complete": int(y) in full_years} for y in sorted(set(yrs))}
    use = full_years[-3:] or [int(y) for y in sorted(set(yrs))][-1:]
    msk = np.isin(yrs, use)
    dvals = np.array([v for d, v in dm.items() if d.year in use])
    diwali = []
    for y, ds in DIWALI.items():
        d0 = dt.date.fromisoformat(ds)
        if d0 > t[-1].date() or d0 not in dm:
            continue
        base = [dm[d0 + dt.timedelta(days=k)] for k in range(-12, 13) if abs(k) > 3 and (d0 + dt.timedelta(days=k)) in dm]
        peak = max(dm.get(d0, 0), dm.get(d0 + dt.timedelta(days=1), 0))
        if len(base) >= 10:
            diwali.append({"year": y, "date": ds, "day_pm25": round(peak, 1), "baseline_pm25": round(float(np.mean(base)), 1), "ratio": round(peak / max(float(np.mean(base)), 1e-6), 2)})
    return {"cell": raw.get("cell"), "period": [aq["time"][0][:10], aq["time"][-1][:10]], "years_used": use, "by_year": by_year,
            "annual_pm25": round(float(np.nanmean(pm25[msk])), 1), "annual_pm10": round(float(np.nanmean(pm10[msk])), 1),
            "annual_so2": round(float(np.nanmean(so2[msk])), 1), "annual_no2": round(float(np.nanmean(no2[msk])), 1),
            "daily_p98_pm25": round(float(np.percentile(dvals, 98)), 1) if dvals.size else None,
            "days_above_naaqs_24h": round(float(np.mean(dvals > NAAQS_24H) * 365), 1) if dvals.size else None,
            "days_above_who_24h": round(float(np.mean(dvals > WHO_24H) * 365), 1) if dvals.size else None,
            "monthly_pm25": [round(float(np.nanmean(pm25[msk & (mon == m)])), 1) for m in range(1, 13)],
            "diwali": diwali, "resolution": "CAMS analysis, roughly 10 km cell: regional background, not street level and not a ground monitor."}


def severity_pm25(annual: float) -> str:
    """WHO-based screening scale for annual-mean PM2.5 (µg/m3): <=5 NEGLIGIBLE (at the guideline), <=15 LOW (WHO interim target 3),
    <=25 MODERATE (target 2), <=40 HIGH (above target 2, within India NAAQS), >40 CRITICAL (above India NAAQS)."""
    return "NEGLIGIBLE" if annual <= WHO_ANNUAL else "LOW" if annual <= 15 else "MODERATE" if annual <= 25 else "HIGH" if annual <= NAAQS_ANNUAL else "CRITICAL"


# ── layer 2: sources (found + declared) ───────────────────────────────────────────

def _src_path(lat: float, lon: float) -> Path:
    home = Path(os.environ.get("CRI_HOME") or (Path.home() / ".climrisk"))
    d = home / "sources"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{lat:.2f}_{lon:.2f}.json"


def declared(lat: float, lon: float) -> list:
    try:
        return json.loads(_src_path(lat, lon).read_text())
    except Exception:                                              # noqa: BLE001
        return []


def declare(lat: float, lon: float, sources: list) -> list:
    """Store sources the user says are near the site. Replaces an earlier declaration of the same kind."""
    cur = [s for s in declared(lat, lon) if s["kind"] not in {x["kind"] for x in sources}]
    for s in sources:
        s.setdefault("provenance", "declared by the user; not found in OpenStreetMap")
        s.setdefault("declared_at", dt.date.today().isoformat())
    cur += sources
    _src_path(lat, lon).write_text(json.dumps(cur))
    return cur


def parse_declaration(text: str) -> list:
    """'a cement plant and a 13 MW thermal power plant about 3 km north-east, plus fireworks units' -> source dicts."""
    low = text.lower()
    out = []
    dist = re.search(r"(\d+(?:\.\d+)?)\s*km", low)
    d_km = float(dist.group(1)) if dist else None
    brg = None
    for w in sorted(COMPASS, key=len, reverse=True):
        if re.search(r"(?<![a-z])(?:to the |towards? the |on the )?" + re.escape(w) + r"(?:ern|ward|erly)?(?![a-z])", low) and len(w) > 2:
            brg = COMPASS[w]
            break
    common = {"distance_km": d_km, "bearing_deg": brg, "provenance": "declared by the user; not found in OpenStreetMap"}

    def num(rx):
        m = re.search(rx, low)
        return float(m.group(1)) if m else None
    if re.search(r"thermal|coal|power (?:plant|station)|captive power|boiler", low):
        s = {"kind": "thermal", **common, "assumptions": {}}
        mw = num(r"(\d+(?:\.\d+)?)\s*(?:mw|megawatt)")
        if mw:
            s["assumptions"]["mw"] = mw
        out.append(s)
    if re.search(r"cement|clinker|kiln works", low):
        s = {"kind": "cement", **common, "assumptions": {}}
        mt = num(r"(\d+(?:\.\d+)?)\s*(?:mtpa|million (?:tonnes?|tons?))")
        if mt:
            s["assumptions"]["clinker_mtpa"] = mt
        out.append(s)
    if re.search(r"fire\s?works?|crackers?|pyrotech|match (?:factory|units?|industry)|sivakasi", low):
        out.append({"kind": "fireworks", **common, "assumptions": {}})
    return out


def discover_osm(lat: float, lon: float, radius_km: float = 10.0, fetch: Optional[Callable] = None) -> dict:
    """Industrial emitters tagged in OpenStreetMap (power plants other than solar/wind/hydro, works, chimneys, industrial land)."""
    q = (f'[out:json][timeout:25];(nwr(around:{int(radius_km * 1000)},{lat},{lon})["power"~"^(plant|generator)$"];'
         f'nwr(around:{int(radius_km * 1000)},{lat},{lon})["man_made"~"^(works|chimney)$"];'
         f'nwr(around:{int(radius_km * 1000)},{lat},{lon})["industrial"];);out center tags 300;')
    els = None
    for ep in ("https://overpass-api.de/api/interpreter", "https://overpass.kumi.systems/api/interpreter"):
        try:
            if fetch:
                els = fetch(ep, q)
            else:
                els = json.loads(urllib.request.urlopen(urllib.request.Request(ep, data=urllib.parse.urlencode({"data": q}).encode(), headers=_UA), timeout=35).read())["elements"]
            break
        except Exception:                                          # noqa: BLE001
            continue
    if els is None:
        return {"ok": False, "found": [], "note": "OpenStreetMap could not be queried just now."}
    found, skipped = [], 0
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
        kind = "cement" if "cement" in txt else "fireworks" if re.search(r"firework|cracker|pyro", txt) else "thermal" if (t.get("power") and src in ("coal", "gas", "oil", "biomass", "diesel", "")) else "industrial"
        found.append({"kind": kind, "name": t.get("name"), "lat": c["lat"], "lon": c["lon"], "tags": {k: v for k, v in t.items() if k in ("power", "plant:source", "plant:output:electricity", "industrial", "man_made", "operator", "product")},
                      "provenance": "OpenStreetMap"})
    return {"ok": True, "found": found, "skipped_renewable": skipped,
            "note": f"{len(found)} industrial emitters tagged in OpenStreetMap within {radius_km:.0f} km; {skipped} solar/wind/hydro generators ignored. OpenStreetMap is incomplete: absence there is not absence on the ground."}


def _geo(lat0, lon0, lat1, lon1) -> tuple:
    dy = (lat1 - lat0) * 111320.0
    dx = (lon1 - lon0) * 111320.0 * math.cos(math.radians(lat0))
    return math.hypot(dx, dy) / 1000.0, (math.degrees(math.atan2(dx, dy)) + 360) % 360


# ── layer 3: plume screen ───────────────────────────────────────────────────────

def emission(kind: str, assumptions: Optional[dict] = None) -> Optional[dict]:
    """Primary emission rates (g/s) by scenario for a source kind, plus every assumption used."""
    if kind not in DEFAULTS:
        return None
    a = {**DEFAULTS[kind], **(assumptions or {})}
    if kind == "thermal":
        coal_kg_s = a["mw"] * 1000.0 * a["coal_kg_per_kwh"] / 3600.0
        flow = coal_kg_s * a["fluegas_nm3_per_kg_coal"]
        so2 = coal_kg_s * 1000.0 * a["coal_sulphur_pct"] / 100.0 * 2.0
    else:
        flow = a["clinker_mtpa"] * 1e6 * a["fluegas_nm3_per_t_clinker"] / (a["operating_days"] * 86400.0)
        so2 = None
    pm = {k: flow * v / 1000.0 for k, v in a["pm_mg_nm3"].items()}                # mg/Nm3 * Nm3/s = mg/s -> g/s
    return {"flow_nm3_s": round(flow, 1), "pm_g_s": {k: round(v, 3) for k, v in pm.items()}, "pm25_fraction": a["pm25_fraction"],
            "so2_g_s": None if so2 is None else round(so2, 2), "stack_h_m": a["stack_h_m"], "assumptions": a}


def _met(raw: dict) -> dict:
    m = raw["met"]
    u, d = _arr(m["wind_speed_10m"]), _arr(m["wind_direction_10m"])
    cl = _arr(m["cloud_cover"]) / 100.0
    day = _arr(m["is_day"]) > 0.5
    ok = np.isfinite(u) & np.isfinite(d)
    return {"u": u[ok], "d": d[ok], "cloud": np.nan_to_num(cl[ok], nan=0.5), "day": day[ok]}


def _increment(met: dict, q_g_s: float, h_m: float, dist_km: float, bearing: float) -> dict:
    """Receptor concentration (µg/m3) for every hour; annual mean and highest 24 h mean over the record."""
    from ..modellab import air as A
    u = np.maximum(met["u"] * (max(h_m, 10.0) / 10.0) ** 0.25, 1.0)             # power-law wind at stack height, floor 1 m/s
    cls = np.array([A.stability_class(float(a), bool(b), "moderate", float(c)) for a, b, c in zip(met["u"], met["day"], met["cloud"])])
    dd = np.deg2rad(met["d"] - bearing)
    x, y = dist_km * 1000.0 * np.cos(dd), dist_km * 1000.0 * np.sin(dd)
    c = np.zeros_like(u)
    for k in "BCDEF":
        m = cls == k
        if m.any():
            c[m] = A.plume(x[m], y[m], 1.5, q_g_s, u[m], h_m, k) * 1e6                 # g/m3 -> µg/m3
    n = (len(c) // 24) * 24
    d24 = c[:n].reshape(-1, 24).mean(1)
    return {"annual": float(c.mean()), "max24": float(d24.max()), "p98_24": float(np.percentile(d24, 98))}


def screen(lat: float, lon: float, sources: list, raw: dict) -> list:
    """Per source: increments for each emission scenario over the unknown geometry (swept, never guessed)."""
    met = _met(raw)
    res = []
    for s in sources:
        em = emission(s["kind"], s.get("assumptions"))
        item = {"kind": s["kind"], "name": s.get("name"), "provenance": s.get("provenance"), "distance_km": s.get("distance_km"), "bearing_deg": s.get("bearing_deg")}
        if s.get("lat") is not None and s.get("distance_km") is None:
            item["distance_km"], item["bearing_deg"] = (round(v, 2) for v in _geo(lat, lon, s["lat"], s["lon"]))
        if em is None:
            item["modelled"] = False
            item["reason"] = ("Fireworks: no emission model. Manufacture is small-stack and burning is episodic; the Diwali test in the record layer is the only evidence used."
                              if s["kind"] == "fireworks" else "No emission factors for this kind of source.")
            res.append(item)
            continue
        dists = [item["distance_km"]] if item["distance_km"] else [1.0, 3.0, 5.0]
        brgs = [item["bearing_deg"]] if item["bearing_deg"] is not None else list(range(0, 360, 45))
        item["geometry_swept"] = {"distances_km": dists, "bearings_deg": brgs}
        item["emission"] = em
        sc = {}
        for name, q in em["pm_g_s"].items():
            qf = q * em["pm25_fraction"]
            grid = [_increment(met, qf, em["stack_h_m"], d, b) for d in dists for b in brgs]
            sc[name] = {"pm25_annual_ug_m3": {"min": round(min(g["annual"] for g in grid), 2), "max": round(max(g["annual"] for g in grid), 2),
                                              "mean": round(float(np.mean([g["annual"] for g in grid])), 2)},
                        "pm25_worst_24h_ug_m3_max": round(max(g["max24"] for g in grid), 1)}
        if em["so2_g_s"]:
            grid = [_increment(met, em["so2_g_s"], em["stack_h_m"], d, b) for d in dists for b in brgs]
            sc["so2_uncontrolled"] = {"so2_annual_ug_m3": {"min": round(min(g["annual"] for g in grid), 2), "max": round(max(g["annual"] for g in grid), 2),
                                                           "mean": round(float(np.mean([g["annual"] for g in grid])), 2)}}
        item["scenarios"] = sc
        item["modelled"] = True
        res.append(item)
    return res


# ── the entry used by the site profile ───────────────────────────────────────────

def assess(lat: float, lon: float, osm: bool = True, raw: Optional[dict] = None, sources_override: Optional[list] = None, fetch: Callable = _get) -> dict:
    """Everything above in one dict. `status` is modelled (record only), constructed (record + source screen) or null (record unreadable)."""
    research: list = []
    if raw is None:
        try:
            raw = _fetch_series(lat, lon, dt.date.today(), fetch)
        except Exception as e:                                     # noqa: BLE001
            return {"status": "null", "reason": f"CAMS air-quality record could not be read ({type(e).__name__}: {str(e)[:120]}).", "research": research}
    rec = record_stats(raw)
    research.append(f"Read the CAMS record {rec['period'][0]} to {rec['period'][1]}: annual PM2.5 {rec['annual_pm25']} µg/m3 over {rec['years_used']}.")
    srcs = list(sources_override) if sources_override is not None else declared(lat, lon)
    found = None
    if osm and sources_override is None:
        found = discover_osm(lat, lon)
        research.append(found["note"])
        for f in found["found"]:
            if f["kind"] in DEFAULTS and not any(s["kind"] == f["kind"] for s in srcs):
                d, b = _geo(lat, lon, f["lat"], f["lon"])
                srcs.append({**f, "distance_km": round(d, 2), "bearing_deg": round(b)})
    if srcs:
        research.append("Sources used: " + "; ".join(f"{s['kind']} ({s.get('provenance')})" for s in srcs) + ".")
    scr = screen(lat, lon, srcs, raw) if srcs else []
    base = rec["annual_pm25"]
    inc = {}
    for sc in ("low", "central", "high"):
        mx = sum(s["scenarios"][sc]["pm25_annual_ug_m3"]["max"] for s in scr if s.get("modelled"))
        mn = sum(s["scenarios"][sc]["pm25_annual_ug_m3"]["min"] for s in scr if s.get("modelled"))
        inc[sc] = {"min": round(mn, 2), "max": round(mx, 2), "total_with_background_max": round(base + mx, 1)}
    modelled_src = [s for s in scr if s.get("modelled")]
    tot = base + (inc["central"]["max"] if modelled_src else 0.0)
    if modelled_src:
        research.append(f"Plume screen (central scenario, worst bearing): +{inc['central']['max']} µg/m3 on top of the regional {base}; range {inc['low']['max']} to {inc['high']['max']} across low to high emission scenarios.")
    return {"status": "constructed" if modelled_src else "modelled", "record": rec, "sources": scr, "increment": inc, "annual_pm25_total_central": round(tot, 1),
            "severity": severity_pm25(tot), "thresholds": {"who_annual": WHO_ANNUAL, "naaqs_annual": NAAQS_ANNUAL, "naaqs_24h": NAAQS_24H},
            "assumption_note": ASSUMPTION_NOTE if modelled_src else None, "not_modelled": NOT_MODELLED, "osm": found, "research": research,
            "unverified_sources": [s["kind"] for s in srcs if "declared" in str(s.get("provenance"))]}
