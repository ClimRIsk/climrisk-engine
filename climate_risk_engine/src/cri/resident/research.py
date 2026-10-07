"""
cri/resident/research.py — the resident agent's research desk.

Pulls live hazard feeds (no keys, all public), maps each event onto the engine's asset register (Climate TRACE + EPA
GHGRP heavy-industry assets, with precomputed cyclone and elevation screening), and writes a story from what it finds.

Feeds   GDACS (tropical cyclones and floods, UN/EC)   USGS (M6+ earthquakes, context only)   NOAA CPC ONI (ENSO state)
Mapping an asset is "in the footprint" when it lies within a radius of the event point: tropical cyclone 150 km (hurricane
        class) or 100 km (storm class); flood 75 km. These radii are deliberately coarse — GDACS gives a centroid, not a
        footprint — so the result is "assets in the vicinity", never "assets damaged". The story says so.
"""
from __future__ import annotations

import datetime as dt
import math
import time
from typing import Optional

import numpy as np
import requests

UA = {"User-Agent": "ClimRisk-resident/1.0 (research desk)"}
GDACS = "https://www.gdacs.org/gdacsapi/api/events/geteventlist/SEARCH"
USGS = "https://earthquake.usgs.gov/fdsnws/event/1/query"
ONI = "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt"

_CACHE: dict = {}


def _get(url: str, params: Optional[dict] = None, ttl: int = 1800, as_text: bool = False):
    key = (url, tuple(sorted((params or {}).items())))
    hit = _CACHE.get(key)
    if hit and time.time() - hit[0] < ttl:
        return hit[1]
    last = None
    for k in range(3):
        try:
            r = requests.get(url, params=params, headers=UA, timeout=30)
            r.raise_for_status()
            v = r.text if as_text else r.json()
            _CACHE[key] = (time.time(), v)
            return v
        except Exception as e:                                  # noqa: BLE001
            last = e; time.sleep(1.5 * (k + 1))
    raise RuntimeError(f"{url}: {last}")


def fetch_gdacs(days: int = 21) -> list:
    t = dt.date.today(); f = t - dt.timedelta(days=days)
    j = _get(GDACS, {"eventlist": "TC;FL", "fromdate": str(f), "todate": str(t), "alertlevel": "Green;Orange;Red"})
    out = []
    for x in j.get("features", []):
        p = x["properties"]; g = x.get("geometry") or {}
        if g.get("type") != "Point":
            continue
        lon, lat = g["coordinates"][:2]
        sev = p.get("severitydata") or {}
        out.append({"id": f'{p["eventtype"]}{p["eventid"]}', "type": p["eventtype"], "name": p.get("eventname") or p.get("name"),
                    "title": p.get("name"), "alert": p.get("alertlevel"), "score": float(p.get("alertscore") or 0), "country": p.get("country"),
                    "from": p.get("fromdate"), "to": p.get("todate"), "current": str(p.get("iscurrent")).lower() == "true",
                    "lat": float(lat), "lon": float(lon), "severity": sev.get("severity"), "severity_text": sev.get("severitytext"),
                    "unit": sev.get("severityunit"), "url": (p.get("url") or {}).get("report"), "source": "GDACS"})
    return out


def fetch_usgs(days: int = 30, minmag: float = 6.0) -> list:
    t = dt.date.today(); f = t - dt.timedelta(days=days)
    j = _get(USGS, {"format": "geojson", "starttime": str(f), "endtime": str(t), "minmagnitude": minmag})
    return [{"id": x["id"], "mag": x["properties"]["mag"], "place": x["properties"]["place"], "lat": x["geometry"]["coordinates"][1],
             "lon": x["geometry"]["coordinates"][0], "url": x["properties"].get("url")} for x in j.get("features", [])]


def fetch_enso() -> dict:
    txt = _get(ONI, ttl=6 * 3600, as_text=True)
    last = [ln.split() for ln in txt.strip().splitlines() if ln.strip()][-1]
    season, year, _, anom = last[0], int(last[1]), last[2], float(last[3])
    state = "El Niño" if anom >= 0.5 else "La Niña" if anom <= -0.5 else "neutral"
    strength = "strong" if abs(anom) >= 1.5 else "moderate" if abs(anom) >= 1.0 else "weak" if abs(anom) >= 0.5 else ""
    return {"season": f"{season} {year}", "oni": anom, "state": state, "strength": strength,
            "text": f"ONI {anom:+.2f} °C for {season} {year}: {strength + ' ' if strength else ''}{state}", "url": "https://www.cpc.ncep.noaa.gov/products/analysis_monitoring/ensostuff/ONI_v5.php"}


def _hav(lat0, lon0, lats, lons):
    p = math.pi / 180
    a = np.sin((lats - lat0) * p / 2) ** 2 + math.cos(lat0 * p) * np.cos(lats * p) * np.sin((lons - lon0) * p / 2) ** 2
    return 12742 * np.arcsin(np.sqrt(a))


def radius_km(ev: dict) -> float:
    if ev["type"] == "TC":
        return 150.0 if (ev.get("severity") or 0) >= 119 else 100.0
    return 75.0


def assets_near(lat: float, lon: float, radius: float, limit: int = 40) -> dict:
    from ..data.asset_register import _load, _cyclone_level, _flood_level, LEVELS
    R = _load()
    d = _hav(lat, lon, R["lat"].astype(float), R["lon"].astype(float))
    ii = np.where(d <= radius)[0]
    ii = ii[np.argsort(d[ii])]
    rows = []
    for j in ii:
        em = None if not np.isfinite(R["emissions_t"][j]) else float(R["emissions_t"][j])
        el = None if not np.isfinite(R["elev_m"][j]) else float(R["elev_m"][j])
        cyc, fl = _cyclone_level(float(R["tc_rp100_ms"][j])), _flood_level(float(R["elev_m"][j]))
        rows.append({"i": int(j), "name": str(R["name"][j]), "lat": float(R["lat"][j]), "lon": float(R["lon"][j]),
                     "country": str(R["countries"][R["country"][j]]), "sector": str(R["sectors"][R["sector"][j]]), "type": str(R["atype"][j]),
                     "emissions_t": em, "elev_m": el, "rp100_ms": round(float(R["tc_rp100_ms"][j]), 1), "dist_km": round(float(d[j]), 1),
                     "screen": max(cyc, fl, key=LEVELS.index)})
    em_tot = sum(r["emissions_t"] or 0 for r in rows)
    rows_by_em = sorted(rows, key=lambda r: -(r["emissions_t"] or 0))
    sectors: dict = {}
    for r in rows:
        sectors[r["sector"]] = sectors.get(r["sector"], 0) + 1
    return {"n": len(rows), "radius_km": radius, "emissions_t": em_tot, "sectors": dict(sorted(sectors.items(), key=lambda kv: -kv[1])),
            "top": rows_by_em[:limit], "nearest": rows[:5]}


def scan(days: int = 21, company_keys: Optional[list] = None) -> dict:
    """Run the full research scan. Feed failures are reported, not hidden."""
    status: dict = {}
    events: list = []
    try:
        events = fetch_gdacs(days); status["gdacs"] = f"ok ({len(events)} events)"
    except Exception as e:                                       # noqa: BLE001
        status["gdacs"] = f"failed: {str(e)[:120]}"
    quakes: list = []
    try:
        quakes = fetch_usgs(); status["usgs"] = f"ok ({len(quakes)} M6+)"
    except Exception as e:                                       # noqa: BLE001
        status["usgs"] = f"failed: {str(e)[:120]}"
    enso = None
    try:
        enso = fetch_enso(); status["noaa_cpc"] = "ok"
    except Exception as e:                                       # noqa: BLE001
        status["noaa_cpc"] = f"failed: {str(e)[:120]}"
    keyset = None
    if company_keys:
        try:
            from ..data.asset_register import company_exposure
            keyset = {a["i"] for a in company_exposure(company_keys)["assets"]}
        except Exception:                                        # noqa: BLE001
            keyset = None
    scored = []
    for ev in events:
        try:
            near = assets_near(ev["lat"], ev["lon"], radius_km(ev))
        except Exception as e:                                   # noqa: BLE001
            status["register"] = f"failed: {str(e)[:120]}"; near = {"n": 0, "emissions_t": 0, "top": [], "sectors": {}, "radius_km": radius_km(ev), "nearest": []}
        if keyset is not None:
            near["top"] = [a for a in near["top"] if a["i"] in keyset]
            near["n_company"] = len(near["top"])
        alert_w = {"Red": 3.0, "Orange": 2.0, "Green": 1.0}.get(ev["alert"], 1.0)
        ev["near"] = near
        ev["priority"] = round(alert_w * (1 + math.log1p(near["n"])) * (1.4 if ev["current"] else 1.0), 2)
        scored.append(ev)
    scored.sort(key=lambda e: -e["priority"])
    return {"asof": dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"), "events": scored, "earthquakes": quakes, "enso": enso, "feeds": status,
            "note": "Footprints are centroid + radius (TC 100–150 km, flood 75 km): assets are 'in the vicinity', not 'damaged'."}


def pick_story_event(sc: dict) -> Optional[dict]:
    ev = [e for e in sc["events"] if e["near"]["n"] > 0]
    return ev[0] if ev else None


# ── the story ────────────────────────────────────────────────────────────────

def _fmt_t(x: Optional[float]) -> str:
    if x is None:
        return "n/a"
    return f"{x / 1e6:.1f} Mt" if x >= 1e6 else f"{x / 1e3:.0f} kt" if x >= 1e3 else f"{x:.0f} t"


def build_story(sc: dict, ev: dict, brand: Optional[dict] = None) -> dict:
    """Event → asset-register exposure → story (same format as every ClimRisk report)."""
    from .. import story as S
    near = ev["near"]
    kind = "tropical cyclone" if ev["type"] == "TC" else "flood"
    st = S.Story(kicker="Research desk · " + sc["asof"][:10],
                 title=f'{ev["title"]}: what sits in its path',
                 standfirst=(f'GDACS rates the event {ev["alert"]}. Within {near["radius_km"]:.0f} km of its centre the ClimRisk asset register holds '
                             f'{near["n"]} heavy-industry assets that emit {_fmt_t(near["emissions_t"])} CO₂e a year.'),
                 byline="ClimRisk resident agent", brand=brand)
    st.source("gdacs", f'GDACS — {ev["title"]}', ev.get("url") or "https://www.gdacs.org", "UN OCHA / European Commission Global Disaster Alert and Coordination System")
    st.source("reg", "ClimRisk asset register", "", "Climate TRACE v6 + EPA GHGRP 2023 with IBTrACS cyclone and terrain-elevation screening")
    st.section("event", "What is happening")
    sev = f' ({ev["severity_text"]})' if ev.get("severity_text") else ""
    st.p(f'{ev["title"]} {"affects" if ev["type"] == "FL" else "is near"} {ev.get("country") or "the region"}: {ev["from"][:10]} to {ev["to"][:10]}{sev} [[S:gdacs]].')
    if sc.get("enso"):
        st.source("enso", "NOAA CPC Oceanic Niño Index", sc["enso"]["url"])
        st.p(f'Background state: {sc["enso"]["text"]} [[S:enso]].')
    st.section("assets", "What sits in the vicinity")
    if near["top"]:
        rows = [[a["name"], a["sector"], a["country"], f'{a["dist_km"]:.0f} km', _fmt_t(a["emissions_t"]),
                 "n/a" if a["elev_m"] is None else f'{a["elev_m"]:.0f} m', f'{a["rp100_ms"]:.0f} m/s', a["screen"]] for a in near["top"][:10]]
        st.table(["Asset", "Sector", "Country", "From centre", "CO₂e / yr", "Ground", "1-in-100 wind", "Screen"], rows,
                 caption="Largest emitters inside the radius, from the register's precomputed screening [[S:reg]].")
        st.p("Sectors in range: " + ", ".join(f"{k} ({v})" for k, v in list(near["sectors"].items())[:6]) + ".")
        try:
            lons = [a["lon"] for a in near["top"]]; lats = [a["lat"] for a in near["top"]]
            sizes = [max(a["emissions_t"] or 1e4, 1e4) for a in near["top"]]
            cols = [["NEGLIGIBLE", "LOW", "MODERATE", "HIGH", "CRITICAL"].index(a["screen"]) for a in near["top"]]
            st.figure(S.chart_scatter_map(lons + [ev["lon"]], lats + [ev["lat"]], sizes + [max(sizes)], cols + [4], title=f'{ev["title"]} (event centre) and register assets', cbar="screen level"),
                      caption="Event centre and nearby assets sized by emissions.")
        except Exception:                                        # noqa: BLE001
            pass
    else:
        st.p("No register asset lies within the radius.")
    st.section("loud", "The loud risk and the quiet one")
    nearest = near["nearest"][0] if near["nearest"] else None
    st.p(f'The loud risk is the headline: a {kind} rated {ev["alert"]}. The quiet one is what it does to a specific site — equipment standing on a floor that is '
         f'10 cm too low, a substation behind a wall built for last century\'s storm — and none of that is in a centroid and a radius.'
         + (f' The nearest register asset, {nearest["name"]}, is {nearest["dist_km"]:.0f} km from the centre.' if nearest else ""))
    st.callout("Next step in the engine: open the Asset Twin for any of these coordinates to replay the event on the real footprint with the hydraulic solver and test protection measures.")
    st.section("limits", "What this does not say")
    st.ul([sc["note"], "The register covers heavy industry only (power, cement, steel, refining, chemicals, mining, aluminium); other assets are not counted.",
           "Screening values are precomputed; the site-level twin and risk analyst are the confirmation."])
    return st.build()


def post_from_story(story: dict, ev: dict, sc: dict, link: str = "") -> dict:
    """LinkedIn copy + card image from the same facts as the story (no text the story does not support)."""
    from .. import story as S
    near = ev["near"]
    pts = [f'{near["n"]} heavy-industry assets within {near["radius_km"]:.0f} km, emitting {_fmt_t(near["emissions_t"])} CO₂e a year']
    if near["top"]:
        a = near["top"][0]
        pts.append(f'largest: {a["name"]} ({a["country"]}, {a["sector"]}), {a["dist_km"]:.0f} km from the centre')
    if near["sectors"]:
        pts.append("sectors in range: " + ", ".join(list(near["sectors"])[:3]))
    if sc.get("enso"):
        pts.append(sc["enso"]["text"])
    hook = f'{ev["title"]} is rated {ev["alert"]} by GDACS. The headline is the storm. The risk is what sits under it.'
    text = S.linkedin(story, hook=hook, points=pts, closing="Footprints are centroid + radius, so this is exposure in the vicinity, not damage. Site-level replay in the ClimRisk Asset Twin.",
                      tags=["#ClimateRisk", "#PhysicalRisk", "#Resilience"], link=link)
    img = S.chart_card(f'{ev["title"]}', [(str(near["n"]), "assets in range"), (_fmt_t(near["emissions_t"]), "CO₂e a year"), (ev["alert"], "GDACS alert")],
                       footer="Source: GDACS · ClimRisk asset register. Centroid + radius, exposure not damage.")
    return {"text": text, "image": img, "title": story["title"]}
