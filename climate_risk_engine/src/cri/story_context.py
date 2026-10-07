"""
cri/story_context.py — the real-world context a story opens with. Everything here fails soft: no network, no context, never a made-up one.

  oni_series()      NOAA CPC Oceanic Niño Index since 1950 (decimal year, anomaly) + latest state
  nearest_event()   the most recent GDACS tropical cyclone / flood within reach of a point (UN/EC), or None
  site_image()      real Copernicus Sentinel-2 true-colour image of the site (AWS Earth Search), or None
  radar_pair()      Sentinel-1 before/after flood-extent image when an event window is given, or None
  event_from_payload()  operator- or analyst-supplied event facts, each with its source, used verbatim
"""
from __future__ import annotations

import datetime as dt
import math
import os
import tempfile
from pathlib import Path
from typing import Optional

ONI_URL = "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt"
ONI_PAGE = "https://www.cpc.ncep.noaa.gov/products/analysis_monitoring/ensostuff/ONI_v5.php"
_SEAS = {"DJF": 1, "JFM": 2, "FMA": 3, "MAM": 4, "AMJ": 5, "MJJ": 6, "JJA": 7, "JAS": 8, "ASO": 9, "SON": 10, "OND": 11, "NDJ": 12}


def _offline() -> bool:
    return os.environ.get("CRI_STORY_OFFLINE", "").strip() not in ("", "0")


def oni_series() -> Optional[dict]:
    if _offline():
        return None
    try:
        from .resident.research import _get
        txt = _get(ONI_URL, ttl=6 * 3600, as_text=True)
    except Exception:                                                # noqa: BLE001
        return None
    s = []
    for ln in txt.strip().splitlines()[1:]:
        p = ln.split()
        if len(p) >= 4 and p[0] in _SEAS:
            try:
                s.append((int(p[1]) + (_SEAS[p[0]] - 1) / 12.0, float(p[3]), p[0], int(p[1])))
            except ValueError:
                continue
    if len(s) < 100:
        return None
    last = s[-1]
    state = "El Niño" if last[1] >= 0.5 else "La Niña" if last[1] <= -0.5 else "neutral"
    over2 = sum(1 for i in range(1, len(s)) if s[i][1] >= 2.0 > s[i - 1][1])
    return {"series": [(a, b) for a, b, _, _ in s], "latest": {"season": f"{last[2]} {last[3]}", "oni": last[1], "state": state}, "episodes_over_2c": over2,
            "url": ONI_PAGE, "title": "NOAA Climate Prediction Center, Oceanic Niño Index (ERSSTv5)"}


def _hav(lat0, lon0, lat1, lon1) -> float:
    p = math.pi / 180
    a = math.sin((lat1 - lat0) * p / 2) ** 2 + math.cos(lat0 * p) * math.cos(lat1 * p) * math.sin((lon1 - lon0) * p / 2) ** 2
    return 12742 * math.asin(math.sqrt(a))


def nearest_event(lat: float, lon: float, days: int = 45, within_km: float = 400.0) -> Optional[dict]:
    if _offline():
        return None
    try:
        from .resident import research as R
        evs = R.fetch_gdacs(days)
    except Exception:                                                # noqa: BLE001
        return None
    best = None
    for e in evs:
        try:
            d = _hav(lat, lon, e["lat"], e["lon"])
        except KeyError:
            continue
        if d <= max(within_km, R.radius_km(e) * 2) and (best is None or d < best[0]):
            best = (d, e)
    if not best:
        return None
    d, e = best
    if e.get("alert") not in ("Orange", "Red") and d > 120:          # a distant Green alert is not a hook worth opening a report with
        return None
    return {**e, "distance_km": round(d), "source": "GDACS (UN / European Commission)", "url": "https://www.gdacs.org/"}


def site_image(lat: float, lon: float, days_back: int = 150, half_km: float = 3.0) -> Optional[dict]:
    if _offline():
        return None
    try:
        from .research import figures as F
        end = dt.date.today(); start = end - dt.timedelta(days=days_back)
        out = Path(tempfile.mkdtemp()) / "site.png"
        r = F.satellite_image(lat, lon, out, str(start), str(end), half_km=half_km, max_cloud=25.0, title="The site from orbit")
        if r.get("ok"):
            return {"png": out.read_bytes(), "date": r["date"], "credit": r["credit"], "cloud": r.get("cloud")}
    except Exception:                                                # noqa: BLE001
        pass
    return None


def radar_pair(lat: float, lon: float, before: tuple, after: tuple, half_km: float = 8.0) -> Optional[dict]:
    if _offline():
        return None
    """Sentinel-1 before/after only when the caller names the event windows (start, end ISO dates)."""
    try:
        from .research import figures as F
        out = Path(tempfile.mkdtemp()) / "radar.png"
        r = F.radar_flood_image(lat, lon, out, before, after, half_km=half_km)       # type: ignore[arg-type]
        if r.get("ok"):
            return {"png": Path(r["path"]).read_bytes(), "credit": r.get("credit", "Contains modified Copernicus Sentinel-1 data"), "dates": r.get("dates")}
    except Exception:                                                # noqa: BLE001
        pass
    return None


def event_from_payload(p: dict) -> Optional[dict]:
    """payload['event'] = {headline, date, place, notes:[{text, source, url}]} — used verbatim, never extended."""
    e = p.get("event")
    if not e or not (e.get("headline") or e.get("notes")):
        return None
    notes = [n for n in (e.get("notes") or []) if n.get("text") and n.get("source")]
    return {"headline": e.get("headline", ""), "date": e.get("date", ""), "place": e.get("place", ""), "notes": notes}


def parse_event_text(text: str) -> Optional[dict]:
    """Workbench helper: one line per fact, 'fact text | source name | url (optional)'. First line may start with 'HEADLINE:'."""
    if not text or not text.strip():
        return None
    head, notes = "", []
    for ln in text.strip().splitlines():
        ln = ln.strip()
        if not ln:
            continue
        if ln.upper().startswith("HEADLINE:"):
            head = ln.split(":", 1)[1].strip(); continue
        parts = [x.strip() for x in ln.split("|")]
        if len(parts) >= 2 and parts[0] and parts[1]:
            notes.append({"text": parts[0], "source": parts[1], "url": parts[2] if len(parts) > 2 else ""})
    return {"headline": head, "date": "", "place": "", "notes": notes} if (head or notes) else None
