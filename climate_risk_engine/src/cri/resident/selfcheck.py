"""
cri/resident/selfcheck.py — does this machine reach every data source the engine depends on?

Uses the same HTTPS path as the real fetches (urllib), so it catches the failures that otherwise show up as a
flat 3D site or empty layers: missing root certificates, no internet, a blocked host. Each check is short and
independent; the result says what failed and why, in plain words.
"""
from __future__ import annotations

import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

_UA = {"User-Agent": "ClimRisk-selfcheck/1.0"}

CHECKS = [
    ("Terrain (AWS elevation tiles)", "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/10/805/445.png"),
    ("Satellite catalogue (Earth Search STAC)", "https://earth-search.aws.element84.com/v1"),
    ("Disaster alerts (GDACS)", "https://www.gdacs.org/gdacsapi/api/events/geteventlist/SEARCH?eventlist=TC&alertlevel=Green;Orange;Red&fromdate=2026-01-01&todate=2026-01-02"),
    ("El Niño index (NOAA CPC)", "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt"),
    ("Buildings (OpenStreetMap Overpass)", "https://overpass-api.de/api/status"),
]


def _hint(msg: str) -> str:
    m = msg.lower()
    if "certificate" in m or "ssl" in m:
        return "Python cannot verify HTTPS certificates on this computer. Update to ClimRisk 0.4.1 or later (it ships its own certificate bundle) and reopen the app."
    if "timed out" in m or "timeout" in m:
        return "The server did not answer in time. Check the internet connection or try again."
    if "name or service" in m or "nodename" in m or "getaddrinfo" in m or "urlopen error" in m:
        return "No route to the server. Check the internet connection, VPN or firewall."
    return "The request failed; see the detail."


def _one(item) -> dict:
    name, url = item
    t0 = time.time()
    try:
        req = urllib.request.Request(url, headers=_UA)
        with urllib.request.urlopen(req, timeout=12) as r:
            r.read(256)
            return {"name": name, "ok": True, "ms": round((time.time() - t0) * 1000), "detail": f"HTTP {r.status}"}
    except urllib.error.HTTPError as e:                      # the host answered: TLS and routing are fine
        return {"name": name, "ok": True, "ms": round((time.time() - t0) * 1000), "detail": f"HTTP {e.code} (reachable)"}
    except Exception as e:                                    # noqa: BLE001
        msg = f"{type(e).__name__}: {e}"
        return {"name": name, "ok": False, "ms": round((time.time() - t0) * 1000), "detail": msg[:200], "fix": _hint(msg)}


def run() -> dict:
    with ThreadPoolExecutor(max_workers=len(CHECKS)) as pool:
        rows = list(pool.map(_one, CHECKS))
    bad = [r for r in rows if not r["ok"]]
    return {"ok": not bad, "checked": len(rows), "failed": len(bad), "rows": rows,
            "summary": "All data sources reachable." if not bad else f"{len(bad)} of {len(rows)} data sources unreachable: " + "; ".join(r["name"] for r in bad)}
