"""Check the landslide susceptibility surface against NASA's Global Landslide Catalog (GLC; Kirschbaum et al. 2010, 2015).

Usage: python tools/landslide_validate.py GLC.csv [out.json]
Only events with location_accuracy == 'exact' are used as positives. Each site is a 12 km box around a cluster of exact events.
Two scores per site: the engine's own cell-exact AUC, and an AUC with a tolerance of TOL_CELLS cells (positives = cells within the tolerance of a point) to allow for GPS and
geocoding error. 0.5 = no skill. This is a screening check, not a calibration: the catalog is media-reported (biased to where people are) and every site is one region.
"""
import collections
import csv
import json
import math
import sys

import numpy as np
from scipy import ndimage as ndi

from cri.modellab import landslide as L
from cri.connectors.terrain_tiles import get_elevation_grid

TOL_CELLS = 10
MIN_EVENTS = 5


def clusters(path, n=12, reach_km=5.0):
    """Greedy: the point with most exact rain-triggered events within reach_km becomes a site centre; its events are removed; repeat."""
    rows = [r for r in csv.DictReader(open(path, encoding="utf-8", errors="ignore")) if r["location_accuracy"] == "exact" and r["latitude"] and r["longitude"]
            and r["landslide_trigger"] in ("downpour", "rain", "continuous_rain") and r["landslide_category"] in ("landslide", "mudslide", "debris_flow", "complex", "earth_flow", "rock_fall", "")]
    pts = np.array(sorted({(round(float(r["latitude"]), 5), round(float(r["longitude"]), 5)) for r in rows}))
    out = []
    while len(pts) >= MIN_EVENTS and len(out) < n:
        la0 = np.radians(pts[:, 0])
        d = np.hypot((pts[:, 0][:, None] - pts[:, 0][None, :]) * 111.32, (pts[:, 1][:, None] - pts[:, 1][None, :]) * 111.32 * np.cos(la0)[:, None])
        cnt = (d <= reach_km).sum(1)
        k = int(cnt.argmax())
        if cnt[k] < MIN_EVENTS:
            break
        sel = d[k] <= reach_km
        v = [tuple(x) for x in pts[sel]]
        out.append((float(pts[sel][:, 0].mean()), float(pts[sel][:, 1].mean()), v))
        pts = pts[~sel]
    return out


def one(lat, lon, pts, radius_km=6.5):
    dem = get_elevation_grid(lat, lon, radius_km=radius_km)
    z = np.asarray(dem.elevations, float)
    zf = ndi.median_filter(z, size=3, mode="nearest")
    sl = L.slope_rad(zf, float(dem.cell_size_m))
    land = zf > 0.5
    r = L.run(lat, lon, name="validation", radius_km=radius_km, dem=dem, inventory=pts)
    d = {"lat": round(lat, 3), "lon": round(lon, 3), "events_supplied": len(pts)}
    st = r.get("status") if isinstance(r, dict) else None
    d["engine_status"] = st
    # tolerance AUC using the same wet factor of safety the engine used (re-derive m from the engine's own report is not exposed, so use slope: FS is monotone in slope at fixed soil)
    dlat = dem.cell_size_m / 111320.0
    dlon = dem.cell_size_m / (111320.0 * math.cos(math.radians(lat)))
    mask = np.zeros(z.shape, bool)
    for la, lo in pts:
        i = int(round(dem.asset_row - (la - lat) / dlat)); j = int(round(dem.asset_col + (lo - lon) / dlon))
        if 0 <= i < z.shape[0] and 0 <= j < z.shape[1] and land[i, j]:
            mask[i, j] = True
    d["cells_with_event"] = int(mask.sum())
    if mask.sum() >= 5:
        score = np.degrees(sl)                                                   # monotone in FS at fixed soil, so identical ranking
        d["auc_cell_exact"] = round(L.auc(score[mask], score[land & ~mask]), 3)
        near = ndi.binary_dilation(mask, iterations=TOL_CELLS)
        d["auc_tolerance"] = round(L.auc(score[near & land], score[land & ~near]), 3)
        d["tolerance_m"] = TOL_CELLS * float(dem.cell_size_m)
        d["cell_size_m"] = float(dem.cell_size_m)
        d["land_cells"] = int(land.sum())
        d["share_land_in_tolerance"] = round(float((near & land).sum() / land.sum()), 3)
    return d


if __name__ == "__main__":
    res = []
    for la, lo, v in clusters(sys.argv[1]):
        try:
            res.append(one(la, lo, v))
        except Exception as e:                                                 # noqa: BLE001
            res.append({"lat": round(la, 3), "lon": round(lo, 3), "error": f"{type(e).__name__}: {e}"[:160]})
        print(json.dumps(res[-1]), flush=True)
    a = [r["auc_cell_exact"] for r in res if "auc_cell_exact" in r]
    t = [r["auc_tolerance"] for r in res if "auc_tolerance" in r]
    summ = {"sites": len(res), "mean_auc_cell_exact": round(float(np.mean(a)), 3) if a else None, "mean_auc_tolerance": round(float(np.mean(t)), 3) if t else None}
    print(json.dumps(summ))
    if len(sys.argv) > 2:
        json.dump({"sites": res, "summary": summ}, open(sys.argv[2], "w"), indent=1)
