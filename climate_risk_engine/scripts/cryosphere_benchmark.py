"""Benchmarks for cri.cryosphere. Usage:
  python scripts/cryosphere_benchmark.py freeze_thaw  <stations.json> <out.json>
  python scripts/cryosphere_benchmark.py avalanche    <slf.csv> <out.json> [n_events]
  python scripts/cryosphere_benchmark.py permafrost   <perprob.tif> <out.json>
Needs network (GHCN-Daily, Open-Meteo archive, AWS terrain tiles). Results are resumable: re-run to continue.
"""
import json
import math
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cri import cryosphere as C  # noqa: E402


def auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    allv = np.concatenate([pos, neg])
    ranks = allv.argsort().argsort() + 1.0
    # average ranks for ties
    for v in np.unique(allv):
        m = allv == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ra, rb = a.argsort().argsort(), b.argsort().argsort()
    return float(np.corrcoef(ra, rb)[0, 1])


def ghcn_daily(sid: str):
    url = f"https://www.ncei.noaa.gov/pub/data/ghcn/daily/all/{sid}.dly"
    last = None
    for i in range(3):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "climrisk"}), timeout=60) as r:
                return r.read().decode()
        except Exception as e:                       # noqa: BLE001
            last = e
            time.sleep(2)
    raise RuntimeError(str(last))


def station_series(txt: str, y0: int, y1: int):
    """-> dict date -> (tmin, tmax) deg C, only days where both elements exist with a blank quality flag."""
    tmin, tmax = {}, {}
    for line in txt.splitlines():
        el = line[17:21]
        if el not in ("TMIN", "TMAX"):
            continue
        y, m = int(line[11:15]), int(line[15:17])
        if not (y0 <= y <= y1):
            continue
        for d in range(31):
            o = 21 + d * 8
            v = int(line[o:o + 5])
            q = line[o + 6]
            if v == -9999 or q != " ":
                continue
            key = f"{y:04d}-{m:02d}-{d + 1:02d}"
            (tmin if el == "TMIN" else tmax)[key] = v / 10.0
    days = sorted(set(tmin) & set(tmax))
    return days, [tmin[k] for k in days], [tmax[k] for k in days]


def bench_freeze_thaw(stations_json, out):
    st = json.load(open(stations_json))
    res = json.load(open(out)) if Path(out).exists() else {}
    t0 = time.time()
    for sid, (la, lo, el, nm) in st.items():
        if sid in res or time.time() - t0 > 140:
            continue
        try:
            days, a, b = station_series(ghcn_daily(sid), C.FIRST_YEAR, C.LAST_YEAR)
            by = {}
            for d, x, y in zip(days, a, b):
                by.setdefault(d[:4], ([], []))
                by[d[:4]][0].append(x)
                by[d[:4]][1].append(y)
            obs = [C.freeze_thaw_counts(*v) for v in by.values() if len(v[0]) >= 350]
            if len(obs) < 10:
                res[sid] = {"skip": "station record incomplete", "name": nm}
                json.dump(res, open(out, "w"))
                continue
            mod = C.freeze_thaw(la, lo)
            res[sid] = {"name": nm, "lat": la, "lon": lo, "station_elev_m": el, "grid_elev_m": mod["grid_elevation_m"], "years_obs": len(obs),
                        **{f"obs_{k}": round(float(np.mean([o[k] for o in obs])), 1) for k in ("frost_days", "ice_days", "freeze_thaw_days")},
                        "mod_frost_days": mod["frost_days_per_year"], "mod_ice_days": mod["ice_days_per_year"], "mod_freeze_thaw_days": mod["freeze_thaw_days_per_year"],
                        "obs_band": C.freeze_thaw_band(float(np.mean([o["freeze_thaw_days"] for o in obs]))), "mod_band": mod["band"]}
        except Exception as e:                        # noqa: BLE001
            res[sid] = {"skip": str(e)[:100], "name": nm}
        json.dump(res, open(out, "w"))
    json.dump(res, open(out, "w"))
    return res


def summarise_freeze_thaw(out):
    r = {k: v for k, v in json.load(open(out)).items() if "skip" not in v}
    ok = list(r.values())
    summ = {"stations": len(ok)}
    for k in ("frost_days", "ice_days", "freeze_thaw_days"):
        o = np.array([x[f"obs_{k}"] for x in ok])
        m = np.array([x[f"mod_{k}"] for x in ok])
        summ[k] = {"mean_obs": round(float(o.mean()), 1), "mean_bias": round(float((m - o).mean()), 1), "mae": round(float(np.abs(m - o).mean()), 1),
                   "rank_corr": round(spearman(o, m), 3), "within_10_days": round(float((np.abs(m - o) <= 10).mean()), 3)}
    summ["band_match"] = round(float(np.mean([x["obs_band"] == x["mod_band"] for x in ok])), 3)
    summ["band_within_one"] = round(float(np.mean([abs([b for _, b in C.FT_BANDS].index(x["obs_band"]) - [b for _, b in C.FT_BANDS].index(x["mod_band"])) <= 1 for x in ok])), 3)
    elev = np.array([x["station_elev_m"] - (x["grid_elev_m"] or 0) for x in ok])
    d = np.array([x["mod_freeze_thaw_days"] - x["obs_freeze_thaw_days"] for x in ok])
    summ["elevation_offset_corr_with_ft_bias"] = round(float(np.corrcoef(elev, d)[0, 1]), 3)
    return summ


def load_slf(path):
    import csv
    import io
    txt = open(path, encoding="utf-8", errors="replace").read().splitlines()
    hi = [i for i, l in enumerate(txt) if l.startswith('"avalanche.id')][0]
    hdr = [h.strip('"') for h in txt[hi].strip('"').replace('""', '"').split(",")]
    rows = list(csv.DictReader(io.StringIO("\n".join(txt[hi + 1:])), fieldnames=hdr))
    out = []
    for r in rows:
        try:
            la, lo = float(r["start.zone.coordinates.latitude"]), float(r["start.zone.coordinates.longitude"])
            q = float(r["coordinates.quality"] or 9999)
        except (TypeError, ValueError, KeyError):
            continue
        if r.get("activity") in ("tour", "offpiste") and q <= 100 and 45.7 < la < 47.9 and 5.9 < lo < 10.6:
            out.append({"lat": la, "lon": lo, "elev": float(r["start.zone.elevation"]) if r.get("start.zone.elevation") else None,
                        "incl": float(r["start.zone.inclination"]) if r.get("start.zone.inclination") else None, "date": r["date"]})
    return out


def one_point(args):
    la, lo, rad = args
    from cri.connectors.terrain_tiles import get_elevation_grid
    g = get_elevation_grid(la, lo, radius_km=0.9, zoom=13)
    r = C.release_zone_share(g.elevations, g.cell_size_m, (g.asset_row, g.asset_col), rad)
    slope, _ = C.slope_aspect(g.elevations, g.cell_size_m)
    return {"share": r["share"], "max_slope": r.get("max_slope_deg"), "elev": float(g.elevations[g.asset_row, g.asset_col]),
            "slope_here": float(slope[g.asset_row, g.asset_col]), "cell_m": float(g.cell_size_m)}


def bench_avalanche(slf_csv, out, n_events=300, radius=300.0):
    import random
    from concurrent.futures import ThreadPoolExecutor
    ev = load_slf(slf_csv)
    random.seed(11)
    random.shuffle(ev)
    ev = ev[:n_events]
    res = json.load(open(out)) if Path(out).exists() else {"events": [], "controls": []}
    done_e, done_c = len(res["events"]), len(res["controls"])
    elevs = [e["elev"] for e in ev if e["elev"]]
    lo_el, hi_el = np.percentile(elevs, 5), np.percentile(elevs, 95)
    rng = random.Random(23)
    controls = []
    while len(controls) < n_events:
        controls.append((rng.uniform(45.9, 47.6), rng.uniform(6.1, 10.4)))
    t0 = time.time()

    def run(batch, store):
        with ThreadPoolExecutor(14) as ex:
            futs = [(b, ex.submit(one_point, (b[0], b[1], radius))) for b in batch]
            for b, f in futs:
                try:
                    r = f.result(timeout=max(5, 150 - (time.time() - t0)))
                    r.update({"lat": b[0], "lon": b[1]})
                    store.append(r)
                except Exception:                     # noqa: BLE001
                    pass
    if done_e < len(ev):
        run([(e["lat"], e["lon"]) for e in ev[done_e:done_e + 120]], res["events"])
    elif len(res["controls"]) < n_events:
        run(controls[done_c:done_c + 120], res["controls"])
    json.dump({**res, "lo_el": float(lo_el), "hi_el": float(hi_el), "n_events_target": len(ev)}, open(out, "w"))
    print("events", len(res["events"]), "controls", len(res["controls"]))


def one_point2(args):
    la, lo = args
    from cri.connectors.terrain_tiles import get_elevation_grid
    g = get_elevation_grid(la, lo, radius_km=0.5, zoom=14)
    slope, aspect = C.slope_aspect(g.elevations, g.cell_size_m)
    rr, cc = np.indices(g.elevations.shape)
    dist = np.hypot((rr - g.asset_row) * g.cell_size_m, (cc - g.asset_col) * g.cell_size_m)
    out = {"elev": float(g.elevations[g.asset_row, g.asset_col]), "cell_m": float(g.cell_size_m)}
    for rad in (60, 150, 300):
        m = (dist <= rad) & np.isfinite(slope)
        rel = m & (slope >= C.RELEASE_MIN_DEG) & (slope <= C.RELEASE_MAX_DEG)
        out[f"share{rad}"] = float(rel.sum() / m.sum()) if m.any() else None
        out[f"max{rad}"] = float(np.nanmax(slope[m])) if m.any() else None
        out[f"mean{rad}"] = float(np.nanmean(slope[m])) if m.any() else None
    return out


def bench_avalanche2(slf_csv, out, n_events=300):
    import random
    from concurrent.futures import ThreadPoolExecutor
    ev = load_slf(slf_csv)
    random.seed(11)
    random.shuffle(ev)
    ev = ev[:n_events]
    rng = random.Random(5)
    ctl = []
    for e in ev:                                   # matched control: same neighbourhood, 2-8 km away
        for _ in range(1):
            br, dk = rng.uniform(0, 2 * math.pi), rng.uniform(2, 8)
            ctl.append((e["lat"] + dk * math.cos(br) / 111.32, e["lon"] + dk * math.sin(br) / (111.32 * math.cos(math.radians(e["lat"]))), e["elev"]))
    res = json.load(open(out)) if Path(out).exists() else {"events": [], "controls": []}
    t0 = time.time()

    def run(items, store, key):
        with ThreadPoolExecutor(14) as ex:
            futs = [(it, ex.submit(one_point2, (it[0], it[1]))) for it in items]
            for it, f in futs:
                try:
                    r = f.result(timeout=max(5, 150 - (time.time() - t0)))
                    r.update({"lat": it[0], "lon": it[1], "ref_elev": it[2]})
                    store.append(r)
                except Exception:                     # noqa: BLE001
                    pass
    if len(res["events"]) < len(ev):
        run([(e["lat"], e["lon"], e["elev"]) for e in ev[len(res["events"]):len(res["events"]) + 140]], res["events"], "e")
    elif len(res["controls"]) < len(ctl):
        run(ctl[len(res["controls"]):len(res["controls"]) + 140], res["controls"], "c")
    json.dump(res, open(out, "w"))
    print("events", len(res["events"]), "controls", len(res["controls"]), "of", len(ev), len(ctl))


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "avalanche2":
        bench_avalanche2(sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 300)
    if mode == "avalanche":
        bench_avalanche(sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 300)
    if mode == "freeze_thaw":
        bench_freeze_thaw(sys.argv[2], sys.argv[3])
        s = json.load(open(sys.argv[3]))
        print("done", len(s), "of", len(json.load(open(sys.argv[2]))))
        if len(s) >= len(json.load(open(sys.argv[2]))):
            print(json.dumps(summarise_freeze_thaw(sys.argv[3]), indent=1))
