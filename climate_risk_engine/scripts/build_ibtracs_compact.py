"""
Build the compact global tropical-cyclone track file shipped with the engine.

Source: NOAA NCEI IBTrACS v04r01, "since1980" (all basins), public domain.
  https://www.ncei.noaa.gov/products/international-best-track-archive

Kept per track point (main track, tropical/mixed nature, valid position):
  storm index, time (hours since 1980-01-01), lat, lon,
  vmax_kt   1-minute sustained wind: USA_WIND (NHC/JTWC, near-global since 1980);
            else WMO_WIND converted from 10-min to 1-min (÷ 0.88, WMO 2010 guidance)
  rmw_nmi   radius of maximum wind (USA_RMW) — NaN when not reported
  pres_mb, dist2land_km, speed_kt, dir_deg
Per storm: SID, name, season, basin.

Output: src/cri/data/ibtracs_since1980.npz (compressed)

Usage: python scripts/build_ibtracs_compact.py /path/to/ibtracs.since1980.list.v04r01.csv
"""
from __future__ import annotations

import csv
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "src" / "cri" / "data" / "ibtracs_since1980.npz"
T0 = datetime(1980, 1, 1)
BASINS = ["NA", "EP", "WP", "NI", "SI", "SP", "SA", "MM"]


def f(x: str) -> float:
    x = (x or "").strip()
    try:
        return float(x)
    except ValueError:
        return float("nan")


def main(path: str) -> None:
    sids, names, seasons, basins = [], [], [], []
    sid_ix: dict[str, int] = {}
    cols = {k: [] for k in ("storm", "t", "lat", "lon", "vmax", "rmw", "pres", "d2l", "spd", "dir")}
    with open(path, newline="", encoding="latin-1") as fh:
        rd = csv.DictReader(fh)
        next(rd)                                           # units row
        for r in rd:
            if r["TRACK_TYPE"].strip() != "main" or r["NATURE"].strip() not in ("TS", "MX", "NR"):
                continue
            lat, lon = f(r["LAT"]), f(r["LON"])
            if np.isnan(lat) or np.isnan(lon):
                continue
            v = f(r["USA_WIND"])
            if np.isnan(v):
                w = f(r["WMO_WIND"])
                v = w / 0.88 if not np.isnan(w) else float("nan")
            if np.isnan(v) or v < 20:                      # below tropical-depression strength
                continue
            sid = r["SID"].strip()
            if sid not in sid_ix:
                sid_ix[sid] = len(sids)
                sids.append(sid); names.append(r["NAME"].strip()); seasons.append(int(f(r["SEASON"])))
                b = r["BASIN"].strip()
                basins.append(BASINS.index(b) if b in BASINS else 7)
            t = datetime.strptime(r["ISO_TIME"].strip(), "%Y-%m-%d %H:%M:%S")
            cols["storm"].append(sid_ix[sid])
            cols["t"].append((t - T0).total_seconds() / 3600.0)
            cols["lat"].append(lat); cols["lon"].append(lon); cols["vmax"].append(v)
            cols["rmw"].append(f(r["USA_RMW"])); cols["pres"].append(f(r["USA_PRES"]) if r["USA_PRES"].strip() else f(r["WMO_PRES"]))
            cols["d2l"].append(f(r["DIST2LAND"])); cols["spd"].append(f(r["STORM_SPEED"])); cols["dir"].append(f(r["STORM_DIR"]))

    np.savez_compressed(
        OUT,
        storm=np.array(cols["storm"], np.int32), t=np.array(cols["t"], np.float32),
        lat=np.array(cols["lat"], np.float32), lon=np.array(cols["lon"], np.float32),
        vmax=np.array(cols["vmax"], np.float32), rmw=np.array(cols["rmw"], np.float32),
        pres=np.array(cols["pres"], np.float32), d2l=np.array(cols["d2l"], np.float32),
        spd=np.array(cols["spd"], np.float32), dir=np.array(cols["dir"], np.float32),
        sid=np.array(sids), name=np.array(names), season=np.array(seasons, np.int16),
        basin=np.array(basins, np.int8), basins=np.array(BASINS),
        source=np.array("NOAA NCEI IBTrACS v04r01 since1980 (public domain)"),
    )
    print(f"{len(sids)} storms, {len(cols['t'])} track points, seasons {min(seasons)}-{max(seasons)} → {OUT} "
          f"({OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main(sys.argv[1])
