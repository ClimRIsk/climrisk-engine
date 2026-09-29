"""
Precompute hazard screening for every asset in the register so company and
portfolio views load instantly (no network, no modelling at query time).

Per asset (written back into src/cri/data/asset_register.npz):
  tc_rate_ts, tc_rate_cat1   annual rate of tropical-storm-force / hurricane-force
                             winds (exact IBTrACS + Holland model, validated)
  tc_rp100_ms                1-in-100-year open-terrain 1-min wind
  tc_worst                   index into tc_worst_names ("Haiyan 2013")
  elev_m                     ground elevation sampled from AWS terrain tiles at
                             zoom 10 (~150 m) — screening only; the site analysis
                             uses 10 m data

Stages: cyclone | elevation | merge   (each cached, safe to re-run)
"""
from __future__ import annotations

import argparse
import io
import math
import os
import sys
import time
import urllib.request
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
NPZ = ROOT / "src" / "cri" / "data" / "asset_register.npz"
RAW = ROOT / "data" / "asset_register" / "raw"
TILES = RAW / "tiles_z10"
TILES.mkdir(parents=True, exist_ok=True)
Z = 10


def _uniq():
    z = np.load(NPZ)
    ll = np.round(np.c_[z["lat"].astype(float), z["lon"].astype(float)], 2)
    uniq, inv = np.unique(ll, axis=0, return_inverse=True)
    return uniq, inv.ravel()


# ── cyclone ──────────────────────────────────────────────────────────────────

def _tc_chunk(rows):
    from cri.climate import tropical_cyclone as tc
    out = []
    for la, lo in rows:
        h = tc.site_hazard(float(la), float(lo))
        w = f"{h.worst['name']} {h.worst['season']}" if h.worst else ""
        out.append((h.annual_rate_ge_ts, h.annual_rate_ge_cat1, h.rp_wind_ms.get(100, 0.0) or 0.0, w))
    return out


def stage_cyclone(budget_s: float = 140.0):
    uniq, _ = _uniq()
    f = RAW / "tc_screen.npz"
    done = dict(np.load(f, allow_pickle=True)) if f.exists() else {}
    start = int(done.get("n_done", 0))
    ts = list(done.get("ts", [])); c1 = list(done.get("c1", [])); rp = list(done.get("rp", [])); ws = list(done.get("ws", []))
    t0 = time.time()
    step = 800
    with ProcessPoolExecutor(4) as pool:
        while start < len(uniq) and time.time() - t0 < budget_s:
            block = uniq[start:start + step * 4]
            chunks = [block[i:i + step] for i in range(0, len(block), step)]
            for res in pool.map(_tc_chunk, chunks):
                for a, b, c, d in res:
                    ts.append(a); c1.append(b); rp.append(c); ws.append(d)
            start += len(block)
            np.savez(f, n_done=start, ts=np.array(ts, np.float32), c1=np.array(c1, np.float32),
                     rp=np.array(rp, np.float32), ws=np.array(ws, dtype=object))
            print(f"cyclone {start}/{len(uniq)}", flush=True)
    print("cyclone complete" if start >= len(uniq) else "cyclone partial — re-run")


# ── elevation (AWS terrain tiles, terrarium) ─────────────────────────────────

def _tile_xy(lat, lon, z=Z):
    n = 2 ** z
    x = (lon + 180.0) / 360.0 * n
    lr = math.radians(max(min(lat, 85.0), -85.0))
    y = (1 - math.log(math.tan(lr) + 1 / math.cos(lr)) / math.pi) / 2 * n
    return x, y


def _fetch_tile(tx, ty):
    """Decoded terrarium tile (256×256 m) — reuses any tile cached by earlier runs."""
    f = TILES / f"{tx}_{ty}.npy"
    if f.exists():
        return np.load(f)
    from PIL import Image
    url = f"https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{Z}/{tx}/{ty}.png"
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            rgb = np.asarray(Image.open(io.BytesIO(r.read())).convert("RGB"), dtype=np.float32)
        return rgb[..., 0] * 256 + rgb[..., 1] + rgb[..., 2] / 256 - 32768
    except Exception:                          # noqa: BLE001
        return None


def stage_elevation(budget_s: float = 100.0):
    """Sample each asset's elevation tile-by-tile; only the sampled values are kept."""
    uniq, _ = _uniq()
    xy = [_tile_xy(a, b) for a, b in uniq]
    by_tile: dict = {}
    for i, (x, y) in enumerate(xy):
        by_tile.setdefault((int(x), int(y)), []).append(i)
    f = RAW / "elev_screen.npz"
    elev = np.load(f)["elev"] if f.exists() else np.full(len(uniq), np.nan, np.float32)
    done_tiles = set(map(tuple, np.load(f)["done"])) if f.exists() else set()
    todo = [t for t in by_tile if t not in done_tiles]
    print(f"tiles: {len(by_tile)} needed, {len(todo)} to go", flush=True)

    def work(t):
        a = _fetch_tile(*t)
        vals = []
        for i in by_tile[t]:
            x, y = xy[i]
            px, py = min(int((x - t[0]) * 256), 255), min(int((y - t[1]) * 256), 255)
            vals.append((i, float(a[py, px]) if a is not None else np.nan))
        return t, vals, a is not None

    t0 = time.time()
    with ThreadPoolExecutor(32) as pool:
        for k in range(0, len(todo), 320):
            for t, vals, ok in pool.map(work, todo[k:k + 320]):
                for i, v in vals:
                    elev[i] = v
                if ok:
                    done_tiles.add(t)
            np.savez(f, elev=elev, done=np.array(sorted(done_tiles), np.int32).reshape(-1, 2))
            if time.time() - t0 > budget_s:
                print(f"elevation partial ({len(done_tiles)}/{len(by_tile)} tiles) — re-run"); return
    print(f"elevation complete ({len(done_tiles)}/{len(by_tile)} tiles)")


def _sample_elev(uniq):
    return np.load(RAW / "elev_screen.npz")["elev"]


# ── merge ────────────────────────────────────────────────────────────────────

def stage_merge():
    uniq, inv = _uniq()
    tcf = np.load(RAW / "tc_screen.npz", allow_pickle=True)
    assert int(tcf["n_done"]) >= len(uniq), "run --stage cyclone until complete"
    elev = _sample_elev(uniq)
    worst = [str(w) for w in tcf["ws"]]
    names = sorted(set(worst))
    z = dict(np.load(NPZ))
    z["tc_rate_ts"] = tcf["ts"][:len(uniq)][inv]
    z["tc_rate_cat1"] = tcf["c1"][:len(uniq)][inv]
    z["tc_rp100_ms"] = tcf["rp"][:len(uniq)][inv]
    z["tc_worst"] = np.array([names.index(worst[i]) for i in inv], np.int32)
    z["tc_worst_names"] = np.array(names)
    z["elev_m"] = elev[inv]
    np.savez_compressed(NPZ, **z)
    print(f"merged: {len(inv)} assets | cyclone-exposed (≥TS 0.05/yr): {(z['tc_rate_ts'] >= 0.05).sum()} | "
          f"elevation sampled: {np.isfinite(z['elev_m']).sum()} | low-lying (<3 m): {(z['elev_m'] < 3).sum()} | "
          f"{NPZ.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["cyclone", "elevation", "merge"], required=True)
    a = ap.parse_args()
    t = time.time()
    {"cyclone": stage_cyclone, "elevation": stage_elevation, "merge": stage_merge}[a.stage]()
    print(f"({time.time() - t:.0f}s)")
