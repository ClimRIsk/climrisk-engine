"""
cri/hydraulics.py — a 2D shallow-water flood solver for site scale, driven by the engine's regional flood maps.

Why it exists
  The JRC (river) and Deltares (coast) maps are 90 m and give a final depth. They cannot say how water gets onto a site
  that sits among buildings, how fast it arrives, how fast it moves, or what a wall changes. This module routes water
  over the 9 m terrain with buildings as impermeable obstructions.

Solver     Local-inertial shallow-water equations (Bates, Horritt & Fewtrell 2010, the LISFLOOD-FP formulation) on a
           staggered grid; Manning friction; semi-implicit friction term; adaptive time step by the CFL condition;
           flux limiter so a cell cannot give away more water than it holds. Mass is tracked and checked.
Forcing    The 90 m flood map for the chosen scenario gives a water-surface elevation (map depth + terrain). That level
           is imposed on (a) the model edge where the map is wet, (b) sea cells, (c) deep interior wet cells (≥ 1 m:
           channels and basins). The imposed level rises from the lowest boundary cell to the map level over a ramp
           (2 h), then holds (to 4 h). Shallower land is flooded only by routing, so arrival time, velocity and the
           effect of obstructions are computed, not assigned.
Honest limits
  • The map level is the ceiling: the solver cannot flood to a level above it. The map itself is 90 m and undefended.
  • Event duration is a stated assumption (1.5 h rise, 1.5 h hold); a longer event only moves depths toward the map level.
  • Terrain is the AWS terrain tiles (≈9 m, includes some building and tree height); the map used NASADEM. Their
    difference is a source of error of the order of a metre in dense or hilly land; it is reported, not hidden.
  • No rainfall, drains, culverts, pumps, bridges or wave action. Buildings are treated as solid.
"""
from __future__ import annotations

import math
import time
from typing import Optional

import numpy as np

G = 9.81
PX_DEG = 1.0 / 1200.0
M_LAT = 111320.0 * PX_DEG                  # metres per 3-arc-second pixel, north-south


# ── solver ───────────────────────────────────────────────────────────────────

def solve(z: np.ndarray, block: np.ndarray, stage: np.ndarray, target: np.ndarray, dx: float, dy: float = None,
          t_end: float = 10800.0, ramp_s: float = 5400.0, manning: float = 0.04, n_frames: int = 8, dt_max: float = 4.0,
          alpha: float = 0.7, hmin: float = 0.005, max_steps: int = 60000) -> dict:
    """Run the local-inertial solver. z, block, stage, target are (R, C) arrays; target is the held water level (m) in stage cells."""
    dy = dy or dx
    R, C = z.shape
    z = z.astype(np.float64)
    h = np.zeros((R, C))
    qx = np.zeros((R, C - 1))
    qy = np.zeros((R - 1, C))
    open_x = ~(block[:, :-1] | block[:, 1:])
    open_y = ~(block[:-1, :] | block[1:, :])
    stage = stage & ~block
    lmin = float(z[stage].min()) if stage.any() else 0.0
    tgt = np.where(stage, target, 0.0)
    hmax = np.zeros((R, C)); vmax = np.zeros((R, C)); arrive = np.full((R, C), np.nan)
    frames, f_times = [], [t_end * (k + 1) / n_frames for k in range(n_frames)]
    added = 0.0
    t, step, nf = 0.0, 0, 0
    n2 = manning ** 2
    cell_area = dx * dy

    def hold(t_now, h):
        lvl = lmin + (tgt - lmin) * min(t_now / ramp_s, 1.0)
        want = np.where(stage, np.maximum(lvl - z, 0.0), h)
        return want

    h = hold(0.0, h)
    while t < t_end and step < max_steps:
        hm = float(h.max())
        dt = dt_max if hm < 0.02 else min(dt_max, alpha * min(dx, dy) / math.sqrt(G * hm))
        dt = max(dt, 0.05)
        eta = z + h
        # x faces
        zf = np.maximum(z[:, :-1], z[:, 1:])
        hf = np.maximum(np.maximum(eta[:, :-1], eta[:, 1:]) - zf, 0.0)
        ok = open_x & (hf > hmin)
        slope = (eta[:, 1:] - eta[:, :-1]) / dx
        hf_s = np.where(ok, hf, 1.0)
        qx = np.where(ok, (qx - G * hf * dt * slope) / (1.0 + G * dt * n2 * np.abs(qx) / hf_s ** (7.0 / 3.0)), 0.0)
        cap = 0.25 * dx * np.where(qx > 0, h[:, :-1], h[:, 1:]) / dt
        qx = np.clip(qx, -cap, cap)
        # y faces (row index grows southward; eta difference sign handled consistently with flux direction = +row)
        zf = np.maximum(z[:-1, :], z[1:, :])
        hf = np.maximum(np.maximum(eta[:-1, :], eta[1:, :]) - zf, 0.0)
        ok = open_y & (hf > hmin)
        slope = (eta[1:, :] - eta[:-1, :]) / dy
        hf_s = np.where(ok, hf, 1.0)
        qy = np.where(ok, (qy - G * hf * dt * slope) / (1.0 + G * dt * n2 * np.abs(qy) / hf_s ** (7.0 / 3.0)), 0.0)
        cap = 0.25 * dy * np.where(qy > 0, h[:-1, :], h[1:, :]) / dt
        qy = np.clip(qy, -cap, cap)
        # continuity
        div = np.zeros((R, C))
        div[:, :-1] += qx / dx; div[:, 1:] -= qx / dx
        div[:-1, :] += qy / dy; div[1:, :] -= qy / dy
        h = np.maximum(h - dt * div, 0.0)
        t += dt
        step += 1
        want = hold(t, h)
        added += float(((want - h) * stage).sum()) * cell_area
        h = want
        wet = h > 0.05
        newly = wet & np.isnan(arrive)
        arrive[newly] = t
        hmax = np.maximum(hmax, h)
        if step % 4 == 0:                                     # velocity at cell centres (average of the four faces)
            ux = np.zeros((R, C)); uy = np.zeros((R, C))
            ux[:, :-1] += qx; ux[:, 1:] += qx; uy[:-1, :] += qy; uy[1:, :] += qy
            sp = np.hypot(ux, uy) / 2.0
            v = np.where(h > 0.05, sp / np.maximum(h, 0.05), 0.0)
            vmax = np.maximum(vmax, np.minimum(v, 8.0))
        while nf < n_frames and t >= f_times[nf]:
            frames.append(h.copy()); nf += 1
    while len(frames) < n_frames:
        frames.append(h.copy())
    v_final = float((h * cell_area).sum())
    inner = ~stage
    return {"hmax": hmax, "vmax": vmax, "arrive_s": arrive, "frames": frames, "frame_t": f_times, "h_final": h, "steps": step,
            "t_sim": t, "volume_end_m3": v_final, "volume_added_m3": added,
            "volume_in_non_stage_m3": float((h * inner * cell_area).sum())}


# ── regional forcing ─────────────────────────────────────────────────────────

def _grid_xy(R: int, C: int, ac: float, ar: float, dx: float, dy: float):
    c, r = np.meshgrid(np.arange(C), np.arange(R))
    return (c - ac) * dx, (ar - r) * dy


def _sample(ground, x, y):
    """Bilinear sample of a Ground-like object at arrays of local metres."""
    from scipy.ndimage import map_coordinates
    c = np.clip(ground.ac + x / ground.dx, 0, ground.cols - 1.001)
    r = np.clip(ground.ar - y / ground.dy, 0, ground.rows - 1.001)
    return map_coordinates(ground.z, [r, c], order=1, mode="nearest")


def regional_scenario_window(lat: float, lon: float, half_m: float, kind: str, year: Optional[int], rp: int) -> Optional[dict]:
    """Depth (m) of one mapped scenario on the 3-arc-second grid around the site, north-up, with pixel local metres."""
    from .finance import flood_hazard as F
    n = int(math.ceil(half_m / M_LAT))
    kx = M_LAT * math.cos(math.radians(lat))
    for attempt in range(3):
        try:
            return _window(lat, lon, half_m, kind, year, rp, n, kx, F)
        except Exception:                                      # noqa: BLE001
            if attempt == 2:
                raise
            time.sleep(2.0 * (attempt + 1))


_WIN = {}


def _window(lat, lon, half_m, kind, year, rp, n, kx, F):
    key = (round(lat, 4), round(lon, 4), int(half_m), kind, year, rp)
    if key in _WIN:
        return _WIN[key]
    out = _window_uncached(lat, lon, half_m, kind, year, rp, n, kx, F)
    if len(_WIN) > 24:
        _WIN.clear()
    _WIN[key] = out
    return out


def _window_uncached(lat, lon, half_m, kind, year, rp, n, kx, F):
    if kind == "river":
        import rasterio
        from rasterio.windows import Window
        j = F._jrc()
        name = F._tile_name(lat, lon)
        tid = j["tiles"].get(name)
        if tid is None:
            return None
        url = f"{j['base']}RP{rp}/ID{tid}_{name}_RP{rp}_depth.tif"
        with rasterio.Env(**F._GDAL_ENV):
            with rasterio.open("/vsicurl/" + url) as ds:
                r0, c0 = ds.index(lon, lat)
                a = ds.read(1, window=Window(c0 - n, r0 - n, 2 * n + 1, 2 * n + 1), boundless=True,
                            fill_value=ds.nodata if ds.nodata is not None else 0).astype(np.float32)
                cx, cy = ds.xy(r0, c0)
                nod = ds.nodata
        if nod is not None:
            a[a == nod] = np.nan
        off_x, off_y = (cx - lon) * 111320.0 * math.cos(math.radians(lat)), (cy - lat) * 111320.0
    else:
        D = F._delt()
        i = int(round((lat + 90.0) / PX_DEG)); jx = int(round((lon + 180.0) / PX_DEG))
        w = F._chunk(year, rp, i, jx, n)
        if w is None:
            return {"depth": np.zeros((2 * n + 1, 2 * n + 1), np.float32), "x": None, "y": None, "n": n, "empty": True}
        a = np.where((w < -9000) | ~np.isfinite(w), 0.0, w).astype(np.float32)[::-1]      # rows ascend northward in the file
        off_x = (jx * PX_DEG - 180.0 - lon) * 111320.0 * math.cos(math.radians(lat))
        off_y = (i * PX_DEG - 90.0 - lat) * 111320.0
    a = np.nan_to_num(a, nan=0.0)
    c, r = np.meshgrid(np.arange(2 * n + 1), np.arange(2 * n + 1))
    return {"depth": a, "x": (c - n) * kx + off_x, "y": (n - r) * M_LAT + off_y, "n": n, "empty": False}


def regional_wse(win: dict, dem_coarse) -> dict:
    """Water-surface elevation of the wet pixels of a window: map depth + local terrain (90 m mean)."""
    if win is None or win.get("empty"):
        return {"wet": False, "x": np.array([]), "y": np.array([]), "wse": np.array([]), "depth": np.array([])}
    from scipy.ndimage import uniform_filter
    d = win["depth"]
    wet = d > 0.05
    if not wet.any():
        return {"wet": False, "x": np.array([]), "y": np.array([]), "wse": np.array([]), "depth": np.array([])}
    k = max(1, int(round(90.0 / dem_coarse.dx)))
    smooth = type("G", (), {})()
    smooth.z, smooth.dx, smooth.dy, smooth.ac, smooth.ar, smooth.rows, smooth.cols = (uniform_filter(dem_coarse.z, size=k, mode="nearest"),
        dem_coarse.dx, dem_coarse.dy, dem_coarse.ac, dem_coarse.ar, dem_coarse.rows, dem_coarse.cols)
    xs, ys = win["x"][wet], win["y"][wet]
    zc = _sample(smooth, xs, ys)
    return {"wet": True, "x": xs, "y": ys, "wse": zc + d[wet], "depth": d[wet], "dem": zc}


def rasterize_blocks(components: list, R: int, C: int, ac: float, ar: float, dx: float, dy: float) -> tuple:
    """Boolean building mask for the grid and a per-component cell mask list."""
    from PIL import Image, ImageDraw
    allm = Image.new("1", (C, R), 0)
    dr = ImageDraw.Draw(allm)
    per = {}
    for c in components:
        if c.get("removed"):
            continue
        poly = [((ac + (c["x"] + px) / dx), (ar - (c["y"] + py) / dy)) for px, py in c["footprint"]]
        if len(poly) < 3:
            continue
        dr.polygon(poly, fill=1)
        m = Image.new("1", (C, R), 0)
        ImageDraw.Draw(m).polygon(poly, fill=1)
        mm = np.array(m, dtype=bool)
        if not mm.any():                                        # sub-cell footprint: take the cell it sits in
            rr, cc = int(round(ar - c["y"] / dy)), int(round(ac + c["x"] / dx))
            if 0 <= rr < R and 0 <= cc < C:
                mm[rr, cc] = True
        per[c["id"]] = mm
    return np.array(allm, dtype=bool), per


def barrier_ring(R, C, ac, ar, dx, dy, radius_m: float) -> np.ndarray:
    x, y = _grid_xy(R, C, ac, ar, dx, dy)
    d = np.hypot(x, y)
    return np.abs(d - radius_m) <= 0.75 * max(dx, dy)


def simulate(lat: float, lon: float, ground, components: list, scen: dict, dem_coarse, add_m: float = 0.0,
             barrier_m: float = 0.0, barrier_radius_m: float = 90.0, t_end: float = 10800.0, ramp_s: float = 5400.0,
             _win: Optional[dict] = None, n_frames: int = 8, manning: float = 0.04) -> dict:
    """Simulate one mapped flood scenario over the site grid and summarise per component."""
    t0 = time.time()
    R, C = ground.rows, ground.cols
    dx, dy = ground.dx, ground.dy
    kind = scen["kind"]
    out = {"scenario": scen["id"], "label": scen.get("label"), "dx_m": round(dx, 2), "rows": R, "cols": C, "add_m": add_m,
           "barrier_m": barrier_m, "duration_h": {"rise": ramp_s / 3600.0, "total": t_end / 3600.0}}
    half = max(ground.rows, ground.cols) * dx / 2.0 + 1200.0
    win = _win if _win is not None else regional_scenario_window(lat, lon, half, kind, scen.get("year"), scen["rp"])
    wse = regional_wse(win, dem_coarse)
    out["regional"] = {"window_half_km": round(half / 1000.0, 1), "wet_pixels": int(len(wse["wse"])),
                       "source": scen.get("source")}
    if not wse["wet"]:
        out.update(dry=True, comp={cid: {"depth_at_ground_m": 0.0, "velocity_ms": 0.0, "arrival_min": None} for cid in
                                   [c["id"] for c in components if not c.get("removed")]},
                   note="The regional flood map shows no water within about %.1f km of the site for this scenario." % (half / 1000.0),
                   frames=None, elapsed_s=round(time.time() - t0, 1))
        return out
    from scipy.spatial import cKDTree
    tree = cKDTree(np.c_[wse["x"], wse["y"]])
    x, y = _grid_xy(R, C, ground.ac, ground.ar, dx, dy)
    zf = ground.z.astype(np.float64)
    block, per = rasterize_blocks(components, R, C, ground.ac, ground.ar, dx, dy)
    # stage cells: model edge, sea, deep interior wet cells
    edge = np.zeros((R, C), bool); edge[0, :] = edge[-1, :] = True; edge[:, 0] = edge[:, -1] = True
    sea = zf < 0.0
    pts = np.c_[x.ravel(), y.ravel()]
    dist, idx = tree.query(pts)
    dist, idx = dist.reshape(R, C), idx.reshape(R, C)
    deep_px = wse["depth"][idx] >= 1.0
    # robust water-surface level: median over the nearest wet pixels within 600 m (terrain mismatch makes single pixels noisy)
    kk = min(12, len(wse["wse"]))
    dd, ii = tree.query(pts, k=kk, distance_upper_bound=600.0)
    dd, ii = np.atleast_2d(dd.reshape(len(pts), -1)), np.atleast_2d(ii.reshape(len(pts), -1))
    vals = np.where(np.isfinite(dd), wse["wse"][np.minimum(ii, len(wse["wse"]) - 1)], np.nan)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        med = np.nanmedian(vals, axis=1)
    med = np.where(np.isfinite(med), med, wse["wse"][idx.ravel()]).reshape(R, C)
    level = med + add_m
    stage = ((edge & (dist <= 250.0)) | (sea & (dist <= 1500.0)) | (deep_px & (dist <= 70.0))) & ~block
    out["boundary"] = {"stage_cells": int(stage.sum()), "edge_cells": int((stage & edge).sum()), "sea_cells": int((stage & sea).sum()),
                       "level_m_p10_p50_p90": [round(float(v), 2) for v in np.percentile(level[stage], [10, 50, 90])] if stage.any() else None,
                       "nearest_wet_pixel_to_pin_m": round(float(tree.query([0.0, 0.0])[0]), 0)}
    zz = zf.copy()
    if barrier_m > 0:
        ring = barrier_ring(R, C, ground.ac, ground.ar, dx, dy, barrier_radius_m)
        zz = np.where(ring, zf + barrier_m, zf)
        stage &= ~ring
    if not stage.any():
        out.update(dry=True, comp={cid: {"depth_at_ground_m": 0.0, "velocity_ms": 0.0, "arrival_min": None} for cid in per},
                   note="Regional flood water exists in the map but none of it reaches the edge of the site model (nearest wet "
                        "pixel %.0f m from the pin), so none is routed onto the site." % out["boundary"]["nearest_wet_pixel_to_pin_m"],
                   frames=None, elapsed_s=round(time.time() - t0, 1))
        return out
    res = solve(zz, block, stage, np.where(stage, level, 0.0), dx, dy, t_end=t_end, ramp_s=ramp_s, manning=manning, n_frames=n_frames)
    hm, vm, ar_ = res["hmax"], res["vmax"], res["arrive_s"]
    inner = ~stage & ~block
    comp = {}
    for cid, m in per.items():
        from scipy.ndimage import binary_dilation
        halo = binary_dilation(m, iterations=1) & ~block | (binary_dilation(m, iterations=1) & ~m)
        halo &= ~stage
        sel = halo if halo.any() else m
        d = float(hm[sel].max()) if sel.any() else 0.0
        v = float(vm[sel][hm[sel] >= 0.9 * d].max()) if (sel.any() and d > 0.05) else 0.0
        a = float(np.nanmin(ar_[sel])) / 60.0 if (sel.any() and np.isfinite(ar_[sel]).any()) else None
        comp[cid] = {"depth_at_ground_m": round(d, 2), "velocity_ms": round(v, 2), "arrival_min": None if a is None else round(a, 0),
                     "depth_velocity_m2s": round(d * v, 2)}
    # validation against the map inside the model (non-stage land)
    map_wet = (wse["depth"][idx] > 0.05) & (dist <= 60.0)
    both = inner & (hm > 0.05) & map_wet
    out["validation"] = {
        "solver_wet_share_of_open_land": round(float((inner & (hm > 0.05)).sum() / max(inner.sum(), 1)), 3),
        "map_wet_share_of_open_land": round(float((inner & map_wet).sum() / max(inner.sum(), 1)), 3),
        "agreement_of_map_wet_cells": round(float(both.sum() / max((inner & map_wet).sum(), 1)), 3),
        "max_depth_m": round(float(hm[inner].max()) if inner.any() else 0.0, 2),
        "exceeds_map_level_by_m": round(float(np.max((zz + hm - level)[inner & (hm > 0.05)], initial=0.0)), 2),
        "mass_residual_pct": round(100.0 * abs(res["volume_end_m3"] - res["volume_added_m3"]) / max(res["volume_added_m3"], 1.0), 2),
        "steps": res["steps"], "simulated_h": round(res["t_sim"] / 3600.0, 2),
    }
    sw, mw = out["validation"]["solver_wet_share_of_open_land"], out["validation"]["map_wet_share_of_open_land"]
    if abs(sw - mw) <= 0.15:
        out["validation"]["consistency"] = "solver and 90 m map flood similar shares of the land"
    elif sw > mw:
        out["validation"]["consistency"] = ("the solver floods more land than the 90 m map: it finds low pockets and channels the map cannot see, "
                                            "or the terrain tiles sit lower than the map's terrain. Treat these depths as a low-lying-ground warning, "
                                            "not as map-validated depths.")
    else:
        out["validation"]["consistency"] = ("the solver floods less land than the 90 m map within the simulated 3 h: water has not yet filled "
                                            "the land the map shows wet; a longer event would raise the share toward the map.")
    cm = lambda a: np.round(a * 100).astype(np.int16)
    out.update(dry=False, comp=comp, frame_t_s=[round(t) for t in res["frame_t"]],
               frames=[cm(f).ravel().tolist() for f in res["frames"]], hmax_cm=cm(hm).ravel().tolist(),
               block=block.ravel().astype(np.uint8).tolist(), elapsed_s=round(time.time() - t0, 1))
    return out
