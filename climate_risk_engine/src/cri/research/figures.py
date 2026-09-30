"""
cri/research/figures.py — factual figures for Research Desk articles.

Every image is made from real data and carries its source on the image itself:
  • satellite_image  Copernicus Sentinel-2 L2A true colour (AWS Earth Search), a real
                     scene for a place and date window — e.g. before / after a flood
  • assets_map       a company's plants from the asset register on a Natural Earth map
  • chart helpers    house style for matplotlib charts drawn from downloaded data
                     (save() stamps the source line; no "representative" numbers)
No generative / AI imagery is produced or used.
"""
from __future__ import annotations

import datetime as _dt
import json
import urllib.request
from pathlib import Path
from typing import Optional

import numpy as np

NAVY, GOLD, INK, MUTED, GRID = "#0b1220", "#d4af37", "#e5e7eb", "#94a3b8", "#1f2a3d"
SEV = {"CRITICAL": "#ef4444", "HIGH": "#f97316", "MODERATE": "#eab308", "LOW": "#22c55e", "NEGLIGIBLE": "#64748b"}
PALETTE = ["#38bdf8", "#d4af37", "#f97316", "#22c55e", "#a855f7", "#ef4444", "#14b8a6"]


def house_style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.facecolor": NAVY, "axes.facecolor": NAVY, "savefig.facecolor": NAVY,
        "axes.edgecolor": GRID, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
        "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False, "font.size": 12,
        "axes.titlesize": 16, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.prop_cycle": matplotlib.cycler(color=PALETTE), "legend.frameon": False,
    })
    return plt


def save(fig, path: str | Path, source: str, note: Optional[str] = None, dpi: int = 200) -> str:
    """Save with the data source (required) and ClimRisk mark along the bottom."""
    if not source or "representative" in source.lower():
        raise ValueError("Every chart needs a real data source line (no representative/illustrative data).")
    fig.text(0.01, 0.012, f"Source: {source}" + (f"  ·  {note}" if note else ""), fontsize=8.5, color=MUTED, ha="left")
    fig.text(0.99, 0.012, "climrisk.io", fontsize=9, color=GOLD, ha="right", weight="bold")
    fig.subplots_adjust(bottom=max(fig.subplotpars.bottom, 0.14))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    return str(path)


# ── Sentinel-2 true colour ────────────────────────────────────────────────────

def _search(lat, lon, start, end, max_cloud, limit=20):
    from ..connectors.sentinel2_pixels import COLLECTION, STAC_URL
    payload = {"collections": [COLLECTION], "intersects": {"type": "Point", "coordinates": [lon, lat]},
               "datetime": f"{start}T00:00:00Z/{end}T23:59:59Z", "query": {"eo:cloud_cover": {"lt": max_cloud}},
               "sortby": [{"field": "properties.eo:cloud_cover", "direction": "asc"}], "limit": limit}
    req = urllib.request.Request(STAC_URL, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json", "User-Agent": "ClimRisk-Research/1.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read()).get("features", [])


def satellite_image(lat: float, lon: float, out: str | Path, start: str, end: str, half_km: float = 4.0,
                    max_cloud: float = 20.0, title: Optional[str] = None, marker: bool = True) -> dict:
    """Least-cloudy real Sentinel-2 scene in [start, end] around a point, saved as PNG with credit."""
    from PIL import Image, ImageDraw, ImageFont
    from ..connectors.sentinel2_pixels import _read_band, asset_bbox

    feats = _search(lat, lon, start, end, max_cloud)
    if not feats:
        return {"ok": False, "reason": f"no Sentinel-2 scene under {max_cloud}% cloud between {start} and {end}"}
    f = feats[0]
    bbox = asset_bbox(lat, lon, half_km)
    px = int(min(1400, max(600, half_km * 2000 / 10)))           # ~10 m pixels, capped
    rgb = _read_band(f["assets"]["visual"]["href"], bbox, (px, px), count=3)
    rgb = np.nan_to_num(np.moveaxis(rgb, 0, -1), nan=0).clip(0, 255).astype(np.uint8)
    img = Image.fromarray(rgb, "RGB")
    date = f["properties"]["datetime"][:10]
    cloud = f["properties"].get("eo:cloud_cover")
    d = ImageDraw.Draw(img)
    if marker:
        c = px // 2
        d.ellipse([c - 14, c - 14, c + 14, c + 14], outline=(212, 175, 55), width=4)
    band = Image.new("RGB", (px, 64), (11, 18, 32))
    bd = ImageDraw.Draw(band)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 18)
        small = ImageFont.truetype("DejaVuSans.ttf", 14)
    except OSError:
        font = small = ImageFont.load_default()
    bd.text((14, 8), title or f"{lat:.3f}, {lon:.3f}", fill=(229, 231, 235), font=font)
    bd.text((14, 36), f"Copernicus Sentinel-2 L2A true colour · {date} · cloud {cloud:.0f}% · "
                      f"{2 * half_km:.0f} km across · processed by ClimRisk", fill=(148, 163, 184), font=small)
    canvas = Image.new("RGB", (px, px + 64))
    canvas.paste(img, (0, 0)); canvas.paste(band, (0, px))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out, optimize=True)
    return {"ok": True, "path": str(out), "scene": f["id"], "date": date, "cloud": cloud,
            "credit": f"Contains modified Copernicus Sentinel-2 data ({date[:4]}), via AWS Earth Search"}


# ── Company assets map ────────────────────────────────────────────────────────

def assets_map(company_keys: list[str], out: str | Path, title: str) -> dict:
    from ..data.asset_register import company_exposure
    plt = house_style()
    e = company_exposure(company_keys)
    if not e.get("found"):
        return {"ok": False, "reason": "company not in register"}
    land = json.loads((Path(__file__).resolve().parents[1] / "api" / "world_land_110m.json").read_text())
    A = e["assets"]
    lats, lons = np.array([a["lat"] for a in A]), np.array([a["lon"] for a in A])
    w, east = lons.min() - 8, lons.max() + 8
    s, n = max(-60, lats.min() - 6), min(80, lats.max() + 6)
    aspect = (n - s) / max(east - w, 1e-6)
    fig, ax = plt.subplots(figsize=(12, max(4.5, min(9.0, 12 * aspect + 1.4))))
    for ring in land:
        xs, ys = zip(*ring)
        ax.fill(xs, ys, color="#16233a", edgecolor="#26395a", linewidth=0.6)
    for lv in ["NEGLIGIBLE", "LOW", "MODERATE", "HIGH", "CRITICAL"]:
        sel = [a for a in A if a["screen_level"] == lv]
        if sel:
            ax.scatter([a["lon"] for a in sel], [a["lat"] for a in sel], s=46, color=SEV[lv], edgecolor=NAVY,
                       linewidth=0.8, label=f"{lv.title()} ({len(sel)})", zorder=3)
    ax.set_xlim(w, east); ax.set_ylim(s, n); ax.set_aspect("equal"); ax.grid(False)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title)
    ax.legend(loc="lower left", fontsize=10, title="Physical screening level", title_fontsize=10)
    path = save(fig, out, "Climate TRACE v6 / EPA GHGRP asset register; screening: IBTrACS cyclone model + "
                          "terrain elevation (ClimRisk engine); Natural Earth")
    plt.close(fig)
    return {"ok": True, "path": path, "n_assets": len(A)}


# ── Sentinel-1 radar flood map (sees through cloud) ──────────────────────────

def radar_flood_image(lat: float, lon: float, out: str | Path, before: tuple[str, str], after: tuple[str, str],
                      half_km: float = 5.0, title: Optional[str] = None, water_db: float = -18.0) -> dict:
    """
    Real Sentinel-1 RTC (VV) scenes before and after an event, same orbit direction.
    Left: before; right: after, with newly water-covered land (VV < −18 dB after, not before)
    in blue. Radar sees through the cloud that hides floods from optical satellites.
    """
    from PIL import Image, ImageDraw, ImageFont
    from scipy import ndimage
    from ..connectors.sentinel1_sar import STAC, _sas
    from ..connectors.sentinel2_pixels import _read_band, asset_bbox

    def search(s, e):
        body = {"collections": ["sentinel-1-rtc"], "intersects": {"type": "Point", "coordinates": [lon, lat]},
                "datetime": f"{s}T00:00:00Z/{e}T23:59:59Z", "limit": 50}
        req = urllib.request.Request(STAC, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as r:
            return [f for f in json.loads(r.read()).get("features", []) if "vv" in f.get("assets", {})]

    fa = sorted(search(*after), key=lambda f: f["properties"]["datetime"])
    if not fa:
        return {"ok": False, "reason": f"no Sentinel-1 scene between {after[0]} and {after[1]}"}
    a_scene = fa[0]                                    # first pass after the event
    orbit = a_scene["properties"].get("sat:orbit_state")
    fb = [f for f in search(*before) if f["properties"].get("sat:orbit_state") == orbit] or search(*before)
    if not fb:
        return {"ok": False, "reason": f"no Sentinel-1 scene between {before[0]} and {before[1]}"}
    fb = sorted(fb, key=lambda f: f["properties"]["datetime"])
    b_scene = fb[-1]                                   # last pass before (shown)
    b_ref = fb[-4:]                                    # up to 4 passes define "normal" water
    bbox = asset_bbox(lat, lon, half_km)
    px = int(min(900, max(400, half_km * 2000 / 20)))
    tok = _sas()

    def read(f):
        href = f["assets"]["vv"]["href"]
        a = _read_band(f"{href}{'&' if '?' in href else '?'}{tok}", bbox, (px, px), "average")[0]
        a = ndimage.median_filter(np.nan_to_num(a, nan=np.nanmedian(a)), size=3)
        return 10 * np.log10(np.maximum(a, 1e-6))

    db_b, db_a = read(b_scene), read(a_scene)
    # Permanent water: dark in ANY pre-event pass (wind can roughen a lake on one date) or
    # mapped as water by ESA WorldCover — so lakes and rivers are never counted as flooding.
    before_min = np.minimum.reduce([db_b] + [read(f) for f in b_ref if f is not b_scene])
    perm_mask = before_min < water_db
    try:
        from ..connectors.worldcover import lc_grid
        lc, _ = lc_grid(lat, lon, half_km)
        ri = np.minimum((np.arange(px) * lc.shape[0] / px).astype(int), lc.shape[0] - 1)
        ci = np.minimum((np.arange(px) * lc.shape[1] / px).astype(int), lc.shape[1] - 1)
        wc = lc[np.ix_(ri, ci)]
        perm_mask |= ndimage.binary_dilation(wc == 80, iterations=2)
    except Exception:                                  # noqa: BLE001
        pass
    new_water = (db_a < water_db) & ~perm_mask
    perm = (db_a < water_db) & perm_mask

    def grey(db):
        g = np.clip((db + 25) / 25, 0, 1) ** 0.9 * 255
        return np.stack([g, g, g], -1).astype(np.uint8)

    left, right = grey(db_b), grey(db_a)
    right[perm] = (40, 60, 90)
    right[new_water] = (56, 189, 248)
    gap = 10
    canvas = Image.new("RGB", (2 * px + gap, px + 78), (11, 18, 32))
    canvas.paste(Image.fromarray(left), (0, 0)); canvas.paste(Image.fromarray(right), (px + gap, 0))
    d = ImageDraw.Draw(canvas)
    try:
        font, small = ImageFont.truetype("DejaVuSans.ttf", 17), ImageFont.truetype("DejaVuSans.ttf", 13)
    except OSError:
        font = small = ImageFont.load_default()
    bd, ad = b_scene["properties"]["datetime"][:10], a_scene["properties"]["datetime"][:10]
    for x, lab in ((8, f"Before · {bd}"), (px + gap + 8, f"After · {ad}  (blue = flooded land · dark blue = normal water)")):
        d.rectangle([x - 4, 6, x + 8 * len(lab) + 6, 30], fill=(11, 18, 32))
        d.text((x, 9), lab, fill=(229, 231, 235), font=small)
    d.text((12, px + 10), title or f"{lat:.3f}, {lon:.3f}", fill=(229, 231, 235), font=font)
    d.text((12, px + 40), f"Copernicus Sentinel-1 radar (VV, RTC, Microsoft Planetary Computer) · water = backscatter < {water_db:.0f} dB · "
                          f"{2 * half_km:.0f} km across · processed by ClimRisk", fill=(148, 163, 184), font=small)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out, optimize=True)
    cell_km2 = (2 * half_km / px) ** 2
    return {"ok": True, "path": str(out), "before": b_scene["id"], "after": a_scene["id"], "orbit": orbit,
            "reference_passes": [f["id"] for f in b_ref],
            "new_water_km2": round(float(new_water.sum() * cell_km2), 2),
            "credit": f"Contains modified Copernicus Sentinel-1 data ({ad[:4]}), RTC via Microsoft Planetary Computer"}
