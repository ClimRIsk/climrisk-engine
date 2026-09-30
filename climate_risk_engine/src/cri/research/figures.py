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
