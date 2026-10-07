"""
cri/story_figs.py — the ClimRisk research-desk figure style for story reports (dark navy, gold accent, source on every image).

Same look as the climrisk.io articles: navy canvas, gold for the headline number, sky blue for the main series, the data source
stamped along the bottom of each image. No chart is drawn without a source line. All functions return PNG bytes.
"""
from __future__ import annotations

import io
import json
import math
from pathlib import Path
from typing import Optional

import numpy as np

NAVY, PANEL, GOLD, INK, MUTED, GRID, BLUE = "#0b1220", "#111c30", "#d4af37", "#e5e7eb", "#94a3b8", "#1f2a3d", "#38bdf8"
SEV = {"CRITICAL": "#ef4444", "HIGH": "#f97316", "MODERATE": "#eab308", "LOW": "#22c55e", "NEGLIGIBLE": "#64748b"}
STATE = {"dry": "#9fb3d1", "wet_floor": "#eab308", "equipment_inundated": "#ef4444", "submerged": "#b91c1c", "protected": "#22c55e"}
PALETTE = [BLUE, GOLD, "#f97316", "#22c55e", "#a855f7", "#ef4444", "#14b8a6"]


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _rc():
    import matplotlib
    return {"figure.facecolor": NAVY, "axes.facecolor": NAVY, "savefig.facecolor": NAVY, "axes.edgecolor": GRID, "axes.labelcolor": INK,
            "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
            "axes.spines.top": False, "axes.spines.right": False, "font.size": 12, "axes.titlesize": 16, "axes.titleweight": "bold",
            "axes.titlelocation": "left", "axes.prop_cycle": matplotlib.cycler(color=PALETTE), "legend.frameon": False, "font.family": "DejaVu Sans"}


def _out(fig, source: str, dpi: int = 160) -> bytes:
    if not source:
        raise ValueError("every figure needs a source line")
    fig.text(0.012, 0.012, f"Source: {source}", fontsize=8.5, color=MUTED, ha="left")
    fig.text(0.99, 0.012, "climrisk.io", fontsize=9, color=GOLD, ha="right", weight="bold")
    fig.subplots_adjust(bottom=max(fig.subplotpars.bottom, 0.14))
    b = io.BytesIO(); fig.savefig(b, format="png", dpi=dpi); _plt().close(fig)
    return b.getvalue()


def _wrap(t: str, n: int) -> str:
    import textwrap
    return "\n".join(textwrap.wrap(t, n))


# ── six numbers ──────────────────────────────────────────────────────────────

def numbers(title: str, subtitle: str, cards: list, source: str) -> bytes:
    """cards = [(value, label, source_note)] — up to six, 3×2 grid."""
    plt = _plt()
    with plt.rc_context(_rc()):
        fig = plt.figure(figsize=(12, 7.2)); ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.text(0.033, 0.94, title, fontsize=22, fontweight="bold", color=INK, va="top")
        ax.text(0.033, 0.875, subtitle, fontsize=12, color=MUTED, va="top")
        for i, (v, l, s) in enumerate(cards[:6]):
            r, c = divmod(i, 3); x = 0.033 + c * 0.318; y = 0.77 - r * 0.375
            ax.add_patch(plt.Rectangle((x, y - 0.33), 0.30, 0.33, facecolor=PANEL, edgecolor=GRID, lw=1.2))
            ax.text(x + 0.02, y - 0.07, str(v), fontsize=26 if len(str(v)) < 14 else 19, fontweight="bold", color=GOLD, va="top")
            ax.text(x + 0.02, y - 0.17, _wrap(l, 34), fontsize=12.5, color=INK, va="top", linespacing=1.3)
            ax.text(x + 0.02, y - 0.30, _wrap(s, 48), fontsize=8.8, color=MUTED, va="top")
        return _out(fig, source)


# ── scenario losses: physical (gold) + transition (blue) with P5–P95 whisker ──

def scenario_stack(labels: list, physical: list, total: list, p5: list, p95: list, title: str, source: str, unit: str = "US$ billion", scale: float = 1e9,
                   note_physical: str = "of which physical damage") -> bytes:
    plt = _plt()
    with plt.rc_context(_rc()):
        fig, ax = plt.subplots(figsize=(12, max(3.8, 0.62 * len(labels) + 1.9)))
        y = np.arange(len(labels))[::-1]
        phys = np.array(physical) / scale; tot = np.array(total) / scale
        ax.barh(y, tot, color=BLUE, height=0.55, label="Expected loss (transition + physical)")
        ax.barh(y, np.minimum(phys, tot), color=GOLD, height=0.55, label=note_physical)
        for yi, t, lo, hi in zip(y, tot, np.array(p5) / scale, np.array(p95) / scale):
            ax.plot([lo, hi], [yi, yi], color=INK, lw=1.6); ax.plot([lo, lo], [yi - .12, yi + .12], color=INK, lw=1.6); ax.plot([hi, hi], [yi - .12, yi + .12], color=INK, lw=1.6)
            ax.text(hi + max(tot) * 0.012, yi, f"${t:,.1f}bn  (P95 ${hi:,.1f}bn)" if scale == 1e9 else f"{t:,.1f}  (P95 {hi:,.1f})", va="center", fontsize=10.5, color=INK)
        ax.set_yticks(y); ax.set_yticklabels(labels); ax.set_xlim(0, max(np.array(p95) / scale) * 1.7)
        ax.set_xlabel(f"present value 2026–2050, {unit}"); ax.set_title(title); ax.legend(loc="lower right", fontsize=10)
        ax.grid(axis="y", visible=False)
        return _out(fig, source)


def bars(labels: list, values: list, title: str, source: str, unit: str = "", horizontal: bool = True, color=None, fmt: str = "{:,.3g}", highlight: Optional[int] = None,
         xlabel: str = "", size=(12, 5)) -> bytes:
    plt = _plt()
    with plt.rc_context(_rc()):
        fig, ax = plt.subplots(figsize=(size[0], max(size[1] * 0.7, 0.5 * len(labels) + 2) if horizontal else size[1]))
        cols = [GOLD if highlight == i else (color or BLUE) for i in range(len(values))]
        if horizontal:
            y = np.arange(len(labels))[::-1]; ax.barh(y, values, color=cols, height=0.6); ax.set_yticks(y); ax.set_yticklabels(labels)
            for yi, v in zip(y, values):
                ax.text(v + max(values) * 0.012, yi, fmt.format(v) + unit, va="center", fontsize=10.5, color=INK)
            ax.set_xlim(0, max(values) * 1.18); ax.grid(axis="y", visible=False)
        else:
            x = np.arange(len(labels)); ax.bar(x, values, color=cols, width=0.62); ax.set_xticks(x); ax.set_xticklabels(labels, rotation=0)
            for xi, v in zip(x, values):
                ax.text(xi, v + max(values) * 0.015, fmt.format(v) + unit, ha="center", fontsize=10, color=INK)
            ax.set_ylim(0, max(values) * 1.15); ax.grid(axis="x", visible=False)
        ax.set_title(title)
        if xlabel: (ax.set_xlabel if horizontal else ax.set_ylabel)(xlabel)
        return _out(fig, source)


def lines(x: list, series: dict, title: str, source: str, ylabel: str = "", highlight_last: bool = True, size=(12, 5.2)) -> bytes:
    plt = _plt()
    with plt.rc_context(_rc()):
        fig, ax = plt.subplots(figsize=size)
        for i, (k, y) in enumerate(series.items()):
            ax.plot(x[:len(y)], y, lw=2.6, label=k)
            if highlight_last:
                ax.text(x[len(y) - 1], y[-1], f" {y[-1]:,.0f}", color=PALETTE[i % len(PALETTE)], va="center", fontsize=10)
        ax.set_ylabel(ylabel); ax.set_title(title)
        if len(series) > 1: ax.legend(fontsize=10, loc="upper left")
        return _out(fig, source)


def events(years: list, values: list, title: str, source: str, ylabel: str = "", highlight: Optional[dict] = None, threshold: Optional[tuple] = None, xlim: Optional[tuple] = None) -> bytes:
    plt = _plt()
    with plt.rc_context(_rc()):
        fig, ax = plt.subplots(figsize=(12, 4.9))
        ax.vlines(years, 0, values, color=BLUE, lw=2.2, alpha=0.8); ax.scatter(years, values, s=34, color=BLUE, zorder=3)
        for yy, lab in (highlight or {}).items():
            if yy in years:
                v = values[list(years).index(yy)]; ax.scatter([yy], [v], s=90, color=GOLD, zorder=4)
                ax.annotate(lab, (yy, v), textcoords="offset points", xytext=(6, 8), fontsize=10.5, color=INK)
        if threshold:
            ax.axhline(threshold[0], color="#ef4444", lw=1.1, ls="--"); ax.text(min(years), threshold[0], " " + threshold[1], va="bottom", fontsize=9.5, color="#ef4444")
        if xlim: ax.set_xlim(*xlim)
        ax.set_ylabel(ylabel); ax.set_title(title); ax.set_ylim(0, max(values) * 1.18)
        return _out(fig, source)


def oni(series: list, title: str, source: str) -> bytes:
    """series = [(decimal_year, anomaly)] since 1950; shades El Niño / La Niña thresholds and marks the latest value."""
    plt = _plt()
    with plt.rc_context(_rc()):
        fig, ax = plt.subplots(figsize=(12, 4.8))
        x = np.array([s[0] for s in series]); y = np.array([s[1] for s in series])
        ax.fill_between(x, 0, y, where=y >= 0.5, color="#f97316", alpha=0.85, lw=0); ax.fill_between(x, 0, y, where=y <= -0.5, color=BLUE, alpha=0.85, lw=0)
        ax.plot(x, y, color=MUTED, lw=0.9)
        ax.axhline(0.5, color=GRID, lw=1); ax.axhline(-0.5, color=GRID, lw=1); ax.axhline(2.0, color="#ef4444", lw=1, ls="--")
        ax.text(x[0], 2.03, " +2 °C", color="#ef4444", fontsize=9.5, va="bottom")
        ax.scatter([x[-1]], [y[-1]], s=80, color=GOLD, zorder=5); ax.annotate(f"{y[-1]:+.2f} °C now", (x[-1], y[-1]), textcoords="offset points", xytext=(-90, 10), color=GOLD, fontsize=11, fontweight="bold")
        ax.set_ylabel("Oceanic Niño Index (°C)"); ax.set_title(title)
        return _out(fig, source)


# ── maps ─────────────────────────────────────────────────────────────────────

def _land():
    p = Path(__file__).resolve().parent / "api" / "world_land_110m.json"
    return json.loads(p.read_text()) if p.exists() else []


def asset_map(assets: list, title: str, source: str, mark: Optional[tuple] = None, pad: float = 7.0, legend_title: str = "Physical screening level", size_by: str = "emissions_t", size: Optional[tuple] = None, legend: bool = True) -> bytes:
    """assets = [{lat, lon, screen|screen_level, emissions_t?, name?}] on a land map; mark=(lat, lon, label) draws a gold ring."""
    plt = _plt()
    with plt.rc_context(_rc()):
        lats = np.array([a["lat"] for a in assets] + ([mark[0]] if mark else [])); lons = np.array([a["lon"] for a in assets] + ([mark[1]] if mark else []))
        w, e = lons.min() - pad, lons.max() + pad; s, n = max(-60, lats.min() - pad * 0.7), min(80, lats.max() + pad * 0.7)
        fig, ax = plt.subplots(figsize=size or (12, max(4.6, min(8.6, 12 * (n - s) / max(e - w, 1e-6) + 1.3))))
        for ring in _land():
            xs, ys = zip(*ring); ax.fill(xs, ys, color="#16233a", edgecolor="#26395a", linewidth=0.6)
        em = np.array([(a.get(size_by) or 1e5) for a in assets], float)
        sz = 40 + 360 * (em / max(em.max(), 1.0)) ** 0.6 if len(em) else []
        for lv in ["NEGLIGIBLE", "LOW", "MODERATE", "HIGH", "CRITICAL"]:
            ii = [i for i, a in enumerate(assets) if (a.get("screen") or a.get("screen_level")) == lv]
            if ii:
                ax.scatter([assets[i]["lon"] for i in ii], [assets[i]["lat"] for i in ii], s=[sz[i] for i in ii], color=SEV[lv], edgecolor=NAVY, linewidth=0.8, label=f"{lv.title()} ({len(ii)})", zorder=3, alpha=0.95)
        if mark:
            ax.scatter([mark[1]], [mark[0]], s=520, facecolors="none", edgecolors=GOLD, linewidths=2.6, zorder=5)
            ax.annotate(mark[2], (mark[1], mark[0]), textcoords="offset points", xytext=(14, 12), color=GOLD, fontsize=11, fontweight="bold")
        ax.set_xlim(w, e); ax.set_ylim(s, n); ax.set_aspect("equal" if not size else "auto"); ax.grid(False); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(title)
        if legend:
            ax.legend(loc="lower left", fontsize=10, title=legend_title, title_fontsize=10)
        return _out(fig, source)


# ── the site itself ──────────────────────────────────────────────────────────

def site_plan(twin: dict, sim: Optional[dict], states: dict, title: str, source: str, show_depth: bool = True, label_n: int = 7, barrier_m: float = 0.0) -> bytes:
    """Plan view of the modelled site: terrain shading, solver maximum depth, every structure coloured by its flood state, worst labelled."""
    from matplotlib.patches import Polygon
    plt = _plt()
    with plt.rc_context(_rc()):
        T = twin["terrain"]; R, C, dx, dy = T["rows"], T["cols"], T["dx_m"], T["dy_m"]
        z = np.array(T["z"], float).reshape(R, C); x0, y0 = T["x0_m"], T["y0_m"]
        ext = [x0, x0 + C * dx, y0 - R * dy, y0]
        comps = [c for c in twin["components"] if not c.get("removed") and c.get("on_site", True)]
        if any(c["kind"] != "building" for c in comps):
            comps_main = [c for c in comps if c["kind"] != "building"]; ctx = [c for c in comps if c["kind"] == "building"]
        else:
            comps_main, ctx = comps, []
        xs = [c["x"] for c in comps_main] or [0]; ys = [c["y"] for c in comps_main] or [0]
        span = max(max(xs) - min(xs), max(ys) - min(ys), 120.0) * 0.5 + 80.0; cx, cy = (max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2
        fig, ax = plt.subplots(figsize=(12, 7.4))
        ax.imshow(z, extent=ext, cmap="Greys_r", alpha=0.55, zorder=0, interpolation="bilinear")
        ax.contour(np.linspace(x0 + dx / 2, x0 + C * dx - dx / 2, C), np.linspace(y0 - dy / 2, y0 - R * dy + dy / 2, R), z, levels=8, colors="#475569", linewidths=0.5, alpha=0.7)
        if show_depth and sim and sim.get("hmax_cm") and not sim.get("dry"):
            h = np.array(sim["hmax_cm"], float).reshape(sim["rows"], sim["cols"]) / 100.0
            hm = np.ma.masked_where(h < 0.03, h)
            im = ax.imshow(hm, extent=ext, cmap="Blues", vmin=0, vmax=max(1.0, float(np.nanpercentile(h[h > 0.03], 98)) if (h > 0.03).any() else 1.0), alpha=0.8, zorder=1, interpolation="nearest")
            cb = fig.colorbar(im, ax=ax, shrink=0.65, pad=0.01); cb.set_label("maximum water depth (m), 2D solver", color=INK); cb.ax.yaxis.set_tick_params(color=MUTED)
        for c in ctx:
            ax.add_patch(Polygon([(c["x"] + px, c["y"] + py) for px, py in c["footprint"]], closed=True, facecolor="#334155", edgecolor="#475569", lw=0.6, alpha=0.6, zorder=2))
        order = sorted(comps_main, key=lambda c: -(({"submerged": 3, "equipment_inundated": 2, "wet_floor": 1}.get(states.get(c["id"]), 0)) * 100 + c.get("height_m", 0)))
        for c in comps_main:
            col = STATE.get(states.get(c["id"], "dry"), STATE["dry"])
            ax.add_patch(Polygon([(c["x"] + px, c["y"] + py) for px, py in c["footprint"]], closed=True, facecolor=col, edgecolor="#0b1220", lw=1.0, zorder=3, alpha=0.95))
        labelled = 0
        for c in order:
            if labelled >= label_n:
                break
            if states.get(c["id"], "dry") == "dry" and labelled >= 3:
                continue
            ax.annotate(c["name"][:22], (c["x"], c["y"]), textcoords="offset points", xytext=(7, 7), fontsize=9, color=INK, zorder=6,
                        bbox=dict(boxstyle="round,pad=0.15", fc=NAVY, ec="none", alpha=0.7)); labelled += 1
        if barrier_m > 0:
            r = max([math.hypot(c["x"], c["y"]) for c in comps_main] or [40.0]) + 25.0
            th = np.linspace(0, 2 * np.pi, 200); ax.plot(r * np.cos(th), r * np.sin(th), color="#22c55e", lw=2.4, ls="--", zorder=5, label=f"flood barrier {barrier_m:.1f} m")
        ax.set_xlim(cx - span, cx + span); ax.set_ylim(cy - span * 0.62, cy + span * 0.62); ax.set_aspect("equal"); ax.grid(False)
        ax.set_xlabel("metres east of the pin"); ax.set_ylabel("metres north")
        from matplotlib.lines import Line2D
        hand = [Line2D([0], [0], marker="s", ls="", color=STATE[k], label=lab, markersize=10) for k, lab in
                (("dry", "dry"), ("wet_floor", "water on floor"), ("equipment_inundated", "equipment under water"), ("protected", "protected"))]
        if barrier_m > 0:
            hand.append(Line2D([0], [0], color="#22c55e", lw=2.4, ls="--", label=f"flood barrier {barrier_m:.1f} m"))
        fig.subplots_adjust(bottom=0.24)
        ax.legend(handles=hand, loc="upper center", bbox_to_anchor=(0.5, -0.1), fontsize=9.5, ncol=5, facecolor=NAVY, framealpha=0.0)
        ax.set_title(title)
        return _out(fig, source)


def gust_vs_design(rows: list, title: str, source: str) -> bytes:
    """rows = [(name, gust_ms, design_ms)] — bar of gust at structure height against its design wind."""
    plt = _plt()
    with plt.rc_context(_rc()):
        rows = rows[:12]
        fig, ax = plt.subplots(figsize=(12, 0.5 * len(rows) + 2.2))
        y = np.arange(len(rows))[::-1]
        ax.barh(y + 0.18, [r[2] for r in rows], height=0.32, color="#475569", label="assumed design gust")
        ax.barh(y - 0.18, [r[1] for r in rows], height=0.32, color=[("#ef4444" if r[1] > r[2] else BLUE) for r in rows], label="gust at the structure's height")
        for yi, r in zip(y, rows):
            ax.text(max(r[1], r[2]) + 1, yi, f"{r[1]:.0f} vs {r[2]:.0f} m/s  ({r[1] / r[2]:.2f}×)", va="center", fontsize=10, color=INK)
        ax.set_yticks(y); ax.set_yticklabels([r[0][:28] for r in rows]); ax.set_xlim(0, max(max(r[1], r[2]) for r in rows) * 1.35)
        fig.subplots_adjust(bottom=0.24)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, fontsize=9.5); ax.set_title(title); ax.grid(axis="y", visible=False)
        return _out(fig, source)


def distribution(edges: list, counts: list, title: str, source: str, marks: dict, unit: str = "US$ million", scale: float = 1e6) -> bytes:
    plt = _plt()
    with plt.rc_context(_rc()):
        e = np.array(edges) / scale; c = np.array(counts, float)
        fig, ax = plt.subplots(figsize=(12, 4.8))
        ax.bar((e[:-1] + e[1:]) / 2, c / c.sum(), width=(e[1] - e[0]) * 0.92, color=BLUE, alpha=0.9)
        for i, (k, v) in enumerate(marks.items()):
            ax.axvline(v / scale, color=[GOLD, "#f97316", "#ef4444"][i % 3], lw=2, ls="--"); ax.text(v / scale, ax.get_ylim()[1] * (0.95 - 0.1 * i), f" {k}: {v / scale:,.0f}", color=INK, fontsize=10.5)
        ax.set_xlabel(f"total loss, PV 2026–2050, {unit}"); ax.set_ylabel("share of Monte Carlo draws"); ax.set_title(title)
        return _out(fig, source)


def gauge_bar(items: list, title: str, source: str) -> bytes:
    """items = [(label, value, total, colour?)] — stacked single bar split (e.g. scopes)."""
    plt = _plt()
    with plt.rc_context(_rc()):
        fig, ax = plt.subplots(figsize=(12, 2.9))
        left = 0.0; tot = sum(i[1] for i in items)
        for k, (lab, v, col) in enumerate([(i[0], i[1], i[2] if len(i) > 2 else PALETTE[k % len(PALETTE)]) for k, i in enumerate(items)]):
            ax.barh([0], [v], left=left, color=col, height=0.5);
            if v / tot > 0.06: ax.text(left + v / 2, 0, f"{lab}\n{v:,.0f} t ({100 * v / tot:.0f}%)", ha="center", va="center", fontsize=10.5, color=NAVY, fontweight="bold")
            left += v
        ax.set_xlim(0, tot); ax.set_yticks([]); ax.grid(False); ax.set_xlabel("t CO₂e"); ax.set_title(title)
        return _out(fig, source)


# ── covers ───────────────────────────────────────────────────────────────────

def cover(title: str, subtitle: str, credit: str, image_png: Optional[bytes] = None, tag: str = "", crop: bool = True) -> bytes:
    """1200×630: title panel on the left, a real image (satellite / chart / map) on the right. crop=False fits a chart without trimming it."""
    from PIL import Image
    plt = _plt()
    with plt.rc_context(_rc()):
        fig = plt.figure(figsize=(12, 6.3)); ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.add_patch(plt.Rectangle((0, 0), 1, 1, color=NAVY))
        if image_png:
            im = Image.open(io.BytesIO(image_png)).convert("RGB"); w, h = im.size
            pw, ph = 0.52 * 1200.0, 630.0
            if crop:
                tgt = pw / ph
                if w / h > tgt:
                    nw = int(h * tgt); im = im.crop(((w - nw) // 2, 0, (w - nw) // 2 + nw, h))
                else:
                    nh = int(w / tgt); im = im.crop((0, 0, w, nh))
                ax.imshow(np.asarray(im), extent=[0.48, 1.0, 0, 1], aspect="auto", zorder=1)
            else:
                sc = min(pw / w, ph / h); iw, ih = w * sc / 1200.0, h * sc / 630.0
                x0 = 0.48 + (0.52 - iw) / 2; y0 = (1 - ih) / 2
                ax.imshow(np.asarray(im), extent=[x0, x0 + iw, y0, y0 + ih], aspect="auto", zorder=1)
        if tag:
            ax.text(0.04, 0.92, tag.upper(), fontsize=11, color=GOLD, fontweight="bold", va="top")
        ax.text(0.04, 0.80, _wrap(title, 22), fontsize=28 if len(title) > 70 else 30, fontweight="bold", color=INK, va="top", linespacing=1.15)
        ax.text(0.04, 0.30 if len(title) > 44 else 0.45, _wrap(subtitle, 38), fontsize=15, color=GOLD, va="top", linespacing=1.3)
        ax.text(0.04, 0.05, _wrap(credit, 52), fontsize=8.5, color=MUTED, va="bottom")
        b = io.BytesIO(); fig.savefig(b, format="png", dpi=100); plt.close(fig)
        return b.getvalue()
