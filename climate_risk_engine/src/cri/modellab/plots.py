"""Figures shared by the hazard studies (matplotlib through cri.story's house style)."""
from __future__ import annotations

import numpy as np

from .. import story as S


def return_level_plot(x, model_fn, ci: dict, unit: str, title: str, trend_fn=None) -> bytes:
    plt = S._mpl()
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    xs = np.sort(np.asarray(x, float))[::-1]
    n = len(xs)
    T_emp = (n + 1) / np.arange(1, n + 1)                                  # Weibull plotting position
    ax.scatter(T_emp, xs, s=22, color="#0f172a", zorder=3, label="Observed annual maxima")
    Ts = np.logspace(0.15, math_log10(max(120, n * 3)), 80)
    ax.plot(Ts, [model_fn(t) for t in Ts], color="#0ea5e9", lw=2, label="Fitted")
    if trend_fn:
        ax.plot(Ts, [trend_fn(t) for t in Ts], color="#f97316", lw=1.8, ls="--", label="With trend, 2050")
    if ci:
        Tk = sorted(ci)
        ax.fill_between(Tk, [ci[t][0] for t in Tk], [ci[t][1] for t in Tk], color="#0ea5e9", alpha=0.15, lw=0, label="90 % bootstrap range")
    ax.set_xscale("log"); ax.set_xlabel("return period (years)"); ax.set_ylabel(unit)
    ax.legend(frameon=False, fontsize=9)
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold", color=S.INK)
    fig.tight_layout()
    return S._png(fig)


def math_log10(v):
    import math
    return math.log10(v)


def hindcast_plot(years, x, cut: int, levels: dict, unit: str, title: str) -> bytes:
    plt = S._mpl()
    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    years = np.asarray(years); x = np.asarray(x, float)
    ax.axvspan(years[cut] - 0.5, years[-1] + 0.5, color="#f1f5f9", zorder=0)
    ax.vlines(years, x.min() * 0.98, x, color="#cbd5e1", lw=1)
    ax.scatter(years[:cut], x[:cut], color="#64748b", s=24, zorder=3, label="Fitting period")
    ax.scatter(years[cut:], x[cut:], color="#f97316", s=28, zorder=3, label="Held-out years")
    for (T, lv), c in zip(levels.items(), ("#22c55e", "#0ea5e9", "#a855f7")):
        ax.axhline(lv, color=c, lw=1.2, ls="--"); ax.text(years[0], lv, f" {T}-yr level from the fitting period", fontsize=8, va="bottom", color=c)
    ax.set_ylabel(unit); ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold", color=S.INK)
    fig.tight_layout()
    return S._png(fig)


def scatter_fit(xs, ys, fit_x, fit_y, xlabel, ylabel, title, extra=None) -> bytes:
    plt = S._mpl()
    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    ax.scatter(xs, ys, s=10, alpha=0.35, color="#64748b", label="Observations")
    ax.plot(fit_x, fit_y, color="#0ea5e9", lw=2.2, label="Fitted")
    if extra:
        for lab, (ex, ey, col) in extra.items():
            ax.plot(ex, ey, color=col, lw=1.6, ls="--", label=lab)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.legend(frameon=False, fontsize=9)
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold", color=S.INK)
    fig.tight_layout()
    return S._png(fig)


def series_fig(x, series: dict, ylabel: str, title: str, cut: int = None) -> bytes:
    plt = S._mpl()
    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    cols = ["#94a3b8", "#0ea5e9", "#f97316", "#22c55e"]
    for (k, y), c in zip(series.items(), cols):
        ax.plot(x, y, color=c, lw=1.3, label=k)
    if cut is not None:
        ax.axvspan(x[cut], x[-1], color="#f1f5f9", zorder=0)
    ax.set_ylabel(ylabel); ax.legend(frameon=False, fontsize=9, ncol=3)
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold", color=S.INK)
    fig.tight_layout()
    return S._png(fig)
