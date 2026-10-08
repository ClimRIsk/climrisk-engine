"""Rain-triggered shallow landslide susceptibility: infinite-slope stability on the DEM with a rainfall-driven pore-pressure term."""
from __future__ import annotations

import math

import numpy as np

from . import hydrology, plots
from .generic_report import Notebook

G_W = 9.81


def factor_of_safety(theta, c_kpa=5.0, phi_deg=30.0, gamma=18.0, z=2.0, m=0.0):
    """Infinite-slope factor of safety (Skempton & DeLory 1957 form):
    FS = c' / (γ z sinθ cosθ) + (1 − m γw/γ) tanφ / tanθ,   m = saturated fraction of the soil depth."""
    th = np.clip(np.asarray(theta, float), math.radians(0.5), math.radians(89.0))
    return c_kpa / (gamma * z * np.sin(th) * np.cos(th)) + (1.0 - m * G_W / gamma) * math.tan(math.radians(phi_deg)) / np.tan(th)


def saturation_fraction(infiltrated_mm, depth_m=2.0, porosity=0.40):
    """Fraction of the soil column filled by the water that infiltrated (all of it assumed to stay in the column)."""
    return float(np.clip(infiltrated_mm / (porosity * depth_m * 1000.0), 0.0, 1.0))


def slope_rad(z, dx):
    gy, gx = np.gradient(z, dx)
    return np.arctan(np.hypot(gx, gy))


def auc(score_pos, score_neg) -> float:
    """Probability a landslide cell ranks above a non-landslide cell (Mann–Whitney)."""
    from scipy import stats
    a, b = np.asarray(score_pos, float), np.asarray(score_neg, float)
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    u = stats.mannwhitneyu(a, b, alternative="greater").statistic
    return float(u / (len(a) * len(b)))


def run_benchmarks() -> list:
    out = []
    phi = 30.0
    f1 = float(factor_of_safety(math.radians(phi), c_kpa=0.0, phi_deg=phi, m=0.0))
    out.append({"name": "Dry cohesionless slope is marginally stable at the friction angle", "metric": f"FS(θ = φ = 30°) = {f1:.4f} (exact 1)", "value": abs(f1 - 1), "limit": 1e-9, "ok": abs(f1 - 1) < 1e-9, "note": "With c' = 0 and m = 0, FS = tanφ / tanθ."})
    f2 = float(factor_of_safety(math.radians(20.0), c_kpa=0.0, phi_deg=30.0, gamma=18.0, m=1.0))
    hand = (1 - 9.81 / 18.0) * math.tan(math.radians(30)) / math.tan(math.radians(20))
    out.append({"name": "Fully saturated slope: hand calculation", "metric": f"FS = {f2:.4f} (hand {hand:.4f}); saturation removes {100 * (1 - f2 / (math.tan(math.radians(30)) / math.tan(math.radians(20)))):.0f} % of the margin", "value": abs(f2 - hand), "limit": 1e-9, "ok": abs(f2 - hand) < 1e-9, "note": "Pore pressure cancels about γw/γ of the frictional resistance."})
    th = np.radians(np.linspace(5, 60, 40))
    mono = bool(np.all(np.diff(factor_of_safety(th, m=0.3)) < 0))
    fm = [float(factor_of_safety(math.radians(30), m=m)) for m in np.linspace(0, 1, 11)]
    mono_m = bool(np.all(np.diff(fm) < 0))
    out.append({"name": "Steeper and wetter always means less stable", "metric": f"monotone in slope: {mono}; monotone in saturation: {mono_m}", "value": 0.0, "limit": 0.0, "ok": mono and mono_m, "note": "Basic physical monotonicity."})
    s = slope_rad(np.fromfunction(lambda i, j: 0.5 * j * 30.0, (20, 20)), 30.0)
    out.append({"name": "Slope from a DEM tilted at 0.5", "metric": f"{math.degrees(s[10, 10]):.2f}° (exact {math.degrees(math.atan(0.5)):.2f}°)", "value": abs(s[10, 10] - math.atan(0.5)), "limit": 1e-9, "ok": abs(s[10, 10] - math.atan(0.5)) < 1e-9, "note": "Finite differences are exact on a plane."})
    rng = np.random.default_rng(2)
    pos, neg = rng.normal(1.0, 1, 400), rng.normal(0.0, 1, 4000)
    a = auc(pos, neg)
    out.append({"name": "Rank score (AUC) recovers a known separation", "metric": f"AUC = {a:.3f} (theory {0.5 * (1 + math.erf(1 / 2)):.3f})", "value": abs(a - 0.760), "limit": 0.03, "ok": abs(a - 0.760) < 0.03, "note": "Two unit-variance normals one σ apart."})
    return out


def run(lat: float, lon: float, name: str = "Study site", radius_km: float = 3.0, return_period: int = 100, progress=None, dem=None, rain_fetch=None,
        c_kpa: float = 5.0, phi_deg: float = 30.0, gamma: float = 18.0, depth_m: float = 2.0, porosity: float = 0.40, cn: float = 75.0, inventory=None) -> dict:
    nb = Notebook(progress)
    nb("1", "Problem", f"Which slopes around {name} ({lat:.3f}, {lon:.3f}) fail in a {return_period}-year rainstorm, and how much of the area is that?")
    nb("2", "Choice of physics", "Shallow landslides on hillslopes are governed by the balance between the driving shear stress of the soil weight and the resisting friction and cohesion. Rain raises the pore-water pressure and cuts the frictional resistance. The infinite-slope model captures that balance in closed form, cell by cell, which is what a 30 m terrain model can support; full finite-element slope models need soil layers that no public data provide.")
    if dem is None:
        from ..connectors.terrain_tiles import get_elevation_grid
        dem = get_elevation_grid(lat, lon, radius_km=radius_km)
    z = np.asarray(dem.elevations, float)
    from scipy import ndimage as ndi
    zf = ndi.median_filter(z, size=3, mode="nearest")
    sl = slope_rad(zf, float(dem.cell_size_m))
    deg = np.degrees(sl)
    land = zf > 0.5
    nb("3", "Terrain", f"{z.shape[0]}×{z.shape[1]} cells at {dem.cell_size_m:.0f} m; median slope {np.median(deg[land]):.1f}°, 95th percentile {np.percentile(deg[land], 95):.1f}°, maximum {deg[land].max():.1f}°.")
    try:
        from .flood_study import hydrology_fetch
        am = (rain_fetch or hydrology_fetch)(lat, lon)
        mu, beta = hydrology.gumbel_fit(am["maxima_mm"])
        d_T = hydrology.gumbel_quantile(mu, beta, return_period)
        src = am["source"]
    except Exception as e:                                           # noqa: BLE001
        d_T, src = 150.0, f"generic fallback ({type(e).__name__})"
    antecedent = 1.5 * d_T                                            # a triggering storm sits on 2-3 wet days
    q = hydrology.scs_cn_runoff(antecedent, cn)
    inf = max(antecedent - q, 0.0)
    m = saturation_fraction(inf, depth_m, porosity)
    nb("4", "Rainfall to pore pressure", f"The {return_period}-year daily rain is {d_T:.0f} mm ({src}); three wet days bring {antecedent:.0f} mm, of which {inf:.0f} mm infiltrates (SCS curve number {cn:g}) and saturates {m:.0%} of a {depth_m:g} m soil column.")

    def unstable(**kw):
        a = dict(c_kpa=c_kpa, phi_deg=phi_deg, gamma=gamma, z=depth_m, m=m); a.update(kw)
        return float(np.mean((factor_of_safety(sl, **a) < 1.0)[land]))

    fs_dry = factor_of_safety(sl, c_kpa, phi_deg, gamma, depth_m, 0.0)
    fs_wet = factor_of_safety(sl, c_kpa, phi_deg, gamma, depth_m, m)
    f_dry, f_wet = float(np.mean((fs_dry < 1)[land])), float(np.mean((fs_wet < 1)[land]))
    nb("5", "Stability", f"Factor of safety below 1 on {f_dry:.1%} of land cells when dry and {f_wet:.1%} after the design storm.")
    sens = []
    for lab, kw in (("Friction angle ±5°", [dict(phi_deg=phi_deg - 5), dict(phi_deg=phi_deg + 5)]), ("Cohesion 0–10 kPa", [dict(c_kpa=0.0), dict(c_kpa=10.0)]), ("Soil depth 1–4 m", [dict(z=1.0), dict(z=4.0)]),
                    ("Saturation fraction halved / doubled", [dict(m=m * 0.5), dict(m=min(1.0, m * 2))])):
        a, b = unstable(**kw[0]), unstable(**kw[1])
        sens.append({"label": lab, "low": a, "high": b, "span": abs(b - a)})
    sens.sort(key=lambda r: -r["span"])
    bench = run_benchmarks()
    nfail = sum(1 for b in bench if not b["ok"])
    nb("6", "Verification", f"{len(bench) - nfail} of {len(bench)} tests passed." if not nfail else f"{nfail} verification test(s) FAILED.", warn=bool(nfail))
    val = None
    if inventory:
        # inventory: list of (lat, lon); grid is centred on the asset, cells are cell_size_m on a side
        dlat = dem.cell_size_m / 111320.0
        dlon = dem.cell_size_m / (111320.0 * math.cos(math.radians(lat)))
        pos_idx = set()
        for la_, lo_ in inventory:
            i = int(round(dem.asset_row - (la_ - lat) / dlat)); j = int(round(dem.asset_col + (lo_ - lon) / dlon))
            if 0 <= i < z.shape[0] and 0 <= j < z.shape[1] and land[i, j]:
                pos_idx.add((i, j))
        if len(pos_idx) >= 10:
            mask = np.zeros(z.shape, bool)
            for i, j in pos_idx:
                mask[i, j] = True
            a_ = auc(-fs_wet[mask], -fs_wet[land & ~mask])
            val = {"n": len(pos_idx), "auc": a_}
            nb("7", "Validation", f"{len(pos_idx)} mapped landslide cells vs {int((land & ~mask).sum())} others: AUC = {a_:.2f} (0.5 = no skill, 1 = perfect).", warn=a_ < 0.6)
    if val is None:
        nb("7", "Validation", "No landslide inventory was supplied or found for this site, so the susceptibility map is verified against the physics but NOT validated against real landslides.", warn=True)
    status = "FAILED verification" if nfail else "verified only (no landslide inventory to validate against)" if val is None else ("verified and validated against the inventory" if val["auc"] >= 0.6 else "verified; NO SKILL against the inventory")
    fs_show = np.where(land, np.clip(fs_wet, 0, 3), np.nan)
    from .. import story as S
    plt = S._mpl()
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    im = ax.imshow(fs_show, cmap="RdYlGn", vmin=0.5, vmax=2.0)
    ax.contour(fs_show, levels=[1.0], colors="k", linewidths=0.8)
    ax.set_xticks([]); ax.set_yticks([]); fig.colorbar(im, label="factor of safety (after design storm)", shrink=0.8)
    ax.set_title("Slope stability after the design storm (black line: FS = 1)", loc="left", fontsize=10, fontweight="bold", color=S.INK)
    fig.tight_layout()
    fig_map = S._png(fig)
    head = [(f"{f_wet:.1%}", "land cells unstable after the storm"), (f"{f_dry:.1%}", "unstable when dry"), (f"{np.percentile(deg[land], 95):.0f}°", "95th-percentile slope"), (f"{m:.0%}", "soil column saturated")]
    blocks = [("h2", "question", "The question"), ("p", f"Which hillsides near {name} are prone to rain-triggered shallow landslides, and how much does a {return_period}-year storm change that?"),
              ("h2", "physics", "The physics the engine chose"),
              ("p", "On a long uniform slope of angle θ, a soil layer of depth z slides when the driving shear stress exceeds the shear strength. The result is the infinite-slope factor of safety, **FS = c′ / (γ z sinθ cosθ) + (1 − m γw / γ) tanφ′ / tanθ**, where c′ and φ′ are the soil's cohesion and friction angle, γ its unit weight and m the fraction of the soil depth that is saturated. FS below 1 means failure."),
              ("p", "Rain enters only through m: infiltrated water fills the pore space and the pore pressure cancels part of the friction. The engine converts the design storm to infiltrated depth with the SCS curve number and divides by the pore volume of the soil column."), ("basis",),
              ("h2", "data", "The data"), ("p", f"Slope comes from a {dem.cell_size_m:.0f} m global elevation model ({z.shape[0]}×{z.shape[1]} cells, median-filtered to remove single-cell noise). Soil strength is a generic residual-soil value (c′ = {c_kpa:g} kPa, φ′ = {phi_deg:g}°, γ = {gamma:g} kN/m³, depth {depth_m:g} m): no soil map is used, which is the largest source of uncertainty below."),
              ("fig", fig_map, "Green: stable. Red: unstable. The black contour is FS = 1."),
              ("h2", "result", "How much is unstable"),
              ("table", ["Condition", "Land cells with FS < 1"], [["Dry soil", f"{f_dry:.1%}"], [f"After the {return_period}-year storm ({inf:.0f} mm infiltrated, {m:.0%} saturated)", f"{f_wet:.1%}"]], "Share of land cells in the study area."),
              ("table", ["Assumption varied", "Unstable cells (low case)", "Unstable cells (high case)"], [[r["label"], f'{r["low"]:.1%}', f'{r["high"]:.1%}'] for r in sens], "One-at-a-time, after the design storm; sorted by influence."),
              ("callout", "Not validated: there is no landslide inventory behind this map. It tells you where the physics says slopes are marginal, not where slides have occurred." if val is None else f"Validated against {val['n']} mapped landslide cells: AUC = {val['auc']:.2f}." + (" Skilful." if val["auc"] >= 0.6 else " Not better than chance: do not use the map for decisions."), "warn" if val is None else ("good" if val["auc"] >= 0.6 else "bad"))]
    steps = ["Supply a landslide inventory (point locations of past slides) and the engine will score the map with an ROC curve.", "Replace the generic soil strength with a soil map or a few borehole tests; the sensitivity table shows it matters most.", "Add runout modelling for any area flagged unstable above assets; this study stops at initiation."]
    return {"domain": "landslide", "label": "Landslide", "kind": "landslide", "name": name, "lat": lat, "lon": lon, "title": f"Landslide susceptibility at {name}: which slopes fail in a {return_period}-year storm",
            "standfirst": f"The engine applied the infinite-slope stability equation to {name}'s terrain: {f_dry:.1%} of land cells are unstable dry and {f_wet:.1%} after a {return_period}-year storm ({d_T:.0f} mm/day). No inventory of real landslides was available, so the map is verified against the physics but not validated. Status: {status}.",
            "status": status, "ok": nfail == 0, "headline": head, "blocks": blocks, "verification": bench, "sensitivity": [],
            "assumptions": [f"Generic soil: c′ = {c_kpa:g} kPa, φ′ = {phi_deg:g}°, γ = {gamma:g} kN/m³, depth {depth_m:g} m.", "Infiltrated rain stays in the soil column (no lateral drainage), a conservative assumption.", "Failure is shallow and translational; deep-seated slides are out of scope."],
            "limitations": ["30 m terrain smooths steep local slopes, which understates instability.", "Initiation only: no runout, no earthquake trigger, no vegetation root cohesion."], "next_steps": steps, "notebook": nb.rows,
            "numbers": {"frac_unstable_dry": f_dry, "frac_unstable_wet": f_wet, "saturation": m, "rain_mm": d_T, "sens": sens}, "elapsed_s": round(__import__("time").time() - nb.t0, 1)}
