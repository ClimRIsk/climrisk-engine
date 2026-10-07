"""Turn a study result into a methods-and-results story (HTML / PDF through cri.story)."""
from __future__ import annotations

import numpy as np

from .. import story as S


def _tone(status: str) -> str:
    return "bad" if "FAILED" in status else "warn" if ("not validated" in status or "verified only" in status or "no skill" in status) else "good"


def _fmt(v, nd=3):
    if isinstance(v, float):
        if v == 0:
            return "0"
        return f"{v:.2e}" if abs(v) < 1e-3 else f"{v:.{nd}g}"
    return str(v)


def _verification(st: S.Story, rows: list):
    st.table(["Test", "What it proves", "Result", "Limit", ""], [[r["name"], r["note"], f'{_fmt(r["value"])}', f'< {_fmt(r["limit"])}', "PASS" if r["ok"] else "FAIL"] for r in rows],
             "Verification asks one question: is the model solving its own equations correctly? It says nothing yet about whether the equations describe the real place.")


def _notebook(st: S.Story, nb: list):
    st.ul([f'**{n["step"]}. {n["title"]}.** {n["detail"]}' for n in nb])


def _basis(st: S.Story, basis: list):
    if not basis:
        return
    st.p("What the engine read before choosing these equations:")
    st.ul([f'[[S:{b["key"]}]] {b["text"]}' for b in basis])
    for b in basis:
        st.source(b["key"], b["title"], b["url"])


def _sens_chart(rows, unit):
    return S.chart_bars([r["label"] for r in rows], [r["swing"] for r in rows], title=f"How far the answer moves across each input's plausible range ({unit})", unit="", horizontal=True)


def flood_story(r: dict, narrative: str = "", basis: list = None) -> dict:
    o, P = r["out"], r["plan"]
    st = S.Story("ClimRisk Model Lab · Flood", f'Flood model for {r["name"]}: a {r["return_period"]}-year rainstorm',
                 narrative or (f'The engine built a two-dimensional inundation model of {r["name"]} from first principles, tested it, and found {o["site_depth_m"]:.2f} m of water at the site, '
                               + (f'lasting {o["wet_hours"]:.1f} hours' if o["wet_hours"] >= 0.1 else "which drains almost at once") +
                               f', with {o["flooded_area_km2"]:.2f} km² of the {o["land_km2"]:.1f} km² of land deeper than 30 cm. Status: {r["status"]}.'),
                 byline="Written and run by the ClimRisk engine")
    st.callout(f'Model status: **{r["status"]}**. ' + ("Every number below comes from a model that passed its verification tests." if r["ok"] else "Verification failed; do not use these results."), _tone(r["status"]))
    st.facts([(f'{o["site_depth_m"]:.2f} m', "peak depth at the site"), (f'{o["wet_hours"]:.1f} h' if o["wet_hours"] >= 0.1 else "< 0.1 h", "time the site stays under 5 cm or more of water"), (f'{o["flooded_area_km2"]:.2f} km²', "area deeper than 30 cm (patches of 4+ cells)"),
              (f'{r["storm_mm"]:.0f} mm', f'{r["storm_hours"]:g}-hour storm rainfall'), (f'{r["runoff_mm"]:.0f} mm', "runoff after infiltration"), (f'{P["dx"]:.0f} m', "model grid")])
    st.section("question", "The question")
    st.p(f'How deep does water get at {r["name"]} ({r["lat"]:.4f}, {r["lon"]:.4f}), for how long, and how far does it spread, in the {r["return_period"]}-year storm? '
         "A doctoral student would answer it in five moves: choose the physics, gather the data, build and run the model, prove it is solved correctly, and find out which assumption the answer depends on most. The engine did the same, in that order.")
    st.section("physics", "The physics the engine chose")
    st.p("Water depth h, free surface η = z + h and unit discharge q obey mass and momentum balance: **∂h/∂t + ∇·q = r** and **∂q/∂t + g h ∇η + g n² |q| q / h^(7/3) = 0**, where r is runoff, g gravity and n the Manning roughness. "
         "The second equation keeps the slope of the water surface and bed friction and drops convective acceleration (the local-inertial form, Bates et al. 2010).")
    for a, b in r["model_choice"]:
        st.p(f"**{a}.** {b}")
    _basis(st, basis or [])
    st.section("data", "The data")
    rn = r["rain"]
    if rn.get("ok"):
        st.figure(S.chart_events(rn["years"], rn["maxima"], title="Largest daily rainfall each year at the site (mm)", ylabel="mm per day", threshold=(r["daily_T_mm"], f'{r["return_period"]}-year: {r["daily_T_mm"]:.0f} mm')),
                  f'{rn["n_years"]} annual maxima from {rn["source"]}; the dashed line is the {r["return_period"]}-year value from a fitted Gumbel distribution.')
    st.p(f'Terrain: elevation tiles resampled to a {P["dx"]:.0f} m grid; the site sits at {o["site_elev_m"]:.1f} m with {o["domain_relief_m"]:.0f} m of relief across the area. '
         f'Storm: {r["storm_hours"]:g} hours carrying 65% of the {r["return_period"]}-year daily total ({r["storm_mm"]:.0f} mm); after infiltration {r["runoff_mm"]:.0f} mm runs off.')
    st.section("result", "What the model shows")
    g = r["grids"]
    st.figure(_depth_map(g, r["name"]), "Maximum water depth during the storm over the terrain; the marker is the site.")
    ser = g["series"]
    if ser["t"]:
        st.figure(S.chart_lines(np.array(ser["t"]) / 3600, {"Depth at the site (m)": ser["probe"]}, title="Water depth at the site through the storm", ylabel="m"), "The rise follows the storm's hyetograph; the long tail is water draining across flat ground.")
    st.section("verification", "Is the model solved correctly?")
    _verification(st, r["verification"])
    if r["sensitivity"]:
        st.section("sensitivity", "What the answer depends on")
        st.figure(_sens_chart(r["sensitivity"], "m of depth at the site"), "One input varied at a time, the rest held at their base values.")
        st.table(["Input", "Range tested", "Depth at low", "Depth at base", "Depth at high", "Elasticity"],
                 [[x["label"], f'{x["low"]:.3g} – {x["high"]:.3g}', f'{x["y_low"]:.2f} m', f'{x["y_base"]:.2f} m', f'{x["y_high"]:.2f} m', f'{x["elasticity"]:+.2f}'] for x in r["sensitivity"]],
                 "Elasticity: % change in depth per % change in the input over the tested range.")
    st.section("observations", "Comparison with the real world")
    if r["validation"]["done"]:
        v = r["validation"]
        st.p(f'Manning n was tuned to {r["calibration"]["best"]:.3f} on {r["calibration"]["n_points"]} observation(s) and the model was then scored on {v["n_points"]} it had not seen: RMSE {v["rmse_m"]:.2f} m, bias {v["bias_m"]:+.2f} m, correlation {v["r"]:.2f} (source: {v["source"]}).')
    elif r["calibration"]["done"]:
        st.p(f'Manning n was tuned to {r["calibration"]["best"]:.3f}, but with too few observations to hold any back the model is calibrated, not validated.')
    else:
        st.callout("No observed flood depths or extents were supplied, so this model is verified but neither calibrated nor validated. Flood marks, gauge records or a satellite flood mask would change that.", "warn")
    if r.get("reference"):
        ref = r["reference"]
        st.p(f'Independent reference: the {ref["source"]} give {ref["depth_m"]:.2f} m at the site for the {r["return_period"]}-year river flood, against {ref["model_m"]:.2f} m here for local rainfall. They describe different flood mechanisms, so this is context rather than validation.')
    st.section("assumptions", "Assumptions and limits")
    st.ul(r["assumptions"] + r["limitations"])
    _next(st, r)
    st.section("notebook", "Lab notebook")
    _notebook(st, r["notebook"])
    return st.build()


def _next(st, r):
    if r.get("next_steps"):
        st.section("next", "What the engine would do next")
        st.ul(r["next_steps"])


def _depth_map(g, name):
    plt = S._mpl()
    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    z = g["z"]; sea = g.get("sea")
    h = np.ma.masked_less(np.where(sea, 0.0, g["h_max"]) if sea is not None else g["h_max"], 0.05)
    ax.imshow(z, cmap="Greys", alpha=0.85)
    if sea is not None and sea.any():
        ax.imshow(np.ma.masked_where(~sea, np.ones_like(z)), cmap="winter", alpha=0.35, vmin=0, vmax=2)
        ax.text(0.02, 0.97, "open water (outlet)", transform=ax.transAxes, va="top", fontsize=8, color="#0369a1")
    im = ax.imshow(h, cmap="Blues", vmin=0, vmax=max(0.5, float(np.nanmax(g["h_max"]))))
    ai, aj = g["asset"]
    ax.plot(aj, ai, marker="*", color="#ef4444", ms=15, mec="white")
    ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.04, label="max depth (m)")
    ax.set_title(f"{name}: peak flood depth", loc="left", fontsize=11, fontweight="bold")
    fig.tight_layout()
    return S._png(fig)


def air_story(r: dict, narrative: str = "", basis: list = None) -> dict:
    v, sr = r["validation"], r["series"]
    st = S.Story("ClimRisk Model Lab · Air quality", f'Air-quality model for {r["name"]}: PM2.5 and the AQI',
                 narrative or f'The engine built a physical box model of {r["name"]}, let three competing structures fight it out on the first 60% of {len(sr["obs"]) / 24:.0f} days of data, '
                              f'and tested the winner on the rest: correlation {v["r"]:.2f}, skill {v["skill_vs_clim"]:+.0%} against the mean daily cycle. Status: {r["status"]}.',
                 byline="Written and run by the ClimRisk engine")
    st.callout(f'Model status: **{r["status"]}**.', _tone(r["status"]))
    facts = [(f'{v["r"]:.2f}', "correlation on unseen hours"), (f'{v["rmse"]:.1f} µg/m³', "RMSE on unseen hours"), (f'{v["nmb"]:+.0%}', "mean bias"), (f'{v["fac2"]:.0%}', "hours within a factor of 2"),
             (f'{v["skill_vs_clim"]:+.0%}', "skill vs mean daily cycle"), (f'{v["skill_vs_pers"]:+.0%}', "skill vs yesterday's value")]
    st.facts(facts)
    st.section("question", "The question")
    st.p(f'Can physics alone, with no machine learning, reproduce the PM2.5 and the Air Quality Index of {r["name"]} from wind and the depth of the boundary layer? '
         "The engine treats the city as a box: pollution enters from emissions, is diluted by the growth of the boundary layer, and is flushed out by the wind.")
    st.section("physics", "The physics the engine chose")
    st.p("Mass balance of a well-mixed box of length L and depth H: **dC/dt = E/H − (u/L)(C − C_b) − (v_d/H)C**, with emission E, wind u, background C_b and deposition velocity v_d. "
         "With forcing held constant over each hour the equation is solved exactly, so numerical error cannot be blamed for any disagreement with the data.")
    st.p("A Gaussian plume (σy, σz from Briggs urban curves) is also available for the near field of single stacks; it is verified here but a whole city is not a single point source.")
    _basis(st, basis or [])
    st.section("selection", "Competing models, one winner")
    st.table(["Model", "Structure", "Parameters", "AIC (training)", "RMSE on unseen hours", "Correlation on unseen hours"], [[k, x["label"], len(x["params"]), f'{x["fit"]["aic"]:.0f}', f'{x["test_rmse"]:.1f}', f'{x["test_r"]:.2f}'] for k, x in r["structures"].items()],
             f'Lower AIC is better and penalises extra parameters. The model was chosen on the training AIC alone: {r["structure"]}. The last two columns are shown for transparency and played no part in the choice.')
    st.p("Fitted parameters of the winner: " + "; ".join(f"{k} = {x:.3g}" for k, x in r["params"].items()) + ". Emission E is in µg m⁻² s⁻¹, C_b in µg m⁻³, phase in hour of day.")
    st.section("result", "How it performs on data it never saw")
    st.figure(_ts_fig(r), "Observed (grey) and modelled (blue) PM2.5; the shaded part was held out of the fit.")
    st.table(["Measure", "Model", "Mean daily cycle", "Same hour yesterday"], [["RMSE (µg/m³)", f'{v["rmse"]:.1f}', f'{v["rmse_clim"]:.1f}', f'{v["rmse_pers"]:.1f}']], "A model earns its keep only by beating both naive baselines.")
    if "aqi_category_hit" in v:
        st.p(f'On {v["aqi_days"]} unseen days the model put the day in the right AQI category {v["aqi_category_hit"]:.0%} of the time; the mean daily AQI was {v["aqi_obs_mean"]:.0f} observed and {v["aqi_sim_mean"]:.0f} modelled.')
    pl = r["plume"]
    st.section("plume", "Near-source view: a 100 g/s stack")
    st.figure(S.chart_lines(np.array(pl["x_m"]) / 1000, {"Ground-level concentration (µg/m³)": pl["ugm3"]}, title=f'Gaussian plume, stability class {pl["class"]}, wind {pl["wind"]:.1f} m/s, 40 m release', ylabel="µg/m³"),
              "How a single stack's contribution falls with distance; shown to illustrate the verified dispersion module.")
    st.section("verification", "Is the model solved correctly?")
    _verification(st, r["verification"])
    st.section("sensitivity", "What the answer depends on")
    st.figure(_sens_chart(r["sensitivity"], "µg/m³ of mean PM2.5"), "One input varied at a time over its plausible range.")
    st.section("assumptions", "Assumptions and limits")
    st.ul(r["assumptions"] + r["limitations"])
    _next(st, r)
    st.section("notebook", "Lab notebook")
    _notebook(st, r["notebook"])
    return st.build()


def _ts_fig(r):
    plt = S._mpl()
    sr = r["series"]; n = len(sr["obs"]); x = np.arange(n) / 24
    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    ax.plot(x, sr["obs"], color="#94a3b8", lw=1.4, label="Observed")
    sim = np.array(sr["sim"]); cut = sr["cut"]
    ax.plot(x[:cut], sim[:cut], color="#0ea5e9", lw=1.4, label="Model (fitted)")
    ax.plot(x[cut:], sr["test_sim"], color="#f97316", lw=1.6, label="Model (unseen)")
    ax.axvspan(x[cut], x[-1], color="#f1f5f9", zorder=0)
    ax.set_xlabel("days"); ax.set_ylabel("PM2.5 µg/m³"); ax.legend(frameon=False, fontsize=9, ncol=3)
    fig.tight_layout()
    return S._png(fig)
