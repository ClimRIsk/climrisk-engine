"""Uncertainty ledger: every source of doubt behind a site's number, what is known about its size, and what is not.  (12 Nov 2026)

The engine's outputs were single numbers with caveats attached. A ledger puts the doubts side by side so a reader can see which one dominates and which have never been
measured. Each entry has a TYPE (the usual three from climate modelling plus data and parameter doubts), a STATUS, and where quantified a low-to-high range.

Types      internal_variability  chance sequence of weather in the record or in a model run
           scenario              which emissions pathway / warming level
           structural            different models or data products disagree (model form, source of the data)
           parameter             an input that is assumed (curve number, roughness, drainage)
           sampling              finite record length (confidence interval of a fitted return level)
Statuses   quantified            a range computed from data in the engine (the source and method are in the entry)
           assumed_range         a range somebody chose; shown, never presented as measured
           not_assessed          nothing in the engine measures it; the reason says what would be needed

Nothing here invents a number. The `relative_range` of an entry is (high - low) / mid of the QUANTITY THAT ENTRY IS ABOUT (rain in mm, a change ratio, a damage fraction). They are
ranges of different quantities, so the ranking says which doubt is relatively widest, not how they combine into one loss range; combining needs the loss model run per case, which
is stated in `combination`.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

KINDS = ("internal_variability", "scenario", "structural", "parameter", "sampling")


def _rel(lo: float, hi: float) -> Optional[float]:
    mid = (lo + hi) / 2.0
    return None if not math.isfinite(mid) or mid == 0 else round(abs(hi - lo) / abs(mid), 3)


def entry(key: str, label: str, kind: str, status: str, *, low=None, high=None, central=None, unit: str = "", source: str = "", method: str = "", note: str = "", reason: str = "") -> dict:
    assert kind in KINDS and status in ("quantified", "assumed_range", "not_assessed"), (kind, status)
    if status == "not_assessed":
        low = high = central = None
    out = {"key": key, "label": label, "kind": kind, "status": status, "low": low, "central": central, "high": high, "unit": unit,
           "relative_range": None if low is None or high is None else _rel(float(low), float(high)), "source": source, "method": method, "note": note}
    if status == "not_assessed":
        out["reason"] = reason
    return out


def rain_sources(design: Optional[dict]) -> dict:
    """Disagreement between the design-rain products (ERA5, NASA POWER, nearest gauge) at the chosen return period, from `finance.rain_design.design`."""
    if not design or not design.get("available"):
        return entry("rain_sources", "Design rainfall: which data product", "structural", "not_assessed", reason="design rainfall unavailable: " + str((design or {}).get("reason", "not run"))[:120])
    by = {n: s["d1_mm"] for n, s in design["by_source"].items() if "d1_mm" in s}
    if len(by) < 2:
        return entry("rain_sources", "Design rainfall: which data product", "structural", "not_assessed", reason="only one rainfall source could be read (" + ", ".join(by) + "); a second is needed to see disagreement")
    lo, hi = min(by.values()), max(by.values())
    gauge = design["by_source"].get("gauge")
    return entry("rain_sources", "Design rainfall: which data product", "structural", "quantified", low=round(lo, 1), high=round(hi, 1), central=design.get("d1_mm"), unit="mm in a day",
                 source="ERA5, NASA POWER" + (f", gauge {gauge.get('km')} km away" if gauge else ""), method="Gumbel return level of annual maxima for each product",
                 note=f"Products: " + "; ".join(f"{n} {v:.0f} mm" for n, v in sorted(by.items())) + ". The design value is the largest. A gauge is a point; the grids are area averages, so neither is 'truth'.")


def scenario_spread(rain: Optional[dict], pathway: str = "") -> dict:
    """Spread of the CMIP6 change in extreme daily rain across models: {p10, median, p90} as ratios (1.12 = 12 % heavier)."""
    if not rain or any(k not in rain for k in ("p10", "median", "p90")):
        return entry("scenario_models", "Future change in extreme rain (climate models)", "scenario", "not_assessed", reason="CMIP6 change not read for this place and pathway (needs the climate-scenario module)")
    return entry("scenario_models", "Future change in extreme rain (climate models)", "structural", "quantified", low=round(rain["p10"], 3), central=round(rain["median"], 3), high=round(rain["p90"], 3), unit="ratio to today",
                 source="CMIP6 ensemble (World Bank CCKP)", method="p10 / median / p90 across models" + (f", {pathway}" if pathway else ""),
                 note="The spread is across models for one pathway; the choice of pathway is a separate (scenario) doubt and must be shown by running each.")


def return_level_sampling(levels: Optional[tuple], central: Optional[float] = None, unit: str = "mm", T: int = 100) -> dict:
    """5 to 95 % bootstrap interval of a fitted return level (`modellab.evt.bootstrap_levels`): the doubt from having only a few decades of record."""
    if not levels:
        return entry("return_level_sampling", f"1-in-{T} level: length of record", "sampling", "not_assessed", reason="no bootstrap interval was computed for this series")
    lo, hi = levels
    return entry("return_level_sampling", f"1-in-{T} level: length of record", "sampling", "quantified", low=round(float(lo), 1), high=round(float(hi), 1), central=central, unit=unit,
                 source="bootstrap of the annual maxima", method="Gumbel refit on 200 resamples, 5th to 95th percentile",
                 note="A 30-year record says little about a 100-year event; this interval is the honest size of that.")


def damage_curve_transfer(depth_m: float, continent: str, asset_class: str = "industrial") -> dict:
    """Damage fraction at `depth_m` from the site's own JRC continental curve against the range across all continents published for that class.

    This is the spread you would get by applying another continent's curve, a proxy for how well a curve transfers. It is NOT the published confidence band of the curve."""
    from .finance import vulnerability as V
    try:
        own, basis = V.jrc_curve(continent, asset_class)
        tab = {k: v for k, v in V._JRC[asset_class].items() if not (asset_class == "industrial" and k == "oceania")}
    except Exception as e:                                           # noqa: BLE001
        return entry("damage_curve", "Damage curve (depth to loss)", "structural", "not_assessed", reason=f"curve not available ({str(e)[:80]})")
    d = float(np.clip(depth_m, 0, 12))
    vals = {k: float(np.interp(d, V.DEPTH, c)) for k, c in tab.items()}
    c0 = float(np.interp(d, V.DEPTH, own))
    if len(vals) < 2:
        return entry("damage_curve", "Damage curve (depth to loss)", "structural", "not_assessed", reason="only one continental curve is published for this asset class")
    return entry("damage_curve", "Damage curve (depth to loss)", "structural", "quantified", low=round(min(vals.values()), 3), high=round(max(vals.values()), 3), central=round(c0, 3), unit=f"damage fraction at {d:.1f} m",
                 source="JRC Huizinga et al. 2017 continental curves", method=f"{basis}; range across {len(vals)} published continents: " + ", ".join(f"{k} {v:.2f}" for k, v in sorted(vals.items())),
                 note="Transfer spread across continents, not the published confidence band of the curve. Asset class is a classification of the site, not a measurement.")


def tc_sources(winds: Optional[dict], rp: int = 100) -> dict:
    """Disagreement between independent tropical-cyclone wind sources at one site and return period: {source name: {return period: 1-min sustained wind, m/s}}.

    The engine's own winds come from IBTrACS tracks with a parametric profile. A second, independent stochastic event set (IRIS, the Imperial College Storm Model, is the intended
    one) can be passed in under its own name; nothing is read from it here, so with one source this is not_assessed."""
    srcs = {n: float(t[rp]) for n, t in (winds or {}).items() if t and rp in t and t[rp] is not None and math.isfinite(float(t[rp])) and float(t[rp]) > 0}
    if len(srcs) < 2:
        return entry("tc_sources", f"Cyclone wind 1-in-{rp}: which model", "structural", "not_assessed",
                     reason=("only one source of cyclone winds was supplied (" + ", ".join(srcs) + "); an independent second event set is needed to see the disagreement") if srcs else "no cyclone wind return level supplied")
    lo, hi = min(srcs.values()), max(srcs.values())
    return entry("tc_sources", f"Cyclone wind 1-in-{rp}: which model", "structural", "quantified", low=round(lo, 1), high=round(hi, 1), central=round(float(np.median(list(srcs.values()))), 1), unit="m/s, 1-min sustained",
                 source=", ".join(sorted(srcs)), method="return level of each source at the same site",
                 note="; ".join(f"{n} {v:.0f} m/s" for n, v in sorted(srcs.items())) + ". Wind enters damage roughly as a power law, so a modest wind gap is a larger loss gap.")


def map_disagreement(ratings: dict) -> dict:
    """Ordinal disagreement between independent hazard sources for the same peril, from their severity labels, e.g. {'JRC river map': 'NEGLIGIBLE', 'terrain model': 'MODERATE', 'rain run': 'HIGH'}."""
    order = ["NEGLIGIBLE", "LOW", "MODERATE", "HIGH", "CRITICAL"]
    r = {k: v for k, v in (ratings or {}).items() if v in order}
    if len(r) < 2:
        return entry("map_disagreement", "Flood: do independent sources agree", "structural", "not_assessed", reason="fewer than two independent hazard sources gave a rating for this site")
    lv = {k: order.index(v) for k, v in r.items()}
    lo, hi = min(lv.values()), max(lv.values())
    return entry("map_disagreement", "Flood: do independent sources agree", "structural", "quantified", low=lo, high=hi, unit="severity steps (0 negligible to 4 critical)",
                 source="site profile entries", method="lowest and highest severity across sources: " + ", ".join(f"{k} {v}" for k, v in r.items()),
                 note=("The sources disagree by %d step(s): they see different mechanisms (river network, terrain position, rain on the ground), so a disagreement is information, not noise." % (hi - lo)) if hi > lo else "The sources agree.")

# Which sources in a site profile look at the same peril, and what each one actually sees. Different mechanisms can legitimately differ (a site can be safe from a river and
# flood from rain), so a verdict is the WORST credible reading, never an average, and a source that could not be run is named as not assessed, never counted as safe.
# `shared` says honestly what inputs the sources have in common, so agreement is not mistaken for independence.
PERILS = {
    "flood": {"label": "Flood", "sources": {
        "river_flood": "river network (JRC CEMS-GloFAS map)", "river": "river flow (rainfall-runoff study)",
        "surface_water": "rain on the ground (rain-on-grid run)", "flood": "rain-driven 2-D flood study",
        "flood_terrain_radar": "terrain position and observed water history", "coastal_flood": "coast and sea level (Deltares maps)",
        "surge": "storm surge study", "sealevel": "sea-level rise study"},
        "shared": "the river map and the terrain models share elevation data; the surge and sea-level studies share the coastline"},
    "wind": {"label": "Wind and cyclone", "sources": {"cyclone": "tropical-cyclone track record", "wind": "all-cause extreme wind (ERA5 study)"},
             "shared": "both lean on reanalysis wind in places"},
    "heat": {"label": "Heat", "sources": {"thermal_stress": "site heat-stress screening", "heat": "WBGT extremes study", "urban_heat_landcover": "land-cover urban heat"},
             "shared": "the screening and the WBGT study can use the same temperature reanalysis"},
    "cold": {"label": "Cold", "sources": {"cold_extreme": "cold-extreme screening"}, "shared": ""},
    "fire": {"label": "Wildfire", "sources": {"fire": "fire-weather index study (Canadian FWI)", "wildfire_fuel": "fuel and land cover at the site"},
             "shared": "weather-based and fuel-based views are different mechanisms; neither sees ignition"},
    "water": {"label": "Water stress and drought", "sources": {"water_stress": "WRI Aqueduct basin indicators", "drought": "soil water balance study"},
              "shared": "Aqueduct is basin scale, not site supply"},
    "seismic": {"label": "Earthquake", "sources": {"earthquake": "earthquake shaking study"}, "shared": ""},
    "slope": {"label": "Landslide and debris flow", "sources": {"landslide_slope": "slope and land-cover screening", "landslide": "infinite-slope physics", "debris_flow": "debris-flow screening"},
              "shared": "all three use the same terrain model"},
    "air": {"label": "Air quality", "sources": {"air_local": "local air-quality screening", "air": "PM2.5 study"}, "shared": "may share the same PM2.5 reanalysis"},
}
FLOOD_MECHANISMS = PERILS["flood"]["sources"]
_ORDER = ["NEGLIGIBLE", "LOW", "MODERATE", "HIGH", "CRITICAL"]


def peril_concordance(profile: Optional[dict], peril: str = "flood") -> dict:
    """Do the independent sources for one peril in a site profile agree? Reads `siteprofile.profile(...)` output; nothing is re-run.

    Returns a ledger entry plus `sources` (source, what it sees, severity), `not_assessed` (sources that gave no answer and why), `agreement`, `verdict`, and for flood `observed`
    (recorded events whose mapped box contains the site). Rules: severity is the source's own rating; `none_found` counts as NEGLIGIBLE only because the engine established
    absence; a `null` entry is NOT safe. The verdict is the highest severity among assessed sources, with agreement stated separately, because a quiet source that cannot see a
    mechanism does not cancel a loud one that can. If the engine holds only one source for a peril the answer says so: that peril cannot be cross-checked here."""
    spec = PERILS[peril]
    hz = (profile or {}).get("hazards") or {}
    rows, na, unrated = [], [], []
    for k, what in spec["sources"].items():
        e = hz.get(k)
        if e is None:
            continue
        if e.get("status") in ("modelled", "constructed", "none_found") and e.get("severity") in _ORDER:
            rows.append({"source": e.get("label") or k, "key": k, "sees": what, "severity": e["severity"], "status": e["status"]})
        elif e.get("status") in ("modelled", "constructed"):
            unrated.append({"source": e.get("label") or k, "key": k, "sees": what, "value": e.get("value"), "unit": e.get("unit"),
                            "why": "ran and gave a value, but no severity rating, so it cannot be compared on the severity scale"})
        else:
            na.append({"source": e.get("label") or k, "key": k, "sees": what, "why": e.get("reason") or "no answer"})
    label = f"{spec['label']}: do independent sources agree"
    ent = map_disagreement({r["source"]: r["severity"] for r in rows})
    ent["label"] = label
    n = len(rows)
    worst = max((r["severity"] for r in rows), key=_ORDER.index) if rows else None
    can_cross = len(spec["sources"]) >= 2
    if n == 0 and unrated:
        agreement, verdict = "unrated", f"{spec['label']}: the source(s) that ran gave values without a severity rating, so there is nothing to compare on the severity scale."
    elif n == 0:
        agreement, verdict = "none", f"No {spec['label'].lower()} source gave an answer for this site: it is NOT ASSESSED here (not zero)."
    elif n == 1:
        agreement = "single_source"
        verdict = (f"One source ({rows[0]['source']}) rates this {worst}. " + ("Not cross-checked at this site; the other sources gave no answer." if can_cross
                   else "The engine holds only one source for this peril, so it cannot be cross-checked here."))
    else:
        spread = _ORDER.index(worst) - min(_ORDER.index(r["severity"]) for r in rows)
        agreement = "agree" if spread <= 1 else "disagree"
        loud = [r["source"] for r in rows if r["severity"] == worst]
        verdict = (f"Worst credible reading {worst} ({', '.join(loud)}); {n} sources "
                   + ("agree within one step." if agreement == "agree" else f"disagree by {spread} steps, so look at what each one can see before trusting the quiet one."))
        if spec["shared"]:
            verdict += f" They are not fully independent: {spec['shared']}."
    if unrated:
        verdict += " Ran but not rated (value only, so not compared): " + ", ".join(x["source"] for x in unrated) + "."
    if na:
        verdict += " Not assessed: " + ", ".join(x["source"] for x in na) + "."
    out = {"peril": peril, "sources": rows, "not_assessed": na, "unrated": unrated, "agreement": agreement, "verdict": verdict, "worst": worst, "n_sources_available": len(spec["sources"])}
    if peril == "flood":
        hist = (profile or {}).get("history") or {}
        ev = (profile or {}).get("flood_events") or hist.get("flood_events") or {}
        hits = [x for x in (ev.get("events") or []) if x.get("bbox_contains_point")]
        out["observed"] = {"available": bool(ev.get("available")), "events_with_box_over_site": len(hits), "latest": hits[0]["date"] if hits else None,
                           "note": "Recorded flood events (Copernicus EMS / Global Flood Database footprints). A box over the site is not proof it was wet: the box is the mapped area, not the water."}
        if hits and worst in ("NEGLIGIBLE", "LOW"):
            out["verdict"] += f" Caution: {len(hits)} recorded flood event(s) have a mapped box over this site, yet the models rate it {worst}."
        at = (profile or {}).get("flood_atlas") or hist.get("flood_atlas")
        if at and at.get("in_india"):
            out["observed"]["atlas"] = {k: at.get(k) for k in ("district", "state", "status", "atlas_ha", "national_percentile", "state_rank", "state_districts_listed", "reading", "reason", "source")}
            if at.get("status") == "listed":
                if at["national_percentile"] >= 0.75 and worst in ("NEGLIGIBLE", "LOW"):
                    out["verdict"] += (f" Caution: {at['district']} district is in the top quarter of the NRSC atlas for observed flood area ({at['atlas_ha']:,} ha, above "
                                       f"{round(100 * at['national_percentile'])} % of districts), yet the models rate this site {worst}. The atlas is district scale; check the site's own terrain.")
                else:
                    out["verdict"] += f" NRSC atlas: {at['district']} district mapped {at['atlas_ha']:,} ha flooded 1998-2022 (above {round(100 * at['national_percentile'])} % of districts); a low rank does not clear a site."
            else:
                out["verdict"] += " NRSC atlas: no usable district statistic here (" + (at.get("reason") or "not listed") + ")."
    if peril == "slope":
        hist = (profile or {}).get("history") or {}
        ls = (profile or {}).get("landslide_exposure") or hist.get("landslide_exposure")
        if ls and ls.get("in_india"):
            out["observed"] = {"available": True, "note": "NRSC Landslide Atlas of India (2023), Table 3: district exposure rank. Exposure, not hazard.",
                               **{k: ls.get(k) for k in ("district", "state", "status", "rank", "of", "reading", "reason", "source")}}
            if ls.get("status") == "ranked":
                top = ls["rank"] <= 50
                if top and worst in (None, "NEGLIGIBLE", "LOW"):
                    out["verdict"] += (f" Caution: NRSC ranks {ls['district']} district {ls['rank']} of {ls['of']} for landslide exposure (top third), yet the slope screening here gives "
                                       f"{worst or 'no rating'}. The rank is district scale and about exposure; check this site's own slope.")
                else:
                    out["verdict"] += f" NRSC atlas: {ls['district']} district ranks {ls['rank']} of {ls['of']} for landslide exposure; a lower rank does not clear a site."
            else:
                out["verdict"] += " NRSC atlas: this district was not ranked, which is not assessed and not safe."
    ent.update(out)
    ent["key"] = f"{peril}_concordance"
    return ent


def flood_concordance(profile: Optional[dict]) -> dict:
    """Flood only (kept for callers that already use it)."""
    return peril_concordance(profile, "flood")


def hazard_concordance(profile: Optional[dict]) -> dict:
    """All perils at once: for each, whether the sources agree, and a summary of which could and could not be cross-checked."""
    per = {p: peril_concordance(profile, p) for p in PERILS}
    ran = {p: e for p, e in per.items() if e["sources"] or e["not_assessed"] or e["unrated"]}
    return {"perils": per,
            "cross_checked": sorted(p for p, e in ran.items() if e["agreement"] in ("agree", "disagree")),
            "disagree": sorted(p for p, e in ran.items() if e["agreement"] == "disagree"),
            "single_source": sorted(p for p, e in ran.items() if e["agreement"] == "single_source"),
            "unrated": sorted(p for p, e in ran.items() if e["agreement"] == "unrated" or e["unrated"]),
            "no_answer": sorted(p for p in PERILS if p not in ran or ran[p]["agreement"] == "none"),
            "reading": ("Perils with two or more answering sources are cross-checked; those with one are not, and the answer says why. A peril with no answer is not assessed, never safe.")}


def projection_sources(proj: Optional[dict]) -> dict:
    """From `modellab.projection_check.rain_projection_sources`: do two different climate-model pipelines agree on how much heavier extreme rain gets?"""
    if not proj or not proj.get("available"):
        return entry("projection_sources", "Future extreme rain: second model pipeline", "structural", "not_assessed",
                     reason=(proj or {}).get("reason") or "the HighResMIP cross-check was not run for this site (about 20 s of network calls, rate-limited by the provider)")
    c = proj.get("cckp_ssp585_interpolated") or {}
    note = (f"CMIP6 ensemble (CCKP/NEX-GDDP) p10 {c.get('p10', float('nan')):.2f} / median {c.get('median', float('nan')):.2f} / p90 {c.get('p90', float('nan')):.2f}. "
            + ("The HighResMIP median sits inside that band. " if proj.get("highres_median_inside_cckp_p10_p90") else "The HighResMIP median is OUTSIDE that band: the projection depends on the model pipeline. ")
            + f"{len(proj['per_model'])} of 7 models returned usable years. " + proj.get("note", ""))
    return entry("projection_sources", "Future extreme rain: second model pipeline", "structural", "quantified", low=proj["low"], high=proj["high"], central=proj["median"], unit="ratio of 20-yr mean Rx1day, 2031-2050 over 1995-2014",
                 source=proj.get("source", ""), method="per-model ratio of mean annual-maximum daily rain; range is across the HighResMIP models", note=note)


def rain_event_check(chk: Optional[dict]) -> dict:
    """From `finance.rain_design.event_cross_check`: does a satellite-plus-gauge blend (CHIRPS) agree with the record's largest storm day?"""
    if not chk or not chk.get("available"):
        return entry("rain_event_check", "Largest storm day: third rain source (CHIRPS)", "structural", "not_assessed",
                     reason=(chk or {}).get("reason") or "the CHIRPS check was not run for this site (about 10 s of network reads)")
    rec = chk["record"]["mm"]
    lo, hi = sorted((chk["best_day_mm"], rec))
    return entry("rain_event_check", "Largest storm day: third rain source (CHIRPS)", "structural", "quantified", low=lo, high=hi, central=chk["two_day_mm"], unit="mm, one storm",
                 source=chk.get("source", ""), method="record's largest day versus CHIRPS best single day (low or high) and CHIRPS two-day sum (central), because a gauge day straddles two UTC days", note=chk["reading"])


def parameter_sensitivity(sens: Optional[dict]) -> dict:
    """From `finance.pluvial.surface_water_sensitivity`: how far the surface-water depth moves when assumed inputs are varied, and whether it saturates."""
    if not sens or not sens.get("available"):
        return entry("surface_water_parameters", "Surface-water depth: assumed inputs", "parameter", "not_assessed", reason="the sensitivity run was not done for this site (about 2 minutes of solver time)")
    lo, hi = sens["range_m"]
    note = sens.get("text", "")
    return entry("surface_water_parameters", "Surface-water depth: assumed inputs", "parameter", "assumed_range", low=lo, high=hi, central=sens["base_m"], unit="m",
                 source="surface_water_sensitivity", method="one input at a time: design rain, return period, drainage, curve number, roughness (ranges are stated assumptions)", note=note)


NOT_ASSESSED = (
    ("internal_variability", "Internal climate variability", "needs initial-condition ensemble members of a climate model; the engine uses one realisation per model and the observed record"),
    ("scenario", "Choice of emissions pathway", "run each pathway and show them side by side; a single ledger holds one"),
    ("structural", "Hydrological and hydraulic model form (JRC river maps)", "the JRC maps are one model chain (LISFLOOD into LISFLOOD-FP); no second global flood model is held to compare"),
    ("parameter", "Asset values and exposure", "the user's own values are used as given; their error is not assessed"),
)


def build(site: dict, entries: list) -> dict:
    """Assemble a ledger. `entries` come from the functions above (any subset). The standard not-assessed items are added unless an entry already covers them."""
    have = {e["key"] for e in entries}
    extra = [entry("na_" + k.replace(" ", "_")[:24] + str(i), lab, k, "not_assessed", reason=why) for i, (k, lab, why) in enumerate(NOT_ASSESSED)]
    allv = list(entries) + extra
    q = [e for e in allv if e["status"] in ("quantified", "assumed_range") and e["relative_range"] is not None]
    q.sort(key=lambda e: -e["relative_range"])
    return {"site": site, "entries": allv, "ranking": [{"key": e["key"], "label": e["label"], "relative_range": e["relative_range"], "status": e["status"]} for e in q],
            "counts": {s: sum(1 for e in allv if e["status"] == s) for s in ("quantified", "assumed_range", "not_assessed")},
            "dominant_quantified": q[0]["label"] if q else None,
            "combination": ("These ranges are for different quantities and are NOT combined here. A combined loss range needs the loss model run across the cases (for example the 5 to 95 % bootstrap "
                            "level, each damage curve, each scenario) and that run is not done by this ledger."),
            "reading": (f"{sum(1 for e in allv if e['status'] == 'quantified')} doubts are quantified from data, {sum(1 for e in allv if e['status'] == 'assumed_range')} are ranges someone chose, and "
                        f"{sum(1 for e in allv if e['status'] == 'not_assessed')} are not assessed" + (f"; the relatively widest quantified one is: {q[0]['label']}." if q else "."))}


def site_ledger(lat: float, lon: float, depth_m: Optional[float] = None, continent: str = "asia", asset_class: str = "industrial", design: Optional[dict] = None,
                scenario_rain: Optional[dict] = None, bootstrap: Optional[tuple] = None, ratings: Optional[dict] = None, sens: Optional[dict] = None, fetch_rain: bool = True,
                tc_winds: Optional[dict] = None, profile: Optional[dict] = None, projection: Optional[dict] = None, event_check: Optional[dict] = None) -> dict:
    """Ledger for one site from whatever has been computed (pass the pieces in; the slow ones are never run silently). With fetch_rain, the design-rain products are read (cached)."""
    if design is None and fetch_rain:
        try:
            from .finance import rain_design as RD
            design = RD.design(lat, lon)
        except Exception as e:                                       # noqa: BLE001
            design = {"available": False, "reason": f"{type(e).__name__}: {str(e)[:100]}"}
    ents = [rain_sources(design), scenario_spread(scenario_rain), return_level_sampling(bootstrap, central=(design or {}).get("d1_mm") if design else None), (None if profile else map_disagreement(ratings or {})),
            parameter_sensitivity(sens), tc_sources(tc_winds), projection_sources(projection), rain_event_check(event_check)]
    if profile:
        ents += [e for e in (peril_concordance(profile, p) for p in PERILS) if e["sources"] or e["not_assessed"] or e["unrated"]]
    ents = [e for e in ents if e]
    ents.append(damage_curve_transfer(depth_m, continent, asset_class) if depth_m is not None else
                entry("damage_curve", "Damage curve (depth to loss)", "structural", "not_assessed", reason="no flood depth supplied to evaluate the curve at"))
    return build({"lat": lat, "lon": lon, "continent": continent, "asset_class": asset_class}, ents)
