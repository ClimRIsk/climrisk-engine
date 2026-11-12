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
                tc_winds: Optional[dict] = None) -> dict:
    """Ledger for one site from whatever has been computed (pass the pieces in; the slow ones are never run silently). With fetch_rain, the design-rain products are read (cached)."""
    if design is None and fetch_rain:
        try:
            from .finance import rain_design as RD
            design = RD.design(lat, lon)
        except Exception as e:                                       # noqa: BLE001
            design = {"available": False, "reason": f"{type(e).__name__}: {str(e)[:100]}"}
    ents = [rain_sources(design), scenario_spread(scenario_rain), return_level_sampling(bootstrap, central=(design or {}).get("d1_mm") if design else None), map_disagreement(ratings or {}),
            parameter_sensitivity(sens), tc_sources(tc_winds)]
    ents.append(damage_curve_transfer(depth_m, continent, asset_class) if depth_m is not None else
                entry("damage_curve", "Damage curve (depth to loss)", "structural", "not_assessed", reason="no flood depth supplied to evaluate the curve at"))
    return build({"lat": lat, "lon": lon, "continent": continent, "asset_class": asset_class}, ents)
