"""Hazard registry: every hazard the Model Lab can study, how to recognise it in a request, and which study builds and validates it."""
from __future__ import annotations

import re
from typing import Callable, Optional

from . import dossier

# key → label, regex, needs, validates_with, tier
REG = {
    "flood": {"label": "Flood inundation (rain-driven, 2-D shallow-water)", "rx": r"flood\w*|inundat\w*|storm\s*water|pluvial|rain(?:fall)?[\s\-]*runoff|waterlogg\w*|drainage", "needs": "a place", "validates_with": "observed flood depths or extents"},
    "air": {"label": "Air quality (PM2.5 and AQI, physical box model)", "rx": r"air[\s\-]*quality|aqi|pm\s*2\.?5|pm\s*10|particulate\w*|smog|air pollution|pollution", "needs": "a city", "validates_with": "ground-station PM2.5 series; CAMS analysis"},
    "heat": {"label": "Extreme heat (wet-bulb globe temperature, extreme-value hindcast)", "rx": r"heat[\s\-]*(?:wave|stress|risk)\w*|extreme heat|wbgt|wet[\s\-]*bulb|hot days|heat", "needs": "a place", "validates_with": "held-out years of ERA5 (hindcast)"},
    "rain": {"label": "Extreme rainfall (return levels, extreme-value hindcast)", "rx": r"(?:extreme|heavy|intense|severe)\s+(?:rain\w*|precipitation)|cloudburst|rainfall extremes?|precipitation extremes?|design rain\w*", "needs": "a place", "validates_with": "held-out years of ERA5 (hindcast)"},
    "wind": {"label": "Extreme wind (design gusts, extreme-value hindcast)", "rx": r"extreme wind\w*|wind[\s\-]*(?:speed|storm|hazard|load)\w*|gust\w*|windstorm\w*|design wind", "needs": "a place", "validates_with": "held-out years of ERA5 (hindcast)"},
    "fire": {"label": "Fire weather (Canadian FWI, extreme-value hindcast)", "rx": r"wild[\s\-]*fire\w*|bush[\s\-]*fire\w*|forest[\s\-]*fire\w*|fire[\s\-]*(?:weather|danger|risk|index)|fwi", "needs": "a place", "validates_with": "held-out years of ERA5 (hindcast)"},
    "drought": {"label": "Drought (soil water balance)", "rx": r"drought\w*|dry spell\w*|soil moisture|water stress|water scarcity", "needs": "a place", "validates_with": "ERA5-Land soil moisture"},
    "river": {"label": "River flow (rainfall-runoff, linear reservoirs)", "rx": r"river[\s\-]*(?:flow|flood\w*|discharge|level)|stream[\s\-]*flow|discharge|hydrograph|fluvial|rainfall[\s\-]*runoff model", "needs": "a place on a river", "validates_with": "USGS gauge (US) or GloFAS discharge"},
    "cyclone": {"label": "Tropical cyclone wind (Holland profile on IBTrACS)", "rx": r"cyclone\w*|hurricane\w*|typhoon\w*|tropical storm\w*", "needs": "a place", "validates_with": "held-out seasons of IBTrACS (hindcast)"},
    "earthquake": {"label": "Earthquake shaking (Gutenberg–Richter, Joyner–Boore)", "rx": r"earthquake\w*|seismic\w*|quake\w*|ground shaking|\bpga\b|gutenberg", "needs": "a place", "validates_with": "held-out years of the USGS catalogue (hindcast)"},
    "landslide": {"label": "Landslide susceptibility (infinite slope + pore pressure)", "rx": r"landslide\w*|slope (?:failure|stability)|mudslide\w*|debris flow\w*", "needs": "a place", "validates_with": "a landslide inventory (if supplied)"},
}


def _tables() -> dict:
    out = {k: re.compile(r"\b(?:" + v["rx"] + r")", re.I) for k, v in REG.items()}
    return out


_RX = _tables()


def detect(text: str) -> Optional[str]:
    """First hazard named in the text (by position); then hazards the engine has no solver for; None if none."""
    best = None
    for k, rx in _RX.items():
        m = rx.search(text)
        if m and (best is None or m.start() < best[0] or (m.start() == best[0] and len(m.group(0)) > best[2])):
            best = (m.start(), k, len(m.group(0)))
    # 'heat' is a loose alias: do not let it beat a more specific hazard that starts at the same place
    d = dossier.match(text)
    if best is None:
        return d
    if d is not None:
        m = re.search(dossier.CATALOG[d][1], text, re.I)
        if m and m.start() < best[0]:
            return d
    return best[1]


def all_domains() -> dict:
    d = {k: {"label": v["label"], "needs": v["needs"], "validates_with": v["validates_with"]} for k, v in REG.items()}
    for k, v in dossier.CATALOG.items():
        d[k] = {"label": v[0] + " (research dossier: no verified solver yet)", "needs": "a place", "validates_with": "n/a, it lists the data and physics needed"}
    return d


def run_generic(key: str, lat: float, lon: float, name: str, options: dict, progress: Optional[Callable] = None) -> dict:
    """Build, verify and validate a non-flood, non-air hazard; returns the standard study result dict. 'No model here' findings are results, not errors."""
    try:
        from . import evidence
        return evidence.attach(_run_generic(key, lat, lon, name, options, progress))
    except dossier.NoModel as e:
        label = (REG.get(key, {}).get("label") or key).split(" (")[0]
        return dossier.no_model_result(key, label, name, lat, lon, str(e), ["Use the study that fits this climate, or move the site."])


def _run_generic(key: str, lat: float, lon: float, name: str, options: dict, progress: Optional[Callable] = None) -> dict:
    opts = dict(options or {})
    if key in ("heat", "rain", "wind", "fire"):
        from . import studies_evt
        return studies_evt.run(key, lat, lon, name, progress=progress, **{k: v for k, v in opts.items() if k in ("series", "fetch", "bootstrap", "scenarios", "scenario_fetch")})
    if key in ("drought", "river"):
        from . import studies_series
        try:
            return getattr(studies_series, key)(lat, lon, name, progress=progress, **{k: v for k, v in opts.items() if k in ("fetch", "data", "flow")})
        except dossier.NoModel as e:
            labs = {"drought": "Drought", "river": "River flow"}
            return dossier.no_model_result(key, labs[key], name, lat, lon, str(e), ["Use a nearby site on a river or with land cover, or a gauge record if one exists."] if key == "river" else ["Move the site a few kilometres inland."])
    if key == "cyclone":
        from . import cyclone
        return cyclone.run(lat, lon, name, progress=progress, **{k: v for k, v in opts.items() if k in ("search_km",)})
    if key == "earthquake":
        from . import studies_seismic
        return studies_seismic.run(lat, lon, name, progress=progress, **{k: v for k, v in opts.items() if k in ("radius_km", "catalog", "fetch")})
    if key == "landslide":
        from . import landslide
        return landslide.run(lat, lon, name, progress=progress, **{k: v for k, v in opts.items() if k in ("radius_km", "return_period", "c_kpa", "phi_deg", "depth_m", "inventory", "dem", "rain_fetch")})
    if key in dossier.CATALOG:
        return dossier.learn_hazard(key, name, lat, lon, progress=progress)
    raise ValueError(f"unknown hazard {key!r}")
