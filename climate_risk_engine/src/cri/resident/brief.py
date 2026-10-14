"""
cri/resident/brief.py — the keyless 'AI deep dive'.

Turns a computed site assessment (the dict /asset/assess returns) into a written brief, with no LLM and
no API key. Every number quoted is read from the assessment's own evidence rows; the mechanism and 2050
paragraphs are fixed engineering text, labelled as qualitative. Nothing is invented: if the assessment
has gaps (a layer that did not load) the brief says so first.
"""
from __future__ import annotations

from typing import Optional

_ORDER = {"CRITICAL": 5, "HIGH": 4, "MODERATE": 3, "LOW": 2, "NEGLIGIBLE": 1}

_MECH = {
    "tropical_cyclone": "Cyclone damage comes from wind pressure on roofs, cladding and openings, then from wind-borne debris and the loss of power and access during the storm. The exposure is set by how often storms pass close and how strong they are when they do.",
    "flood": "Flood damage depends on how deep and how long water stands on the site, and on whether switchgear, control rooms and stock sit below that level. Low flat ground drains slowly, so a modest rise in water can reach the whole site.",
    "urban_heat": "Heat risk grows with built-up cover and little shade: hot roofs and paving raise indoor and surface temperatures, which cuts equipment capacity, raises cooling load and limits safe outdoor working hours.",
    "wildfire": "Wildfire exposure follows the fuel around the site (trees, shrub and grass) and the dryness of that fuel. Embers can start fires well ahead of the front.",
    "debris_flow": "Debris flow needs a steep catchment above the site and enough water to mobilise loose material; the site is exposed when it sits at the foot of such a channel.",
    "landslide": "Landslide exposure rises with slope at and above the site and with saturated, weathered ground.",
    "water_stress": "Water stress means the catchment's users already withdraw a large share of the renewable supply, so a dry year leaves little buffer. For a site this shows up as supply cuts, higher water cost or abstraction limits. The rating describes the whole basin, not this site's own wells or utility connection.",
}
_LABEL = {"tropical_cyclone": "Tropical cyclone", "flood": "Flood", "urban_heat": "Urban heat (land-cover screening)", "wildfire": "Wildfire", "debris_flow": "Debris flow", "landslide": "Landslide", "water_stress": "Water stress (basin scale)"}
_FUTURE = {
    "tropical_cyclone": "Under Current Policies the strongest storms are expected to get stronger and wetter; under Net Zero 2050 that rise is smaller but is still locked in for the next two decades. The engine's wind figures here use 1980–2025 history with no uplift, so treat them as a floor.",
    "flood": "Sea level and heavy-rain intensity keep rising on both pathways to 2050; the gap between Current Policies and Net Zero 2050 opens mostly after 2050. For planning, assume today's flood level is the minimum.",
    "urban_heat": "Hot days increase on both pathways; Current Policies adds more of them by 2050. Design for hotter conditions than the historic record.",
    "wildfire": "Fire weather lengthens under Current Policies; Net Zero 2050 slows that change.",
    "debris_flow": "More intense rainfall raises the chance of triggering events.",
    "landslide": "More intense rainfall raises the chance of triggering events.",
    "water_stress": "The Aqueduct business-as-usual path in the evidence above shows how the basin score is projected to change; treat it as a screening trend, not a forecast for this site.",
}


def _ev_line(e: dict) -> str:
    unit = f" {e['unit']}" if e.get("unit") else ""
    return f"{e.get('metric')}: {e.get('value')}{unit}"


def site_brief(a: dict) -> dict:
    """a = the assessment dict. Returns {'narrative': str, 'sections': [...], 'model_used': 'resident-engine', 'gaps': [...]}"""
    name = a.get("asset_name") or "The site"
    lat, lon = a.get("lat"), a.get("lon")
    findings = sorted(a.get("findings") or [], key=lambda f: (_ORDER.get(f.get("severity"), 0), f.get("score", 0)), reverse=True)
    material = [f for f in findings if _ORDER.get(f.get("severity"), 0) >= 3]
    gaps = list(a.get("data_gaps") or [])

    sections = []
    if gaps:
        sections.append({"title": "Read this first", "text": "Not all data loaded for this run: " + "; ".join(g.split(" — ")[0] for g in gaps) + ". Ratings that depend on the missing layer are not reliable. Run it again once the connection is back."})

    head = a.get("headline") or ""
    sig = a.get("risk_signal") or "LOW"
    lead = f"{name} ({lat}, {lon}) is rated {sig}. {head}".strip()
    if material:
        lead += " The ratings that drive this: " + "; ".join(f"{_LABEL.get(f['hazard'], f['hazard'])} {f['severity']}" for f in material) + "."
    sections.append({"title": "Risk signal", "text": lead})

    for f in material:
        ev = [e for e in (f.get("evidence") or []) if e.get("value") not in (None, "")][:4]
        cite = ("Measured evidence — " + "; ".join(_ev_line(e) for e in ev) + ".") if ev else ""
        sections.append({"title": f"{_LABEL.get(f['hazard'], f['hazard'])}: {f['severity']}", "text": f"{_MECH.get(f['hazard'], '')} {cite}".strip()})

    if not material:
        sections.append({"title": "Why nothing rates higher", "text": "No hazard reached MODERATE on the evidence that loaded. That is a statement about this data, not a guarantee: the 3D twin and the flood solver can still find equipment-level exposure that the screening layers miss."})

    cov = [c for c in (a.get("coverage") or []) if c.get("status") in ("missing", "not_assessed")]
    if cov:
        sections.append({"title": "Not rated in this result", "text": "These were not rated, which is different from low risk: " + "; ".join(f"{c['label']} ({'missing' if c['status'] == 'missing' else 'not assessed'})" for c in cov) + "."})

    fut = [f"{_LABEL.get(f['hazard'], f['hazard'])}: {_FUTURE[f['hazard']]}" for f in (material or findings[:1]) if f.get("hazard") in _FUTURE]
    if fut:
        sections.append({"title": "By 2050 (qualitative, Current Policies vs Net Zero 2050)", "text": " ".join(fut) + " This section is qualitative; the quantified scenarios are in the Risk Analyst."})

    acts, seen = [], set()
    for f in (material or findings):
        for x in f.get("actions") or []:
            if x not in seen:
                seen.add(x); acts.append(x)
            if len(acts) >= 3:
                break
        if len(acts) >= 3:
            break
    if acts:
        sections.append({"title": "Three actions, most urgent first", "text": " ".join(f"{i + 1}. {x}" for i, x in enumerate(acts))})

    sections.append({"title": "Next step", "text": f"Ask ClimRisk to \"build a 3D twin of {lat}, {lon}\" to see which structures and equipment are exposed and which protection options the solver verifies."})
    text = "\n\n".join(f"{s['title']}\n{s['text']}" for s in sections)
    return {"narrative": text, "sections": sections, "model_used": "resident-engine", "tool_calls_made": [], "gaps": gaps}
