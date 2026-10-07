"""
cri/story_reports.py — story-format reports for every ClimRisk module.

Seven report types, one format (cri.story), written the way the ClimRisk research articles are: a dated hook, a thesis computed from the
engine's own numbers (the loud risk against the quiet one), research-desk figures with the source on every image, what it means per
audience, a prose method-and-limits paragraph, "How ClimRisk measured this", and numbered sources. Each is customisable — sector template, section picker, client branding — and renders to
HTML, PDF, Markdown or JSON. Every number in a story comes from an engine call (or a result the caller passes in); nothing is
written from memory. Where the engine has no value, the story says so instead of filling the gap.

    generate(report_type, payload, sections=None, template=None, brand=None, fmt="json")
    REPORTS  — registry used by the workbench (label, sections, form fields, worked example)
"""
from __future__ import annotations

import dataclasses
import datetime as dt
from typing import Optional

from . import story as S
from . import story_figs as F

# ── sector templates: framing only (no numbers) ─────────────────────────────

TEMPLATES = {
    "cement": {"label": "Cement and lime", "chain": "A cement works is a chain of single points of failure: the kiln that cannot be stopped cheaply, the silos and conveyors, the electrical rooms that run the mills, and the roads that move the product.",
               "quiet": "a motor-control room whose floor sits 15 cm above the yard, behind a kiln that took a decade to commission", "carbon": "Process emissions from calcination are fixed by chemistry; only fuel, clinker ratio and capture move the footprint."},
    "power": {"label": "Power generation", "chain": "A power station is a switchyard, a fuel path and a cooling-water path; losing any one of them stops generation whatever the state of the turbine.",
              "quiet": "a transformer bay one course of bricks below the flood line of the last storm", "carbon": "Fuel carbon intensity and run-hours dominate; carbon prices bite through dispatch before they bite through the balance sheet."},
    "refinery": {"label": "Refining and oil and gas", "chain": "A refinery is a tank farm, a set of process units, a jetty and a flare, tied together by pipe racks that nobody wants to disturb.",
                 "quiet": "an empty floating-roof tank that lifts when the bund fills, shearing the pipework it was connected to", "carbon": "Process heat and hydrogen units carry most of the emissions; throughput risk follows product demand."},
    "steel": {"label": "Iron and steel", "chain": "A steelworks puts molten metal next to water, so the physical risk is rarely the water alone but what the water does to the hot parts of the process.",
              "quiet": "a casting-floor pit that takes on water faster than the pumps were sized for", "carbon": "The blast-furnace route locks in emissions for decades; the scrap and direct-reduction routes change the profile."},
    "chemicals": {"label": "Chemicals", "chain": "A chemical site is units, storage and effluent. The loss that matters is rarely the unit — it is loss of containment during the event.",
                  "quiet": "a bunded storage area whose capacity assumed rain, not a flood", "carbon": "Feedstock and steam carbon intensity drive the footprint; product pricing decides who can pass costs on."},
    "mining": {"label": "Mining and minerals", "chain": "A mine depends on pumps, haul roads and tailings — three things that fail quietly under water and wind.",
               "quiet": "a haul road that is cut for a week, stopping a mine whose pit is perfectly dry", "carbon": "Diesel and electricity dominate; the transition exposure is customers' demand as much as the operation's own emissions."},
    "aluminium": {"label": "Aluminium", "chain": "A smelter runs on uninterrupted power; a pot line that loses power for hours freezes, and that is a loss measured in months.",
                  "quiet": "a rectifier hall whose supply depends on a single line across low ground", "carbon": "Electricity carbon intensity is the footprint; the grid decides the product's CBAM exposure."},
    "port": {"label": "Ports and logistics", "chain": "A terminal is cranes, quay, yard and the road and rail that connect it; its risk is as much access as assets.",
             "quiet": "a gate road that closes at a depth of 30 cm, long before the quay is touched", "carbon": "Scope 3 and fuel are the footprint; customers' supply-chain rules drive the transition risk."},
    "generic": {"label": "General industrial", "chain": "An industrial site is a set of structures whose failure modes depend on how it floods and how it is built, not on its sector label.",
                "quiet": "the one piece of equipment whose floor is lower than the rest of the plant", "carbon": "Where the emissions sit decides which policies reach the site."},
}
_SECT_MAP = {"cement": "cement", "lime": "cement", "electricity-generation": "power", "power": "power", "utilities": "power", "oil-and-gas-refining": "refinery", "refinery": "refinery",
             "oil_gas": "refinery", "iron-and-steel": "steel", "steel": "steel", "chemicals": "chemicals", "petrochemicals": "chemicals", "mining": "mining", "aluminum": "aluminium",
             "aluminium": "aluminium", "port": "port", "logistics": "port", "transport": "port"}


def template_for(sector: Optional[str], key: Optional[str] = None) -> dict:
    k = key or _SECT_MAP.get((sector or "").lower().replace(" ", "-"), "generic")
    return {**TEMPLATES.get(k, TEMPLATES["generic"]), "key": k if k in TEMPLATES else "generic"}


def _usd(x: Optional[float]) -> str:
    if x is None:
        return "n/a"
    a = abs(x)
    return f"${x / 1e9:,.1f}bn" if a >= 1e9 else f"${x / 1e6:,.1f}m" if a >= 1e6 else f"${x / 1e3:,.0f}k" if a >= 1e3 else f"${x:,.0f}"


def _pct(x: Optional[float], d: int = 0) -> str:
    return "n/a" if x is None else f"{x:.{d}f}%"


def _brand(p: dict, brand: Optional[dict]) -> dict:
    b = dict(brand or {})
    if p.get("client") and not b.get("client"):
        b["client"] = p["client"]
    return b


def _kick(label: str) -> str:
    return f"{label} · {dt.date.today().strftime('%d %B %Y')}"


def _try(fn, *a, **k):
    try:
        return fn(*a, **k)
    except Exception:                                               # noqa: BLE001
        return None


def _ratio(quiet: float, loud: float) -> tuple:
    """(ratio, wording) for quiet-versus-loud, from the engine's own present values. Never asserts a direction the numbers do not show."""
    if not loud or loud <= 0 or quiet is None:
        return None, "The loud risk could not be priced separately for this asset, so the comparison is not made."
    r = quiet / loud
    if r >= 1.5:
        return r, f"The quiet risk is about {r:.0f} times the loud one" if r >= 10 else f"The quiet risk is about {r:.1f} times the loud one"
    if r >= 0.8:
        return r, "The quiet risk is about the same size as the loud one"
    return r, f"Here the loud risk is the larger: the quiet one is {r:.1f} times its size"


def _cmp(trans: float, phys: float, scen: str) -> str:
    if not phys or phys <= 0:
        return f"Transition cost under the {scen}: {_usd(trans)}."
    r = trans / phys
    rel = f"{r:.0f} times" if r >= 10 else f"{r:.1f} times" if r >= 1.2 else "about the same as" if r >= 0.8 else f"{r:.1f} times"
    return f"Under the {scen} the transition cost ({_usd(trans)}) is {rel}{' ' if rel.endswith('times') else ' '}the modelled physical damage ({_usd(phys)})."


def _oni_para(st, oni: Optional[dict]) -> None:
    if not oni:
        return
    st.source("oni", oni["title"], oni["url"], "monthly 3-month running mean, 1950 to date")
    la = oni["latest"]
    st.p(f'The background state matters for the season. The Oceanic Niño Index for {la["season"]} stands at {la["oni"]:+.2f} °C ({la["state"]}); since 1950 it has crossed +2 °C on only '
         f'{oni["episodes_over_2c"]} occasions [[S:oni]]. ENSO shifts cyclone and flood odds in many basins, but the engine does not condition its return periods on it: treat the index as context, not as an adjustment to the figures below.')
    _try(lambda: st.figure(F.oni(oni["series"], "Oceanic Niño Index since 1950", "NOAA Climate Prediction Center, ERSSTv5"), caption="Orange: El Niño episodes (≥ +0.5 °C); blue: La Niña (≤ −0.5 °C). Gold dot: latest season [[S:oni]]."))


def _event_hook(st, p: dict, lat: float, lon: float, name: str, storms: list) -> Optional[dict]:
    """Opening paragraphs: a dated, sourced real-world hook. Priority: facts the user supplied → live GDACS event in reach → the strongest storm on record."""
    from . import story_context as C
    ev = C.event_from_payload(p)
    if ev:
        if ev["headline"]:
            st.p(f'**{ev["headline"]}**' + (f' ({ev["date"]})' if ev["date"] else ""))
        for i, n in enumerate(ev["notes"]):
            st.source(f"ev{i}", n["source"], n.get("url", ""), "supplied with the request")
            st.p(f'{n["text"]} [[S:ev{i}]]')
        return {"kind": "supplied"}
    g = C.nearest_event(lat, lon) if p.get("live_event", True) else None
    if g:
        st.source("gdacs", f'GDACS — {g["title"]}', g["url"], g["source"])
        kind = "tropical cyclone" if g["type"] == "TC" else "flood"
        st.p(f'**{g["title"]}.** The UN and European Commission\'s alert system rates this {kind} {g["alert"]}, active {str(g["from"])[:10]} to {str(g["to"])[:10]}, with its centre {g["distance_km"]} km from {name} [[S:gdacs]]. '
             f'That is a distance, not a damage report: GDACS gives a point and a rating, not a footprint. What the point does to a particular structure depends on how that structure is built and where it stands, which is what the rest of this report works out.')
        return {"kind": "gdacs", "event": g}
    if storms:
        s = storms[0]
        nm = s["name"] if s["name"] and not s["name"].lower().startswith("unnamed") else "an unnamed tropical system"
        st.p(f'**{s["year"]}.** The strongest storm in the track archive within reach of {name} was {nm}: modelled sustained wind of {s["peak_wind_ms"]:.0f} m/s at the site ({s["category_at_site"]}), '
             f'with the centre {s["closest_km"]:.0f} km away at its closest [[S:ibt]]. The archive holds {len(storms)} such passages since 1980. That is the loud part of the record. It is not, on its own, a statement about what the site can take.')
        return {"kind": "archive"}
    return None


def _wind_v10(tw: dict, sid: str) -> Optional[float]:
    for s in tw["scenarios"]["wind"]:
        if s["id"] == sid:
            return s["v10_ms"]
    for s in tw["history"].get("storms", []):
        if s["id"] == sid:
            return s["peak_wind_ms"]
    return None


def _uniq_names(comps: list) -> dict:
    total: dict = {}
    for c in comps:
        total[c["name"]] = total.get(c["name"], 0) + 1
    seen: dict = {}
    out = {}
    for c in comps:
        k = c["name"]; seen[k] = seen.get(k, 0) + 1
        out[c["id"]] = f"{k} {seen[k]}" if total[k] > 1 else k
    return out


# ═════════════════════════════════════════════════════════════════════════════
# 1. PHYSICAL — the asset story
# ═════════════════════════════════════════════════════════════════════════════

def physical(p: dict, tpl: dict, brand: dict) -> dict:
    from . import asset_twin as AT
    from . import story_context as C
    name = p.get("name") or "the asset"
    tw = p.get("twin")
    if tw is None:
        tw = AT.build_twin(float(p["lat"]), float(p["lon"]), name, p.get("sector"), p.get("asset_type"), p.get("commissioned_year"),
                           float(p.get("radius_m", 300)), p.get("edits"), p.get("incidents"), True, run_hydraulics=p.get("hydraulics", True))
    an = p.get("analysis") or AT.analyse(tw)
    fnd = an["findings"]
    lab = p.get("lab")
    if lab is None and fnd and p.get("scenario_lab", True):
        lab = _try(lambda: __import__("cri.resident.scenario", fromlist=["lab"]).lab(tw, an))
    name = tw["asset"]["name"] if tw.get("asset", {}).get("name") else name
    lat, lon = tw["asset"]["lat"], tw["asset"]["lon"]
    site, hist, scen = tw["site"], tw["history"], tw["scenarios"]
    comps = [c for c in tw["components"] if not c.get("removed") and c.get("on_site", True)]
    onsite = [c for c in comps if c["kind"] != "building"] if any(c["kind"] != "building" for c in comps) else comps
    by = {c["id"]: c for c in tw["components"]}
    label_of = _uniq_names(onsite)
    storms = sorted(hist.get("storms", []), key=lambda e: -e["peak_wind_ms"])
    fl_f = sorted([f for f in fnd if f["hazard"] == "flood"], key=lambda f: -len(f["components"]))
    wd_f = sorted([f for f in fnd if f["hazard"] == "wind"], key=lambda f: -(_wind_v10(tw, f["scenario"]) or 0))
    fixed = sum(1 for f in fnd if any(m.get("resolves_equipment") or m.get("residual_beyond_design") == 0 for m in f.get("protection", [])))
    max_map = max((s["depth_m"] for s in scen["flood"]), default=0.0)

    # the priced loud risk (risk analyst), when the register/hazard data can be reached
    ra = p.get("analyst") or _try(lambda: __import__("cri.finance.analyst", fromlist=["x"]).analyse_asset(lat, lon, value_usd=p.get("value_usd"), n=int(p.get("n", 1000))))
    ph = (ra or {}).get("financial", {}).get("physical") if ra else None
    cyc = (ra or {}).get("physical", {}).get("cyclone") if ra else None
    value = ((ra or {}).get("asset", {}).get("value", {}) or {}).get("mode") if ra else None
    vbasis = ((ra or {}).get("asset", {}).get("value", {}) or {}).get("basis", "") if ra else ""

    # flood view: the scenario that reaches most structures
    fv = {"states": {}, "depth": {}, "sim": None, "sc": None, "tally": None, "maxd": 0.0, "n": 0}
    if fl_f:
        sc = next((s for s in AT._key_flood(tw, solver=True) if s["id"] == fl_f[0]["scenario"]), None)
        if sc:
            def _ev():
                hz = AT._hz_flood(sc, tw); hz.pop("_sim", None)
                return AT.evaluate(onsite, site, hz, None, tw.get("linear"))
            base = _try(_ev)
            if base:
                fv.update(sc=sc, sim=(tw.get("sims") or {}).get(sc["id"]), tally=base["tally"]["flood"],
                          states={cid: x["flood"]["state"] for cid, x in base["components"].items()}, depth={cid: x["flood"]["depth_m"] for cid, x in base["components"].items()})
                fv["n"] = len(fl_f[0]["components"]); fv["maxd"] = float(((fv["sim"] or {}).get("validation") or {}).get("max_depth_m") or max(fv["depth"].values() or [0]))
    n_eq = (fv["tally"]["equipment_inundated"] + fv["tally"]["submerged"]) if fv["tally"] else 0

    # wind view: strongest scenario that exceeds design
    wv = {"rows": [], "v10": None, "label": "", "n": 0}
    if wd_f:
        f = wd_f[0]; v10 = _wind_v10(tw, f["scenario"])
        base = _try(lambda: AT.evaluate(onsite, site, {"wind": {"v10_ms": v10}}))
        if base:
            wv.update(v10=v10, label=f["title"].split(":")[0], n=len(f["components"]))
            ids = sorted(f["components"], key=lambda c: -base["components"][c]["wind"]["ratio"])[:10]
            grp: dict = {}
            for c in sorted(f["components"], key=lambda c: -base["components"][c]["wind"]["ratio"]):
                if by[c].get("design_wind_ms"):
                    k = (by[c]["name"], round(float(base["components"][c]["wind"]["gust_ms"]), 0), float(by[c]["design_wind_ms"]))
                    grp[k] = grp.get(k, 0) + 1
            wv["rows"] = [((f"{k[0]} (×{n})" if n > 1 else k[0]), k[1], k[2]) for k, n in sorted(grp.items(), key=lambda kv: -(kv[0][1] / kv[0][2]))][:10]

    # ── the story ──
    if fv["n"] and max_map < 0.05:
        title = f"{name}: the flood map is blank, the solver is not"
    elif fv["n"]:
        title = f"{name}: {fv['n']} structures in the water, {wv['n']} beyond their wind rating" if wv["n"] else f"{name}: where the water gets in"
    elif wv["n"]:
        title = f"{name}: the wind the plant was not built for"
    else:
        title = f"{name}: what the water and the wind do not reach"
    parts = [f'{len(comps)} structures modelled on real footprints and terrain.']
    if fv["n"]:
        parts.append(f'In the worst mapped flood ({fv["sc"]["label"]}) a 2D solver puts water on {fv["n"]} of them, {n_eq} with equipment under water' + (f', up to {fv["maxd"]:.1f} m deep, on a site where the hazard map shows {max_map:.1f} m.' if max_map < 0.05 else f', up to {fv["maxd"]:.1f} m deep.'))
    elif scen["flood"]:
        parts.append("The flood scenarios do not reach any structure.")
    if wv["n"]:
        parts.append(f'{wv["n"]} exceed their assumed design wind in the {wv["label"].lower()} scenario.')
    standfirst = " ".join(parts)
    st = S.Story(_kick("Physical risk · asset story"), title, standfirst, byline="ClimRisk Asset Twin", brand=brand)
    for k, t, u, n in [("osm", "OpenStreetMap contributors — building and industrial footprints", "https://www.openstreetmap.org/copyright", "ODbL"),
                       ("terr", "AWS Terrain Tiles (Terrarium), ~9 m", "https://registry.opendata.aws/terrain-tiles/", ""),
                       ("jrc", "JRC CEMS-GloFAS river flood hazard maps v2.1", "https://data.jrc.ec.europa.eu/collection/id-0054", "undefended 3 arc-second depth maps"),
                       ("del", "Deltares Global Flood Maps v2021.06 (GTSMip6 surge + tide)", "https://deltares-research.github.io/global-flood-maps/", "undefended coastal depth maps"),
                       ("flopros", "Scussolini et al. (2016), FLOPROS: an evolving global database of flood protection standards, NHESS 16:1049", "https://doi.org/10.5194/nhess-16-1049-2016", "protection standard, as a return period"),
                       ("ibt", "NOAA IBTrACS tropical-cyclone best tracks", "https://www.ncei.noaa.gov/products/international-best-track-archive", "1980–2025, Holland wind model"),
                       ("era5", "ERA5 reanalysis via Open-Meteo archive", "https://open-meteo.com/en/docs/historical-weather-api", "daily maximum gust, 1950 onward"),
                       ("huiz", "Huizinga, de Moel & Szewczyk (2017), Global flood depth-damage functions, JRC", "https://doi.org/10.2760/16510", "industrial curve"),
                       ("eman", "Emanuel (2011), Global warming effects on U.S. hurricane damage, Weather, Climate & Society 3:261", "https://doi.org/10.1175/WCAS-D-11-00007.1", "wind damage sigmoid"),
                       ("bates", "Bates, Horritt & Fewtrell (2010), local-inertial shallow-water formulation, J. Hydrology 387:33", "https://doi.org/10.1016/j.jhydrol.2010.03.027", "")]:
        st.source(k, t, u, n)

    # cover: orbit image if reachable, otherwise the solver plan view
    site_img = C.site_image(lat, lon) if p.get("satellite", True) else None
    plan_png = _try(lambda: F.site_plan(tw, fv["sim"], fv["states"], f"{name}: structures by flood state", "OpenStreetMap footprints; AWS Terrain Tiles; ClimRisk 2D solver (local-inertial)", True,
                                         label_n=7)) if (fv["sc"] or onsite) else None
    cover_img = (site_img or {}).get("png") or plan_png
    if cover_img:
        cv = F.cover(title, f'Physical risk assessment · {dt.date.today().strftime("%d %B %Y")}', "ClimRisk Asset Twin · " + ((site_img or {}).get("credit") or "OpenStreetMap, AWS Terrain Tiles, ClimRisk 2D solver"), (site_img or {}).get("png") or plan_png, tag="physical risk", crop=bool(site_img))
        st.cover(cv)

    # ── the hook ──
    hook = _event_hook(st, p, lat, lon, name, storms)
    cards = [(str(len(comps)), "structures modelled on mapped footprints", "OpenStreetMap + 9 m terrain"),
             (f'{fv["maxd"]:.1f} m' if fv["n"] else "none", "deepest solver water on the site" if fv["n"] else "water reaches no structure", fv["sc"]["label"] if fv["sc"] else "all mapped scenarios"),
             (f'{fv["n"]} of {len(onsite)}' if fv["sc"] else f"0 of {len(onsite)}", "structures reached in the worst flood", "2D shallow-water solver"),
             (f'{storms[0]["peak_wind_ms"]:.0f} m/s' if storms else "none", f'strongest storm on record ({storms[0]["year"]})' if storms else "tropical-storm-force passage on record", "IBTrACS best tracks, Holland model"),
             (f'{ph["aal_pct"]:.2f}%' if ph else "n/a", "modelled annual damage, share of value", "JRC / Emanuel curves × hazard"),
             (str(fixed), "of " + str(len(fnd)) + " findings with a verified fix", "solver / re-evaluation")]
    _try(lambda: st.figure(F.numbers(f"{name} in six numbers", "What the engine found, measured at the site", cards, "ClimRisk Asset Twin; IBTrACS; JRC CEMS-GloFAS; Deltares; OpenStreetMap; AWS Terrain Tiles"),
                           caption="Every number is computed by the engine for this site; the source is stamped on the image."))
    if hook is None:
        st.p(f'{name} stands {site["ground_ref_m"]} m above datum. This report asks what happens to it when the water and the wind come, structure by structure.')

    # ── the thesis ──
    q_txt = ""
    if fv["n"]:
        q_txt = (f'The loud risk is what the maps show: ' + (f'a flood layer with {max_map:.1f} m on the site' if max_map < 0.05 else f'mean mapped depths up to {max_map:.1f} m') +
                 f', and a cyclone record whose worst passage brought {storms[0]["peak_wind_ms"]:.0f} m/s. ' if storms else '. ')
        q_txt += (f'The quiet one is what the site does with the water. The solver finds low ground the 90 m map cannot resolve and puts water on {fv["n"]} of {len(onsite)} structures, {n_eq} of them with equipment under water. ')
    elif wv["n"]:
        q_txt = f'The loud risk is the storm. The quiet one is the gap between the wind and what was assumed at each structure: {wv["n"]} of {len(onsite)} exceed their assumed design gust in the {wv["label"].lower()}. '
    st.pull(f'{fv["n"]} of {len(onsite)} structures reached by water' if fv["n"] else (f'{wv["n"]} of {len(onsite)} beyond design wind' if wv["n"] else "No structure reached in the modelled scenarios"),
            (q_txt + tpl["chain"]) if q_txt else "In the scenarios modelled, nothing at this site is reached by water or pushed beyond its design wind. That is a result to check against as-built levels, not a clearance.")

    # ── history ──
    st.section("history", "What has already happened here")
    ev = [(e["year"], e["peak_wind_ms"], f'{e["name"]} {e["year"]}') for e in storms[:14] if e.get("year")]
    if storms:
        st.p(f'The track archive holds {len(hist["storms"])} tropical storm or stronger passages within reach of this site since 1980 [[S:ibt]]. The strongest, {storms[0]["name"]} in {storms[0]["year"]}, '
             f'brought about {storms[0]["peak_wind_ms"]:.0f} m/s ({storms[0]["category_at_site"]}) at the site, passing {storms[0]["closest_km"]:.0f} km away.')
        yrs = sorted({e["year"] for e in storms}); sv = {e["year"]: e["peak_wind_ms"] for e in storms}
        _try(lambda: st.figure(F.events(yrs, [sv[y] for y in yrs], "Modelled wind at the site, storm by storm", "NOAA IBTrACS v04r01; Holland (1980) wind field; ClimRisk engine", ylabel="sustained wind at 10 m (m/s)",
                                        highlight={storms[0]["year"]: f'{storms[0]["peak_wind_ms"]:.0f} m/s'}, threshold=(17.5, "tropical-storm force")),
                               caption="Each stem is one storm's peak modelled wind at the site. Gold: the strongest [[S:ibt]]."))
        st.table(["Storm", "Year", "Peak wind at site", "Category", "Closest approach"], [[e["name"], e["year"], f'{e["peak_wind_ms"]:.0f} m/s', e["category_at_site"], f'{e["closest_km"]:.0f} km'] for e in storms[:6]])
    else:
        st.p("No tropical storm has been recorded within reach of this site in the track archive [[S:ibt]].")
    ws = hist.get("windstorms", [])
    if ws:
        st.p("The all-cause record (reanalysis, 1950 onward) adds non-cyclone wind. The five strongest daily gusts: " + "; ".join(f'{w["date"]} {w["gust_ms"]:.0f} m/s' for w in ws[:5]) + " [[S:era5]]. Reanalysis smooths peaks, so these are floors, not ceilings.")
    inc = hist.get("incidents", [])
    st.p(("Reported by the operator: " + "; ".join(f'{i["date"]}: {i.get("label") or i.get("note")}' for i in inc[:6]) + ". The replay is checked against these.") if inc else
         "No operator-reported incidents were supplied. Adding them lets the replay be checked against what actually happened, which is the best test of any of the numbers here.")
    _oni_para(st, C.oni_series() if p.get("context", True) else None)

    # ── loud ──
    st.section("loud", "The loud risk, priced")
    if cyc and ph:
        rp = ph.get("annual_loss_rp") or {}
        st.p(f'Cyclone wind is the risk everyone plans for. Here the archive implies {cyc["lam"]:.2f} tropical-storm-force passages a year and a 1-in-100-year wind of {cyc["rp100_ms"]:.0f} m/s ({cyc["category_100yr"]}) [[S:ibt]]. '
             f'Run through the damage curve [[S:eman]] against an asset value of {_usd(value)} ({vbasis}), the modelled average annual damage is **{ph["aal_pct"]:.2f}% of value** ({_usd(ph["aal_usd"])} a year)'
             + (f', and a 1-in-100-year event year costs **{_usd(rp["100"])}**, {100 * ph["annual_damage_ratio_rp"]["100"]:.0f}% of value.' if "100" in rp else "."))
        if rp:
            _try(lambda: st.figure(F.bars([f"1-in-{k} year" for k in rp], [v / 1e6 for v in rp.values()], "Physical loss in a single year, by return period", "ClimRisk risk analyst: IBTrACS cyclone event set × Emanuel (2011) curve × JRC flood × FLOPROS defences",
                                          unit=" m", fmt="${:,.0f}", horizontal=False, color=F.BLUE, highlight=list(rp).index("100") if "100" in rp else None, xlabel="US$ million"),
                                   caption="Single-year direct physical loss at each return period. Direct damage only; business interruption is not included [[S:eman]]."))
    else:
        st.p(f'Cyclone wind is the risk everyone plans for. The site-specific loss figures could not be priced here (the risk analyst was unavailable or has no value basis), so only the wind scenarios below are used. 1-in-100 sustained wind: ' +
             (", ".join(f'{w["v10_ms"]:.0f} m/s ({w.get("category", "")})' for w in scen["wind"] if w.get("rp") == 100 and w["kind"] == "design") or "see scenario table") + ".")
    if scen["flood"]:
        rv = next((s for s in scen["flood"] if s["kind"] == "river"), None); cs = next((s for s in scen["flood"] if s["kind"] != "river"), None)
        df = (ra or {}).get("physical", {}).get("defences") if ra else None
        st.p("The flood maps are the other loud number. " + (f'River and coastal layers are undefended; the engine credits a protection standard of about 1-in-{df["river_rp"]:.0f} (river) and 1-in-{df["coastal_rp"]:.0f} (coast) from FLOPROS [[S:flopros]]. ' if df else "")
             + f'Mean mapped depth over the site reaches {max_map:.2f} m in the deepest scenario [[S:jrc]][[S:del]].')
        if max_map < 0.005:
            st.p(f'All {len(scen["flood"])} mapped flood scenarios (river 1-in-10 to 1-in-500, coastal surge today and with sea-level rise) put 0.00 m of water on the site. That is the loud number, and it is zero.')
        else:
            st.table(["Flood scenario", "Mean depth", "90th percentile", "Share of site wet", "Source"],
                     [[s["label"], f'{s["depth_m"]:.2f} m', f'{s["depth_p90_m"]:.2f} m', f'{round(100 * s["wet_share"])}%', "JRC [[S:jrc]]" if s["kind"] == "river" else "Deltares [[S:del]]"] for s in scen["flood"][:8]])
    dw = [w for w in scen["wind"] if w["kind"] == "design"]
    if dw:
        st.table(["Wind scenario", "Sustained at 10 m", "Category"], [[w["label"], f'{w["v10_ms"]:.0f} m/s', w.get("category", "")] for w in scen["wind"][:8]], caption="Return-period winds from the cyclone model and the all-cause reanalysis fit [[S:ibt]][[S:era5]].")

    # ── quiet ──
    st.section("quiet", "The quiet risk: what the site does to the water")
    st.p(f'{tpl["chain"]} The maps stop at a pixel 90 m wide. The site does not. At {name} the twin carries {len(comps)} structures on their mapped footprints, each with a floor level taken from 9 m terrain, and a 2D solver moves the water between them. '
         f'The quiet risk is {tpl["quiet"]}.')
    if fv["sc"] and plan_png:
        v = (fv["sim"] or {}).get("validation") or {}
        st.figure(plan_png, caption=f'Maximum water depth in the worst scenario ({fv["sc"]["label"]}), with each structure coloured by what the water does to it. Blue shading: solver depth; contours: terrain [[S:terr]][[S:osm]][[S:bates]].')
        if v:
            st.p(f'Solver check: mass residual {v.get("mass_residual_pct", 0)}%, solver wet land {round(100 * v.get("solver_wet_share_of_open_land", 0))}% against {round(100 * v.get("map_wet_share_of_open_land", 0))}% on the map ({v.get("consistency", "")}).')
    if fv["n"] and max_map < 0.05:
        st.callout("The flood maps show no depth on the site itself, yet the 2D solver reaches structures. The solver takes its water level from the nearest wet map pixels (up to 600 m away) and routes it over 9 m terrain, "
                   "which can fill low pockets that the 90 m flood map smooths over. Read the flood findings below as a warning to check ground levels on site, not as a mapped flood.", "warn")
    if not fnd:
        st.p("Nothing in the modelled scenarios reaches a component or exceeds its design wind. Heights, equipment levels and design winds are assumptions by structure kind until the operator edits them, so this is the result to check first. It is not a clearance.")
    if fv["n"]:
        worst = sorted([c for c in onsite if fv["states"].get(c["id"]) in ("wet_floor", "equipment_inundated", "submerged")], key=lambda c: -fv["depth"].get(c["id"], 0))[:8]
        st.h3(fl_f[0]["title"])
        st.table(["Structure", "Kind", "Floor above datum", "Water depth", "State"], [[label_of.get(c["id"], c["name"]), c["kind"], f'{c.get("floor_elev_m", 0):.2f} m', f'{fv["depth"].get(c["id"], 0):.2f} m', fv["states"].get(c["id"], "").replace("_", " ")] for c in worst],
                 caption="Worst-reached structures. Floor levels come from the terrain model plus an assumed plinth until the operator supplies as-built values.")
        st.p("**Why:** " + fl_f[0]["cause"]); st.p("**What it costs:** " + fl_f[0]["consequence"])
        if fl_f[0].get("access"):
            st.p("**Access:** " + "; ".join(f'{a["name"]} ({a["type"]}) {a["state"]} at {a["max_depth_m"]} m' for a in fl_f[0]["access"][:3]) + ".")
        for f in fl_f[1:3]:
            st.p(f'**{f["title"]}.** ' + "; ".join(f["problem"][:2]))
    if wv["n"] and wv["rows"]:
        st.h3(wd_f[0]["title"])
        _try(lambda: st.figure(F.gust_vs_design(wv["rows"], f'Gust at the structure vs its assumed design wind — {wv["label"]}', "ClimRisk engine: power-law profile × gust factor at each structure's height; design winds assumed by kind"),
                               caption=f'A {wv["v10"]:.0f} m/s sustained wind at 10 m becomes a larger gust at the top of a tall structure, and load grows with the square of speed. Red bars exceed the assumed design gust.'))
        st.p("**Why:** " + wd_f[0]["cause"]); st.p("**What it costs:** " + wd_f[0]["consequence"])
        st.callout("Design winds here are assumed by structure kind (tank, silo, building). Replace them with the as-built design values in the twin editor: that is the single most valuable edit for this section.", "warn")
    if len(fnd) > 1:
        st.table(["Finding", "Hazard", "Structures", "Verified fix"], [[f["title"], f["hazard"], len(f["components"]), "yes" if any(m.get("resolves_equipment") or m.get("residual_beyond_design") == 0 for m in f.get("protection", [])) else "no"] for f in fnd],
                 caption="Every finding the Asset Twin agent raised, with whether a measure re-run through the solver or structure check resolves it.")
    if an.get("notes"):
        uniq: dict = {}
        for n_ in an["notes"]:
            lab_, _, body_ = n_.partition(": ")
            uniq.setdefault(body_ or n_, []).append(lab_ if body_ else "")
        st.callout(" ".join((", ".join(l for l in labs if l) + ": " if any(labs) else "") + body for body, labs in list(uniq.items())[:2]), "warn")

    # ── protection ──
    st.section("protection", "What protects it, ranked and verified")
    prot_rows = []
    for f in (fl_f[:1] + wd_f[:1]):
        for m in f.get("protection", []):
            what = f'raise flood-critical equipment {m["delta_m"]} m' if m["type"] == "raise_equipment" else f'perimeter flood barrier {m["height_m"]} m' if m["type"] == "flood_barrier" else f'wind retrofit to {m["design_wind_ms"]:.0f} m/s'
            res = (f'{m.get("residual_equipment_inundated", 0)} equipment inundated, {m.get("residual_wet_floor", 0)} wet floor' if "residual_wet_floor" in m else f'{m.get("residual_beyond_design", 0)} still beyond design')
            prot_rows.append([f["hazard"], what, res, m.get("verified_by", "re-evaluation")])
    if prot_rows:
        before = (fv["tally"]["equipment_inundated"] + fv["tally"]["submerged"] + fv["tally"]["wet_floor"]) if fv["tally"] else 0
        fl_m = [m for m in (fl_f[0].get("protection", []) if fl_f else []) if "residual_wet_floor" in m]
        if fl_m:
            labs = ["As modelled"] + [("raise equipment" if m["type"] == "raise_equipment" else "barrier") for m in fl_m]
            vals = [float(before)] + [float(m["residual_equipment_inundated"] + m["residual_wet_floor"]) for m in fl_m]
            _try(lambda: st.figure(F.bars(labs, vals, f'Structures reached by water, before and after ({fl_f[0]["title"].split(":")[0]})', "ClimRisk 2D solver / component re-evaluation; measures proposed and verified by the engine", horizontal=False, fmt="{:,.0f}",
                                          color=F.BLUE, highlight=0), caption="Each bar re-runs the structure check with the measure applied. The barrier is re-run through the solver with the wall in the model."))
        st.table(["Hazard", "Measure", "What remains", "Verified by"], prot_rows)
    if lab and lab.get("solutions"):
        st.p(lab["recommendation"])
        st.table(["Solution", "Resolves base scenarios", "Flood margin", "Wind margin"],
                 [[s["name"], "yes" if s["resolves_base"] else "no", f'+{s["flood_margin_m"]} m' if s["flood_margin_m"] is not None else "–", f'+{s["wind_margin_pct"]}%' if s["wind_margin_pct"] is not None else "–"] for s in lab["solutions"][:6]],
                 caption="Scenario lab: every mapped flood scenario shifted by +0 to +1.0 m and every wind scaled ×1.0 to ×1.2. Ranked by physical performance; no costs are invented. " + " ".join(lab.get("limits", [])[:1]))
    elif not prot_rows:
        st.p("No protection is proposed because nothing is reached in the modelled scenarios." if not fnd else "The scenario lab did not run, so protection options are listed under each finding only.")

    # ── audiences ──
    st.section("audiences", "What this means, by reader")
    worst = fl_f[0] if fl_f else (wd_f[0] if wd_f else None)
    st.p("**For the operator.** " + (f'Start with: {worst["title"].lower()}. ' if worst else "Confirm the heights and equipment levels in the twin, because every result depends on them. ")
         + "The cheapest verified fix is listed above; edit the twin with as-built values and the findings change with it.")
    st.p("**For the lender or insurer.** " + (f'{len(fnd)} scenario(s) produce a finding at this site' + (f', and the flood map alone would have shown none' if max_map < 0.05 and fv["n"] else "") + '. Ask which verified measures are in the capital plan and when. ' if fnd else "No scenario produces a finding on the assumed values; ask for the as-built levels that make that true. ")
         + "The hazard maps are undefended; local defences are credited only at the FLOPROS standard unless the operator supplies better.")
    st.p("**For the investor.** " + (f'Direct physical loss is {ph["aal_pct"]:.2f}% of value a year on average, and {_usd((ph.get("annual_loss_rp") or {}).get("100"))} in a 1-in-100 year. ' if ph and (ph.get("annual_loss_rp") or {}).get("100") else "")
         + "These are direct-damage figures. A site that stops for weeks costs more than the equipment it loses, and that is not in these numbers.")

    # ── method ──
    st.section("method", "Method and limitations")
    st.p("Flood depths come from a 2D shallow-water solver (local-inertial form [[S:bates]]) on 9 m terrain [[S:terr]] with every building a solid obstruction, forced with the water level implied by the flood map [[S:jrc]][[S:del]]. "
         "Event duration is assumed; rainfall, drains, pumps and waves are not modelled. Wind at a structure's height uses a power-law profile and a gust factor, and design winds are assumed by kind until edited. "
         "Tornado and downburst are not modelled. Footprints come from OpenStreetMap [[S:osm]]" + ("; no footprints were mapped here, so an illustrative layout is shown and flagged." if site.get("layout_source") == "parametric" else "; unmapped process equipment must be added with the twin editor.")
         + " " + " ".join(list(tw.get("data_gaps", []))[:2]))
    st.ul(an.get("assumptions", [])[:5])
    st.h3("How ClimRisk measured this")
    st.ul([f'Terrain and footprints → a 3D twin of {len(comps)} structures on {site["radius_m"]:.0f} m radius.', "Hazards → JRC river, Deltares coastal, IBTrACS cyclone and ERA5 all-cause wind, each cited.",
           "Quantify → a 2D hydraulic solver per flood scenario and a height-adjusted gust check per structure.", "Protect → measures proposed by the engine and re-run through the solver or the structure check before they are reported.",
           "Price → damage curves and the analyst's Monte Carlo for the loss figures. Every number above can be regenerated from the twin."])
    return st.build()


# ═════════════════════════════════════════════════════════════════════════════
# 2/3. TRANSITION and FINANCIAL — from the risk analyst
# ═════════════════════════════════════════════════════════════════════════════

ANN = (1 - 1.07 ** -25) / 0.07            # 25-year annuity factor at the engine's 7% discount rate (used only to turn a company's annual physical loss into a present value)


def _analyst(p: dict) -> dict:
    if p.get("result"):
        return p["result"]
    from .finance import analyst as AN
    if p.get("lat") is not None and p.get("lon") is not None:
        return AN.analyse_asset(float(p["lat"]), float(p["lon"]), value_usd=p.get("value_usd"), n=int(p.get("n", 1500)),
                                **{k: p[k] for k in ("sector", "asset_type", "capacity", "emissions_t", "commissioned_year", "retirement_year", "export_share_eu") if p.get(k) is not None})
    if p.get("keys") or p.get("company"):
        keys = p.get("keys")
        if not keys:
            from .data.asset_register import search
            res = search(p["company"], limit=10)
            if not res:
                raise ValueError(f'"{p["company"]}" is not in the asset register (heavy industry only).')
            keys = [r["key"] for r in res if r["match"] == "group"] or [res[0]["key"]]
        return AN.analyse_company(keys, top_n=int(p.get("top_n", 15)), n=int(p.get("n", 1500)))
    raise ValueError("Give lat and lon (one asset) or a company / owner keys, or pass an analyst result.")


def _is_company(r: dict) -> bool:
    return "companies" in r and "asset" not in r


def _short(k: str) -> str:
    return (k.replace("Nationally Determined Contributions (NDCs)", "NDCs").replace("?C", "°C").replace("Below 2°C", "Below 2 °C"))[:24]


def _core_asset(r: dict) -> dict:
    fin = r["financial"]; scs = list(fin["scenarios"]); phys = fin["physical"]["npv"]["mean"]
    rows = {}
    for k in scs:
        s = fin["scenarios"][k]; tot = s["total_npv"]
        rows[k] = {"total": tot["mean"], "p5": tot["p5"], "p50": tot["p50"], "p95": tot["p95"], "p99": tot["p99"], "es": tot.get("es95"), "pct": tot["pct_of_value"]["mean"],
                   "phys": min(phys, tot["mean"]), "trans": s["transition_loss"]["mean"], "unab": (s.get("total_npv_unabated") or {}).get("mean"), "stranded": (s.get("stranded") or {}).get("mean"),
                   "carbon": s["transition_npv"]["mean"], "share_phys": s.get("share_physical")}
    return {"scs": scs, "rows": rows, "phys": phys, "years": fin["years"], "value": r["asset"]["value"]["mode"]}


def _core_company(r: dict) -> dict:
    scs = list(r["scenarios"]); A = r["assets"]
    phys = sum(((a.get("physical_aal_pct") or 0) / 100.0) * (a.get("value_mode") or 0) for a in A) * ANN
    rows = {}
    for k in scs:
        t = r["scenarios"][k]["total_npv"]
        rows[k] = {"total": t["mean"], "p5": t["p5"], "p50": t["p50"], "p95": t["p95"], "p99": t.get("p99"), "es": t.get("es95"), "phys": min(phys, t["mean"]), "trans": max(0.0, t["mean"] - min(phys, t["mean"]))}
    return {"scs": scs, "rows": rows, "phys": phys, "value": r.get("portfolio_value_mode")}


def _pick(scs: list, *needles) -> str:
    for n in needles:
        for k in scs:
            if n.lower() in k.lower():
                return k
    return scs[0]


def _names(r: dict, p: dict) -> str:
    c = p.get("company") or p.get("company_name")
    if c:
        return str(c).strip().title() if str(c).islower() else str(c).strip()
    first = (r.get("companies") or ["the portfolio"])[0]
    return first.split(" (")[0].split(",")[0]


def _company_assets_map(r: dict, keys_hint: Optional[list], title: str, cover: bool = False) -> Optional[bytes]:
    A = None
    try:
        from .data.asset_register import company_exposure, search
        ks = keys_hint
        if not ks and r.get("companies"):
            res = search(r["companies"][0], limit=10)
            ks = [x["key"] for x in res if x["match"] == "group"] or ([res[0]["key"]] if res else None)
        if ks:
            e = company_exposure(ks)
            A = [{"lat": a["lat"], "lon": a["lon"], "screen": a.get("flood_screen") if False else max([a["cyclone"]["level"], a.get("flood_screen") or "NEGLIGIBLE"], key=["NEGLIGIBLE", "LOW", "MODERATE", "HIGH", "CRITICAL"].index), "emissions_t": a.get("emissions_t")} for a in e["assets"]]
    except Exception:                                               # noqa: BLE001
        A = None
    if not A:
        return None
    return F.asset_map(A, "" if cover else title, "Climate TRACE v6 / EPA GHGRP asset register; screening: IBTrACS cyclone model + terrain elevation (ClimRisk engine); Natural Earth", pad=5.0 if cover else 7.0, size=(6.4, 6.4) if cover else None, legend=not cover)


def _sources_core(st) -> None:
    st.source("ngfs", "NGFS Phase 5 climate scenarios (IIASA Scenario Explorer)", "https://data.ene.iiasa.ac.at/ngfs/", "carbon price and sector output, 7 scenarios, 3 models")
    st.source("wb", "World Bank Carbon Pricing Dashboard", "https://carbonpricingdashboard.worldbank.org/", "instruments in force and pipeline")
    st.source("ibt", "NOAA IBTrACS tropical-cyclone best tracks", "https://www.ncei.noaa.gov/products/international-best-track-archive", "1980–2025")
    st.source("eman", "Emanuel (2011), Global warming effects on U.S. hurricane damage, Weather, Climate & Society 3:261", "https://doi.org/10.1175/WCAS-D-11-00007.1", "wind damage sigmoid")
    st.source("huiz", "Huizinga, de Moel & Szewczyk (2017), Global flood depth-damage functions, JRC", "https://doi.org/10.2760/16510", "industrial curve")
    st.source("jrc", "JRC CEMS-GloFAS river flood hazard maps v2.1", "https://data.jrc.ec.europa.eu/collection/id-0054", "")
    st.source("del", "Deltares Global Flood Maps v2021.06", "https://deltares-research.github.io/global-flood-maps/", "")
    st.source("flopros", "Scussolini et al. (2016), FLOPROS, NHESS 16:1049", "https://doi.org/10.5194/nhess-16-1049-2016", "")
    st.source("reg", "ClimRisk asset register", "", "Climate TRACE v6 + EPA GHGRP 2023 with IBTrACS cyclone and terrain-elevation screening")


def _oni_hook(st, oni) -> None:
    if not oni:
        return
    st.source("oni", oni["title"], oni["url"], "")
    la = oni["latest"]
    st.p(f'**{la["season"]}.** The Oceanic Niño Index reads {la["oni"]:+.2f} °C: {la["state"]}. Since 1950 it has crossed +2 °C only {oni["episodes_over_2c"]} times [[S:oni]]. '
         "The ocean is the loud part of the climate story this year. What follows is about the part that does not make the news: what climate policy and climate hazards do to the value of a specific set of industrial assets.")


def _price_figs(st, r, core, tr) -> None:
    scs = [k for k in ("Net Zero 2050", "Delayed transition", "Current Policies", "Below 2?C") if k in tr["scenarios"]]
    if scs:
        n = min(len(tr["scenarios"][k]["price_median"]) for k in scs)
        yrs = core["years"][:n]
        _try(lambda: st.figure(F.lines(yrs, {_short(k): tr["scenarios"][k]["price_median"][:n] for k in scs}, "Carbon price the plant would face, by scenario (US$/t CO₂)", "NGFS Phase 5 (IIASA), median of GCAM / MESSAGEix / REMIND, regionalised; ClimRisk engine", ylabel="US$ per tonne"),
                               caption="Median carbon price by scenario for this asset's region [[S:ngfs]]."))


def transition(p: dict, tpl: dict, brand: dict) -> dict:
    from . import story_context as C
    r = _analyst(p)
    st = S.Story(_kick("Transition risk · story"), "", byline="ClimRisk risk analyst", brand=brand)
    _sources_core(st)
    oni = C.oni_series() if p.get("context", True) else None
    if _is_company(r):
        core = _core_company(r); scs = core["scs"]; rows = core["rows"]
        names = _names(r, p)
        nz, cp, dl = _pick(scs, "Net Zero"), _pick(scs, "Current"), _pick(scs, "Delayed")
        ratio, rtxt = _ratio(rows[dl]["trans"], core["phys"])
        st.d["title"] = f"{names}: {r['n_assets']} plants and the carbon bill behind them"
        st.d["standfirst"] = (f'{r["n_assets"]} register assets worth about {_usd(core["value"])}. Expected cost to 2050 runs from {_usd(rows[cp]["total"])} under current policies to {_usd(rows[nz]["total"])} under Net Zero 2050, '
                              f'and the range between scenarios is the transition story [[S:ngfs]].')
        mp = _company_assets_map(r, p.get("keys"), f"{names}: the assets, by physical screening level", cover=True)
        if mp:
            st.cover(F.cover(st.d["title"], f"Transition risk · {dt.date.today().strftime('%d %B %Y')}", "ClimRisk asset register · Climate TRACE v6, EPA GHGRP", mp, tag="transition risk", crop=False))
        _oni_hook(st, oni)
        cards = [(str(r["n_assets"]), "register assets in the portfolio", "Climate TRACE v6 + EPA GHGRP"), (_usd(core["value"]), "estimated portfolio value", "capacity-based proxy unless supplied"),
                 (_usd(rows[cp]["total"]), "expected cost, current policies", "Monte Carlo mean, PV 2026–2050"), (_usd(rows[dl]["total"]), "expected cost, delayed transition", "Monte Carlo mean, PV 2026–2050"),
                 (_usd(rows[nz]["total"]), "expected cost, Net Zero 2050", "Monte Carlo mean, PV 2026–2050"), (_usd(rows[dl]["p95"]), "95th percentile, delayed transition", "1-in-20 outcome")]
        _try(lambda: st.figure(F.numbers(f"{names} in six numbers", "Transition and physical cost to 2050, portfolio level", cards, "ClimRisk risk analyst; NGFS Phase 5; asset register"), caption="Portfolio numbers from the engine's Monte Carlo over NGFS carbon-price and activity paths."))
        ratio_txt = (f"{rtxt}: {_usd(rows[dl]['trans'])} of expected cost is transition against about {_usd(core['phys'])} physical, in the delayed transition." if ratio is not None else rtxt)
        st.pull((f"{ratio:.0f}× quiet to loud" if ratio and ratio >= 10 else f"{ratio:.1f}× quiet to loud") if ratio is not None else "Transition vs physical", ratio_txt + " Physical is approximated as each asset's modelled annual loss over 25 years at 7%.")
        st.section("loud", "The loud risk: the scenario spread")
        sp = rows[nz]["total"] / max(rows[cp]["total"], 1.0)
        st.p(f'The headline for transition risk is a range, not a point. Across the seven NGFS scenarios the portfolio\'s expected cost to 2050 runs from {_usd(min(v["total"] for v in rows.values()))} to {_usd(max(v["total"] for v in rows.values()))}; Net Zero 2050 costs {sp:.1f} times current policies [[S:ngfs]]. '
             f'Which scenario the plan sits in is the first question for any lender.')
        _try(lambda: st.figure(F.scenario_stack([_short(k) for k in scs], [rows[k]["phys"] for k in scs], [rows[k]["total"] for k in scs], [rows[k]["p5"] for k in scs], [rows[k]["p95"] for k in scs],
                                                 f"{names}: expected loss to 2050 by scenario", "ClimRisk risk analyst Monte Carlo over NGFS Phase 5 (IIASA); physical share approximated from annual modelled loss"),
                               caption="Gold: the physical share. Blue: the full expected loss. Whisker: 5th to 95th percentile [[S:ngfs]]."))
        st.table(["Scenario", "Mean", "Median", "95th pct", "Expected shortfall"], [[k, _usd(rows[k]["total"]), _usd(rows[k]["p50"]), _usd(rows[k]["p95"]), _usd(rows[k]["es"])] for k in scs])
        st.section("quiet", "The quiet risk: which assets carry it")
        A = sorted([a for a in r["assets"] if a.get("npv_delayed_mean")], key=lambda a: -a["npv_delayed_mean"])[:10]
        top3 = sum(a["npv_delayed_mean"] for a in A[:3]) / max(sum(a["npv_delayed_mean"] for a in r["assets"] if a.get("npv_delayed_mean")), 1.0)
        st.p(f'{tpl["carbon"]} The portfolio number hides concentration: the three largest assets carry {top3:.0%} of the delayed-transition cost.')
        _try(lambda: st.figure(F.bars([a["name"].title()[:32] for a in A], [a["npv_delayed_mean"] / 1e6 for a in A], "Delayed-transition cost by asset, PV 2026–2050 (US$ m)", "ClimRisk risk analyst; NGFS Phase 5 (IIASA); asset register", unit=" m", fmt="${:,.0f}", highlight=0),
                               caption="The ten assets with the largest expected delayed-transition cost [[S:reg]]."))
        st.table(["Asset", "Country", "Sector", "Emissions", "Carbon price now", "Delayed-transition cost"],
                 [[a["name"], a["country"], a["sector"], f'{(a["emissions_t"] or 0) / 1e6:.2f} Mt', f'${a["carbon_price_now"]:.0f}/t' if a.get("carbon_price_now") is not None else "n/a", _usd(a.get("npv_delayed_mean"))] for a in A])
        st.section("audiences", "What this means, by reader")
        st.p("**For the company.** The lever is which assets abate first; the chart above is the order to ask about.")
        st.p("**For the lender or investor.** The exposure is a range. Ask which scenario the plan assumes, then read the matching row and its 95th percentile.")
        st.section("method", "Method and limitations")
        st.p(r.get("method", "Monte Carlo over NGFS carbon-price and activity paths.") + " Asset values are estimated from capacity unless supplied, so replace them for a decision-grade figure. Direct costs only; second-round effects are not included. "
             "Where the expected cost approaches the estimated value, read it as the asset not being worth running at that price, and note that the value basis is a proxy with a wide band.")
        st.h3("How ClimRisk measured this")
        st.ul(["Register → every asset snapped to Climate TRACE / EPA GHGRP with emissions and capacity.", "Policy → World Bank carbon-pricing instruments by jurisdiction and sector.", "Scenarios → NGFS Phase 5 carbon price and output paths by region and model.",
               "Monte Carlo → shared draws across assets, 7 scenarios, discounted at 7%.", "Output → expected cost, percentiles and expected shortfall, asset by asset."])
        return st.build()

    a = r["asset"]; pol = r["policy"]; tr = r["transition"]; fin = r["financial"]; core = _core_asset(r); scs = core["scs"]; rows = core["rows"]
    nz, cp, dl = _pick(scs, "Net Zero"), _pick(scs, "Current"), _pick(scs, "Delayed")
    ratio, rtxt = _ratio(rows[dl]["trans"], core["phys"])
    pn = tr["scenarios"][nz]["price_median"]
    st.d["title"] = f'{a["name"]}: the carbon bill and when it arrives'
    st.d["standfirst"] = (f'{a["name"]} emits {a["emissions_t"] / 1e3:,.0f} kt CO₂e a year. The carbon price it pays today is ${pol["effective_price_usd_t"]:.0f}/t; under Net Zero 2050 the modelled price passes ${pn[-1]:,.0f}/t by {core["years"][len(pn) - 1]}. '
                          f'Present-value cost to 2050 runs from {_usd(rows[cp]["trans"])} (current policies) to {_usd(rows[nz]["trans"])} (Net Zero 2050) [[S:ngfs]].')
    _try(lambda: st.cover(F.cover(st.d["title"], f'Transition risk · {dt.date.today().strftime("%d %B %Y")}', "NGFS Phase 5 (IIASA); World Bank Carbon Pricing Dashboard; ClimRisk risk analyst",
                                   F.lines(core["years"][:len(pn)], {_short(k): tr["scenarios"][k]["price_median"][:len(pn)] for k in scs if k in tr["scenarios"] and len(tr["scenarios"][k]["price_median"]) >= len(pn)}, "Carbon price by scenario", "NGFS Phase 5 (IIASA)", ylabel="US$/t", highlight_last=False, size=(7, 6.3)), tag="transition risk", crop=False)))
    _oni_hook(st, oni)
    cards = [(f'{a["emissions_t"] / 1e3:,.0f} kt', "CO₂e a year", "Climate TRACE v6 register"), (_usd(core["value"]), "asset value", a["value"]["basis"].split(";")[0][:48]), (f'${pol["effective_price_usd_t"]:.0f}/t', "carbon price today", "World Bank Carbon Pricing Dashboard"),
             (_usd(rows[cp]["trans"]), "transition cost, current policies", "Monte Carlo mean, PV 2026–2050"), (_usd(rows[nz]["trans"]), "transition cost, Net Zero 2050", "carbon + abatement + stranding"), (_usd(rows[nz]["stranded"]), "stranded value, Net Zero 2050", "output falls below the run-rate")]
    _try(lambda: st.figure(F.numbers(f'{a["name"]} in six numbers', "The carbon bill, as modelled", cards, "ClimRisk risk analyst; NGFS Phase 5 (IIASA); World Bank Carbon Pricing Dashboard"), caption="Each number is a Monte Carlo mean or an input from the register; sources on the image."))
    st.pull(f"{rows[nz]['trans'] / max(rows[cp]['trans'], 1):.1f}× between scenarios" if rows[cp]["trans"] else "Policy is the whole story",
            f'The same plant costs {_usd(rows[cp]["trans"])} under current policies and {_usd(rows[nz]["trans"])} under Net Zero 2050. ' + _cmp(rows[nz]["trans"], core["phys"], "Net Zero 2050 scenario"))
    st.section("loud", "The loud risk: the carbon price path")
    st.p(f'Carbon pricing is the visible part. Today {len(pol["in_force"])} instrument(s) in force reach this asset, with an effective price of ${pol["effective_price_usd_t"]:.0f}/t [[S:wb]]. The scenario paths are what changes: ' +
         "; ".join(f'{_short(k)} ${tr["scenarios"][k]["price_median"][0]:.0f}/t in {core["years"][0]} to ${tr["scenarios"][k]["price_median"][-1]:,.0f}/t in {core["years"][len(tr["scenarios"][k]["price_median"]) - 1]}' for k in (nz, dl, cp) if k in tr["scenarios"]) + " [[S:ngfs]].")
    _price_figs(st, r, core, tr)
    _try(lambda: st.figure(F.scenario_stack([_short(k) for k in scs], [0.0] * len(scs), [rows[k]["trans"] for k in scs], [fin["scenarios"][k]["transition_loss"]["p5"] for k in scs], [fin["scenarios"][k]["transition_loss"]["p95"] for k in scs],
                                             f'{a["name"]}: transition cost to 2050 by scenario', "ClimRisk risk analyst Monte Carlo over NGFS Phase 5 (IIASA), PV at 7%", note_physical="physical (not part of this report)"),
                           caption="Carbon cost paid or abated plus stranded value, with the 5th to 95th percentile [[S:ngfs]]."))
    st.table(["Scenario", "Carbon cost PV (no action)", "Cost with abatement response", "Stranded value", "Transition total"],
             [[k, _usd(rows[k]["carbon"]), _usd(((fin["scenarios"][k].get("abatement") or {}).get("response_cost_npv") or {}).get("mean")) if (fin["scenarios"][k].get("abatement") or {}).get("enabled") else "n/a", _usd(rows[k]["stranded"]), _usd(rows[k]["trans"])] for k in scs],
             caption="Present value 2026–2050, mean of the Monte Carlo draws [[S:ngfs]].")
    st.section("policy", "The quiet risk: which policy actually reaches the asset")
    if pol["in_force"]:
        st.table(["Instrument in force", "Type", "Price", "Covers this sector"], [[i["name"], i["type"], f'${i["price_usd_t"]:.0f}/t' if i.get("price_usd_t") is not None else "n/a", "yes" if i.get("covers_asset_sector") else "no"] for i in pol["in_force"]])
    st.p(" ".join(pol["notes"]) + " [[S:wb]]")
    if pol.get("pipeline"):
        st.p("In the pipeline: " + "; ".join(i["name"] for i in pol["pipeline"][:5]) + ".")
    st.p(tpl["carbon"] + (" The asset exports into the EU, so CBAM adds a border charge on top." if pol.get("cbam_exporter") else ""))
    act = fin["scenarios"][nz].get("activity_median") or tr["scenarios"][nz].get("activity_median")
    if act and tr["scenarios"].get(cp, {}).get("activity_median"):
        n = min(len(act), len(tr["scenarios"][cp]["activity_median"]), len(core["years"]))
        _try(lambda: st.figure(F.lines(core["years"][:n], {_short(nz): [100 * x for x in act[:n]], _short(cp): [100 * x for x in tr["scenarios"][cp]["activity_median"][:n]]}, "What happens to output: activity index (2026 = 100)", "NGFS Phase 5 (IIASA) sector output path for this region; ClimRisk engine", ylabel="index"),
                               caption="Output under each scenario, from the NGFS model path for the asset's sector and region. Stranded value is what the falling run-rate leaves behind [[S:ngfs]]."))
    st.section("abatement", "What the operator can do about it")
    ab = tr.get("abatement", {})
    if ab.get("levers"):
        st.table(["Lever", "Share of emissions", "Marginal cost (US$/t)", "Available from", "Basis"],
                 [[l["name"], "–".join(f"{x:.0%}" for x in l["share"]) if isinstance(l["share"], (list, tuple)) else f'{l["share"]:.0%}', "–".join(f"{x:.0f}" for x in l["mac"]) if isinstance(l["mac"], (list, tuple)) else str(l["mac"]), str(l["avail"]), l.get("basis", "")] for l in ab["levers"]],
                 caption="Library ranges unless the operator supplies its own levers. Get site quotes and rerun.")
    if rows[nz].get("unab") and rows[nz]["total"] is not None:
        st.p(f'Under Net Zero 2050 the model abates when the price reaches each lever\'s cost. Without any abatement the mean total loss is {_usd(rows[nz]["unab"])}, against {_usd(rows[nz]["total"])} with the sector levers: {_usd(rows[nz]["unab"] - rows[nz]["total"])} avoided.')
    for act_ in r.get("actions", []):
        if act_["hazard"] in ("carbon_cost", "abatement"):
            st.p(f'**{act_["problem"]}** {act_["action"]}')
    st.section("audiences", "What this means, by reader")
    st.p("**For the operator.** The cost depends on timing as much as price: abatement started late is paid for at the highest prices.")
    st.p(f'**For the lender or investor.** The question is which scenario the plan sits in, and whether the stranded-value row ({_usd(rows[nz]["stranded"])} under Net Zero) is covered by the asset\'s remaining life.')
    st.section("method", "Method and limitations")
    st.p("A Monte Carlo over NGFS Phase 5 carbon-price and sector-output paths [[S:ngfs]], regionalised to the asset and run for seven scenarios. The carbon cost is emissions times the price the plant faces, passed through to customers in part; abatement levers switch on when the price reaches their marginal cost; "
         "stranded value is the asset value lost when output falls below the run-rate that justified it. " + str((tr.get("assumptions") or {}).get("emissions", "") if isinstance(tr.get("assumptions"), dict) else "") + " " + " ".join(r.get("caveats", [])[:2]))
    st.h3("How ClimRisk measured this")
    st.ul(["Register → asset, emissions, capacity and value basis.", "Policy → World Bank instruments in force and pipeline for the jurisdiction and sector.", "Scenarios → NGFS Phase 5 price and output paths, three models, seven scenarios.",
           "Monte Carlo → draws on price, pass-through, charged share and value; discounted at 7%.", "Abatement → sector lever library with marginal costs; the plant abates when the price crosses each one."])
    return st.build()


def financial(p: dict, tpl: dict, brand: dict) -> dict:
    from . import story_context as C
    r = _analyst(p)
    st = S.Story(_kick("Financial risk · story"), "", byline="ClimRisk risk analyst", brand=brand)
    _sources_core(st)
    oni = C.oni_series() if p.get("context", True) else None
    if _is_company(r):
        core = _core_company(r); scs = core["scs"]; rows = core["rows"]
        names = _names(r, p)
        worst = max(scs, key=lambda k: rows[k]["total"]); dl = _pick(scs, "Delayed")
        ratio, rtxt = _ratio(rows[dl]["trans"], core["phys"])
        st.d["title"] = f"{names}: the loss distribution" + (f", and the part nobody prices ({ratio:.0f}×)" if ratio and ratio >= 10 else (f", and the part nobody prices ({ratio:.1f}×)" if ratio and ratio >= 1.5 else ""))
        st.d["standfirst"] = (f'Portfolio value about {_usd(core["value"])} across {r["n_assets"]} assets. Expected loss to 2050 is {_usd(rows[worst]["total"])} in the worst scenario ({_short(worst)}), '
                              f'with a 95th-percentile outcome of {_usd(rows[worst]["p95"])} [[S:ngfs]].')
        mp = _company_assets_map(r, p.get("keys"), f"{names}: the assets, by physical screening level", cover=True)
        mp_body = _company_assets_map(r, p.get("keys"), f"{names}: the assets, by physical screening level")
        if mp:
            st.cover(F.cover(st.d["title"], f"Financial risk · {dt.date.today().strftime('%d %B %Y')}", "ClimRisk risk analyst · asset register", mp, tag="financial risk", crop=False))
        if mp_body:
            st.figure(mp_body, caption="Every register asset, coloured by the higher of its cyclone and flood screening level and sized by emissions [[S:reg]].")
        _oni_hook(st, oni)
        cards = [(_usd(core["value"]), "estimated portfolio value", "capacity-based proxy unless supplied"), (_usd(rows[dl]["total"]), "expected loss, delayed transition", "Monte Carlo mean, PV 2026–2050"), (_usd(rows[dl]["p95"]), "95th percentile", "1-in-20 outcome"),
                 (_usd(rows[dl]["es"]), "expected shortfall", "mean of the worst 5% of draws"), (_usd(core["phys"]), "physical share (approx.)", "annual loss × 25 years at 7%"), (_usd(rows[dl]["trans"]), "transition share", "total less physical")]
        _try(lambda: st.figure(F.numbers(f"{names} in six numbers", "Where the expected loss comes from", cards, "ClimRisk risk analyst; NGFS Phase 5; asset register"), caption="Portfolio-level Monte Carlo summary."))
        st.pull((f"{ratio:.0f}× quiet to loud" if ratio and ratio >= 10 else f"{ratio:.1f}× quiet to loud") if ratio is not None else "Quiet vs loud", rtxt + f": {_usd(rows[dl]['trans'])} of transition cost against {_usd(core['phys'])} of physical damage in the delayed transition." if ratio is not None else rtxt)
        st.section("loud", "The loud risk: the expected loss")
        _try(lambda: st.figure(F.scenario_stack([_short(k) for k in scs], [rows[k]["phys"] for k in scs], [rows[k]["total"] for k in scs], [rows[k]["p5"] for k in scs], [rows[k]["p95"] for k in scs],
                                                 f"{names}: expected loss to 2050 by scenario", "ClimRisk risk analyst Monte Carlo over NGFS Phase 5 (IIASA); physical share approximated from annual modelled loss"), caption="Gold: physical. Blue: total. Whisker: 5th to 95th percentile [[S:ngfs]]."))
        near_all = rows[worst]["total"] >= 0.9 * (core["value"] or 1e30)
        if near_all:
            st.callout(f'In {_short(worst)} the expected cost is {_pct(100 * rows[worst]["total"] / core["value"])} of the estimated portfolio value. Read that as the assets not being worth running at those prices, not as a forecast of total loss, and note the value is a proxy with a wide band.', "warn")
        st.section("quiet", "The quiet risk: the tail and the concentration")
        st.table(["Scenario", "Mean", "Median", "95th pct", "Expected shortfall"], [[k, _usd(rows[k]["total"]), _usd(rows[k]["p50"]), _usd(rows[k]["p95"]), _usd(rows[k]["es"])] for k in scs])
        A = sorted([a for a in r["assets"] if a.get("physical_aal_pct") is not None], key=lambda a: -a["physical_aal_pct"])[:8]
        st.table(["Asset", "Country", "Physical annual loss (% of value)", "Cyclone", "River", "Coast"], [[a["name"], a["country"], f'{a["physical_aal_pct"]:.2f}%', a.get("cyclone", ""), a.get("river", ""), a.get("coast", "")] for a in A],
                 caption=f'Flood depths were read live for the {r.get("flood_assessed_assets", 0)} largest emitters only; the others use screening values.')
        st.section("audiences", "What this means, by reader")
        st.p("**For the lender.** Size covenants to the 95th percentile and expected shortfall, not the mean.")
        st.p("**For the investor.** The mean hides the tail; the tail is where a few coastal assets and the highest-emitting plants sit.")
        st.section("method", "Method and limitations")
        st.p(r.get("method", "Monte Carlo loss distribution.") + " Direct physical damage and carbon cost only; business interruption is not included. The physical share at company level is approximated as each asset's modelled annual loss times a 25-year annuity at 7%.")
        st.h3("How ClimRisk measured this")
        st.ul(["Register → assets, emissions, capacity.", "Physical → IBTrACS cyclone set, terrain screening, live JRC and Deltares flood depths for the largest emitters.", "Transition → NGFS price and output paths.", "Monte Carlo → shared draws, expected loss, percentiles, expected shortfall."])
        return st.build()

    a = r["asset"]; fin = r["financial"]; core = _core_asset(r); scs = core["scs"]; rows = core["rows"]; ph = fin["physical"]
    dl = _pick(scs, "Delayed"); d = rows[dl]
    ratio, rtxt = _ratio(d["trans"], core["phys"])
    st.d["title"] = f'{a["name"]}: what the loss distribution says' + (f', and the part nobody prices ({ratio:.1f}×)' if ratio and ratio >= 1.5 else "")
    st.d["standfirst"] = r["headline"]
    cv = _try(lambda: F.cover(st.d["title"], f'Financial risk · {dt.date.today().strftime("%d %B %Y")}', "ClimRisk risk analyst; NGFS Phase 5 (IIASA)",
                              F.distribution(fin["scenarios"][dl]["histogram"]["edges"], fin["scenarios"][dl]["histogram"]["counts"], "Loss distribution", "ClimRisk Monte Carlo", {"mean": d["total"], "P95": d["p95"]}, scale=1e6), tag="financial risk", crop=False))
    if cv:
        st.cover(cv)
    _oni_hook(st, oni)
    cards = [(_usd(core["value"]), "asset value", a["value"]["basis"].split(";")[0][:48]), (_usd(d["total"]), f"expected loss, {_short(dl).lower()}", "Monte Carlo mean, PV 2026–2050"), (_usd(d["p95"]), "1-in-20 outcome (P95)", "Monte Carlo"),
             (_usd(d["es"]), "expected shortfall (worst 5%)", "Monte Carlo"), (f'{ph["aal_pct"]:.2f}%', "physical annual loss, of value", "IBTrACS × Emanuel; JRC / Deltares × FLOPROS"), (_usd(core["phys"]), "physical share of the expected loss", "present value 2026–2050")]
    _try(lambda: st.figure(F.numbers(f'{a["name"]} in six numbers', "The loss distribution, as modelled", cards, "ClimRisk risk analyst; NGFS Phase 5 (IIASA); JRC; Deltares; IBTrACS"), caption="Monte Carlo means and percentiles; each source is stamped on the image."))
    st.pull((f"{ratio:.0f}× quiet to loud" if ratio and ratio >= 10 else f"{ratio:.1f}× quiet to loud") if ratio is not None else "Quiet vs loud", (rtxt + f": {_usd(d['trans'])} of transition loss against {_usd(core['phys'])} of physical damage ({_short(dl).lower()})." if ratio is not None else rtxt))
    st.section("loud", "The loud risk: physical damage, priced")
    rp = ph.get("annual_loss_rp") or {}
    st.p(f'Physical damage is the risk that makes the news. Modelled average annual damage at {a["name"]} is **{ph["aal_pct"]:.2f}% of value** ({_usd(ph["aal_usd"])} a year), and its present value to 2050 is {_usd(core["phys"])}'
         + (f'. A 1-in-100-year event year costs {_usd(rp["100"])}, {100 * ph["annual_damage_ratio_rp"]["100"]:.0f}% of value.' if "100" in rp else ".") + " Leading hazard: " + ", ".join(f"{k} ({v:.2f}% a year)" for k, v in (ph.get("by_hazard_aal_pct") or {}).items()) + " [[S:ibt]][[S:eman]].")
    if rp:
        _try(lambda: st.figure(F.bars([f"1-in-{k} year" for k in rp], [v / 1e6 for v in rp.values()], "Physical loss in a single year, by return period", "ClimRisk risk analyst: IBTrACS × Emanuel (2011) × JRC flood × FLOPROS defences", unit=" m", fmt="${:,.0f}", horizontal=False,
                                      highlight=list(rp).index("100") if "100" in rp else None, xlabel="US$ million"), caption="Single-year direct physical loss. Direct damage only [[S:eman]][[S:huiz]]."))
    st.section("quiet", "The quiet risk: the transition, and what the mean hides")
    _try(lambda: st.figure(F.scenario_stack([_short(k) for k in scs], [rows[k]["phys"] for k in scs], [rows[k]["total"] for k in scs], [rows[k]["p5"] for k in scs], [rows[k]["p95"] for k in scs],
                                             f'{a["name"]}: expected loss to 2050 by scenario', "ClimRisk risk analyst Monte Carlo over NGFS Phase 5 (IIASA); physical + transition, PV at 7%"), caption="Gold: physical damage. Blue: total expected loss. Whisker: 5th to 95th percentile [[S:ngfs]]."))
    for para in r["narrative"][:3]:
        st.p(para)
    h = fin["scenarios"][dl]["histogram"]
    _try(lambda: st.figure(F.distribution(h["edges"], h["counts"], f"Loss distribution, {_short(dl)}", "ClimRisk risk analyst Monte Carlo over NGFS Phase 5 (IIASA)", {"mean": d["total"], "P95": d["p95"], "expected shortfall": d["es"]}, scale=1e6),
                           caption="Share of Monte Carlo draws by total loss. The right tail is where lender covenants live."))
    st.table(["Scenario", "Expected loss", "Share of value", "Median", "95th pct", "99th pct", "Expected shortfall"],
             [[k, _usd(rows[k]["total"]), _pct(rows[k]["pct"], 1), _usd(rows[k]["p50"]), _usd(rows[k]["p95"]), _usd(rows[k]["p99"]), _usd(rows[k]["es"])] for k in scs], caption="Present value 2026–2050 [[S:ngfs]].")
    st.p(tpl["chain"])
    st.section("inaction", "What inaction costs")
    if d.get("unab"):
        st.p(f'Without any abatement the mean loss in the {_short(dl).lower()} would be {_usd(d["unab"])}, against {_usd(d["total"])} with the sector levers: {_usd(d["unab"] - d["total"])} avoided.')
    for act in r.get("actions", [])[:3]:
        st.p(f'**{act["problem"]}** {act["action"]}')
    st.section("audiences", "What this means, by reader")
    st.p(f'**For the lender.** The 95th percentile ({_usd(d["p95"])}) and expected shortfall ({_usd(d["es"])}) are the covenant numbers, not the mean.')
    st.p("**For the insurer.** The physical share is a direct-damage figure; check deductibles against the single-year losses above.")
    st.p("**For the investor.** The distribution is wide; the decision depends on which scenario the plan assumes.")
    st.section("method", "Method and limitations")
    st.p(f'{fin["n_draws"]:,} Monte Carlo draws, discounted at {fin["discount_rate"]:.0%}, over NGFS Phase 5 scenarios [[S:ngfs]] for the transition and the engine\'s cyclone and flood models for the physical side [[S:ibt]][[S:jrc]][[S:del]]. ' + " ".join(r.get("caveats", [])[:3]))
    st.h3("How ClimRisk measured this")
    st.ul(["Physical → IBTrACS cyclone event set, JRC river and Deltares coastal depth, FLOPROS defences, damage curves.", "Transition → NGFS price and output paths, abatement levers, stranding.", "Monte Carlo → joint draws; expected loss, percentiles, expected shortfall.", "Value → the register's capacity-based estimate unless the operator supplies a value."])
    return st.build()


# ═════════════════════════════════════════════════════════════════════════════
# 4/5. LCA and ISO 14067
# ═════════════════════════════════════════════════════════════════════════════

def _lca_run(p: dict) -> dict:
    if p.get("result"):
        return p["result"]
    prod, par, n = p.get("product", "cement"), dict(p.get("params") or {}), int(p.get("n", 2000))
    from .lca import products
    if prod == "cement":
        from .lca.cement import CementPlant, assess
        fields = {f.name for f in dataclasses.fields(CementPlant)}
        return assess(CementPlant(**{k: v for k, v in par.items() if k in fields}), n=n, cbam_year=int(par.get("cbam_year", 2026)))
    country = par.get("country")
    if prod == "steel":
        return products.steel(par["route"], kwh_per_t=float(par["kwh_per_t"]), country=country, g_per_kwh=par.get("g_per_kwh"), n=n)
    if prod == "aluminium":
        return products.aluminium(kwh_per_t=float(par["kwh_per_t"]), country=country, g_per_kwh=par.get("g_per_kwh"), n=n)
    if prod == "ammonia":
        return products.ammonia(gas_gj_per_t=par.get("gas_gj_per_t"), kwh_per_t=float(par["kwh_per_t"]), country=country, g_per_kwh=par.get("g_per_kwh"), n=n)
    raise ValueError("product must be cement, steel, aluminium or ammonia")


def _lca_sources(st, r) -> None:
    st.source("std", "ISO 14067:2018 Greenhouse gases — Carbon footprint of products", "https://www.iso.org/standard/71206.html", "")
    st.source("ipcc", "IPCC AR6 WG1 Ch.7, GWP100 values", "https://www.ipcc.ch/report/ar6/wg1/chapter/chapter-7/", "")
    for i, f in enumerate(r.get("factor_sources", [])[:6]):
        st.source(f"f{i}", str(f), "", "emission-factor source")


def lca(p: dict, tpl: dict, brand: dict) -> dict:
    r = _lca_run(p)
    u = r["uncertainty"]; fp = r["footprint_kg_co2e_per_t"]
    det = sorted(r["detail"], key=lambda d: -d["kg_co2e"]); top = det[0]; share = 100 * top["kg_co2e"] / fp if fp else 0
    prod = r["product"]
    title = f'One tonne of {prod}: {fp:,.0f} kg of CO₂e, and the line that makes {share:.0f}% of it'
    st = S.Story(_kick("Life cycle assessment · story"), title, f'Cradle-to-gate footprint of {r["declared_unit"]} made in {r["country"]}: {fp:,.0f} kg CO₂e (5th–95th percentile {u["p5"]:,.0f}–{u["p95"]:,.0f}). '
                 f'The largest single line is {top["item"].lower()} ({top["stage"]}), at {top["kg_co2e"]:,.0f} kg.', byline="ClimRisk LCA engine", brand=brand)
    _lca_sources(st, r)
    bs = {k: v for k, v in r["by_stage"].items() if v}
    cv = _try(lambda: F.cover(title, f'Life cycle assessment · {dt.date.today().strftime("%d %B %Y")}', "ClimRisk LCA engine · ISO 14067 method",
                               F.gauge_bar([(k, v) for k, v in bs.items()], "Where the footprint sits (kg CO₂e per tonne)", "ClimRisk LCA engine; emission factors as cited"), tag="life cycle assessment", crop=False))
    if cv:
        st.cover(cv)
    st.p(f'**{dt.date.today().year}.** Every tonne of {prod} that leaves a gate carries a number, and almost nobody sees how it is made up. For this one, made in {r["country"]}, the engine adds up {len(det)} lines into {fp:,.0f} kg CO₂e. '
         f'One line, {top["item"].lower()}, accounts for {share:.0f}% of it [[S:std]].')
    cards = [(f"{fp:,.0f} kg", "CO₂e per tonne", "ClimRisk LCA engine"), (f'{u["p5"]:,.0f}–{u["p95"]:,.0f}', "5th–95th percentile, kg", "Monte Carlo on factor uncertainty"), (f'{share:.0f}%', f'from {top["item"].lower()[:30]}', top["basis"][:56]),
             (f'{r["biogenic_co2_kg_per_t"]:,.1f} kg', "biogenic CO₂, reported separately", "ISO 14067 §6.3.5"), (r["data_quality"]["rating"], f'data quality, score {r["data_quality"]["weighted_score"]}', "pedigree matrix"), (f'{r["data_quality"]["primary_data_share"]:.0%}', "primary-data share", "of emissions")]
    _try(lambda: st.figure(F.numbers(f"One tonne of {prod} in six numbers", "A cradle-to-gate footprint, taken apart", cards, "ClimRisk LCA engine; factor sources as listed"), caption="Each number is computed from the inventory. Sources are stamped on the image."))
    st.pull(f'{top["item"]}: {share:.0f}% of the footprint', f'The headline number is one figure; the decision is in the line that dominates it. {tpl["carbon"]}')
    st.section("loud", "The loud number: where it comes from")
    _try(lambda: st.figure(F.bars([f'{d["item"][:34]} ({d["stage"]})' for d in det[:9]], [d["kg_co2e"] for d in det[:9]], "Largest lines in the inventory (kg CO₂e per tonne)", "ClimRisk LCA engine; factor sources as cited", highlight=0, fmt="{:,.0f}"),
                           caption="Largest contributions in the inventory; each line keeps its basis."))
    st.table(["Stage", "Item", "kg CO₂e / t", "Basis"], [[d["stage"], d["item"], f'{d["kg_co2e"]:.1f}', d["basis"]] for d in det[:10]])
    st.section("quiet", "The quiet part: what moves it, and what it hides")
    if r.get("clinker"):
        c = r["clinker"]
        st.p(f'Clinker is {c["clinker_ratio"]:.0%} of the cement. Calcination alone releases {c["calcination_kg_co2_per_t_clinker"]:.0f} kg per tonne of clinker, and that is chemistry: {c["calcination_basis"]}. {tpl["carbon"]}')
    else:
        st.p(tpl["carbon"])
    st.p(f'The uncertainty band is {100 * (u["p95"] - u["p5"]) / 2 / u["p50"]:.0f}% either side of the median ({u["method"][:140]}). Excluded: ' + "; ".join(r.get("exclusions", [])[:3]) + ".")
    if r.get("trajectory"):
        t = r["trajectory"]
        sc = {k: v for k, v in t["scenarios"].items() if k in ("Net Zero 2050", "Current Policies", "Delayed transition")} or dict(list(t["scenarios"].items())[:3])
        n = min(len(v) for v in sc.values())
        _try(lambda: st.figure(F.lines(t["years"][:n], {_short(k): v[:n] for k, v in sc.items()}, "Footprint if only the grid decarbonises (kg CO₂e per tonne)", "NGFS Phase 5 (IIASA) grid pathway; ClimRisk LCA engine; process held constant", ylabel="kg CO₂e per tonne"),
                               caption="Grid-pathway effect only: the plant's own process is held constant, which is exactly why the line flattens."))
    if r.get("cbam"):
        c = r["cbam"]
        st.section("cbam", "What the border sees")
        st.source("cbamreg", "Regulation (EU) 2023/956 establishing a carbon border adjustment mechanism", "https://eur-lex.europa.eu/eli/reg/2023/956/oj", "")
        st.p(f'Under EU CBAM the embedded emissions are {c["direct_tco2_per_t_cement"] + c["indirect_tco2_per_t_cement"]:.2f} t per tonne ({c["direct_tco2_per_t_cement"]:.2f} direct, {c["indirect_tco2_per_t_cement"]:.2f} indirect). '
             f'In {c["year"]} the phase-in share is {c["cbam_phase_in_share"]:.1%}, giving an indicative cost of ${c["indicative_cbam_cost_usd_per_t_cement"]:.2f} per tonne at an EU ETS price of ${c["eu_ets_price_usd_t"]:.0f}/t [[S:cbamreg]]. ' + " ".join(c.get("notes", [])[:1]))
    st.section("audiences", "What this means, by reader")
    st.p(f'**For the producer.** The hotspot rows are the order of work; the biggest line ({top["item"].lower()}) is rarely the one with the cheapest fix.')
    st.p("**For the buyer.** Ask for the 5th–95th percentile band and the data-quality rating with any single number.")
    st.section("method", "Method and limitations")
    st.p(f'{r["standard"]} Emission factors are cited per line [[S:f0]]. Data quality is rated {r["data_quality"]["rating"]}, with {r["data_quality"]["primary_data_share"]:.0%} of emissions from primary data. ' + " ".join(r.get("notes", [])[:3]))
    st.h3("How ClimRisk measured this")
    st.ul(["Inventory → fuels, electricity, process and transport flows per tonne, with factors from named sources.", "GWP → IPCC AR6 100-year values [[S:ipcc]].", "Uncertainty → Monte Carlo on factor and activity-data ranges.", "Output → total, stage split, hotspots and the sensitivity of the total to each line."])
    return st.build()


def iso14067(p: dict, tpl: dict, brand: dict) -> dict:
    from .lca import cfp
    if p.get("result"):
        r = p["result"]
    else:
        prod, par = p.get("product", "cement"), dict(p.get("params") or {})
        from .lca import products
        if prod == "cement":
            from .lca.cement import CementPlant, inventory
            fields = {f.name for f in dataclasses.fields(CementPlant)}
            plant = CementPlant(**{k: v for k, v in par.items() if k in fields}); inv, country = inventory(plant), plant.resolved_country()
        else:
            fn = {"steel": products.steel_inventory, "aluminium": products.aluminium_inventory, "ammonia": products.ammonia_inventory, "custom": products.custom_inventory}[prod]
            inv, _ = fn(**{k: v for k, v in par.items() if k != "cbam_year"}); country = par.get("country")
        r = cfp.run_study(inv, p.get("study") or {"boundary": "cradle_to_gate"}, prod, country, n=int(p.get("n", 2000)))
    rep = cfp.report(r, p.get("study"))
    S5, S7, S8 = rep["sections"]["5_results"], rep["sections"]["7_interpretation"], rep["sections"]["8_review_and_communication"]
    u = r["uncertainty"]; b = r["boundary"]
    title = f'{r["product"].title()} carbon footprint study: {r["cfp_kg_co2e"]:,.0f} kg CO₂e per {r["declared_unit"].replace("1 ", "")}'
    hs = S7["hotspots_to_80pct"]
    st = S.Story(_kick("ISO 14067 · carbon footprint study"), title,
                 f'{b["label"]}. {r["cfp_kg_co2e"]:,.0f} kg CO₂e (P5–P95 {u["p5"]:,.0f}–{u["p95"]:,.0f}), {r["gwp"]}, reference year {r["reference_year"]}. {len(hs)} item(s) make up 80% of the footprint. '
                 + ("A full CFP." if b.get("is_full_cfp") else "A partial CFP: stages after the boundary are not included."), byline="ClimRisk LCA engine", brand=brand)
    st.source("iso", "ISO 14067:2018 Greenhouse gases — Carbon footprint of products", "https://www.iso.org/standard/71206.html", "")
    for i, f in enumerate(r.get("factor_sources", [])[:6]):
        st.source(f"f{i}", str(f), "", "emission-factor source")
    bs = {k: v for k, v in (r.get("a1_a3_by_component") or r["by_stage"]).items() if v}
    cv = _try(lambda: F.cover(title, f'ISO 14067 study · {dt.date.today().strftime("%d %B %Y")}', "ClimRisk LCA engine · reproducible: study fingerprint " + str(r["fingerprint"]), F.gauge_bar([(k, v) for k, v in bs.items()], "Contribution by component (kg CO₂e)", "ClimRisk LCA engine"), tag="iso 14067", crop=False))
    if cv:
        st.cover(cv)
    cards = [(f'{r["cfp_kg_co2e"]:,.0f} kg', "CO₂e per declared unit", b["label"][:50]), (f'{u["p5"]:,.0f}–{u["p95"]:,.0f}', "5th–95th percentile", "Monte Carlo"), (str(len(hs)), "items make 80% of the total", "ISO 14067 §7 hotspot analysis"),
             (r["data_quality"]["overall_rating"], f'data quality, score {r["data_quality"]["overall_score"]}', r["data_quality"]["scale"][:50]), (f'{r["data_quality"]["primary_data_share"]:.0%}', "primary-data share", "of emissions"), (str(r["reference_year"]), "reference year", r["gwp"][:40])]
    _try(lambda: st.figure(F.numbers("The study in six numbers", "Goal, scope and results at a glance", cards, "ClimRisk LCA engine; ISO 14067:2018"), caption="Study summary computed from the inventory."))
    st.pull(f'{hs[0]["item"]}: {hs[0]["share_pct"]:.0f}% of the footprint', f'{len(hs)} item(s) reach 80% cumulatively. The study is reproducible: the same inputs give fingerprint {r["fingerprint"]}.')
    st.section("goal", "Goal and scope")
    g, sc = rep["sections"]["2_goal"], rep["sections"]["3_scope"]
    st.p(f'Intended application: {g["intended_application"]}. Audience: {g["intended_audience"]}. {g["note"]} [[S:iso]]')
    st.table(["Item", "Value"], [["Product", sc["product"]], ["Declared unit", sc["declared_unit"]], ["System boundary", b["label"]], ["Excluded stages", ", ".join(b.get("excluded_stages", [])) or "none"],
                                 ["Time period", sc["time_period"]], ["Geography", sc["geography"]], ["GWP basis", sc["gwp"]], ["Allocation", str(sc["allocation"])]])
    st.section("results", "Results")
    _try(lambda: st.figure(F.bars(list(bs), list(bs.values()), "Contribution by component (kg CO₂e)", "ClimRisk LCA engine; factor sources as cited", fmt="{:,.0f}", highlight=0), caption="Calculated from the inventory flows."))
    gt = {k.replace("_", " "): v for k, v in S5["by_ghg_type"].items() if v}
    st.table(["GHG type (ISO 14067 §6.3.5)", "kg CO₂e"], [[k, f"{v:,.2f}"] for k, v in gt.items()], caption="Biogenic CO₂ emissions and removals are reported separately and are not included in the fossil total.")
    st.section("interpretation", "Interpretation: hotspots, sensitivity and methodological choices")
    st.table(["Stage", "Item", "kg CO₂e", "Share", "Cumulative"], [[h["stage"], h["item"], f'{h["kg_co2e"]:.1f}', f'{h["share_pct"]:.1f}%', f'{h["cumulative_pct"]:.1f}%'] for h in hs], caption="Items making up 80% of the footprint.")
    sens = S7["sensitivity"][:6]
    _try(lambda: st.figure(F.bars([x["item"][:34] for x in sens], [x["effect_of_plus_10pct_on_total_pct"] for x in sens], "Sensitivity: effect on the total of +10% on each item (%)", "ClimRisk LCA engine, one-at-a-time perturbation", unit="%", fmt="{:,.2f}", highlight=0), caption="Which input errors move the total most."))
    mc = r.get("methodological_choices") or S7.get("methodological_choices")
    if mc:
        st.table(["Methodological choice", "Total kg CO₂e", "Change vs selected"], [[c["choice"], f'{c.get("total_kg_co2e", c.get("total_kg_co2e_per_unit", 0)):.1f}', f'{c.get("change_vs_selected_pct", 0):+.1f}%'] for c in mc])
    st.section("quality", "Data quality and cut-off")
    dq = r["data_quality"]
    st.p(f'Overall data quality: {dq["overall_rating"]} (score {dq["overall_score"]}, {dq["scale"]}); primary data cover {dq["primary_data_share"]:.0%} of emissions.')
    st.table(["Criterion", "Score", "Rating", "Basis"], [[c["criterion"], c["score"], c["rating"], c["basis"]] for c in dq["criteria"]])
    st.p(f'Cut-off: {r["cutoff"]["status"]}. {r["cutoff"]["note"]}')
    st.section("review", "Review and communication")
    st.p(S8["verification"]["statement"])
    st.callout(S8["communication_statement"], "info")
    st.section("method", "Method and limitations")
    st.p(" ".join(list(r.get("notes", []))[:3] + list(r.get("exclusions", []))[:2]) + f' Study fingerprint {r["fingerprint"]}: the same inputs reproduce this result.')
    st.h3("How ClimRisk measured this")
    st.ul(["Inventory → named flows with emission factors from cited sources.", "Boundary and allocation → as declared above, per ISO 14067 §6.", "Uncertainty → Monte Carlo on factor and activity ranges; sensitivity by one-at-a-time perturbation.", "Reproducibility → a fingerprint of the inputs and factor library."])
    return st.build()


# ═════════════════════════════════════════════════════════════════════════════
# 6. CARBON AUDIT
# ═════════════════════════════════════════════════════════════════════════════

def carbon_audit(p: dict, tpl: dict, brand: dict) -> dict:
    if p.get("result"):
        r = p["result"]
    else:
        from .carbon.inventory import calculate, from_dict
        r = calculate(from_dict(p["inventory"]))
    t = r["totals_t"]; q = r["data_quality"]
    cats = {k: v for k, v in r["by_category_t"].items() if v}
    top_k = max(cats, key=cats.get) if cats else "n/a"; top_sh = 100 * cats.get(top_k, 0) / max(t["total_location"], 1.0)
    miss = [c for c in r["completeness"] if not c["reported"]]
    title = f'{r["organisation"]}: {t["total_location"]:,.0f} tonnes of CO₂e, and how far to trust the number'
    st = S.Story(_kick("Carbon audit · story"), title, f'Reporting year {r["reporting_year"]}, {r["consolidation"].replace("_", " ")} consolidation, GWP basis {r["gwp"]}. Scope 1 {t["scope1"]:,.0f} t, scope 2 {t["scope2_location"]:,.0f} t (location) or {t["scope2_market"]:,.0f} t (market), scope 3 {t["scope3"]:,.0f} t. '
                 f'{len(miss)} of {len(r["completeness"])} categories have no data.', byline="ClimRisk carbon engine", brand=brand)
    for i, s_ in enumerate(r.get("standards", [])[:5]):
        st.source(f"s{i}", str(s_), "", "")
    st.source("lib", f'ClimRisk emission-factor library {r["factor_library"]}', "", "UK DESNZ 2026, US EPA eGRID 2023, Green-e residual mix, AIB, Ember")
    cv = _try(lambda: F.cover(title, f'Carbon audit · {dt.date.today().strftime("%d %B %Y")}', "ClimRisk carbon engine · fingerprint " + str(r["fingerprint"]),
                               F.gauge_bar([("Scope 1", t["scope1"]), ("Scope 2 (location)", t["scope2_location"]), ("Scope 3", t["scope3"])], "Emissions by scope (t CO₂e)", "ClimRisk carbon engine; factor library " + str(r["factor_library"])), tag="carbon audit", crop=False))
    if cv:
        st.cover(cv)
    st.p(f'**{r["reporting_year"]}.** An emissions number is only as good as the lines behind it. For {r["organisation"]} the engine audits {len(r["lines"])} lines into {t["total_location"]:,.0f} t CO₂e; '
         f'{top_k} is {top_sh:.0f}% of it, and {len(miss)} of {len(r["completeness"])} categories carry no data at all [[S:lib]].')
    cards = [(f'{t["scope1"]:,.0f} t', "scope 1", "direct emissions"), (f'{t["scope2_location"]:,.0f} t', "scope 2, location-based", "grid-average factors"), (f'{t["scope2_market"]:,.0f} t', "scope 2, market-based", "contractual / residual-mix factors"),
             (f'{t["scope3"]:,.0f} t', "scope 3 reported", f'{len(r["completeness"]) - len(miss)} of {len(r["completeness"])} categories'), (f'{q["weighted_score"]}', "data-quality score", "1 best, 5 worst"), (f'{q["share_primary"]:.0%}', "primary-data share", "the rest uses spend or default factors")]
    _try(lambda: st.figure(F.numbers(f'{r["organisation"]} in six numbers', "A carbon inventory, audited line by line", cards, "ClimRisk carbon engine; GHG Protocol; factor library " + str(r["factor_library"])), caption="Each number is a sum of audited lines."))
    st.pull(f'{len(miss)} categories with no data', f'{top_k} carries {top_sh:.0f}% of the reported total. A category with no data must be explained, not dropped; the auditor will ask first.')
    st.section("loud", "The loud number")
    _try(lambda: st.figure(F.bars([k.replace("Cat ", "C")[:36] for k in list(cats)[:10]], list(cats.values())[:10], "Emissions by category (t CO₂e)", "ClimRisk carbon engine; GHG Protocol categories", fmt="{:,.0f}", highlight=list(cats).index(top_k) if top_k in list(cats)[:10] else None),
                           caption="Every bar is the sum of audited lines; each line keeps its input, factor and evidence reference."))
    if r.get("intensity"):
        st.p(f'Intensity: {r["intensity"]["t_per_musd_revenue"]:.1f} t per US$ m revenue and {r["intensity"]["t_per_employee"]:.2f} t per employee ({r["intensity"]["basis"]}).')
    st.section("quiet", "The quiet part: what is missing, and what an auditor will ask")
    st.p(f'{len(miss)} of {len(r["completeness"])} categories have no data. ' + "; ".join(f'{c["name"]} ({c["status"]})' for c in miss[:6]) + ".")
    st.table(["Level", "Check", "Message"], [[c["level"], c["check"], c["message"]] for c in r["checks"]])
    lines = sorted(r["lines"], key=lambda l: -(l["t_co2e"] or 0))[:8]
    st.table(["Line", "Entity", "Scope", "t CO₂e", "Factor", "Evidence", "DQ"], [[l["description"][:40], l["entity_name"], l["scope"], f'{l["t_co2e"]:,.1f}', (l["factor"] or {}).get("id", (l["factor"] or {}).get("name", ""))[:30] if isinstance(l["factor"], dict) else str(l["factor"])[:30], l.get("evidence") or "none", l["dq"]] for l in lines],
             caption="Largest lines. A line with no evidence reference will be queried first [[S:lib]].")
    st.section("audiences", "What this means, by reader")
    st.p("**For the sustainability lead.** The completeness table is the work list before assurance.")
    st.p("**For the assurer.** Fingerprint " + str(r["fingerprint"]) + " reproduces the calculation from the same inputs and factor library.")
    st.section("method", "Method and limitations")
    st.p(f'Scope 2 is reported both ways; market-based uses residual-mix factors where supplied. Primary data cover {q["share_primary"]:.0%} of the total; the rest uses spend or default factors. ' + " ".join(str(s_) for s_ in r.get("standards", [])[:3]))
    st.h3("How ClimRisk measured this")
    st.ul(["Inputs → activity data per entity, scope and category, each with an evidence reference.", "Factors → the ClimRisk emission-factor library, versioned, with named sources.", "Checks → completeness, consistency and data-quality rules.", "Output → totals by scope and category and a reproducible fingerprint."])
    return st.build()


# ═════════════════════════════════════════════════════════════════════════════
# 7. CBAM
# ═════════════════════════════════════════════════════════════════════════════

def cbam(p: dict, tpl: dict, brand: dict) -> dict:
    if p.get("result"):
        r = p["result"]
    else:
        from .climate.cbam import assess_cbam_exposure
        r = dataclasses.asdict(assess_cbam_exposure(company_name=p["company_name"], sector=p["sector"], revenue_usd_m=float(p["revenue_usd_m"]), eu_revenue_fraction=float(p.get("eu_revenue_fraction", 0.2)),
                                                    production_tonnes=p.get("production_tonnes"), origin_country=p.get("origin_country"), carbon_price_paid_eur=float(p.get("carbon_price_paid_eur", 0.0)),
                                                    eur_usd_rate=float(p.get("eur_usd_rate", 1.08))))
    st = S.Story(_kick("EU CBAM · story"), "", byline="ClimRisk transition engine", brand=brand)
    st.source("reg", "Regulation (EU) 2023/956 establishing a carbon border adjustment mechanism", "https://eur-lex.europa.eu/eli/reg/2023/956/oj", "")
    if not r["is_applicable"]:
        st.d["title"] = f'{r["company_name"]}: CBAM does not apply to this sector'
        st.d["standfirst"] = f'{r["sector"]} is outside Annex I of the regulation [[S:reg]], so no certificate cost is modelled.'
        st.section("loud", "The finding")
        st.p("The sector is not a CBAM good today. Scope can widen; check downstream products that contain covered goods.")
        return st.build()
    traj = r["scenarios"]["cbam_trajectory"]["years"]
    first, last = traj[0], traj[-1]
    peak = max(traj, key=lambda t: t["cbam_cost_usd_m"])
    title = f'{r["company_name"]}: the EU border bill, {first["year"]}–{last["year"]}'
    st.d["title"] = title
    st.d["standfirst"] = (f'{r["cbam_sector_name"]}. EU-bound emissions of {first["eu_embedded_co2e_t"]:,.0f} t CO₂e a year turn into a certificate cost of {_usd(first["cbam_cost_usd_m"] * 1e6)} in {first["year"]} '
                          f'rising to {_usd(last["cbam_cost_usd_m"] * 1e6)} by {last["year"]}; the peak year is {_usd(r["peak_annual_cost_usd_m"] * 1e6)} [[S:reg]].')
    st.source("ets", "EU ETS price path: NGFS Phase 5 / EMBER-calibrated (ClimRisk engine)", "https://ember-climate.org/data/data-tools/carbon-price-viewer/", "")
    cv = _try(lambda: F.cover(title, f'EU CBAM exposure · {dt.date.today().strftime("%d %B %Y")}', "ClimRisk transition engine · Regulation (EU) 2023/956",
                               F.bars([str(t["year"]) for t in traj], [t["cbam_cost_usd_m"] for t in traj], "CBAM certificate cost (US$ m)", "ClimRisk engine: EU ETS path × EU-bound embedded emissions, net of origin carbon price", horizontal=False, fmt="{:,.0f}", size=(7, 6.3)), tag="eu cbam", crop=False))
    if cv:
        st.cover(cv)
    st.p(f'**{first["year"]}.** The definitive period of the EU carbon border mechanism puts a price on the carbon inside imported {r["cbam_sector_name"].lower()}. For {r["company_name"]}, the engine takes the EU-bound share of production, multiplies by the embedded intensity of {r["embedded_carbon_intensity"]:.2f} t per unit, '
         f'and prices it at the EU ETS path less any carbon price already paid at origin [[S:reg]]. The result starts at {_usd(first["cbam_cost_usd_m"] * 1e6)} in {first["year"]} ({first["cbam_cost_pct_revenue"]:.1f}% of revenue) and peaks at {_usd(r["peak_annual_cost_usd_m"] * 1e6)} in {peak["year"]}.')
    cards = [(f'{r["embedded_carbon_intensity"]:.2f} t', "embedded CO₂ per unit", "regulation default or measured"), (f'{first["eu_embedded_co2e_t"]:,.0f} t', "EU-bound emissions a year", "production × EU share × intensity"), (f'€{first["net_cbam_price_eur_t"]:.0f}/t', f'net price in {first["year"]}', "EU ETS less origin price"),
             (f'{first["cbam_cost_pct_revenue"]:.1f}%', f'of revenue in {first["year"]}', "certificate cost / revenue"), (_usd(r["peak_annual_cost_usd_m"] * 1e6), f'peak annual cost ({peak["year"]})', "ClimRisk transition engine"), (f'€{first["origin_carbon_price_eur_t"]:.0f}/t', "origin carbon price deducted", "Art. 9 deduction")]
    _try(lambda: st.figure(F.numbers(f'{r["company_name"]} at the EU border in six numbers', "What the certificate bill is made of", cards, "ClimRisk transition engine; Regulation (EU) 2023/956; EU ETS path (NGFS / EMBER)"), caption="Each number is computed by the engine from the inputs on the cover."))
    st.pull(f'{_usd(r["peak_annual_cost_usd_m"] * 1e6)} in {peak["year"]}', f'From {_usd(first["cbam_cost_usd_m"] * 1e6)} in {first["year"]}: the bill is emissions times net price, and both move. ' + (f'Every €10/t of origin carbon price removes {_usd(10 * first["eu_embedded_co2e_t"])} a year.' if first["eu_embedded_co2e_t"] else ""))
    st.section("loud", "The loud number: the bill")
    _try(lambda: st.figure(F.bars([str(t["year"]) for t in traj], [t["cbam_cost_usd_m"] for t in traj], "CBAM certificate cost (US$ m)", "ClimRisk engine: EU ETS path (NGFS / EMBER) × EU-bound embedded emissions, net of origin carbon price", horizontal=False, fmt="{:,.0f}", highlight=traj.index(peak)),
                           caption="EU ETS price path × EU-bound embedded emissions, net of the origin carbon price [[S:reg]][[S:ets]]."))
    st.table(["Year", "EU ETS €/t", "Origin price €/t", "Net €/t", "Cost US$ m", "% of revenue"], [[t["year"], f'{t["eu_ets_price_eur_t"]:.0f}', f'{t["origin_carbon_price_eur_t"]:.0f}', f'{t["net_cbam_price_eur_t"]:.0f}', f'{t["cbam_cost_usd_m"]:.1f}', f'{t["cbam_cost_pct_revenue"]:.2f}%'] for t in traj])
    st.section("quiet", "The quiet part: what moves the bill")
    st.p(tpl["carbon"] + " The bill is emissions times net price. Two levers sit outside the price path: the embedded intensity (a measured value beats the default) and the origin carbon price, which is deducted under Article 9.")
    st.p(f'The default intensity used is {r["embedded_carbon_intensity"]:.2f} t per unit. A plant with verified emissions below the default pays less; one above it pays more. Origin price deduction today: €{first["origin_carbon_price_eur_t"]:.0f}/t, so every €10/t of origin price removes {_usd(10 * first["eu_embedded_co2e_t"])} a year.')
    if r.get("data_gaps"):
        st.callout("Data gaps: " + "; ".join(r["data_gaps"]), "warn")
    st.section("audiences", "What this means, by reader")
    st.p("**For the exporter.** Get the plant's verified embedded emissions: the default is a ceiling you may not need to accept.")
    st.p("**For the EU importer.** The certificate cost lands on the importer of record and will appear in price negotiations; ask for the verified figure.")
    st.section("method", "Method and limitations")
    st.p(r["methodology"] + " The certificate price follows the EU ETS path in the model; actual prices will differ. Only direct emissions (and, for cement, indirect) of covered goods are counted.")
    st.h3("How ClimRisk measured this")
    st.ul(["Scope → Annex I sector check against Regulation (EU) 2023/956.", "Emissions → production × EU share × embedded intensity (default unless measured).", "Price → EU ETS path calibrated to NGFS / EMBER, less the origin carbon price.", "Output → annual certificate cost and its share of revenue."])
    return st.build()


# ═════════════════════════════════════════════════════════════════════════════
# registry
# ═════════════════════════════════════════════════════════════════════════════

REPORTS = {
    "physical": {"label": "Physical risk — asset story", "fn": physical,
                 "fields": [{"k": "name", "t": "text", "l": "Asset name"}, {"k": "lat", "t": "number", "l": "Latitude"}, {"k": "lon", "t": "number", "l": "Longitude"},
                            {"k": "sector", "t": "text", "l": "Sector (e.g. cement)"}, {"k": "radius_m", "t": "number", "l": "Model radius (m)", "d": 300},
                            {"k": "value_usd", "t": "number", "l": "Asset value US$ (optional; else estimated)"},
                            {"k": "event", "t": "area", "l": "Event hook (optional): HEADLINE: … then one fact per line as  fact | source | url"},
                            {"k": "satellite", "t": "select", "l": "Satellite image of the site", "o": ["yes", "no"]}],
                 "example": {"name": "Chattogram refinery", "lat": 22.2645, "lon": 91.805, "sector": "refinery"},
                 "sections": ["history", "loud", "quiet", "protection", "audiences", "method"]},
    "transition": {"label": "Transition risk", "fn": transition,
                   "fields": [{"k": "company", "t": "text", "l": "Company (asset register)"}, {"k": "lat", "t": "number", "l": "…or asset latitude"}, {"k": "lon", "t": "number", "l": "…and longitude"}, {"k": "value_usd", "t": "number", "l": "Asset value US$ (optional)"}],
                   "example": {"lat": 22.2645, "lon": 91.805, "value_usd": 500000000}, "sections": ["loud", "policy", "abatement", "audiences", "method"]},
    "financial": {"label": "Financial risk", "fn": financial,
                  "fields": [{"k": "company", "t": "text", "l": "Company (asset register)"}, {"k": "lat", "t": "number", "l": "…or asset latitude"}, {"k": "lon", "t": "number", "l": "…and longitude"}, {"k": "value_usd", "t": "number", "l": "Asset value US$ (optional)"}],
                  "example": {"lat": 22.2645, "lon": 91.805, "value_usd": 500000000}, "sections": ["loud", "quiet", "inaction", "audiences", "method"]},
    "lca": {"label": "Life cycle assessment", "fn": lca,
            "fields": [{"k": "product", "t": "select", "l": "Product", "o": ["cement", "steel", "aluminium", "ammonia"]}, {"k": "params", "t": "json", "l": "Plant parameters (JSON, as /lca/<product>)"}],
            "example": {"product": "cement", "params": {"name": "Plant", "country": "IND", "clinker_ratio": 0.72, "kiln_thermal_gj_per_t_clinker": 3.4, "fuel_mix": {"coal_bituminous": 0.7, "petcoke": 0.2, "biomass_other": 0.1}, "kwh_per_t_clinker": 60, "kwh_per_t_cement_grinding": 40}},
            "sections": ["loud", "quiet", "cbam", "audiences", "method"]},
    "iso14067": {"label": "ISO 14067 study report", "fn": iso14067,
                 "fields": [{"k": "product", "t": "select", "l": "Product", "o": ["cement", "steel", "aluminium", "ammonia"]}, {"k": "params", "t": "json", "l": "Plant parameters (JSON)"}, {"k": "study", "t": "json", "l": "Study options (boundary, reference_year, commissioner…)"}],
                 "example": {"product": "cement", "params": {"name": "Plant", "country": "IND", "clinker_ratio": 0.72, "kiln_thermal_gj_per_t_clinker": 3.4, "fuel_mix": {"coal_bituminous": 0.7, "petcoke": 0.2, "biomass_other": 0.1}, "kwh_per_t_clinker": 60, "kwh_per_t_cement_grinding": 40},
                             "study": {"boundary": "cradle_to_gate", "reference_year": 2025, "commissioner": "Client", "practitioner": "ClimRisk"}},
                 "sections": ["goal", "results", "interpretation", "quality", "review", "method"]},
    "carbon_audit": {"label": "Carbon audit", "fn": carbon_audit,
                     "fields": [{"k": "inventory", "t": "json", "l": "Inventory (JSON, as /carbon/inventory)"}], "example": None, "sections": ["loud", "quiet", "audiences", "method"]},
    "cbam": {"label": "CBAM exposure", "fn": cbam,
             "fields": [{"k": "company_name", "t": "text", "l": "Company"}, {"k": "sector", "t": "select", "l": "Sector", "o": ["cement", "steel", "aluminium", "fertilizers", "hydrogen", "electricity"]},
                        {"k": "revenue_usd_m", "t": "number", "l": "Revenue (US$ m)"}, {"k": "eu_revenue_fraction", "t": "number", "l": "EU share of revenue (0–1)", "d": 0.2},
                        {"k": "production_tonnes", "t": "number", "l": "Production (t/yr)"}, {"k": "origin_country", "t": "text", "l": "Origin country (ISO2)"}, {"k": "carbon_price_paid_eur", "t": "number", "l": "Origin carbon price (€/t)", "d": 0}],
             "example": {"company_name": "Demo Cement", "sector": "cement", "revenue_usd_m": 1200, "eu_revenue_fraction": 0.25, "production_tonnes": 2000000, "origin_country": "IN", "carbon_price_paid_eur": 0},
             "sections": ["loud", "quiet", "audiences", "method"]},
}


def catalogue() -> list:
    return [{"id": k, "label": v["label"], "fields": v["fields"], "example": v["example"], "sections": v["sections"]} for k, v in REPORTS.items()] + [
        {"id": "_templates", "templates": {k: v["label"] for k, v in TEMPLATES.items()}}]


def build(report_type: str, payload: dict, template: Optional[str] = None, brand: Optional[dict] = None) -> dict:
    if report_type not in REPORTS:
        raise KeyError(report_type)
    payload = dict(payload)
    if isinstance(payload.get("event"), str):
        from .story_context import parse_event_text
        payload["event"] = parse_event_text(payload["event"])
    if isinstance(payload.get("satellite"), str):
        payload["satellite"] = payload["satellite"].strip().lower() not in ("no", "false", "0")
    tpl = template_for(payload.get("sector"), template)
    return REPORTS[report_type]["fn"](payload, tpl, _brand(payload, brand))


def generate(report_type: str, payload: dict, sections: Optional[list] = None, template: Optional[str] = None, brand: Optional[dict] = None, fmt: str = "json"):
    """Build, apply the section picker, render. fmt: json (story model + section list) | html | markdown | pdf (bytes)."""
    st = build(report_type, payload, template, brand)
    out = S.pick_sections(st, sections)
    if fmt == "html":
        return S.html(out)
    if fmt == "markdown":
        return S.markdown(out)
    if fmt == "pdf":
        return S.pdf(out)
    return {"story": out, "sections": S.sections_of(st), "kept": sections or [s["key"] for s in S.sections_of(st)], "template": template_for(payload.get("sector"), template)["key"]}
