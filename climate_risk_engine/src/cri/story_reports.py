"""
cri/story_reports.py — story-format reports for every ClimRisk module.

Seven report types, one format (cri.story): a hook, dated evidence with numbered sources, the asset-level story, what it means per
audience, and the method and its limits. Each is customisable — sector template, section picker, client branding — and renders to
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


# ═════════════════════════════════════════════════════════════════════════════
# 1. PHYSICAL — the asset story
# ═════════════════════════════════════════════════════════════════════════════

def physical(p: dict, tpl: dict, brand: dict) -> dict:
    from . import asset_twin as AT
    name = p.get("name") or "the asset"
    tw = p.get("twin")
    if tw is None:
        tw = AT.build_twin(float(p["lat"]), float(p["lon"]), name, p.get("sector"), p.get("asset_type"), p.get("commissioned_year"),
                           float(p.get("radius_m", 300)), p.get("edits"), p.get("incidents"), True, run_hydraulics=p.get("hydraulics", True))
    an = p.get("analysis") or AT.analyse(tw)
    fnd = an["findings"]
    lab = p.get("lab")
    if lab is None and fnd and p.get("scenario_lab", True):
        try:
            from .resident import scenario as SC
            lab = SC.lab(tw, an)
        except Exception:                                           # noqa: BLE001
            lab = None
    name = tw["asset"]["name"] if tw.get("asset", {}).get("name") else name
    site, hist, scen = tw["site"], tw["history"], tw["scenarios"]
    comps = [c for c in tw["components"] if not c.get("removed") and c.get("on_site", True)]
    storms = sorted(hist.get("storms", []), key=lambda e: -e["peak_wind_ms"])
    fl_f = [f for f in fnd if f["hazard"] == "flood"]; wd_f = [f for f in fnd if f["hazard"] == "wind"]
    fixed = sum(1 for f in fnd if any(m.get("resolves_equipment") or m.get("residual_beyond_design") == 0 for m in f.get("protection", [])))

    max_map = max((s["depth_m"] for s in scen["flood"]), default=0.0)
    sf = (f'{len(comps)} structures modelled on real footprints and terrain; {len(fnd)} finding(s) — {len(fl_f)} from water, {len(wd_f)} from wind. '
          + (f'The deepest mapped scenario reaches {max_map:.1f} m on average across the site. ' if max_map > 0 else ("The flood maps put no water on the site itself. " if scen["flood"] else ""))
          + ("" if fnd else "In the scenarios modelled, nothing at this site is reached by water or pushed beyond design wind."))
    st = S.Story(_kick("Physical risk · asset story"), f"{name}: what the water and the wind do to the asset", sf, byline="ClimRisk Asset Twin agent", brand=brand)
    for k, t, u, n in [("osm", "OpenStreetMap contributors — building and industrial footprints", "https://www.openstreetmap.org/copyright", "ODbL"),
                       ("terr", "AWS Terrain Tiles (Terrarium), ~9 m", "https://registry.opendata.aws/terrain-tiles/", ""),
                       ("jrc", "JRC CEMS-GloFAS river flood hazard maps v2.1", "", "undefended 3 arc-second depth maps"),
                       ("del", "Deltares Global Flood Maps v2021.06 (GTSMip6 surge + tide)", "", "undefended coastal depth maps"),
                       ("ibt", "NOAA IBTrACS tropical-cyclone best tracks", "https://www.ncei.noaa.gov/products/international-best-track-archive", "1980–2025, Holland wind model"),
                       ("era5", "ERA5 reanalysis via Open-Meteo archive", "https://open-meteo.com/en/docs/historical-weather-api", "daily maximum gust, 1950 onward"),
                       ("bates", "Bates, Horritt & Fewtrell (2010) local-inertial shallow-water formulation, J. Hydrology 387", "", "")]:
        st.source(k, t, u, n)

    st.facts([(str(len(comps)), "structures modelled"), (f'{site["ground_ref_m"]} m', "ground above datum"), (str(len(scen["flood"])), "flood scenarios"),
              (f'{storms[0]["peak_wind_ms"]:.0f} m/s' if storms else "none", "strongest storm on record" if storms else "recorded storm at the site"),
              (str(len(fnd)), "findings"), (str(fixed), "with a verified fix")])

    st.section("history", "What has already happened here")
    ev = [(e["year"], e["peak_wind_ms"], f'{e["name"]} {e["year"]}') for e in storms[:12] if e.get("year")]
    if storms:
        st.p(f'The track archive holds {len(hist["storms"])} tropical storm or stronger passages within reach of this site since 1980 [[S:ibt]]. The strongest, {storms[0]["name"]} in {storms[0]["year"]}, '
             f'brought about {storms[0]["peak_wind_ms"]:.0f} m/s ({storms[0]["category_at_site"]}) at the site, passing {storms[0]["closest_km"]:.0f} km away.')
        try:
            st.figure(S.chart_events([e[0] for e in ev], [e[1] for e in ev], title="Modelled wind at the site, by storm", ylabel="m/s", highlight={ev[0][0]: ev[0][2]} if ev else None),
                      caption="Peak sustained wind at the site from each storm's track (Holland model) [[S:ibt]].")
        except Exception:                                           # noqa: BLE001
            pass
        st.table(["Storm", "Year", "Peak wind at site", "Category", "Closest approach"],
                 [[e["name"], e["year"], f'{e["peak_wind_ms"]:.0f} m/s', e["category_at_site"], f'{e["closest_km"]:.0f} km'] for e in storms[:6]])
    else:
        st.p("No tropical storm of tropical-storm strength has been recorded within reach of this site in the track archive [[S:ibt]].")
    ws = hist.get("windstorms", [])
    if ws:
        st.p("The all-cause wind record (reanalysis, 1950 onward) adds non-cyclone events — the five strongest daily gusts: "
             + "; ".join(f'{w["date"]} {w["gust_ms"]:.0f} m/s' for w in ws[:5]) + " [[S:era5]]. Reanalysis smooths peaks, so these are floors, not ceilings.")
    inc = hist.get("incidents", [])
    if inc:
        st.p("Reported by the operator: " + "; ".join(f'{i["date"]}: {i.get("label") or i.get("note")}' for i in inc[:6]) + ".")
    else:
        st.p("No operator-reported incidents were supplied. Adding them lets the replay be checked against what actually happened.")

    st.section("loud", "The loud risk: what the hazard maps say")
    if scen["flood"]:
        st.table(["Flood scenario", "Mean depth", "90th percentile", "Share of site wet", "Source"],
                 [[s["label"], f'{s["depth_m"]:.2f} m', f'{s["depth_p90_m"]:.2f} m', f'{round(100 * s["wet_share"])}%', "JRC [[S:jrc]]" if s["kind"] == "river" else "Deltares [[S:del]]"] for s in scen["flood"][:8]])
        try:
            st.figure(S.chart_bars([s["label"].replace("Coastal surge + tide, ", "Coast ").replace("River flood, ", "River ")[:26] for s in scen["flood"][:8]],
                                   [s["depth_m"] for s in scen["flood"][:8]], title="Mean modelled depth over the site", unit=" m", horizontal=True),
                      caption="Map depths are undefended and are the ceiling the solver is held to.")
        except Exception:                                           # noqa: BLE001
            pass
    else:
        st.p("The flood maps hold no depth at this site for the scenarios available — which is itself worth checking against the ground level of "
             f'{site["ground_ref_m"]} m above datum.')
    if fl_f and max_map == 0:
        st.callout("The flood maps show no depth on the site itself, yet the 2D solver reaches structures. The solver takes its water level from the nearest wet map pixels (up to 600 m away) and routes it over 9 m terrain, "
                   "which can fill low pockets that the 90 m flood map smooths over. Read the flood findings below as a warning to check ground levels on site, not as a mapped flood.", "warn")
    dw = [w for w in scen["wind"] if w["kind"] == "design"]
    if dw:
        st.table(["Wind scenario", "Sustained at 10 m", "Category"], [[w["label"], f'{w["v10_ms"]:.0f} m/s', w.get("category", "")] for w in scen["wind"][:8]],
                 caption="Return-period winds from the cyclone model and the all-cause reanalysis fit [[S:ibt]][[S:era5]].")

    st.section("quiet", "The quiet risk: what it does to this asset")
    st.p(f'{tpl["chain"]} The loud risk is the map above. The quiet one is {tpl["quiet"]}. That is what the twin looks for, structure by structure.')
    if not fnd:
        st.p("Nothing in the modelled scenarios reaches a component or exceeds its design wind. Heights, equipment levels and design winds are assumptions by structure kind until the operator edits them, "
             "so this is the result to check first, not a clearance.")
    for f in (fl_f[:3] + wd_f[:3]):
        st.h3(f["title"])
        st.ul(f["problem"][:6])
        st.p("**Why:** " + f["cause"])
        st.p("**What it costs:** " + f["consequence"])
        if f.get("access"):
            st.p("**Access:** " + "; ".join(f'{a["name"]} ({a["type"]}) {a["state"]} at {a["max_depth_m"]} m' for a in f["access"][:3]) + ".")
        if f.get("protection"):
            rows = []
            for m in f["protection"]:
                what = f'raise equipment {m["delta_m"]} m' if m["type"] == "raise_equipment" else f'flood barrier {m["height_m"]} m' if m["type"] == "flood_barrier" else f'wind retrofit to {m["design_wind_ms"]:.0f} m/s'
                res = (f'{m.get("residual_equipment_inundated", 0)} inundated, {m.get("residual_wet_floor", 0)} wet' if "residual_wet_floor" in m else f'{m.get("residual_beyond_design", 0)} still beyond design')
                rows.append([what, res, m.get("verified_by", "re-evaluation")])
            st.table(["Protection", "What remains", "Verified by"], rows)
    if an.get("notes"):
        st.callout(" ".join(an["notes"][:3]), "warn")

    st.section("protection", "What protects it, ranked")
    if lab and lab.get("solutions"):
        st.p(lab["recommendation"])
        st.table(["Solution", "Resolves base scenarios", "Flood margin", "Wind margin"],
                 [[s["name"], "yes" if s["resolves_base"] else "no", f'+{s["flood_margin_m"]} m' if s["flood_margin_m"] is not None else "–", f'+{s["wind_margin_pct"]}%' if s["wind_margin_pct"] is not None else "–"] for s in lab["solutions"][:6]],
                 caption="Ranked by physical performance; no costs are invented. " + " ".join(lab.get("limits", [])[:1]))
    else:
        st.p("No protection is proposed because nothing is reached in the modelled scenarios." if not fnd else "The scenario lab did not run, so protection options are listed under each finding only.")

    st.section("audiences", "What this means, by reader")
    worst = fl_f[0] if fl_f else (wd_f[0] if wd_f else None)
    st.p("**For the operator.** " + (f'Start with: {worst["title"].lower()}. ' if worst else "Confirm the heights and equipment levels in the twin, because the result depends on them. ")
         + "The cheapest verified fix is listed under each finding; edit the twin with as-built values and the findings change.")
    st.p("**For the lender or insurer.** " + (f'{len(fnd)} scenario(s) produce a finding at this site; ask which of the verified measures are in the capital plan and when. ' if fnd else "No scenario produces a finding on the assumed values; ask for the as-built levels that make that true. ")
         + "The hazard maps are undefended; local defences are not credited unless the operator supplies them.")
    st.p("**For the investor.** Physical damage here is a direct-loss figure, not a business-interruption figure; a site that stops for weeks costs more than the equipment it loses.")

    st.section("method", "Method and limits")
    st.ul(an.get("assumptions", [])[:6] + [
        "Flood depths come from a 2D shallow-water solver (local-inertial form [[S:bates]]) on 9 m terrain [[S:terr]] with every building a solid obstruction, forced with the level implied by the flood map; "
        "event duration is assumed, and rainfall, drains, pumps and waves are not modelled.",
        "Wind at a structure's height uses a power-law profile and a gust factor; design winds are assumed by kind until edited. Tornado and downburst are not modelled.",
        "Footprints come from OpenStreetMap [[S:osm]]; where none exist, an illustrative layout is shown and flagged." if site.get("layout_source") == "parametric" else "Footprints come from OpenStreetMap [[S:osm]]; unmapped process equipment must be added with the editor."]
        + list(tw.get("data_gaps", [])[:3]))
    return st.build()


# ═════════════════════════════════════════════════════════════════════════════
# 2/3. TRANSITION and FINANCIAL — from the risk analyst
# ═════════════════════════════════════════════════════════════════════════════

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


def _scen(r: dict, label: str) -> dict:
    return (r.get("financial") or r).get("scenarios", {}).get(label, {})


def transition(p: dict, tpl: dict, brand: dict) -> dict:
    r = _analyst(p)
    st = S.Story(_kick("Transition risk · story"), "", byline="ClimRisk risk analyst", brand=brand)
    st.source("ngfs", "NGFS Phase 5 scenarios (IIASA)", "https://data.ene.iiasa.ac.at/ngfs/", "carbon price and sector output, 7 scenarios")
    st.source("wb", "World Bank Carbon Pricing Dashboard", "https://carbonpricingdashboard.worldbank.org/", "instruments in force and pipeline")
    if _is_company(r):
        names = ", ".join(r["companies"][:3]) + ("…" if len(r["companies"]) > 3 else "")
        d = r["scenarios"]; lab = list(d)
        st.d["title"] = f"{names}: what the transition does to {r['n_assets']} assets"
        mean = {k: d[k]["total_npv"]["mean"] for k in lab}
        st.d["standfirst"] = (f'Across {r["n_assets"]} assets worth about {_usd(r.get("portfolio_value_mode"))}, expected climate cost to 2050 ranges from {_usd(min(mean.values()))} to {_usd(max(mean.values()))} '
                              f'depending on the scenario — the spread is the transition story [[S:ngfs]].')
        st.section("loud", "The loud risk: the scenario spread")
        st.figure(S.chart_bars([k.replace(" (NDCs)", "")[:22] for k in lab], [mean[k] / 1e6 for k in lab], title="Expected total climate cost, PV 2026–2050 (US$ m)", unit="", horizontal=True,
                               errors=None), caption="Mean of the Monte Carlo distribution per scenario [[S:ngfs]].")
        st.table(["Scenario", "Mean", "Median", "95th pct", "Expected shortfall"], [[k, _usd(d[k]["total_npv"]["mean"]), _usd(d[k]["total_npv"]["p50"]), _usd(d[k]["total_npv"]["p95"]), _usd(d[k]["total_npv"].get("es95"))] for k in lab])
        st.section("quiet", "The quiet risk: which assets carry it")
        A = sorted(r["assets"], key=lambda a: -(a.get("npv_delayed_mean") or 0))[:10]
        st.p(f'{tpl["carbon"]} The portfolio number hides concentration: the ten assets below carry the largest delayed-transition cost.')
        st.table(["Asset", "Country", "Sector", "Emissions", "Carbon price now", "Delayed-transition cost"],
                 [[a["name"], a["country"], a["sector"], f'{(a["emissions_t"] or 0) / 1e6:.2f} Mt', f'${a["carbon_price_now"]:.0f}/t' if a.get("carbon_price_now") is not None else "n/a", _usd(a.get("npv_delayed_mean"))] for a in A])
        st.section("audiences", "What this means, by reader")
        st.p("**For the company.** The lever is which assets abate first; the table is the order to ask about.")
        st.p("**For the lender or investor.** The exposure is a range, not a point: ask which scenario the plan assumes, then read the matching row.")
        st.section("method", "Method and limits")
        st.ul([r.get("method", "Monte Carlo over NGFS carbon-price and activity paths."), "Asset values are estimated from capacity unless supplied; replace them for a decision-grade figure.", "Direct costs only; second-round effects are not included."])
        return st.build()

    a = r["asset"]; pol = r["policy"]; tr = r["transition"]; fin = r["financial"]
    scs = list(fin["scenarios"]); yrs = fin["years"]
    delayed = next((k for k in scs if "elayed" in k), scs[0]); nz = next((k for k in scs if "Net Zero" in k), scs[0]); cp = next((k for k in scs if "Current" in k), scs[-1])
    st.d["title"] = f'{a["name"]}: the carbon bill and when it arrives'
    pv = lambda k, key="transition_npv": fin["scenarios"][k][key]["mean"]            # noqa: E731
    st.d["standfirst"] = (f'{a["name"]} emits {a["emissions_t"] / 1e3:,.0f} kt CO₂e a year. Present-value carbon cost to 2050 runs from {_usd(pv(cp))} under current policies to {_usd(pv(nz))} under Net Zero 2050 '
                          f'if nothing is done [[S:ngfs]].')
    st.facts([(f'{a["emissions_t"] / 1e3:,.0f} kt', "CO₂e a year"), (_usd(a["value"]["mode"]), "asset value (" + a["value"]["basis"].split(";")[0][:24] + ")"), (f'${pol["effective_price_usd_t"]:.0f}/t', "carbon price today"),
              (_usd(pv(cp)), "cost, current policies"), (_usd(pv(nz)), "cost, Net Zero 2050")])
    st.section("loud", "The loud risk: the carbon price path")
    try:
        ser = {k.replace("?C", "°C"): tr["scenarios"][k]["price_median"] for k in tr["scenarios"] if k in ("Net Zero 2050", "Delayed transition", "Current Policies", "Below 2?C")}
        n = min(len(v) for v in ser.values())
        st.figure(S.chart_lines(yrs[:n], {k: v[:n] for k, v in ser.items()}, title="Median carbon price by scenario (US$/t CO₂)", ylabel="US$/t"), caption="NGFS Phase 5 median carbon price [[S:ngfs]].")
    except Exception:                                               # noqa: BLE001
        pass
    def _ab(k):
        ab = fin["scenarios"][k].get("abatement") or {}
        return _usd((ab.get("response_cost_npv") or {}).get("mean")) if ab.get("enabled") else "n/a"
    st.table(["Scenario", "Carbon cost PV (no action)", "Cost with abatement response", "Stranded value"],
             [[k, _usd(fin["scenarios"][k]["transition_npv"]["mean"]), _ab(k), _usd((fin["scenarios"][k].get("stranded") or {}).get("mean"))] for k in scs],
             caption="Present value 2026–2050, mean of the Monte Carlo draws [[S:ngfs]].")
    st.section("policy", "The quiet risk: which policy actually reaches the asset")
    if pol["in_force"]:
        st.table(["Instrument in force", "Type", "Price", "Covers this sector"], [[i["name"], i["type"], f'${i["price_usd_t"]:.0f}/t' if i.get("price_usd_t") is not None else "n/a", "yes" if i.get("covers_asset_sector") else "no"] for i in pol["in_force"]])
    st.p(" ".join(pol["notes"]) + " [[S:wb]]")
    if pol.get("pipeline"):
        st.p("In the pipeline: " + "; ".join(i["name"] for i in pol["pipeline"][:5]) + ".")
    st.p(tpl["carbon"] + (" The asset exports into the EU, so CBAM adds a border charge on top." if pol.get("cbam_exporter") else ""))
    st.section("abatement", "What the operator can do about it")
    ab = tr.get("abatement", {})
    if ab.get("levers"):
        st.table(["Lever", "Share of emissions", "Marginal cost (US$/t)", "Available from", "Basis"],
                 [[l["name"], "–".join(f"{x:.0%}" for x in l["share"]) if isinstance(l["share"], (list, tuple)) else f'{l["share"]:.0%}', "–".join(f"{x:.0f}" for x in l["mac"]) if isinstance(l["mac"], (list, tuple)) else str(l["mac"]), str(l["avail"]), l.get("basis", "")] for l in ab["levers"]],
                 caption="Library ranges unless the operator supplies its own levers — get site quotes and rerun.")
    for act in r.get("actions", []):
        if act["hazard"] in ("carbon_cost", "abatement"):
            st.p(f'**{act["problem"]}** {act["action"]}')
    st.section("audiences", "What this means, by reader")
    st.p("**For the operator.** The cost depends on timing as much as price: abatement started late is paid for at the highest prices.")
    st.p("**For the lender or investor.** The question is which scenario the plan sits in and whether the stranded-value row is covered by the asset's remaining life.")
    st.section("method", "Method and limits")
    st.ul(list(tr.get("assumptions", []))[:5] + r.get("caveats", [])[:3] + [f'Source: {pol["source"][:160]}…'])
    return st.build()


def financial(p: dict, tpl: dict, brand: dict) -> dict:
    r = _analyst(p)
    st = S.Story(_kick("Financial risk · story"), "", byline="ClimRisk risk analyst", brand=brand)
    st.source("ngfs", "NGFS Phase 5 scenarios (IIASA)", "https://data.ene.iiasa.ac.at/ngfs/", "")
    if _is_company(r):
        d = r["scenarios"]; lab = list(d); names = ", ".join(r["companies"][:3])
        worst = max(lab, key=lambda k: d[k]["total_npv"]["mean"])
        st.d["title"] = f"{names}: the loss distribution"
        st.d["standfirst"] = f'Portfolio value {_usd(r.get("portfolio_value_mode"))} across {r["n_assets"]} assets. Expected loss to 2050 is {_usd(d[worst]["total_npv"]["mean"])} in the worst scenario ({worst}), with a 95th-percentile outcome of {_usd(d[worst]["total_npv"]["p95"])}.'
        st.section("loud", "The loud risk: the expected loss")
        st.figure(S.chart_bars([k[:22] for k in lab], [d[k]["total_npv"]["mean"] / 1e6 for k in lab], title="Mean total loss, PV 2026–2050 (US$ m)", horizontal=True), caption="Monte Carlo mean per scenario [[S:ngfs]].")
        st.section("quiet", "The quiet risk: the tail and the concentration")
        st.table(["Scenario", "Mean", "Median", "95th pct", "Expected shortfall"], [[k, _usd(d[k]["total_npv"]["mean"]), _usd(d[k]["total_npv"]["p50"]), _usd(d[k]["total_npv"]["p95"]), _usd(d[k]["total_npv"].get("es95"))] for k in lab])
        A = sorted(r["assets"], key=lambda a: -(a.get("physical_aal_pct") or 0))[:8]
        st.table(["Asset", "Country", "Physical annual loss (% of value)", "Cyclone", "River", "Coast"], [[a["name"], a["country"], f'{a["physical_aal_pct"]:.2f}%' if a.get("physical_aal_pct") is not None else "n/a", a.get("cyclone", ""), a.get("river", ""), a.get("coast", "")] for a in A],
                 caption=f'Flood depths were read live for the {r.get("flood_assessed_assets", 0)} largest emitters only; the others use screening values.')
        st.section("audiences", "What this means, by reader")
        st.p("**For the lender.** Size covenants to the 95th percentile and expected shortfall, not the mean.")
        st.p("**For the investor.** The mean hides the tail; the tail is where a few coastal assets sit.")
        st.section("method", "Method and limits")
        st.ul([r.get("method", "Monte Carlo loss distribution."), "Direct physical damage and carbon cost only; business interruption is not included."])
        return st.build()

    a = r["asset"]; fin = r["financial"]; scs = list(fin["scenarios"]); ph = fin["physical"]
    delayed = next((k for k in scs if "elayed" in k), scs[0])
    d = fin["scenarios"][delayed]
    st.d["title"] = f'{a["name"]}: what the loss distribution says'
    st.d["standfirst"] = r["headline"]
    st.facts([(_usd(a["value"]["mode"]), "asset value"), (_usd(d["total_npv"]["mean"]), f"expected loss, {delayed.lower()}"), (_usd(d["total_npv"]["p95"]), "1-in-20 outcome"),
              (_usd(d["total_npv"].get("es95")), "expected shortfall (worst 5%)"), (f'{ph["aal_pct"]:.2f}%', "physical annual loss, of value")])
    st.section("loud", "The loud risk: the expected loss")
    mean = [fin["scenarios"][k]["total_npv"]["mean"] / 1e6 for k in scs]
    lo = [(fin["scenarios"][k]["total_npv"]["mean"] - fin["scenarios"][k]["total_npv"]["p5"]) / 1e6 for k in scs]
    hi = [(fin["scenarios"][k]["total_npv"]["p95"] - fin["scenarios"][k]["total_npv"]["mean"]) / 1e6 for k in scs]
    st.figure(S.chart_bars([k[:20] for k in scs], mean, title="Total loss, PV 2026–2050 (US$ m), bars show P5–P95", errors=[lo, hi]), caption="Mean with 5th–95th percentile range [[S:ngfs]].")
    for para in r["narrative"][:3]:
        st.p(para)
    st.section("quiet", "The quiet risk: the tail, and what the mean hides")
    st.table(["Scenario", "Mean", "Median", "95th pct", "99th pct", "Expected shortfall", "Mean, % of value"],
             [[k, _usd(fin["scenarios"][k]["total_npv"]["mean"]), _usd(fin["scenarios"][k]["total_npv"]["p50"]), _usd(fin["scenarios"][k]["total_npv"]["p95"]),
               _usd(fin["scenarios"][k]["total_npv"]["p99"]), _usd(fin["scenarios"][k]["total_npv"].get("es95")), _pct(fin["scenarios"][k]["total_npv"]["pct_of_value"]["mean"], 1)] for k in scs])
    rp = ph.get("annual_loss_rp") or {}
    if rp:
        st.p("Physical loss in a single year at return period: " + "; ".join(f'1-in-{k}: {_usd(v)} ({100 * ph["annual_damage_ratio_rp"][k]:.0f}% of value)' for k, v in rp.items()) + ". "
             + f'{tpl["chain"]}')
    st.section("inaction", "What inaction costs")
    ab = d.get("total_npv_unabated")
    if ab:
        st.p(f'Without any abatement the mean loss in the {delayed.lower()} would be {_usd(ab["mean"])}, against {_usd(d["total_npv"]["mean"])} with the sector levers — {_usd(ab["mean"] - d["total_npv"]["mean"])} avoided.')
    for act in r.get("actions", [])[:3]:
        st.p(f'**{act["problem"]}** {act["action"]}')
    st.section("audiences", "What this means, by reader")
    st.p(f'**For the lender.** The 95th percentile ({_usd(d["total_npv"]["p95"])}) and expected shortfall ({_usd(d["total_npv"].get("es95"))}) are the covenant numbers, not the mean.')
    st.p("**For the insurer.** The physical share is a direct-damage figure; check deductibles against the 1-in-50 and 1-in-100 single-year losses above.")
    st.p("**For the investor.** The distribution is wide; the decision depends on which scenario the plan assumes.")
    st.section("method", "Method and limits")
    st.ul(r.get("caveats", [])[:4] + [f'Draws: {fin["n_draws"]}; discount rate {fin["discount_rate"]:.0%}.'])
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


def lca(p: dict, tpl: dict, brand: dict) -> dict:
    r = _lca_run(p)
    u = r["uncertainty"]; fp = r["footprint_kg_co2e_per_t"]
    st = S.Story(_kick("Life cycle assessment · story"), f'One tonne of {r["product"]}: {fp:,.0f} kg of CO₂e, and where it comes from',
                 f'Cradle-to-gate footprint of {r["declared_unit"]} made in {r["country"]}: {fp:,.0f} kg CO₂e (5th–95th percentile {u["p5"]:,.0f}–{u["p95"]:,.0f}). {r["standard"].split(".")[0]}.', byline="ClimRisk LCA engine", brand=brand)
    st.source("std", "ISO 14067:2018 Carbon footprint of products", "https://www.iso.org/standard/71206.html", "")
    for i, f in enumerate(r.get("factor_sources", [])[:6]):
        st.source(f"f{i}", str(f), "", "")
    st.facts([(f"{fp:,.0f} kg", "CO₂e per tonne"), (f'{u["p5"]:,.0f}–{u["p95"]:,.0f}', "P5–P95, kg"), (f'{r["biogenic_co2_kg_per_t"]:,.1f} kg', "biogenic CO₂, reported separately"),
              (r["data_quality"]["rating"], f'data quality (score {r["data_quality"]["weighted_score"]})')])
    st.section("loud", "The loud number")
    bs = {k: v for k, v in r["by_stage"].items() if v}
    st.figure(S.chart_bars(list(bs), list(bs.values()), title="Where the footprint comes from (kg CO₂e per tonne)", horizontal=True), caption="Stage and component contributions from the inventory.")
    st.table(["Stage", "Item", "kg CO₂e / t", "Basis"], [[d["stage"], d["item"], f'{d["kg_co2e"]:.1f}', d["basis"]] for d in r["detail"][:10]])
    st.section("quiet", "The quiet part: what moves it, and what it hides")
    if r.get("clinker"):
        c = r["clinker"]
        st.p(f'Clinker is {c["clinker_ratio"]:.0%} of the cement. Calcination alone releases {c["calcination_kg_co2_per_t_clinker"]:.0f} kg per tonne of clinker, and that is chemistry: {c["calcination_basis"]}. {tpl["carbon"]}')
    else:
        st.p(tpl["carbon"])
    st.p(f'The uncertainty band is {100 * (u["p95"] - u["p5"]) / 2 / u["p50"]:.0f}% either side of the median ({u["method"][:120]}…). Excluded: ' + "; ".join(r.get("exclusions", [])[:3]) + ".")
    if r.get("trajectory"):
        t = r["trajectory"]
        try:
            sc = {k: v for k, v in t["scenarios"].items() if k in ("Net Zero 2050", "Current Policies", "Delayed transition")} or dict(list(t["scenarios"].items())[:3])
            st.figure(S.chart_lines(t["years"], sc, title="Footprint if only the grid decarbonises (kg CO₂e per tonne)", ylabel="kg CO₂e / t"), caption="Grid pathway effect only — the plant's own process is held constant.")
        except Exception:                                           # noqa: BLE001
            pass
    if r.get("cbam"):
        c = r["cbam"]
        st.section("cbam", "What the border sees")
        st.p(f'Under EU CBAM the embedded emissions are {c["direct_tco2_per_t_cement"] + c["indirect_tco2_per_t_cement"]:.2f} t per tonne ({c["direct_tco2_per_t_cement"]:.2f} direct, {c["indirect_tco2_per_t_cement"]:.2f} indirect). '
             f'In {c["year"]} the phase-in share is {c["cbam_phase_in_share"]:.1%}, giving an indicative cost of ${c["indicative_cbam_cost_usd_per_t_cement"]:.2f} per tonne at an EU ETS price of ${c["eu_ets_price_usd_t"]:.0f}/t. ' + " ".join(c.get("notes", [])[:1]))
    st.section("audiences", "What this means, by reader")
    st.p("**For the producer.** The hotspot rows are the order of work; the biggest line is rarely the one with the cheapest fix.")
    st.p("**For the buyer.** Ask for the P5–P95 band and the data-quality rating with any single number.")
    st.section("method", "Method and limits")
    st.ul([r["standard"], f'Data quality: {r["data_quality"]["rating"]} (primary-data share {r["data_quality"]["primary_data_share"]:.0%}).'] + r.get("notes", [])[:3])
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
    st = S.Story(_kick("ISO 14067 · carbon footprint study"), f'{r["product"].title()} carbon footprint study: {r["cfp_kg_co2e"]:,.0f} kg CO₂e per {r["declared_unit"].replace("1 ", "")}',
                 f'{b["label"]}. {r["cfp_kg_co2e"]:,.0f} kg CO₂e (P5–P95 {u["p5"]:,.0f}–{u["p95"]:,.0f}), {r["gwp"]}, reference year {r["reference_year"]}. '
                 + ("A full CFP." if b.get("is_full_cfp") else "A partial CFP: stages after the boundary are not included."), byline="ClimRisk LCA engine", brand=brand)
    st.source("iso", "ISO 14067:2018 Greenhouse gases — Carbon footprint of products", "https://www.iso.org/standard/71206.html", "")
    for i, f in enumerate(r.get("factor_sources", [])[:6]):
        st.source(f"f{i}", str(f), "", "")
    st.section("goal", "Goal and scope")
    g, sc = rep["sections"]["2_goal"], rep["sections"]["3_scope"]
    st.p(f'Intended application: {g["intended_application"]}. Audience: {g["intended_audience"]}. {g["note"]} [[S:iso]]')
    st.table(["Item", "Value"], [["Product", sc["product"]], ["Declared unit", sc["declared_unit"]], ["System boundary", b["label"]], ["Excluded stages", ", ".join(b.get("excluded_stages", [])) or "none"],
                                 ["Time period", sc["time_period"]], ["Geography", sc["geography"]], ["GWP basis", sc["gwp"]], ["Allocation", str(sc["allocation"])]])
    st.section("results", "Results")
    bs = {k: v for k, v in (r.get("a1_a3_by_component") or r["by_stage"]).items() if v}
    st.figure(S.chart_bars(list(bs), list(bs.values()), title="Contribution by component (kg CO₂e)", horizontal=True), caption="Calculated from the inventory flows in §4.")
    gt = {k.replace("_", " "): v for k, v in S5["by_ghg_type"].items() if v}
    st.table(["GHG type (ISO 14067 §6.3.5)", "kg CO₂e"], [[k, f"{v:,.2f}"] for k, v in gt.items()], caption="Biogenic CO₂ emissions and removals are reported separately and are not included in the fossil total.")
    st.section("interpretation", "Interpretation: hotspots, sensitivity and methodological choices")
    st.table(["Stage", "Item", "kg CO₂e", "Share", "Cumulative"], [[h["stage"], h["item"], f'{h["kg_co2e"]:.1f}', f'{h["share_pct"]:.1f}%', f'{h["cumulative_pct"]:.1f}%'] for h in S7["hotspots_to_80pct"]], caption="Items making up 80% of the footprint.")
    st.table(["Item", "Effect of +10% on the total"], [[x["item"], f'{x["effect_of_plus_10pct_on_total_pct"]:.2f}%'] for x in S7["sensitivity"][:6]])
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
    st.section("method", "Method and limits")
    st.ul(list(r.get("notes", []))[:4] + list(r.get("exclusions", []))[:3] + [f'Study fingerprint {r["fingerprint"]}: the same inputs reproduce this result.'])
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
    st = S.Story(_kick("Carbon audit · story"), f'{r["organisation"]}: {t["total_location"]:,.0f} tonnes of CO₂e, and how far to trust the number',
                 f'Reporting year {r["reporting_year"]}, {r["consolidation"].replace("_", " ")} consolidation, GWP basis {r["gwp"]}. Scope 1 {t["scope1"]:,.0f} t, scope 2 {t["scope2_location"]:,.0f} t (location) or {t["scope2_market"]:,.0f} t (market), scope 3 {t["scope3"]:,.0f} t.',
                 byline="ClimRisk carbon engine", brand=brand)
    for i, s in enumerate(r.get("standards", [])[:5]):
        st.source(f"s{i}", str(s), "", "")
    st.source("lib", f'ClimRisk emission-factor library {r["factor_library"]}', "", "UK DESNZ 2026, US EPA eGRID 2023, Green-e residual mix, AIB, Ember")
    st.facts([(f'{t["scope1"]:,.0f} t', "scope 1"), (f'{t["scope2_location"]:,.0f} t', "scope 2, location"), (f'{t["scope2_market"]:,.0f} t', "scope 2, market"), (f'{t["scope3"]:,.0f} t', "scope 3 reported"),
              (f'{q["weighted_score"]}', "data-quality score (1 best, 5 worst)")])
    st.section("loud", "The loud number")
    cats = r["by_category_t"]
    st.figure(S.chart_bars([k.replace("Cat ", "C") for k in cats][:10], list(cats.values())[:10], title="Emissions by category (t CO₂e)", horizontal=True), caption="Every bar is the sum of audited lines; each line keeps its input, factor and evidence reference.")
    if r.get("intensity"):
        st.p(f'Intensity: {r["intensity"]["t_per_musd_revenue"]:.1f} t per US$ m revenue and {r["intensity"]["t_per_employee"]:.2f} t per employee ({r["intensity"]["basis"]}).')
    st.section("quiet", "The quiet part: what is missing, and what an auditor will ask")
    miss = [c for c in r["completeness"] if not c["reported"]]
    st.p(f'{len(miss)} of {len(r["completeness"])} categories have no data. A category with no data must be explained, not dropped: ' + "; ".join(f'{c["name"]} ({c["status"]})' for c in miss[:6]) + ".")
    st.table(["Level", "Check", "Message"], [[c["level"], c["check"], c["message"]] for c in r["checks"]])
    lines = sorted(r["lines"], key=lambda l: -(l["t_co2e"] or 0))[:8]
    st.table(["Line", "Entity", "Scope", "t CO₂e", "Factor", "Evidence", "DQ"], [[l["description"][:40], l["entity_name"], l["scope"], f'{l["t_co2e"]:,.1f}', (l["factor"] or {}).get("id", (l["factor"] or {}).get("name", ""))[:30] if isinstance(l["factor"], dict) else str(l["factor"])[:30], l.get("evidence") or "none", l["dq"]] for l in lines],
             caption="Largest lines. A line with no evidence reference will be queried first [[S:lib]].")
    st.section("audiences", "What this means, by reader")
    st.p("**For the sustainability lead.** The completeness table is the work list before assurance.")
    st.p("**For the assurer.** Fingerprint " + r["fingerprint"] + " reproduces the calculation from the same inputs and factor library.")
    st.section("method", "Method and limits")
    st.ul([f'Scope 2 reported both ways; market-based uses residual-mix factors where supplied.', f'Primary-data share {q["share_primary"]:.0%}; the rest uses spend or default factors.'] + [str(s) for s in r.get("standards", [])[:4]])
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
    st.d["title"] = f'{r["company_name"]}: the EU border bill, {first["year"]}–{last["year"]}'
    st.d["standfirst"] = (f'{r["cbam_sector_name"]}. EU-bound emissions of {first["eu_embedded_co2e_t"]:,.0f} t CO₂e a year turn into a certificate cost of {_usd(first["cbam_cost_usd_m"] * 1e6)} in {first["year"]} '
                          f'rising to {_usd(last["cbam_cost_usd_m"] * 1e6)} by {last["year"]}; the peak year is {_usd(r["peak_annual_cost_usd_m"] * 1e6)} [[S:reg]].')
    st.facts([(f'{r["embedded_carbon_intensity"]:.2f} t', "embedded CO₂ per unit"), (f'{first["eu_embedded_co2e_t"]:,.0f} t', "EU-bound emissions a year"), (f'€{first["net_cbam_price_eur_t"]:.0f}/t', f'net price {first["year"]}'),
              (f'{first["cbam_cost_pct_revenue"]:.1f}%', f'of revenue, {first["year"]}'), (_usd(r["peak_annual_cost_usd_m"] * 1e6), "peak annual cost")])
    st.section("loud", "The loud number: the bill")
    st.figure(S.chart_bars([str(t["year"]) for t in traj], [t["cbam_cost_usd_m"] for t in traj], title="CBAM certificate cost (US$ m)", unit=""), caption="EU ETS price path (NGFS / EMBER calibrated) × EU-bound embedded emissions, net of origin carbon price [[S:reg]].")
    st.table(["Year", "EU ETS €/t", "Origin price €/t", "Net €/t", "Cost US$ m", "% of revenue"], [[t["year"], f'{t["eu_ets_price_eur_t"]:.0f}', f'{t["origin_carbon_price_eur_t"]:.0f}', f'{t["net_cbam_price_eur_t"]:.0f}', f'{t["cbam_cost_usd_m"]:.1f}', f'{t["cbam_cost_pct_revenue"]:.2f}%'] for t in traj])
    st.section("quiet", "The quiet part: what moves the bill")
    st.p(tpl["carbon"] + " The bill is emissions × net price. Two levers sit outside the price path: the embedded intensity (a measured value beats the default) and the origin carbon price, which is deducted under Article 9.")
    inten = r["embedded_carbon_intensity"]
    st.p(f'The default intensity used is {inten:.2f} t per unit. A plant with verified emissions below the default pays less; one above it pays more. Origin price deduction today: €{first["origin_carbon_price_eur_t"]:.0f}/t, so every €10/t of origin price removes €{10 * first["eu_embedded_co2e_t"] / 1e6:.1f} m a year.')
    if r.get("data_gaps"):
        st.callout("Data gaps: " + "; ".join(r["data_gaps"]), "warn")
    st.section("audiences", "What this means, by reader")
    st.p("**For the exporter.** Get the plant's verified embedded emissions: the default is a ceiling you may not need to accept.")
    st.p("**For the EU importer.** The certificate cost lands on the importer of record and will appear in price negotiations; ask for the verified figure.")
    st.section("method", "Method and limits")
    st.ul([r["methodology"], "Certificate price follows the EU ETS path in the model; actual prices will differ.", "Only direct (and, for cement, indirect) emissions of covered goods are counted."])
    return st.build()


# ═════════════════════════════════════════════════════════════════════════════
# registry
# ═════════════════════════════════════════════════════════════════════════════

REPORTS = {
    "physical": {"label": "Physical risk — asset story", "fn": physical,
                 "fields": [{"k": "name", "t": "text", "l": "Asset name"}, {"k": "lat", "t": "number", "l": "Latitude"}, {"k": "lon", "t": "number", "l": "Longitude"},
                            {"k": "sector", "t": "text", "l": "Sector (e.g. cement)"}, {"k": "radius_m", "t": "number", "l": "Model radius (m)", "d": 300}],
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
