"""Model validation pack: a self-contained report, in the shape a bank's model-risk team or an auditor expects, generated from what the engine measures about itself.

Nothing in it is typed by hand. Layer 1 (analytic benchmarks) and layer 2 (synthetic calibration of the hindcast test) are executed when the pack is built; layer 3 (the
global sweep) is read from the sweep results file; layer 4 (independent-source cross-check) is read from the comparison file when one exists. Failures are listed as prominently
as passes. usage:  python -m cri.modellab.validation_pack sweep.jsonl [compare.md] [out_dir]
"""
from __future__ import annotations

import collections
import datetime
import html
import json
import statistics as st
import sys
from pathlib import Path

import numpy as np

NOT_MODELLED = [
    ("Storm surge and coastal flooding from the sea", "Needs hydrodynamic surge modelling with bathymetry and tide gauges; the flood study handles rain and river flooding only."),
    ("Sea-level rise", "No open, commercially licensed regional sea-level product is wired in yet (NASA/IPCC AR6 projections are the candidate)."),
    ("Hail, tornado, convective wind", "Reanalysis and climate models do not resolve them; needs radar and storm-report climatologies."),
    ("Subsidence, erosion, permafrost", "Needs ground-motion and soil data not yet integrated."),
    ("Wind, cyclone, drought and fire under future climate", "Projections cover extreme rain and extreme heat only; the CMIP6 products used do not provide a wind-extreme or fire index, and cyclone change is not robust at grid scale."),
    ("Independent audit and station validation", "Reanalysis and satellite-based products are not weather stations. Station-level validation (GHCN-Daily, national services) is the next independent check."),
    ("Scale", "A study takes seconds to minutes per hazard per place; screening a million assets needs a batch pipeline on pre-computed grid cells, which the portfolio mode implements per 0.25° cell."),
]
SOURCES = [
    ("NASA POWER (MERRA-2 / IMERG)", "Daily rain, temperature, humidity, wind, 1981 onward", "CC BY 4.0: commercial use permitted with attribution"),
    ("ERA5 via Open-Meteo free tier", "Daily/hourly reanalysis, 1979 onward (development cross-check only)", "Free tier is non-commercial; not used for production numbers"),
    ("Copernicus ERA5 / ERA5-Land / GloFAS (CDS)", "Reanalysis, land and river discharge", "Free licence, commercial use with attribution (requires user registration; adapter planned)"),
    ("World Bank CCKP CMIP6 0.25° (NEX-GDDP-CMIP6 downscaled)", "Climate-scenario ensemble, 1950-2100", "CMIP6 output CC0; portal content MPL; commercial use with acknowledgement"),
    ("USGS earthquake catalogue (ComCat/FDSN)", "Earthquakes M4+, 1973 onward", "US Government work: public domain"),
    ("IBTrACS", "Tropical-cyclone best tracks", "Open, with citation (NOAA NCEI)"),
    ("Terrain tiles (AWS Open Data / Copernicus DEM)", "Elevation for flood and landslide studies", "Open data licences, attribution"),
]


def _category(d: dict) -> str:
    s = d.get("status", "")
    if s == "ERROR": return "error"
    if s.startswith("FAILED"): return "verification failed"
    if "no model" in s: return "no model (no exposure / too little data)"
    if "small drift" in s: return "validated, small drift"
    if "hindcast FAILED" in s or "no skill" in s.lower(): return "hindcast failed / no skill"
    if "diagnosed" in s: return "validated after diagnosis"
    if "validated" in s: return "validated"
    if "verified only" in s: return "verified only"
    return "other"


def layer1() -> list:
    """Execute every analytic benchmark now. Returns [(module, n_pass, n_total, [failed names])]."""
    from . import bucket, cyclone, evt, fire, heat, landslide, seismic
    out = []
    for nm, fn in (("Extreme-value fitting and return levels", evt.run_benchmarks), ("Heat stress (WBGT, Magnus)", heat.run_benchmarks), ("Fire Weather Index", fire.run_benchmarks), ("Soil-water bucket / river routing", bucket.run_benchmarks),
                   ("Earthquake (Gutenberg–Richter, ground motion)", seismic.run_benchmarks), ("Tropical-cyclone wind field (Holland)", cyclone.bench_holland), ("Landslide infinite slope", landslide.run_benchmarks)):
        try:
            b = fn()
            out.append((nm, sum(1 for x in b if x["ok"]), len(b), [x["name"] for x in b if not x["ok"]]))
        except Exception as e:                                         # noqa: BLE001
            out.append((nm, 0, 1, [f"could not run: {e}"]))
    return out


def layer2(n_seeds: int = 150) -> dict:
    """Calibrate the hindcast test itself: false-alarm rate on stationary synthetic records and detection rate on records with a strong trend."""
    from . import evt
    rng = np.random.default_rng(2026)
    fa = det = tot = 0
    for _ in range(n_seeds):
        x = rng.gumbel(60, 12, 60)
        h = evt.hindcast(x)
        if h["ok"]:
            tot += 1; fa += 0 if h["passed"] else 1
        x2 = x + 0.5 * np.arange(60)
        h2 = evt.hindcast(x2)
        if h2["ok"] and not h2["passed"]:
            det += 1
    return {"n": tot, "false_fail": fa / max(tot, 1), "detect": det / max(n_seeds, 1)}


def layer3(rows: list) -> dict:
    by = collections.defaultdict(collections.Counter)
    for d in rows:
        by[d["hazard"]][_category(d)] += 1
    zones = collections.defaultdict(collections.Counter)
    for d in rows:
        zones[d.get("zone", "?")][_category(d)] += 1
    fails = [d for d in rows if _category(d) in ("hindcast failed / no skill", "verification failed", "error")]
    drift = [abs(d["drift"]) for d in rows if d.get("drift") is not None]
    ks = [d for d in rows if d["hazard"] in ("rain", "wind", "heat", "fire") and d.get("ks_p") is not None]
    return {"by": by, "zones": zones, "fails": fails, "n": len(rows), "sites": len({d["site"] for d in rows}),
            "hc_pass": sum(1 for d in ks if d.get("hc_passed")), "hc_n": len(ks), "drift_median": (st.median(drift) if drift else None)}


def build(sweep_path: str, compare_path: str = None, out_dir: str = ".", seeds: int = 150) -> dict:
    from .. import __version__
    rows = [json.loads(l) for l in Path(sweep_path).read_text().splitlines() if l.strip()]
    l1, l2, l3 = layer1(), layer2(seeds), layer3(rows)
    cmp_txt = Path(compare_path).read_text() if compare_path and Path(compare_path).exists() else ""
    e = html.escape
    when = datetime.date.today().isoformat()
    p1 = sum(a for _, a, _, _ in l1); t1 = sum(b for _, _, b, _ in l1)
    md = [f"# ClimRisk Model Lab: validation pack\n", f"Engine version {__version__}, generated {when}. Every number below is computed or read from a results file when this report is built; none is typed by hand.\n",
          "## 1. Scope and method\n", "For each hazard the engine builds a model on its own, then tests it in four independent layers. A model is called *validated* only when it predicted data it had not been shown. The layers: (1) analytic benchmarks, (2) calibration of the hindcast test on synthetic records with known answers, (3) a global sweep of real data at many sites in every climate zone, (4) comparison with an independent second data source.\n",
          f"## 2. Layer 1: analytic benchmarks ({p1} of {t1} passed)\n", "| Component | Passed | Failures |\n|---|---|---|"]
    md += [f"| {nm} | {a} of {b} | {', '.join(f) or 'none'} |" for nm, a, b, f in l1]
    md += ["", f"## 3. Layer 2: is the hindcast test itself calibrated?\n", f"On {l2['n']} synthetic stationary records the test wrongly failed {l2['false_fail'] * 100:.0f} % (nominal about 10 %); on records with a strong built-in trend it detected the problem {l2['detect'] * 100:.0f} % of the time.\n",
           f"## 4. Layer 3: global sweep ({l3['n']} studies at {l3['sites']} sites)\n"]
    cats = sorted({c for v in l3["by"].values() for c in v})
    md += ["| Hazard | n | " + " | ".join(cats) + " |", "|---|---|" + "---|" * len(cats)]
    md += [f"| {h} | {sum(c.values())} | " + " | ".join(str(c.get(k, 0)) for k in cats) + " |" for h, c in sorted(l3["by"].items())]
    missing = [h for h in ("rain", "wind", "heat", "fire", "drought", "river", "cyclone", "earthquake", "flood", "landslide") if h not in l3["by"]]
    if missing:
        md += ["", "Not covered by this sweep: " + ", ".join(missing) + ". They are validated site by site when run, but have no worldwide pass rate yet."]
    md += ["", f"Extreme-value hindcast: {l3['hc_pass']} of {l3['hc_n']} passed outright (nominal false-fail rate about 10 %)." + (f" Median size of the design-level drift where a hindcast failed or flagged: {l3['drift_median'] * 100:.0f} %." if l3["drift_median"] is not None else ""), "",
           "### Failures and no-skill results (listed in full)\n"]
    md += [f"- {d['site']} ({d.get('zone', '?')}) / {d['hazard']}: {d['status']}" for d in l3["fails"]] or ["- none"]
    md += ["", "### By climate zone\n", "| Zone | " + " | ".join(cats) + " |", "|---|" + "---|" * len(cats)]
    md += [f"| {z} | " + " | ".join(str(c.get(k, 0)) for k in cats) + " |" for z, c in sorted(l3["zones"].items())]
    md += ["", "## 5. Layer 4: independent-source cross-check\n", cmp_txt.strip() or "Not yet run: needs the second data source to be available (see the sweep's part 2).", "", "## 6. Data sources and licences\n", "| Source | Used for | Licence |\n|---|---|---|"]
    md += [f"| {a} | {b} | {c} |" for a, b, c in SOURCES]
    md += ["", "## 7. What the engine does not model (stated, not hidden)\n"] + [f"- **{a}.** {b}" for a, b in NOT_MODELLED]
    md += ["", "## 8. Reproduce\n", "`python -m cri.modellab.validation_pack sweep.jsonl compare.md out/` regenerates this report. The sweep is `python tools/global_sweep.py`.\n", "This report is generated evidence, not an independent audit."]
    text = "\n".join(md)
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "validation-pack.md").write_text(text, encoding="utf-8")
    (out / "validation-pack.html").write_text(_to_html(text), encoding="utf-8")
    return {"layer1": l1, "layer2": l2, "layer3": {k: v for k, v in l3.items() if k not in ("by", "zones", "fails")}, "files": [str(out / "validation-pack.md"), str(out / "validation-pack.html")], "n_failures": len(l3["fails"])}


def _to_html(md: str) -> str:
    """Minimal markdown to HTML (headings, tables, bullets, paragraphs): enough for the pack, no dependencies."""
    e = html.escape
    out = ['<!doctype html><meta charset="utf-8"><title>Model validation pack</title><style>body{font:16px/1.55 Georgia,serif;max-width:920px;margin:2rem auto;padding:0 1rem;color:#1b2430}table{border-collapse:collapse;margin:1rem 0;font:14px system-ui}td,th{border:1px solid #cfd6df;padding:5px 9px;text-align:left}th{background:#eef2f7}code{background:#eef2f7;padding:1px 4px}</style>']
    lines = md.split("\n"); i = 0
    import re

    def inline(s):
        s = e(s)
        s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
        s = re.sub(r"\*(.+?)\*", r"<i>\1</i>", s)
        return re.sub(r"`(.+?)`", r"<code>\1</code>", s)
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("|"):
            tbl = []
            while i < len(lines) and lines[i].startswith("|"):
                tbl.append(lines[i]); i += 1
            rows = [[c.strip() for c in r.strip("|").split("|")] for r in tbl if not set(r.replace("|", "").strip()) <= set("-: ")]
            out.append("<table><tr>" + "".join(f"<th>{inline(c)}</th>" for c in rows[0]) + "</tr>" + "".join("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in r) + "</tr>" for r in rows[1:]) + "</table>")
            continue
        if ln.startswith("#"):
            n = len(ln) - len(ln.lstrip("#")); out.append(f"<h{n}>{inline(ln[n:].strip())}</h{n}>")
        elif ln.startswith("- "):
            items = []
            while i < len(lines) and lines[i].startswith("- "):
                items.append(lines[i][2:]); i += 1
            out.append("<ul>" + "".join(f"<li>{inline(x)}</li>" for x in items) + "</ul>"); continue
        elif ln.strip():
            out.append(f"<p>{inline(ln)}</p>")
        i += 1
    return "\n".join(out)


if __name__ == "__main__":
    r = build(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None, sys.argv[3] if len(sys.argv) > 3 else ".")
    print(json.dumps({k: v for k, v in r.items() if k != "layer1"}, indent=1))
