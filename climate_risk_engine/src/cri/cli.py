"""
`cri` command line.

  cri ingest portfolio.xlsx [-o results.csv] [--draws 2000]
        Run every asset (name, lat, lon [, sector, type, capacity, emissions_t, value_usd, company]) through the
        Risk Analyst and write one consolidated CSV.
  cri run --company ultratech [--scenarios all|nze_2050,current_policies]
        Legacy company engine: NPV / EV per scenario.
  cri water LAT LON [--sector mining]
        WRI Aqueduct 4.0 indicators at a coordinate.
  cri lca steel --route eaf --kwh 450 --country DEU      (also: aluminium, ammonia)
        Cradle-to-gate carbon footprint per tonne.
  cri serve [--port 8000]
        Start the local API.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _ingest(a) -> int:
    from .portfolio_export import analyse_rows, read_portfolio, to_csv
    try:
        rows = read_portfolio(a.file)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    bad = sum(1 for r in rows if r.get("error"))
    print(f"Parsing {len(rows)} assets... ✓" + (f"  ({bad} with invalid coordinates)" if bad else ""))

    def progress(k, n, item):
        r = item["input"]
        if item.get("error"):
            print(f"  [{k}/{n}] {r['name'][:40]:40s}  ✗ {item['error']}")
        else:
            res = item["result"]
            fin = res["financial"]
            cur = next((v for v in fin["scenarios"].values() if v["label"] == "Current Policies"), None)
            pv = cur["total_npv"]["mean"] if cur else None
            print(f"  [{k}/{n}] {r['name'][:40]:40s}  AAL {fin['physical']['aal_pct']:.2f}%"
                  + (f"  PV loss (Current Policies) ${pv / 1e6:,.1f}M" if pv else "  (no asset value — give value_usd or capacity)"))

    items = analyse_rows(rows, n=a.draws, progress=progress)
    out = Path(a.output or Path(a.file).with_suffix("").as_posix() + "_climrisk.csv")
    out.write_text(to_csv(items), encoding="utf-8")
    print(f"Consolidated CSV: {out}  ({len(items)} rows)")
    return 0


def _run(a) -> int:
    from .api.main import COMPANY_REGISTRY, SCENARIO_REGISTRY, run_engine
    key = a.company.lower()
    if key not in COMPANY_REGISTRY:
        print(f"unknown company '{a.company}'. Known: {', '.join(sorted(COMPANY_REGISTRY))}", file=sys.stderr)
        return 2
    scen = list(SCENARIO_REGISTRY) if a.scenarios == "all" else [s.strip() for s in a.scenarios.split(",")]
    c = COMPANY_REGISTRY[key]
    print(f"{c.name}")
    print("─" * 60)
    for s in scen:
        if s not in SCENARIO_REGISTRY:
            print(f"  {s}: unknown scenario (known: {', '.join(SCENARIO_REGISTRY)})")
            continue
        r = run_engine(company=c, scenario=SCENARIO_REGISTRY[s])
        y30 = next((y for y in r.years if y.year == 2030), None)
        print(f"  {s:22s} EV ${r.enterprise_value:,.0f}M" + (
            f" · 2030 carbon cost ${y30.carbon_cost:,.1f}M · physical loss ${y30.physical_loss_cost:,.1f}M" if y30 else ""))
    return 0


def _water(a) -> int:
    from .connectors.wri_aqueduct import point
    print(json.dumps(point(a.lat, a.lon, a.sector), indent=2, ensure_ascii=False))
    return 0


def _lca(a) -> int:
    from .lca import products
    try:
        if a.product == "steel":
            if not a.route:
                print("error: --route bf_bof|eaf|dri_eaf is required", file=sys.stderr)
                return 2
            r = products.steel(a.route, a.kwh, a.country, a.g_per_kwh)
        elif a.product == "aluminium":
            r = products.aluminium(a.kwh, a.country, a.g_per_kwh)
        else:
            r = products.ammonia(a.kwh, a.country, a.g_per_kwh, gas_gj_per_t=a.gas_gj)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    u = r["uncertainty"]
    print(f"{r['declared_unit']}: {r['footprint_kg_co2e_per_t']:,.0f} kg CO2e  (P5–P95 {u['p5']:,.0f}–{u['p95']:,.0f})")
    for k, v in r["by_stage"].items():
        if v:
            print(f"  {k:45s} {v:10,.1f}")
    for e in r["exclusions"]:
        print(f"  excluded: {e}")
    return 0


def _serve(a) -> int:
    import uvicorn
    uvicorn.run("cri.api.main:app", host=a.host, port=a.port)
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="cri", description="ClimRisk engine command line")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("ingest", help="portfolio xlsx/csv → consolidated risk CSV")
    s.add_argument("file")
    s.add_argument("-o", "--output")
    s.add_argument("--draws", type=int, default=2000)
    s.set_defaults(fn=_ingest)

    s = sub.add_parser("run", help="company engine run across scenarios")
    s.add_argument("--company", required=True)
    s.add_argument("--scenarios", default="all")
    s.set_defaults(fn=_run)

    s = sub.add_parser("water", help="WRI Aqueduct 4.0 at a coordinate")
    s.add_argument("lat", type=float)
    s.add_argument("lon", type=float)
    s.add_argument("--sector")
    s.set_defaults(fn=_water)

    s = sub.add_parser("lca", help="cradle-to-gate product carbon footprint")
    s.add_argument("product", choices=["steel", "aluminium", "ammonia"])
    s.add_argument("--kwh", type=float, required=True, help="plant electricity kWh per tonne")
    s.add_argument("--country")
    s.add_argument("--route", choices=["bf_bof", "eaf", "dri_eaf"])
    s.add_argument("--g-per-kwh", dest="g_per_kwh", type=float)
    s.add_argument("--gas-gj", dest="gas_gj", type=float)
    s.set_defaults(fn=_lca)

    s = sub.add_parser("serve", help="start the local API")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.set_defaults(fn=_serve)

    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
