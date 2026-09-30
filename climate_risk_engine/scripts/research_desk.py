"""
Research Desk toolkit — engine numbers and factual images for climrisk.io articles.

All outputs are JSON on stdout (numbers) or PNG files (images). Every number comes
from the ClimRisk engine and the open datasets it uses; the article writer must cite
them as such.

  python scripts/research_desk.py asset   --lat 29.48 --lon -95.63 [--value 5e8 --emissions 2e5 --sector cement --type "integrated dry"]
  python scripts/research_desk.py company --name "Holcim"
  python scripts/research_desk.py hazard  --lat 23.76 --lon 90.40
  python scripts/research_desk.py policy  --lat 51.87 --lon 4.33 --sector cement
  python scripts/research_desk.py search  --name "Tata"
  python scripts/research_desk.py satellite --lat 23.76 --lon 90.40 --start 2024-06-01 --end 2024-07-15 --out x.png [--title "..."] [--km 4]
  python scripts/research_desk.py assets-map --name "Holcim" --out map.png
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _dump(x):
    import numpy as np

    def conv(o):
        if isinstance(o, dict):
            return {k: conv(v) for k, v in o.items() if not str(k).startswith("_")}
        if isinstance(o, (list, tuple)):
            return [conv(v) for v in o]
        if isinstance(o, np.ndarray):
            return conv(o.tolist())
        if isinstance(o, np.generic):
            return o.item()
        return o
    print(json.dumps(conv(x), indent=1, default=str))


def _keys(name: str) -> list[str]:
    from cri.data.asset_register import search
    res = search(name, limit=10)
    if not res:
        raise SystemExit(json.dumps({"found": False, "note": f"'{name}' not in the asset register"}))
    return [r["key"] for r in res if r["match"] == "group"] or [res[0]["key"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["asset", "company", "hazard", "policy", "search", "satellite", "assets-map"])
    ap.add_argument("--lat", type=float); ap.add_argument("--lon", type=float)
    ap.add_argument("--name"); ap.add_argument("--value", type=float); ap.add_argument("--emissions", type=float)
    ap.add_argument("--sector"); ap.add_argument("--type", default="")
    ap.add_argument("--start"); ap.add_argument("--end"); ap.add_argument("--out"); ap.add_argument("--title")
    ap.add_argument("--km", type=float, default=4.0); ap.add_argument("--top", type=int, default=20)
    a = ap.parse_args()

    if a.cmd == "asset":
        from cri.finance.analyst import analyse_asset
        r = analyse_asset(a.lat, a.lon, name=a.name, sector=a.sector, atype=a.type or None,
                          emissions_t=a.emissions, value_usd=a.value, n=3000)
        r["financial"]["scenarios"] = {k: {kk: v[kk] for kk in ("label", "total_npv", "transition_npv", "stranded",
                                                                  "closure_share", "price_median", "annual_cost_median")}
                                       for k, v in r["financial"]["scenarios"].items()}
        _dump(r)
    elif a.cmd == "company":
        from cri.finance.analyst import analyse_company
        r = analyse_company(_keys(a.name), top_n=a.top, n=1500)
        for s in r.get("scenarios", {}).values():
            s.pop("histogram", None)
        _dump(r)
    elif a.cmd == "hazard":
        from cri.climate.tropical_cyclone import site_hazard
        from cri.finance import geo
        from cri.finance.flood_hazard import site_flood
        h = site_hazard(a.lat, a.lon).to_dict()
        fl = site_flood(a.lat, a.lon)
        mean = lambda d: {k: (round(sum(v) / len(v), 2) if v else 0.0) for k, v in (d or {}).items()}  # noqa: E731
        _dump({"admin": geo.admin_at(a.lat, a.lon), "cyclone": h,
               "river_flood_mean_depth_m_by_rp": mean(fl["river"].get("depths")),
               "coastal_flood_mean_depth_m_by_year_rp": mean(fl["coastal"].get("depths")),
               "flood_sources": [fl["river"].get("source"), fl["coastal"].get("source")]})
    elif a.cmd == "policy":
        from cri.finance import asset_value, geo, policy
        adm = geo.admin_at(a.lat, a.lon)
        _dump(policy.applicable(adm.get("iso3"), adm.get("name"), a.sector or "", a.type,
                                asset_value.cost_key(a.sector or "", a.type)))
    elif a.cmd == "search":
        from cri.data.asset_register import search
        _dump(search(a.name, limit=15))
    elif a.cmd == "satellite":
        from cri.research.figures import satellite_image
        _dump(satellite_image(a.lat, a.lon, a.out, a.start, a.end, half_km=a.km, title=a.title))
    elif a.cmd == "assets-map":
        from cri.research.figures import assets_map
        _dump(assets_map(_keys(a.name), a.out, a.title or f"{a.name}: assets by physical screening level"))


if __name__ == "__main__":
    main()
