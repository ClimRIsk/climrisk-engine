"""
data/asset_register.py — company → physical assets, with precomputed hazard screening.

Built offline by scripts/build_asset_register.py + scripts/precompute_asset_screening.py
from Climate TRACE v6 (global heavy-industry assets with owners) and EPA GHGRP 2023
(US facilities with parent companies). Loaded into memory once; every lookup is local
and takes milliseconds — no network, no modelling at query time.

Screening per asset (precomputed):
  cyclone   exact IBTrACS + Holland model: annual rates, 1-in-100-year wind
  flood     ground elevation (AWS terrain tiles, ~150 m) — validated signal is
            built-up land below 3 m in low-lying coastal plains; this point
            elevation is a screen, the 10 m site analysis is the confirmation
Coverage: heavy industry (power, cement, steel, refining, petrochemicals, aluminium,
mining) 65–99 % with owners; food, textiles, glass, lime have no owner data.
Ownership is as reported by the sources (no parent/subsidiary roll-up).
"""
from __future__ import annotations

import json
import re
import threading
import time
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np

_DATA = Path(__file__).resolve().parent
_SUFFIX = re.compile(
    r"\b(the|inc|incorporated|corp|corporation|co|company|companies|llc|l\.l\.c|lp|l\.p|ltd|limited|plc|"
    r"holdings?|group|sa|nv|ag|se|na|usa|us|trust|partners|pvt|private|pte|bhd|tbk|spa|s\.a|gmbh|kk|oyj|asa|ab)\b")

SAFFIR = [(17.5, "tropical storm"), (33.0, "Cat 1"), (43.0, "Cat 2"), (50.0, "Cat 3"), (58.0, "Cat 4"), (70.0, "Cat 5")]
LEVELS = ["NEGLIGIBLE", "LOW", "MODERATE", "HIGH", "CRITICAL"]


def norm(name: str) -> str:
    s = str(name).lower().replace("&", " and ")
    s = re.sub(r"\(.*?\)", " ", s)
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    s = _SUFFIX.sub(" ", s)
    return re.sub(r"\s+", " ", s).strip()


@lru_cache(maxsize=1)
def _load() -> dict:
    t0 = time.time()
    z = np.load(_DATA / "asset_register.npz", allow_pickle=False)
    a = {k: z[k] for k in z.files}
    meta = json.loads((_DATA / "asset_register_companies.json").read_text())
    comps = meta["companies"]
    keys = list(comps.keys())
    a["_companies"] = comps
    a["_keys"] = keys
    a["_nassets"] = np.array([len(comps[k]["assets"]) for k in keys])
    a["_meta"] = {"built": meta.get("built"), "n_assets": meta.get("n_assets"),
                  "n_companies": meta.get("n_companies"), "load_s": round(time.time() - t0, 2)}
    return a


def warm_up_async() -> None:
    """Load the register in the background at engine start-up so the first query is instant."""
    threading.Thread(target=_load, daemon=True).start()


def stats() -> dict:
    return _load()["_meta"]


def search(query: str, limit: int = 12) -> list[dict]:
    """Rank owners by exact > prefix > all-words match, then by number of assets."""
    R = _load()
    q = norm(query)
    if not q:
        return []
    toks = q.split()
    scored = []
    for i, k in enumerate(R["_keys"]):
        if k == q:
            s = 3
        elif k.startswith(q):
            s = 2
        elif all(t in k.split() for t in toks):
            s = 1
        else:
            continue
        scored.append((s, int(R["_nassets"][i]), k))
    scored.sort(key=lambda x: (-x[0], -x[1]))
    out = []
    for s, n, k in scored[:limit]:
        c = R["_companies"][k]
        idx = np.array(c["assets"])
        countries = sorted({str(R["countries"][j]) for j in R["country"][idx]})
        out.append({"key": k, "name": c["name"], "n_assets": n, "countries": countries[:8],
                    "n_countries": len(countries), "match": ["", "words", "prefix", "exact"][s]})
    return out


def _category(v: float) -> str:
    lab = "below tropical-storm force"
    for thr, name in SAFFIR:
        if v >= thr:
            lab = name
    return lab


def _cyclone_level(rp100: float) -> str:
    return ("CRITICAL" if rp100 >= 58 else "HIGH" if rp100 >= 50 else "MODERATE" if rp100 >= 33
            else "LOW" if rp100 >= 17.5 else "NEGLIGIBLE")


def _flood_level(elev: float) -> str:
    """
    Validated signal = land within ~3 m of SEA level (coastal plains). Readings below −2 m
    are inland depressions (Caspian, Dead Sea) or sea-floor pixels at the ~150 m screening
    resolution — not flagged here; the 10 m site analysis decides.
    """
    if not np.isfinite(elev) or elev < -2.0:
        return "NEGLIGIBLE"
    return "HIGH" if elev < 3.0 else "MODERATE" if elev < 6.0 else "NEGLIGIBLE"


def company_exposure(keys: list[str], max_assets: int = 2000) -> dict:
    """Assets for one or more owner keys (from search), with screening and a portfolio summary."""
    R = _load()
    idx: list[int] = []
    names = []
    for k in keys:
        c = R["_companies"].get(k) or R["_companies"].get(norm(k))
        if c:
            idx += c["assets"]
            names.append(c["name"])
    idx = sorted(set(idx))[:max_assets]
    if not idx:
        return {"found": False, "keys": keys}
    ii = np.array(idx)
    em = R["emissions_t"][ii]
    assets = []
    for j in ii:
        cyc = _cyclone_level(float(R["tc_rp100_ms"][j]))
        fl = _flood_level(float(R["elev_m"][j]))
        top = max(cyc, fl, key=LEVELS.index)
        assets.append({
            "i": int(j), "name": str(R["name"][j]), "lat": round(float(R["lat"][j]), 5),
            "lon": round(float(R["lon"][j]), 5), "country": str(R["countries"][R["country"][j]]),
            "sector": str(R["sectors"][R["sector"][j]]), "type": str(R["atype"][j]),
            "capacity": None if not np.isfinite(R["capacity"][j]) else float(R["capacity"][j]),
            "emissions_t": None if not np.isfinite(R["emissions_t"][j]) else float(R["emissions_t"][j]),
            "source": str(R["source"][j]),
            "elev_m": None if not np.isfinite(R["elev_m"][j]) else round(float(R["elev_m"][j]), 1),
            "cyclone": {"rate_ts": round(float(R["tc_rate_ts"][j]), 3),
                        "rate_cat1": round(float(R["tc_rate_cat1"][j]), 3),
                        "rp100_ms": round(float(R["tc_rp100_ms"][j]), 1),
                        "rp100_category": _category(float(R["tc_rp100_ms"][j])),
                        "worst": str(R["tc_worst_names"][R["tc_worst"][j]]) or None,
                        "level": cyc},
            "flood_screen": fl,
            "screen_level": top,
        })
    assets.sort(key=lambda a: (-LEVELS.index(a["screen_level"]), -(a["emissions_t"] or 0)))

    def share(pred):
        n = sum(1 for a in assets if pred(a))
        return round(n / len(assets), 3)

    em_known = np.nansum(em) if np.isfinite(em).any() else None
    def em_share(pred):
        if not em_known:
            return None
        return round(float(sum((a["emissions_t"] or 0) for a in assets if pred(a)) / em_known), 3)

    countries: dict = {}
    sectors: dict = {}
    for a in assets:
        countries[a["country"]] = countries.get(a["country"], 0) + 1
        sectors[a["sector"]] = sectors.get(a["sector"], 0) + 1
    hi = lambda a: LEVELS.index(a["screen_level"]) >= 3               # noqa: E731
    return {
        "found": True, "companies": names, "n_assets": len(assets),
        "summary": {
            "countries": dict(sorted(countries.items(), key=lambda kv: -kv[1])),
            "sectors": dict(sorted(sectors.items(), key=lambda kv: -kv[1])),
            "emissions_t_known": None if em_known is None else float(em_known),
            "share_assets_high_or_critical": share(hi),
            "share_emissions_high_or_critical": em_share(hi),
            "share_assets_cyclone_cat1plus_100yr": share(lambda a: a["cyclone"]["rp100_ms"] >= 33),
            "share_assets_cyclone_cat3plus_100yr": share(lambda a: a["cyclone"]["rp100_ms"] >= 50),
            "share_assets_low_lying_lt3m": share(lambda a: a["elev_m"] is not None and -2 <= a["elev_m"] < 3),
            "by_level": {lv: sum(1 for a in assets if a["screen_level"] == lv) for lv in LEVELS},
        },
        "assets": assets,
        "method": ("Precomputed screening: cyclone = IBTrACS 1980–2025 + Holland model (validated vs station "
                   "winds); flood = point elevation (~150 m) with the validated <3 m low-lying signal. "
                   "Open any asset for the full 10 m satellite site analysis."),
        "sources": ["Climate TRACE v6 asset ownership", "EPA GHGRP 2023 facilities", "NOAA IBTrACS",
                    "AWS Terrain Tiles"],
        "caveats": ["Ownership as reported by the sources — subsidiaries under different names are separate "
                    "entries; select all that apply.",
                    "Heavy-industry coverage only (power, cement, steel, refining, chemicals, mining); offices, "
                    "retail and light manufacturing sites are not in the register."],
    }
