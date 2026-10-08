"""Portfolio screening: run the hazard studies for a list of assets, sharing work between assets that fall in the same 0.25° grid cell.

Why grid cells: every data source behind the studies (NASA POWER 0.5°x0.625°, ERA5 0.25°, CMIP6 0.25°) is a grid, so two assets in one 0.25° cell receive the same hazard
numbers; running the study once per cell and reusing it is exact, not an approximation. A 1,000-asset portfolio concentrated in 60 cities costs 60 studies per hazard.
Results are appended to a JSON-lines store as they finish, so an interrupted run resumes where it stopped and a re-run costs nothing.
"""
from __future__ import annotations

import csv
import io
import json
import re
from pathlib import Path
from typing import Callable, Optional

DEFAULT_HAZARDS = ("rain", "heat", "wind", "drought", "cyclone", "earthquake")
MAX_ASSETS = 500
_ALIASES = {"name": ("name", "asset", "asset_name", "site", "id", "asset_id"), "lat": ("lat", "latitude", "y"), "lon": ("lon", "lng", "long", "longitude", "x"),
            "value": ("value", "asset_value", "exposure", "tiv", "sum_insured")}


def cell(lat: float, lon: float) -> tuple:
    """Centre of the 0.25° cell containing the point (the shared study location)."""
    la = round((int((lat + 90) // 0.25) * 0.25 + 0.125) - 90, 3)
    lo = round((int(((lon + 180) % 360) // 0.25) * 0.25 + 0.125) - 180, 3)
    return la, lo


def parse_assets(text: str) -> list:
    """Read a CSV (header row with name, lat, lon and optionally value). Rejects bad coordinates with the row number, never guesses."""
    rows = list(csv.reader(io.StringIO(text.strip().replace("\r\n", "\n"))))
    if len(rows) < 2:
        raise ValueError("the file needs a header row and at least one asset")
    head = [h.strip().lower() for h in rows[0]]
    idx = {}
    for k, names in _ALIASES.items():
        for n in names:
            if n in head:
                idx[k] = head.index(n); break
    if "lat" not in idx or "lon" not in idx:
        raise ValueError("the header must contain latitude and longitude columns (lat, lon)")
    out = []
    for r_i, r in enumerate(rows[1:], start=2):
        if not any(c.strip() for c in r):
            continue
        try:
            la, lo = float(r[idx["lat"]]), float(r[idx["lon"]])
        except (ValueError, IndexError):
            raise ValueError(f"row {r_i}: latitude/longitude are not numbers")
        if not (-90 <= la <= 90 and -180 <= lo <= 180):
            raise ValueError(f"row {r_i}: coordinates {la}, {lo} are outside the valid range")
        nm = r[idx["name"]].strip() if "name" in idx and idx["name"] < len(r) and r[idx["name"]].strip() else f"Asset {len(out) + 1}"
        val = None
        if "value" in idx and idx["value"] < len(r):
            try:
                val = float(re.sub(r"[^0-9.\-eE]", "", r[idx["value"]]))
            except ValueError:
                val = None
        out.append({"name": nm[:80], "lat": la, "lon": lo, "value": val})
    if not out:
        raise ValueError("no assets found")
    if len(out) > MAX_ASSETS:
        raise ValueError(f"at most {MAX_ASSETS} assets per run (got {len(out)}); split the file")
    return out


def _lvl(n: dict):
    lv = (n or {}).get("levels") or {}
    for k in (100, "100", 475, "475"):
        if k in lv and lv[k] is not None:
            return k, lv[k]
    return None, None


def brief(res: dict) -> dict:
    """Compact, JSON-safe summary of a study result for the portfolio table."""
    n = res.get("numbers") or {}
    k, v = _lvl(n)
    ev = n.get("evidence") or {}
    out = {"status": res.get("status"), "ok": res.get("ok"), "grade": ev.get("grade", "n/a"), "label": res.get("label"), "level_T": k, "level": v}
    sc = n.get("scenarios") if isinstance(n.get("scenarios"), dict) else None
    if sc and sc.get("table"):
        t = sc["table"]
        for tag, key in (("mid_245", "ssp245:2040-2059"), ("end_585", "ssp585:2080-2099")):
            if key in t:
                f = t[key]["change"]; out[tag] = f["median"]
        out["scen_mode"] = sc.get("mode")
    bt = (sc or {}).get("backtest")
    if isinstance(bt, dict) and bt.get("ok"):
        out["backtest"] = bt["verdict"]
    return out


def _key(h, la, lo):
    return f"{h}|{la}|{lo}"


def screen(assets: list, hazards=DEFAULT_HAZARDS, store: Optional[str] = None, progress: Optional[Callable] = None, runner: Optional[Callable] = None) -> dict:
    """Run every hazard for every distinct cell. Returns {"assets": [...], "results": {key: brief}, "cells": n, "runs": n_new, "reused": n_cached}."""
    if runner is None:
        from . import hazards as H
        runner = lambda h, la, lo, nm: H.run_generic(h, la, lo, nm, {}, None)       # noqa: E731
    done: dict = {}
    sp = Path(store) if store else None
    if sp and sp.exists():
        for line in sp.read_text().splitlines():
            try:
                j = json.loads(line); done[j["key"]] = j["brief"]
            except Exception:                                          # noqa: BLE001
                continue
    cells: dict = {}
    for a in assets:
        c = cell(a["lat"], a["lon"]); a["cell"] = list(c)
        cells.setdefault(c, []).append(a["name"])
    todo = [(h, c) for c in cells for h in hazards if _key(h, *c) not in done]
    new = 0
    for n_i, (h, c) in enumerate(todo, start=1):
        if progress:
            progress(f"{h} at cell {c[0]:.3f}, {c[1]:.3f} ({n_i} of {len(todo)})")
        try:
            b = brief(runner(h, c[0], c[1], f"cell {c[0]:.2f},{c[1]:.2f}"))
        except Exception as e:                                         # noqa: BLE001
            msg = str(e)[:140]
            if "daily allowance" in msg or "rate-limiting" in msg:
                b = {"status": "deferred (data service allowance used; run again later)", "ok": None, "grade": "n/a", "deferred": True}
            else:
                b = {"status": f"error: {msg}", "ok": False, "grade": "n/a"}
        if not b.get("deferred"):
            done[_key(h, *c)] = b
            if sp:
                with sp.open("a") as f:
                    f.write(json.dumps({"key": _key(h, *c), "brief": b}) + "\n")
        else:
            done[_key(h, *c)] = b                                      # shown, but not stored: a re-run tries again
        new += 1
    return {"assets": assets, "hazards": list(hazards), "results": {k: v for k, v in done.items()}, "cells": len(cells), "runs": new, "reused": len(cells) * len(hazards) - new}


def _fmt(b: dict) -> str:
    if not b:
        return "—"
    if b.get("deferred"):
        return "deferred"
    if str(b.get("status", "")).startswith("no model"):
        return "no model"
    s = b.get("status") or ""
    if b.get("level") is None:
        return ("model built" if b.get("ok") else "failed") + f" [{b.get('grade', 'n/a')}]"
    v = b["level"]
    t = b.get("level_T")
    txt = f"{v:.2g}" if abs(v) < 10 else f"{v:.0f}"
    flag = "!" if "FAILED" in s else ""
    return f"{txt} ({t}-yr) [{b.get('grade', 'n/a')}]{flag}"


def table(res: dict) -> tuple:
    """(head, rows) for display: one row per asset, one column per hazard: level, return period and evidence grade; '!' marks a failed hindcast."""
    hz = res["hazards"]
    head = ["Asset", "Latitude", "Longitude", "Grid cell"] + hz
    rows = []
    for a in res["assets"]:
        c = tuple(a["cell"])
        rows.append([a["name"], f'{a["lat"]:.4f}', f'{a["lon"]:.4f}', f"{c[0]:.3f}, {c[1]:.3f}"] + [_fmt(res["results"].get(_key(h, *c))) for h in hz])
    return head, rows


def digest(res: dict) -> dict:
    """Portfolio-level facts: grade mix per hazard, the assets with the largest projected increase, and where evidence is thin."""
    out = {"hazards": {}, "movers": [], "weak": []}
    for h in res["hazards"]:
        g: dict = {}
        for a in res["assets"]:
            b = res["results"].get(_key(h, *a["cell"]))
            if b:
                g[b.get("grade", "n/a")] = g.get(b.get("grade", "n/a"), 0) + 1
        out["hazards"][h] = g
    movers = []
    for a in res["assets"]:
        for h in res["hazards"]:
            b = res["results"].get(_key(h, *a["cell"])) or {}
            if b.get("mid_245") is not None:
                chg = (b["mid_245"] - 1.0) * 100 if b.get("scen_mode") == "ratio" else b["mid_245"]
                movers.append((a["name"], h, chg, "%" if b.get("scen_mode") == "ratio" else "°C"))
            if b.get("grade") in ("C", "D") or "FAILED" in str(b.get("status", "")):
                out["weak"].append((a["name"], h, b.get("grade"), b.get("status")))
    movers.sort(key=lambda m: -m[2] if m[3] == "%" else -m[2] * 10)
    out["movers"] = movers[:10]
    return out


def to_csv(res: dict) -> str:
    head, rows = table(res)
    sio = io.StringIO(); w = csv.writer(sio); w.writerow(head)
    for r in rows:
        w.writerow(r)
    return sio.getvalue()
