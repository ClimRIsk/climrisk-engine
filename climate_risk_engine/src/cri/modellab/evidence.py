"""Evidence grade (A-D) for every study result: how much weight do these numbers deserve, and why?

The grade is a transparent score, not an opinion: each factor is stated with the points it earned, so a reader (or an auditor) can disagree with a single line instead of with the whole.

Factors (only those that apply to a study count towards its maximum):
  verification   the physics passed analytic benchmarks (a failed benchmark forces grade D)             0..1
  validation     predicted data it had not seen (held-out years, gauge, inventory)                      0..2
  licence        the data may be used commercially (not a development-only free tier)                   0..1
  record length  enough years for the return periods quoted (extreme-value studies)                     -1..1
  precision      the bootstrap range of the 100-year level is narrow enough to act on                   -1..1
  projections    the climate-model ensemble reproduced what was observed at this site (if projected)    -1..1
  cross-check    an independent second source agreed (when one has been run)                            -1..1
Grade = share of the applicable maximum: A >= 80 %, B >= 55 %, C >= 30 %, D below; "n/a" when no model was built.
"""
from __future__ import annotations


def _validation(status: str):
    s = status.lower()
    if "validated against era5-land" in s or "glofas" in s or "model product" in s:
        return 1, "validated against another model's product, not an observation"
    if "hindcast-validated" in s or "validated against a gauge" in s or "validated against the inventory" in s:
        return 2, "predicted data it had never seen"
    if "small drift" in s:
        return 1, "predicted held-out data with a small, practically minor drift"
    if "diagnosed" in s:
        return 1, "passed only after a diagnosis chosen with the answer in view"
    if "no skill" in s or "hindcast failed" in s:
        return 0, "did NOT predict held-out data"
    return 0, "not tested against unseen data (record or inventory too limited)"


def grade(res: dict) -> dict:
    """Score a study result dict. Returns {grade, score, max, items, summary}."""
    status = str(res.get("status", ""))
    n = res.get("numbers") or {}
    if res.get("kind") == "no_model" or status.lower().startswith("no model"):
        return {"grade": "n/a", "score": 0, "max": 0, "items": [], "summary": "No model was built, so there is nothing to grade."}
    items = []

    def add(name, pts, mx, note):
        items.append({"factor": name, "points": pts, "max": mx, "note": note})

    bench = res.get("verification") or []
    failed = [b for b in bench if not b.get("ok", True)]
    add("Physics verified", 0 if failed else 1, 1, f"{len(bench) - len(failed)} of {len(bench)} analytic benchmarks passed" if bench else "no analytic benchmark for this study")
    v, vn = _validation(status)
    add("Validated on unseen data", v, 2, vn)
    blob = " ".join([str(n.get("source", "")), " ".join(map(str, res.get("limitations") or []))]).lower()
    dev_only = "non-commercial" in blob or "via open-meteo" in str(n.get("source", "")).lower()
    add("Data licence", 0 if dev_only else 1, 1, "free tier licensed for non-commercial use only: development, not production" if dev_only else "source may be used commercially with attribution")
    if n.get("n_years"):
        ny = n["n_years"]
        add("Record length", 1 if ny >= 40 else -1 if ny < 25 else 0, 1, f"{ny} years of data")
    ci, lv = n.get("ci"), n.get("levels")
    try:
        if ci and lv:
            c = ci.get("100") or ci.get(100)
            w = (c[1] - c[0]) / abs(lv[100] if 100 in lv else lv["100"])
            add("Precision of the 100-year level", 1 if w <= 0.30 else -1 if w >= 0.70 else 0, 1, f"90 % range is {w * 100:.0f} % of the estimate")
    except Exception:                                                  # noqa: BLE001
        pass
    sc = (n.get("scenarios") or {}).get("backtest") if isinstance(n.get("scenarios"), dict) else None
    if sc and sc.get("ok"):
        add("Climate projections backtested", {"pass": 1, "partial": 0, "fail": -1}[sc["verdict"]], 1, sc["summary"])
    elif isinstance(n.get("scenarios"), dict) and n["scenarios"].get("table"):
        add("Climate projections backtested", 0, 1, "projections shown but could not be checked against observations here")
    cc = n.get("crosscheck")
    if isinstance(cc, dict) and cc.get("agree") is not None:
        add("Independent second source", 1 if cc["agree"] else -1, 1, cc.get("note", ""))
    score = sum(i["points"] for i in items)
    mx = sum(i["max"] for i in items)
    frac = score / mx if mx else 0
    g = "D" if failed else "A" if frac >= 0.80 else "B" if frac >= 0.55 else "C" if frac >= 0.30 else "D"
    weakest = min(items, key=lambda i: i["points"] - i["max"])
    why = {"A": "strong evidence on every factor that applies", "B": "good evidence; the weakest point is listed below", "C": "usable for screening, not for decisions that carry money without a check", "D": "treat as indicative only"}[g]
    return {"grade": g, "score": score, "max": mx, "items": items, "summary": f"Evidence grade {g}: {why}. Score {score} of {mx}; weakest factor: {weakest['factor'].lower()} ({weakest['note']})."}


def attach(res: dict) -> dict:
    """Add the grade to a result: headline tile, a table in the report, and the standfirst."""
    try:
        g = grade(res)
    except Exception:                                                  # noqa: BLE001
        return res
    if g["grade"] == "n/a":
        return res
    res.setdefault("numbers", {})["evidence"] = g
    res["headline"] = list(res.get("headline") or []) + [(g["grade"], "evidence grade (A best)")]
    blocks = list(res.get("blocks") or [])
    sec = [("h2", "evidence", "How much weight these numbers deserve"),
           ("table", ["Factor", "What the engine found", "Points"], [[i["factor"], i["note"], f'{i["points"]:+d} of {i["max"]}' if i["points"] < 0 else f'{i["points"]} of {i["max"]}'] for i in g["items"]],
            "Every factor is scored the same way for every hazard and every place, so grades can be compared across sites. Grade: A at least 80 % of the applicable points, B at least 55 %, C at least 30 %, otherwise D; a failed benchmark forces D."),
           ("callout", g["summary"], "good" if g["grade"] in ("A", "B") else "warn" if g["grade"] == "C" else "bad")]
    at = 2 if len(blocks) >= 2 and blocks[0][0] == "h2" else 0
    res["blocks"] = blocks[:at] + sec + blocks[at:]
    res["standfirst"] = str(res.get("standfirst", "")) + f" Evidence grade: {g['grade']}."
    return res
