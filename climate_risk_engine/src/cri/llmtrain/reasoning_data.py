"""
cri/llmtrain/reasoning_data.py — scenarios and training rows for the reasoning task (see cri.reasoning).

Scenarios are HYPOTHETICAL inputs (invented company, invented figures) so that the numbers the model reads are all supplied in the prompt; the engine (riskmgmt.translate) computes the effect and the
rule-based reasoner writes the gold chain from it. Precedent passages are real, from the case index. Variation that teaches reasoning rather than phrasing:
    - base scenarios across hazards, sizes, insurance levels and covenant regimes, with a third of them placed right at a covenant limit (so the comparison, not the topic, decides the verdict)
    - counterfactual siblings: the same scenario with ONE input changed (insurance, covenant limit, loss size) and the gold chain changed accordingly
    - ablation: the engine results removed from the sources -> the right answer is NOT-COMPUTABLE
    - distractor: an unrelated case passage added -> the answer must not use it
Held-out split by precedent event (whole events never seen in training).

  python -m cri.llmtrain.reasoning_data OUT_DIR [--n 3000] [--seed 5]
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path

from .. import reasoning as R
from .. import riskmgmt as RM

HAZARDS = [
    ("cyclone", ["a coastal refinery", "a port terminal", "a network of coastal substations"]),
    ("flood", ["a chemical plant on a river", "three cement plants", "a food processing plant"]),
    ("wildfire", ["a network of overhead power lines", "a timber mill", "a mining camp"]),
    ("drought", ["a hydropower fleet", "a paper mill", "a bottling plant"]),
    ("heat", ["a data-centre campus", "a steel mill", "a chain of retail warehouses"]),
    ("cold", ["a gas-fired power plant", "a petrochemical complex", "a gas distribution network"]),
    ("hail", ["a car-assembly plant", "a chain of retail warehouses", "a solar farm"]),
    ("earthquake", ["a semiconductor fab", "a hospital group", "a cement works"]),
]
_A = ["Harbor", "Northfield", "Redwood", "Silverline", "Bluewater", "Ironbridge", "Kestrel", "Marlow", "Oakhurst", "Pinecrest", "Stonegate", "Tidewell", "Larkspur", "Westmere"]
_B = ["Materials", "Energy", "Foods", "Chemicals", "Logistics", "Power", "Metals", "Paper", "Retail", "Industries", "Networks", "Systems"]
_C = ["plc", "Inc", "Group", "Holdings"]


def _ln(rng, lo, hi):
    return math.exp(rng.uniform(math.log(lo), math.log(hi)))


def gen_scenario(rng: random.Random, wide: bool = False) -> dict:
    """wide=True draws sizes, leverage and loss multiples from ranges OUTSIDE those used for training (an out-of-distribution test)."""
    hz, assets_l = rng.choice(HAZARDS)
    assets = rng.choice(assets_l)
    rev = round(_ln(rng, 8000, 60000) if wide and rng.random() < 0.5 else _ln(rng, 20, 150) if wide else _ln(rng, 200, 6000), 1)
    margin = rng.uniform(0.04, 0.55) if wide else rng.uniform(0.07, 0.38)
    ebitda = round(rev * margin, 1)
    nd = round(ebitda * (rng.uniform(4.2, 9.0) if wide else rng.uniform(0.4, 4.2)), 1)
    intr = round(max(nd * rng.uniform(0.03, 0.085), 1.0), 1)
    ta = round(rev * rng.uniform(0.8, 2.4), 1)
    eq = round(ta * rng.uniform(0.15, 0.6), 1)
    dmg = round(ebitda * (_ln(rng, 2.4, 9.0) if wide and rng.random() < 0.6 else _ln(rng, 0.02, 2.4)), 1)
    share = rng.choice([0.0, 0.0, rng.uniform(0.2, 0.95), rng.uniform(0.2, 0.95), 1.0])
    rec = round(dmg * share, 1)
    if rng.random() < 0.04:
        rec = round(dmg * rng.uniform(1.05, 1.5), 1)                 # an inconsistent input the engine flags
    loss = {"asset_damage": dmg, "insured_recovery": rec}
    if rng.random() < 0.65:
        loss["bi_revenue_lost"] = round(rev * rng.uniform(0.005, 0.12), 1)
        if rng.random() < 0.4:
            loss["bi_margin"] = round(rng.uniform(margin, min(0.7, margin * 2.2)), 3)
    if rng.random() < 0.4:
        loss["extra_costs"] = round(ebitda * rng.uniform(0.005, 0.08), 1)
    fin = {"revenue": rev, "ebitda": ebitda, "net_debt": nd, "interest_expense": intr, "equity": eq, "total_assets": ta}
    a = RM.translate(fin, loss, None)["after"]
    cov = {}
    regime = rng.choices(["breach", "tight", "pass", "none"], [0.32, 0.22, 0.32, 0.14])[0]
    if regime != "none":
        cands = [k for k, ok in (("max_net_debt_to_ebitda", bool(a.get("net_debt_to_ebitda"))), ("min_interest_cover", bool(a.get("interest_cover"))),
                                 ("min_equity_ratio", bool(a.get("equity_ratio")) and a["equity_ratio"] > 0)) if ok]
        rng.shuffle(cands)
        chosen = cands[:rng.choice([1, 1, 2, 3])]
        for j, key in enumerate(chosen):
            val = {"max_net_debt_to_ebitda": a.get("net_debt_to_ebitda"), "min_interest_cover": a.get("interest_cover"), "min_equity_ratio": a.get("equity_ratio")}[key]
            mx = key == "max_net_debt_to_ebitda"
            role = regime if j == 0 else ("pass" if rng.random() < 0.7 else regime)          # the first covenant carries the regime; the others are mostly comfortable
            if role == "breach":
                f = rng.uniform(0.85, 0.995) if mx else rng.uniform(1.005, 1.25)
            elif role == "tight":
                f = rng.uniform(1.005, 1.16) if mx else rng.uniform(0.875, 0.995)
            else:
                f = rng.uniform(1.25, 2.0) if mx else rng.uniform(0.45, 0.78)
            cov[key] = round(max(val * f, 0.5), 2) if key != "min_equity_ratio" else round(min(max(val * f, 0.02), 0.9), 3)
    name = f"{rng.choice(_A)} {rng.choice(_B)} {rng.choice(_C)}"
    return {"name": name, "hazard": hz, "assets": assets, "fin": fin, "loss": loss, "cov": cov}


def perturb(scn: dict, rng: random.Random) -> tuple:
    """-> (variant scenario, description). Exactly one input changes."""
    v = copy.deepcopy(scn)
    opts = ["insurance", "loss_size"]
    if v["cov"]:
        opts += ["covenant_tighter", "covenant_looser"]
    k = rng.choice(opts)
    if k == "insurance":
        d = v["loss"]["asset_damage"]
        v["loss"]["insured_recovery"] = 0.0 if v["loss"].get("insured_recovery", 0) > 0 else round(d * 0.9, 1)
    elif k == "loss_size":
        f = rng.choice([0.3, 3.0])
        for key in ("asset_damage", "insured_recovery", "bi_revenue_lost", "extra_costs"):
            if v["loss"].get(key):
                v["loss"][key] = round(v["loss"][key] * f, 1)
    else:
        f = rng.choice([0.75, 0.85]) if k == "covenant_tighter" else rng.choice([1.2, 1.35])        # a max-type limit scales by f, a min-type limit by 1/f: f < 1 is tighter for both
        for key, kind in (("max_net_debt_to_ebitda", "max"), ("min_interest_cover", "min"), ("min_equity_ratio", "min")):
            if key in v["cov"]:
                v["cov"][key] = round(v["cov"][key] * f if kind == "max" else v["cov"][key] / f, 3 if key == "min_equity_ratio" else 2)
    return v, k


def _analogs(scn: dict, rng: random.Random, CB) -> tuple:
    """-> (passages, label, channel phrases). One precedent: its frame and one fact or response."""
    try:
        an = CB.analogs(scn["hazard"], scn["assets"], None, k=6)
    except Exception:
        an = []
    if not an:
        return [], None, [], None
    a = rng.choice(an[:4])
    items = a["items"]
    pick = [items[0]]
    rest = [r for r in items[1:] if r["kind"] in ("fact", "resp")]
    if rest:
        pick.append(rng.choice(rest))
    from .cases import CH_DESC
    ch = [CH_DESC[c] for c in (a.get("shared_channels") or []) if c in CH_DESC][:3]
    return [{"text": r["text"]} for r in pick], a["title"].split(" (")[0], ch, a["event"]


def _distractor(scn: dict, rng: random.Random, CB, avoid_event) -> list:
    others = [h for h, _ in HAZARDS if h != scn["hazard"]]
    try:
        for h in rng.sample(others, len(others)):
            for a in CB.analogs(h, "", None, k=3):
                if a["event"] != avoid_event and h not in scn["hazard"]:
                    return [{"text": a["items"][0]["text"]}]
    except Exception:
        pass
    return []


def _row(scn, led, S, ids, ans, kind, split, meta, qi, engine=True):
    return {"kind": kind, "split": split, "meta": meta,
            "messages": [{"role": "system", "content": R.SYS_REASON}, {"role": "user", "content": R.user_message(scn, S, qi)}, {"role": "assistant", "content": ans}]}


def make_rows(scn: dict, rng: random.Random, CB, split: str, pid: str, role: str = "base", perturbation: str = None, allow_extras: bool = True) -> list:
    """All rows derived from one scenario: the base row plus (sometimes) an ablation row and a distractor row. Returns rows with their own meta."""
    rows = []
    led = R.ledger(scn)
    qi = rng.randrange(4)
    passages, label, channels, event = scn.get("_analog") or (None, None, None, None)
    scn["analogs"] = passages or []
    S, ids = R.sources(scn, led)
    ans = R.compose(scn, led, ids, channels, True, label)
    base_meta = {"pair": pid, "role": role, "perturbation": perturbation, "expected": led["verdict"], "engine": True, "irrelevant": [], "analog_event": event,
                 "min_headroom": led["binding"]["headroom_pct"] if led["binding"] else None, "sources": S,
                 "attribution": {"new": [R.COV_NAMES[k] for k in led["breach_new"]], "pre": [R.COV_NAMES[k] for k in led["breach_pre"]]}}
    rows.append(_row(scn, led, S, ids, ans, "reason_base" if role == "base" else "reason_variant", split, base_meta, qi))
    if allow_extras and rng.random() < 0.18:
        S2, ids2 = R.sources(scn, led, engine=False)
        ans2 = R.compose(scn, led, ids2, channels, False, label)
        rows.append(_row(scn, led, S2, ids2, ans2, "reason_ablation", split, {**base_meta, "role": "ablation", "expected": "NOT-COMPUTABLE", "engine": False, "sources": S2}, qi))
    if allow_extras and passages and rng.random() < 0.18:
        d = _distractor(scn, rng, CB, event)
        if d:
            first = rng.random() < 0.5
            S3, ids3 = R.sources(scn, led, True, d, first)
            ans3 = R.compose(scn, led, ids3, channels, True, label)
            rows.append(_row(scn, led, S3, ids3, ans3, "reason_distractor", split, {**base_meta, "role": "distractor", "irrelevant": ids3["distractor"], "sources": S3}, qi))
    return rows


def _split_of(event, seed_key: str) -> str:
    h = int(hashlib.sha1(str(event or seed_key).encode()).hexdigest(), 16) % 10
    return "test" if h < 2 else "valid" if h < 3 else "train"


def build(out: str, n: int = 3000, seed: int = 5, index_path: str = None) -> dict:
    from ..resident import casebase as CB
    if index_path:
        import os
        os.environ["CRI_CASE_INDEX"] = index_path
    rng = random.Random(seed)
    outd = Path(out)
    outd.mkdir(parents=True, exist_ok=True)
    data = {"train": [], "valid": [], "test": []}
    for i in range(n):
        scn = gen_scenario(rng)
        passages, label, ch, event = _analogs(scn, rng, CB)
        scn["_analog"] = (passages, label, ch, event)
        split = _split_of(event, f"s{i}")
        pid = f"p{seed}-{i}"
        rows = make_rows(scn, rng, CB, split, pid)
        if rng.random() < 0.4:                                       # a counterfactual sibling: same scenario, one input changed
            v, k = perturb(scn, rng)
            v["_analog"] = scn["_analog"]
            rows += make_rows(v, rng, CB, split, pid, "variant", k, allow_extras=False)
        data[split] += rows
    for i in range(max(10, n // 12)):                                  # out-of-distribution scenarios, test only
        for _try in range(60):                                         # precedent must belong to the held-out events, so no precedent leaks from training
            scn = gen_scenario(rng, wide=True)
            passages, label, ch, event = _analogs(scn, rng, CB)
            if event is None or _split_of(event, "x") == "test":
                break
        else:
            continue
        scn["_analog"] = (passages, label, ch, event)
        rows = make_rows(scn, rng, CB, "test", f"ood{seed}-{i}", allow_extras=False)
        for r in rows:
            r["meta"]["ood"] = True
        data["test"] += rows
    stats = {}
    for s, rs in data.items():
        with open(outd / f"{s}.jsonl", "w") as f, open(outd / f"{s}_meta.jsonl", "w") as g:
            for r in rs:
                f.write(json.dumps({"messages": r["messages"]}, ensure_ascii=False) + "\n")
                g.write(json.dumps(r, ensure_ascii=False) + "\n")
        stats[s] = {"rows": len(rs), "kinds": dict(Counter(r["kind"] for r in rs)), "verdicts": dict(Counter(r["meta"]["expected"] for r in rs if r["kind"] != "reason_ablation")),
                    "analog_events": len({r["meta"]["analog_event"] for r in rs if r["meta"]["analog_event"]})}
    (outd / "dataset_report.json").write_text(json.dumps(stats, indent=1))
    return stats


def main() -> None:
    a = sys.argv[1:]
    out = a[0]
    n = int(a[a.index("--n") + 1]) if "--n" in a else 3000
    seed = int(a[a.index("--seed") + 1]) if "--seed" in a else 5
    idx = a[a.index("--index") + 1] if "--index" in a else None
    print(json.dumps(build(out, n, seed, idx), indent=1))


if __name__ == "__main__":
    main()
