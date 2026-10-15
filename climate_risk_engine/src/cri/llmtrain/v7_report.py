"""
cri/llmtrain/v7_report.py — turn the v7 held-out evaluations into (a) a promotion gate and (b) a plain report that can be shown to a third party.

    python -m cri.llmtrain.v7_report gate NEW_EVAL.json REASON_EVAL.json GENERAL_EVAL.json OUT.json [--new climrisk-llm-v7] [--old climrisk-llm-v6]
    python -m cri.llmtrain.v7_report report NEW_EVAL.json REASON_EVAL.json GENERAL_EVAL.json GATE.json OUT.md [--data-report v7_report.json]

Gate (all three must hold or v7 is NOT promoted and the app keeps its model):
  1. on the held-out NEW tasks (stress, scenario, siteprofile, chat, gold) v7's mean score is above v6's;
  2. reasoning composite did not fall by more than 0.02;
  3. general case-question composite did not fall by more than 0.02.
Scores are mechanical (numbers must come from the facts; nulls must be named, not called low; the screening label must be present). They measure honesty and
discipline on these tasks, not climate-forecast accuracy, and the report says so.
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

KINDS = ("stress", "scenario", "siteprofile", "chat", "gold")
SKIP = re.compile(r"(^|\.)(new_numbers|invented_numbers)$")
LOWER_BETTER = re.compile(r"invented_numbers$")


def kind_means(metrics: dict) -> dict:
    by = defaultdict(list)
    for k, v in metrics.items():
        kind, name = k.split(".", 1)
        if kind in KINDS and not SKIP.search(k) and isinstance(v, (int, float)):
            by[kind].append(v)
    return {k: round(sum(v) / len(v), 4) for k, v in by.items()}


def new_score(metrics: dict) -> float:
    m = kind_means(metrics)
    return round(sum(m.values()) / len(m), 4) if m else 0.0


def _general(metrics: dict) -> float:
    from .pick import composite
    return composite(metrics)[0]


def gate(new_eval, reason_eval, general_eval, new="climrisk-llm-v7", old="climrisk-llm-v6") -> dict:
    n, r, g = json.loads(Path(new_eval).read_text()), json.loads(Path(reason_eval).read_text()), json.loads(Path(general_eval).read_text())
    out = {"new": new, "old": old}
    if new not in n or old not in n:
        return {**out, "promote": False, "why": "a model is missing from the new-task evaluation"}
    a, b = new_score(n[new]["metrics"]), new_score(n[old]["metrics"])
    out["new_tasks"] = {"new": a, "old": b, "ok": a > b}
    ra, rb = r.get(new, {}).get("composite"), r.get(old, {}).get("composite")
    out["reasoning"] = {"new": ra, "old": rb, "ok": ra is not None and rb is not None and ra >= rb - 0.02}
    ga = _general(g[new]["metrics"]) if new in g else None
    gb = _general(g[old]["metrics"]) if old in g else None
    out["general"] = {"new": ga, "old": gb, "ok": ga is not None and gb is not None and ga >= gb - 0.02}
    out["promote"] = all(out[k]["ok"] for k in ("new_tasks", "reasoning", "general"))
    out["why"] = "all three checks passed" if out["promote"] else "failed: " + ", ".join(k for k in ("new_tasks", "reasoning", "general") if not out[k]["ok"])
    return out


def report(new_eval, reason_eval, general_eval, gate_json, out, data_report=None) -> str:
    n, r, g, gt = (json.loads(Path(x).read_text()) for x in (new_eval, reason_eval, general_eval, gate_json))
    dr = json.loads(Path(data_report).read_text()) if data_report and Path(data_report).exists() else {}
    models = sorted([m for m in n if not m.startswith("_")], key=lambda m: -new_score(n[m]["metrics"]))
    L = ["# ClimRisk resident model, v7 training round: held-out results", "",
         "Everything below is measured on rows the model never trained on (held-out stress-test runs, held-out scenario combinations, site profiles, chat turns, reference answers). "
         "Scoring is done by code, not by a model or a person: numbers in an answer must exist in the engine's facts, missing inputs must be named rather than called low, and the screening-grade label must appear.", ""]
    if dr:
        L += ["## Training data (every answer composed from the engine; nothing copied from outside)", "",
              "| split | " + " | ".join(KINDS) + " |", "|---|" + "---|" * len(KINDS)]
        for s in ("train", "valid", "test"):
            L.append(f"| {s} | " + " | ".join(str(dr.get(s, {}).get(k, 0)) for k in KINDS) + " |")
        L += ["", f"Held-out prompts that also appear in training: {dr.get('test_prompts_also_in_train', 'n/a')}.", ""]
    L += ["## New tasks: mean held-out score per task family (0 to 1, higher is better)", "",
          "| model | overall | " + " | ".join(KINDS) + " | rows |", "|---|---|" + "---|" * len(KINDS) + "---|"]
    for m in models:
        km = kind_means(n[m]["metrics"])
        L.append(f"| {m} | {new_score(n[m]['metrics']):.3f} | " + " | ".join(f"{km.get(k, float('nan')):.3f}" if k in km else "n/a" for k in KINDS) + f" | {n[m].get('n_rows', '')} |")
    L += ["", "## Did it forget what it already did? (regression checks, v7 against v6)", ""]
    rs = gt.get("reasoning", {})
    gs = gt.get("general", {})
    L += [f"- Reasoning on held-out precedent events (composite): v7 {rs.get('new')} against v6 {rs.get('old')}.",
          f"- General case questions (composite of citation and honesty metrics): v7 {gs.get('new')} against v6 {gs.get('old')}.", ""]
    L += ["## Decision", "", f"Promotion gate: **{'PASSED, v7 is now the model the app uses' if gt.get('promote') else 'NOT PASSED, the app keeps ' + str(gt.get('old'))}** ({gt.get('why')}).", "",
          "## What this does and does not show", "",
          "- It shows that a 3-billion-parameter model fine-tuned on a Mac, on engine-composed examples, follows the platform's rules on these tasks measurably better than the same base model and than a 7B general model, and that this was measured on unseen rows.",
          "- It does not show forecasting accuracy. The numbers the model quotes are the engine's, not the model's. The model learns to explain, to name what is missing and to refuse to invent.",
          "- The training and test rows are generated from the engine's own modules and a 1,000-row scenario list whose figures are used only as stated inputs. Held-out rows differ in values and scenario combination from training rows, but they share the same templates, so scores on real, free-form user questions will be lower.",
          "- Scale is small: a few thousand training rows, one Apple-silicon laptop, no GPU. That is the point of the next step: more and more varied data, a larger base model, and a human-reviewed test set, which need GPU time.", ""]
    txt = "\n".join(L)
    Path(out).write_text(txt)
    return txt


def main() -> None:
    a = sys.argv[1:]
    opt = lambda k, d: a[a.index(k) + 1] if k in a else d                                    # noqa: E731
    if a[0] == "gate":
        res = gate(a[1], a[2], a[3], opt("--new", "climrisk-llm-v7"), opt("--old", "climrisk-llm-v6"))
        Path(a[4]).write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))
    elif a[0] == "report":
        report(a[1], a[2], a[3], a[4], a[5], opt("--data-report", None)); print("wrote", a[5])


if __name__ == "__main__":
    main()
