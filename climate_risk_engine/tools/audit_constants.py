"""Scan src/cri for numbers that reach an output without a stated source, and for words that admit a guess.

Three signals, each reported per file and per line so a human can judge:
  UNSOURCED   a module-level UPPER_CASE constant, or a dict/list of numbers, with no source marker (a citation word, DOI, URL, table/eq/fig reference,
              or 'user input') on the same line or the 4 lines above/below.
  ADMISSION   a comment/docstring/string containing a guess word: rough, proxy, assume, illustrative, heuristic, placeholder, arbitrary, hand-tuned,
              ballpark, typical, plausible, made up, fudge, 'for now'.
  SILENTDEF   a .get(key, <non-trivial number>) or `x or <number>` fallback in code, which turns a missing input into an answer.

Usage:  python tools/audit_constants.py [--json out.json] [--top 40] [path]
This finds places to look. It does not decide that a number is wrong: a sourced constant can carry its source three lines away, and an unsourced one can be right.
"""
from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "src" / "cri"
SRC = re.compile(r"(source|doi|https?://|et al|\bper\b|epa|ipcc|iso\s?\d|cfr|table\s?\w|eq\.|equation|fig\.|ngfs|imo|jrc|ar[456]\b|world bank|noaa|fema|unfccc|ghg protocol|pcaf|"
                 r"user input|user-supplied|supplied by|ecb|eba|basel|tcfd|issb|csrd|iea|eurostat|usgs|nasa|copernicus|wri|undrr|oecd|\b(19|20)\d\d\b)", re.I)
ADMIT = re.compile(r"\b(rough(ly)?|proxy|assum(e|ed|es|ption)|illustrative|heuristic|placeholder|arbitrary|hand[- ]tuned|ballpark|typical(ly)?|plausible|made[- ]up|fudge|for now|"
                   r"back-of-envelope|guess|stylised|stylized|synthetic|dummy|mock)\b", re.I)
TRIVIAL = {0, 1, 2, 3, 4, 5, 10, 12, 24, 60, 100, 1000, 0.5, 0.25, 0.1, 0.01, 0.001, 365, 365.25, 1e6, 1e9, 1e3, 90, 180, 360, 3600}


def _trivial(v) -> bool:
    return isinstance(v, bool) or (isinstance(v, (int, float)) and (abs(v) in TRIVIAL or abs(v) < 1e-9))


def _nums(node):
    return [n.value for n in ast.walk(node) if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)) and not isinstance(n.value, bool) and not _trivial(n.value)]


def scan_file(path: Path) -> dict:
    text = path.read_text(errors="replace")
    lines = text.splitlines()
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return {}
    out = {"UNSOURCED": [], "ADMISSION": [], "SILENTDEF": []}

    def window(i, a=4, b=4):
        return "\n".join(lines[max(0, i - 1 - a): min(len(lines), i + b)])

    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            tg = node.targets if isinstance(node, ast.Assign) else [node.target]
            names = [t.id for t in tg if isinstance(t, ast.Name)]
            if names and any(n.isupper() or n.startswith("_") and n[1:].isupper() for n in names) and node.value is not None:
                nums = _nums(node.value)
                if len(nums) >= 1 and not SRC.search(window(node.lineno, 4, getattr(node, "end_lineno", node.lineno) - node.lineno + 4)):
                    out["UNSOURCED"].append((node.lineno, names[0], len(nums)))
    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "get" and len(n.args) == 2:
            d = n.args[1]
            if isinstance(d, ast.Constant) and isinstance(d.value, (int, float)) and not _trivial(d.value):
                out["SILENTDEF"].append((n.lineno, lines[n.lineno - 1].strip()[:110]))
        if isinstance(n, ast.BoolOp) and isinstance(n.op, ast.Or):
            last = n.values[-1]
            if isinstance(last, ast.Constant) and isinstance(last.value, (int, float)) and not _trivial(last.value):
                out["SILENTDEF"].append((n.lineno, lines[n.lineno - 1].strip()[:110]))
    for i, ln in enumerate(lines, 1):
        s = ln.strip()
        if (s.startswith("#") or '"""' in s or "'" in s or '"' in s) and ADMIT.search(s) and len(s) < 260:
            out["ADMISSION"].append((i, s[:140]))
    return out


def main(argv):
    top, jout, root = 40, None, ROOT
    a = list(argv)
    while a:
        x = a.pop(0)
        if x == "--top":
            top = int(a.pop(0))
        elif x == "--json":
            jout = a.pop(0)
        else:
            root = Path(x)
    res = {}
    for p in sorted(root.rglob("*.py")):
        r = scan_file(p)
        if r and any(r.values()):
            res[str(p.relative_to(ROOT.parent.parent)) if ROOT in p.parents else str(p)] = r
    tot = {k: sum(len(v[k]) for v in res.values()) for k in ("UNSOURCED", "ADMISSION", "SILENTDEF")}
    print("totals:", tot, "files with findings:", len(res))
    ranked = sorted(res.items(), key=lambda kv: -(3 * len(kv[1]["UNSOURCED"]) + 2 * len(kv[1]["SILENTDEF"]) + len(kv[1]["ADMISSION"])))
    for f, r in ranked[:top]:
        print(f"{f}: unsourced {len(r['UNSOURCED'])}, silent-default {len(r['SILENTDEF'])}, admissions {len(r['ADMISSION'])}")
    if jout:
        Path(jout).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:])
