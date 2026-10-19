"""AST audit: every broad exception handler in src/cri, classified by what it does with the failure.

Classes (worst first):
  VALUE   handler returns/assigns a concrete-looking value (number, string, list/dict literal) -> failure can pass as an answer
  SWALLOW handler is only pass/continue/break -> failure vanishes
  LOGONLY handler logs and falls through (result depends on what follows)
  RERAISE re-raises / raises a new error / returns an explicit error structure   (fine)
  NONE    returns None / False   (usually fine: reads as "no result")
Output JSONL + summary. Usage: python tools/audit_fallbacks.py [--out audit.jsonl]
"""
import ast, json, sys
from pathlib import Path
SRC = Path(__file__).resolve().parent.parent / "src" / "cri"
BROAD = {"Exception", "BaseException"}


def is_broad(h):
    if h.type is None: return True
    ts = h.type.elts if isinstance(h.type, ast.Tuple) else [h.type]
    return any(isinstance(t, ast.Name) and t.id in BROAD for t in ts)


def lit(n):
    if isinstance(n, ast.Constant): return repr(n.value)[:40]
    if isinstance(n, (ast.List, ast.Dict, ast.Tuple, ast.Set)): return ast.unparse(n)[:60]
    return None


def classify(h):
    body = h.body
    stmts = [s for s in body if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant))]
    if not stmts: return "SWALLOW", ""
    if all(isinstance(s, (ast.Pass, ast.Continue, ast.Break)) for s in stmts): return "SWALLOW", type(stmts[0]).__name__
    for s in stmts:
        if isinstance(s, ast.Raise): return "RERAISE", ""
    for s in stmts:
        if isinstance(s, ast.Return):
            v = s.value
            if v is None or (isinstance(v, ast.Constant) and v.value in (None, False)): return "NONE", ""
            if isinstance(v, ast.Dict) and any(isinstance(k, ast.Constant) and k.value in ("error", "available", "ok", "failed", "reason", "status") for k in v.keys if k is not None):
                return "RERAISE", "error-structure"
            return "VALUE", "return " + ast.unparse(v)[:80]
    for s in stmts:
        if isinstance(s, ast.Assign) and lit(s.value) is not None and not (isinstance(s.value, ast.Constant) and s.value.value is None):
            return "VALUE", ast.unparse(s)[:80]
        if isinstance(s, ast.AugAssign): return "VALUE", ast.unparse(s)[:80]
    return "LOGONLY", ast.unparse(stmts[0])[:60]


class V(ast.NodeVisitor):
    def __init__(self, f): self.f, self.fn, self.out = f, ["<module>"], []
    def visit_FunctionDef(self, n):
        self.fn.append(n.name); self.generic_visit(n); self.fn.pop()
    visit_AsyncFunctionDef = visit_FunctionDef
    def visit_ExceptHandler(self, h):
        if is_broad(h):
            c, d = classify(h)
            self.out.append({"file": str(self.f.relative_to(SRC)), "line": h.lineno, "func": self.fn[-1], "class": c, "detail": d})
        self.generic_visit(h)


def main():
    rows = []
    for f in sorted(SRC.rglob("*.py")):
        try: t = ast.parse(f.read_text(encoding="utf-8"))
        except SyntaxError: continue
        v = V(f); v.visit(t); rows += v.out
    out = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else "/tmp/audit.jsonl"
    with open(out, "w") as fh:
        for r in rows: fh.write(json.dumps(r) + "\n")
    from collections import Counter
    c = Counter(r["class"] for r in rows)
    print(len(rows), dict(c))


if __name__ == "__main__": main()
