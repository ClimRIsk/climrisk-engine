"""Static check for latent ImportErrors: every `from X import name` (including lazy ones inside functions, which only fail when that
code path runs) is resolved against the real module. Catches renamed/removed symbols hiding behind `except Exception`."""
import ast, importlib, sys
from pathlib import Path
SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))
bad, checked, skipped = [], 0, []
for f in sorted((SRC / "cri").rglob("*.py")):
    rel = f.relative_to(SRC).with_suffix("")
    pkg_parts = list(rel.parts)
    is_pkg = pkg_parts[-1] == "__init__"
    if is_pkg: pkg_parts = pkg_parts[:-1]
    try: tree = ast.parse(f.read_text(encoding="utf-8"))
    except SyntaxError as e: bad.append((str(rel), 0, f"SyntaxError {e}")); continue
    for n in ast.walk(tree):
        if not isinstance(n, ast.ImportFrom): continue
        if n.level:
            base = pkg_parts if is_pkg else pkg_parts[:-1]
            base = base[:len(base) - (n.level - 1)] if n.level > 1 else base
            mod = ".".join(base + ([n.module] if n.module else []))
        else:
            mod = n.module
        if not mod or not mod.startswith("cri"): continue
        try: m = importlib.import_module(mod)
        except Exception as e:
            skipped.append((str(rel), n.lineno, mod, type(e).__name__)); continue
        for a in n.names:
            if a.name == "*": continue
            checked += 1
            if hasattr(m, a.name): continue
            try: importlib.import_module(mod + "." + a.name); continue
            except Exception: pass
            bad.append((str(rel), n.lineno, f"from {mod} import {a.name}"))
print("checked", checked, "names;", len(bad), "unresolvable;", len(skipped), "modules could not be imported here")
for b in bad: print("BAD ", *b)
for s in skipped[:40]: print("SKIP", *s)
