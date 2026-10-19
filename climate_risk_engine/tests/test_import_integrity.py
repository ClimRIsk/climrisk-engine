"""Every `from cri... import name` (including lazy imports inside functions that only run on a fallback path) must resolve.
Found 3 broken ones by static audit (tools/audit_imports.py): two had been silently disabling a code path behind `except Exception`."""
import subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_no_unresolvable_internal_imports():
    r = subprocess.run([sys.executable, str(ROOT / "tools" / "audit_imports.py")], capture_output=True, text=True, timeout=150)
    bad = [l for l in r.stdout.splitlines() if l.startswith("BAD")]
    assert not bad, "\n".join(bad)
