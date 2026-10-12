"""
cri/licence.py — time- and usage-limited trial licences that work offline on Windows, macOS and Linux.

How it works
  * You (the vendor) hold a private signing key. `issue` signs a small JSON payload: who it is for, how long it lasts after first use, and the usage caps.
  * The app contains only the PUBLIC key (src/cri/data/licence_public.pem, or CRI_LICENCE_PUBKEY for tests). It verifies the signature locally; no server is needed.
  * First activation starts the clock. Usage counters and the last-seen date live in ~/.climrisk/licence_state.json, authenticated with a key derived from the licence id and this machine,
    so casual edits are detected. A system clock moved backwards is detected and refused.
  * The engine enforces it in an HTTP middleware (cri.api.main) when CRI_LICENCE_MODE=required; the desktop app starts the engine with that setting.

What it is not
  * Not tamper-proof: a determined user with the source and the machine can remove a local check. It stops casual copying, and it is the right weight for a reviewer or a pilot.
    For paying customers add online activation with a short offline grace period; the pilot contract does the rest.

CLI
  python -m cri.licence keygen [--dir DIR]                      create a signing key pair (private stays in DIR, outside any repository) and write the public key into the engine
  python -m cri.licence issue --name N --email E [--days 14] [--runs 30] [--ai 100] [--activate-by YYYY-MM-DD] [--dir DIR]
  python -m cri.licence status | install TOKEN | install-file PATH
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import platform
import sys
import time
import uuid
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

PREFIX = "CRILIC1"
DEFAULTS = {"days": 14, "runs": 30, "ai": 100}

# Paths that spend the "runs" allowance (analysis actions) and the "ai" allowance (language-model calls). Everything else is free while the licence is active.
RUN_PREFIXES = ("/insurance/analyse", "/portfolio", "/asset", "/assess", "/analyze", "/physical", "/reports", "/generate", "/scenarios", "/climate-scenario", "/carbon", "/regulatory",
                "/stress", "/sensitivity", "/screen", "/runs", "/resident/oed/run", "/resident/lab", "/story", "/transition-plan", "/real-estate", "/materiality", "/lca", "/predictive")
AI_PREFIXES = ("/resident/ask", "/resident/reason", "/resident/cases", "/agent/")
EXEMPT_PREFIXES = ("/licence", "/license", "/health", "/docs", "/redoc", "/openapi")


class LicenceError(Exception):
    def __init__(self, state: str, message: str):
        super().__init__(message)
        self.state, self.message = state, message


# ── helpers ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

def home() -> Path:
    return Path(os.environ.get("CRI_HOME") or (Path.home() / ".climrisk"))


def _b64(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).decode().rstrip("=")


def _unb64(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def _today() -> date:
    return datetime.now(timezone.utc).date()


def _public_key():
    from cryptography.hazmat.primitives import serialization
    pem = os.environ.get("CRI_LICENCE_PUBKEY")
    if pem:
        data = pem.encode() if "BEGIN" in pem else Path(pem).read_bytes()
    else:
        f = Path(__file__).parent / "data" / "licence_public.pem"
        if not f.exists():
            raise LicenceError("unconfigured", "this build has no licence public key")
        data = f.read_bytes()
    return serialization.load_pem_public_key(data)


# ── vendor side ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────

def keygen(dirpath: Optional[str] = None, write_public: bool = True) -> dict:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import ed25519
    d = Path(dirpath or (Path.home() / ".climrisk-signing"))
    d.mkdir(parents=True, exist_ok=True)
    priv_f = d / "licence_private.pem"
    if priv_f.exists():
        raise FileExistsError(f"{priv_f} already exists; refusing to overwrite a signing key")
    k = ed25519.Ed25519PrivateKey.generate()
    priv_f.write_bytes(k.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()))
    try:
        os.chmod(priv_f, 0o600)
    except OSError:
        pass
    pub = k.public_key().public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)
    (d / "licence_public.pem").write_bytes(pub)
    if write_public:
        out = Path(__file__).parent / "data" / "licence_public.pem"
        out.parent.mkdir(exist_ok=True)
        out.write_bytes(pub)
    return {"private_key": str(priv_f), "public_key": str(d / "licence_public.pem"), "note": "Keep the private key offline and out of every repository. Anyone holding it can issue licences."}


def issue(name: str, email: str, days: int = DEFAULTS["days"], runs: int = DEFAULTS["runs"], ai: int = DEFAULTS["ai"], activate_by: Optional[str] = None,
          dirpath: Optional[str] = None, private_pem: Optional[bytes] = None, note: str = "Evaluation licence") -> str:
    from cryptography.hazmat.primitives import serialization
    pem = private_pem or (Path(dirpath or (Path.home() / ".climrisk-signing")) / "licence_private.pem").read_bytes()
    key = serialization.load_pem_private_key(pem, password=None)
    payload = {"v": 1, "id": uuid.uuid4().hex[:12], "licensee": name, "email": email, "issued": _today().isoformat(), "days": int(days),
               "activate_by": activate_by or (_today() + timedelta(days=30)).isoformat(), "limits": {"runs": int(runs), "ai": int(ai)}, "note": note}
    body = _b64(json.dumps(payload, separators=(",", ":"), sort_keys=True).encode())
    return f"{PREFIX}.{body}.{_b64(key.sign(body.encode()))}"


# ── client side ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────

def verify(token: str) -> dict:
    """-> payload, or raises LicenceError('invalid')."""
    from cryptography.exceptions import InvalidSignature
    try:
        pre, body, sig = token.strip().split(".")
        if pre != PREFIX:
            raise ValueError
        _public_key().verify(_unb64(sig), body.encode())
        return json.loads(_unb64(body))
    except LicenceError:
        raise
    except (ValueError, InvalidSignature, Exception):                # noqa: BLE001
        raise LicenceError("invalid", "this licence key is not valid; check that it was copied completely")


def _mac_key(lic_id: str) -> bytes:
    return hashlib.sha256(f"{lic_id}|{uuid.getnode()}|{platform.system()}".encode()).digest()


def _sign_state(st: dict) -> str:
    body = json.dumps({k: v for k, v in st.items() if k != "mac"}, sort_keys=True, separators=(",", ":")).encode()
    return hmac.new(_mac_key(st.get("id", "")), body, hashlib.sha256).hexdigest()


def _paths() -> tuple:
    return home() / "licence.key", home() / "licence_state.json"


def _load_state(lic_id: str) -> dict:
    _, sf = _paths()
    if not sf.exists():
        return {}
    try:
        st = json.loads(sf.read_text())
    except ValueError:
        raise LicenceError("tampered", "the licence usage record is unreadable; contact ClimRisk for a new key")
    if st.get("id") != lic_id:
        return {}
    if not hmac.compare_digest(st.get("mac", ""), _sign_state(st)):
        raise LicenceError("tampered", "the licence usage record was modified; contact ClimRisk for a new key")
    return st


def _save_state(st: dict) -> None:
    _, sf = _paths()
    sf.parent.mkdir(parents=True, exist_ok=True)
    st = {k: v for k, v in st.items() if k != "mac"}
    st["mac"] = _sign_state(st)
    tmp = sf.with_suffix(".tmp")
    tmp.write_text(json.dumps(st))
    os.replace(tmp, sf)


def install(token: str) -> dict:
    p = verify(token)
    if date.fromisoformat(p["activate_by"]) < _today():
        raise LicenceError("expired", f"this key had to be activated by {p['activate_by']}; ask ClimRisk for a new one")
    kf, _ = _paths()
    kf.parent.mkdir(parents=True, exist_ok=True)
    kf.write_text(token.strip())
    st = _load_state(p["id"])
    if not st:
        _save_state({"id": p["id"], "activated": _today().isoformat(), "last_seen": _today().isoformat(), "used": {"runs": 0, "ai": 0}})
    return status()


def status(count: Optional[str] = None) -> dict:
    """The licence state; with count='runs'|'ai' it also spends one unit (and raises LicenceError if not allowed)."""
    kf, _ = _paths()
    if not kf.exists():
        return {"state": "none", "message": "No licence is installed. Enter the key you were sent.", "allowed": False}
    try:
        p = verify(kf.read_text())
        st = _load_state(p["id"])
        if not st:
            return {"state": "inactive", "message": "Licence found but not activated.", "allowed": False, "licensee": p["licensee"]}
        today, last = _today(), date.fromisoformat(st["last_seen"])
        if today < last - timedelta(days=1):
            raise LicenceError("clock", "the computer's date is earlier than the last use of this licence; correct the clock")
        end = date.fromisoformat(st["activated"]) + timedelta(days=p["days"])
        left = (end - today).days
        lim = p["limits"]
        out = {"state": "active", "allowed": True, "licensee": p["licensee"], "days_left": max(left, 0), "ends": end.isoformat(),
               "limits": lim, "used": dict(st["used"]), "remaining": {k: max(lim[k] - st["used"].get(k, 0), 0) for k in lim}}
        if left < 0:
            out.update(state="expired", allowed=False, message=f"The evaluation period ended on {end.isoformat()}. Saved results can still be viewed; new analyses need a new licence.")
            return out
        if count:
            if st["used"].get(count, 0) >= lim.get(count, 0):
                out.update(state="exhausted", allowed=False, message=f"The evaluation allowance for {'analysis runs' if count == 'runs' else 'AI questions'} ({lim.get(count, 0)}) is used up.")
                return out
            st["used"][count] = st["used"].get(count, 0) + 1
            out["used"], out["remaining"][count] = dict(st["used"]), out["remaining"][count] - 1
        if today > last:
            st["last_seen"] = today.isoformat()
        if count or today > last:
            _save_state(st)
        return out
    except LicenceError as e:
        return {"state": e.state, "allowed": False, "message": e.message}


def classify(method: str, path: str) -> Optional[str]:
    """-> None (free), 'runs', 'ai', or 'block' (not metered, but blocked when the licence is not active)."""
    if path.startswith(EXEMPT_PREFIXES):
        return None
    if method in ("GET", "HEAD", "OPTIONS"):
        return None
    if path.startswith(AI_PREFIXES):
        return "ai"
    if path.startswith(RUN_PREFIXES):
        return "runs"
    return "block"


def enforcement_required() -> bool:
    return os.environ.get("CRI_LICENCE_MODE", "").lower() == "required"


def gate(method: str, path: str) -> Optional[dict]:
    """None = allowed; otherwise the status dict to return with HTTP 402."""
    if not enforcement_required():
        return None
    kind = classify(method, path)
    if kind is None:
        return None
    s = status(None if kind == "block" else kind)
    return None if s.get("allowed") else s


# ── CLI ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

def _arg(a, flag, default=None):
    return a[a.index(flag) + 1] if flag in a else default


def main() -> None:
    a = sys.argv[1:]
    if not a:
        print(__doc__)
        return
    c = a[0]
    if c == "keygen":
        print(json.dumps(keygen(_arg(a, "--dir")), indent=1))
    elif c == "issue":
        t = issue(_arg(a, "--name"), _arg(a, "--email"), int(_arg(a, "--days", DEFAULTS["days"])), int(_arg(a, "--runs", DEFAULTS["runs"])), int(_arg(a, "--ai", DEFAULTS["ai"])), _arg(a, "--activate-by"), _arg(a, "--dir"))
        print(t)
    elif c == "status":
        print(json.dumps(status(), indent=1))
    elif c == "install":
        print(json.dumps(install(a[1]), indent=1))
    elif c == "install-file":
        print(json.dumps(install(Path(a[1]).read_text()), indent=1))


if __name__ == "__main__":
    main()
