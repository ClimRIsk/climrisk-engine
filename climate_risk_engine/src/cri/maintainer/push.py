"""Authenticated, two-step git push for the repositories you list in ~/.climrisk/maintainer.json.

Safety design (the agent never holds a GitHub token: git uses your own credential helper / keychain):
  1. propose(): shows, per repo, the branch, the commits that would go out, the files changed and a secret scan of the added lines. Nothing is pushed. Returns a one-time token bound to the exact HEAD commits.
  2. execute(token, pin): needs your PIN (stored only as a salted PBKDF2 hash), a valid unexpired token, unchanged HEADs and a clean secret scan; then runs a plain `git push origin <branch>` (never --force, only allow-listed branches).
  Wrong PINs lock the tool for 15 minutes after 3 tries. Every proposal and push is appended to ~/.climrisk/maintainer-audit.jsonl.
  Only accepts requests from this computer (loopback), so a deployed copy of the engine can never push.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import secrets
import subprocess
import time
from pathlib import Path
from typing import Optional

HOME = Path(os.environ.get("CRI_MAINTAINER_HOME", str(Path.home() / ".climrisk")))
CONFIG = HOME / "maintainer.json"
AUDIT = HOME / "maintainer-audit.jsonl"
TOKEN_TTL = 600.0
MAX_TRIES, LOCK_SECONDS = 3, 900.0
ALLOWED_BRANCHES = {"main", "master"}
_TOKENS: dict = {}
_FAILS: dict = {"n": 0, "until": 0.0}

SECRET_RX = [r"AKIA[0-9A-Z]{16}", r"ghp_[A-Za-z0-9]{30,}", r"github_pat_[A-Za-z0-9_]{30,}", r"xox[baprs]-[A-Za-z0-9-]{10,}", r"-----BEGIN [A-Z ]*PRIVATE KEY-----", r"sk-[A-Za-z0-9]{32,}", r"gsk_[A-Za-z0-9]{30,}",
             r"(?i)(?:api[_-]?key|secret|token|password|access_token)\s*[=:]\s*['\"][A-Za-z0-9_\-\.]{16,}['\"]", r"LINKEDIN_ACCESS_TOKEN\s*=\s*\S{10,}"]
BAD_FILES = re.compile(r"(^|/)(\.env(\..*)?|.*\.pem|id_rsa|id_ed25519|maintainer\.json|.*\.p12)$")


# ── configuration / authentication ──────────────────────────────────────────────────────────────────────────────────────────
def hash_pin(pin: str, salt: Optional[bytes] = None) -> str:
    salt = salt or secrets.token_bytes(16)
    return salt.hex() + "$" + hashlib.pbkdf2_hmac("sha256", pin.encode(), salt, 200_000).hex()


def verify_pin(pin: str, stored: str) -> bool:
    try:
        s, h = stored.split("$")
        return hmac.compare_digest(hashlib.pbkdf2_hmac("sha256", pin.encode(), bytes.fromhex(s), 200_000).hex(), h)
    except Exception:                                              # noqa: BLE001
        return False


def load_config() -> Optional[dict]:
    try:
        c = json.loads(CONFIG.read_text())
        return c if c.get("pin_hash") and c.get("repos") else None
    except Exception:                                              # noqa: BLE001
        return None


def save_config(pin: str, repos: list) -> None:
    HOME.mkdir(parents=True, exist_ok=True)
    CONFIG.write_text(json.dumps({"pin_hash": hash_pin(pin), "repos": [str(Path(r).expanduser()) for r in repos]}, indent=1))
    try:
        os.chmod(CONFIG, 0o600)
    except OSError:
        pass


def _audit(event: str, **kw) -> None:
    try:
        HOME.mkdir(parents=True, exist_ok=True)
        with AUDIT.open("a") as f:
            f.write(json.dumps({"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "event": event, **kw}) + "\n")
    except Exception:                                              # noqa: BLE001
        pass


def status() -> dict:
    c = load_config()
    return {"enabled": bool(c), "repos": (c or {}).get("repos", []), "locked": _FAILS["until"] > time.time(),
            "how_to": None if c else "Run: python -m cri.maintainer.setup   (sets your PIN and the repositories this tool may push)"}


# ── git helpers ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
def _git(repo: str, *args, timeout: int = 60) -> tuple:
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}
    p = subprocess.run(["git", "-C", repo, *args], capture_output=True, text=True, timeout=timeout, env=env)
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def scan_diff(repo: str, base: str) -> list:
    """Secret-like content in the lines this push would add, and forbidden file names."""
    hits = []
    rc, names = _git(repo, "diff", "--name-only", f"{base}..HEAD")
    for n in names.splitlines():
        if BAD_FILES.search(n.strip()):
            hits.append(f"file {n.strip()} must not be pushed")
    rc, d = _git(repo, "diff", "-U0", f"{base}..HEAD", timeout=120)
    for ln in d.splitlines():
        if ln.startswith("+") and not ln.startswith("+++"):
            for rx in SECRET_RX:
                if re.search(rx, ln):
                    hits.append("secret-like text: " + re.sub(r"[A-Za-z0-9_\-\.]{12,}", "…", ln[:90]))
                    break
    return hits[:10]


def _repo_plan(repo: str) -> dict:
    out = {"repo": repo, "name": Path(repo).name}
    if not (Path(repo) / ".git").exists():
        return {**out, "ok": False, "why": "not a git repository"}
    rc, br = _git(repo, "branch", "--show-current"); br = br.strip()
    if br not in ALLOWED_BRANCHES:
        return {**out, "ok": False, "why": f"branch '{br}' is not allowed (only {sorted(ALLOWED_BRANCHES)})"}
    rc, up = _git(repo, "rev-parse", "--abbrev-ref", f"{br}@{{u}}")
    base = up.strip() if rc == 0 else f"origin/{br}"
    rc, cnt = _git(repo, "rev-list", "--count", f"{base}..HEAD")
    if rc != 0:
        return {**out, "ok": False, "why": "no remote tracking branch to compare with"}
    n = int(cnt.strip() or 0)
    rc, head = _git(repo, "rev-parse", "HEAD")
    rc, log = _git(repo, "log", "--oneline", f"{base}..HEAD")
    rc, stat = _git(repo, "diff", "--shortstat", f"{base}..HEAD")
    rc, dirty = _git(repo, "status", "--porcelain")
    plan = {**out, "ok": True, "branch": br, "base": base, "ahead": n, "head": head.strip(), "commits": log.strip().splitlines()[:15], "summary": stat.strip(),
            "uncommitted": len([x for x in dirty.splitlines() if x.strip()]), "secrets": scan_diff(repo, base) if n else []}
    return plan


# ── the two steps ───────────────────────────────────────────────────────────────────────────────────────────────────────────
def propose() -> dict:
    c = load_config()
    if not c:
        return {"ok": False, "error": "maintainer tools are not set up", "status": status()}
    plans = [_repo_plan(r) for r in c["repos"]]
    todo = [p for p in plans if p.get("ok") and p["ahead"] > 0]
    blocked = [p for p in todo if p["secrets"]]
    token = None
    if todo and not blocked:
        token = secrets.token_urlsafe(16)
        _TOKENS[token] = {"exp": time.time() + TOKEN_TTL, "heads": {p["repo"]: p["head"] for p in todo}, "branches": {p["repo"]: p["branch"] for p in todo}}
    _audit("propose", repos=[(p["name"], p.get("ahead"), bool(p.get("secrets"))) for p in plans])
    return {"ok": True, "plans": plans, "token": token, "expires_in_s": int(TOKEN_TTL) if token else None, "blocked": bool(blocked),
            "note": "Nothing has been pushed." + (" A secret scan blocked the push; remove the flagged content first." if blocked else "" if todo else " Nothing to push: every listed repo is up to date.")}


def execute(token: str, pin: str, client_ok: bool = True) -> dict:
    if not client_ok:
        return {"ok": False, "error": "only this computer may push"}
    c = load_config()
    if not c:
        return {"ok": False, "error": "maintainer tools are not set up"}
    if _FAILS["until"] > time.time():
        return {"ok": False, "error": "locked after wrong PIN attempts; try again later"}
    if not verify_pin(pin or "", c["pin_hash"]):
        _FAILS["n"] += 1
        if _FAILS["n"] >= MAX_TRIES:
            _FAILS.update(n=0, until=time.time() + LOCK_SECONDS)
        _audit("bad_pin"); return {"ok": False, "error": "wrong PIN"}
    _FAILS["n"] = 0
    t = _TOKENS.pop(token or "", None)
    if not t or t["exp"] < time.time():
        return {"ok": False, "error": "proposal expired or unknown: ask for a new proposal and review it"}
    results = []
    for repo, head in t["heads"].items():
        if repo not in c["repos"]:
            results.append({"repo": repo, "ok": False, "error": "not an allowed repository"}); continue
        plan = _repo_plan(repo)
        if not plan.get("ok") or plan["head"] != head or plan["secrets"]:
            results.append({"repo": repo, "ok": False, "error": "the repository changed since you reviewed it, or the secret scan failed; request a new proposal"}); continue
        rc, out = _git(repo, "push", "origin", t["branches"][repo], timeout=180)
        results.append({"repo": repo, "ok": rc == 0, "output": out.strip()[-500:]})
        _audit("push", repo=Path(repo).name, ok=rc == 0, head=head[:8])
    return {"ok": all(r["ok"] for r in results) and bool(results), "results": results}
