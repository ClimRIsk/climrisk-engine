"""Private maintainer push: PIN, token, secret scan, loopback, allow-list, and exclusion from the commercial wheel."""
import subprocess
from pathlib import Path

import pytest

from cri.maintainer import push, ext


def _git(r, *a):
    subprocess.run(["git", "-C", str(r), *a], check=True, capture_output=True)


@pytest.fixture()
def env(tmp_path, monkeypatch):
    bare = tmp_path / "remote.git"; subprocess.run(["git", "init", "-q", "--bare", str(bare)], check=True)
    work = tmp_path / "work"; subprocess.run(["git", "clone", "-q", str(bare), str(work)], check=True, capture_output=True)
    _git(work, "config", "user.email", "t@t"); _git(work, "config", "user.name", "t"); _git(work, "checkout", "-q", "-b", "main")
    (work / "a.txt").write_text("a\n"); _git(work, "add", "."); _git(work, "commit", "-qm", "first"); _git(work, "push", "-q", "-u", "origin", "main")
    monkeypatch.setattr(push, "HOME", tmp_path / "home"); monkeypatch.setattr(push, "CONFIG", tmp_path / "home" / "maintainer.json"); monkeypatch.setattr(push, "AUDIT", tmp_path / "home" / "audit.jsonl")
    push._TOKENS.clear(); push._FAILS.update(n=0, until=0.0)
    push.save_config("correct-horse", [str(work)])
    return work, bare


def test_pin_hash_roundtrip():
    h = push.hash_pin("abc12345")
    assert push.verify_pin("abc12345", h) and not push.verify_pin("abc12346", h) and "abc12345" not in h


def test_nothing_ahead_means_no_token(env):
    p = push.propose()
    assert p["ok"] and p["token"] is None


def test_push_needs_pin_and_works_once(env):
    work, bare = env
    (work / "b.txt").write_text("b\n"); _git(work, "add", "."); _git(work, "commit", "-qm", "second")
    p = push.propose(); assert p["token"] and p["plans"][0]["ahead"] == 1
    assert push.execute(p["token"], "wrong-pin")["error"] == "wrong PIN"
    assert push.execute(p["token"], "correct-horse", client_ok=False)["ok"] is False          # not loopback
    r = push.execute(p["token"], "correct-horse")
    assert r["ok"], r
    again = push.execute(p["token"], "correct-horse"); assert not again["ok"]                 # token is single-use
    out = subprocess.run(["git", "-C", str(bare), "log", "--oneline", "main"], capture_output=True, text=True).stdout
    assert "second" in out


def test_lockout_after_three_bad_pins(env):
    for _ in range(3):
        push.execute("x", "nope")
    assert "locked" in push.execute("x", "correct-horse")["error"]


def test_secret_scan_blocks(env):
    work, _ = env
    (work / "c.py").write_text('TOKEN = "ghp_' + "a" * 36 + '"\n'); _git(work, "add", "."); _git(work, "commit", "-qm", "oops")
    p = push.propose()
    assert p["blocked"] and p["token"] is None and p["plans"][0]["secrets"]
    (work / ".env").write_text("x=1\n")


def test_head_change_after_review_invalidates(env):
    work, _ = env
    (work / "d.txt").write_text("d\n"); _git(work, "add", "."); _git(work, "commit", "-qm", "d")
    p = push.propose()
    (work / "e.txt").write_text("e\n"); _git(work, "add", "."); _git(work, "commit", "-qm", "e")
    r = push.execute(p["token"], "correct-horse")
    assert not r["ok"] and "changed" in r["results"][0]["error"]


def test_wrong_branch_refused(env):
    work, _ = env
    _git(work, "checkout", "-q", "-b", "feature")
    assert push.propose()["plans"][0]["ok"] is False


def test_chat_command_only_matches_push_requests(env):
    assert ext.handle("s", "what is the flood risk in Dhaka") is None
    r = ext.handle("s", "push the changes to github")
    assert r and "nothing is pushed yet" in r["reply"].lower()


def test_wheel_excludes_maintainer_package():
    t = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text()
    assert 'exclude = ["cri.maintainer", "cri.maintainer.*"]' in t.split("[tool.setuptools.packages.find]")[1].split("[tool.setuptools.package-data]")[0]
