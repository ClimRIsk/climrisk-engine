import json
import os
from datetime import timedelta

import pytest

from cri import licence as L


@pytest.fixture()
def lic(tmp_path, monkeypatch):
    monkeypatch.setenv("CRI_HOME", str(tmp_path / "home"))
    k = L.keygen(str(tmp_path / "keys"), write_public=False)
    monkeypatch.setenv("CRI_LICENCE_PUBKEY", k["public_key"])
    monkeypatch.setenv("CRI_LICENCE_MODE", "required")
    pem = open(k["private_key"], "rb").read()
    return lambda **kw: L.issue("Charles", "c@example.com", private_pem=pem, **kw)


def test_install_counts_and_exhausts(lic):
    assert L.status()["state"] == "none"
    s = L.install(lic(days=14, runs=2, ai=1))
    assert s["state"] == "active" and s["remaining"] == {"runs": 2, "ai": 1} and s["days_left"] == 14
    assert L.status("runs")["remaining"]["runs"] == 1
    assert L.status("runs")["remaining"]["runs"] == 0
    e = L.status("runs")
    assert e["state"] == "exhausted" and not e["allowed"] and "used up" in e["message"]
    assert L.status("ai")["allowed"] and not L.status("ai")["allowed"]


def test_forged_and_edited_keys_are_refused(lic):
    t = lic()
    pre, body, sig = t.split(".")
    forged = L._b64(json.dumps({**json.loads(L._unb64(body)), "days": 9999}).encode())
    with pytest.raises(L.LicenceError):
        L.install(f"{pre}.{forged}.{sig}")
    with pytest.raises(L.LicenceError):
        L.install("not a key")
    with pytest.raises(L.LicenceError):                                  # a key signed by someone else
        other = L.keygen(os.path.join(os.environ["CRI_HOME"], "other"), write_public=False)
        L.install(L.issue("x", "y", private_pem=open(other["private_key"], "rb").read()))


def test_expiry_clock_rollback_and_state_tampering(lic, monkeypatch):
    L.install(lic(days=3))
    real = L._today()
    monkeypatch.setattr(L, "_today", lambda: real + timedelta(days=2))
    assert L.status()["allowed"] and L.status()["days_left"] == 1
    monkeypatch.setattr(L, "_today", lambda: real + timedelta(days=4))
    assert L.status()["state"] == "expired"
    monkeypatch.setattr(L, "_today", lambda: real)                       # clock wound back after use
    assert L.status()["state"] == "clock"
    monkeypatch.setattr(L, "_today", lambda: real + timedelta(days=4))
    sf = L._paths()[1]
    st = json.loads(sf.read_text()); st["used"]["runs"] = -500; sf.write_text(json.dumps(st))
    assert L.status()["state"] == "tampered"


def test_late_activation_refused(lic):
    with pytest.raises(L.LicenceError) as e:
        L.install(lic(activate_by="2020-01-01"))
    assert e.value.state == "expired"


def test_classify_and_gate(lic, monkeypatch):
    assert L.classify("GET", "/insurance/analyse") is None
    assert L.classify("POST", "/insurance/analyse") == "runs" and L.classify("POST", "/resident/reason") == "ai" and L.classify("POST", "/licence/install") is None
    assert L.classify("POST", "/something/else") == "block"
    assert L.gate("POST", "/insurance/analyse")["state"] == "none"       # nothing installed: analysis refused
    assert L.gate("GET", "/health") is None
    L.install(lic(runs=1))
    assert L.gate("POST", "/insurance/analyse") is None
    assert L.gate("POST", "/insurance/analyse")["state"] == "exhausted"
    assert L.gate("POST", "/something/else") is None                     # free POST while active
    monkeypatch.delenv("CRI_LICENCE_MODE")
    assert L.gate("POST", "/insurance/analyse") is None                  # development mode: not enforced


def test_http_gate(lic):
    from fastapi.testclient import TestClient
    from cri.api.main import app
    c = TestClient(app)
    r = c.post("/portfolio/run", json={})
    assert r.status_code == 402 and r.json()["state"] == "none"
    assert c.get("/licence/status").json()["state"] == "none"
    k = lic(runs=1)
    assert c.post("/licence/install", json={"key": k}).json()["state"] == "active"
    assert c.post("/licence/install", json={"key": "junk"}).status_code == 400
    assert c.get("/health").status_code == 200
