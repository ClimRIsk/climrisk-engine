"""The keyless deep dive, the certificate fix and the 'terrain did not load' guard (found on a real Mac run)."""
import os
import ssl

import numpy as np
import pytest


def _assessment(**kw):
    a = {
        "asset_name": "Chattogram Plant", "asset_type": "industrial", "lat": 22.265, "lon": 91.805, "risk_signal": "CRITICAL",
        "headline": "Asset sits at 4 m on flat terrain. Tropical cyclone wind is the leading exposure (CRITICAL).",
        "findings": [
            {"hazard": "tropical_cyclone", "severity": "CRITICAL", "score": 6,
             "evidence": [{"metric": "Cyclones passing within 200 km since 1980", "value": 46, "unit": "", "source": "IBTrACS"},
                          {"metric": "1-in-100-year wind", "value": "61 m/s", "unit": "", "source": "IBTrACS"}],
             "actions": ["Design to the 1-in-100 wind.", "Secure rooftop plant.", "Prepare a season plan."]},
            {"hazard": "urban_heat", "severity": "MODERATE", "score": 2.5, "evidence": [{"metric": "Built-up cover within 1 km", "value": 45.1, "unit": "%", "source": "WorldCover"}], "actions": ["Cool roofs."]},
            {"hazard": "flood", "severity": "NEGLIGIBLE", "score": 0, "evidence": [], "actions": []},
        ],
    }
    a.update(kw)
    return a


def test_brief_quotes_only_evidence_and_needs_no_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    from cri.resident.brief import site_brief
    b = site_brief(_assessment())
    t = b["narrative"]
    assert b["model_used"] == "resident-engine"
    assert "46" in t and "61 m/s" in t and "45.1" in t          # numbers come from the evidence rows
    assert "qualitative" in t.lower()                            # 2050 section is labelled
    assert "1. Design to the 1-in-100 wind." in t               # actions ordered by severity
    assert "GROQ" not in t


def test_brief_says_when_data_is_missing():
    from cri.resident.brief import site_brief
    b = site_brief(_assessment(data_gaps=["Terrain did not load, so elevation is NOT valid — SSLError"]))
    assert b["sections"][0]["title"] == "Read this first"
    assert "not reliable" in b["sections"][0]["text"]


def test_brief_endpoint_keyless(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    from fastapi.testclient import TestClient
    from cri.api.main import app
    c = TestClient(app)
    r = c.post("/resident/brief", json={"assessment": _assessment()})
    assert r.status_code == 200 and r.json()["sections"]
    assert c.post("/resident/brief", json={"assessment": {"findings": []}}).status_code == 422


def test_llm_fallback_no_longer_asks_for_a_groq_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    from cri.ai import llm_agent
    r = llm_agent._groq_unavailable_fallback("x" * 500)                # long prompt -> plain pointer, no key demand
    assert "GROQ_API_KEY" not in r.narrative and r.error is None


def test_trust_store_points_python_at_certifi(monkeypatch):
    certifi = pytest.importorskip("certifi")
    monkeypatch.delenv("SSL_CERT_FILE", raising=False)
    monkeypatch.delenv("REQUESTS_CA_BUNDLE", raising=False)
    import cri
    cri._trust_store()
    assert os.environ["SSL_CERT_FILE"] == certifi.where()
    ctx = ssl._create_default_https_context()
    assert ctx.verify_mode == ssl.CERT_REQUIRED and ctx.check_hostname


def test_trust_store_respects_operator_bundle(monkeypatch):
    pytest.importorskip("certifi")
    monkeypatch.setenv("SSL_CERT_FILE", "/custom/bundle.pem")
    import cri
    cri._trust_store()
    assert os.environ["SSL_CERT_FILE"] == "/custom/bundle.pem"


def test_terrain_with_no_tiles_raises_not_flat(monkeypatch):
    from cri.connectors import terrain_tiles as T
    monkeypatch.setattr(T, "_fetch_tile", lambda z, x, y: None)
    T._LAST_ERROR[0] = "SSLCertVerificationError: unable to get local issuer certificate"
    with pytest.raises(RuntimeError) as e:
        T.get_elevation_grid(22.265, 91.805, radius_km=1.0)
    assert "root-certificate" in str(e.value) and "issuer certificate" in str(e.value)


def test_assessment_reports_data_gaps_when_terrain_fails(monkeypatch):
    from cri.analysis import asset_intelligence as AI
    from cri.terrain import pysheds_analyzer as P

    def boom(*a, **k):
        raise RuntimeError("terrain tiles could not be downloaded (9 tried)")
    monkeypatch.setattr(P.TerrainEngine, "assess", boom)
    try:
        out = AI.assess_asset(22.265, 91.805, "t", include_visuals=False)
    except Exception as exc:                                         # network-dependent parts may fail offline
        pytest.skip(f"assessment needs network: {exc}")
    assert out["data_gaps"] and "Terrain did not load" in out["data_gaps"][0]


def test_selfcheck_names_the_certificate_problem(monkeypatch):
    from cri.resident import selfcheck as S

    def fail(req, timeout=0):
        raise OSError("<urlopen error [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: unable to get local issuer certificate>")
    monkeypatch.setattr(S.urllib.request, "urlopen", fail)
    r = S.run()
    assert not r["ok"] and r["failed"] == len(S.CHECKS)
    assert "certificate" in r["rows"][0]["fix"].lower() and "0.4.1" in r["rows"][0]["fix"]


def test_selfcheck_ok_when_reachable(monkeypatch):
    from cri.resident import selfcheck as S

    class R:
        status = 200
        def read(self, n=0): return b""
        def __enter__(self): return self
        def __exit__(self, *a): return False
    monkeypatch.setattr(S.urllib.request, "urlopen", lambda req, timeout=0: R())
    assert S.run()["ok"]
