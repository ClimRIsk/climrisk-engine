import os, tempfile, time
import pytest
from cri.resident import agent as AG, answer as AN, converse as CV, memory as M


@pytest.fixture(autouse=True)
def _home(monkeypatch, tmp_path):
    monkeypatch.setenv("CRI_HOME", str(tmp_path))
    monkeypatch.setenv("CRI_DATA_DIR", str(tmp_path))          # the resident database must never be the real one
    yield


def _agent():
    return AG.Agent(twin_builder=lambda **k: {}, analyser=lambda t: {}, scanner=lambda **k: {}, lab=lambda *a: {})


def test_no_model_never_asks_for_a_key(monkeypatch):
    monkeypatch.setattr(AN, "llm_status", lambda force=False: {"available": False, "model": None, "why": "Ollama is not running on this computer"})
    out = _agent().handle("c1", "hello there")
    assert out["intent"] == "smalltalk"
    low = out["reply"].lower()
    assert "no key or account" in low and "api key" not in low and "ollama" in low


def test_model_conversation_uses_context_and_history(monkeypatch):
    seen = {}
    monkeypatch.setattr(AN, "llm_status", lambda force=False: {"available": True, "model": "climrisk-llm", "why": "ok"})
    def fake_post(path, payload, total_s):
        import json
        seen["msgs"] = json.loads(payload)["messages"]
        return {"message": {"content": "Air at this place is MODERATE: the engine reports PM2.5 19.3 µg/m3. A monitor would settle the local part."}}
    monkeypatch.setattr(AN, "_ollama_post", fake_post)
    M.set_state("last_place", {"lat": 9.4713, "lon": 77.9107, "name": "Home"})
    M.set_state("last_result", {"summary": "site hazard profile", "explain": "Air quality MODERATE 19.3 µg/m3"})
    ag = _agent()
    out = ag.handle("c2", "hi")                                   # smalltalk goes through the model
    assert out["mode"].startswith("local language model")
    out = ag.handle("c2", "why is the air a concern for us here?")
    assert out["reply"].startswith("Air at this place") and "unsupported_numbers" in out and out["unsupported_numbers"] == []
    last = seen["msgs"][-1]["content"]
    assert "CONTEXT" in last and "Home" in last and "19.3" in last
    assert any(m["role"] == "assistant" for m in seen["msgs"])      # the earlier turn is carried


def test_invented_numbers_are_flagged(monkeypatch):
    monkeypatch.setattr(AN, "llm_status", lambda force=False: {"available": True, "model": "m", "why": "ok"})
    monkeypatch.setattr(AN, "_ollama_post", lambda *a, **k: {"message": {"content": "Your site gets 847 mm of rain a year and PM2.5 of 19.3."}})
    M.set_state("last_result", {"summary": "air", "explain": "PM2.5 19.3"})
    M.set_state("last_place", {"lat": 1.0, "lon": 2.0, "name": "X"})
    out = _agent().handle("c3", "how worried should we be about this?")
    assert "847" in out["unsupported_numbers"] and "did not come from the engine" in out["reply"]


def test_tool_requests_still_route_to_tools(monkeypatch):
    monkeypatch.setattr(AN, "llm_status", lambda force=False: {"available": True, "model": "m", "why": "ok"})
    called = []
    monkeypatch.setattr(AN, "_ollama_post", lambda *a, **k: called.append(1) or {"message": {"content": "x"}})
    out = _agent().handle("c4", "all hazards at 22.26, 91.80")
    assert out["intent"] == "site_profile" and not called


def test_model_error_is_reported_not_hidden(monkeypatch):
    monkeypatch.setattr(AN, "llm_status", lambda force=False: {"available": True, "model": "m", "why": "ok"})
    def boom(*a, **k):
        raise TimeoutError("slow")
    monkeypatch.setattr(AN, "_ollama_post", boom)
    out = _agent().handle("c5", "hello")
    assert "did not answer" in out["reply"] and "key" not in out["reply"].lower()
