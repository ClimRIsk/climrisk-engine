"""A question about a place is answered from that place's own hazard numbers, not from unrelated literature."""
import pytest

from cri.resident import agent as A


@pytest.fixture()
def ag():
    return A.get_agent() if hasattr(A, "get_agent") else A.Agent()


def test_site_question_with_coordinates_routes_to_the_site_profile(ag, monkeypatch):
    monkeypatch.setattr(A, "_job", lambda fn, label: {"id": "job-test"})          # do not start the real profile run
    r = ag._profile_commands("s-test", "What is the heat stress risk at 29.7255, -95.0 and what data is missing?")
    assert r and r["intent"] == "site_profile" and r["job"] == "job-test"
    assert "null" in r["reply"].lower() and "never as zero" in r["reply"].lower()


def test_general_questions_and_tool_requests_are_not_hijacked(ag):
    assert ag._profile_commands("s-test", "What is a storm surge?") is None
    assert ag._profile_commands("s-test", "build a 3D twin at 29.7255, -95.0") is None
