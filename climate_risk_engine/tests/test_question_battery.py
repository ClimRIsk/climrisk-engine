"""Offline question batteries for the shipping and building handlers. A phrasing that stops parsing correctly fails here."""
import battery_cases as B
import pytest

from cri.maritime import answer as A
from cri.resident import feasibility as FE

MARKET = ("bunker_usd_per_t", "charter_usd_per_day", "eua_price")


def _same(a, b):
    return abs(float(a) - float(b)) < 1e-6 if isinstance(b, (int, float)) and isinstance(a, (int, float)) else str(a).lower() == str(b).lower()


def _ship_errs(text, exp):
    errs = [] if A.matches(text) else ["not routed"]
    p = A.parse(text)
    for k, v in exp.items():
        got = (p.get("market") or {}).get(k) if k in MARKET else p.get(k)
        if got is None or not _same(got, v):
            errs.append(f"{k}: want {v!r} got {got!r}")
    return errs


def _build_errs(text, exp):
    errs = [] if FE.matches(text) else ["not routed"]
    p = FE.parse(text)
    for k, v in exp.items():
        if p.get(k) is None or not _same(p[k], v):
            errs.append(f"{k}: want {v!r} got {p.get(k)!r}")
    return errs


@pytest.mark.parametrize("text,exp", B.SHIP + B.SHIP_HELDOUT, ids=lambda x: x[:50] if isinstance(x, str) else "")
def test_ship_question(text, exp):
    assert _ship_errs(text, exp) == []


@pytest.mark.parametrize("text", B.SHIP_NEGATIVE)
def test_ship_negatives_not_claimed(text):
    assert not A.matches(text)


@pytest.mark.parametrize("text,exp", B.BUILD + B.BUILD_HELDOUT, ids=lambda x: x[:50] if isinstance(x, str) else "")
def test_build_question(text, exp):
    assert _build_errs(text, exp) == []


@pytest.mark.parametrize("text", B.BUILD_NEGATIVE + B.BUILD_HELDOUT_NEGATIVE)
def test_build_negatives_not_claimed(text):
    assert not FE.matches(text)


def test_building_area_is_not_read_as_coordinates():
    from cri.resident import nlu as N
    for t in ("a 5,000 square metre warehouse in Rotterdam", "a factory of 20,000 m2 in Chennai", "a 30,000 sq ft factory near Houston"):
        assert "lat" not in N.entities(t), t
    assert N.entities("build at 25.77, -80.19")["lat"] == 25.77


def test_ship_and_building_never_claim_each_other():
    for text, _ in B.SHIP:
        assert not FE.matches(text), text
    for text, _ in B.BUILD:
        assert not A.matches(text), text


@pytest.mark.parametrize("text", B.SHIP_NEEDS_ROUTE)
def test_ship_question_without_route_asks_for_it(text):
    assert A.needs_route(text) and not A.matches(text)


@pytest.mark.parametrize("text,_", B.BUILD_NO_SPEC)
def test_building_question_without_size_still_routes(text, _):
    assert FE.matches(text)


@pytest.mark.parametrize("text", B.BUILD_NOT_BUILDING)
def test_software_and_team_building_not_routed(text):
    assert not FE.matches(text)


def test_followups_merge_into_the_last_voyage():
    last = {"origin": "Rotterdam", "dest": "Shanghai", "distance_nm": 16000, "speed_kn": 14, "name": "Ever Given", "market": {"bunker_usd_per_t": 600}, "flags": []}
    for text, key, val in (("what if she slows to 12 knots", "speed_option_kn", 12), ("and with a 5 day delay", "delay_days", 5), ("same ship to Singapore", "dest", "Singapore")):
        assert A.is_followup(text)
        m = A.merge(last, A.parse(text))
        assert m[key] == val and m["origin"] == "Rotterdam" and m["name"] == "Ever Given"
    m = A.merge(last, A.parse("bunker at $700/t instead"))
    assert m["market"]["bunker_usd_per_t"] == 700 and m["speed_kn"] == 14
    assert not A.is_followup("tell me a joke") and not A.is_followup("what if I build a 10 storey building")
