"""The shipped intent model (src/cri/data/resident_nlu.json) against two independent utterance sets, plus the corpus rules."""
import json
import re

import pytest

import nlu_battery as A
import nlu_battery_b as B
import nlu_battery_c as BC
from pathlib import Path
from cri.resident import corpus as C, nlu as N

SHORT_GENERIC = {"why", "help", "bye", "cool", "hello", "thanks", "thanks a lot", "good morning", "good evening", "what can you do", "what are you able to do", "what do you know about kiln 2",
                 "what does that rating mean", "hi there", "thank you very much"}      # one-line stock phrases that legitimately appear in both


@pytest.fixture(scope="module")
def model():
    return N.Model.from_json(json.loads(N.MODEL_PATH.read_text()))


def _score(m, bat):
    ok = wrong = n = 0
    for intent, qs in bat.items():
        for q in qs:
            g = m.predict(q)["intent"]
            n += 1
            ok += g == intent
            wrong += (g != intent and g != "unknown")
    return ok / n, wrong / n


def test_model_has_the_new_classes(model):
    assert {"voyage", "feasibility", "offtopic"} <= set(model.classes)


def test_accuracy_on_independent_set_a(model):
    acc, wrong = _score(model, A.BATTERY)
    assert acc >= 0.92 and wrong <= 0.03          # measured 0.952 / 0.014 before the real data, 0.9375 / 0.014 after (29 Oct 2026)


def test_accuracy_on_clean_set_b(model):
    acc, wrong = _score(model, B.BATTERY)
    assert acc >= 0.88 and wrong <= 0.04          # measured 0.918 / 0.020 before the real data; 0.939 after, but B misses guided the last templates, so C is the fresh check


def test_accuracy_on_fresh_set_c(model):
    acc, wrong = _score(model, BC.BATTERY)
    assert acc >= 0.80 and wrong <= 0.08          # measured 0.842 / 0.053 on 29 Oct 2026 (0.868 for the model trained without the real data)


def test_real_out_of_scope_questions_are_not_sent_to_a_tool(model):
    """185 REAL out-of-scope queries (CLINC150 test split, CC BY 3.0) the model never saw. Measured 175 scope replies, 6 asks, 4 wrong tool (2.2%);
    the model before the real data sent 24 of 185 (13%) to a climate tool."""
    qs = [l.strip() for l in Path(__file__).with_name("data").joinpath("clinc_oos_heldout.txt").read_text().splitlines() if l.strip()]
    assert len(qs) >= 180
    scope = wrong = 0
    for q in qs:
        p = model.predict(q)
        if p["intent"] == "unknown":
            scope += bool(p.get("offtopic"))
        else:
            wrong += 1
    assert wrong / len(qs) <= 0.04 and scope / len(qs) >= 0.90


def test_heldout_real_queries_are_not_in_training():
    norm = lambda s: re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()          # noqa: E731
    held = {norm(l) for l in Path(__file__).with_name("data").joinpath("clinc_oos_heldout.txt").read_text().splitlines() if l.strip()}
    trained = {norm(u) for u, _ in C.real_rows()} | {norm(u) for u, _, _ in C.utterances()}
    assert not (held & trained)
    assert len(C.real_rows()) == 350


def test_offtopic_is_never_sent_to_a_tool(model):
    for bat in (A.BATTERY, B.BATTERY):
        for q in bat["unknown"]:
            p = model.predict(q)
            assert p["intent"] == "unknown", (q, p)
            assert "offtopic" not in [t["intent"] for t in p["top"]]


def test_climate_questions_are_not_called_offtopic(model):
    for bat in (A.BATTERY, B.BATTERY):
        for intent, qs in bat.items():
            if intent in ("unknown", "smalltalk"):
                continue
            for q in qs:
                assert not model.predict(q).get("offtopic"), q


def test_batteries_do_not_leak_into_the_corpus():
    norm = lambda s: re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()          # noqa: E731
    train = {norm(u) for u, _, _ in C.utterances()}
    for bat in (A.BATTERY, B.BATTERY, BC.BATTERY):
        leaked = {norm(q) for qs in bat.values() for q in qs if norm(q) in train} - SHORT_GENERIC
        assert not leaked, leaked


def test_corpus_templates_are_all_live():
    """A dict key repeated in corpus_more silently replaced 20 voyage templates with 3 during development; guard the counts."""
    assert len(C.T["voyage"]) >= 20 and len(C.T["feasibility"]) >= 20 and len(C.T["offtopic"]) >= 30


def test_floor_is_stated():
    assert N.CONF_FLOOR == pytest.approx(0.30)


def test_agent_replies_for_the_new_classes(monkeypatch, tmp_path):
    from cri.resident import agent as AG
    ag = AG.Agent()
    off = ag._dispatch("s", "what's the capital of Peru", {"intent": "unknown", "confidence": 0.8, "offtopic": True, "top": []}, {}, None)
    assert "outside what I do" in off["reply"] and all(a["type"] != "intent" for a in off["actions"])
    assert "where it is going" in ag._dispatch("s", "ship emissions", {"intent": "voyage", "confidence": 0.9, "top": []}, {}, None)["reply"]
    assert "do not price it" in ag._dispatch("s", "build a hotel", {"intent": "feasibility", "confidence": 0.9, "top": []}, {}, None)["reply"]
