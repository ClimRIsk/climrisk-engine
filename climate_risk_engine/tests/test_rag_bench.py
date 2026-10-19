"""Retrieval quality guard for the case library (see tools/rag_bench.py). Floors are set below the measured values so noise cannot fail them, but a regression will."""
import importlib.util
import os
from pathlib import Path

import pytest

from cri.resident import casebase as CB

_spec = importlib.util.spec_from_file_location("rag_bench", Path(__file__).resolve().parent.parent / "tools" / "rag_bench.py")
B = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(B)


def test_entity_questions_always_found():
    assert B.run_b()["hit@1"] >= 0.95                                   # "what did <company> report about <event>" must be exact


def test_no_false_routes_to_the_case_library():
    assert B.run_a()["false_routes"]["routed"] == 0


@pytest.mark.skipif(not os.environ.get("CRI_EMBED_DIR"), reason="needs the local embedding model (set CRI_EMBED_DIR)")
def test_dense_hybrid_beats_keywords_on_held_out_paraphrases():
    CB._IDX["path"] = None
    dense = B.run_a()
    old = CB.SEM_W
    CB.SEM_W = 0.0
    try:
        bm25 = B.run_a()
    finally:
        CB.SEM_W = old
    assert dense["p2"]["hit@6"] >= 0.80 and dense["p2"]["mrr"] > bm25["p2"]["mrr"] + 0.08
    assert dense["p2"]["routed_to_case_library"] >= 0.7


@pytest.mark.skipif(not os.environ.get("CRI_EMBED_DIR"), reason="needs the local embedding model (set CRI_EMBED_DIR)")
def test_questions_about_events_without_curated_cards_are_routed_and_found():
    CB._IDX["path"] = None
    p3 = B.run_p3()
    assert p3["filings_only"]["routed_to_case_library"] >= 0.75           # was 0.24 before event centroids were used for routing
    assert p3["filings_only"]["hit@6"] >= 0.85 and p3["curated"]["hit@1"] >= 0.8


@pytest.mark.skipif(not os.environ.get("CRI_EMBED_DIR"), reason="needs the local embedding model (set CRI_EMBED_DIR)")
def test_off_topic_past_event_questions_are_mostly_not_routed():
    CB._IDX["path"] = None
    fr = B.run_a()["false_routes_p3"]
    assert fr["routed"] <= 4, fr["which"]                                 # measured 3 of 25 (two come from the older rules, one from the event route)
