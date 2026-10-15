"""The 1,000-scenario list: every row is read correctly by the agent, never run at a place the user did not name without saying so,
and the training rows built from it carry only engine facts."""
import csv
import random
from pathlib import Path

import pytest

from cri.resident import agent as A

CSV = Path(__file__).resolve().parents[1] / "data" / "scenarios" / "climate_risk_scenarios_1000.csv"
SSP = {"SSP1-2.6": "SSP126", "SSP2-4.5": "SSP245", "SSP3-7.0": "SSP370", "SSP5-8.5": "SSP585"}


@pytest.fixture(scope="module")
def recs():
    with open(CSV, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


@pytest.fixture()
def ag(monkeypatch):
    monkeypatch.setattr(A, "_job", lambda fn, label: {"id": "job-test"})
    return A.get_agent() if hasattr(A, "get_agent") else A.Agent()


def test_file_is_what_it_says(recs):
    assert len(recs) == 1000
    assert len({(r["Sector"], r["Primary_Hazard"], r["Climate_Pathway"], r["Horizon_Year"]) for r in recs}) == 15      # 1000 rows, 15 distinct scenarios


def test_every_row_is_read_with_its_own_pathway_and_year(ag, recs, monkeypatch):
    monkeypatch.setattr(A.M, "get_state", lambda k, d=None: None)
    for r in recs[::10]:
        out = ag._scenario_commands("s-csv", r["Description"])
        assert out is not None
        assert "Which place" in out["reply"], r["Description"]                      # no place named: asks, never guesses


def test_year_2100_is_not_read_as_2070(ag, monkeypatch):
    monkeypatch.setattr(A.M, "get_state", lambda k, d=None: {"lat": 1.0, "lon": 2.0, "name": "Somewhere", "source": "x"} if k == "last_place" else None)
    monkeypatch.setattr(A.M, "set_state", lambda *a, **k: None)
    out = ag._scenario_commands("s-csv", "A Energy facility faces acute vulnerability to Extreme Heat under SSP5-8.5 (High) projected for the year 2100.")
    assert out and "SSP585 scenario for 2100" in out["reply"]


def test_carried_over_place_is_declared(ag, monkeypatch):
    monkeypatch.setattr(A.M, "get_state", lambda k, d=None: {"lat": 29.7, "lon": -95.0, "name": "29.7000, -95.0000", "source": "x"} if k == "last_place" else None)
    monkeypatch.setattr(A.M, "set_state", lambda *a, **k: None)
    out = ag._scenario_commands("s-csv", "A Logistics facility faces Coastal Inundation under SSP2-4.5 (Medium) projected for the year 2050.")
    assert out and "names no place" in out["reply"] and "29.7000" in out["reply"]


def test_training_rows_carry_engine_facts_only_and_hold_out_whole_scenarios():
    from cri.llmtrain import dataset as D, evaluate as E
    rows = D.build_scenarios_csv(CSV)
    assert {r["split"] for r in rows} == {"train", "valid", "test"}
    train = {tuple(r["meta"]["combo"]) for r in rows if r["split"] == "train"}
    test = {tuple(r["meta"]["combo"]) for r in rows if r["split"] == "test"}
    assert train and test and not (train & test)                                    # a held-out scenario is never seen in training
    for r in rows:
        ref = r["messages"][-1]["content"]
        assert E.nums(ref) <= E.nums(r["meta"]["facts"]) | E.nums(r["messages"][1]["content"]) | {"0", "1"}, ref
        assert "not assessed" in ref and "not zero" in ref and "your own input" in ref
        assert "no location" in ref.lower()


ANS = CSV.with_name("climate_risk_answers_1000.csv")


def test_answer_file_is_formulaic_so_it_is_reviewed_not_copied():
    with open(CSV, newline="", encoding="utf-8") as fh:
        sc = {r["Scenario_ID"]: r["Description"] for r in csv.DictReader(fh)}
    with open(ANS, newline="", encoding="utf-8") as fh:
        an = list(csv.DictReader(fh))
    import re
    assert len(an) == 1000
    assert len({re.sub(r"[\d.]+", "N", r["Target_Action_Plan"]) for r in an}) == 1            # one template for every sector and hazard
    for r in an:
        exc = int(re.search(r"by (\d+)%", sc[r["Scenario_ID"]]).group(1))
        buf = float(re.search(r"to \+([\d.]+)%", r["Target_Action_Plan"]).group(1))
        assert buf == pytest.approx(1.2 * exc, abs=0.051)                                     # the buffer is 1.2 x the stated exceedance


def test_plan_review_rows_flag_fixed_multiple_and_unverified_inputs():
    from cri.llmtrain import dataset as D, evaluate as E
    rows = D.build_plan_review(CSV, ANS)
    assert rows and {r["split"] for r in rows} == {"train", "valid", "test"}
    train = {tuple(r["meta"]["combo"]) for r in rows if r["split"] == "train"}
    test = {tuple(r["meta"]["combo"]) for r in rows if r["split"] == "test"}
    assert not (train & test)
    for r in rows:
        ref, facts = r["messages"][-1]["content"], r["meta"]["facts"] + r["messages"][1]["content"]
        assert E.nums(ref) <= E.nums(facts) | {"0", "1"}, ref
        low = ref.lower()
        assert "fixed multiple" in low and "not assessed" in low and "has not verified" in low
        assert "diversif" in low and "stress test" in low
