import json
from pathlib import Path

from cri import uncertainty as U
from cri.analysis import india_landslide_exposure as L

DATA = Path(L.__file__).resolve().parents[1] / "data" / "india_landslide_exposure.json"


def test_table_is_complete_and_matches_the_atlas_spot_checks():
    d = json.loads(DATA.read_text())
    assert d["parse_check"]["rows"] == 147 and d["parse_check"]["missing_ranks"] == []
    assert sorted(r["rank"] for r in d["districts"]) == list(range(1, 148))
    by = {r["rank"]: r for r in d["districts"]}
    assert (by[1]["district"], by[1]["state"]) == ("Rudraprayag", "Uttarakhand")      # the atlas text names Rudraprayag as highest
    assert [by[i]["district"] for i in (2, 3, 4, 5, 6)] == ["Tehri Garhwal", "Thrissur", "Rajauri", "Palakkad", "Poonch"]
    assert by[147]["district"] == "Udham Singh Nagar" and by[13]["district"] == "Wayanad" and by[126]["district"] == "Lahul & Spiti"


def test_splits_are_left_unmatched_and_spelling_variants_match():
    d = {r["district"]: r for r in json.loads(DATA.read_text())["districts"]}
    assert d["Karbi Anglong"]["matched"] is None and d["Jaintia Hills"]["matched"] is None      # split since 2011, not guessed
    assert d["Poonch"]["matched"] == "Punch" and d["Budgam"]["matched"] == "Badgam" and d["Leh-ladakh"]["matched"] == "Leh(Ladakh)"


def test_lookup_ranked_unranked_and_outside_india():
    r = L.lookup(30.2846, 78.9811)                                   # Rudraprayag town
    assert r["status"] == "ranked" and r["rank"] == 1 and r["of"] == 147 and "not a hazard probability" in r["reading"]
    w = L.lookup(11.6854, 76.1320)                                   # Kalpetta, Wayanad
    assert w["status"] == "ranked" and w["rank"] == 13
    ch = L.lookup(12.9815, 80.2180)                                  # Chennai: plains, not in the mountain table
    assert ch["status"] == "not_ranked" and ch["rank"] is None and "NOT ASSESSED" in ch["reason"]
    assert L.lookup(51.5, -0.12)["in_india"] is False


def _prof(ls, sev):
    hz = {"landslide_slope": {"label": "Slope screening", "status": "modelled", "severity": sev}} if sev else {}
    return {"hazards": hz, "history": {"landslide_exposure": ls}}


def test_concordance_cautions_on_low_slope_rating_in_top_third_and_never_clears():
    top = L.lookup(30.2846, 78.9811)
    e = U.peril_concordance(_prof(top, "LOW"), "slope")
    assert "Caution: NRSC ranks Rudraprayag district 1 of 147" in e["verdict"] and e["observed"]["rank"] == 1
    low = L.lookup(30.2846, 78.9811)
    low = {**low, "rank": 120}                                       # a low-ranked district must not read as clearing the site
    e2 = U.peril_concordance(_prof(low, "LOW"), "slope")
    assert "does not clear a site" in e2["verdict"] and "Caution" not in e2["verdict"]
    e3 = U.peril_concordance(_prof(L.lookup(12.9815, 80.2180), None), "slope")
    assert "not assessed and not safe" in e3["verdict"]
