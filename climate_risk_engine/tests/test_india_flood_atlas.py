import json
from pathlib import Path

import pytest

from cri import uncertainty as U
from cri.analysis import india_flood_atlas as A

DATA = Path(A.__file__).resolve().parents[1] / "data" / "india_flood_atlas.json"


def test_parse_matches_the_atlas_own_totals_state_by_state():
    d = json.loads(DATA.read_text())
    assert len(d["districts"]) == 439 and len(d["parse_check"]) == 25
    assert all(c["ok"] for c in d["parse_check"].values()), [s for s, c in d["parse_check"].items() if not c["ok"]]
    tn = {r["district"]: r["ha"] for r in d["districts"] if r["state"] == "Tamil Nadu"}
    assert tn["Chennai"] == 4892 and tn["Kanchipuram"] == 89794 and len(tn) == 24


def test_district_lookup_for_known_places():
    p = A.lookup(25.5941, 85.1376)                                    # Patna
    assert p["status"] == "listed" and p["district"] == "Patna" and p["atlas_ha"] == 170333 and p["national_percentile"] > 0.9
    c = A.lookup(12.9815, 80.2180)                                    # Velachery, Chennai district
    assert c["status"] == "listed" and c["atlas_ha"] == 4892 and c["national_percentile"] < 0.5
    assert any("2015 Chennai" in x for x in c["caveats"])


def test_no_record_is_never_zero():
    assert A.lookup(51.5, -0.12)["in_india"] is False and "atlas_ha" not in A.lookup(51.5, -0.12)
    goa = A.lookup(15.4909, 73.8278)
    assert goa["status"] == "state_not_covered" and goa["atlas_ha"] is None
    for r in (A.lookup(19.076, 72.8777), A.lookup(13.63, 79.42)):    # a split or unlisted district
        assert r["atlas_ha"] is None and r["status"] == "not_matched" and "not shown as zero" in r["reason"]


def _prof(atlas, sev):
    return {"hazards": {"river_flood": {"label": "River flood", "status": "modelled", "severity": sev}, "surface_water": {"label": "Rain", "status": "modelled", "severity": sev}},
            "history": {"flood_events": {"available": True, "events": []}, "flood_atlas": atlas}}


def test_concordance_warns_when_models_are_low_in_a_top_quarter_district_and_never_clears_on_low_rank():
    hi = A.lookup(25.5941, 85.1376)
    e = U.peril_concordance(_prof(hi, "LOW"), "flood")
    assert "Caution: Patna district is in the top quarter" in e["verdict"] and e["observed"]["atlas"]["atlas_ha"] == 170333
    lo = A.lookup(12.9815, 80.2180)
    e2 = U.peril_concordance(_prof(lo, "LOW"), "flood")
    assert "does not clear a site" in e2["verdict"] and "Caution: Chennai" not in e2["verdict"]
    e3 = U.peril_concordance(_prof(A.lookup(15.4909, 73.8278), "HIGH"), "flood")
    assert "no usable district statistic" in e3["verdict"]
