"""Golden sites: coarse expert expectations vs what the engine said on the day the snapshot was taken (tests/golden/).

Three checks:
 1. every expectation the engine misses is listed in KNOWN_GAPS with a reason (nothing fails silently);
 2. every KNOWN_GAPS entry still fails: when a fix makes one pass, remove it (the gap list must shrink, never rot);
 3. the zones computable without the network (tropical-cyclone rate, WRI water stress) are recomputed live and must agree with the snapshot
    ratings' direction, so a change in those models shows up here.
The snapshot itself is refreshed deliberately (python tests/golden/refresh.py) and the diff reviewed."""
import json, sys
from pathlib import Path
import pytest

HERE = Path(__file__).resolve().parent / "golden"
sys.path.insert(0, str(HERE))
from sites import SITES, KNOWN_GAPS      # noqa: E402

LV = ["NEGLIGIBLE", "LOW", "MODERATE", "HIGH", "CRITICAL"]
SNAP = json.loads(next(HERE.glob("snapshot_*.json")).read_text())["sites"]


def _fails(got, c):
    if got is None:
        return "min" in c
    return LV.index(got) < LV.index(c["min"]) if "min" in c else LV.index(got) > LV.index(c["max"])


def _cases():
    for n, (_, _, exp, _) in SITES.items():
        for z, c in exp.items():
            yield n, z, c


def test_every_miss_is_a_documented_known_gap():
    undocumented = [(n, z, c, SNAP[n].get(z)) for n, z, c in _cases() if _fails(SNAP[n].get(z), c) and (n, z) not in KNOWN_GAPS]
    assert not undocumented, f"new disagreement with expert expectation, document it in KNOWN_GAPS or fix the engine: {undocumented}"


def test_known_gaps_still_fail():
    exp = {(n, z): c for n, z, c in _cases()}
    fixed = [(n, z) for (n, z) in KNOWN_GAPS if not _fails(SNAP[n].get(z), exp[(n, z)])]
    assert not fixed, f"these now meet the expectation, remove them from KNOWN_GAPS: {fixed}"


def test_snapshot_covers_all_sites_without_errors():
    assert set(SNAP) == set(SITES)
    assert [n for n, z in SNAP.items() if "error" in z] == []


def test_agreement_rate_is_reported_and_not_worse_than_today():
    total = sum(1 for _ in _cases())
    met = sum(1 for n, z, c in _cases() if not _fails(SNAP[n].get(z), c))
    assert total >= 40 and met / total >= 0.58, (met, total)      # 28 of 46 on 19 Oct 2026; raise this floor as gaps are fixed


def test_live_cyclone_rate_agrees_with_expectations():
    from cri.climate import tropical_cyclone as TC
    for n, (la, lo, exp, _) in SITES.items():
        if "cyclone" not in exp:
            continue
        rate = TC.site_hazard(la, lo).annual_rate_ge_ts or 0.0
        c = exp["cyclone"]
        if "min" in c and LV.index(c["min"]) >= 2:
            assert rate >= 0.05, (n, rate)           # MODERATE or higher expected -> a tropical storm passes within range most decades
        if "max" in c and c["max"] == "NEGLIGIBLE":
            assert rate < 0.05, (n, rate)


def test_live_water_stress_agrees_with_expectations():
    from cri.connectors.wri_aqueduct import WRIAqueductConnector
    w = WRIAqueductConnector()
    for n, (la, lo, exp, _) in SITES.items():
        if exp.get("water", {}).get("min") == "HIGH":
            r = w.get_water_risk(la, lo, 2019)
            assert r.get("water_stress") is not None and r["water_stress"] >= 3.0, (n, r.get("water_stress"))


# ── insurer view: ranking sanity at golden sites (snapshot of insurance.run, 10 m TIV each) ─────────────────────────────────
INS = json.loads(next(HERE.glob("insurer_snapshot_*.json")).read_text())["sites"]


def test_insurer_cyclone_ranking_is_plausible():
    w = {n: v["wind"] for n, v in INS.items()}
    assert min(w["miami"], w["new_orleans"], w["houston"]) > 2 * w["tokyo"] > 0     # Gulf/Florida hurricane belt >> Tokyo at equal TIV
    assert w["new_orleans"] == max(w.values())
    assert w["rotterdam"] == 0 == w["phoenix"]                                       # no tropical-cyclone record: 0 here means NOT MODELLED (see below)


def test_insurer_flags_what_it_did_not_model_instead_of_reading_zero():
    for n in ("rotterdam", "phoenix"):
        assert "wind" in INS[n]["na"], f"{n}: wind AAL of 0 must carry a 'not assessed' flag"
    for n in INS:
        assert "flood: surface-water and small-river" in INS[n]["na"]


def test_insurer_known_gap_coastal_loss_is_zero_at_surge_prone_sites():
    # Miami and New Orleans show zero coastal AAL because the Deltares map reads dry at the pin (see KNOWN_GAPS). Recorded, not endorsed:
    # when the coastal map improves this test should fail and be rewritten.
    assert INS["miami"]["coast"] == 0 and INS["new_orleans"]["coast"] == 0
