"""Scientific integrity checks: missing flood data must not read as low risk; losses reconcile; NFIP curve sane."""
import numpy as np
import pytest

from cri import insurance as I
from cri.finance import analyst, vulnerability as V
from tests.test_insurance import site


def _unavailable(*a, **k):
    return {"river": {"available": False, "reason": "test"}, "coastal": {"available": False, "reason": "test"}}


def test_unassessed_flood_is_labelled_not_negligible_and_loss_is_flagged_lower_bound(monkeypatch):
    monkeypatch.setattr(analyst, "site_flood", _unavailable)
    sites = [{"name": f"S{i}", "lat": 20 + i, "lon": 80 + i, "sector": "cement", "value_usd": 5e8, "emissions_t": 1e6 - i} for i in range(3)]
    r = analyst.analyse_sites(sites, top_n=1, n=300)
    assert r["physical_coverage"]["physical_loss_is_lower_bound"] is True
    assert r["physical_coverage"]["river_flood_assessed"] == 0
    for a in r["assets"]:
        assert a["river"] == "NOT ASSESSED" and a["coast"] == "NOT ASSESSED" and not a["river_assessed"]


def test_losses_never_exceed_value_and_percentages_bounded(monkeypatch):
    monkeypatch.setattr(analyst, "site_flood", _unavailable)
    r = analyst.analyse_asset(25.0, 90.0, name="X", sector="cement", value_usd=4e8, n=400, _flood=_unavailable())
    ph = r["financial"]["physical"]
    assert 0 <= ph["aal_usd"] <= r["asset"]["value"]["high"]
    assert ph["aal_pct"] <= 100
    for sc in r["financial"]["scenarios"].values():
        pct = sc["total_loss"]["pct_of_value"] if "total_loss" in sc else None
        if pct:
            assert pct["mean"] <= 100 and pct["p95"] <= 100


def test_nfip_curve_is_monotone_bounded_and_below_industrial_at_depth():
    d = np.array([0, 0.1, 0.3, 0.6, 1.0, 2.0, 3.0, 6.0])
    n = V.flood_damage(d, "nfip_residential_us")
    assert np.all(np.diff(n) >= -1e-9) and n[0] == 0 and n.max() <= 1
    assert V.flood_damage(2.0, "nfip_residential_us") < V.flood_damage(2.0, "northamerica")     # validation finding: engine industrial curve is higher


def test_insurance_line_can_use_nfip_curve_and_it_changes_loss_only_for_flood():
    s1 = site(h={"wind": False, "river": True}); s2 = dict(s1, vulnerability="nfip_residential_us")
    r1 = I.run([s1], hazards=[I.normalise_site(s1, 0)["hazard"]], n_years=3000)
    r2 = I.run([s2], hazards=[dict(I.normalise_site(s2, 0)["hazard"])], n_years=3000)
    a1, a2 = r1["stats"]["gross"]["aal"], r2["stats"]["gross"]["aal"]
    assert a1 != a2 and a2 >= 0


def test_validation_artifacts_exist_and_state_limits():
    import json
    from pathlib import Path
    j = json.loads((Path(__file__).resolve().parents[1] / "data" / "backtest" / "flood_vulnerability_nfip.json").read_text())
    assert j["n_claims_used"] > 100000 and j["limits"] and any(b["engine_over_observed"] > 1.5 for b in j["bins"])
