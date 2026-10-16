from cri.finance import analyst as A


def _r(findings, quality="exact", value=1e6):
    sev = {"severity": "NEGLIGIBLE"}
    return {"asset": {"name": "X", "value": {"mode": value, "quality": quality}, "emissions_t": 100},
            "financial": {"scenarios": {"Delayed transition": {"total_npv": {"mean": 1.0, "p95": 2.0, "pct_of_value": {"mean": 5.0}}}},
                          "physical": {"aal_pct": 0.0, "annual_damage_ratio_rp": {"100": 0.0}}},
            "physical": {"cyclone": sev, "river_flood": sev, "coastal_flood": sev, "site_analysis": {"findings": findings}}}


def test_headline_names_unpriced_hazards_and_estimated_value():
    h = A._headline(_r([{"hazard": "debris_flow", "severity": "HIGH"}, {"hazard": "wildfire", "severity": "LOW"}], quality="proxy"))
    assert "not priced in this loss: debris flow (HIGH)" in h and "wildfire" not in h
    assert "asset value is itself an estimate (from emissions)" in h


def test_headline_clean_when_nothing_unpriced():
    h = A._headline(_r([{"hazard": "flood", "severity": "HIGH"}]))
    assert "not priced" not in h and "estimate" not in h
