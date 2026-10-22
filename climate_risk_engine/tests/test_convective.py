import pytest

from cri import convective as C


def test_hail_and_tornado_alley_beat_the_pacific_northwest():
    ok = C.severe_hail(35.47, -97.52)           # Oklahoma City
    sea = C.severe_hail(47.61, -122.33)         # Seattle
    assert ok["covered"] and sea["covered"]
    assert ok["hail_days_per_year"] > 5 * max(sea["hail_days_per_year"], 0.1)
    assert ok["hail_2in_days_per_year"] <= ok["hail_days_per_year"]
    assert ok["band"] in ("High", "Very high") and sea["band"] in ("Low", "Moderate")
    assert C.tornado(35.47, -97.52)["tornado_days_per_year"] > 5 * max(C.tornado(47.61, -122.33)["tornado_days_per_year"], 0.01)


def test_ef_counts_are_nested():
    t = C.tornado(35.47, -97.52)
    assert t["tornado_days_per_year"] >= t["ef1_plus_days_per_year"] >= t["ef2_plus_days_per_year"] >= 0
    assert list(t["years"]) == [1996, 2024] and "no probability" in t["basis"]


@pytest.mark.parametrize("lat,lon", [(51.5, -0.12), (13.08, 80.27), (-33.9, 151.2), (49.0, -123.0), (32.52, -117.03), (42.33, -83.05)])
def test_outside_the_united_states_is_not_covered_never_zero(lat, lon):
    for f in (C.severe_hail, C.tornado):
        r = f(lat, lon)
        assert r["covered"] is False and "hail_days_per_year" not in r and "tornado_days_per_year" not in r
        assert "United States" in r["reason"] and "border" in r["reason"]


def test_alaska_and_hawaii_are_inside_and_windows_are_respected():
    assert C.in_united_states(61.2, -149.9) and C.in_united_states(21.3, -157.8)
    assert not C.in_united_states(25.0, -110.0)
    a = C.days_with_report("hail", 35.47, -97.52, 40.0, y0=1996, y1=2009, min_val=1.0)
    b = C.days_with_report("hail", 35.47, -97.52, 40.0, y0=2010, y1=2024, min_val=1.0)
    assert list(a["years"]) == [1996, 2009] and list(b["years"]) == [2010, 2024]


def test_report_zones_are_context_only(monkeypatch):
    from cri import physical_report as pr
    z = {x["key"]: x for x in pr._convective_marine_zones(35.47, -97.52)}
    assert set(z) == {"hail", "tornado"}                      # marine heatwave is opt-in
    assert all(k in pr._CONTEXT_KEYS for k in z)
    assert "Reports, not measurements" in z["hail"]["zone"] and "not the chance" in z["tornado"]["zone"]
    assert pr._convective_marine_zones(51.5, -0.12) == []       # UK: dropped, not 'low'
    monkeypatch.setenv("CRI_CONVECTIVE", "0")
    assert pr._convective_marine_zones(35.47, -97.52) == []
