import datetime as dt

import numpy as np
import pytest

from cri import marine as M


def _series(years=(1982, 2020), noise=0.0, seed=1):
    d0, d1 = dt.date(years[0], 1, 1), dt.date(years[1], 12, 31)
    dates = [(d0 + dt.timedelta(k)).isoformat() for k in range((d1 - d0).days + 1)]
    doy = np.array([dt.date.fromisoformat(x).timetuple().tm_yday for x in dates])
    rng = np.random.default_rng(seed)
    sst = 15 + 5 * np.sin(2 * np.pi * (doy - 100) / 365.25) + noise * rng.standard_normal(len(dates))
    return dates, sst


def test_a_known_heatwave_is_found_with_its_dates_and_category():
    dates, sst = _series(noise=0.3)
    i0 = dates.index("2015-07-10")
    sst[i0:i0 + 20] += 3.0
    ev = [e for e in M.detect(dates, sst)["events"] if e["start"] <= "2015-07-29" and e["end"] >= "2015-07-10"]
    assert len(ev) == 1
    e = ev[0]
    assert abs(e["duration_days"] - 20) <= 4 and e["peak_anomaly_c"] > 2.5 and e["category"] >= 2


def test_short_runs_are_not_events_and_small_gaps_are_joined():
    dates, sst = _series(noise=0.0)
    i = dates.index("2010-06-01")
    sst[i:i + 4] += 3.0                                    # 4 days: below the 5-day minimum
    j = dates.index("2012-06-01")
    sst[j:j + 6] += 3.0
    sst[j + 8:j + 14] += 3.0                               # 2-day gap: joined
    k = dates.index("2014-06-01")
    sst[k:k + 6] += 3.0
    sst[k + 10:k + 16] += 3.0                              # 4-day gap: two events
    ev = M.detect(dates, sst)["events"]
    near = lambda a, b: [e for e in ev if a <= e["start"] <= b]
    assert near("2010-05-25", "2010-06-10") == []
    assert len(near("2012-05-25", "2012-06-20")) == 1 and near("2012-05-25", "2012-06-20")[0]["duration_days"] == 14
    assert len(near("2014-05-25", "2014-06-20")) == 2


def test_baseline_period_has_90th_percentile_exceedance_near_ten_percent():
    dates, sst = _series(noise=0.5, seed=3)
    mean, thr = M.climatology(dates, sst)
    yr = np.array([int(x[:4]) for x in dates])
    doy = M._doy(dates) - 1
    base = (yr >= 1982) & (yr <= 2011)
    share = float(np.mean(sst[base] > thr[doy[base]]))
    assert 0.07 < share < 0.13
    assert np.all(thr >= mean)


def test_marine_heatwave_summary_on_a_fake_series(monkeypatch):
    dates, sst = _series(noise=0.3)
    i = dates.index("2016-02-01")
    sst[i:i + 40] += 3.5
    monkeypatch.setattr(M, "sst_series", lambda *a, **k: {"time": dates, "sst": [float(x) for x in sst], "cell_lat": -16.4, "cell_lon": 146.1, "distance_km": 12.0})
    r = M.marine_heatwave(-16.5, 146.0)
    assert r["covered"] and r["strongest_event"]["start"] <= "2016-03-10" and r["strongest_event"]["end"] >= "2016-02-01"
    assert r["heatwave_days_per_year_last_10_years"] > r["heatwave_days_per_year_baseline_1982_2011"] * 0.5
    assert "Surface sea temperature" in r["basis"]


def test_marine_zone_is_opt_in_and_failure_drops_the_zone(monkeypatch):
    from cri import physical_report as pr
    monkeypatch.delenv("CRI_MARINE", raising=False)
    assert [z for z in pr._convective_marine_zones(51.5, -0.12) if z["key"] == "marine_heatwave"] == []
    monkeypatch.setenv("CRI_MARINE", "1")
    def boom(*a, **k):
        raise RuntimeError("no ocean cell")
    monkeypatch.setattr(M, "marine_heatwave", boom)
    assert [z for z in pr._convective_marine_zones(48.0, 11.0) if z["key"] == "marine_heatwave"] == []
    monkeypatch.setattr(M, "marine_heatwave", lambda *a, **k: {"covered": True, "heatwave_days_per_year_last_10_years": 40.0, "heatwave_days_per_year_baseline_1982_2011": 15.0,
        "distance_to_ocean_cell_km": 9.0, "strongest_event": {"start": "2022-06-01", "end": "2022-08-01", "duration_days": 61, "peak_anomaly_c": 4.7, "category": 3},
        "category_names": M.CATEGORIES, "source": "test"})
    z = [z for z in pr._convective_marine_zones(41.0, 4.0) if z["key"] == "marine_heatwave"][0]
    assert z["rating"] is None and "Surface water only" in z["zone"] and "marine_heatwave" in pr._CONTEXT_KEYS


def test_cell_index_wraps_the_dateline():
    assert M.cell_index(0.1, 179.9)[1] == 719 and M.cell_index(0.1, -179.9)[1] == 720
    la, lo = M.cell_centre(*M.cell_index(-16.5, 146.0))
    assert abs(la + 16.5) < 0.13 and abs(lo - 146.0) < 0.13
