import datetime as dt
import urllib.parse

import numpy as np
import pytest

from cri.analysis import thermal_stress as TS


def _fake(lat_hot=True):
    """Synthetic ERA5-like hourly weather: seasonal + daily cycle, so the pipeline can run offline."""
    def fetch(url):
        q = dict(urllib.parse.parse_qsl(urllib.parse.urlparse(url).query))
        a, b = dt.date.fromisoformat(q["start_date"]), dt.date.fromisoformat(q["end_date"])
        n = ((b - a).days + 1) * 24
        t = np.datetime64(a.isoformat() + "T00:00") + np.arange(n) * np.timedelta64(1, "h")
        doy = ((t.astype("datetime64[D]") - t.astype("datetime64[Y]").astype("datetime64[D]")).astype(int) + 1).astype(float)
        hr = (t - t.astype("datetime64[D]")).astype("timedelta64[h]").astype(int).astype(float)
        base = 31.0 if lat_hot else 5.0
        temp = base + 6 * np.sin(2 * np.pi * (doy - 110) / 365) + 5 * np.sin(2 * np.pi * (hr - 9) / 24)
        sw = np.maximum(0.0, 900 * np.sin(np.pi * (hr - 6) / 12)) * (hr >= 6) * (hr <= 18)
        return {"hourly": {"time": [str(x)[:16] for x in t], "temperature_2m": temp.tolist(), "relative_humidity_2m": [60.0] * n, "surface_pressure": [1000.0] * n,
                           "wind_speed_10m": [2.0] * n, "shortwave_radiation": sw.tolist()}}
    return fetch


def test_assess_hot_site_counts_heat_hours_not_cold(tmp_path, monkeypatch):
    monkeypatch.setenv("CRI_CACHE_DIR", str(tmp_path))
    a = TS.assess(10.0, 78.0, today=dt.date(2026, 10, 13), fetch=_fake(True), future=False)
    assert a["status"] == "modelled"
    s = a["summary"]
    assert s["heat_hours_per_year_last10"]["moderate_heat_ge_32"] > 500
    assert s["cold_hours_per_year_last10"]["strong_cold_le_minus13"] == 0
    assert a["region_group"] == "elsewhere" and a["bias"]["hot_tail_bias_c"] < 0
    adj = a["bias"]["heat_hours_per_year_last10_bias_adjusted"]["moderate_heat_ge_32"]
    assert adj >= s["heat_hours_per_year_last10"]["moderate_heat_ge_32"]            # adding the cool bias back never lowers the count
    assert s["years"][0] == 1981 and s["years"][1] == 2025


def test_assess_cold_site_counts_cold_stress(tmp_path, monkeypatch):
    monkeypatch.setenv("CRI_CACHE_DIR", str(tmp_path))
    a = TS.assess(60.0, 11.0, today=dt.date(2026, 10, 13), fetch=_fake(False), future=False)
    assert a["region_group"] == "Europe"
    assert a["summary"]["cold_hours_per_year_last10"]["moderate_cold_le_0"] > 500
    assert a["summary"]["heat_hours_per_year_last10"]["moderate_heat_ge_32"] == 0


def test_null_when_reanalysis_unreadable(tmp_path, monkeypatch):
    monkeypatch.setenv("CRI_CACHE_DIR", str(tmp_path))

    def boom(url):
        raise OSError("offline")
    a = TS.assess(10.0, 78.0, today=dt.date(2026, 10, 13), fetch=boom, future=False)
    assert a["status"] == "null" and "could not be read" in a["reason"]


def test_future_sensitivity_more_heat_less_cold(tmp_path, monkeypatch):
    monkeypatch.setenv("CRI_CACHE_DIR", str(tmp_path))
    from cri.modellab import scenarios as SC

    def fake_cf(hazard, lat, lon, fetch=None, ssps=SC.SSPS):
        sh = {"p10": 1.0, "median": 2.0, "p90": 3.5}
        return {"periods": ["2040-2059"], "ssp": {s: {"2040-2059": sh} for s in ssps}}
    monkeypatch.setattr(SC, "change_factors", fake_cf)
    a = TS.assess(45.0, 5.0, today=dt.date(2026, 10, 13), fetch=_fake(True), future=True)
    f = a["future"]
    assert f["status"] == "ok" and f["rows"]
    r = f["rows"][0]
    base = f["baseline_hours_per_year"]["moderate_heat_ge_32"]
    assert r["heat_hours_per_year"]["moderate_heat_ge_32"]["central_rh_held"] >= base
    assert r["heat_hours_per_year"]["moderate_heat_ge_32"]["low"] <= r["heat_hours_per_year"]["moderate_heat_ge_32"]["high"]


def test_region_groups():
    assert TS.region_group(48.8, 2.3) == "Europe" and TS.region_group(51.0, -114.0) == "Canada" and TS.region_group(33.4, -112.0) == "US" and TS.region_group(9.8, 78.0) == "elsewhere"
