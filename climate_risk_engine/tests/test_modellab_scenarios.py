"""Climate scenarios (CMIP6 point reader, change factors, backtest), evidence grades, portfolio screening and the validation pack: offline, with synthetic NetCDF files."""
import io
import math

import numpy as np
import pytest

from cri.modellab import evidence, scenarios as SC, studies_evt

scipy_io = pytest.importorskip("scipy.io")


def _cdf(nrec, grid_fn, name="anomaly-x-annual-mean", with_sig=True):
    """A small classic NetCDF file (CDF-1) with a record dimension, like the CCKP files: grid 5 x 8, record k = grid_fn(k)."""
    import tempfile, os
    p = tempfile.mktemp(suffix=".nc")
    f = scipy_io.netcdf_file(p, "w", version=1)
    f.createDimension("time", None); f.createDimension("lat", 5); f.createDimension("lon", 8)
    t = f.createVariable("time", "d", ("time",))
    v = f.createVariable(name, "f", ("time", "lat", "lon"))
    if with_sig:
        s = f.createVariable("anomalysignificance-x-annual-mean", "f", ("time", "lat", "lon"))
    for k in range(nrec):
        t[k] = float(k); v[k] = grid_fn(k).astype("f4")
        if with_sig:
            s[k] = np.ones((5, 8), "f4")
    f.close()
    b = open(p, "rb").read(); os.remove(p)
    return b


@pytest.fixture(autouse=True)
def _small_grid(monkeypatch):
    monkeypatch.setattr(SC, "grid_ij", lambda lat, lon, nlat=721, nlon=1440: (2, 3))
    SC._HDR.clear()
    monkeypatch.setenv("CRI_CACHE_DIR", "/tmp/cri-test-cache-scen")


def _fetcher(files):
    def f(url, a, b):
        for key, data in files:
            if key(url):
                return data[a:b + 1]
        raise AssertionError("unexpected url " + url)
    return f


def test_point_read_and_masked_cell_uses_neighbour():
    g = np.arange(40, dtype="f4").reshape(5, 8)
    g[2, 3] = 1e20                                                   # masked (sea)
    data = _cdf(1, lambda k: g)
    fetch = _fetcher([(lambda u: True, data)])
    assert SC._read("u", 0, 2, 4, fetch) == pytest.approx(20.0)
    assert SC._read("u", 0, 2, 3, fetch) is None
    assert SC._cell_value("u", 0, 2, 3, fetch) == pytest.approx(20.0)  # (2,3) masked: first neighbour (2,4)
    two = _cdf(3, lambda k: g + 100 * k)
    SC._HDR.clear()
    assert SC._read("v", 2, 1, 1, _fetcher([(lambda u: True, two)])) == pytest.approx(g[1, 1] + 200)


def _files(anom_by_pct, hist_years):
    """anomaly files (one record) per percentile + a historical yearly series file (record k = year-1950)."""
    anom = {p: _cdf(1, lambda k, d=d: np.full((5, 8), d), "anomaly-x-annual-mean") for p, d in anom_by_pct.items()}
    hist = _cdf(65, lambda k: np.full((5, 8), hist_years(1950 + k)), "timeseries-x-annual-mean", with_sig=False)
    def pick(url):
        return None
    return anom, hist


def _fake_fetch(anom, hist):
    def f(url, a, b):
        if "ensemble-all-historical" in url:
            return hist[a:b + 1]
        for p, data in anom.items():
            if f"climatology_{p}_" in url:
                return data[a:b + 1]
        raise AssertionError(url)
    return f


def test_change_factors_ratio_and_shift():
    anom, hist = _files({"p10": 3.0, "median": 6.0, "p90": 12.0}, lambda y: 60.0)
    fac = SC.change_factors("rain", 23.8, 90.4, fetch=_fake_fetch(anom, hist))
    assert fac["baseline_1995_2014"] == pytest.approx(60.0)
    d = fac["ssp"]["ssp245"]["2080-2099"]
    assert d["median"] == pytest.approx(1.10) and d["p10"] == pytest.approx(1.05) and d["p90"] == pytest.approx(1.20)
    anom, hist = _files({"p10": 1.0, "median": 3.0, "p90": 5.0}, lambda y: 37.0)
    ft = SC.change_factors("heat", 23.8, 90.4, fetch=_fake_fetch(anom, hist))
    assert ft["ssp"]["ssp585"]["2040-2059"]["median"] == pytest.approx(3.0)


def test_apply_ratio_shift_and_wbgt_range():
    f = {"p10": 1.05, "median": 1.10, "p90": 1.20}
    assert SC.apply(f, 100.0, "ratio") == pytest.approx((105.0, 110.0, 120.0))
    h = {"p10": 1.0, "median": 3.0, "p90": 5.0}
    assert SC.apply(h, 30.0, "shift") == pytest.approx((31.0, 33.0, 35.0))
    lo, mid, hi = SC.apply(h, 30.0, "shift", wbgt=True)
    assert lo == pytest.approx(30 + 0.567 * 3) and hi == pytest.approx(30 + 1.36 * 3) and lo < mid < hi


def test_backtest_pass_and_fail_verdicts():
    yrs = list(range(1981, 2024))
    rng = np.random.default_rng(3)
    model = lambda y: 60.0 + 0.5 * (y - 1981)                         # +5 mm per decade
    anom, hist = _files({"p10": 0, "median": 0, "p90": 0}, model)
    fetch = _fake_fetch(anom, hist)
    obs = [model(y) + rng.normal(0, 6) for y in yrs]                  # same trend, noise
    b = SC.backtest("rain", 0, 0, yrs, obs, fetch=fetch)
    assert b["ok"] and b["verdict"] == "pass" and b["level_ok"] and b["trend_ok"]
    SC._HDR.clear()
    bad = [200.0 - 4.0 * (y - 1981) + rng.normal(0, 3) for y in yrs]   # wrong level and opposite trend
    b2 = SC.backtest("rain", 0, 0, yrs, bad, fetch=fetch)
    assert b2["verdict"] == "fail" and not b2["same_sign"]
    short = SC.backtest("rain", 0, 0, yrs[:10], obs[:10], fetch=fetch)
    assert not short["ok"] and "at least 20" in short["reason"]


def test_projection_table_rows():
    fac = {"mode": "ratio", "periods": ["2040-2059"], "ssp": {"ssp245": {"2040-2059": {"p10": 1.0, "median": 1.1, "p90": 1.3}}}}
    tab = SC.projection_table(fac, {10: 100.0, 50: 150.0, 100: 200.0}, "{:.0f}", "mm")
    assert tab["rows"][0][2].startswith("+10 %") and "220" in tab["rows"][0][3]


def test_evt_study_adds_scenario_section_and_grade(monkeypatch):
    fac = {"hazard": "rain", "var": "rx1day", "mode": "ratio", "unit": "mm/day", "label": "annual maximum one-day rainfall (Rx1day)", "baseline_1995_2014": 60.0, "cell": [23.75, 90.5], "periods": list(SC.PERIODS),
           "ssp": {s: {p: {"p10": 0.95, "median": 1.08, "p90": 1.25} for p in SC.PERIODS} for s in SC.SSPS}, "source": SC.NOTE, "attribution": SC.ATTRIBUTION}
    monkeypatch.setattr(SC, "change_factors", lambda *a, **k: fac)
    yrs = list(range(1981, 2024)); rng = np.random.default_rng(5); x = rng.gumbel(60, 12, len(yrs))
    monkeypatch.setattr(SC, "observed_series", lambda *a, **k: (yrs, list(x)))
    monkeypatch.setattr(SC, "backtest", lambda *a, **k: {"ok": True, "verdict": "partial", "summary": "level matches, trend does not", "bias_text": "obs 70 vs model 60", "obs_slope_decade": 1.0, "obs_slope_range": [-1.0, 3.0],
                                                           "model_slope_decade": 0.2, "trend_resolvable": False, "level_ok": True, "trend_ok": True, "same_sign": True})
    r = studies_evt.run("rain", 23.8, 90.4, "Test", series=(yrs, x), bootstrap=30, scenarios=True)
    titles = [b[2] for b in r["blocks"] if b[0] == "h2"]
    assert any("climate scenarios" in t.lower() for t in titles) and any("backtest" in t.lower() for t in titles)
    assert r["numbers"]["scenarios"]["backtest"]["verdict"] == "partial"
    assert any("Scenario data" in l for l in r["limitations"])
    # without the option nothing is fetched
    r0 = studies_evt.run("rain", 23.8, 90.4, "Test", series=(yrs, x), bootstrap=30)
    assert r0["numbers"]["scenarios"] is None


def test_scenario_failure_is_a_note_not_an_error(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("offline")
    monkeypatch.setattr(SC, "change_factors", boom)
    yrs = list(range(1981, 2024)); x = np.random.default_rng(5).gumbel(60, 12, len(yrs))
    r = studies_evt.run("rain", 23.8, 90.4, "Test", series=(yrs, x), bootstrap=30, scenarios=True)
    assert r["ok"] and any("not available" in l for l in r["limitations"])


# ── evidence grade ───────────────────────────────────────────────────────────────────────────────────────────────────────

def _res(**kw):
    base = {"kind": "evt", "status": "verified and hindcast-validated", "verification": [{"ok": True}] * 6, "limitations": [],
            "numbers": {"n_years": 43, "source": "NASA POWER", "levels": {10: 80.0, 50: 100.0, 100: 110.0}, "ci": {"100": [95.0, 125.0]}}, "headline": [], "blocks": [("h2", "q", "Q"), ("p", "x")], "standfirst": "S."}
    base.update(kw)
    return base


def test_grade_strong_weak_failed_and_nomodel():
    assert evidence.grade(_res())["grade"] in ("A", "B")
    strong = evidence.grade(_res())
    assert strong["score"] == strong["max"] - 0 or strong["score"] >= 0.8 * strong["max"]
    weak = _res(status="verified; hindcast FAILED (the early record does not predict the recent one)", numbers={"n_years": 20, "source": "ERA5 reanalysis (ECMWF) via Open-Meteo", "levels": {100: 100.0}, "ci": {"100": [40.0, 180.0]}})
    assert evidence.grade(weak)["grade"] == "D"
    failed = _res(verification=[{"ok": True}, {"ok": False}])
    assert evidence.grade(failed)["grade"] == "D"
    assert evidence.grade({"kind": "no_model", "status": "no model (x)"})["grade"] == "n/a"


def test_grade_uses_backtest_and_attaches_report_blocks():
    good = _res(); good["numbers"]["scenarios"] = {"table": {"a": 1}, "backtest": {"ok": True, "verdict": "pass", "summary": "ok"}}
    bad = _res(); bad["numbers"]["scenarios"] = {"table": {"a": 1}, "backtest": {"ok": True, "verdict": "fail", "summary": "no"}}
    assert evidence.grade(good)["score"] > evidence.grade(bad)["score"]
    out = evidence.attach(good)
    assert out["headline"][-1][1].startswith("evidence grade") and any(b[0] == "table" for b in out["blocks"]) and "Evidence grade" in out["standfirst"]
    assert out["blocks"][2][0] == "h2"                                 # inserted after the question


# ── portfolio screening ──────────────────────────────────────────────────────────────────────────────────────────────────

from cri.modellab import portfolio as PF, validation_pack as VP


def test_parse_assets_and_errors():
    a = PF.parse_assets("Asset,Latitude,Longitude,Value\nPlant A,23.81,90.41,\"$1,200,000\"\nPlant B,23.82,90.42,500\n")
    assert len(a) == 2 and a[0]["value"] == 1200000 and a[1]["name"] == "Plant B"
    with pytest.raises(ValueError, match="row 2"):
        PF.parse_assets("name,lat,lon\nX,95,10\n")
    with pytest.raises(ValueError, match="latitude and longitude"):
        PF.parse_assets("name,x1,x2\nA,1,2\n")
    with pytest.raises(ValueError, match="at most"):
        PF.parse_assets("name,lat,lon\n" + "\n".join(f"a{i},1,2" for i in range(501)))


def test_cell_groups_nearby_assets_and_separates_distant_ones():
    assert PF.cell(23.81, 90.41) == PF.cell(23.76, 90.30)
    assert PF.cell(23.81, 90.41) != PF.cell(24.30, 90.41)
    assert PF.cell(-90.0, -180.0) == (-89.875, -179.875)


def test_screen_shares_cells_and_resumes(tmp_path):
    calls = []

    def runner(h, la, lo, nm):
        calls.append((h, la, lo))
        n = {"levels": {100: 123.0}, "evidence": {"grade": "B"}}
        if h == "rain":
            n["scenarios"] = {"mode": "ratio", "table": {"ssp245:2040-2059": {"change": {"p10": 1.0, "median": 1.12, "p90": 1.3}}, "ssp585:2080-2099": {"change": {"p10": 1.0, "median": 1.3, "p90": 1.6}}}, "backtest": {"ok": True, "verdict": "pass"}}
        return {"status": "verified and hindcast-validated", "ok": True, "label": h, "numbers": n}
    assets = [{"name": "A", "lat": 23.81, "lon": 90.41}, {"name": "B", "lat": 23.77, "lon": 90.35}, {"name": "C", "lat": 51.5, "lon": -0.12}]
    store = str(tmp_path / "p.jsonl")
    r = PF.screen(assets, ("rain", "heat"), store=store, runner=runner)
    assert r["cells"] == 2 and len(calls) == 4 and r["runs"] == 4                  # A and B share a cell
    r2 = PF.screen([dict(a) for a in assets], ("rain", "heat"), store=store, runner=runner)
    assert len(calls) == 4 and r2["runs"] == 0 and r2["reused"] == 4              # resumed: nothing re-run
    head, rows = PF.table(r)
    assert head[-2:] == ["rain", "heat"] and rows[0][-2].startswith("123") and rows[0][3] == rows[1][3]
    d = PF.digest(r)
    assert d["movers"][0][2] == pytest.approx(12.0) and d["hazards"]["rain"] == {"B": 3}
    assert "Asset" in PF.to_csv(r)


def test_screen_records_errors_and_defers_on_quota(tmp_path):
    def runner(h, la, lo, nm):
        if h == "rain":
            raise RuntimeError("the free weather-data service has used up its daily allowance")
        raise ValueError("boom")
    r = PF.screen([{"name": "A", "lat": 1.0, "lon": 2.0}], ("rain", "heat"), store=str(tmp_path / "p.jsonl"), runner=runner)
    res = r["results"]
    assert any(v.get("deferred") for v in res.values()) and any(v["status"].startswith("error") for v in res.values())
    assert not (tmp_path / "p.jsonl").read_text().count("deferred")                # deferred results are not stored


# ── validation pack ──────────────────────────────────────────────────────────────────────────────────────────────────────

def test_validation_pack_is_generated_from_results(tmp_path):
    import json
    rows = [{"site": "X", "zone": "z1", "hazard": "rain", "status": "verified and hindcast-validated", "ks_p": 0.4, "hc_passed": True},
            {"site": "Y", "zone": "z2", "hazard": "rain", "status": "verified; hindcast FAILED (the early record does not predict the recent one)", "ks_p": 0.01, "hc_passed": False, "drift": 0.2},
            {"site": "Y", "zone": "z2", "hazard": "cyclone", "status": "no model (no tropical cyclone exposure)"}]
    sp = tmp_path / "s.jsonl"; sp.write_text("\n".join(json.dumps(r) for r in rows))
    out = VP.build(str(sp), None, str(tmp_path / "o"), seeds=20)
    md = (tmp_path / "o" / "validation-pack.md").read_text()
    assert "26 of 26" in md and "Y (z2) / rain" in md and "Not covered by this sweep" in md and "storm surge" in md.lower()
    assert out["n_failures"] == 1 and (tmp_path / "o" / "validation-pack.html").read_text().count("<table>") >= 3


def test_portfolio_api_validates_input_and_serves_csv(monkeypatch, tmp_path):
    import time
    from fastapi.testclient import TestClient
    from cri.api.main import app
    from cri.resident import agent as A
    monkeypatch.setenv("CRI_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(PF, "screen", lambda assets, hz, store=None, progress=None, runner=None: {
        "assets": [dict(a, cell=list(PF.cell(a["lat"], a["lon"]))) for a in assets], "hazards": list(hz), "results": {}, "cells": 1, "runs": 2, "reused": 0})
    c = TestClient(app)
    assert c.post("/resident/portfolio/run", json={"csv": "name,lat,lon\nA,95,10\n"}).status_code == 422
    j = c.post("/resident/portfolio/run", json={"csv": "name,lat,lon\nA,23.8,90.4\nB,23.81,90.41\n", "hazards": ["rain", "heat", "flood"]}).json()
    for _ in range(40):
        s = c.get(f"/resident/job/{j['job']}").json()
        if s["status"] != "running":
            break
        time.sleep(0.25)
    assert s["status"] == "done" and "2 assets in 1 grid cells" in s["reply"]
    url = next(b for b in s["blocks"] if b["type"] == "files")["items"][0]["url"]
    r = c.get(url)
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/csv") and "Asset" in r.text
