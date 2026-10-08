"""Model Lab v2: every hazard's physics against hand values and limits, hindcast tests that must pass on stationary synthetic records and fail on broken ones, and routing."""
import math
from types import SimpleNamespace as NS

import numpy as np
import pytest

from cri.modellab import (bucket, counts, cyclone, dossier, evt, fire, hazards, heat, landslide, lab, seismic, studies_evt, studies_seismic, studies_series)


def _ok(benches):
    bad = [b["name"] for b in benches if not b["ok"]]
    assert not bad, bad


# ── building blocks ──────────────────────────────────────────────────────────

def test_all_physics_benchmarks_pass():
    for fn in (evt.run_benchmarks, heat.run_benchmarks, fire.run_benchmarks, bucket.run_benchmarks, seismic.run_benchmarks, cyclone.bench_holland, landslide.run_benchmarks):
        _ok(fn())


def test_fire_isi_hand_value_and_heat_magnus():
    assert fire.isi(90.0, 10.0) == pytest.approx(7.09, abs=0.1)
    assert heat.sat_vapour_pressure_hpa(20.0) == pytest.approx(23.4, abs=0.2)
    assert heat.wbgt_shade(30.0, 70.0) == pytest.approx(0.567 * 30 + 0.393 * heat.vapour_pressure_hpa(30.0, 70.0) + 3.94, abs=1e-6)


def test_gumbel_return_level_formula_and_recovery():
    rng = np.random.default_rng(1)
    x = rng.gumbel(50, 10, 4000)
    mu, beta = evt.gumbel_fit(x)
    assert mu == pytest.approx(50, abs=0.8) and beta == pytest.approx(10, abs=0.6)
    p = evt.best_model(x)["params"]
    assert evt.level(p, 100) == pytest.approx(50 - 10 * math.log(-math.log(1 - 0.01)), abs=2.0)


def test_hindcast_passes_stationary_and_fails_on_a_regime_shift():
    ok = 0
    for s in range(120):
        x = np.random.default_rng(100 + s).gumbel(40, 8, 74)
        ok += int(evt.hindcast(x)["passed"])
    assert ok >= 98                                                    # false-fail rate stays small
    x = np.concatenate([np.random.default_rng(1).gumbel(40, 8, 50), np.random.default_rng(2).gumbel(70, 8, 24)])
    h = evt.hindcast(x)
    assert h["ok"] and not h["passed"]


def test_trend_test_detects_a_trend_but_not_noise():
    yrs = np.arange(1950, 2024)
    rng = np.random.default_rng(3)
    assert evt.trend_test(yrs, rng.gumbel(40, 8, len(yrs)) + 0.3 * (yrs - 1950))["significant"]
    assert not evt.trend_test(yrs, np.random.default_rng(4).gumbel(40, 8, len(yrs)))["significant"]


def test_poisson_rate_hindcast_flags_a_rate_change():
    r = np.random.default_rng(0)
    stable = counts.rate_hindcast(r.poisson(5.0, 50))
    jump = counts.rate_hindcast(np.concatenate([r.poisson(3.0, 33), r.poisson(9.0, 17)]))
    assert stable["p"] > 0.01 and jump["p"] < 0.01


# ── studies on injected data (no network) ────────────────────────────────────

def test_evt_study_validated_on_stationary_series_and_flags_trend():
    yrs = list(range(1950, 2024))
    x = np.random.default_rng(7).gumbel(60, 12, len(yrs))
    r = studies_evt.run("rain", 23.8, 90.4, "Test", series=(yrs, x), bootstrap=40)
    assert r["ok"] and r["domain"] == "rain" and "FAILED" not in r["status"]
    assert r["numbers"]["levels"][100] > r["numbers"]["levels"][10]
    xt = x + 0.5 * np.arange(len(x))
    r2 = studies_evt.run("rain", 23.8, 90.4, "Test", series=(yrs, xt), bootstrap=40)
    assert r2["numbers"]["trend"]["significant"]


def test_seismic_study_on_synthetic_gutenberg_richter_catalogue():
    rng = np.random.default_rng(9)
    n = 3000
    t = np.sort(1973 + 52 * rng.random(n))
    mag = np.round(4.0 + rng.exponential(1.0 / math.log(10), n), 1)
    ang = rng.random(n) * 2 * math.pi; rr = 250 * np.sqrt(rng.random(n)) / 111.0
    cat = {"t": t, "mag": mag, "lat": 35 + rr * np.sin(ang), "lon": 139 + rr * np.cos(ang), "depth": np.full(n, 10.0), "source": "synthetic"}
    r = studies_seismic.run(35.0, 139.0, "Synthetic", catalog=cat)
    assert r["ok"] and abs(r["numbers"]["b"] - 1.0) < 0.15
    assert "FAILED" not in r["status"]
    assert 0 < r["numbers"]["levels"]["475"] < r["numbers"]["levels"]["2475"] < 2.0
    sparse = {k: (v[:5] if hasattr(v, "__len__") and not isinstance(v, str) else v) for k, v in cat.items()}
    assert studies_seismic.run(35.0, 139.0, "Sparse", catalog=sparse)["status"].startswith("no model")


def test_seismic_declustering_removes_aftershocks():
    t = np.array([2000.0, 2000.001, 2000.002, 2010.0]); mag = np.array([7.0, 5.0, 4.5, 5.5])
    lat = np.array([0.0, 0.05, 0.02, 3.0]); lon = np.zeros(4)
    keep = seismic.decluster_gk(t, mag, lat, lon)
    assert list(keep) == [True, False, False, True]


def test_cyclone_study_uses_bundled_tracks():
    r = cyclone.run(22.3, 91.8, "Chattogram")
    assert r["ok"] and r["numbers"]["n_storms"] > 50 and r["numbers"]["rates"]["tropical storm"] > 0
    far = cyclone.run(52.0, 10.0, "Hamburg")                       # no tropical cyclones reach here
    assert far["numbers"]["n_storms"] == 0 and not far["numbers"]["levels"]


def test_landslide_study_on_a_steep_synthetic_hillside():
    ny = nx = 40
    z = np.fromfunction(lambda i, j: 600 + 0.55 * 30.0 * j, (ny, nx))          # 29° slope across the grid
    dem = NS(elevations=z, cell_size_m=30.0, asset_row=20, asset_col=20)
    rain = lambda la, lo: {"maxima_mm": [120, 150, 180, 200, 220, 160, 140, 170, 190, 210], "years": list(range(2000, 2010)), "source": "test"}
    r = landslide.run(0.0, 0.0, "Hill", dem=dem, rain_fetch=rain, c_kpa=0.0, cn=60.0)
    n = r["numbers"]
    assert n["frac_unstable_dry"] < 0.1 and n["frac_unstable_wet"] > 0.9          # cohesionless 29° slope: stable dry (tan30/tan29 > 1), fails once pore pressure builds
    assert "verified only" in r["status"]


def test_drought_and_river_twin_experiments_recover_the_water_balance():
    rng = np.random.default_rng(5)
    n = 365 * 9
    P = np.where(rng.random(n) < 0.35, rng.gamma(1.4, 9.0, n), 0.0) * (1 + 0.8 * np.sin(2 * np.pi * np.arange(n) / 365.25 - 1.0) ** 2)
    E0 = 3.5 + 1.8 * np.sin(2 * np.pi * np.arange(n) / 365.25 - 1.5)
    S, ET, D, Q = bucket.soil(P, E0, 200.0, 0.9, 0.06, 2.5, 0.5, s0=0.5)
    days = [str(np.datetime64("2010-01-01") + int(i)) for i in range(n)]
    sm = 0.05 + 0.4 * S / 200.0 + rng.normal(0, 0.004, n)
    data = {"daily": {"time": days, "precipitation_sum": P.tolist(), "et0_fao_evapotranspiration": E0.tolist(), "soil_moisture_7_to_28cm_mean": sm.tolist()}}
    d = studies_series.drought(23.8, 90.4, "Twin", data=data)
    assert d["ok"] and d["numbers"]["validation"]["r_anom"] > 0.5
    flow = bucket.route(Q, D, 0.3, 0.1, 40.0)
    r = studies_series.river(23.8, 90.4, "Twin", data=data, flow={"series": dict(zip(days, flow.tolist()))})
    assert r["ok"] and r["numbers"]["validation"]["nse"] > 0.5


# ── routing and the dossier ──────────────────────────────────────────────────

@pytest.mark.parametrize("text,key", [("build a flood model for Chattogram", "flood"), ("earthquake hazard for Tokyo", "earthquake"), ("extreme heat model for Dhaka", "heat"), ("river flood model for Dhaka", "river"),
                                      ("air quality Delhi", "air"), ("landslide risk near Kathmandu", "landslide"), ("extreme rainfall for Mumbai", "rain"), ("hurricane risk Miami", "cyclone"),
                                      ("drought model Nairobi", "drought"), ("wildfire Sydney", "fire"), ("windstorm Hamburg", "wind"), ("storm surge model for Khulna", "surge"), ("tsunami for Chennai", "tsunami")])
def test_detects_every_hazard(text, key):
    assert hazards.detect(text) == key


def test_no_hazard_in_unrelated_text():
    assert hazards.detect("write me a poem") is None


def test_unmodelled_hazard_gives_dossier_not_numbers(tmp_path, monkeypatch):
    monkeypatch.setenv("CRI_DATA_DIR", str(tmp_path)); monkeypatch.setenv("CRI_STORY_OFFLINE", "1"); monkeypatch.setenv("CRI_LLM", "0")
    out = lab.run_study("tsunami", 13.0, 80.3, "Chennai", write=True)
    r = out["result"]
    assert r["status"].startswith("no model") and r["numbers"] == {}
    html = open(str(lab.lab_dir() / out["files"][0]["url"].split("/")[-1]), encoding="utf-8").read()
    assert "does not guess" in html and "GEBCO" in html


def test_generic_study_renders_a_report_and_chat_lists_all_hazards(tmp_path, monkeypatch):
    monkeypatch.setenv("CRI_DATA_DIR", str(tmp_path)); monkeypatch.setenv("CRI_STORY_OFFLINE", "1"); monkeypatch.setenv("CRI_LLM", "0")
    out = lab.run_study("cyclone", 22.3, 91.8, "Chattogram", write=True)
    assert out["files"] and out["story"]["title"]
    rows = lab.summary(out["result"])
    assert rows[0][0] == "Status" and any("storm" in r[0] for r in rows)
    from cri.resident import agent as AG
    ag = AG.Agent(twin_builder=lambda **k: {}, analyser=lambda t: {}, scanner=lambda **k: {}, lab=lambda *a: {})
    cap = ag.handle("s", "what can the model lab do?")
    txt = (cap["reply"] + str(cap["blocks"])).lower()
    assert "earthquake" in txt and "landslide" in txt and "cyclone" in txt
    hz = ag.handle("s", "earthquake hazard")
    assert "Which place" in hz["reply"] or hz.get("job")


# ── precision across the world: what the global sweep taught the engine ──────

def test_step_change_separates_a_data_step_from_a_trend():
    yrs = np.arange(1979, 2024)
    rng = np.random.default_rng(11)
    step = np.where(yrs < 2000, 30.0, 45.0) + rng.normal(0, 3, len(yrs))
    trend = 30.0 + 0.4 * (yrs - 1979) + rng.normal(0, 3, len(yrs))
    s1, s2 = evt.step_change(yrs, step), evt.step_change(yrs, trend)
    assert s1["step_like"] and abs(s1["year"] - 2000) <= 2
    assert not s2["step_like"]


def test_a_failed_hindcast_with_a_small_effect_is_called_minor_not_failed():
    yrs = list(range(1979, 2024))
    rng = np.random.default_rng(21)
    x = rng.gumbel(12.0, 0.6, len(yrs)) + 0.025 * np.arange(len(yrs))      # significant trend, but ~1 m/s over 45 years on a 12 m/s base
    r = studies_evt.run("wind", 1.0, 100.0, "Equator", series=(yrs, x), bootstrap=30)
    if not r["numbers"]["hindcast"]["passed"]:
        assert "practically minor" in r["status"] and abs(r["numbers"]["drift50"]) < 0.10
    big = np.concatenate([rng.gumbel(40, 8, 30), rng.gumbel(80, 8, 15)])
    r2 = studies_evt.run("rain", 1.0, 100.0, "Shift", series=(yrs, big), bootstrap=30)
    assert "FAILED" in r2["status"] and r2["numbers"]["step"]["step_like"]


def test_no_cyclone_exposure_is_stated_not_modelled():
    r = cyclone.run(52.0, 10.0, "Hamburg")
    assert r["status"].startswith("no model") and r["ok"] and r["numbers"]["levels"] == {}


def test_river_without_discharge_is_a_finding_not_an_error():
    days = [str(np.datetime64("2010-01-01") + i) for i in range(365 * 3)]
    data = {"daily": {"time": days, "precipitation_sum": [1.0] * len(days), "et0_fao_evapotranspiration": [3.0] * len(days)}}
    out = hazards.run_generic("river", 14.7, -17.4, "Desert", {"data": data, "flow": {"series": {}}})
    assert out["status"].startswith("no model") and out["numbers"] == {}


def test_fire_index_stays_finite_in_extreme_dry_heat():
    f = fire.ffmc(95.0, 50.0, 2.0, 120.0, 0.0)
    assert 0 <= f <= 101.0
    assert isinstance(fire.isi(101.5, 60.0), float) and isinstance(fire.isi(-5.0, 10.0), float)       # was a complex number in the Riyadh sweep


def test_hyper_arid_site_gets_no_fire_model(monkeypatch):
    monkeypatch.setitem(studies_evt.HAZ["fire"], "extract", lambda la, lo, fetch=None: (list(range(1982, 2024)), [200.0] * 42, {"days_over_30": [300] * 42, "_rain_mm": 46.0}))
    r = hazards.run_generic("fire", 24.7, 46.7, "Desert", {})
    assert r["status"].startswith("no model") and "arid" in r["standfirst"]


def test_second_data_source_takes_over_when_the_first_is_out_of_quota(monkeypatch):
    from cri.modellab import data as D

    def boom(*a, **k):
        raise D.QuotaExceeded("daily")
    monkeypatch.setitem(studies_evt.HAZ["rain"], "extract", boom)
    days = [str(np.datetime64("1981-01-01") + i) for i in range(365 * 40)]
    rng = np.random.default_rng(3)
    vals = np.where(rng.random(len(days)) < 0.3, rng.gamma(1.2, 12.0, len(days)), 0.0).tolist()
    monkeypatch.setattr(D, "power_daily", lambda lat, lon, y0, y1, params, fetch=None: {"time": days, "PRECTOTCORR": vals})
    monkeypatch.setitem(studies_evt.POWER, "rain", studies_evt._power_max("PRECTOTCORR"))
    r = studies_evt.run("rain", 1.0, 2.0, "Fallback", bootstrap=20)
    assert "NASA POWER" in r["numbers"]["source"] and r["ok"]
    assert any(n["title"] == "Data source" for n in r["notebook"])


def test_trend_adjusted_level_is_reported_when_the_trend_is_significant():
    yrs = list(range(1981, 2024))
    x = np.random.default_rng(5).gumbel(50, 6, len(yrs)) + 0.6 * np.arange(len(yrs))
    r = studies_evt.run("rain", 1.0, 2.0, "Trend", series=(yrs, x), bootstrap=20)
    lt = r["numbers"]["levels_today"]
    assert lt and lt[10] > r["numbers"]["levels"][10] * 0.98        # trend variance inflates a stationary tail, so compare the 10-year level, which is not tail-dominated
