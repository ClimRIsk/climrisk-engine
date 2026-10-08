"""Offline tests for sea level, surge, station check and future-climate delta methods (no network)."""
import numpy as np

from cri.modellab import coastal as C, futureclimate as FC, slr, stationcheck as SK, surge


def test_geometry_round_trip():
    la, lo = C.destination(10.0, 20.0, 90.0, 111.19)
    assert abs(la - 10.0) < 0.01 and abs(lo - 21.0) < 0.02
    assert abs(float(C.haversine_km(0, 0, 0, 1)) - 111.19) < 0.1


def test_surge_benchmarks_all_pass():
    assert all(b["ok"] for b in surge.benchmarks())


def test_slr_benchmarks_all_pass():
    assert all(b["ok"] for b in slr.benchmarks())


def test_loo_skill_detects_no_signal():
    rng = np.random.default_rng(1)
    ib = rng.uniform(0, 0.5, 40); sw = rng.uniform(0.2, 2, 40); obs = rng.uniform(0.2, 2.5, 40)   # unrelated to the model
    assert surge.loo_skill(ib, sw, obs)["skill"] < 0.2


def test_gap_year_patch_fills_only_missing_years():
    import datetime as dt
    rows = [{"date": dt.datetime(2008, 9, 13), "tot": 2.9, "storm": "Ike"}, {"date": dt.datetime(2005, 8, 29), "tot": 9.9, "storm": "X"}]
    ys, x, added = surge.patch_gap_years([2005, 2006, 2007, 2009], [1.0, 0.8, 0.7, 0.9], rows, lambda y: 0.1)
    assert ys == [2005, 2006, 2007, 2008, 2009] and len(added) == 1 and added[0][0] == 2008
    assert abs(x[3] - 2.8) < 1e-9 and x[0] == 1.0                      # 2005 already complete: untouched


def test_cyclone_scaling_hand_values():
    r = FC.cyclone_scaling({50: 60.0, 100: 70.0})
    row = [n for n in r["numbers"] if n["ssp"] == "ssp585"][0]
    assert abs(row["wind_change"] - 0.05 * 3.55 / 2.0) < 1e-12
    assert r["rows"][0][3].startswith("6")


def test_drought_projection_dry_climate_means_more_drought_days():
    n = 365 * 8
    rng = np.random.default_rng(0)
    P = rng.exponential(2.0, n) * (rng.random(n) < 0.4); E0 = np.full(n, 3.0); doy = np.arange(n) % 365 + 1.0

    def sim(P_, E_, d_):
        s = np.zeros(n); cur = 0.5
        for i in range(n):
            cur = min(1.0, max(0.0, cur + 0.02 * P_[i] - 0.02 * E_[i] * cur)); s[i] = cur
        return s
    dr = {"cell": [0, 0], "ssp": {"ssp245": {"2040-2059": {"dT": 2.0, "P": 0.8, "dT_rng": (1, 3), "P_rng": (0.7, 0.9)}}}}
    out = FC.drought_projection(sim, P, E0, doy, 365, dr)
    assert out["numbers"][0]["by_et"]["0.03"][1] > 0.10 and out["numbers"][0]["by_et"]["0.03"][0] < 0
    wet = {"cell": [0, 0], "ssp": {"ssp245": {"2040-2059": {"dT": 0.0, "P": 1.0, "dT_rng": (0, 0), "P_rng": (1, 1)}}}}
    same = FC.drought_projection(sim, P, E0, doy, 365, wet)
    assert abs(same["numbers"][0]["by_et"]["0.03"][1] - 0.10) < 0.01   # no change in climate: 10 % by construction


def test_fire_projection_warming_raises_fwi():
    n = 365 * 3; rng = np.random.default_rng(2)
    t = 25 + 8 * np.sin(np.arange(n) / 58) + rng.normal(0, 2, n); rh = np.clip(45 - 10 * np.sin(np.arange(n) / 58) + rng.normal(0, 5, n), 8, 95)
    rain = rng.exponential(3, n) * (rng.random(n) < 0.15); wind = np.full(n, 20.0)
    months = [((i % 365) // 31) % 12 + 1 for i in range(n)]; years = [2000 + i // 365 for i in range(n)]
    dr = {"cell": [0, 0], "ssp": {"ssp585": {"2080-2099": {"dT": 4.0, "P": 1.0, "dT_rng": (3, 5), "P_rng": (1, 1)}}}}
    out = FC.fire_projection(t, rh, wind, rain, months, years, dr, False)
    nm = out["numbers"][0]
    assert nm["ann_const_rh"] > out["baseline_annual_max"] and nm["ann_const_q"] >= nm["ann_const_rh"]


def test_station_check_with_mocked_download():
    # a station whose annual maxima track the grid: verdict consistent
    rows = ["STATION,NAME,DATE,PRCP"]
    rng = np.random.default_rng(5); ys = list(range(1990, 2012)); g = {}
    for y in ys:
        peak = 40 + 10 * rng.random(); g[y] = peak * 0.95
        for d in range(300):
            v = peak if d == 10 else 2.0 * rng.random()
            rows.append(f"X,Site,{y}-{1 + d // 28:02d}-{1 + d % 28:02d},{v:.1f}")
    txt = "\n".join(rows)
    one = SK._one("rain", {"id": "X", "name": "Site", "km": 5.0}, ys, [g[y] for y in ys], lambda url: txt, None)
    assert one["n"] == 22 and one["rho"] > 0.9 and 0.6 < one["ratio50"] < 1.7


def test_hargreaves_hand_value_and_power_drought_data():
    from cri.modellab import studies_series as SS
    e = float(SS.hargreaves_et0([30.0], [20.0], [80.0], 0.0)[0])          # equator near equinox: Ra ≈ 36 MJ/m²/day -> 0.0023*(25+17.8)*sqrt(10)*36*0.408 ≈ 4.6 mm
    assert 4.3 < e < 4.9
    assert float(SS.hargreaves_et0([30.0], [30.0], [80.0], 0.0)[0]) == 0.0   # no temperature range, no demand
    days = [f"2000{m:02d}{d:02d}" for m in (1, 2) for d in range(1, 29)]
    par = {k: {dd: v for dd in days} for k, v in (("PRECTOTCORR", 1.0), ("GWETROOT", 0.5), ("T2M_MAX", 20.0), ("T2M_MIN", 10.0))}
    out = SS.power_drought_data(51.5, 0.0, fetch=lambda url: {"properties": {"parameter": par}})
    dd = out["daily"]
    assert len(dd["time"]) == 56 and dd["soil_moisture_7_to_28cm_mean"][0] == 0.5 and dd["et0_fao_evapotranspiration"][0] > 0
