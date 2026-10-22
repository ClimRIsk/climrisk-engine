import numpy as np
import pytest

from cri import cryosphere as C


def test_freeze_thaw_counts_definitions():
    tmin = [-5, -1, 1, -2, None, -3]
    tmax = [-1, 3, 4, 0.5, 2, -2]
    c = C.freeze_thaw_counts(tmin, tmax)
    assert c == {"days": 5, "frost_days": 4, "ice_days": 2, "freeze_thaw_days": 2}


def test_freeze_thaw_band_edges():
    assert C.freeze_thaw_band(None) is None
    assert [C.freeze_thaw_band(x) for x in (0, 9.9, 10, 39.9, 40, 79.9, 80, 200)] == \
        ["Low", "Low", "Moderate", "Moderate", "High", "High", "Very high", "Very high"]


def _fake_archive(url):
    days, tmin, tmax = [], [], []
    for y in range(2004, 2024):
        for d in range(365):
            days.append(f"{y}-01-01")   # dates only need the year prefix
            tmin.append(-2.0 if d % 2 == 0 else 3.0)
            tmax.append(4.0 if d % 2 == 0 else 8.0)
    return {"daily": {"time": days, "temperature_2m_min": tmin, "temperature_2m_max": tmax}, "elevation": 120.0}


def test_freeze_thaw_from_series(tmp_path, monkeypatch):
    monkeypatch.setenv("CRI_CACHE_DIR", str(tmp_path))
    r = C.freeze_thaw(50.0, 8.0, fetch=_fake_archive)
    assert r["freeze_thaw_days_per_year"] == 183.0 and r["frost_days_per_year"] == 183.0 and r["ice_days_per_year"] == 0.0
    assert r["band"] == "Very high" and r["complete_years"] == 20 and r["years"] == [2004, 2023]


def test_freeze_thaw_raises_instead_of_neutral(tmp_path, monkeypatch):
    monkeypatch.setenv("CRI_CACHE_DIR", str(tmp_path))
    with pytest.raises(RuntimeError):
        C.freeze_thaw(50.0, 8.1, fetch=lambda u: {"error": True})


def test_slope_aspect_orientation():
    # ground rising to the west (elevation falls to the east): slope faces east, 45 degrees at 1 m per 1 m
    cols = np.arange(20, dtype=float)
    dem = np.tile(20.0 - cols, (20, 1))
    slope, aspect = C.slope_aspect(dem, 1.0)
    assert abs(slope[10, 10] - 45.0) < 1e-6 and abs(aspect[10, 10] - 90.0) < 1e-6 and C.aspect_name(aspect[10, 10]) == "E"
    # ground falling to the south (rows go south): faces south
    rows = np.arange(20, dtype=float)[:, None]
    dem2 = np.tile(20.0 - rows, (1, 20))
    _, a2 = C.slope_aspect(dem2, 1.0)
    assert abs(a2[10, 10] - 180.0) < 1e-6


def test_release_zone_share_counts_only_release_slopes():
    # left half 40 degrees, right half flat
    dem = np.zeros((41, 41))
    for j in range(21):
        dem[:, j] = (20 - j) * np.tan(np.radians(40.0))
    r = C.release_zone_share(dem, 1.0, (20, 20), 15.0)
    assert 0.3 < r["share"] < 0.7 and r["max_slope_deg"] > 39.0
    flat = C.release_zone_share(np.zeros((41, 41)), 1.0, (20, 20), 15.0)
    assert flat["share"] == 0.0


class _Grid:
    def __init__(self, dem, cell):
        self.elevations, self.cell_size_m = dem, cell
        self.asset_row, self.asset_col = dem.shape[0] // 2, dem.shape[1] // 2
        self.source = "synthetic"


def test_avalanche_terrain_bands_and_lidar_resampling():
    steep = np.fromfunction(lambda r, c: c * np.tan(np.radians(38.0)) * 10.0, (101, 101))
    a = C.avalanche_terrain(46.0, 7.8, elevation_grid=_Grid(steep, 10.0))
    assert a["band"] == "High" and a["release_terrain_share"] > 0.9 and a["tested"]["auc_ci_95_elevation_matched"] == [0.52, 0.66]
    # the same surface sampled at 1 m must give the same answer (resampled to 10 m before slopes are taken)
    fine = np.fromfunction(lambda r, c: c * np.tan(np.radians(38.0)), (1001, 1001))
    b = C.avalanche_terrain(46.0, 7.8, elevation_grid=_Grid(fine, 1.0))
    assert b["band"] == "High" and b["cell_m"] == 10.0
    flat = C.avalanche_terrain(46.0, 7.8, elevation_grid=_Grid(np.zeros((101, 101)), 10.0))
    assert flat["band"] == "None"


def test_permafrost_lookup_and_coverage():
    assert C.permafrost(71.29, -156.79)["zone"] == "Continuous"
    assert C.permafrost(51.5, -0.12)["permafrost_probability"] == 0.0
    q = C.permafrost(-0.2, -78.5)
    assert q["covered"] is False and "northern" in q["reason"]
    assert C.permafrost(40.0, -40.0)["covered"] is False          # open ocean: not covered, never zero


def test_glof_history_is_a_record_with_distance():
    z = C.glof_history(46.02, 7.75, 50.0)
    assert z["events"] > 50 and z["nearest"][0]["distance_km"] <= z["nearest"][-1]["distance_km"] or z["events"] <= 10
    assert "not a hazard model" in z["basis"] or "Reported past events only" in z["basis"]
    assert C.glof_history(54.89, -2.93, 50.0)["events"] == 0
    assert C.glof_history(-0.2, -78.5, 50.0)["nearest"] == []


def test_cryosphere_zones_skip_warm_sites_without_network(monkeypatch):
    from cri import physical_report as pr
    called = []
    monkeypatch.setattr(C, "freeze_thaw", lambda *a, **k: called.append(1) or {})
    assert pr._cryosphere_zones(13.08, 80.27, 6.0) == [] and not called       # Chennai: cannot freeze, no request made


def test_cryosphere_zones_for_a_cold_alpine_site(monkeypatch):
    from cri import physical_report as pr
    monkeypatch.setattr(C, "freeze_thaw", lambda *a, **k: {"band": "Very high", "frost_days_per_year": 299.0, "freeze_thaw_days_per_year": 119.0, "ice_days_per_year": 179.0,
                                                          "freeze_thaw_days_p10_p90": [100.0, 130.0], "years": [2004, 2023], "source": "test"})
    monkeypatch.setattr(C, "avalanche_terrain", lambda *a, **k: {"band": "High", "release_terrain_share": 0.44, "radius_m": 150.0, "max_slope_deg": 52.0,
                                                                  "dominant_release_aspect": "S", "source": "test", "tested": C.AVALANCHE_TEST})
    z = {x["key"]: x for x in pr._cryosphere_zones(45.98, 7.78, 3000.0)}
    assert set(z) >= {"freeze_thaw", "avalanche", "permafrost", "glof"}
    assert "not a hazard model" in z["glof"]["zone"] and z["glof"]["rating"] is None
    assert "not a rating" in z["avalanche"]["zone"] and "weak test" in z["avalanche"]["title"]
    assert all(k in pr._CONTEXT_KEYS for k in z)
