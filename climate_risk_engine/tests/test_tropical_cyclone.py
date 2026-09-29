"""Offline tests for the global tropical-cyclone wind model (uses the packaged IBTrACS file)."""
import numpy as np
import pytest

from cri.climate import tropical_cyclone as tc
from cri.analysis import asset_intelligence as ai


def test_track_file_ships_and_covers_all_basins():
    T = tc.tracks()
    assert len(T["sid"]) > 4000 and T["years"] >= 45
    assert {"NA", "EP", "WP", "NI", "SI", "SP"} <= {str(T["basins"][b]) for b in np.unique(T["basin"])}


def test_holland_profile_peaks_at_rmax():
    r = np.linspace(1, 300, 600)
    v = tc.holland_wind(r, 50.0, 30.0, 1.5)
    assert abs(r[np.argmax(v)] - 30.0) < 2.0
    assert v.max() == pytest.approx(50.0, rel=0.01)


def test_known_landfalls_are_the_worst_storm():
    assert tc.site_hazard(11.24, 125.0).worst["name"] == "Haiyan"          # Tacloban
    assert tc.site_hazard(30.07, -85.58).worst["name"] == "Michael"        # Tyndall AFB
    assert tc.site_hazard(-19.83, 34.84).worst["name"] == "Idai"           # Beira


def test_no_cyclone_hazard_outside_the_belts():
    h = tc.site_hazard(51.5, -0.12)                                         # London
    assert h.events_ge_ts == 0 and h.rp_wind_ms[100] == 0.0


def test_return_periods_increase():
    rp = tc.site_hazard(25.77, -80.19).rp_wind_ms                           # Miami
    assert rp[10] < rp[50] < rp[100] < rp[250]


def test_validation_ships_with_calibration():
    v = tc.validation()
    assert 0.4 < v["calibration"]["k_land_exposure"] < 0.8
    assert abs(v["model_calibrated_heldout"]["median_ratio_model_over_obs"] - 1) < 0.15


def test_finding_severity_and_evidence():
    f = ai._tropical_cyclone(11.24, 125.0)
    assert f["hazard"] == "tropical_cyclone" and f["severity"] in ("HIGH", "CRITICAL")
    assert any("Haiyan" in str(e["value"]) for e in f["evidence"])
    assert "Validated" in f["caveat"]
    assert ai._tropical_cyclone(51.5, -0.12)["severity"] == "NEGLIGIBLE"
