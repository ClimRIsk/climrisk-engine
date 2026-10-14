import numpy as np
import pytest

from cri.modellab import utci as U


def test_matches_reference_package_over_valid_range():
    pytc = pytest.importorskip("pythermalcomfort.models")
    rng = np.random.default_rng(3)
    t = rng.uniform(-45, 45, 400); d = rng.uniform(-25, 60, 400); v = rng.uniform(0.5, 16, 400); rh = rng.uniform(5, 100, 400)
    ref = pytc.utci(tdb=t, tr=t + d, v=v, rh=rh, round_output=False).utci
    ours = U.utci(t, rh, v, t + d)
    ok = np.isfinite(ref) & np.isfinite(ours)
    assert ok.sum() > 300 and np.max(np.abs(ref[ok] - ours[ok])) < 0.05


def test_published_checks_and_physics():
    # no radiation excess, 50 % humidity, light wind, mild air: UTCI close to air temperature
    assert abs(float(U.utci(20.0, 50.0, 0.5, 20.0)) - 20.0) < 0.6
    # sun raises it, wind lowers it in the heat, humidity raises it in the heat
    base = float(U.utci(33.0, 40.0, 1.0, 40.0))
    assert float(U.utci(33.0, 40.0, 1.0, 55.0)) > base
    assert float(U.utci(33.0, 40.0, 5.0, 40.0)) < base
    assert float(U.utci(33.0, 80.0, 1.0, 40.0)) > base
    # strong wind makes cold feel colder
    assert float(U.utci(-10.0, 70.0, 8.0, -10.0)) < float(U.utci(-10.0, 70.0, 1.0, -10.0))


def test_category_and_limits():
    assert U.category(30.0) == "moderate heat stress" and U.category(-30.0) == "very strong cold stress" and U.category(20.0) == "no thermal stress"
    assert np.isnan(U.utci(60.0, 50.0, 2.0, 60.0))


def test_mrt_equals_air_temperature_when_globe_equals_air():
    assert abs(float(U.mrt_from_globe(20.0, 20.0, 1.0)) - 20.0) < 1e-6


def test_standing_mrt_is_sane_day_and_night():
    t = np.array([33.0, 33.0, 5.0]); rh = np.array([50.0, 50.0, 70.0])
    # noon sun (global 900, mostly direct, high sun) vs night vs clear cold night
    m = U.mrt_standing(t, rh, np.array([900.0, 0.0, 0.0]), np.array([0.75, 0.0, 0.0]), np.array([0.95, 0.0, 0.0]))
    assert 45 < m[0] < 65          # sun adds roughly 15 to 30 degrees to the radiant temperature
    assert m[1] < 33.0 + 0.5       # night: the sky is colder than the ground, so radiant temperature is not above air temperature
    assert m[2] < 5.0 + 0.5


def test_hourly_utci_runs_and_orders_sun_over_shade_night():
    tt = np.datetime64("2020-06-21T06:30:00") + np.arange(3) * np.timedelta64(6, "h")      # 06:30, 12:30, 18:30 UTC at 10N 78E
    r = U.hourly_utci(np.full(3, 33.0), np.full(3, 50.0), np.full(3, 1000.0), np.full(3, 2.0), np.array([300.0, 900.0, 50.0]), tt, 10.0, 78.0)
    assert np.all(np.isfinite(r["utci"])) and r["utci"][1] > r["utci"][2]
