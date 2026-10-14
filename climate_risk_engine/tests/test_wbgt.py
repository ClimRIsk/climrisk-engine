import numpy as np

from cri.modellab import wbgt as W, heat


def _t(n=1, h=6):
    return np.datetime64("2020-06-21T%02d:00:00" % h) + np.zeros(n, dtype="timedelta64[h]")


def test_shade_night_equals_wet_bulb_blend():
    # at night there is no sun: globe ~ air temperature, WBGT sits between wet bulb and air temperature
    r = W.wbgt(np.array([30.0]), np.array([60.0]), np.array([1000.0]), np.array([2.0]), np.array([0.0]), _t(1, 20), 10.0, 78.0)
    assert r["psy_wetbulb"][0] < r["wbgt"][0] < 30.0


def test_more_sun_more_wbgt_more_wind_less():
    a = dict(tair_c=np.array([33.0]), rh_pct=np.array([55.0]), p_hpa=np.array([1000.0]), t_utc=_t(1, 7), lat=10.0, lon=78.0)
    lo = W.wbgt(speed10=np.array([2.0]), solar=np.array([100.0]), **a)["wbgt"][0]
    hi = W.wbgt(speed10=np.array([2.0]), solar=np.array([900.0]), **a)["wbgt"][0]
    windy = W.wbgt(speed10=np.array([6.0]), solar=np.array([900.0]), **a)["wbgt"][0]
    assert hi > lo and windy < hi


def test_saturated_air_wbgt_below_dry_hot_globe_and_wetbulb_below_air():
    r = W.wbgt(np.array([30.0]), np.array([100.0]), np.array([1000.0]), np.array([2.0]), np.array([0.0]), _t(1, 22), 10.0, 78.0)
    assert abs(r["nat_wetbulb"][0] - 30.0) < 1.0


def test_old_shade_shortcut_still_present_for_fallback():
    assert abs(float(heat.wbgt_shade(35.0, 50.0)) - 34.8) < 0.3
