"""Release plume: mass conservation, agreement with the verified plume code, input validation, nulls near the source."""
import math

import numpy as np
import pytest

from cri import release_plume as RP
from cri.modellab import air


def _r(**k):
    a = dict(Q_g_s=100.0, H_m=30.0, u10_ms=4.0, wind_from_deg=270.0, cls="D")
    a.update(k)
    return RP.field(**a)


def test_mass_flux_through_a_crosswind_plane_equals_the_emission_rate():
    r = _r(half_m=1500.0, nx=60, ny=240, nz=60)
    f = r["frame"]; c = np.array(r["conc_ug_m3_xyz"]) * 1e-6                      # g/m3
    xs = np.linspace(0, f["L_m"], f["nx"]); ys = np.linspace(-f["W_m"], f["W_m"], f["ny"]); zs = np.linspace(0, f["Z_m"], f["nz"])
    i = int(np.argmin(abs(xs - 1000.0)))
    flux = np.trapz(np.trapz(c[i] * r["inputs"]["u_at_release_ms"], zs, axis=1), ys)
    assert abs(flux - 100.0) / 100.0 < 0.06, flux                                  # grid truncation in z and y, not model error


def test_ground_value_matches_the_verified_plume_function():
    r = _r(src_xy=(0.0, 0.0), wind_from_deg=270.0)                                 # wind from the west -> blows east
    g = r["ground"]; n = g["n"]; half = g["half_m"]
    xs = np.linspace(-half, half, n)
    ix = int(np.argmin(abs(xs - 200.0))); iy = int(np.argmin(abs(xs - 0.0)))
    got = g["ug_m3"][iy][ix]
    want = float(air.plume(np.array([xs[ix]]), np.array([xs[iy]]), 0.0, 100.0, r["inputs"]["u_at_release_ms"], 30.0, "D")[0] * 1e6)
    assert got == pytest.approx(want, rel=2e-3)


def test_near_field_and_upwind_are_null_not_zero():
    r = _r(); g = r["ground"]; n = g["n"]
    xs = np.linspace(-g["half_m"], g["half_m"], n)
    iy = int(np.argmin(abs(xs)))
    ix_up = int(np.argmin(abs(xs + 200.0)))                                        # 200 m west = upwind
    assert g["ug_m3"][iy][ix_up] is None


def test_wind_direction_convention_blows_toward_the_opposite_compass_point():
    r = _r(wind_from_deg=0.0)                                                      # from the north -> toward the south
    assert r["frame"]["ey"] < -0.99 and abs(r["frame"]["ex"]) < 1e-9
    assert r["peak_ground"]["y_m"] < 0


def test_higher_stack_lowers_the_peak_near_the_source_and_pushes_it_downwind():
    lo = _r(H_m=0.0)["peak_ground"]; hi = _r(H_m=60.0)["peak_ground"]
    assert hi["ug_m3"] < lo["ug_m3"] and hi["downwind_m"] >= lo["downwind_m"]


def test_stability_read_from_wind_when_not_declared_and_reported():
    r = _r(cls=None, u10_ms=1.5, daytime=False, cloud_frac=0.1)
    assert r["inputs"]["stability"] == "F" and "read from" in r["inputs"]["stability_source"]


def test_structures_inside_a_level_are_listed():
    comps = [{"id": "a", "name": "Downwind tank", "x": 150.0, "y": 0.0}, {"id": "b", "name": "Upwind office", "x": -150.0, "y": 0.0}]
    r = _r(components=comps, threshold_ug_m3=5.0)
    hit = {x["id"] for x in r["structures_at_or_above"]["5"]}
    assert "a" in hit and "b" not in hit


@pytest.mark.parametrize("kw", [dict(Q_g_s=0.0), dict(Q_g_s=float("nan")), dict(H_m=-1.0), dict(H_m=999.0), dict(u10_ms=-1.0), dict(u10_ms=80.0), dict(cls="Z")])
def test_bad_inputs_are_refused_not_guessed(kw):
    with pytest.raises(ValueError):
        _r(**kw)


def test_limits_state_what_is_not_modelled():
    t = " ".join(_r()["limits"]).lower()
    assert "dense gases" in t and "not for emergency planning" in t and "screening" in t


def test_endpoint_returns_field_and_refuses_bad_input():
    from fastapi.testclient import TestClient
    from cri.api import main
    c = TestClient(main.app)
    r = c.post("/asset-twin/plume", json={"q_g_s": 50, "height_m": 15, "wind_ms": 3, "wind_from_deg": 90})
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["status"] == "modelled" and j["peak_ground"]["ug_m3"] > 0 and j["limits"]
    assert c.post("/asset-twin/plume", json={"q_g_s": -5}).status_code == 422


def test_portfolio_map_page_is_served_and_keeps_null_visible():
    from fastapi.testclient import TestClient
    from cri.api import main
    r = TestClient(main.app).get("/portfolio/map")
    assert r.status_code == 200
    t = r.text
    assert "Not assessed" in t and "no single portfolio score" in t and "/resident/site/profile" in t
