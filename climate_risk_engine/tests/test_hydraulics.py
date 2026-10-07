"""2D shallow-water solver and its regional forcing. Synthetic terrain only, no network."""
import numpy as np
import pytest
from types import SimpleNamespace

from cri import hydraulics as H


def test_mass_is_conserved_and_level_is_capped():
    R = C = 40
    z = (np.abs(np.arange(C) - 20)[None, :] * 0.05) * np.ones((R, 1))
    stage = np.zeros((R, C), bool); stage[:, 0] = True
    r = H.solve(z, np.zeros((R, C), bool), stage, np.where(stage, 3.0, 0.0), 10.0, t_end=3600, ramp_s=600)
    assert abs(r["volume_end_m3"] - r["volume_added_m3"]) / r["volume_added_m3"] < 1e-9
    assert float((z + r["h_final"]).max()) < 3.0 + 0.3                     # no overshoot beyond a small inertial transient


def test_still_water_stays_still():
    R = C = 30
    z = np.zeros((R, C))
    stage = np.zeros((R, C), bool); stage[:, 0] = True
    r = H.solve(z, np.zeros((R, C), bool), stage, np.where(stage, 1.0, 0.0), 10.0, t_end=7200, ramp_s=60)
    lvl = z + r["h_final"]
    assert float(lvl.max() - lvl.min()) < 0.02                              # a lake at rest is flat
    assert float(np.abs(r["frames"][-1] - r["frames"][-2]).max()) < 0.02    # and nothing changes between the last two frames


def test_a_wall_stops_water_and_arrival_increases_with_distance():
    R, C = 30, 60
    z = np.zeros((R, C))
    stage = np.zeros((R, C), bool); stage[:, 0] = True
    blk = np.zeros((R, C), bool); blk[:, 30] = True
    r = H.solve(z, blk, stage, np.where(stage, 1.0, 0.0), 10.0, t_end=3600, ramp_s=300)
    assert r["hmax"][:, 31:].max() == 0.0                                   # nothing passes a solid wall
    a = r["arrive_s"][15]
    assert a[5] < a[15] < a[25]                                             # farther cells wet later


def test_raised_crest_overtops_only_when_level_exceeds_it():
    R, C = 20, 50
    z = np.zeros((R, C)); z[:, 25] = 1.5
    stage = np.zeros((R, C), bool); stage[:, 0] = True
    low = H.solve(z, np.zeros((R, C), bool), stage, np.where(stage, 1.0, 0.0), 10.0, t_end=3600, ramp_s=300)
    high = H.solve(z, np.zeros((R, C), bool), stage, np.where(stage, 2.5, 0.0), 10.0, t_end=3600, ramp_s=300)
    assert low["hmax"][:, 30:].max() == 0.0
    assert high["hmax"][:, 30:].max() > 0.2


class _G:
    def __init__(self, z, dx):
        self.z = z; self.rows, self.cols = z.shape; self.dx = self.dy = dx; self.ac = self.cols / 2; self.ar = self.rows / 2


def _scene(wet_left=True, depth=2.0):
    n = 120; dx = 10.0
    x = (np.arange(n) - n / 2) * dx
    z = np.tile(np.linspace(0.0, 1.5, n), (n, 1))                           # gently rising to the east
    g = _G(z, dx)
    dem = _G(np.tile(np.linspace(0.0, 1.5, 60), (60, 1)), 20.0)
    m = 40
    c, r = np.meshgrid(np.arange(m), np.arange(m))
    wx = (c - m / 2) * 92.8; wy = (m / 2 - r) * 92.8
    d = np.where(wx < -300, depth, 0.0).astype(np.float32) if wet_left else np.zeros((m, m), np.float32)
    win = {"depth": d, "x": wx, "y": wy, "n": m // 2, "empty": False}
    comp = [{"id": "a", "x": 0.0, "y": 0.0, "footprint": [[-15, -15], [15, -15], [15, 15], [-15, 15]], "removed": False},
            {"id": "b", "x": 300.0, "y": 0.0, "footprint": [[-15, -15], [15, -15], [15, 15], [-15, 15]]}]
    return g, dem, win, comp


def test_simulate_routes_water_and_reports_per_component():
    g, dem, win, comp = _scene()
    sc = {"id": "t", "kind": "river", "rp": 100, "year": None, "label": "t"}
    s = H.simulate(0, 0, g, comp, sc, dem, _win=win)
    assert not s["dry"] and s["comp"]["a"]["depth_at_ground_m"] > 0
    assert s["comp"]["a"]["depth_at_ground_m"] >= s["comp"]["b"]["depth_at_ground_m"]      # downstream (higher, farther) is shallower
    assert s["comp"]["a"]["arrival_min"] <= (s["comp"]["b"]["arrival_min"] or 1e9)
    assert s["validation"]["mass_residual_pct"] < 0.1
    assert len(s["frames"]) == 8 and len(s["frames"][0]) == 120 * 120


def test_barrier_in_the_model_protects_the_inside():
    g, dem, win, comp = _scene()
    sc = {"id": "t", "kind": "river", "rp": 100, "year": None, "label": "t"}
    base = H.simulate(0, 0, g, comp, sc, dem, _win=win)
    walled = H.simulate(0, 0, g, comp, sc, dem, _win=win, barrier_m=6.0, barrier_radius_m=60.0)
    assert walled["comp"]["a"]["depth_at_ground_m"] < base["comp"]["a"]["depth_at_ground_m"]
    assert walled["comp"]["a"]["depth_at_ground_m"] < 0.05


def test_no_regional_water_gives_dry_with_reason():
    g, dem, win, comp = _scene(wet_left=False)
    s = H.simulate(0, 0, g, comp, {"id": "t", "kind": "river", "rp": 10, "year": None}, dem, _win=win)
    assert s["dry"] and "no water" in s["note"] and all(v["depth_at_ground_m"] == 0 for v in s["comp"].values())


def test_regional_water_that_does_not_reach_the_site_is_explained():
    g, dem, win, comp = _scene()
    win["depth"][:] = 0.0
    win["depth"][0, 0] = 1.0                                                # one wet pixel, ~3.5 km away
    s = H.simulate(0, 0, g, comp, {"id": "t", "kind": "river", "rp": 10, "year": None}, dem, _win=win)
    assert s["dry"] and "none of it reaches" in s["note"]
