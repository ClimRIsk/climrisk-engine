"""Model Lab: physics components against analytic solutions, twin experiments that must recover known parameters, and the chat path."""
import math
import time
from types import SimpleNamespace as NS

import numpy as np
import pytest

from cri.modellab import air, air_study, calib, flood, flood_study, hydrology, lab, metrics, report


# ── building blocks ──────────────────────────────────────────────────────────

def test_metrics_hand_values():
    o, s = [1, 2, 3, 4], [1, 2, 3, 6]
    assert metrics.rmse(o, s) == pytest.approx(1.0)
    assert metrics.bias(o, s) == pytest.approx(0.5)
    assert metrics.nse(o, o) == 1.0 and metrics.nse(o, [2.5] * 4) == pytest.approx(0.0)
    assert metrics.fac2([10, 10, 10, 10], [5, 20, 4, 25]) == pytest.approx(0.5)
    c = metrics.contingency([1, 1, 0, 0], [1, 0, 1, 0])
    assert c["hits"] == 1 and c["misses"] == 1 and c["false_alarms"] == 1 and c["csi"] == pytest.approx(1 / 3)


def test_scs_cn_known_value_and_effective_rain_sums():
    assert hydrology.scs_cn_runoff(100, 80) == pytest.approx(50.5, abs=0.1)          # textbook: S=63.5, Ia=12.7
    assert hydrology.scs_cn_runoff(10, 80) == 0.0                                    # below initial abstraction
    hy = hydrology.block_hyetograph(120, 6, 300)
    assert hy.sum() == pytest.approx(120)
    assert hydrology.effective_rain_series(hy, 80).sum() == pytest.approx(hydrology.scs_cn_runoff(120, 80))


def test_gumbel_roundtrip():
    rng = np.random.default_rng(0)
    x = rng.gumbel(100, 25, 4000)
    mu, beta = hydrology.gumbel_fit(x)
    assert mu == pytest.approx(100, abs=2) and beta == pytest.approx(25, abs=2)
    assert hydrology.gumbel_quantile(100, 25, 100) == pytest.approx(100 + 25 * 4.60015, abs=0.01)


def test_flood_benchmarks_pass():
    for b in flood.run_benchmarks():
        assert b["ok"], b


def test_air_benchmarks_pass():
    for b in air.run_benchmarks():
        assert b["ok"], b


def test_flood_rain_on_slope_reaches_the_analytic_equilibrium_depth():
    """Steady rain r on a plane of slope S: unit discharge at distance L is r·L, so h = (r L n / √S)^(3/5). Steeper → shallower."""
    n_c, dx, nm, r = 24, 25.0, 0.04, 20e-3 / 3600
    depths = {}
    for s0 in (0.004, 0.02):
        z = (n_c - 1 - np.arange(n_c))[:, None] * dx * s0 * np.ones((1, 6))
        res = flood.simulate(z, dx, 4 * 3600, rain_m_s=r, manning_n=nm, open_boundary="S", dt_max=10.0)
        row = n_c - 4
        L = (row + 1) * dx
        expect = (r * L * nm / math.sqrt(s0)) ** 0.6
        got = float(res.h_end[row, 3])
        assert got == pytest.approx(expect, rel=0.12), (s0, got, expect)
        depths[s0] = got
    assert depths[0.02] < depths[0.004]


def test_oat_sensitivity_recovers_exponents():
    rows = calib.oat_sensitivity(lambda p: p["a"] ** 2 * p["b"], {"a": 2.0, "b": 3.0}, {"a": (1.9, 2.1), "b": (2.9, 3.1)})
    el = {r["param"]: r["elasticity"] for r in rows}
    assert el["a"] == pytest.approx(2.0, abs=0.05) and el["b"] == pytest.approx(1.0, abs=0.02)


# ── studies ──────────────────────────────────────────────────────────────────

def _valley():
    n = 100
    y, x = np.mgrid[0:n, 0:n]
    z = 2 + 0.015 * y + 0.00045 * ((x - 50) * 30) ** 2 / 30
    return NS(elevations=z, cell_size_m=30.0, asset_row=70, asset_col=50)


def _rain(seed=1):
    return {"years": list(range(1991, 2021)), "maxima_mm": list(np.random.default_rng(seed).gumbel(120, 35, 30)), "source": "synthetic"}


def test_flood_study_runs_verifies_and_is_honest_about_validation():
    r = flood_study.run(22.26, 91.8, "Valley", dem=_valley(), rain_fetch=lambda a, b: _rain(), reference_depth=lambda *a: -1.0)
    assert r["ok"] and r["status"] == "verified only"
    assert r["out"]["site_depth_m"] > 0 and r["sensitivity"] and not r["validation"]["done"] and not r["calibration"]["done"]
    assert all(v["ok"] for v in r["verification"])
    assert any("NOT validated" in n["detail"] for n in r["notebook"])


def test_flood_study_twin_experiment_recovers_roughness():
    truth = flood_study.run(22.26, 91.8, "Valley", dem=_valley(), rain_fetch=lambda a, b: _rain(), manning=0.085, sensitivity=False, reference_depth=lambda *a: -1.0)
    g = truth["grids"]; f = flood_study.plan(1.5, 30.0)["factor"]
    pts = [{"row": i * f, "col": j * f, "depth_m": float(g["h_max"][i, j])} for (i, j) in [(27, 25), (30, 24), (32, 26), (35, 25), (25, 25), (22, 24), (40, 25)] if g["h_max"][i, j] > 0.02]
    assert len(pts) >= 4
    r = flood_study.run(22.26, 91.8, "Valley", dem=_valley(), rain_fetch=lambda a, b: _rain(), manning=0.05, observed={"points": pts}, sensitivity=False, reference_depth=lambda *a: -1.0)
    assert r["calibration"]["done"] and r["validation"]["done"]
    assert 0.07 <= r["calibration"]["best"] <= 0.1 and r["validation"]["rmse_m"] < 0.08
    assert r["status"] == "verified, calibrated and validated"


def _city(n=21 * 24, seed=2, E0=0.06, Cb=18.0, amp=0.5, ph=8.0):
    rng = np.random.default_rng(seed)
    hrs = np.arange(n) % 24
    H = np.maximum(300 + 900 * np.clip(np.sin((hrs - 6) / 24 * 2 * math.pi), 0, None) + rng.normal(0, 30, n), 100)
    u = np.maximum(2.5 + rng.normal(0, 0.6, n).clip(-1.5, 2), 0.5)
    E = E0 * (1 + amp * np.cos(2 * math.pi * (hrs - ph) / 24))
    c = air.box_series(E, H, u, 20000, Cb, 0.002, c0=Cb) * (1 + rng.normal(0, 0.05, n))
    rows = [{"t": f"2026-09-{1 + i // 24:02d}T{i % 24:02d}:00", "wind": float(u[i]), "blh": float(H[i]), "cloud": 0.3, "day": 6 <= i % 24 <= 18, "pm25": float(c[i])} for i in range(n)]
    return {"rows": rows, "source": "synthetic"}


def test_air_study_twin_experiment_recovers_the_truth_and_selects_a_diurnal_model():
    r = air_study.run(28.6, 77.2, "Synthetic city", data=_city())
    assert r["ok"] and r["structure"] in ("M1", "M2")
    p = r["params"]
    assert p["E"] == pytest.approx(0.06, rel=0.2) and p["Cb"] == pytest.approx(18.0, rel=0.15) and p["amp"] == pytest.approx(0.5, abs=0.2)
    v = r["validation"]
    assert abs(v["nmb"]) < 0.05 and v["fac2"] > 0.95 and v["aqi_category_hit"] >= 0.75
    assert "not stations" in r["status"] or "no skill" in r["status"] or "ground stations" in r["status"]


def test_air_study_reports_no_skill_for_noise():
    d = _city()
    rng = np.random.default_rng(9)
    for row in d["rows"]:
        row["pm25"] = float(rng.uniform(5, 60))
    r = air_study.run(28.6, 77.2, "Noise city", data=d)
    assert not r["skilful"] and "no skill" in r["status"]


def test_reports_render_and_state_the_status():
    from cri import story as S
    rf = flood_study.run(22.26, 91.8, "Valley", dem=_valley(), rain_fetch=lambda a, b: _rain(), reference_depth=lambda *a: 0.6, sensitivity=True)
    st = report.flood_story(rf, "", [{"key": "b0", "title": "Shallow water equations", "url": "https://en.wikipedia.org/wiki/Shallow_water_equations", "text": "Shallow water equations: a set of equations."}])
    h = S.html(st)
    assert "verified only" in h and "Lake at rest" in h and "Shallow water equations" in h and "Elasticity" in h
    assert len(S.pdf(st)) > 20000
    ra = air_study.run(28.6, 77.2, "Synthetic city", data=_city())
    h2 = S.html(report.air_story(ra))
    assert "Competing models, one winner" in h2 and "Gaussian plume" in h2


def test_detect_domain_and_requests():
    assert lab.detect_domain("build a flood model for Chattogram") == "flood"
    assert lab.detect_domain("air quality model for Delhi") == "air"
    assert lab.detect_domain("what is the AQI") == "air"
    assert lab.detect_domain("write a LinkedIn post") is None


# ── chat ─────────────────────────────────────────────────────────────────────

def test_chat_runs_a_study_as_a_job(tmp_path, monkeypatch):
    monkeypatch.setenv("CRI_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("CRI_STORY_OFFLINE", "1")
    monkeypatch.setenv("CRI_LLM", "0")
    from cri.resident import agent as AG
    monkeypatch.setattr(flood_study, "hydrology_fetch", lambda a, b: _rain())
    from cri.connectors import terrain_tiles
    monkeypatch.setattr(terrain_tiles, "get_elevation_grid", lambda lat, lon, radius_km=1.5, **k: _valley())
    monkeypatch.setattr("cri.climate.flood_depth.get_flood_depth", lambda *a, **k: -1.0)
    ag = AG.Agent(twin_builder=lambda **k: {}, analyser=lambda t: {}, scanner=lambda **k: {}, lab=lambda *a: {})
    r = ag.handle("s", "build a flood model for 22.2645, 91.805")
    assert r["job"] and r["intent"] == "modellab" and "flood" in r["reply"]
    for _ in range(600):
        j = AG.job_status(r["job"])
        if j["status"] != "running":
            break
        time.sleep(0.2)
    assert j["status"] == "done", j
    files = next(b for b in j["blocks"] if b["type"] == "files")["items"]
    html = (AG.report_dir() / files[0]["url"].rsplit("/", 1)[1]).read_text(encoding="utf-8")
    assert "Verification" in html and any(b["type"] == "table" for b in j["blocks"])
    # a plain question is not a build request
    q = ag.handle("s", "what is a flood model?")
    assert q.get("intent") != "modellab"
    cap = ag.handle("s", "what can the model lab do?")
    assert "Model Lab" in cap["reply"]
    ask = ag.handle("s", "build an air quality model")
    assert "Which place" in ask["reply"] or ask.get("job")


def test_chat_routes_a_bare_model_request_and_ignores_questions(tmp_path, monkeypatch):
    monkeypatch.setenv("CRI_DATA_DIR", str(tmp_path)); monkeypatch.setenv("CRI_STORY_OFFLINE", "1"); monkeypatch.setenv("CRI_LLM", "0")
    from cri.resident import agent as AG
    from cri.modellab import data as D
    monkeypatch.setattr(D, "hourly_met_and_pm", lambda lat, lon, days=60, fetch=None: _city())
    ag = AG.Agent(twin_builder=lambda **k: {}, analyser=lambda t: {}, scanner=lambda **k: {}, lab=lambda *a: {})
    r = ag.handle("s", "air quality model for 28.6139, 77.2090")
    assert r.get("intent") == "modellab" and r["job"]
    for _ in range(300):
        j = AG.job_status(r["job"])
        if j["status"] != "running":
            break
        time.sleep(0.2)
    assert j["status"] == "done", j
    assert any("Chosen structure" in str(b.get("rows")) for b in j["blocks"] if b["type"] == "table")
    for q in ("tell me about the flood model", "is air quality modelling hard?", "what is a box model for air pollution?"):
        assert ag.handle("s", q).get("intent") != "modellab", q


def test_flood_conditioning_removes_single_cell_pits_from_the_flooded_area():
    z = _valley().elevations.copy()
    rng = np.random.default_rng(4)
    z = z + rng.normal(0, 0.8, z.shape)                                   # speckle: the noise that made the Chattogram map look like static
    dem = NS(elevations=z, cell_size_m=30.0, asset_row=70, asset_col=50)
    r = flood_study.run(22.26, 91.8, "Noisy valley", dem=dem, rain_fetch=lambda a, b: _rain(), reference_depth=lambda *a: -1.0, sensitivity=False)
    assert any(n["step"] == "1b" for n in r["notebook"]) and r["ok"]
    big = flood_study._clusters(np.array([[1, 0, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 0, 0]], bool), 4)
    assert big.sum() == 4 and not big[0, 0]


def test_open_water_is_an_outlet_not_a_flood():
    # coastal valley: the northern third is sea at 0 m
    d = _valley(); z = d.elevations.copy(); z[:30, :] = 0.0
    dem = NS(elevations=z, cell_size_m=30.0, asset_row=70, asset_col=50)
    sea = flood_study._open_water(flood_study._block_mean(z, 2))
    assert sea[:14].all() and not sea[18:].any()
    r = flood_study.run(22.26, 91.8, "Coast", dem=dem, rain_fetch=lambda a, b: _rain(), reference_depth=lambda *a: -1.0, sensitivity=False)
    assert r["ok"] and r["out"]["sea_fraction"] > 0.2 and any(n["step"] == "1c" for n in r["notebook"])
    assert r["grids"]["h_max"][sea].max() == 0.0 and r["out"]["flooded_area_km2"] < r["out"]["land_km2"]
    # mass balance with a sink: in = out (edges + sea) + stored
    zz = np.zeros((20, 20)) + np.linspace(1, 0, 20)[None, :]; s = np.zeros_like(zz, bool); s[:, -3:] = True
    res = flood.simulate(zz, 20.0, 1800, rain_m_s=30e-3 / 3600, open_boundary=False, sink=s)
    assert res.mass_error < 1e-9 and res.vol_out > 0
