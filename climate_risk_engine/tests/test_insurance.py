"""Insurer view: policy terms, reinsurance, simulation against independent calculations, correlation, climate pairing."""
import numpy as np
import pytest

from cri import insurance as I


def _depths(prof):
    return {str(rp): [d] * n + [0.0] * (100 - n) for rp, (n, d) in prof.items()}


RIVER = {"rps": [10, 20, 50, 100, 200, 500],
         "depths": _depths({10: (0, 0.0), 20: (5, 0.3), 50: (20, 0.6), 100: (50, 0.9), 200: (80, 1.2), 500: (100, 2.0)}), "protection": 0.0}
COAST = {"rps": [10, 100], "depths": {"2018_10": [0.4] * 30 + [0.0] * 70, "2018_100": [1.0] * 60 + [0.0] * 40,
                                      "2050_10": [0.5] * 30 + [0.0] * 70, "2050_100": [1.2] * 60 + [0.0] * 40}, "protection": 0.0}


def haz(wind=True, river=True, coast=False, lam=0.6, sigma=9.0):
    return {"cont": "asia", "tc_region": "NI", "iso3": "BGD", "notes": [],
            "wind": {"lam": lam if wind else 0.0, "sigma": sigma if wind else None, "rp100_ms": 17.5 + sigma * np.log(lam * 100)},
            "river": RIVER if river else None, "coast": COAST if coast else None}


def site(name="A", lat=22.0, lon=91.0, tiv=1e6, **kw):
    d = {"name": name, "lat": lat, "lon": lon, "tiv_usd": tiv, "sector": "cement", "hazard": haz(**kw.pop("h", {}))}
    d.update(kw)
    return d


def run(sites, **kw):
    kw.setdefault("n_years", 4000)
    return I.run(sites, hazards=[I.normalise_site(s, k)["hazard"] for k, s in enumerate(sites)], **kw)


# ── independent checks of the simulation ─────────────────────────────────────────────────────────────────────────────────────

def test_ground_up_matches_numerical_integration():
    s = [I.normalise_site(site(), 0)]
    h = [s[0]["hazard"]]
    gu = I.ground_up(s, h, 40000, 3)
    ref = I.analytic_aal(h[0])
    for k in ("wind", "river"):
        sim = float(gu[k].mean())
        assert abs(sim - ref[k]) <= 0.06 * ref[k] + 1e-5, (k, sim, ref[k])
    assert float(gu["coast"].mean()) == 0.0


def test_matches_the_engines_own_loss_simulation():
    """Same hazard definition as finance.distribution.simulate (Poisson storms, annual flood from return periods): average annual damage must agree."""
    from cri.finance.distribution import simulate
    from cri.finance.vulnerability import flood_damage
    h = haz()
    dmg = [float(np.mean(flood_damage(np.array(h["river"]["depths"][str(r)]), "asia"))) for r in h["river"]["rps"]]
    case = {"value": {"low": 1e6, "mode": 1e6, "high": 1e6}, "tc": {"lam": h["wind"]["lam"], "sigma": h["wind"]["sigma"], "region": "NI"},
            "river": {"rps": h["river"]["rps"], "damage": dmg, "protection": 0.0}, "transition": {"emissions_t": 0.0}}
    eng = simulate(case, n=6000)["physical"]
    s = [I.normalise_site(site(), 0)]
    gu = I.ground_up(s, [s[0]["hazard"]], 40000, 5)
    ours = 100 * float((1 - (1 - gu["wind"]) * (1 - gu["river"])).mean())
    assert abs(ours - eng["aal_pct"]) <= 0.08 * eng["aal_pct"], (ours, eng["aal_pct"])


# ── policy terms and reinsurance by hand ─────────────────────────────────────────────────────────────────────────────────────

def test_deductible_limit_sublimit_and_cover():
    sites = [I.normalise_site(site(deductible_usd=100.0, limit_usd=500.0, tiv=1000.0), 0),
             I.normalise_site(site(tiv=1000.0, wind_limit_usd=200.0, covers=["wind"]), 1)]
    z = np.zeros((1, 2), np.float32)
    gu = {"wind": np.array([[0.05, 0.9]], np.float32), "river": np.array([[0.7, 0.7]], np.float32), "coast": z}
    L = I.apply_terms(gu, sites)
    assert L["wind"][0, 0] == pytest.approx(0.0)                     # 50 ground-up < 100 deductible
    assert L["flood"][0, 0] == pytest.approx(500.0)                  # 700 - 100 = 600, capped at 500
    assert L["wind"][0, 1] == pytest.approx(200.0)                   # sublimit
    assert L["flood"][0, 1] == pytest.approx(0.0)                    # flood not covered on site 2
    assert L["total"][0, 0] == pytest.approx(500.0)


def test_overall_limit_caps_both_perils():
    s = [I.normalise_site(site(tiv=1000.0, limit_usd=600.0), 0)]
    gu = {"wind": np.array([[0.5]], np.float32), "river": np.array([[0.5]], np.float32), "coast": np.zeros((1, 1), np.float32)}
    L = I.apply_terms(gu, s)
    assert L["total"][0, 0] == pytest.approx(600.0)
    assert L["wind"][0, 0] + L["flood"][0, 0] == pytest.approx(600.0)


def test_reinsurance_quota_share_then_layer():
    w = np.array([[50.0], [150.0], [1000.0], [0.0]])
    f = np.zeros_like(w)
    r = I.reinsurance(w, f, {"quota_share_pct": 20, "xl": [{"attach": 100.0, "limit": 200.0, "reinstatements": 0}]})
    assert r["ceded"].tolist() == pytest.approx([10.0, 30.0 + 20.0, 200.0 + 200.0, 0.0])
    assert r["net_wind"].tolist() == pytest.approx([40.0, 100.0, 600.0, 0.0])
    assert r["layers"][0]["prob_attach"] == pytest.approx(0.5)


def test_input_validation():
    with pytest.raises(I.InsuranceError):
        I.normalise_site({"lat": 1, "lon": 1}, 0)
    with pytest.raises(I.InsuranceError):
        I.normalise_site({"lat": 1, "lon": 1, "tiv_usd": 100, "deductible_usd": 100, "limit_usd": 50}, 0)
    with pytest.raises(I.InsuranceError):
        I.normalise_site({"lat": 1, "lon": 1, "tiv_usd": 100, "covers": ["hail"]}, 0)
    with pytest.raises(I.InsuranceError):
        I.run([], n_years=100)


# ── portfolio behaviour ──────────────────────────────────────────────────────────────────────────────────────────────────────

def test_spatial_correlation_drives_the_tail():
    near = [site("A", 22.0, 91.0), site("B", 22.05, 91.05)]
    far = [site("A", 22.0, 91.0), site("B", 22.0, 111.0)]
    rn, rf = run(near, n_years=6000), run(far, n_years=6000)
    assert rn["stats"]["gross"]["aal"] == pytest.approx(rf["stats"]["gross"]["aal"], rel=0.08)   # same sites, same AAL
    pn = next(r for r in rn["stats"]["gross"]["rows"] if r["rp"] == 100)["aep"]
    pf = next(r for r in rf["stats"]["gross"]["rows"] if r["rp"] == 100)["aep"]
    assert pn > 1.1 * pf                                                  # clustered portfolio has the fatter tail


def test_curves_are_monotone_and_oep_not_above_aep():
    r = run([site("A"), site("B", 22.4, 91.4, h={"coast": True})], n_years=5000)
    rows = r["stats"]["gross"]["rows"]
    assert all(rows[i]["aep"] <= rows[i + 1]["aep"] + 1e-9 for i in range(len(rows) - 1))
    assert all(x["oep"] <= x["aep"] + 1e-9 for x in rows)
    assert r["stats"]["gross"]["tvar99"] >= rows[5]["aep"] - 1e-9
    assert not rows[-1]["reliable"] or r["n_years"] >= 5000


def test_deductible_ladder_and_premium():
    r = run([site("A", deductible_pct=0), site("B", 22.3, 91.2, tiv=2e6)], n_years=4000)
    aal = [x["aal"] for x in r["deductible_ladder"]]
    assert all(aal[i] >= aal[i + 1] for i in range(len(aal) - 1)) and aal[0] > aal[-1]
    for x in r["sites"]:
        assert x["technical_premium"] > x["aal"] > 0
    assert r["premium"]["technical_total"] == pytest.approx(sum(x["technical_premium"] for x in r["sites"]))
    assert sum(x["aal"] for x in r["sites"]) == pytest.approx(r["stats"]["gross"]["aal"], rel=1e-6)


def test_reinsurance_lowers_net_tail():
    prog = {"quota_share_pct": 0, "xl": [{"attach": 200000.0, "limit": 500000.0, "reinstatements": 1}]}
    r = run([site("A", tiv=5e6), site("B", 22.1, 91.1, tiv=5e6)], reinsurance_program=prog, n_years=5000)
    g, n = r["stats"]["gross"], r["stats"]["net"]
    assert n["aal"] < g["aal"]
    assert next(x for x in n["rows"] if x["rp"] == 100)["aep"] <= next(x for x in g["rows"] if x["rp"] == 100)["aep"]
    assert r["reinsurance"]["layers"][0]["expected_recovery"] == pytest.approx(g["aal"] - n["aal"], rel=1e-6)


# ── climate pairing ──────────────────────────────────────────────────────────────────────────────────────────────────────────

def _stub(f=1.2, add=0.3, ws=1.05):
    def compose(lat, lon, ssp="ssp245", year=2070, fetch=None, tile_fetch=None, with_sea=True):
        return {"river_depth_factor": {"p10": f, "median": f, "p90": f}, "rain": {"p10": f, "median": f, "p90": f}, "warming_k": 1.5,
                "sea": {"available": True, "add_m": {"p17": add, "median": add, "p83": add}},
                "wind": {"tc_exposed": True, "scale": {"low": ws, "median": ws, "high": ws}}, "confidence": [], "limits": [], "attribution": []}
    return compose


def test_climate_run_is_paired_and_directional():
    sites = [site("A", h={"coast": True}), site("B", 22.5, 91.5, h={"coast": True})]
    same = run(sites, climate={"ssp": "ssp245", "year": 2070}, compose_fn=_stub(1.0, 0.0, 1.0), n_years=4000)
    assert same["climate"]["aal"]["change"] == pytest.approx(0.0, abs=1e-6)       # no change in hazard -> exactly zero difference
    worse = run(sites, climate={"ssp": "ssp585", "year": 2070}, compose_fn=_stub(1.3, 0.4, 1.08), n_years=4000)
    c = worse["climate"]
    assert c["aal"]["change"] > 0 and c["aal"]["change_pct"] > 1
    assert c["tail"]["aep100"]["change"] > 0
    assert all(s["premium_scenario"] >= s["premium_today"] for s in c["per_site"])
    assert c["premium_total_scenario"] > worse["premium"]["technical_total"]


def test_climate_failure_keeps_todays_hazard_and_flags_it():
    def boom(*a, **k):
        raise RuntimeError("no projection here")
    r = run([site("A")], climate={"ssp": "ssp245", "year": 2050}, compose_fn=boom, n_years=2000)
    assert r["climate"]["aal"]["change"] == pytest.approx(0.0, abs=1e-6)
    assert r["climate"]["per_site"][0]["climate"]["ok"] is False


def test_api_endpoint_and_dashboard():
    import os
    os.environ.setdefault("CRI_API_KEYS", "")
    from fastapi.testclient import TestClient
    from cri.api.main import app
    c = TestClient(app)
    body = {"sites": [site("A", tiv=3e6, deductible_pct=2), site("B", 22.2, 91.2, tiv=2e6)], "n_years": 1000, "seed": 3,
            "reinsurance": {"quota_share_pct": 10, "xl": [{"attach": 1e5, "limit": 1e6}]}}
    r = c.post("/insurance/analyse", json=body)
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["n_sites"] == 2 and d["stats"]["net"]["aal"] < d["stats"]["gross"]["aal"] and d["method"] and d["limits"]
    assert c.post("/insurance/analyse", json={"sites": []}).status_code == 422
    bad = dict(body, sites=[{"name": "x", "lat": 1, "lon": 1}])
    assert c.post("/insurance/analyse", json=bad).status_code == 422
    assert c.post("/insurance/analyse", json=dict(body, climate={"ssp": "ssp999"})).status_code == 422
    page = c.get("/insurance/dashboard")
    assert page.status_code == 200 and "Insurer view" in page.text


def test_reproducible_and_json_safe():
    import json
    a = run([site("A")], seed=4, n_years=1500)
    b = run([site("A")], seed=4, n_years=1500)
    assert a["stats"]["gross"]["aal"] == b["stats"]["gross"]["aal"]
    json.dumps(a)
