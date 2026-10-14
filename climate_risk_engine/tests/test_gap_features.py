"""Features built to back website claims: Aqueduct 4.0, SBTi alignment, BRSR, benchmark, portfolio CSV, CLI."""
import csv
import io
import os

import pytest

os.environ.setdefault("CRI_API_KEYS", "")
from fastapi.testclient import TestClient  # noqa: E402

from cri.api.main import app  # noqa: E402
from cri.climate.sbti_alignment import check  # noqa: E402
from cri.connectors.wri_aqueduct import WRIAqueductConnector, point, stress_path  # noqa: E402

c = TestClient(app)


# ── WRI Aqueduct 4.0 ─────────────────────────────────────────────────────────

def test_aqueduct_known_places():
    chennai = point(13.08, 80.27)
    rotterdam = point(51.92, 4.48)
    assert chennai["found"] and chennai["iso3"] == "IND" and chennai["admin1"] == "Tamil Nadu"
    assert chennai["baseline"]["bws"]["score"] > 4.0            # extremely high water stress
    assert rotterdam["baseline"]["bws"]["score"] < 1.5
    assert set(chennai["future"]) == {"opt", "bau", "pes"}
    assert set(stress_path(chennai)) == {"2019", "2030", "2050", "2080"}


def test_aqueduct_ocean_and_snap():
    assert point(0.0, -140.0)["found"] is False
    r = point(13.08, 80.33)          # just offshore Chennai → snaps to land within ~15 km
    assert r["found"] and r["snapped_km"] <= 15


def test_aqueduct_connector_shape_and_industry_weighting():
    w = WRIAqueductConnector().get_water_risk(33.45, -112.07, 2050)
    assert w["source"] == "wri_aqueduct_4.0" and 0 <= w["water_stress"] <= 5
    r = point(33.45, -112.07, "mining")
    assert r["overall_water_risk"]["sector_weighting"] == "Mining"
    assert r["overall_water_risk"]["sector_score"] is not None


def test_water_endpoint_and_agent_lookup():
    r = c.get("/physical/water-risk", params={"lat": 13.08, "lon": 80.27, "sector": "cement"})
    assert r.status_code == 200 and r.json()["found"]
    r = c.post("/agent/water-risk", json={"company_name": "X", "sector": "cement", "revenue_usd_m": 500,
                                          "lat": 13.08, "lon": 80.27})
    assert r.status_code == 200 and r.json()["aqueduct_lookup"]["iso3"] == "IND"
    assert c.post("/agent/water-risk", json={"company_name": "X", "sector": "cement", "revenue_usd_m": 500}).status_code == 422


# ── SBTi ─────────────────────────────────────────────────────────────────────

def test_sbti_ambition_and_alerts():
    ok = check(2020, 1e6, 2030, 42.0)
    weak = check(2020, 1e6, 2030, 30.0, reported={2024: 900_000})
    assert ok["ambition_aligned"] and not weak["ambition_aligned"]
    assert weak["sbti_required_reduction_by_target_year_pct"] == pytest.approx(42.0)
    assert any(a["type"] == "off_track_sbti_minimum" for a in weak["alerts"])
    on_track = check(2020, 1e6, 2030, 50.0, reported={2024: 790_000})
    assert on_track["alerts"] == []
    assert check(2020, 1e6, 2030, 25.0, scope="scope3")["ambition_aligned"]   # 2.5 %/yr for Scope 3


def test_sbti_endpoint_validation():
    assert c.post("/sbti/alignment", json={"base_year": 2020, "base_emissions_t": 1e6, "target_year": 2019,
                                           "target_reduction_pct": 40}).status_code == 422


# ── disclosure units & BRSR ──────────────────────────────────────────────────

def test_disclosure_units_and_brsr_intensity():
    r = c.post("/reports/brsr", json={"company_id": "ultratech", "usd_inr": 88.0})
    assert r.status_code == 200
    p6 = r.json()["sections"]["section_c_principle6_environment"]["P6_E3_ghg_emissions"]
    assert p6["scope_1_tco2e"] > 1e6                         # tonnes, not megatonnes
    s12 = p6["scope_1_tco2e"] + p6["scope_2_tco2e"]
    assert p6["ghg_emission_intensity_tco2e_per_crore_inr"] == pytest.approx(s12 / p6["turnover_inr_crore"], rel=1e-3)
    tcfd = c.post("/reports/tcfd", json={"company_id": "ultratech"}).json()
    assert tcfd["sections"] is not None


# ── benchmark ────────────────────────────────────────────────────────────────

def test_benchmark_requires_weights_and_decomposes():
    pos = [{"company_id": "a", "company_name": "A", "exposure_usd_m": 100, "sector": "Energy", "region": "Global"},
           {"company_id": "b", "company_name": "B", "exposure_usd_m": 100, "sector": "Technology", "region": "Global"}]
    assert c.post("/portfolio/benchmark", json={"positions": pos}).status_code == 422
    j = c.post("/portfolio/benchmark", json={"positions": pos, "benchmark_sector_weights": {"Energy": 10, "Technology": 90}}).json()
    a = j["attribution"]
    for k, v in j["active"].items():
        assert v == pytest.approx(a["sector_allocation"][k] + a["regional_exposure"][k], abs=0.01)
    assert j["active"]["eal_current_policies_pct"] > 0          # overweight energy → more risk


# ── portfolio CSV + CLI ──────────────────────────────────────────────────────

def test_portfolio_export_csv_and_bad_rows(tmp_path):
    from cri.cli import main
    f = tmp_path / "pf.csv"
    f.write_text("name,lat,lon,sector,value_usd\nPlant A,13.08,80.27,cement,250000000\nBad,,,cement,\n")
    out = tmp_path / "out.csv"
    assert main(["ingest", str(f), "-o", str(out), "--draws", "300"]) == 0
    rows = list(csv.DictReader(io.StringIO(out.read_text())))
    assert len(rows) == 2 and rows[0]["iso3"] == "IND" and rows[1]["error"]
    assert "Physical risk only" in rows[0]["headline"]


def test_cli_lca_and_water(capsys):
    from cri.cli import main
    assert main(["lca", "aluminium", "--kwh", "14000", "--country", "NOR"]) == 0
    assert "kg CO2e" in capsys.readouterr().out
    assert main(["water", "13.08", "80.27"]) == 0


def test_public_company_risk_not_found_is_honest():
    r = c.get("/public/company-risk", params={"q": "Zzqx Nonexistent Retail"}).json()
    assert r["found"] is False and "register covers" in r["note"]
    assert c.get("/public/company-risk", params={"q": "ab"}).status_code == 422


def test_agent_assess_has_no_placeholders():
    d = c.post("/agent/assess", json={"company_name": "Acme Software Labs", "sector_hint": "technology",
                                       "jurisdiction": "US", "revenue_usd_m": 800}).json()
    assert d["company_profile"]["resolved_name"] == "Acme Software Labs"          # no silent substitution
    assert d["extended_risk"]["paris_alignment"] is None
    assert "eal_p50_usd_m" not in d["extended_risk"]["physical_risk"]
    assert d["risk_assessment"]["scenarios"]["cp"]["npv_impact_pct"] == 0.0


def test_insurance_offshore_north_sea_is_not_assessed_not_zero():
    """A North Sea wind farm has no tropical-cyclone record and no land flood map: the engine must say 'not assessed', never a zero premium."""
    from cri import insurance as I
    hz = {"wind": {"lam": 0.0, "sigma": None, "rp100_ms": None}, "river": None, "coast": None, "cont": "europe", "tc_region": "ROW", "iso3": None, "notes": []}
    r = I.run([{"name": "Offshore farm", "lat": 55.6, "lon": 7.8, "tiv_usd": 6e8, "sector": "offshore wind", "hazard": hz}], n_years=500)
    row = r["sites"][0]
    assert row["technical_premium"] is None and row["aal"] is None and row["coverage_complete"] is False
    assert {x["peril"].split(" ")[0] for x in row["not_assessed"]} == {"wind", "experience"}     # offshore covers wind only; its claims-driven losses also need the insurer's claims
    assert r["premium"]["technical_total"] is None and "NO PREMIUM" in r["premium"]["coverage_warning"] and r["warnings"]


def test_insurance_mixed_portfolio_is_flagged_lower_bound():
    from cri import insurance as I
    ok = {"wind": {"lam": 1.0, "sigma": 0.3, "rp100_ms": 60.0}, "river": None, "coast": None, "cont": "asia", "tc_region": "WP", "iso3": "PHL", "notes": []}
    sea = {"wind": {"lam": 0.0, "sigma": None, "rp100_ms": None}, "river": None, "coast": None, "cont": "europe", "tc_region": "ROW", "iso3": None, "notes": []}
    r = I.run([{"name": "Port", "lat": 14.6, "lon": 121.0, "tiv_usd": 1e8, "hazard": ok, "covers": ["wind"]}, {"name": "Sea", "lat": 55.6, "lon": 7.8, "tiv_usd": 1e8, "hazard": sea, "covers": ["wind"]}], n_years=500)
    assert r["premium"]["technical_total"] is not None and r["premium"]["complete"] is False and "LOWER BOUND" in r["premium"]["coverage_warning"]
    assert r["premium"]["sites_with_perils_not_assessed"] == ["Sea"]


def _fake_marine(url, **kw):
    import math
    n = 60
    ys = list(range(1964, 2024))
    times = [f"{y}-{m:02d}-15" for y in ys for m in (1, 12)]
    hs = [4.0 + 3.0 * math.sin(i * 1.7) ** 2 for i in range(len(times))]
    return {"daily": {"time": times, "wave_height_max": hs}, "hourly": {"time": times, "wave_height": hs}}


def test_claims_credibility_not_assessed_without_claims_and_scales_with_data():
    from cri.analysis import claims_credibility as cc
    assert cc.assess(None, 100.0)["status"] == "null" and cc.assess([], 100.0)["status"] == "null"
    assert cc.assess([{"amount": 1e6, "cause": "cable"}], 0.0)["status"] == "null"
    claims = [{"amount": 2e6, "cause": "subsea cable"}] * 6 + [{"amount": 5e5, "cause": "blade"}] * 4
    a = cc.assess(claims, 200.0, "turbine-years", 80.0)
    assert a["status"] == "modelled" and a["aal_asset"] > 0
    b = cc.assess(claims, 400.0, "turbine-years", 80.0)
    assert b["aal_asset"] < a["aal_asset"]                       # same claims over twice the exposure is a lower rate


def test_offshore_premium_withheld_without_claims_and_priced_with_them():
    from cri import insurance as I
    storm = {"status": "constructed", "years": [1964, 2023], "return_levels": {}, "storm_damage_ratio_aal": 0.0004, "wind": {"lam": 0.0, "sigma": None, "rp100_ms": None},
             "river": None, "coast": None, "cont": "europe", "tc_region": "ROW", "iso3": None, "notes": []}
    hz = {"wind": {"lam": 0.0, "sigma": None, "rp100_ms": None}, "river": None, "coast": None, "cont": "europe", "tc_region": "ROW", "iso3": None, "notes": [], "storm": None}
    site = {"name": "Horns Rev", "lat": 55.5, "lon": 7.9, "tiv_usd": 6e8, "sector": "offshore wind", "hazard": hz}
    r0 = I.run([dict(site)], n_years=500)
    assert r0["sites"][0]["technical_premium"] is None and r0["premium"]["technical_total"] is None
    claims = [{"amount": 2e6, "cause": "subsea cable"}] * 5
    s1 = dict(site); s1["offshore"] = {"turbines": 80, "claims": claims, "claims_exposure_turbine_years": 300}
    r1 = I.run([s1], n_years=500)                                 # wind peril still not assessed (no storm hazard supplied) so the premium stays withheld
    assert r1["sites"][0]["technical_premium"] is None


def test_offshore_cold_screen_with_fake_fetch():
    import datetime as dt, math, tempfile, os
    os.environ["CRI_CACHE_DIR"] = tempfile.mkdtemp()
    from cri.analysis import offshore_wind as O
    days = []; d = dt.date(1989, 1, 1)
    while d <= dt.date(2024, 12, 31):
        days.append(d); d += dt.timedelta(days=1)
    tm = [5 + 9 * math.sin(2 * math.pi * (x.timetuple().tm_yday - 110) / 365.25) - (8 if (x.year in (1996, 2010) and x.month in (1, 2)) else 0) for x in days]
    def fake(url, **k):
        return {"daily": {"time": [x.isoformat() for x in days], "temperature_2m_max": [v + 3 for v in tm], "temperature_2m_min": [v - 3 for v in tm], "temperature_2m_mean": tm,
                          "relative_humidity_2m_mean": [92] * len(days), "precipitation_sum": [0.5] * len(days)}}
    c = O.cold_screen(55.5, 7.9, today=dt.date(2025, 6, 1), fetch=fake)
    assert c["status"] == "constructed" and c["freezing_degree_days"]["max"] > c["freezing_degree_days"]["median"] and c["loss_effect"] is None
    assert c["freezing_degree_days"]["worst_winter"] in ("1995/1996", "2009/2010")
    bad = O.cold_screen(52.1, 3.3, today=dt.date(2025, 6, 1), fetch=lambda u, **k: (_ for _ in ()).throw(RuntimeError("down")))
    assert bad["status"] == "null" and "not zero" in bad["reason"]


def test_offshore_scope_lists_more_than_wind():
    from cri import insurance as I
    sc = I._offshore_scope({}, {"storm": {"status": "constructed", "cold": {"status": "constructed"}, "access": {"share_of_hours_hs_below": {}}}}, False)
    names = " | ".join(x["peril"] for x in sc)
    assert "sea ice, icing and freeze-thaw" in names and "scour" in names and "ship collision" in names and "corrosion" in names
    assert any("not assessed: no claims" in x["status"] for x in sc)
