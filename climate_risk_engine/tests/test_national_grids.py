"""National grid factors for Australia, Canada, Germany, China and Japan, read from the publishers' own documents on 6 Nov 2026."""
import importlib.util
import json
import os
from pathlib import Path

import pytest

os.environ.setdefault("CRI_API_KEYS", "")
from fastapi.testclient import TestClient  # noqa: E402

from cri.api.main import app  # noqa: E402
from cri.carbon import factors as F  # noqa: E402
from cri.carbon import national_grids as NG  # noqa: E402
from cri.carbon import inventory as I  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
c = TestClient(app)


def _build():
    spec = importlib.util.spec_from_file_location("build_national_grids", ROOT / "scripts" / "build_national_grids.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_seed_file_is_exactly_what_the_checked_build_script_writes():
    assert _build().build() == json.loads((ROOT / "src" / "cri" / "data" / "national_grids.json").read_text())


def test_build_script_rejects_a_transcription_error(monkeypatch):
    b = _build()
    monkeypatch.setattr(b, "CN_PROVINCES", b.CN_PROVINCES[:-1])                        # a missing province
    with pytest.raises(AssertionError):
        b.build()
    b = _build()
    monkeypatch.setattr(b, "CN_NATIONAL", 0.5406)                                      # no longer reproduces the -1.1% the MEE states
    with pytest.raises(AssertionError):
        b.build()
    b = _build()
    monkeypatch.setitem(b.DE, 2024, (158, 447, 363, 334, 361, 414, 53, "provisional"))   # CO2 g/kWh no longer equals emissions / consumption
    with pytest.raises(AssertionError):
        b.build()


# ── Australia ────────────────────────────────────────────────────────────────

def test_australia_state_factors_are_the_nga_2026_table():
    nsw, vic, qld, sa = (F.electricity_location("AUS", region=r) for r in ("NSW", "vic", "Queensland", "SA"))
    assert [x["g_per_kwh"] for x in (nsw, vic, qld, sa)] == [600.0, 740.0, 650.0, 210.0]
    assert nsw["scope3_g_per_kwh"] == 70.0 and vic["scope3_g_per_kwh"] == 110.0
    assert nsw["gwp_basis"] == "AR5" and "Table 1" in nsw["source"] and nsw["dq"] == 2 and nsw["note"] is None
    assert F.electricity_location("AU", region="ACT")["g_per_kwh"] == 600.0                  # ACT shares the NSW grid (the table row is NSW and ACT)
    nat = F.electricity_location("AUS")
    assert nat["g_per_kwh"] == 590.0 and nat["dq"] == 3 and "give the state" in nat["note"]
    assert F.electricity_residual("AUS")["g_per_kwh"] == 790.0


def test_australia_western_australia_has_two_grids_and_says_which_it_assumed():
    wa = F.electricity_location("AUS", region="WA")
    assert wa["g_per_kwh"] == 450.0 and "South West Interconnected" in wa["note"]
    assert F.electricity_location("AUS", region="WA-NWIS")["g_per_kwh"] == 560.0


def test_unknown_region_falls_back_to_national_and_says_so():
    r = F.electricity_location("AUS", region="Mars")
    assert r["g_per_kwh"] == 590.0 and "not one this table holds" in r["note"] and r["dq"] == 3


# ── Canada ───────────────────────────────────────────────────────────────────

def test_canada_provinces_generation_and_consumption_intensity():
    ab = F.electricity_location("CAN", region="AB")
    assert ab["g_per_kwh"] == 334.76 and ab["consumption_g_per_kwh"] == 346.10 and ab["year"] == "2024" and "preliminary" in ab["source"]
    assert F.electricity_location("CA", region="alberta", fy="2023")["g_per_kwh"] == 420.30
    assert F.electricity_location("CAN", region="QC")["g_per_kwh"] == 2.10
    pe = F.electricity_location("CAN", region="PE")
    assert pe["g_per_kwh"] == 3.19 and pe["consumption_g_per_kwh"] == 264.55 and "imports and losses" in pe["note"]
    nat = F.electricity_location("CAN")
    assert nat["g_per_kwh"] == 102.17 and "give the province" in nat["note"]
    assert "does not name the GWP set" in nat["note"] and nat["gwp_basis"] == "AR5"
    assert "year 2019 is not held" in F.electricity_location("CAN", region="ON", fy="2019")["note"]


# ── Germany ──────────────────────────────────────────────────────────────────

def test_germany_uba_series_with_status():
    d = F.electricity_location("DEU")
    assert d["g_per_kwh"] == 352.0 and d["co2_only_g_per_kwh"] == 344.0 and d["with_upstream_g_per_kwh"] == 406.0 and d["year"] == "2025" and d["status"] == "estimate"
    assert "estimate" in d["note"] and d["gwp_basis"] == "AR5"
    y23 = F.electricity_location("DEU", fy="2023")
    assert y23["g_per_kwh"] == 387.0 and y23["co2_only_g_per_kwh"] == 379.0 and y23["note"] is None
    assert F.electricity_location("DEU", fy="2024")["status"] == "provisional"
    assert "year 1999 is not held" in F.electricity_location("DEU", fy="1999")["note"]
    assert F.electricity_residual("DEU")["g_per_kwh"] > 352                                      # market-based still uses the AIB residual mix


# ── China ────────────────────────────────────────────────────────────────────

def test_china_national_region_province_and_market_based_variant():
    n = F.electricity_location("CHN")
    assert n["g_per_kwh"] == 530.6 and n["gwp_basis"] == "CO2 only" and n["year"] == "2023"
    assert n["residual_g_per_kwh"] == 609.6 and n["fossil_electricity_g_per_kwh"] == 827.3
    assert F.electricity_residual("CN")["g_per_kwh"] == 609.6
    assert F.electricity_location("CHN", region="EAST")["g_per_kwh"] == 550.0
    assert F.electricity_location("CHN", region="HB")["g_per_kwh"] == 404.4          # Hubei, ISO 3166-2:CN
    assert F.electricity_location("CHN", region="HE")["g_per_kwh"] == 651.6          # Hebei
    assert F.electricity_location("CHN", region="四川")["g_per_kwh"] == 156.4
    assert F.electricity_location("CHN", region="Yunnan")["g_per_kwh"] == 133.3
    assert F.electricity_location("CHN", region="Tibet")["note"].startswith("region 'Tibet' is not one this table holds")
    assert "only the 2023 data year" in F.electricity_location("CHN", fy="2025")["note"]


def test_china_table_is_complete_and_regions_bracket_the_national_value():
    regs = NG.data()["countries"]["CHN"]["regions"]
    assert sum(r["level"] == "province" for r in regs.values()) == 30 and sum(r["level"] == "region" for r in regs.values()) == 7
    assert regs["SC"]["g_per_kwh"] < regs["CN"]["g_per_kwh"] < regs["TJ"]["g_per_kwh"]


# ── Japan ────────────────────────────────────────────────────────────────────

def test_japan_has_only_a_substitute_value_and_says_so():
    j = F.electricity_location("JPN")
    assert j["g_per_kwh"] == 422.0 and j["dq"] == 3 and j["gwp_basis"] == "CO2 only"
    assert "no single national grid factor" in j["note"] and "substitute value" in j["source"]
    assert F.electricity_residual("JPN") is None


# ── inventory and API ────────────────────────────────────────────────────────

def _calc(country, gwp="AR6", **rec):
    return I.calculate(I.from_dict({"organisation": "x", "reporting_year": 2025, "gwp": gwp, "entities": [{"id": "E", "name": "w", "country": country}],
                                    "records": [dict({"id": "r", "entity": "E", "category": "s2_electricity", "method": "electricity", "kwh": 1_000_000, "evidence": "x"}, **rec)]}))


def test_inventory_uses_the_state_and_discloses_the_gwp_basis():
    r = _calc("AUS", grid_region="VIC")
    assert r["totals_t"]["scope2_location"] == pytest.approx(740.0)                          # 1,000,000 kWh x 740 g
    assert any("AR5" in x["message"] and "not restated to AR6" in x["message"] for x in r["checks"])
    assert not any("not restated" in x["message"] for x in _calc("AUS", gwp="AR5", grid_region="VIC")["checks"])        # already on the requested basis
    assert not any("not restated" in x["message"] for x in _calc("CHN", grid_region="BJ")["checks"])                    # CO2 only
    assert _calc("CHN", grid_region="BJ")["totals_t"]["scope2_location"] == pytest.approx(555.4)
    assert _calc("CHN", grid_region="BJ")["totals_t"]["scope2_market"] == pytest.approx(609.6)                          # excluding market-traded non-fossil


def test_inventory_unchanged_for_other_countries():
    assert _calc("GBR")["totals_t"]["scope2_location"] == pytest.approx(F.electricity_location("GBR")["g_per_kwh"])
    assert _calc("IND")["totals_t"]["scope2_location"] == pytest.approx(F.electricity_location("IND")["g_per_kwh"])
    assert _calc("USA", us_state="TX")["totals_t"]["scope2_location"] == pytest.approx(F.electricity_location("USA", us_state="TX")["g_per_kwh"])


def test_api_grid_factor_takes_region():
    r = c.get("/carbon/grid-factor", params={"country": "AUS", "region": "VIC"}).json()
    assert r["location"]["g_per_kwh"] == 740.0 and r["residual_mix"]["g_per_kwh"] == 790.0
    assert c.get("/carbon/grid-factor", params={"country": "CAN", "region": "ON", "fy": "2023"}).json()["location"]["g_per_kwh"] == 53.94
    assert c.get("/carbon/grid-factor", params={"country": "ZZZ"}).status_code == 404


def test_provenance_lists_the_file_and_its_monitors():
    from cri import provenance as P
    e = P.entry("national_grids.json")
    assert e["status"] == "published" and e["retrieved"] == "2026-11-06" and {m["id"] for m in e["monitors"]} == {"nga", "eccc", "uba", "mee", "moe"}
