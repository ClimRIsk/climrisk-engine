"""Client asset lists through the Risk Analyst (/analyst/sites, /analyst/sites/upload)."""
import os

os.environ.setdefault("CRI_API_KEYS", "")
from fastapi.testclient import TestClient  # noqa: E402

from cri.api.main import app  # noqa: E402

c = TestClient(app)
SITES = [{"name": "Rotterdam DC", "lat": 51.9, "lon": 4.4, "value_usd": 5e8},
         {"name": "Dhaka cement", "lat": 23.7, "lon": 90.4, "sector": "cement", "type": "integrated dry",
          "emissions_t": 800000, "value_usd": 2.5e8}]


def test_sites_portfolio_shape_and_value():
    r = c.post("/analyst/sites", json={"label": "Client", "sites": SITES, "top_n": 2})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["found"] and d["n_assets"] == 2 and d["companies"] == ["Client"]
    assert abs(d["portfolio_value_mode"] - 7.5e8) < 1
    sc = d["scenarios"]["Delayed transition"]["total_npv"]
    assert 0 <= sc["mean"] <= 7.5e8 and sc["p95"] >= sc["p50"]
    # carbon cost only where emissions were given
    by = {a["name"]: a for a in d["assets"]}
    dk, rt = by["Dhaka cement"], by["Rotterdam DC"]
    assert dk["npv_nz_mean"] > dk["npv_cp_mean"]                      # emissions given → transition cost differs by scenario
    assert rt["npv_nz_mean"] == rt["npv_cp_mean"]                      # no emissions → physical only
    assert dk["npv_nz_mean"] <= 2.5e8 + 1                              # capped at asset value
    assert by["Rotterdam DC"]["sector"] == "other"


def test_sites_reject_bad_coordinates():
    r = c.post("/analyst/sites", json={"sites": [{"name": "x", "lat": 95, "lon": 0}]})
    assert r.status_code == 422


def test_upload_reports_skipped_rows():
    csv = ("name,lat,lon,sector,value_usd\nRotterdam DC,51.9,4.4,,500000000\n"
           "No coords,,,cement,1000\n").encode()
    r = c.post("/analyst/sites/upload", files={"file": ("sites.csv", csv, "text/csv")}, data={"label": "Upload", "top_n": "1"})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["n_assets"] == 1
    assert d["skipped_rows"] and d["skipped_rows"][0]["name"] == "No coords"


def test_upload_missing_columns():
    r = c.post("/analyst/sites/upload", files={"file": ("x.csv", b"site,latitude\nA,1\n", "text/csv")})
    assert r.status_code == 422


def test_upload_with_financial_columns_adds_earnings_row():
    csv = ("name,lat,lon,sector,type,emissions_t,value_usd,revenue_usd,ebitda_usd,da_usd,interest_usd,net_debt_usd,book_value_usd\n"
           "Dhaka cement,23.7,90.4,cement,integrated dry,800000,250000000,300000000,75000000,25000000,12000000,150000000,250000000\n"
           "Rotterdam DC,51.9,4.4,,,,500000000,,,,,,\n").encode()
    r = c.post("/analyst/sites/upload", files={"file": ("s.csv", csv, "text/csv")}, data={"label": "Fin", "top_n": "1"})
    assert r.status_code == 200, r.text
    by = {a["name"]: a for a in r.json()["assets"]}
    e = by["Dhaka cement"]["earnings_delayed_2040"]
    assert e and e["rating"] and e["leverage"] is not None and e["covenant_breach_prob"] is None   # no covenants given
    assert by["Rotterdam DC"]["earnings_delayed_2040"] is None


def test_upload_lone_financial_column_is_rejected():
    csv = b"name,lat,lon,revenue_usd\nA,51.9,4.4,1000000\n"
    r = c.post("/analyst/sites/upload", files={"file": ("s.csv", csv, "text/csv")})
    assert r.status_code == 422 and "revenue_usd and ebitda_usd" in r.text


def test_analyst_asset_endpoint_validates_financials_and_levers():
    base = {"lat": 23.7, "lon": 90.4, "sector": "cement", "asset_type": "integrated dry", "emissions_t": 8e5, "value_usd": 2.5e8}
    bad = c.post("/analyst/asset", json=dict(base, financials={"revenue_usd": 1e8}))
    assert bad.status_code == 422 and "ebitda_usd" in bad.text
    bad = c.post("/analyst/asset", json=dict(base, levers=[{"name": "x", "share": 2, "mac": 10}]))
    assert bad.status_code == 422
