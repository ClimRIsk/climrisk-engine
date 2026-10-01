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
