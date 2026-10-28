import pytest

from cri import carbon_registry as CR
from cri.carbon_markets import screen_credits

CSV = """Serial Number,Quantity,Vintage Year,Registration Date,Crediting Period Start,Project Type,Status,Host Country,Internal Note
CER-1,1000,2018,2014-06-01,2014-06-01,wind,Retired,India,x
CER-2,500,2012,2011-03-01,2011-03-01,hydro,Issued,Brazil,y
CER-3,200,2019,2015-01-01,2015-01-01,afforestation,Issued,Kenya,z
CER-4,,2017,,,solar,Issued,Chile,w
"""


def test_import_maps_columns_and_keeps_blanks_blank():
    r = CR.import_csv(CSV, "CDM")
    assert r["matched_columns"]["quantity_t"] == "Quantity" and "Internal Note" in r["ignored_columns"]
    lots = {l["id"]: l for l in r["lots"]}
    assert lots["CER-1"]["retired"] is True and lots["CER-2"]["retired"] is False
    assert lots["CER-4"]["quantity_t"] is None and lots["CER-4"]["registered_year"] is None
    assert {"id": "CER-4", "missing": ["quantity_t"]} in r["incomplete"]


def test_cer_rules_follow_published_conditions():
    lots = {l["id"]: l for l in screen_credits(CR.import_csv(CSV, "CDM")["lots"])["lots"]}
    p = lambda i, name: next(x for x in lots[i]["cer"]["pathways"] if x["pathway"].startswith(name))       # noqa: E731
    assert p("CER-1", "Article 6.4")["status"] == "may_be_eligible"
    assert p("CER-2", "Article 6.4")["status"] == "not_eligible"            # registered before 1 Jan 2013
    assert p("CER-3", "Article 6.4")["status"] == "not_eligible"            # afforestation
    assert p("CER-4", "Article 6.4")["status"] == "cannot_tell"             # registration year missing: named, not assumed
    assert "registered_year" in p("CER-4", "Article 6.4")["reasons"][0]
    assert all(p(i, "CORSIA")["status"] == "not_eligible" for i in lots)    # CDM not listed for open CORSIA phases (ICAO Oct 2025)
    assert any("quantity missing" in f for f in lots["CER-4"]["flags"])


def test_post_2020_cer_is_not_a_pre_2021_unit():
    r = CR.assess_cer({"registry": "CDM", "vintage": 2021, "registered_year": 2015, "crediting_period_start_year": 2015})
    assert r["pathways"][0]["status"] == "not_eligible"


def test_non_cdm_lots_are_untouched():
    r = screen_credits([{"id": "v", "registry": "Verra", "project_type": "biochar", "quantity_t": 5, "vintage": 2024}])
    assert r["lots"][0]["cer"] is None


def test_import_api():
    from fastapi.testclient import TestClient
    from cri.api.main import app
    c = TestClient(app)
    r = c.post("/carbon-markets/import-registry", json={"csv_text": CSV, "programme": "CDM"})
    assert r.status_code in (200, 401, 403)
    if r.status_code == 200:
        assert r.json()["screen"]["lots"][0]["cer"]["pathways"]
