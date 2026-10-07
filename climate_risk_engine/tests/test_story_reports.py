"""Story-format report suite: every type renders to html/pdf/markdown/json from engine results; customisation works; numbers trace to inputs."""
import json

import pytest

import test_asset_twin as X
from cri import story as S
from cri import story_reports as SR


@pytest.fixture(scope="module")
def twin():
    return X.twin(incidents=[{"date": "2019-08-02", "hazard": "flood", "note": "MCC room flooded", "depth_m": 0.4}])


@pytest.fixture(scope="module")
def analyst():
    from cri.finance.analyst import analyse_asset
    try:
        return analyse_asset(22.2645, 91.805, value_usd=5e8, n=300)
    except Exception as e:                                      # noqa: BLE001
        pytest.skip(f"analyst data unavailable offline: {e}")


@pytest.fixture(scope="module")
def company():
    from cri.finance.analyst import analyse_company
    try:
        return analyse_company(["group:holcim"], top_n=3, n=200)
    except Exception as e:                                      # noqa: BLE001
        pytest.skip(f"analyst data unavailable offline: {e}")


def _all_formats(typ, p, **kw):
    h = SR.generate(typ, p, fmt="html", **kw); pdf = SR.generate(typ, p, fmt="pdf", **kw); md = SR.generate(typ, p, fmt="markdown", **kw); js = SR.generate(typ, p, **kw)
    assert h.startswith("<!DOCTYPE html") and pdf[:4] == b"%PDF" and md.startswith("#") and json.dumps(js)
    return h, js


def test_physical_story_traces_to_twin_and_has_the_arc(twin):
    h, js = _all_formats("physical", {"twin": twin, "name": "Test plant", "sector": "cement", "scenario_lab": False})
    assert "52 m/s" in h and "Sidr" in h                                   # the storm record comes straight from the twin
    assert [s["key"] for s in js["sections"]] == ["history", "loud", "quiet", "protection", "audiences", "method"]
    assert "MCC room flooded" in h                                         # operator-reported incident carried through
    assert "[1]" in h or "<sup" in h                                       # numbered sources
    assert "Sources" in h and "OpenStreetMap" in h


def test_physical_findings_come_from_the_agent(twin):
    from cri import asset_twin as AT
    an = AT.analyse(twin)
    story = SR.build("physical", {"twin": twin, "analysis": an, "name": "Test plant", "sector": "cement", "scenario_lab": False})
    text = json.dumps(story)
    for f in an["findings"][:3]:
        assert f["title"] in text


def test_physical_no_findings_is_not_a_clearance():
    import numpy as np
    from cri import asset_twin as AT
    dry = X.T.build_twin(X.LAT, X.LON, "Dry", sector="cement", _osm=X._osm(), _grid=X._grid(), _flood={"river": {"available": False}, "coastal": {"available": False}}, _tc={"years": 45, "rp_wind_ms": {10: 20.0, 50: 22.0, 100: 24.0, 250: 26.0}, "rp_category": {10: "", 50: "", 100: "", 250: ""}},
                         _storms=[], _sar={"available": False}, _era=X.ERA, run_hydraulics=False)
    st = SR.build("physical", {"twin": dry, "name": "Dry", "scenario_lab": False})
    assert "not a clearance" in json.dumps(st)


def test_section_picker_and_branding(twin):
    full = SR.build("physical", {"twin": twin, "name": "T", "scenario_lab": False})
    part = SR.generate("physical", {"twin": twin, "name": "T", "scenario_lab": False}, sections=["history"])
    assert len(part["story"]["blocks"]) < len(full["blocks"]) and part["kept"] == ["history"]
    h = SR.generate("physical", {"twin": twin, "name": "T", "scenario_lab": False, "client": "Acme Cement"}, brand={"accent": "#ff0000", "name": "ACME RISK", "footer": "Confidential"}, fmt="html")
    assert "Prepared for Acme Cement" in h and "#ff0000" in h and "ACME RISK" in h and "Confidential" in h


def test_sector_template_changes_framing(twin):
    a = SR.build("physical", {"twin": twin, "name": "T", "scenario_lab": False}, template="cement")
    b = SR.build("physical", {"twin": twin, "name": "T", "scenario_lab": False}, template="refinery")
    assert json.dumps(a) != json.dumps(b) and "tank farm" in json.dumps(b)


def test_transition_and_financial_asset(analyst):
    h, js = _all_formats("transition", {"result": analyst})
    assert "carbon price" in h.lower() and "Net Zero 2050" in h and "World Bank" in h
    h2, js2 = _all_formats("financial", {"result": analyst})
    assert "expected shortfall" in h2.lower() and analyst["headline"].split(";")[0] in h2
    mean_nz = analyst["financial"]["scenarios"]["Net Zero 2050"]["total_npv"]["mean"]
    assert f"${mean_nz / 1e6:,.1f}m" in h2                                   # number traces to the engine result


def test_transition_and_financial_company(company):
    h, _ = _all_formats("transition", {"result": company})
    assert "assets" in h and str(company["n_assets"]) in h
    _all_formats("financial", {"result": company})


def test_lca_and_iso14067_from_inputs():
    ex = SR.REPORTS["lca"]["example"]
    h, js = _all_formats("lca", {**ex, "n": 400})
    assert "kg of CO₂e" in h and "ISO 14067" in h and "calcination" in h.lower()
    h2, js2 = _all_formats("iso14067", {**SR.REPORTS["iso14067"]["example"], "n": 400})
    assert [s["key"] for s in js2["sections"]][:2] == ["goal", "results"] and "fingerprint" in h2.lower()


def test_carbon_audit_from_inventory():
    import re
    from pathlib import Path
    html = (Path(SR.__file__).parent / "api" / "carbon_dashboard.html").read_text()
    demo = json.loads(re.search(r"const DEMO = (\{.*?\});\n", html, re.S).group(1))
    h, js = _all_formats("carbon_audit", {"inventory": demo})
    assert "Demo Manufacturing Group" in h and "completeness" in h.lower() or "no data" in h.lower()


def test_cbam_numbers_and_not_applicable():
    h, js = _all_formats("cbam", SR.REPORTS["cbam"]["example"])
    assert "Regulation (EU) 2023/956" in h and "certificate cost" in h
    na = SR.build("cbam", {"company_name": "Bank", "sector": "financials", "revenue_usd_m": 100})
    assert "does not apply" in na["title"]


def test_unknown_type():
    with pytest.raises(KeyError):
        SR.build("nope", {})


def test_missing_inputs_are_explained():
    with pytest.raises(ValueError):
        SR.build("transition", {})


# ── API ──────────────────────────────────────────────────────────────────────

def test_api_build_render_download():
    from fastapi.testclient import TestClient
    from cri.api.main import app
    c = TestClient(app)
    cat = c.get("/story-reports/catalogue").json()
    assert {r["id"] for r in cat["reports"]} == set(SR.REPORTS)
    b = c.post("/story-reports/build", json={"type": "cbam", "payload": SR.REPORTS["cbam"]["example"] | {"client": "Acme"}}).json()
    assert b["token"] and [s["key"] for s in b["sections"]] == ["loud", "quiet", "audiences", "method"]
    h = c.post("/story-reports/render", json={"token": b["token"], "format": "html", "brand": {"accent": "#123456"}, "sections": ["loud"]})
    assert h.status_code == 200 and "#123456" in h.text and "What this means" not in h.text and "Prepared for Acme" in h.text
    p = c.post("/story-reports/render", json={"token": b["token"], "format": "pdf"})
    assert p.content[:4] == b"%PDF" and "attachment" in p.headers["content-disposition"]
    assert c.post("/story-reports/render", json={"token": "gone", "format": "html"}).status_code == 404
    assert c.post("/story-reports/build", json={"type": "transition", "payload": {}}).status_code == 422
    assert c.post("/story-reports/build", json={"type": "bogus", "payload": {}}).status_code == 404
    d = c.get("/story-reports/dashboard")
    assert d.status_code == 200 and "Story reports" in d.text and "/resident/dock.js" in d.text


def test_zero_map_depth_with_solver_finding_is_explained(twin):
    import copy
    from cri import asset_twin as AT
    tw = copy.deepcopy(twin)
    for s in tw["scenarios"]["flood"]:
        s["depth_m"] = 0.0
    an = AT.analyse(tw)
    st = SR.build("physical", {"twin": tw, "analysis": an, "name": "T", "scenario_lab": False})
    text = json.dumps(st)
    assert any(f["hazard"] == "flood" for f in an["findings"]) and "flood maps show no depth on the site itself" in text
    assert "deepest mapped scenario" not in text
