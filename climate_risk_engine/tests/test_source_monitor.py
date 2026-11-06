"""Edition monitor for DESNZ, eGRID, AIB, Ember (and CEA through its own connector). Pages are fixtures shaped like the real pages read on 6 Nov 2026."""
import pytest

from cri.connectors import cea_baseline as CEA
from cri.connectors import source_monitor as SM

DESNZ = """<ul>
<li><a href="/government/publications/greenhouse-gas-reporting-conversion-factors-2026">Greenhouse gas reporting: conversion factors 2026</a><ul><li>31 July 2026</li><li>Research and analysis</li></ul></li>
<li><a href="/government/publications/greenhouse-gas-reporting-conversion-factors-2025">Greenhouse gas reporting: conversion factors 2025</a><ul><li>10 June 2025</li></ul></li>
<li><a href="/government/publications/greenhouse-gas-reporting-conversion-factors-2024">conversion factors 2024</a><ul><li>30 October 2024</li></ul></li></ul>"""
DESNZ_NEW = DESNZ.replace("<ul>\n", '<ul>\n<li><a href="/government/publications/greenhouse-gas-reporting-conversion-factors-2027">conversion factors 2027</a><ul><li>12 June 2027</li></ul></li>\n', 1)
EGRID = "<h1>Detailed Data</h1><h2>eGRID with 2023 Data</h2><p>Released: 1/15/2025<br>Revision 1 Released: 1/17/2025<br>Revision 2 Released: 6/12/2025</p>"
EGRID_NEW = EGRID.replace("eGRID with 2023 Data", "eGRID with 2024 Data").replace("1/15/2025", "1/20/2027").replace("Revision 1 Released: 1/17/2025", "").replace("Revision 2 Released: 6/12/2025", "")
EGRID_REV = EGRID + "<p>Revision 3 Released: 11/2/2026</p>"
AIB = ('<ul>' + "".join(f'<li><a href="/facts/european-residual-mix/{y}">{y}</a></li>' for y in range(2015, 2026)) +
       "</ul><p>Results of the calculation of Residual Mixes for the calendar year 2025</p><em>Version 1.0, 2026-05-26</em>")
AIB_NEW = AIB.replace("</ul>", '<li><a href="/facts/european-residual-mix/2026">2026</a></li></ul>').replace("calendar year 2025", "calendar year 2026").replace("2026-05-26", "2027-05-25")
AIB_REVISED = AIB.replace("Version 1.0, 2026-05-26", "Version 1.1, 2026-08-14")
EMBER = '<html><head><meta property="article:modified_time" content="2026-11-03T19:06:58+00:00" /></head><body>Yearly Electricity Data</body></html>'
EMBER_NEW = EMBER.replace("2026-11-03", "2027-01-15")
NGA = '<ul><li><a href="/climate-change/publications/national-greenhouse-accounts-factors-2026">National Greenhouse Accounts Factors 2026</a></li><li><a href="/climate-change/publications/national-greenhouse-accounts-factors-2025">2025</a></li></ul>'
NGA_NEW = NGA.replace("<ul>", '<ul><li><a href="/climate-change/publications/national-greenhouse-accounts-factors-2027">2027</a></li>', 1)
ECCC = "<main><h1>Canada's Official Greenhouse Gas Inventory</h1><ul><li><strong>Date Published:</strong> 2026-04-14</li><li><strong>Temporal Coverage:</strong> 1990-01-01 to 2024-12-31</li></ul></main>"
ECCC_NEW = ECCC.replace("2026-04-14", "2027-04-13").replace("to 2024-12-31", "to 2025-12-31")
UBA = '<h3>Publikationen</h3><ul><li><a href="/publikationen/entwicklung-der-spezifischen-treibhausgas-0">Entwicklung der spezifischen Treibhausgas-Emissionen des deutschen Strommix in den Jahren 1990 - 2025</a></li></ul>'
UBA_NEW = UBA.replace("1990 - 2025", "1990 - 2026")
_MOE = "https://policies.env.go.jp/earth/ghg-santeikohyo/files/calc/"
MOE = "".join(f'<a href="{_MOE}{f}">x</a>' for f in ("r08_denki_coefficient_rev4.pdf", "r07_denki_coefficient_rev4.pdf", "r06_denki_coefficient_rev10.pdf", "r05_coefficient_rev4.pdf", "r08_gas_coefficient.pdf"))
MOE_NEW = MOE + f'<a href="{_MOE}r09_denki_coefficient.pdf">x</a>'
MOE_REV = MOE + f'<a href="{_MOE}r08_denki_coefficient_rev5.pdf">x</a>'
PAGES = {"desnz": DESNZ, "egrid": EGRID, "aib": AIB, "ember": EMBER, "nga": NGA, "eccc": ECCC, "uba": UBA, "moe": MOE}
N_SRC = 10                                         # desnz, egrid, aib, ember, cea, nga, eccc, uba, mee, moe


@pytest.fixture(autouse=True)
def _home(tmp_path, monkeypatch):
    monkeypatch.setenv("CRI_HOME", str(tmp_path))
    monkeypatch.delenv("CRI_STORY_OFFLINE", raising=False)


def _fetcher(pages):
    by_url = {SM.BY_ID[k].url: v for k, v in pages.items()}
    by_url[CEA.LISTING_URL] = "<html>Baseline Carbon Dioxide<h4>Version 22.0</h4><span>User Guide – Version 22.0</span><a href=\"/api/cdm-report-version/view-pdf/en/82\">PDF</a></html>"

    def f(url):
        if url not in by_url:
            raise OSError("no page for " + url)
        return by_url[url]
    return f


def _row(rep, i):
    return next(r for r in rep["sources"] if r["id"] == i)


# ── what is held ─────────────────────────────────────────────────────────────

def test_held_editions_are_read_from_the_engine_data():
    assert SM.held_desnz()["edition"] == [2026] and SM.held_egrid()["edition"] == [2023, 2]
    assert SM.held_aib()["edition"] == [2025, "2026-05-26"] and SM.held_ember()["date"] == "2026-09-30"
    assert SM.BY_ID["cea"].held()["edition"] == [22, 0]
    assert SM.BY_ID["nga"].held()["edition"] == [2026] and SM.BY_ID["eccc"].held()["edition"] == [2024, "2026-04-14"]
    assert SM.BY_ID["uba"].held()["edition"] == [2025] and SM.BY_ID["mee"].held()["edition"] == [2023] and SM.BY_ID["moe"].held()["edition"] == [8, 4]
    assert len(SM.SOURCES) == N_SRC


# ── detectors ────────────────────────────────────────────────────────────────

def test_detectors_read_the_newest_edition_and_its_date():
    d = SM.detect_desnz(DESNZ)
    assert d["edition"] == [2026] and d["published"] == "2026-07-31" and d["link"].endswith("conversion-factors-2026")
    e = SM.detect_egrid(EGRID)
    assert e["edition"] == [2023, 2] and e["label"] == "eGRID2023 rev2" and e["published"] == "2025-06-12"
    a = SM.detect_aib(AIB)
    assert a["edition"] == [2025, "2026-05-26"] and a["published"] == "2026-05-26"
    m = SM.detect_ember(EMBER)
    assert m["edition"] == ["2026-11-03"] and m["published"] == "2026-11-03"
    assert SM.detect_nga(NGA)["edition"] == [2026] and SM.detect_nga(NGA)["link"].endswith("factors-2026")
    assert SM.detect_eccc(ECCC)["edition"] == [2024, "2026-04-14"] and SM.detect_eccc(ECCC)["label"] == "NIR (data to 2024)"
    assert SM.detect_uba(UBA)["edition"] == [2025]
    mo = SM.detect_moe(MOE)
    assert mo["edition"] == [8, 4] and mo["label"] == "SHK electricity supplier list R8 rev4"          # the gas list and the r05 file are not electricity supplier lists for a later year
    assert SM.detect_ember(EMBER.replace('property="article:modified_time" content="2026-11-03T19:06:58+00:00"', 'content="2026-11-03T19:06:58+00:00" property="article:modified_time"'))["published"] == "2026-11-03"


@pytest.mark.parametrize("fn", [SM.detect_desnz, SM.detect_egrid, SM.detect_aib, SM.detect_ember, SM.detect_nga, SM.detect_eccc, SM.detect_uba, SM.detect_moe])
def test_unrecognised_layout_is_an_error_not_a_guess(fn):
    with pytest.raises(ValueError, match="layout changed"):
        fn("<html><body>Page not found</body></html>")


# ── status logic ─────────────────────────────────────────────────────────────

def test_unverified_until_read_then_current_when_nothing_newer():
    rep = SM.status()
    assert rep["summary"]["unverified"] == N_SRC and all("cannot say" in r["message"] for r in rep["sources"])
    rep = SM.sync(fetch=_fetcher(PAGES))
    assert {r["id"]: r["freshness"] for r in rep["sources"]} == {**{i: "current" for i in SM.BY_ID}, "mee": "unverified"}, [(r["id"], r["message"]) for r in rep["sources"]]
    assert "rolling release" in _row(rep, "ember")["message"] and "newest listed" in _row(rep, "desnz")["message"]
    mee = _row(rep, "mee")
    assert mee["manual"] and "by hand" in mee["message"] and "2023" in mee["message"] and mee["checked_at"] is None     # a source with no readable page is never "current"


@pytest.mark.parametrize("src, page, expect", [
    ("desnz", DESNZ_NEW, "conversion factors 2027"),
    ("egrid", EGRID_NEW, "eGRID2024"),
    ("egrid", EGRID_REV, "eGRID2023 rev3"),
    ("aib", AIB_NEW, "Residual Mix 2026"),
    ("aib", AIB_REVISED, "Residual Mix 2025"),                 # same year, revised after the held copy's date
    ("ember", EMBER_NEW, "page updated 2027-01-15"),
    ("nga", NGA_NEW, "NGA Factors 2027"),
    ("eccc", ECCC_NEW, "NIR (data to 2025)"),
    ("uba", UBA_NEW, "1990-2026"),
    ("moe", MOE_NEW, "R9"),
    ("moe", MOE_REV, "R8 rev5"),
])
def test_newer_edition_is_flagged_with_what_to_do(src, page, expect):
    rep = SM.sync(fetch=_fetcher({**PAGES, src: page}))
    r = _row(rep, src)
    assert r["freshness"] == "newer_available" and expect in r["message"] and "not loaded yet" in r["message"] and "rebuild with scripts/build_" in r["message"]
    assert all(x["freshness"] == "current" for x in rep["sources"] if x["id"] not in (src, "mee"))


def test_each_source_fails_alone_and_says_why():
    f = _fetcher({k: v for k, v in PAGES.items() if k != "egrid"})
    rep = SM.sync(fetch=f)
    e = _row(rep, "egrid")
    assert e["freshness"] == "unverified" and "OSError" in e["last_error"]
    assert _row(rep, "desnz")["freshness"] == "current"
    rep = SM.sync(force=True, fetch=_fetcher({**PAGES, "aib": "<html>moved</html>"}))
    assert "layout changed" in _row(rep, "aib")["last_error"] and _row(rep, "aib")["freshness"] == "unverified"


def test_one_read_a_day_unless_forced():
    calls = []
    base = _fetcher(PAGES)

    def f(u):
        calls.append(u)
        return base(u)
    SM.sync(fetch=f)
    n = len(calls)
    SM.sync(fetch=f)
    assert len(calls) == n
    SM.sync(force=True, fetch=f)
    assert len(calls) > n


def test_stale_check_goes_back_to_unverified(monkeypatch):
    SM.sync(fetch=_fetcher(PAGES))
    import time as _t
    real = _t.time()
    monkeypatch.setattr(SM.time, "time", lambda: real + 9 * 86400)
    rep = SM.status()
    assert _row(rep, "desnz")["freshness"] == "unverified" and "days ago" in _row(rep, "desnz")["message"]


def test_offline_mode_does_not_touch_the_network(monkeypatch):
    monkeypatch.setenv("CRI_STORY_OFFLINE", "1")
    monkeypatch.setattr(SM, "_fetch", lambda u: (_ for _ in ()).throw(AssertionError("network used")))
    assert SM.sync()["summary"]["unverified"] == N_SRC


def test_a_new_source_can_be_registered():
    s = SM.Source("test", "Test source", "Someone", "https://example.org/x", lambda: {"edition": [1], "label": "v1", "date": "2026-01-01"},
                  lambda h: {"edition": [2], "label": "v2", "published": "2026-02-02"}, rebuild="scripts/x.py")
    SM.register(s)
    try:
        rep = SM.sync(fetch=lambda u: "<html></html>" if "example.org" in u else _fetcher(PAGES)(u))
        assert _row(rep, "test")["freshness"] == "newer_available"
    finally:
        SM.SOURCES.remove(s)
        del SM.BY_ID["test"]


# ── provenance, agent, API ───────────────────────────────────────────────────

def test_provenance_lists_the_monitors_for_a_file():
    from cri import provenance as P
    SM.sync(fetch=_fetcher({**PAGES, "desnz": DESNZ_NEW}))
    e = P.entry("carbon_factors.json")
    assert {m["id"] for m in e["monitors"]} == {"desnz", "egrid", "aib", "ember"}
    assert "conversion factors 2027" in e["newer_edition"]
    assert P.entry("cea_baseline.json")["monitors"][0]["id"] == "cea"


def test_answers_state_editions_and_caveats():
    SM.sync(fetch=_fetcher(PAGES))
    a = SM.answer("is the egrid data up to date?")
    assert "eGRID2023 rev2 is the newest listed by US Environmental Protection Agency" in a and "epa.gov/egrid" in a
    assert "AIB" in SM.answer("which residual mix edition do you use") and "2025" in SM.answer("which residual mix edition do you use")
    allr = SM.answer("which data sources are my emission factors from and are they current")
    assert f"{N_SRC} monitored sources" in allr and "UK DESNZ" in allr and "Ember" in allr
    assert SM.answer("what is the capital of France") is None and SM.answer("flood risk for Houston") is None
    # country sources are asked about only when the question joins a country, a grid word and a currency word
    au = SM.answer("is the Australian grid emission factor up to date?")
    assert "Australia NGA Factors" in au and "NGA Factors 2026 is the newest" in au and "Canada" not in au
    assert "China electricity CO2 emission factors" in SM.answer("what is the latest edition of China's electricity emission factors") and "by hand" in SM.answer("is the china grid factor current")
    assert SM.answer("flood risk for a plant in Canada") is None and SM.answer("what is the carbon price in China") is None and SM.answer("Germany GDP growth latest") is None


def test_agent_routes_source_questions(monkeypatch):
    import time
    from cri.resident import agent as A
    monkeypatch.setenv("CRI_DATA_DIR", str(SM._home()))
    monkeypatch.setattr(SM, "_fetch", _fetcher({**PAGES, "desnz": DESNZ_NEW}))
    monkeypatch.setattr(CEA, "fetch_listing", lambda: _fetcher({})(CEA.LISTING_URL))
    ag = A.Agent(twin_builder=lambda *a, **k: {}, analyser=lambda *a, **k: {}, scanner=lambda *a, **k: [], lab=lambda *a, **k: {})
    o = ag.handle("s", "has DESNZ published new conversion factors?")      # a "new?" question reads the release pages now
    assert o["intent"] == "knowledge" and o.get("job")
    t0 = time.time()
    while time.time() - t0 < 10:
        st = A.job_status(o["job"])
        if st["status"] != "running":
            break
        time.sleep(0.1)
    assert st["status"] == "done" and "conversion factors 2027" in st["reply"] and "not loaded yet" in st["reply"]
    o = ag.handle("s", "which residual mix edition do you use?")           # a plain question is answered at once, from the last read
    assert not o.get("job") and "2025" in o["reply"]


def test_api_endpoint():
    from fastapi.testclient import TestClient
    from cri.api.main import app
    c = TestClient(app)
    r = c.get("/data/sources").json()
    assert {s["id"] for s in r["sources"]} == {"desnz", "egrid", "aib", "ember", "cea", "nga", "eccc", "uba", "mee", "moe"} and r["summary"]["unverified"] == N_SRC
    assert c.get("/data/sources", params={"source": "egrid"}).json()["sources"][0]["held"]["label"] == "eGRID2023 rev2"
    assert c.get("/data/sources", params={"source": "nope"}).status_code == 404
