"""Standards watch: GHG Protocol, ISO 14067, IFRS S2, ESRS, CBAM and PCAF, read from the publishers' own pages on 6 Nov 2026."""
import importlib.util
import json
from pathlib import Path

import pytest

import test_source_monitor as T
from cri.climate import cbam as CB
from cri.connectors import source_monitor as SM, standards_watch as SW
from cri.finance.policy import EU_CBAM_FREE_SHARE

ROOT = Path(__file__).resolve().parents[1]
_home = T._home          # the monitor's state goes to a temp dir


def _build():
    spec = importlib.util.spec_from_file_location("build_standards_watch", ROOT / "scripts" / "build_standards_watch.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# ── the data and its build checks ────────────────────────────────────────────

def test_seed_file_is_exactly_what_the_checked_build_script_writes():
    assert _build().build() == json.loads((ROOT / "src" / "cri" / "data" / "standards_watch.json").read_text())


def test_every_item_has_primary_sources_an_engine_state_and_what_was_not_assessed():
    assert SW.ids() == ["ghgp", "iso14067", "ifrs_s2", "esrs", "cbam", "pcaf"]
    for it in SW.items():
        assert it["sources"] and all(s["url"].startswith("https://") for s in it["sources"])
        assert it["engine"]["state"] in ("matches", "partly", "behind") and it["not_assessed"] and it["monitors"]
        assert all(m in SM.BY_ID for m in it["monitors"])                             # every fingerprint belongs to a registered monitor


def test_build_script_rejects_bad_records(monkeypatch):
    import copy
    b = _build()
    items = copy.deepcopy(b.ITEMS)
    items[0]["key_dates"][0]["date"] = "2027-03"                                        # marked done, but in the future
    with pytest.raises(AssertionError, match="marked done"):
        b._check(items)
    items = copy.deepcopy(b.ITEMS)
    items[0]["key_dates"][-1]["done"] = True                                            # a planned 2028 date marked done
    with pytest.raises(AssertionError):
        b._check(items)
    items = copy.deepcopy(b.ITEMS)
    items[4]["key_values"]["cbam_factor_on_free_allocation"]["2026"] = 1.0              # differs from finance/policy.py
    with pytest.raises(AssertionError):
        b._check(items)
    items = copy.deepcopy(b.ITEMS)
    items[1]["status"] = "made_up"
    with pytest.raises(AssertionError, match="unknown status"):
        b._check(items)
    items = copy.deepcopy(b.ITEMS)
    items[2]["sources"] = []
    with pytest.raises(AssertionError, match="primary source"):
        b._check(items)


def test_what_the_publishers_say_on_6_nov_2026():
    g = SW.item("ghgp")
    assert g["status"] == "under_revision" and "Q2 2027" in g["headline"] and "Q4 2028" in g["headline"]
    dates = {d["date"]: d for d in g["key_dates"]}
    assert dates["2027-Q2"]["done"] is False and dates["2028-Q4"]["done"] is False and dates["2026-07-29"]["done"] is True
    assert "1,100" in dates["2026-07-29"]["event"] and "56 countries" in dates["2026-07-29"]["event"]
    iso = SW.item("iso14067")
    assert "90.92" in iso["headline"] and "confirmed in 2024" in iso["headline"] and "paywalled" in iso["engine"]["note"]
    ifrs = SW.item("ifrs_s2")
    assert "1 January 2027" in ifrs["headline"] and ifrs["engine"]["state"] == "partly"
    esrs = SW.item("esrs")
    assert esrs["status"] == "adopted_pending_scrutiny" and "3 Jul 2026" in esrs["headline"] and any("entered into force" in x for x in esrs["not_assessed"])
    cb = SW.item("cbam")
    assert cb["key_values"]["certificate_price_eur_t_2026"] == {"Q1": 75.36, "Q2": 75.28, "Q3": 82.32}
    assert cb["key_values"]["authorised_declarant_threshold_t"] == 50
    p = SW.item("pcaf")
    assert p["engine"]["state"] == "behind" and "3rd edition" in p["headline"] and p["monitors"]["pcaf"]["edition"] == [2, 2022]


# ── CBAM: the engine's calculator agrees with the regulation ─────────────────

def test_cbam_factor_schedule_is_the_directives_and_the_calculator_uses_it():
    fac = SW.item("cbam")["key_values"]["cbam_factor_on_free_allocation"]
    assert {int(k): v for k, v in fac.items()} == EU_CBAM_FREE_SHARE
    assert fac["2026"] == 0.975 and fac["2030"] == 0.515 and fac["2033"] == 0.14 and fac["2034"] == 0.0
    r = CB.assess_cbam_exposure("Acme", "steel", 1000, 0.2, production_tonnes=1_000_000, origin_country="IN")
    yrs = {y["year"]: y for y in r.scenarios["cbam_trajectory"]["years"]}
    assert [yrs[y]["phase_in_pct"] for y in (2026, 2027, 2030, 2034)] == [2.5, 5.0, 48.5, 100.0]


def test_cbam_2026_cost_is_the_phased_share_at_the_published_price():
    r = CB.assess_cbam_exposure("Acme", "steel", 1000, 0.2, production_tonnes=1_000_000, origin_country="default")
    y26 = r.scenarios["cbam_trajectory"]["years"][0]
    emb = y26["eu_embedded_co2e_t"]                                                      # 200,000 t x 1.85
    mean = round((75.36 + 75.28 + 82.32) / 3, 2)
    assert emb == pytest.approx(370_000) and y26["eu_ets_price_eur_t"] == pytest.approx(round(mean))
    assert y26["cbam_cost_eur_m"] == pytest.approx(emb * 0.025 * mean / 1e6, abs=0.01)   # about EUR 0.72m, not the EUR 22m of a 100% / EUR 60 assumption
    assert y26["cbam_cost_eur_m"] < 1.0
    last = r.scenarios["cbam_trajectory"]["years"][-1]
    assert last["phase_in_pct"] == 100 and last["cbam_cost_eur_m"] == pytest.approx(last["eu_embedded_co2e_t"] * last["net_cbam_price_eur_t"] / 1e6, abs=0.01)
    assert any("scenario assumption" in g for g in r.data_gaps) and "2.5% in 2026" in r.methodology


# ── detectors on page fixtures shaped like the real pages ────────────────────

def test_detectors_read_status_from_the_publishers_pages():
    g = SM.detect_ghgp(T.GHGP)
    assert g["edition"] == ["2026-07-29"] and g["published"] == "2026-07-29"
    i = SM.detect_iso14067(T.ISO)
    assert i["edition"] == ["2018", "90.92", "2024"] and "last confirmed 2024" in i["label"]
    s = SM.detect_ifrs_s2(T.IFRS)
    assert s["edition"] == ["2025-12", "2027-01-01"]
    pr = SM.detect_cbam_price(T.CBAM_PRICE)
    assert pr["edition"] == [3] and "latest Q3: EUR 82.32" in pr["label"] and pr["published"] == "2026-10-05"
    law = SM.detect_cbam_law(T.CBAM_LAW)
    assert law["edition"] == ["2026-08-10", 14]                                          # 13 implementing acts + 1 delegated act
    assert SM.detect_pcaf(T.PCAF)["edition"] == [3, 2025] and SM.detect_pcaf(T.PCAF_HELD)["edition"] == [2, 2022]


@pytest.mark.parametrize("fn", [SM.detect_ghgp, SM.detect_iso14067, SM.detect_ifrs_s2, SM.detect_cbam_price, SM.detect_cbam_law, SM.detect_pcaf])
def test_unrecognised_layout_is_an_error_not_a_guess(fn):
    with pytest.raises(ValueError, match="layout changed"):
        fn("<html><body>Page not found</body></html>")


def test_fingerprints_match_what_the_fixtures_show():
    """The recorded fingerprint is exactly what the detector reads from the page as it was on 6 Nov 2026 (except PCAF, where the engine is behind by design)."""
    for mid, page in (("ghgp", T.GHGP), ("iso14067", T.ISO), ("ifrs_s2", T.IFRS), ("cbam_price", T.CBAM_PRICE), ("cbam_law", T.CBAM_LAW)):
        assert SM.BY_ID[mid].held()["edition"] == SM.BY_ID[mid].detect(page)["edition"], mid


# ── monitor behaviour ────────────────────────────────────────────────────────

def test_standards_are_current_when_their_pages_match_the_record():
    rep = SM.sync(fetch=T._fetcher(T.PAGES))
    for mid in ("ghgp", "iso14067", "ifrs_s2", "cbam_price", "cbam_law", "pcaf"):
        assert T._row(rep, mid)["freshness"] == "current", mid
    esrs = T._row(rep, "esrs")
    assert esrs["freshness"] == "unverified" and esrs["manual"] and "Official Journal" in esrs["message"]


def test_a_changed_status_page_asks_for_a_re_read_not_a_silent_update():
    for mid, page, expect in (("ghgp", T.GHGP.replace("2026.07.29", "2027.03.15"), "2027-03-15"),
                              ("iso14067", T.ISO.replace("[90.92]", "[60.60]"), "stage 60.60"),
                              ("ifrs_s2", T.IFRS.replace("1 January 2027", "1 January 2028"), "effective 2028-01-01"),
                              ("cbam_law", T.CBAM_LAW.replace("10 August 2026", "20 November 2026"), "2026-11-20")):
        r = T._row(SM.sync(force=True, fetch=T._fetcher({**T.PAGES, mid: page})), mid)
        assert r["freshness"] == "newer_available" and "now reads differently" in r["message"] and expect in r["message"] and "build_standards_watch.py" in r["message"], (mid, r["message"])


def test_cbam_price_page_with_q4_published_is_newer():
    q4 = T.CBAM_PRICE.replace("<td>4 January 2027</td><td></td>", "<td>4 January 2027</td><td>80.10</td>")
    r = T._row(SM.sync(fetch=T._fetcher({**T.PAGES, "cbam_price": q4})), "cbam_price")
    assert r["freshness"] == "newer_available" and "4 of 4" in r["message"] and "climate/cbam.py" in r["message"]


def test_pcaf_third_edition_is_flagged_because_the_engine_implements_the_second():
    r = T._row(SM.sync(fetch=T._fetcher({**T.PAGES, "pcaf": T.PCAF})), "pcaf")
    assert r["freshness"] == "newer_available" and "edition 3 (2025)" in r["message"] and "PCAF Part A 2nd edition (2022)" in r["message"]


# ── agent answers ────────────────────────────────────────────────────────────

def test_questions_about_standards_get_status_dates_and_the_engine_state():
    SM.sync(fetch=T._fetcher(T.PAGES))
    a = SM.answer("is the Scope 2 guidance being revised? what is the latest GHG Protocol timeline")
    assert "2027-Q2" in a and "2028-Q4" in a and "In the engine" in a and "carbon/factors.py" in a
    a = SM.answer("what is the latest CBAM certificate price")
    assert "82.32" in a and "2027-01-04" in a and "climate/cbam.py" in a and "forty-fold" in a
    a = SM.answer("which edition of PCAF do you implement - is it the latest?")
    assert "2nd edition (2022)" in a and "3rd edition" in a and "behind" in a
    a = SM.answer("has ESRS been simplified, when does it take effect")
    assert "2027" in a and "Not assessed" in a and "Official Journal" in a
    a = SM.answer("what is the status of ISO 14067, is a revision coming")
    assert "90.92" in a and "paywalled" in a and "lca/cfp.py" in a


def test_task_requests_are_not_hijacked_by_the_standards_watch():
    for q in ("write the CBAM report for Holcim", "how much will CBAM cost Tata Steel", "run PCAF attribution for this loan", "build an ISO 14067 study for cement",
              "ESRS E1 gap check for our disclosure"):
        assert not SM.matches(q), q


def test_knowledge_index_contains_the_standards():
    t = SM.knowledge_text()
    assert "Standards and regulations the engine follows" in t and "GHG Protocol corporate standards" in t and "Part A 2nd edition (2022)" in t


# ── provenance and API ───────────────────────────────────────────────────────

def test_provenance_lists_the_file_and_its_monitors():
    from cri import provenance as P
    e = P.entry("standards_watch.json")
    assert e["status"] == "published" and e["retrieved"] == "2026-11-06" and {m["id"] for m in e["monitors"]} == {"ghgp", "iso14067", "ifrs_s2", "esrs", "cbam_price", "cbam_law", "pcaf"}


def test_api_endpoint():
    from fastapi.testclient import TestClient
    from cri.api.main import app
    c = TestClient(app)
    r = c.get("/data/standards").json()
    assert [i["id"] for i in r["items"]] == SW.ids() and r["items"][0]["monitors"][0]["freshness"] == "unverified"
    one = c.get("/data/standards", params={"item": "cbam"}).json()["items"]
    assert len(one) == 1 and {m["id"] for m in one[0]["monitors"]} == {"cbam_price", "cbam_law"} and one[0]["engine"]["state"] == "matches"
    assert c.get("/data/standards", params={"item": "nope"}).status_code == 404
