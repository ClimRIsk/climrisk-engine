"""Offline tests for the company asset register (shipped data, in-memory lookups)."""
import time

from cri.data import asset_register as ar


def test_register_ships_and_loads_fast():
    t = time.time()
    s = ar.stats()
    assert s["n_assets"] > 25000 and s["n_companies"] > 10000
    assert time.time() - t < 2.0


def test_search_is_instant_and_ranked():
    ar.stats()
    t = time.time()
    r = ar.search("Duke Energy")
    assert (time.time() - t) < 0.1
    assert r[0]["match"] == "group"                                  # one-click group first
    ent = [x for x in r if x["match"] != "group"]
    assert ent[0]["name"].upper().startswith("DUKE ENERGY") and ent[0]["n_assets"] > 10


def test_exposure_has_precomputed_screening():
    key = ar.search("NTPC")[0]["key"]
    e = ar.company_exposure([key])
    assert e["found"] and e["n_assets"] > 20
    a = next(x for x in e["assets"] if "Simhadri" in x["name"])
    assert a["cyclone"]["worst"].startswith("Hudhud")          # Vizag, 2014
    assert set(e["summary"]["by_level"]) == set(ar.LEVELS)


def test_below_sea_level_basins_are_not_flagged():
    assert ar._flood_level(-19.0) == "NEGLIGIBLE"                # Baku / Caspian basin
    assert ar._flood_level(1.5) == "HIGH"
    assert ar._flood_level(4.5) == "MODERATE"
    assert ar._flood_level(40.0) == "NEGLIGIBLE"


def test_unknown_company():
    assert ar.search("zzzz nonexistent co") == []
    assert ar.company_exposure(["zzzz"])["found"] is False


# ── found by the worldwide QA run ───────────────────────────────────────────

def test_accents_do_not_break_search():
    assert ar.search("Comision Federal de Electricidad")[0]["n_assets"] >= 40
    assert ar.search("Comisión Federal de Electricidad")[0]["n_assets"] >= 40


def test_state_owned_groups_are_one_click():
    g = next(x for x in ar.search("China Huaneng") if x["match"] == "group")
    assert g["n_entities"] > 50 and ar.company_exposure([g["key"]])["n_assets"] > 90


def test_group_matching_is_not_too_greedy():
    semen = ar.company_exposure(["group:semen"])
    assert set(semen["summary"]["countries"]) == {"IDN"}               # not Norwegian 'sement'
    assert not any("valero" in c.lower() for c in ar.company_exposure(["group:vale"])["companies"])


def test_generic_words_do_not_form_groups():
    assert not any(x["match"] == "group" for x in ar.search("China Power"))
