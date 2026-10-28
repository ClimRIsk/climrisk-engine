"""Port matching. Fixtures are real OpenStreetMap geocoder responses observed on 28 Oct 2026 (class, type, importance, country), trimmed."""
import pytest

from cri.maritime import answer as A, ports as PT


def C(name, cc, country, cl, ty, imp, lat=0.0, lon=0.0):
    return {"name": name, "country_code": cc, "country": country, "class": cl, "type": ty, "importance": imp, "lat": lat, "lon": lon}


OBS = {
    # place search
    "Shanghai": [C("上海市", "cn", "China", "boundary", "administrative", 0.753, 31.2, 121.5), C("上海市, 黄浦区", "cn", "China", "place", "city", 0.753, 31.23, 121.47)],
    "Hamburg": [C("Hamburg, Deutschland", "de", "Germany", "boundary", "administrative", 0.788, 53.55, 10.0), C("Town of Hamburg, Erie County, New York", "us", "United States", "boundary", "administrative", 0.474, 42.7, -78.8)],
    "Santos": [C("Santos, São Paulo", "br", "Brazil", "boundary", "administrative", 0.626, -23.96, -46.33), C("Sántos, Somogy", "hu", "Hungary", "boundary", "administrative", 0.503, 46.4, 17.6)],
    "Valencia": [C("València", "es", "Spain", "boundary", "administrative", 0.731, 39.47, -0.38), C("Valencia, Parroquia Santa Rosa", "ve", "Venezuela", "place", "city", 0.545, 10.16, -68.0)],
    "Singapore": [C("Singapore", "sg", "Singapore", "boundary", "administrative", 0.803, 1.29, 103.85)],
    # "port of X" searches returning junk, as observed
    "port of Shanghai": [C("Go Go Shanghai, Canterbury Street", "gb", "United Kingdom", "shop", "massage", 0.0)],
    "port of Hamburg": [C("Port, Bei der Erholung, St. Pauli", "de", "Germany", "amenity", "restaurant", 0.0)],
    "port of Santos": [C("Port of General Santos", "ph", "Philippines", "landuse", "industrial", 0.08)],
    "port of Valencia": [C("Puerto de Valencia", "es", "Spain", "landuse", "harbour", 0.407, 39.45, -0.33)],
}


def search(q):
    return OBS.get(q, OBS.get(q.split(",")[0], []))


def test_junk_port_of_results_are_never_used():
    for n in ("Shanghai", "Hamburg"):
        r = PT.resolve(n, None, search)
        assert r["status"] == "ok" and r["port"]["country_code"] in ("CN", "DE")
        assert r["port"]["is_port_feature"] is False                      # a place, said so
    h = PT.resolve("Hamburg", None, search)
    assert any("most prominent" in x for x in h["notes"]) and any("Hamburg, United States" in x for x in h["notes"])      # chosen on prominence, says so, tells how to override


def test_santos_philippines_harbour_is_rejected_by_name():
    r = PT.resolve("Santos", None, search, evidence=lambda la, lo: abs(la + 23.96) < 1)
    assert r["status"] == "ok" and r["port"]["country_code"] == "BR"
    assert "only one with a harbour" in " ".join(r["notes"])


def test_valencia_is_ambiguous_without_evidence_and_resolved_with_it():
    r = PT.resolve("Valencia", None, search)
    assert r["status"] == "ambiguous" and {c["country_code"] for c in r["candidates"]} == {"ES", "VE"}
    r = PT.resolve("Valencia", None, search, evidence=lambda la, lo: la > 30)    # only the Spanish one has a harbour nearby
    assert r["status"] == "ok" and r["port"]["country_code"] == "ES"
    r = PT.resolve("Valencia", None, search, evidence=lambda la, lo: True)       # both claim a harbour: still ask
    assert r["status"] == "ambiguous"


def test_country_hint_filters():
    r = PT.resolve("Valencia", "Venezuela", lambda q: OBS["Valencia"])
    assert r["status"] == "ok" and r["port"]["country_code"] == "VE"
    r = PT.resolve("Valencia", "Atlantis", lambda q: OBS["Valencia"])
    assert any("No candidate matched the country" in x for x in r["notes"])


def test_no_harbour_nearby_is_a_warning():
    r = PT.resolve("Singapore", None, search, evidence=lambda la, lo: False)
    assert r["status"] == "ok" and r["port"]["port_evidence"] is False and any("may not be a seaport" in x for x in r["notes"])


def test_not_found_and_unusable_hits():
    assert PT.resolve("Nowhereville", None, lambda q: [])["status"] == "not_found"
    assert PT.resolve("Shanghai", None, lambda q: [C("x", "gb", "UK", "shop", "massage", 0.1)])["status"] == "not_found"


def test_answer_asks_instead_of_guessing_and_offers_buttons():
    r = A.answer("The ship Aurora is docked at Valencia and moving to Hamburg, 5000 NM at 14 knots, burns 45 t/day at 14 knots", lambda q: None, search=search)
    assert r["result"] is None and "more than one country" in r["reply"]
    assert [a["label"] for a in r["actions"]] == ["Valencia, Spain", "Valencia, Venezuela"]
    assert "Valencia, Spain" in r["actions"][0]["text"] and "Hamburg" in r["actions"][0]["text"]


def test_answer_echoes_ports_and_uses_country_suffix():
    r = A.answer("The ship Aurora is docked at Valencia, Spain and moving to Hamburg, 5000 NM at 14 knots, burns 45 t/day at 14 knots", lambda q: None, search=search,
                 evidence=lambda la, lo: True)
    assert r["result"] is not None
    t = r["blocks"][0]
    assert t["head"][0] == "Port as understood" and t["rows"][0][2] == "Spain" and t["rows"][1][2] == "Germany"
    assert "EU ETS and FuelEU scope" in t["caption"]
