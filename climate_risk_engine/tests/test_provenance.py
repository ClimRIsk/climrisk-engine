from datetime import date

from cri import provenance as P


def test_every_data_file_is_registered():
    a = P.audit()
    assert a["unregistered"] == [], f"data files without a provenance entry: {a['unregistered']}"
    assert a["registry_entries_without_file"] == []


def test_registry_has_no_drift_against_file_headers():
    assert [r["file"] for r in P.audit()["files"] if r.get("drift")] == []


def test_age_and_review_status():
    e = P.entry("epa_ghg_hub_2025.json", today=date(2026, 10, 28))
    assert e["age_days"] == (date(2026, 10, 28) - date(2025, 1, 15)).days
    assert e["freshness"] == "review_due" and e["newer_edition"]
    assert P.entry("imo_dcs_class_aer_2022.json", today=date(2023, 6, 1))["freshness"] == "ok"


def test_unknown_vintage_is_not_invented():
    e = P.entry("ibtracs_since1980.npz")
    assert e["age_days"] is None and e["freshness"] == "unknown"


def test_third_party_files_are_flagged_not_complete():
    a = P.audit()
    assert "carbontlens_tickers.json" in a["third_party"]
    assert P.entry("carbontlens_tickers.json")["documentation"] == "partial"


def test_declared_retrieval_dates_match_files():
    for n in ("epa_ghg_hub_2025.json", "imo_dcs_class_aer_2022.json", "subpart_w_table_w1.json"):
        assert P.entry(n)["retrieved"] == "2026-10-28"


def test_unregistered_file_would_be_detected(tmp_path, monkeypatch):
    (tmp_path / "new_table.json").write_text("{}")
    monkeypatch.setattr(P, "DATA_DIR", tmp_path)
    assert P.audit()["unregistered"] == ["new_table.json"]
