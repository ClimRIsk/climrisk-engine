"""CEA CO2 Baseline Database: seed integrity, listing parser, guarded ingestion, monitor, factor use, agent answers.
No network: the listing and the PDF are fixtures; pdf_text is stubbed. Fixture numbers are the published Version 22.0 values (Annexure I, Table S)."""
import json

import pytest

from cri.connectors import cea_baseline as C

LISTING = """<html><body><h1>CDM-Co2 Baseline Database</h1>
<div><h4>Version 23.0</h4>
 <div><span>1. Baseline Carbon Dioxide Emission Database Version 23.0</span> <a href="/api/cdm-report-version/view-pdf/en/90">XLSX</a></div>
 <div><span>2. User Guide – Version 23.0</span> <a href="/api/cdm-report-version/view-pdf/en/89">PDF</a></div></div>
<div><h4>Version 22.0</h4>
 <div><span>1. Baseline Carbon Dioxide Emission Database Version 22.0</span> <a href="/api/cdm-report-version/view-pdf/en/83">XLSX</a></div>
 <div><span>2. User Guide – Version 22.0</span> <a href="/api/cdm-report-version/view-pdf/en/82">PDF</a></div></div>
<div><h4>Version 21.0</h4>
 <div><span>1.Baseline Carbon Dioxide Emission Database Version 21.0</span> <a href="/api/cdm-report-version/view-pdf/en/78">PDF</a> <small>(782.43 KB)</small></div>
 <div><span>2.User Guide – Version 21.0</span> <a href="/api/cdm-report-version/view-pdf/en/76">PDF</a></div></div>
<div><h4>Draft Version</h4>
 <div><span>2.Draft - User Guide – Version 1.0 – OUTDATED</span> <a href="/api/cdm-report-version/view-pdf/en/60">PDF</a></div></div>
</body></html>"""
LISTING_22 = LISTING.split('<div><h4>Version 22.0</h4>')[0].split('<div><h4>Version 23.0</h4>')[0] + '<div><h4>Version 22.0</h4>' + LISTING.split('<div><h4>Version 22.0</h4>')[1]


@pytest.fixture(autouse=True)
def _home(tmp_path, monkeypatch):
    monkeypatch.setenv("CRI_HOME", str(tmp_path))
    monkeypatch.delenv("CRI_STORY_OFFLINE", raising=False)


def _guide_text(version="23.0", extra_fy="2026-27", factor=0.660, co2=1180.0, total=1787.88, tamper=None, om=0.960, bm=0.420, cm=0.690, drop=0):
    """A user-guide text in the layout of Annexure I and Table S, built from the seed plus one newer year."""
    s = C.seed()["weighted_average"]
    rows = [f"{fy}{'*' if fy == '2022-23' else ''} {r['co2_mt']} {r['net_generation_conventional_bu']} {r['re_generation_bu']} {r['total_net_generation_bu']:.2f} {r['weighted_average_tco2_per_mwh']:.3f}"
            for fy, r in s.items()]
    rows = rows[drop:]
    if extra_fy:
        rows.append(f"{extra_fy} {co2} 1450.00 337.88 {total} {factor:.3f}")
    if tamper:
        rows[tamper[0]] = tamper[1]
    return (f"CO2 Baseline Database for the Indian Power Sector\nUser Guide\nVersion {version}\nAugust 2027\n\nTable S: ...\n"
            f"Average OM BM CM\n{factor:.3f} {om:.3f} {bm:.3f} {cm:.3f}\n\nAnnexure I\nFY (1) (2) (3) (4) (5) (6)\n" + "\n".join(rows) + "\n")


# ── seed ─────────────────────────────────────────────────────────────────────

def test_seed_is_internally_consistent_and_matches_published_headlines():
    d = C.seed()
    assert d["version"] == "22.0" and d["latest_fy"] == "2025-26" and len(d["weighted_average"]) == 13
    for fy, r in d["weighted_average"].items():
        assert r["co2_mt"] / r["total_net_generation_bu"] == pytest.approx(r["weighted_average_tco2_per_mwh"], abs=0.0006), fy
    for fy, m in d["margins"].items():
        assert (m["om"] + m["bm"]) / 2 == pytest.approx(m["cm"], abs=0.0011), fy
    wa = d["weighted_average"]
    assert wa["2013-14"]["weighted_average_tco2_per_mwh"] == 0.774 and wa["2025-26"]["weighted_average_tco2_per_mwh"] == 0.675
    assert d["margins"]["2025-26"] == {"om": 0.963, "bm": 0.446, "cm": 0.705}
    assert wa["2022-23"]["adjustment_bu_captive_and_cross_border"] == pytest.approx(24.07)     # captive + cross-border adjustment from FY2022-23
    assert wa["2021-22"]["adjustment_bu_captive_and_cross_border"] == 0.0
    assert d["revision_history"][-1]["version"] == "22.0" and "state" in d["scope_statement"]


def test_factor_lookup_and_errors():
    f = C.factor()
    assert (f["fy"], f["tco2_per_mwh"], f["g_per_kwh"], f["version"], f["is_latest"]) == ("2025-26", 0.675, 675.0, "22.0", True)
    assert C.factor("2024-25")["g_per_kwh"] == 710.0 and not C.factor("2024-25")["is_latest"]
    assert C.factor("2021-22", "cm")["tco2_per_mwh"] == 0.915 and any("build margin" in n for n in C.factor("2021-22", "cm")["notes"])
    assert any("captive" in n for n in C.factor("2019-20")["notes"])
    with pytest.raises(KeyError, match="2013-14 to 2025-26"):
        C.factor("2030-31")
    with pytest.raises(KeyError, match="2021-22 to 2025-26"):
        C.factor("2015-16", "bm")
    with pytest.raises(KeyError):
        C.factor(kind="state")


# ── listing parser ───────────────────────────────────────────────────────────

def test_parse_listing_versions_roles_and_ids():
    v = C.parse_listing(LISTING)
    assert [x["version"] for x in v] == ["23.0", "22.0", "21.0"]                 # the draft is not a release
    top = C.newest(v)
    assert top["version"] == "23.0" and top["database"]["id"] == 90 and top["user_guide"]["id"] == 89 and top["user_guide"]["kind"] == "PDF"
    assert v[1]["database"]["kind"] == "XLSX" and v[1]["user_guide"]["url"].endswith("/en/82")
    assert v[2]["database"]["kind"] == "PDF" and v[2]["user_guide"]["id"] == 76     # older versions list the database as a download too
    assert C.parse_listing("<html><body>maintenance</body></html>") == [] and C.parse_listing("") == []


# ── guarded ingestion ────────────────────────────────────────────────────────

def test_parse_user_guide_accepts_a_consistent_newer_release():
    r = C.parse_user_guide(_guide_text())
    assert r["ok"], r["problems"]
    assert r["version"] == "23.0" and r["published"] == "August 2027" and r["latest_fy"] == "2026-27" and r["rows_seen"] == 14
    assert r["weighted_average"]["2026-27"]["weighted_average_tco2_per_mwh"] == 0.660
    assert r["weighted_average"]["2026-27"]["adjustment_bu_captive_and_cross_border"] == pytest.approx(0.0, abs=0.01)
    assert r["margins"] == {"2026-27": {"om": 0.960, "bm": 0.420, "cm": 0.690}}


@pytest.mark.parametrize("kw, expect", [
    (dict(factor=0.700), "CO2/generation"),                                      # factor does not equal CO2 / generation
    (dict(extra_fy=None), "not later than"),                                     # nothing newer than the seed
    (dict(drop=6), "only 8 valid"),                                              # too few rows to trust the table
    (dict(cm=0.750), "mean of OM"),                                              # combined margin is not (OM+BM)/2
    (dict(tamper=(3, "2016-17 888.34 1072.84 81.55 1154.39 0.790")), "2016-17"),  # one corrupted historical row
])
def test_parse_user_guide_rejects_anything_that_fails_a_check(kw, expect):
    r = C.parse_user_guide(_guide_text(**kw))
    assert not r["ok"] and r["weighted_average"] == {} and any(expect in p for p in r["problems"]), r["problems"]


def test_parse_user_guide_needs_cover_version_and_table_s():
    t = _guide_text().replace("Version 23.0", "Edition").replace("Average OM BM CM", "x")
    r = C.parse_user_guide(t)
    assert not r["ok"] and any("version number" in p for p in r["problems"]) and any("Table S" in p for p in r["problems"])


# ── the monitor ──────────────────────────────────────────────────────────────

def test_status_is_unverified_until_the_listing_has_been_read():
    s = C.status()
    assert s["freshness"] == "unverified" and s["version"] == "22.0" and "cannot say it is the newest" in s["message"]


def test_sync_says_current_only_after_reading_the_listing_and_limits_requests():
    calls = []
    s = C.sync(fetch=lambda: calls.append(1) or LISTING_22)
    assert s["freshness"] == "current" and s["latest_listed"] == "22.0" and s["checked_at"]
    C.sync(fetch=lambda: calls.append(1) or LISTING_22)                             # inside the day: no second request
    assert len(calls) == 1
    C.sync(force=True, fetch=lambda: calls.append(1) or LISTING_22)
    assert len(calls) == 2


def test_failed_read_is_reported_not_hidden():
    def boom():
        raise OSError("network is down")
    s = C.sync(fetch=boom)
    assert s["freshness"] == "unverified" and "OSError" in s["last_error"] and s["checked_at"] is None
    s = C.sync(force=True, fetch=lambda: "<html>a captive portal</html>")
    assert "no versions recognised" in s["last_error"]


def test_unreadable_guide_is_reported_as_not_accepted():
    s = C.sync(fetch=lambda: LISTING, download=lambda url: b"%PDF-fake" if url.endswith("/89") else b"")
    assert s["freshness"] == "newer_available"                                    # no PDF reader result yet: pdf_text is real and the bytes are not a PDF
    assert s["ingest_result"]["ok"] is False and s["version"] == "22.0"


def test_ingestion_overlay_replaces_the_seed(monkeypatch):
    monkeypatch.setattr(C, "pdf_text", lambda data: _guide_text())
    s = C.sync(fetch=lambda: LISTING, download=lambda url: b"%PDF-fake")
    assert s["freshness"] == "current" and s["version"] == "23.0" and s["source"] == "ingested" and s["latest_fy"] == "2026-27"
    f = C.factor()
    assert (f["fy"], f["tco2_per_mwh"], f["version"]) == ("2026-27", 0.660, "23.0")
    assert C.factor("2024-25")["tco2_per_mwh"] == 0.710                            # earlier years are kept
    assert json.loads(C._overlay_path().read_text())["source_sha256"]


def test_ingestion_failure_keeps_the_seed_and_says_why(monkeypatch):
    monkeypatch.setattr(C, "pdf_text", lambda data: _guide_text(factor=0.700))
    s = C.sync(fetch=lambda: LISTING, download=lambda url: b"%PDF-fake")
    assert s["freshness"] == "newer_available" and s["version"] == "22.0" and C.factor()["tco2_per_mwh"] == 0.675
    assert "not accepted" in s["message"] and "CO2/generation" in s["message"] and "Version 23.0" in s["message"]
    assert not C._overlay_path().exists()


def test_guide_that_names_a_different_version_is_not_accepted(monkeypatch):
    monkeypatch.setattr(C, "pdf_text", lambda data: _guide_text(version="22.5"))
    s = C.sync(fetch=lambda: LISTING, download=lambda url: b"%PDF-fake")
    assert s["version"] == "22.0" and "listing entry" in " ".join(s["ingest_result"]["problems"])


def test_offline_mode_never_touches_the_network(monkeypatch):
    monkeypatch.setenv("CRI_STORY_OFFLINE", "1")
    monkeypatch.setattr(C, "fetch_listing", lambda: (_ for _ in ()).throw(AssertionError("network used")))
    assert C.sync()["freshness"] == "unverified"


# ── use in the engine ────────────────────────────────────────────────────────

def test_india_scope2_uses_cea_and_other_countries_do_not():
    from cri.carbon import factors as F
    loc = F.electricity_location("IND")
    assert loc["g_per_kwh"] == 675.0 and "CEA" in loc["source"] and loc["gwp_basis"] == "CO2 only" and "latest published" in loc["note"]
    assert F.electricity_location("IND", fy="2023-24")["g_per_kwh"] == 727.0 and F.electricity_location("IND", fy="2023-24")["note"] is None
    assert F.electricity_location("IN")["g_per_kwh"] == 675.0
    with pytest.raises(KeyError, match="no weighted_average for FY 2031-32"):
        F.electricity_location("IND", fy="2031-32")
    assert "Ember" in F.electricity_location("BGD")["source"]


def test_inventory_india_line_cites_cea_fiscal_year_and_scales_with_it():
    from cri.carbon import inventory as I

    def run(**kw):
        return I.calculate(I.from_dict({"organisation": "x", "reporting_year": 2025, "gwp": "AR6", "entities": [{"id": "E", "name": "Plant", "country": "IND"}],
                                        "records": [dict({"id": "r", "entity": "E", "category": "s2_electricity", "method": "electricity", "kwh": 1_000_000, "evidence": "meter"}, **kw)]}))
    a, b = run(), run(grid_fy="2019-20")
    assert a["totals_t"]["scope2_location"] == pytest.approx(675.0) and b["totals_t"]["scope2_location"] == pytest.approx(713.0)
    line = a["lines"][0]
    assert "CEA" in line["factor"]["source"] and "FY2025-26" in line["factor"]["source"]
    assert not any("not restated" in c["message"] for c in a["checks"])            # CO2-only factor: no GWP restatement question
    assert any("grid_fy" in c["message"] for c in a["checks"])


# ── what the agent says ──────────────────────────────────────────────────────

def test_answers_are_sourced_versioned_and_do_not_invent_state_factors():
    a = C.answer("What is the CEA grid emission factor for 2024-25?")
    assert "0.710" in a and "710 g" in a and "v22.0" in a and "view-pdf/en/82" in a and "0.727" in a
    s = C.answer("what is CEA's emission factor for Tamil Nadu")
    assert "no state-wise" in s and "not a CEA value" in s
    m = C.answer("CEA combined margin 2025-26")
    assert "0.705" in m and "combined margin" in m
    assert C.answer("is there a newer CEA baseline database version published?").startswith("Using Version 22.0")
    assert C.answer("what is the capital of France") is None and C.answer("transformer flood risk in Houston") is None


def test_knowledge_text_carries_the_numbers_and_the_status():
    t = C.knowledge_text()
    assert "FY2025-26 0.675" in t and "OM 0.963, BM 0.446, CM 0.705" in t and "no state-wise" in t and "Version 22.0" in t
    assert "== CEA state-wise emission factors ==" in t and "ISO 14067" in t


def test_agent_routes_cea_questions_and_checks_on_request(monkeypatch):
    from cri.resident import agent as A
    monkeypatch.setenv("CRI_DATA_DIR", str(C._home()))
    ag = A.Agent(twin_builder=lambda *a, **k: {}, analyser=lambda *a, **k: {}, scanner=lambda *a, **k: [], lab=lambda *a, **k: {})
    o = ag.handle("s", "what is CEA's latest weighted average grid emission factor in India?")
    assert "0.675" in o["reply"] and o["blocks"] and "cea.nic.in" in json.dumps(o["blocks"])
    monkeypatch.setattr(C, "fetch_listing", lambda: LISTING)
    monkeypatch.setattr(C, "pdf_text", lambda data: _guide_text())
    monkeypatch.setattr(C, "_get", lambda *a, **k: b"%PDF-fake")
    o = ag.handle("s", "has CEA published a new version of the baseline database?")
    assert o.get("job")
    import time
    t0 = time.time()
    while time.time() - t0 < 10:
        st = A.job_status(o["job"])
        if st["status"] != "running":
            break
        time.sleep(0.1)
    assert st["status"] == "done" and "Version 23.0" in st["reply"] and "August 2027" in st["reply"]
    assert "0.660" in ag.handle("s", "CEA grid emission factor for 2026-27")["reply"]


def test_provenance_registers_the_file_and_flags_a_newer_edition():
    from cri import provenance as P
    e = P.entry("cea_baseline.json")
    assert e["documentation"] == "complete" and e["status"] == "published" and "drift" not in e and "newer_edition" not in e
    C.sync(fetch=lambda: LISTING, download=lambda url: b"x")
    assert "Version 23.0" in P.entry("cea_baseline.json")["newer_edition"]
    assert P.audit()["unregistered"] == []


def test_real_pdf_round_trip_through_the_text_extractors():
    """A real PDF (cover, Table S, Annexure I as a ruled table) goes through pdf_text and the parser, with pdftotext and with pypdf alone."""
    pytest.importorskip("reportlab")
    pypdf = pytest.importorskip("pypdf")
    import io
    import tempfile
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Table
    st = getSampleStyleSheet()
    buf = io.BytesIO()
    rows = [["FY", "Total CO2 (MT)", "Conventional (BU)", "RE (BU)", "Total (BU)", "tCO2/MWh"], ["(1)", "(2)", "(3)", "(4)", "(5)", "(6)=(2)/(5)"]]
    for fy, r in C.seed()["weighted_average"].items():
        rows.append([fy + ("*" if fy == "2022-23" else ""), str(r["co2_mt"]), str(r["net_generation_conventional_bu"]), str(r["re_generation_bu"]),
                     f"{r['total_net_generation_bu']:.2f}", f"{r['weighted_average_tco2_per_mwh']:.3f}"])
    rows.append(["2026-27", "1180.0", "1450.00", "337.88", "1787.88", "0.660"])
    grid = [("GRID", (0, 0), (-1, -1), 0.5, (0, 0, 0)), ("FONTSIZE", (0, 0), (-1, -1), 6)]
    SimpleDocTemplate(buf).build([Paragraph("User Guide", st["Title"]), Paragraph("Version 23.0", st["Normal"]), Paragraph("August 2027", st["Normal"]), PageBreak(),
                                  Table([["Average", "OM", "BM", "CM"], ["0.660", "0.960", "0.420", "0.690"]], style=grid), PageBreak(),
                                  Paragraph("Annexure I", st["Heading1"]), Table(rows, colWidths=[60, 80, 80, 70, 80, 80], style=grid)])
    data = buf.getvalue()
    r = C.parse_user_guide(C.pdf_text(data))
    assert r["ok"], r["problems"]
    assert r["version"] == "23.0" and r["rows_seen"] == 14 and r["margins"]["2026-27"]["cm"] == 0.69
    with tempfile.NamedTemporaryFile(suffix=".pdf") as f:
        f.write(data)
        f.flush()
        txt = "\n".join(p.extract_text() for p in pypdf.PdfReader(f.name).pages)
    assert C.parse_user_guide(txt)["ok"]


def test_api_endpoints():
    from fastapi.testclient import TestClient
    from cri.api.main import app
    c = TestClient(app)
    r = c.get("/data/cea").json()
    assert r["factor"]["tco2_per_mwh"] == 0.675 and r["status"]["version"] == "22.0" and r["status"]["freshness"] == "unverified"
    assert c.get("/data/cea", params={"fy": "2020-21", "kind": "cm"}).status_code == 404            # margins start at FY2021-22
    assert c.get("/data/cea", params={"fy": "2020-21"}).json()["factor"]["g_per_kwh"] == 703.0
    g = c.get("/carbon/grid-factor", params={"country": "IND", "fy": "2023-24"}).json()
    assert g["location"]["g_per_kwh"] == 727.0 and "CEA" in g["location"]["source"]
