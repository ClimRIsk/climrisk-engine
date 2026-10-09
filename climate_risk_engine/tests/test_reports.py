"""Published climate reports as a source: verbatim sentences only, same record layout as the SEC collector, kept apart from real-event counts."""
import json

import pytest

from cri.llmtrain import edgar_cases as EC
from cri.llmtrain import reports as R
from cri.resident import casebase as CB

HTML = b"""<html><head><title>x</title></head><body>
<p>Our Ludwigshafen site added vessels able to operate in low water and an early warning system, which makes longer low-water periods on the Rhine more manageable.</p>
<p>Under the high-warming scenario, about 5% of substations sit in floodplains and we plan to invest $120 million in flood barriers by 2030.</p>
<p>Click here to read the forward-looking statements about flood risk.</p>
<p>We value our people.</p>
<p>and then the storm season continued without a full stop</p>
<p>Wildfire risk to our timberlands is studied each year and the results are not published.</p>
</body></html>"""


def test_extract_keeps_only_complete_hazard_sentences_with_evidence():
    q = R.extract(R.doc_text(HTML, "text/html", "http://x/r.html"))
    texts = [x["quote"] for x in q]
    assert any(t.startswith("Under the high-warming scenario, about 5% of substations") for t in texts)
    assert any("early warning system" in t for t in texts)
    assert not any("Click here" in t or "We value" in t or "full stop" in t for t in texts)
    top = q[0]
    assert top["amounts"] and top["quote"].endswith(".")                    # most evidence first
    assert all(t in HTML.decode() for t in texts)                          # copied word for word


def _tiny_pdf(text: str) -> bytes:
    stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
    objs = [b"<< /Type /Catalog /Pages 2 0 R >>", b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
            b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream", b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    out, offs = b"%PDF-1.4\n", []
    for i, o in enumerate(objs, 1):
        offs.append(len(out))
        out += b"%d 0 obj\n" % i + o + b"\nendobj\n"
    xr = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objs) + 1) + b"".join(b"%010d 00000 n \n" % o for o in offs)
    return out + b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF" % (len(objs) + 1, xr)


def test_pdf_text_is_read():
    pytest.importorskip("pypdf")
    t = R.doc_text(_tiny_pdf("Flood barriers protect the plant and reduce losses by 40% under the high scenario."), "application/pdf", "http://x/r.pdf")
    q = R.extract(t)
    assert q and "Flood barriers protect the plant" in q[0]["quote"] and q[0]["amounts"]


def test_collect_writes_edgar_layout_and_reports_failures(tmp_path, monkeypatch):
    def fake(url, timeout=60):
        if "bad" in url:
            raise OSError("boom")
        return HTML, "text/html"
    monkeypatch.setattr(R, "_fetch", fake)
    monkeypatch.setattr(R.time, "sleep", lambda s: None)
    seeds = [{"company": "Acme Corp", "year": 2024, "url": "http://x/a.html", "region": "Germany"}, {"company": "Bad Co", "year": 2024, "url": "http://x/bad.html"}]
    s = R.collect(tmp_path, seeds)
    assert s["documents_with_quotes"] == 1 and s["failed"] == ["Bad Co 2024"]
    rec = json.loads((tmp_path / "report_cases.jsonl").read_text().splitlines()[0])
    assert {"event", "event_name", "hazard", "region", "company", "cik", "form", "filed", "url", "quotes"} <= set(rec) and rec["form"] == "climate report"


def _write_reports(tmp_path, n=5):
    d = tmp_path / "reports"
    d.mkdir()
    recs = [{"event": R.EVENT, "event_name": R.EVENT_NAME, "hazard": "physical risk disclosure", "region": "X", "company": f"Firm{chr(65 + i)} Holdings", "cik": f"firm{i}", "form": "climate report", "filed": "2024",
             "url": f"http://x/{i}", "quotes": [{"quote": f"Flood barriers protect the Firm{chr(65 + i)} plant and the company plans to invest $1{i} million by 2030.", "amounts": [f"$1{i} million"], "response": ["operational"]},
                                                  {"quote": f"Firm{chr(65 + i)} studies hurricane scenarios and reinsurance cover for each site every year.", "amounts": [], "response": ["transfer"]}]} for i in range(n)]
    (d / "report_cases.jsonl").write_text("\n".join(json.dumps(r) for r in recs))
    return d / "report_cases.jsonl"


def test_index_includes_reports_but_does_not_count_them_as_events(tmp_path, monkeypatch):
    (tmp_path / "edgar").mkdir()
    ed = tmp_path / "edgar" / "edgar_cases.jsonl"
    ed.write_text(json.dumps({"event": "ian-2022", "event_name": "Hurricane Ian", "hazard": "tropical cyclone", "region": "United States", "company": "Gulf Realty", "cik": "1", "form": "8-K", "filed": "2022-11-01",
                              "url": "http://sec/1", "quotes": [{"quote": "Hurricane Ian caused $12 million of damage to our Fort Myers properties.", "amounts": ["$12 million"], "response": []}]}))
    rp = _write_reports(tmp_path)
    out = tmp_path / "idx.jsonl"
    r = CB.build(tmp_path / "none.json", ed, out)                                 # reports found next to the SEC file automatically
    assert r["filings"] == 1 + 5
    monkeypatch.setenv("CRI_CASE_INDEX", str(out))
    st = CB.stats()
    assert st["company_event_cases"] == 1 and st["company_climate_reports"] == 5
    ps = CB.retrieve("What did FirmA Holdings say about flood barriers and physical climate risk?", 4)
    assert ps and ps[0]["text"].startswith("FirmA Holdings, climate report 2024:") and ps[0]["rec"] == "filing"
    assert CB.check(CB.compose("What did FirmA Holdings say about flood barriers?", ps), ps)["ok"]


def test_training_rows_use_report_wording(tmp_path):
    rows = EC.build_rows(_write_reports(tmp_path), seed=3, per_case=1)
    assert rows and all(r["split"] in ("train", "valid") for r in rows)
    joined = json.dumps(rows[0])
    assert "climate report 2024:" in joined and "Form climate report" not in joined
    assert any(r["meta"]["type"] == "compare" for r in rows)
