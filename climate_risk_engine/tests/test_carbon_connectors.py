import json

import pytest

from cri import carbon_connectors as CC
from test_carbon_registry import CSV


def test_file_drop_picks_newest_csv_and_stamps_provenance(tmp_path):
    (tmp_path / "old.csv").write_text("Serial Number,Quantity,Vintage Year\nX,1,2010\n")
    import os, time
    os.utime(tmp_path / "old.csv", (time.time() - 100, time.time() - 100))
    (tmp_path / "new.csv").write_text(CSV)
    r = CC.sync(CC.FileDrop(str(tmp_path), "CDM"))
    assert r["provenance"]["n_records"] == 4 and r["provenance"]["source"].endswith("new.csv")
    assert len(r["provenance"]["sha256"]) == 64 and "Not verified" in r["provenance"]["statement"]
    assert r["screen"]["lots"][0]["cer"] is not None


def test_http_json_maps_fields_and_reads_token_from_env(monkeypatch):
    payload = {"data": {"units": [{"unit_serial": "S1", "tonnes": "1,500", "reduction_year": 2018, "reg_date": "2014-02-01", "cp_start": "2014-02-01", "sector": "wind", "state": "issued"}]}}
    seen = {}

    class R:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self, n=-1): return json.dumps(payload).encode()

    def fake(req, timeout=0):
        seen["auth"] = req.get_header("Authorization"); return R()

    monkeypatch.setattr(CC.urllib.request, "urlopen", fake)
    monkeypatch.setenv("REG_TOKEN", "abc")
    c = CC.HttpJson("https://example.test/units", "CDM", "REG_TOKEN", "data.units",
                    {"unit_serial": "serial number", "tonnes": "quantity", "reduction_year": "vintage year", "reg_date": "registration date",
                     "cp_start": "crediting period start", "sector": "project type", "state": "status"})
    r = CC.sync(c)
    assert seen["auth"] == "Bearer abc"
    lot = r["lots"][0]
    assert lot["quantity_t"] == 1500 and lot["vintage"] == 2018 and lot["registered_year"] == 2014
    assert r["screen"]["lots"][0]["cer"]["pathways"][0]["status"] == "may_be_eligible"


def test_refuses_plain_http_and_missing_token(monkeypatch):
    with pytest.raises(ValueError):
        CC.sync(CC.HttpJson("http://example.test/x", "CDM"))
    monkeypatch.delenv("NOPE", raising=False)
    with pytest.raises(ValueError):
        CC.sync(CC.HttpJson("https://example.test/x", "CDM", "NOPE"))
