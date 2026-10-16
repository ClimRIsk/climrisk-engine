"""Exports must carry the same numbers as the API/screen: Excel run export vs POST /runs (every summary figure and every projection cell)."""
import io
import pytest

openpyxl = pytest.importorskip("openpyxl")
from fastapi.testclient import TestClient
from cri.api import main as M

BODY = {"company_id": "bhp", "scenario_id": "nze_2050"}
COLS = ['year', 'revenue', 'opex', 'carbon_cost', 'physical_loss_cost', 'ebitda', 'da', 'ebit', 'nopat', 'transition_capex', 'adaptation_capex',
        'maintenance_capex', 'fcf', 'capex_shock', 'opex_shock', 'gross_var', 'insurance_offset', 'net_var']


def test_excel_export_matches_run_json():
    c = TestClient(M.app)
    j = c.post("/runs", json=BODY).json()
    x = c.post("/runs/export", json=BODY)
    assert x.status_code == 200
    wb = openpyxl.load_workbook(io.BytesIO(x.content))
    S = {r[0]: r[1] for r in wb["Summary"].iter_rows(min_row=2, values_only=True) if r[0]}
    for lab, key, mul, nd in [("NPV of FCF (USD M)", "npv_fcf", 1, 1), ("Enterprise Value (USD M)", "enterprise_value", 1, 1), ("Equity Value (USD M)", "equity_value", 1, 1),
                              ("WACC Used (%)", "wacc_used", 100, 2), ("Gross VaR NPV (USD M)", "gross_var_npv", 1, 1), ("Net VaR NPV (USD M)", "net_var_npv", 1, 1),
                              ("Peak Annual Gross VaR (USD M)", "peak_annual_gross_var", 1, 1)]:
        assert abs(S[lab] - round(j[key] * mul, nd)) < 10 ** -nd, lab
    rows = list(wb["Projections"].iter_rows(min_row=2, values_only=True))
    assert len(rows) == len(j["years"])
    for row, y in zip(rows, j["years"]):
        for k, v in zip(COLS, row):
            assert abs((v or 0) - (y[k] or 0)) <= 0.06, (y["year"], k)
