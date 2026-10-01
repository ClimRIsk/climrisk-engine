"""Guards against silent errors in the GHG inventory (found in the Oct 2026 audit)."""
import io

import pytest
from openpyxl import load_workbook

from cri.carbon import assurance as A
from cri.carbon import inventory as I

ENT = [{"id": "UK", "name": "UK Ltd", "country": "GB"}]
GAS = {"id": "g1", "entity": "UK", "category": "s1_stationary", "method": "factor",
       "factor_id": "desnz26:fuels-gaseous-fuels-natural-gas-kwh-gross-cv", "quantity": 1000, "evidence": "inv"}


def _inv(records, status=None):
    return I.from_dict({"organisation": "T", "reporting_year": 2026, "entities": ENT, "records": records,
                        "category_status": status or {}})


def test_gas_line_matches_hand_calculation_ar6():
    r = I.calculate(_inv([GAS]))
    hand = 1000 * (0.18194 + 0.00028 / 28 * 29.8 + 0.00009 / 265 * 273) / 1000
    assert r["lines"][0]["t_co2e"] == pytest.approx(hand, abs=1e-9)


def test_other_year_records_are_reported_not_silently_dropped():
    r = I.calculate(_inv([GAS, dict(GAS, id="old", year=2025)]))
    assert r["totals_t"]["scope1"] == pytest.approx(I.calculate(_inv([GAS]))["totals_t"]["scope1"])
    assert any(c["check"] == "period" and "old" in c["message"] for c in r["checks"])


def test_exclusion_needs_reason():
    r = I.calculate(_inv([GAS], {"s3_c4": {"status": "excluded", "reason": " "}}))
    assert any(c["level"] == "error" and "Upstream transportation" in c["message"] for c in r["checks"])


def test_duplicate_ids_rejected():
    with pytest.raises(I.InventoryError, match="duplicate"):
        I.calculate(_inv([GAS, dict(GAS)]))


def test_template_example_rows_are_ignored_and_reported():
    p = A.parse_workbook(A.template_workbook(), "T", 2026)
    assert p["records"] == [] and p["entities"] == []
    assert len(p["intake_notes"]) == 6


def test_non_usd_spend_without_fx_is_rejected():
    wb = load_workbook(io.BytesIO(A.template_workbook()))
    wb["Spend"]["A2"] = "s9"
    wb["Spend"]["G2"] = None
    b = io.BytesIO()
    wb.save(b)
    p = A.parse_workbook(b.getvalue(), "T", 2026)
    p["entities"] = [{"id": "UK1", "name": "x", "country": "GB"}]
    with pytest.raises(I.InventoryError, match="fx_to_usd"):
        I.calculate(I.from_dict(p))
