import pytest

from cri.carbon import epa_factors as E


def test_every_stationary_row_is_internally_consistent():
    """kg CO2 per unit printed in the Hub must equal heat content x kg CO2 per mmBtu: catches transcription slips."""
    bad = []
    for name, f in E.lib()["table1_stationary"].items():
        calc = f["hhv_mmbtu_per_unit"] * f["kgco2_per_mmbtu"]
        if abs(calc - f["printed_kgco2_per_unit"]) > 0.012 * f["printed_kgco2_per_unit"]:
            bad.append((name, calc, f["printed_kgco2_per_unit"]))
    assert not bad, bad
    assert len(E.lib()["table1_stationary"]) >= 55


def test_natural_gas_and_diesel_by_hand():
    r = E.stationary("Natural Gas", 1000.0, "mmBtu")                         # 53.06 kg CO2, 1.0 g CH4, 0.10 g N2O per mmBtu
    assert r["kg_co2"] == pytest.approx(53060.0)
    assert r["kg_co2e"] == pytest.approx(53060.0 + 1.0 * 1000 / 1000 * 28 + 0.10 * 1000 / 1000 * 265)
    r = E.stationary("Distillate Fuel Oil No. 2", 100.0, "gallon")           # 100 gal x 0.138 = 13.8 mmBtu
    assert r["heat_mmbtu"] == pytest.approx(13.8) and r["kg_co2"] == pytest.approx(13.8 * 73.96)


def test_supplier_heat_content_overrides_default_and_is_stated():
    r = E.stationary("Natural Gas", 1000.0, "scf", hhv_mmbtu_per_unit=0.00105)
    assert r["heat_content_source"] == "supplier value given" and r["heat_mmbtu"] == pytest.approx(1.05)


def test_biogenic_flag_and_mobile_note():
    assert E.stationary("Wood and Wood Residuals", 1.0)["biogenic"] is True
    m = E.mobile_co2("Motor Gasoline", 10.0)
    assert m["kg_co2"] == pytest.approx(87.8) and "CH4 and N2O" in m["excluded"]


def test_electricity_and_transport():
    r = E.electricity("camx", 1000.0)                                         # 436.7 lb/MWh -> kg
    assert r["kg_co2"] == pytest.approx(1000 * 436.7 / 2.2046226218, rel=1e-6)
    assert E.travel_or_transport("Bus", 100.0)["kg_co2"] == pytest.approx(6.6)
    assert E.travel_or_transport("Rail", 10.0, "table8_scope3_transport")["kg_co2"] == pytest.approx(0.21)
    assert E.steam_heat(10.0)["kg_co2"] == pytest.approx(663.3)


def test_errors():
    for call in (lambda: E.stationary("Unobtainium", 1.0), lambda: E.electricity("XXXX", 1.0), lambda: E.stationary("Natural Gas", 1.0, "tonne"),
                 lambda: E.mobile_co2("Water", 1.0), lambda: E.travel_or_transport("Bus", 1.0, "t99")):
        with pytest.raises(E.EpaFactorError):
            call()
