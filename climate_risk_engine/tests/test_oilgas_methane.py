import pytest

from cri.carbon import oilgas_methane as OG


def test_ideal_gas_density_is_the_expected_magnitude():
    assert OG.kg_ch4_per_scf_pure() == pytest.approx(0.0192, rel=0.01)      # ~0.0192 kg/scf at 60 degF, 14.7 psia


def test_population_arithmetic():
    r = OG.population("pneumatic_gas_service", "Continuous High Bleed Pneumatic Device Vents", 10, 8760, 0.9)
    scf = 21 * 10 * 8760
    assert r["whole_gas_scf"] == pytest.approx(scf)
    assert r["ch4_kg"] == pytest.approx(scf * 0.9 * OG.kg_ch4_per_scf_pure())
    assert r["co2e_kg"] == pytest.approx(r["ch4_kg"] * 28) and "AR5" in r["gwp_basis"]
    assert OG.population("gathering_pipelines", "Cast Iron", 2, 100, 0.9, gwp_ch4=29.8)["gwp_basis"] == "given by you"


def test_no_default_composition_and_unknown_entry_rejected():
    with pytest.raises(OG.MethaneError):
        OG.population("gathering_pipelines", "Cast Iron", 1, 1, 0)
    with pytest.raises(OG.MethaneError):
        OG.population("gathering_pipelines", "Moon Pipe", 1, 1, 0.9)
    with pytest.raises(TypeError):
        OG.population("gathering_pipelines", "Cast Iron", 1, 1)


def test_table_transcription_spot_checks():
    rows = OG.lib()["rows"]
    assert len(rows) == 22
    v = lambda g, c: next(r for r in rows if r["group"] == g and r["component"] == c)["scf_whole_gas_per_hour_per_unit"]
    assert v("major_equipment_gas_service", "Wellhead") == 8.87 and v("major_equipment_crude_service", "Meters/Piping") == 12.4
    assert v("gathering_pipelines", "Plastic/Composite") == 0.28


def test_facility_sums_and_says_what_it_skips():
    f = OG.facility([{"group": "major_equipment_gas_service", "component": "Compressor", "units": 2, "hours": 100},
                     {"group": "major_equipment_gas_service", "component": "Heater", "units": 1, "hours": 100}], 0.8)
    assert f["ch4_kg"] == pytest.approx((13.8 * 2 + 5.22) * 100 * 0.8 * OG.kg_ch4_per_scf_pure())
    assert any("flaring" in x for x in f["not_counted"])
