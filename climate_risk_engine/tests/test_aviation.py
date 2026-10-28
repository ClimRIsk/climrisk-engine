import pytest

from cri.lca import aviation as AV


def test_blend_arithmetic():
    r = AV.blend(1_000_000, 0.3, 20.0)
    assert r["blend_lca_g_per_mj"] == pytest.approx(89 * 0.7 + 20 * 0.3)
    assert r["reduction_pct"] == pytest.approx(100 * (1 - (89 * 0.7 + 6) / 89))
    assert r["avoided_vs_fossil_t_co2e"] == pytest.approx(1_000_000 * 0.3 * 69 / 1e6)
    assert AV.blend(10, 0.0, 20)["reduction_pct"] == pytest.approx(0)
    assert AV.blend(10, 1.0, 44.5)["reduction_pct"] == pytest.approx(50.0)


def test_inputs_checked_and_default_not_invented():
    with pytest.raises(AV.AviationError):
        AV.blend(10, 1.5, 20)
    with pytest.raises(TypeError):
        AV.blend(10, 0.5)                    # no default SAF value: certified figure required


def test_jet_combustion_matches_hub():
    assert AV.jet_combustion(1000)["kg_co2"] == pytest.approx(9750.0)
