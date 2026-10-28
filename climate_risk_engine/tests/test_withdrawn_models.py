"""Guards for models withdrawn on 28 Oct 2026 because their numbers were invented or circular. If one returns, it must come back with a source."""
import pytest

from cri.climate import physical_risk as P
from cri.ml.risk_predictor import ModelWithdrawn, RiskPredictor


def test_physical_risk_refuses_country_placeholders():
    with pytest.raises(P.PhysicalRiskInputError) as e:
        P.assess_physical_risk("X", "steel", hq_country="IN", annual_revenue_usd_m=1000, ebitda_usd_m=150)
    assert "flood_baseline_depth_m" in e.value.missing and "asset_replacement_value_usd_m" in e.value.missing


def test_physical_risk_placeholders_are_labelled_and_have_no_invented_bands():
    r = P.assess_physical_risk("X", "steel", hq_country="IN", annual_revenue_usd_m=1000, ebitda_usd_m=150, allow_placeholders=True)
    assert r.confidence_tier == "PLACEHOLDER"
    assert r.eal_p10_usd_m is None and r.eal_p90_usd_m is None and r.npv_p10_usd_m is None and r.npv_p90_usd_m is None
    assert r.eal_tipping_point_stress_usd_m is None and r.npv_tipping_point_stress_usd_m is None
    txt = " ".join(r.model_agreement_notes + r.data_sources)
    assert "34 GCMs" not in txt and "NEX-GDDP" not in txt and "PLACEHOLDER" in txt


def test_physical_risk_with_supplied_inputs_runs_and_is_not_placeholder():
    r = P.assess_physical_risk("X", "steel", hq_country="IN", annual_revenue_usd_m=1000, ebitda_usd_m=150, flood_baseline_depth_m=0.8, heat_baseline_days=30,
                               water_stress_index=3.0, asset_replacement_value_usd_m=1500, elevation_m=40)
    assert r.confidence_tier != "PLACEHOLDER"
    assert r.eal_p10_usd_m is None  # still no invented bands


def test_synthetic_label_trajectory_model_is_withdrawn():
    with pytest.raises(ModelWithdrawn):
        RiskPredictor().predict(base_cri_score=40, sector="steel", jurisdiction="IN", revenue_usd_m=1000, scope1_mt_co2e=1, eal_p50_usd_m=5, wacc_adjusted=0.09)


def test_no_module_attribute_for_invented_tables():
    assert not hasattr(P, "_MODEL_AGREEMENT") and not hasattr(P, "_TIPPING_POINT_MULTIPLIER")
