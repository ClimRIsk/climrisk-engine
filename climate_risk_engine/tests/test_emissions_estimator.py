"""Offline tests for the observed-first, data-calibrated emissions estimator (v2)."""
import importlib.util
import sys
from pathlib import Path

import pytest

_P = Path(__file__).resolve().parents[1] / "src" / "cri" / "ml" / "emissions_estimator.py"
_spec = importlib.util.spec_from_file_location("cri_ee", _P)
ee = importlib.util.module_from_spec(_spec)
sys.modules["cri_ee"] = ee
_spec.loader.exec_module(ee)
E = ee.EmissionsEstimator()


def test_runtime_files_ship_and_load():
    assert ee._calibration().get("sectors"), "emissions_calibration.json missing"
    assert len(ee._observed_index()[0]) > 200, "observed_emissions_us.json missing"


def test_observed_company_by_name_and_ticker_match():
    a = E.estimate(28674, "utilities", company_name="Duke Energy Corp")
    b = E.estimate(28674, "utilities", ticker="duk")
    assert a.scope1_mt_co2e == b.scope1_mt_co2e
    assert a.scope1_coverage == "complete" and "EPA GHGRP" in a.scope1_source
    assert 60 < a.scope1_mt_co2e < 90          # Duke discloses ~72 Mt for 2023


def test_lower_bound_company_uses_observed_floor_not_sector_prior():
    r = E.estimate(171842, "automotive", company_name="General Motors Co")
    assert r.scope1_coverage == "lower_bound"
    assert r.scope1_mt_co2e == r.scope1_lo          # point = observed floor
    assert r.scope1_hi == pytest.approx(r.scope1_lo * 3, rel=1e-3)
    assert r.notes


def test_uncovered_company_gets_measured_wide_interval():
    r = E.estimate(10_000, "utilities")
    assert r.scope1_coverage == "estimate" and "scope1" in r.imputed_fields
    assert r.scope1_hi / r.scope1_lo > 20            # honest ~3.5x typical error, not ±30 %
    assert r.model_confidence < 0.5


def test_caller_value_wins():
    r = E.estimate(28674, "utilities", company_name="Duke Energy Corp", scope1_known=10.0)
    assert r.scope1_mt_co2e == 10.0 and r.scope1_source == "caller-supplied"


def test_jurisdiction_is_optional():
    # api/main.py used to call without jurisdiction → TypeError swallowed → no emissions
    assert E.estimate(revenue_usd_m=5000, sector="cement").scope1_mt_co2e > 0


def test_no_false_training_claim():
    doc = _P.read_text()
    assert "trained on 14,000" not in doc and "cdp-trained" not in doc
