"""
Offline tests for the asset-intelligence scoring and data-handling logic.
No network: every test feeds synthetic inputs to the pure functions.
"""
import base64

import numpy as np
import pytest

from cri.analysis import asset_intelligence as ai
from cri.connectors import sentinel2_pixels as s2


# ── Sentinel-2 reflectance offset (the bug that inflated NDVI) ────────────────

def _feat(baseline: str, applied: bool) -> dict:
    return {"properties": {"s2:processing_baseline": baseline, "earthsearch:boa_offset_applied": applied},
            "assets": {"red": {"raster:bands": [{"scale": 0.0001, "offset": -0.1}]}}}


def test_offset_not_applied_when_earthsearch_already_harmonised():
    assert s2._band_meta(_feat("05.11", True), "red") == {"scale": 0.0001, "offset": 0.0}


def test_offset_applied_for_new_baseline_not_harmonised():
    assert s2._band_meta(_feat("04.00", False), "red")["offset"] == -0.1


def test_offset_not_applied_for_old_baseline():
    assert s2._band_meta(_feat("02.14", False), "red")["offset"] == 0.0


# ── Flood scoring ─────────────────────────────────────────────────────────────

def test_flood_takes_stronger_sensor_not_the_sum():
    s = {"temporal_scenes_used": 8, "temporal_months_covered": ["01", "02"],
         "water_freq_at_asset": 0.0, "ephemeral_water_within_500m": 0.06}        # optical → 2 pts
    sar = {"available": True, "scenes_used": 10, "months_covered": ["07", "08"],
           "water_freq_at_asset": 0.0, "intermittent_water_within_500m": 0.02}  # radar → 1 pt
    f = ai._flood({"elevation_m": 20}, s, {}, sar)
    assert f["score"] == 2.0
    assert "observed_inundation" in f["mechanisms"]


def test_radar_discounted_at_high_altitude():
    sar = {"available": True, "scenes_used": 10, "months_covered": [],
           "water_freq_at_asset": 0.5, "intermittent_water_within_500m": 0.5}
    f = ai._flood({"elevation_m": 3500}, {}, {}, sar)
    assert f["score"] == 0
    assert any("discounted" in e["metric"] for e in f["evidence"])


def test_repeatedly_flooded_site_is_critical_with_relocation_advice(monkeypatch):
    sar = {"available": True, "scenes_used": 10, "months_covered": [],
           "water_freq_at_asset": 0.4, "intermittent_water_within_500m": 0.9}
    f = ai._flood({"flood_exposure": "MEDIUM", "elevation_m": 10}, {"nearest_water_m": 0}, {}, sar)
    assert f["severity"] == "CRITICAL"


def test_optical_gap_caveat_without_radar():
    s = {"temporal_scenes_used": 6, "temporal_months_covered": ["01", "02", "03", "11", "12"]}
    f = ai._flood({}, s, {}, {"available": False})
    assert "06" in f["caveat"] and "not be seen" in f["caveat"]


def test_dense_urban_caveat():
    f = ai._flood({}, {}, {"built_up_within_250m": 0.9}, {"available": False})
    assert "Street-level waterlogging" in f["caveat"]


# ── Wildfire ──────────────────────────────────────────────────────────────────

def test_alpine_dampener():
    lc = {"flammable_veg_within_1km": 0.75}
    s = {"ndvi_area_mean": 0.2}
    low = ai._wildfire({"elevation_m": 3500}, s, lc)
    high = ai._wildfire({"elevation_m": 700}, s, lc)
    assert high["score"] - low["score"] == pytest.approx(2.0)
    assert ai.SEVERITY_ORDER.index(low["severity"]) < ai.SEVERITY_ORDER.index(high["severity"])


def test_dense_urban_reduces_wildfire():
    f = ai._wildfire({}, {}, {"flammable_veg_within_1km": 0.1, "built_up_within_250m": 0.9})
    assert f["severity"] == "NEGLIGIBLE"


# ── Visual payload packing ───────────────────────────────────────────────────

def test_pack_roundtrip_uint8():
    vals = [-1.0, 0.0, 0.5, None, 1.0]
    p = ai._pack(vals, (1, 5), -1.0, 1.0)
    raw = np.frombuffer(base64.b64decode(p["b64"]), dtype=np.uint8)
    assert raw[3] == 255                                 # nodata sentinel
    dec = p["lo"] + raw[[0, 1, 2, 4]] / 254 * (p["hi"] - p["lo"])
    assert np.allclose(dec, [-1.0, 0.0, 0.5, 1.0], atol=0.01)


def test_pack_uint16_precision():
    vals = [100.0, 4787.4]
    p = ai._pack(vals, (1, 2), 100.0, 4787.4, "uint16")
    raw = np.frombuffer(base64.b64decode(p["b64"]), dtype="<u2")
    dec = p["lo"] + raw / 65534 * (p["hi"] - p["lo"])
    assert np.allclose(dec, vals, atol=0.1)


def test_severity_thresholds():
    assert [ai._sev_from_points(x) for x in (0, 1, 2, 4, 6)] == \
        ["NEGLIGIBLE", "LOW", "MODERATE", "HIGH", "CRITICAL"]


# ── Validated built-up elevation component (area backtest, 3 counties) ────────

def test_low_builtup_elevation_drives_flood_score():
    base = ai._flood({"elevation_m": 20}, {}, {}, {"available": False})
    low = ai._flood({"elevation_m": 20}, {}, {}, {"available": False},
                    {"median_elev_m": 2.1, "basis": "built-up land"})
    mid = ai._flood({"elevation_m": 20}, {}, {}, {"available": False},
                    {"median_elev_m": 4.5, "basis": "built-up land"})
    high = ai._flood({"elevation_m": 20}, {}, {}, {"available": False},
                     {"median_elev_m": 25.0, "basis": "built-up land"})
    assert low["score"] - base["score"] == 3
    assert mid["score"] - base["score"] == 1
    assert high["score"] == base["score"]
    assert any("validated" in e["metric"] for e in low["evidence"])
    assert "Validation status" in low["caveat"]
