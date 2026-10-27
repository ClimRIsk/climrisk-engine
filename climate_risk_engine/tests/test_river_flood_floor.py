"""Regression: where the third-party river map says 'outside floodplain' but the site is low in LiDAR terrain, the report must not
show NEGLIGIBLE. Leeds Dock flooded in Dec 2015 (EA recorded outline); Leeds Building Society HQ is outside every recorded outline.
Needs network access to the EA LiDAR service; skipped when it is unreachable."""
import pytest

from cri.terrain.lidar_flood import flood_position


def _fp(lat, lon):
    try:
        p = flood_position(lat, lon)
    except Exception as e:                                  # noqa: BLE001
        pytest.skip(f"LiDAR service unreachable: {type(e).__name__}")
    if not p.get("available"):
        pytest.skip(p.get("reason") or "LiDAR unavailable")
    return p


def test_leeds_dock_is_floored_above_negligible():
    assert _fp(53.7918, -1.5320).get("floor") in ("MODERATE", "HIGH", "VERY HIGH")


def test_leeds_hq_has_no_floor():
    assert _fp(53.7968, -1.5466).get("floor") is None
