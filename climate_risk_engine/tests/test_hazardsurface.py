"""3D hazard viewer back end: hazard surfaces, study-area parsing, honest 'no map' answers."""
import base64
import io
import json
import zipfile

import numpy as np
import pytest

from cri import geo_io, hazardsurface as H
from cri.connectors.terrain_tiles import ElevationGrid

RAIN = lambda la, lo: {"maxima_mm": [80, 95, 110, 120, 140, 90, 100, 130], "source": "test rainfall"}   # noqa: E731


def _dem(sea=True, steep=True):
    rows = cols = 150
    X, Y = np.meshgrid(np.linspace(0, 1, cols), np.linspace(0, 1, rows))
    z = np.where(X < 0.2, -5.0 if sea else 30.0, (X - 0.2) * (400 if steep else 20) + (8 * np.sin(Y * 12) * 20 if steep else 0))
    return ElevationGrid(10.0, 106.0, 1.5, 20.0, rows, cols, 75, 75, np.maximum(z, -5) if sea else np.maximum(z, 0) + 10.0)


KML = (b'<?xml version="1.0"?><kml xmlns="http://www.opengis.net/kml/2.2"><Placemark><Polygon><outerBoundaryIs><LinearRing><coordinates>'
       b'106.0,10.0,0 106.01,10.0,0 106.01,10.01,0 106.0,10.01,0 106.0,10.0,0</coordinates></LinearRing></outerBoundaryIs></Polygon></Placemark></kml>')


def test_kml_kmz_geojson_agree():
    a = geo_io.parse_study_area(KML, "a.kml")
    buf = io.BytesIO(); zipfile.ZipFile(buf, "w").writestr("doc.kml", KML)
    b = geo_io.parse_study_area(buf.getvalue(), "a.kmz")
    gj = json.dumps({"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [[[106.0, 10.0], [106.01, 10.0], [106.01, 10.01], [106.0, 10.01], [106.0, 10.0]]]}}).encode()
    c = geo_io.parse_study_area(gj, "a.geojson")
    assert a["area_km2"] == b["area_km2"] == c["area_km2"]
    assert a["area_km2"] == pytest.approx(1.2122, rel=0.01)           # 0.01 deg x 0.01 deg at 10 N is about 1.21 km2
    assert a["polygons"][0][0] == [106.0, 10.0] and len(a["polygons"][0]) == 4


def test_bad_files_raise_not_guess():
    for data, name in ((b"<x>", "a.kml"), (b"{not json", "a.geojson"), (b'<kml xmlns="http://www.opengis.net/kml/2.2"><Document/></kml>', "a.kml")):
        with pytest.raises(ValueError):
            geo_io.parse_study_area(data, name)


def test_landslide_classes_follow_slope_and_sum_to_one():
    r = H.surface("landslide", 10.0, 106.0, 1.5, dem=_dem(), rain_fetch=RAIN)
    s = r["shares_box"]
    assert r["available"] and sum(s[k] for k in ("Very high", "High", "Moderate", "Low")) == pytest.approx(1.0, abs=1e-3)
    assert s["High"] + s["Very high"] > 0.1                              # the steep ramp must show up
    flat = H.surface("landslide", 10.0, 106.0, 1.5, dem=_dem(steep=False), rain_fetch=RAIN)["shares_box"]
    assert flat["Low"] > 0.99                                            # a gentle slope is stable
    assert s["not_evaluated_cells"] > 0                                  # sea cells are not evaluated, not counted low
    assert "not validated" in r["validated"]


def test_png_is_real_and_transparent_where_not_evaluated():
    from PIL import Image
    r = H.surface("slope", 10.0, 106.0, 1.5, dem=_dem())
    im = Image.open(io.BytesIO(base64.b64decode(r["image_png_b64"])))
    a = np.array(im)
    assert a.shape[2] == 4 and (a[..., 3] == 0).any() and (a[..., 3] > 0).any()


def test_study_area_shares_use_only_cells_inside():
    poly = [[106.0 - 0.004, 10.0 - 0.004], [106.0 + 0.004, 10.0 - 0.004], [106.0 + 0.004, 10.0 + 0.004], [106.0 - 0.004, 10.0 + 0.004]]
    r = H.surface("landslide", 10.0, 106.0, 1.5, polygons=[poly], dem=_dem(), rain_fetch=RAIN)
    assert r["shares_study_area"]["cells"] < r["shares_box"]["cells"]


def test_sea_level_bathtub_needs_sea_and_connectivity():
    r = H.surface("sea_level_rise", 10.0, 106.0, 1.5, dem=_dem(), water_level_m=3.0)
    assert r["shares_box"]["Depth 1 m or more"] > 0 and r["shares_box"]["Not inundated"] > 0.3
    inland = H.surface("sea_level_rise", 10.0, 106.0, 1.5, dem=_dem(sea=False), water_level_m=3.0)
    assert inland["available"] is False and "no sea" in inland["reason"]


def test_point_only_hazards_have_no_invented_map():
    for hz in ("heat", "drought", "earthquake", "wildfire", "cyclone_wind"):
        r = H.surface(hz, 10.0, 106.0)
        assert r["available"] is False and r["reason"] and "image_png_b64" not in r


def test_plume_refuses_to_assume_a_release():
    r = H.surface("plume", 10.0, 106.0, 1.0)
    assert r["available"] is False and "None is assumed" in r["reason"]
    ok = H.surface("plume", 10.0, 106.0, 1.0, Q_g_s=50, H_m=20, u10_ms=3, wind_from_deg=270)
    assert ok["available"] and ok["details"]["peak_ground"]


def test_unknown_hazard_is_an_error():
    with pytest.raises(ValueError):
        H.surface("nonsense", 0, 0)


def test_api_routes():
    from fastapi.testclient import TestClient
    from cri.api.main import app
    c = TestClient(app)
    assert c.get("/hazard3d").status_code == 200
    cat = c.get("/hazard3d/catalogue").json()["hazards"]
    assert any(h["spatial_surface"] for h in cat) and any(not h["spatial_surface"] and h["reason"] for h in cat)
    r = c.post("/hazard3d/study-area", files={"file": ("a.kml", KML)})
    assert r.status_code == 200 and r.json()["area_km2"] > 1
    assert c.post("/hazard3d/study-area", files={"file": ("b.kml", b"<x>")}).status_code == 422
    assert c.post("/hazard3d/surface", json={"hazard": "heat", "lat": 10, "lon": 106}).json()["available"] is False
    assert c.post("/hazard3d/surface", json={"hazard": "zzz", "lat": 10, "lon": 106}).status_code == 400
    assert c.post("/hazard3d/config", json={"google_maps_key": "bad key!"}).status_code == 400


def test_key_is_stored_owner_only_and_never_listed_remotely(tmp_path, monkeypatch):
    monkeypatch.setenv("CRI_HOME", str(tmp_path))
    monkeypatch.delenv("GOOGLE_MAPS_TILES_KEY", raising=False)
    from fastapi.testclient import TestClient
    from cri.api.main import app
    c = TestClient(app)
    assert c.post("/hazard3d/config", json={"google_maps_key": "AbC-123_x"}).json()["google_key_present"] is True
    f = tmp_path / "maps_key.json"
    assert f.exists() and (f.stat().st_mode & 0o077) == 0
    assert c.get("/hazard3d/config").json()["google_key_present"] is True
    assert c.post("/hazard3d/config", json={"google_maps_key": ""}).json()["google_key_present"] is False and not f.exists()
