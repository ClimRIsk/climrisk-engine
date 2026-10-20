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


# ── OSM buildings ────────────────────────────────────────────────────────────────────────────────────────────────────────────
def _osm(ways):
    return {"elements": [{"type": "way", "tags": t, "geometry": [{"lat": la, "lon": lo} for la, lo in g]} for t, g in ways]}


SQ = lambda la, lo, d=0.0004: [(la, lo), (la, lo + d), (la + d, lo + d), (la + d, lo), (la, lo)]   # noqa: E731


def test_height_rules_are_flagged_by_source():
    assert H._height({"height": "12.5 m"}) == (12.5, "height tag")
    h, src = H._height({"building:levels": "4"})
    assert abs(h - 13.3) < 1e-6 and src == "storeys x 3.2 m"
    assert H._height({}) == (7.0, "assumed (no height in OpenStreetMap)")
    assert H._height({"height": "9999"})[1].startswith("assumed")           # implausible tag is not trusted


def test_buildings_without_hazard_layer_are_unclassified_and_honest():
    H.RASTERS.pop(("last", "landslide"), None)
    osm = _osm([({"building": "yes", "height": "10"}, SQ(10.0, 106.0)), ({"building": "yes"}, SQ(10.001, 106.001)), ({"building": "yes"}, [(10, 106), (10, 106.1)])])
    r = H.buildings(10.0, 106.0, 1.0, hazard="landslide", fetch=lambda q: osm)
    assert r["count"] == 2 and r["hazard"] is None and r["class_counts"] is None       # degenerate way dropped, no invented class
    assert all("cls" not in b for b in r["buildings"])
    assert r["height_sources"] == {"height tag": 1, "assumed (no height in OpenStreetMap)": 1}
    assert "not an assessment of the building" in r["caveat"]


def test_buildings_take_worst_class_under_footprint_and_study_area_flag():
    cls = np.full((10, 10), 3, dtype=int); cls[:, :5] = 1            # west half class 1 (worse), east half class 3
    box = {"west": 106.0, "east": 106.01, "south": 10.0, "north": 10.01}
    H.RASTERS[("last", "landslide")] = (cls, box, ["Very high", "High", "Moderate", "Low"])
    straddle = SQ(10.004, 106.0045, 0.001)                              # crosses the class boundary at lon 106.005
    east = SQ(10.002, 106.007, 0.0005)
    outside = SQ(10.5, 106.5)                                           # outside raster: not evaluated
    osm = _osm([({"building": "yes"}, straddle), ({"building": "yes"}, east), ({"building": "yes"}, outside)])
    r = H.buildings(10.005, 106.005, 1.0, hazard="landslide", polygons=[[(106.0, 10.0), (106.006, 10.0), (106.006, 10.01), (106.0, 10.01)]], fetch=lambda q: osm)
    by = {b["label"] for b in r["buildings"]}
    assert r["buildings"][0]["label"] == "High" and r["buildings"][1]["label"] == "Low" and r["buildings"][2]["cls"] is None
    assert r["class_counts"] == {"High": 1, "Low": 1, "not evaluated": 1} and None in by
    assert [b["in_study_area"] for b in r["buildings"]] == [True, False, False]
    assert r["legend"][0] == {"label": "Very high", "color": H.PALETTE[0]}


def test_buildings_cap_and_radius_clamp():
    ways = [({"building": "yes"}, SQ(10 + i * 1e-5, 106.0, 5e-6)) for i in range(H.MAX_BUILDINGS + 50)]
    seen = {}
    r = H.buildings(10.0, 106.0, 99, fetch=lambda q: (seen.setdefault("q", q), _osm(ways))[1])
    assert r["count"] == H.MAX_BUILDINGS and r["capped"] is True
    south = float(seen["q"].split("(")[1].split(",")[0])
    assert abs(south - (10.0 - 1.5 / 111.32)) < 1e-6                    # radius clamped to 1.5 km, not 99


def test_buildings_endpoint_maps_upstream_failure_to_502(monkeypatch):
    from fastapi.testclient import TestClient
    from cri.api.main import app
    monkeypatch.setattr(H, "_overpass_buildings", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("OpenStreetMap buildings unavailable (URLError)")))
    c = TestClient(app)
    r = c.post("/hazard3d/buildings", json={"hazard": "none", "lat": 10, "lon": 106, "radius_km": 1})
    assert r.status_code == 502 and "unavailable" in r.json()["detail"]
    monkeypatch.setattr(H, "_overpass_buildings", lambda *a, **k: _osm([({"building": "yes", "height": "8"}, SQ(10.0, 106.0))]))
    r = c.post("/hazard3d/buildings", json={"hazard": "none", "lat": 10, "lon": 106, "radius_km": 1})
    assert r.status_code == 200 and r.json()["count"] == 1 and r.json()["hazard"] is None


# ---- 3D viewer: LiDAR terrain tiles, imagery, LiDAR building heights ---------------------------------------------------------------------------
def _fake_grid(epsg=27700, e0=400000.0, n0=500000.0, res=2.0, n=200, surface_bump=None):
    from types import SimpleNamespace
    z = np.full((n, n), 50.0)
    if surface_bump:
        r0, r1, c0, c1, h = surface_bump
        z[r0:r1, c0:c1] += h
    return SimpleNamespace(elevations=z, cell_size_m=res, asset_row=n // 2, asset_col=n // 2, epsg=epsg, centre_en=(e0, n0), source="test")


def test_terrarium_round_trip_is_sub_centimetre():
    from cri.connectors.terrain_mesh import decode_terrarium, encode_terrarium
    h = np.array([[-12.3, 0.0, 153.7, 2840.125, 8848.0]])
    assert np.abs(decode_terrarium(encode_terrarium(h)) - h).max() < 0.01


def test_tile_bounds_cover_the_world_and_nest():
    from cri.connectors.terrain_mesh import tile_bounds
    w, s, e, n = tile_bounds(0, 0, 0)
    assert abs(w + 180) < 1e-9 and abs(e - 180) < 1e-9 and 85 < n < 85.1 and -85.1 < s < -85
    w1, s1, e1, n1 = tile_bounds(1, 0, 0)
    assert abs(e1 - 0.0) < 1e-9 and abs(n1 - n) < 1e-9


def test_lidar_tile_only_where_lidar_covers_and_reports_mask(monkeypatch):
    from cri.connectors import terrain_mesh as TM
    assert TM.lidar_tile(15, 0, 0) is None                    # open ocean: no LiDAR, caller keeps the global tile
    from cri.connectors import lidar_dem
    g = _fake_grid()
    monkeypatch.setattr(lidar_dem, "sample_lonlat", lambda grid, lo, la: np.where(np.asarray(lo) > np.mean(lo), 70.0, np.nan))
    import math
    n_ = 2 ** 15
    tx, ty = int((-2.93 + 180) / 360 * n_), int((1 - math.asinh(math.tan(math.radians(54.89))) / math.pi) / 2 * n_)      # the tile holding Carlisle
    got = TM.lidar_tile(15, tx, ty, fetch_grid=lambda *a, **k: g)
    assert got is not None
    h, mask = got
    assert h.shape == (256, 256) and mask.any() and not mask.all()


def test_lidar_building_heights_measure_roofs_not_ground(monkeypatch):
    from pyproj import Transformer
    from cri import hazardsurface as H
    e0, n0 = 400000.0, 500000.0
    dtm = _fake_grid(e0=e0, n0=n0)
    dsm = _fake_grid(e0=e0, n0=n0, surface_bump=(96, 104, 96, 104, 9.0))      # a 16 m square, 9 m tall, near the centre
    lon0, lat0 = Transformer.from_crs(27700, 4326, always_xy=True).transform(e0, n0)
    to_ll = Transformer.from_crs(27700, 4326, always_xy=True)
    ring = [list(to_ll.transform(e0 + dx, n0 + dy)) for dx, dy in ((-6, -6), (6, -6), (6, 6), (-6, 6))]
    far = [list(to_ll.transform(e0 + dx + 100, n0 + dy + 100)) for dx, dy in ((-6, -6), (6, -6), (6, 6), (-6, 6))]
    hs = H.lidar_heights(lat0, lon0, 0.3, [ring, far], grid_fn=lambda la, lo, r, surface=False: dsm if surface else dtm)
    assert hs[0] == 9.0
    assert hs[1] is None                                       # flat ground: no building height invented


def test_lidar_building_heights_none_when_no_surface_model():
    from cri import hazardsurface as H
    assert H.lidar_heights(0.0, 0.0, 0.3, [[[0, 0], [1, 0], [1, 1]]], grid_fn=lambda *a, **k: None) is None


def test_sentinel2_scene_must_contain_the_box_and_prefers_least_cloud():
    from cri.connectors import sentinel2_rgb as S2
    def post(url, body, timeout=40):
        return {"features": [
            {"id": "edge", "bbox": [-3.5, 54.0, -2.95, 55.5], "properties": {"datetime": "2026-04-30T11:00:00Z", "eo:cloud_cover": 0.0}, "assets": {"visual": {"href": "x"}}},
            {"id": "good", "bbox": [-3.5, 54.0, -2.5, 55.5], "properties": {"datetime": "2026-05-30T11:00:00Z", "eo:cloud_cover": 3.0}, "assets": {"visual": {"href": "y"}}}]}
    sc = S2.find_scene(54.89, -2.93, 1.5, post=post)
    assert sc["id"] == "good" and sc["cloud_pct"] == 3.0       # the cloud-free scene only half covers the box, so it is skipped
    assert S2.find_scene(54.89, -2.93, 1.5, post=lambda *a, **k: {"features": []}) is None


def test_coverage_endpoint_reports_lidar_and_aerial():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from cri.api import hazard3d_api
    from fastapi import Depends
    prot, pub = hazard3d_api.make_router(Depends(lambda: None))
    app = FastAPI(); app.include_router(pub)
    c = TestClient(app)
    nl = c.get("/hazard3d/coverage", params={"lat": 52.37, "lon": 4.9}).json()
    assert nl["lidar_building_heights"] is True and nl["aerial"]["provider"] == "pdok"
    uk = c.get("/hazard3d/coverage", params={"lat": 54.89, "lon": -2.93}).json()
    assert uk["lidar"] and uk["aerial"] is None and uk["lidar_building_heights"] is True
    sea = c.get("/hazard3d/coverage", params={"lat": 0.0, "lon": -30.0}).json()
    assert sea["lidar"] is None and sea["aerial"] is None
    assert c.get("/hazard3d/imagery/aerial/3/0/0.jpg").status_code == 404        # no open aerial imagery at that tile
