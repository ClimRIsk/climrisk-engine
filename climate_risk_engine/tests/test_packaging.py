"""The Docker image does `pip install ./climate_risk_engine`, which copies only the files named in [tool.setuptools.package-data]. A data file the code reads at runtime but the
patterns miss is silently absent in production (the atlas district boundaries, the licence public key and the real out-of-scope queries were all missing this way)."""
import fnmatch
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _patterns(package: str) -> list:
    txt = (ROOT / "pyproject.toml").read_text()
    m = re.search(r'"' + re.escape(package) + r'"\s*=\s*\[(.*?)\]', txt, re.S)
    assert m, f"no package-data entry for {package}"
    return re.findall(r'"([^"]+)"', m.group(1))


def test_every_data_file_is_packaged():
    pats = _patterns("cri.data")
    missing = [p.name for p in (ROOT / "src" / "cri" / "data").iterdir()
               if p.is_file() and p.suffix not in (".py", ".pyc") and not any(fnmatch.fnmatch(p.name, q) for q in pats)]
    assert not missing, f"not matched by package-data (absent from the installed image): {missing}"


def test_atlas_files_are_packaged():
    pats = _patterns("cri.data")
    for f in ("india_flood_atlas.json", "india_landslide_exposure.json", "india_districts_adm2.json.gz"):
        assert any(fnmatch.fnmatch(f, q) for q in pats), f
