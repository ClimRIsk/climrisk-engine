"""WRI Aqueduct 4.0 water risk — the published dataset, shipped with the engine.

Data: WRI Aqueduct 4.0 (Y2023M07D05), CC BY 4.0. Kuzma, S. et al. 2023, "Aqueduct 4.0: Updated
decision-relevant global water risk indicators", WRI Technical Note, doi:10.46830/writn.23.00061.
Built by scripts/build_aqueduct_data.py into src/cri/data/aqueduct40.npz:

  * baseline (1979–2019) for 68,506 units (HydroBASINS level-6 basin × GADM admin-1), gridded at 0.05°:
    13 indicator scores 0–5 with WRI's category labels — baseline water stress (bws), water depletion (bwd),
    interannual variability (iav), seasonal variability (sev), groundwater table decline (gtd), riverine flood
    (rfr), coastal flood (cfr), drought (drr), untreated connected wastewater (ucw), coastal eutrophication
    potential (cep), unimproved/no drinking water (udw), unimproved/no sanitation (usa), peak RepRisk (rri);
    overall water risk with WRI's default weights and the nine industry weightings
  * future (per basin): water stress, depletion, interannual and seasonal variability for 2030 / 2050 / 2080
    under optimistic (SSP1 RCP2.6), business-as-usual (SSP3 RCP7.0) and pessimistic (SSP5 RCP8.5)

No network calls. Points that fall just offshore snap to the nearest land cell within ~15 km.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np

from .base import BaseConnector

_DATA = Path(__file__).resolve().parents[1] / "data" / "aqueduct40.npz"
SOURCE = "WRI Aqueduct 4.0 (Y2023M07D05, CC BY 4.0)"

INDICATORS = {
    "bws": "Baseline water stress", "bwd": "Baseline water depletion", "iav": "Interannual variability",
    "sev": "Seasonal variability", "gtd": "Groundwater table decline", "rfr": "Riverine flood risk",
    "cfr": "Coastal flood risk", "drr": "Drought risk", "ucw": "Untreated connected wastewater",
    "cep": "Coastal eutrophication potential", "udw": "Unimproved/no drinking water",
    "usa": "Unimproved/no sanitation", "rri": "Peak RepRisk country ESG risk",
}
RAW_UNITS = {"bws": "withdrawal / available flow", "bwd": "consumption / available flow",
             "gtd": "cm/year", "rfr": "% population exposed (annual expected)",
             "cfr": "% population exposed (annual expected)", "drr": "drought risk index 0–1"}
INDUSTRIES = {"def": "Default weighting", "agr": "Agriculture", "che": "Chemicals", "con": "Construction materials",
              "elp": "Electric power", "fnb": "Food & beverage", "min": "Mining", "ong": "Oil & gas",
              "smc": "Semiconductors", "tex": "Textiles"}
FUTURE_IND = {"ws": "Water stress", "wd": "Water depletion", "iv": "Interannual variability", "sv": "Seasonal variability"}
FUTURE_SCEN = {"opt": "Optimistic (SSP1 RCP2.6)", "bau": "Business as usual (SSP3 RCP7.0)",
               "pes": "Pessimistic (SSP5 RCP8.5)"}
FUTURE_YEARS = [2030, 2050, 2080]
BASELINE_YEAR = 2019  # end of the 1979–2019 baseline period

# engine sector slug → Aqueduct industry weighting
SECTOR_TO_INDUSTRY = {
    "agriculture": "agr", "food": "fnb", "beverages": "fnb", "food_beverage": "fnb", "chemicals": "che",
    "cement": "con", "construction": "con", "steel": "con", "utilities": "elp", "power": "elp",
    "electric_power": "elp", "coal": "min", "mining": "min", "metals_mining": "min", "aluminium": "min",
    "oil_gas": "ong", "oil": "ong", "gas": "ong", "refining": "ong", "semiconductors": "smc",
    "technology": "smc", "textiles": "tex", "apparel": "tex",
}


@lru_cache(maxsize=1)
def _load() -> dict:
    d = np.load(_DATA)
    return {k: d[k] for k in d.files}


def _score(v) -> Optional[float]:
    v = int(v)
    return None if v == 255 else round(v / 50.0, 2)


def _cell(lat: float, lon: float) -> tuple[int, float]:
    """Unit index (1-based, 0 = none) at a point; snaps to the nearest land cell within 3 cells."""
    D = _load()
    g, res = D["grid"], float(D["res"])
    H, W = g.shape
    r = int(np.clip((90.0 - lat) / res, 0, H - 1))
    c = int(np.clip((lon + 180.0) / res, 0, W - 1))
    if g[r, c]:
        return int(g[r, c]), 0.0
    best, bd = 0, 1e9
    for dr in range(-3, 4):
        for dc in range(-3, 4):
            rr, cc = r + dr, (c + dc) % W
            if 0 <= rr < H and g[rr, cc]:
                d = dr * dr + dc * dc
                if d < bd:
                    best, bd = int(g[rr, cc]), d
    return best, (np.sqrt(bd) * res * 111.0 if best else float("nan"))


def point(lat: float, lon: float, sector: Optional[str] = None) -> dict:
    """Full Aqueduct 4.0 record at a coordinate."""
    D = _load()
    idx, snap_km = _cell(lat, lon)
    if not idx:
        return {"found": False, "lat": lat, "lon": lon, "source": SOURCE,
                "note": "No Aqueduct unit within ~15 km (open ocean or outside coverage)."}
    i = idx - 1
    labels = D["labels"]
    lab = lambda k: (str(labels[k]) if k >= 0 else None)
    names = list(INDICATORS)
    raw_names = list(RAW_UNITS)
    baseline = {}
    for j, k in enumerate(names):
        baseline[k] = {"name": INDICATORS[k], "score": _score(D["score"][i, j]), "label": lab(int(D["label"][i, j]))}
    for j, k in enumerate(raw_names):
        v = float(D["raw"][i, j])
        baseline[k]["raw"] = None if np.isnan(v) else round(v, 4)
        baseline[k]["raw_unit"] = RAW_UNITS[k]
    overall = {k: _score(D["total"][i, j]) for j, k in enumerate(INDUSTRIES)}
    ind = SECTOR_TO_INDUSTRY.get((sector or "").lower().replace(" ", "_").replace("&", ""))
    pfaf = int(D["pfaf"][i])
    future = None
    fk = np.searchsorted(D["fut_pfaf"], pfaf)
    if fk < len(D["fut_pfaf"]) and D["fut_pfaf"][fk] == pfaf:
        F = D["future"][fk]
        future = {sc: {k: {str(y): _score(F[a, b, c]) for b, y in enumerate(FUTURE_YEARS)}
                       for c, k in enumerate(FUTURE_IND)} for a, sc in enumerate(FUTURE_SCEN)}
    pl = int(D["place"][i])
    return {
        "found": True, "lat": lat, "lon": lon, "snapped_km": round(snap_km, 1),
        "basin_pfaf_id": pfaf, "iso3": str(D["iso3"][i]), "country": str(D["place_country"][pl]),
        "admin1": str(D["place_admin1"][pl]),
        "baseline_period": "1979–2019", "baseline": baseline,
        "overall_water_risk": {"score": overall["def"], "label": lab(int(D["total_label"][i])),
                               "by_industry": {INDUSTRIES[k]: v for k, v in overall.items() if k != "def"},
                               "sector_weighting": INDUSTRIES.get(ind) if ind else None,
                               "sector_score": overall.get(ind) if ind else None},
        "future": future, "future_scenarios": FUTURE_SCEN, "future_indicators": FUTURE_IND,
        "source": SOURCE,
        "caveat": "Aqueduct indicators are modelled and, per WRI, not directly validated; they describe the "
                  "basin/admin-1 unit, not the site's own water supply.",
    }


def stress_path(rec: dict, scenario: str = "bau") -> dict:
    """Water stress score path: baseline (2019) and the future scores for one Aqueduct scenario."""
    out = {str(BASELINE_YEAR): rec["baseline"]["bws"]["score"]}
    if rec.get("future"):
        out.update(rec["future"][scenario]["ws"])
    return out


def _interp_year(rec: dict, key: str, year: int, scenario: str = "bau") -> Optional[float]:
    base = rec["baseline"][{"ws": "bws", "wd": "bwd", "iv": "iav", "sv": "sev"}[key]]["score"]
    if not rec.get("future") or year <= BASELINE_YEAR:
        return base
    pts = [(BASELINE_YEAR, base)] + [(int(y), v) for y, v in rec["future"][scenario][key].items() if v is not None]
    pts = [(y, v) for y, v in pts if v is not None]
    if not pts:
        return None
    xs, ys = zip(*pts)
    return round(float(np.interp(year, xs, ys)), 2)


class WRIAqueductConnector(BaseConnector):
    """WRI Aqueduct 4.0 lookups (offline, from the published dataset). Scores 0–5, 5 = highest risk."""

    def __init__(self, cache_dir=None):
        # no disk cache needed — the dataset ships with the engine
        pass

    def fetch(self, **kwargs) -> dict:
        return point(kwargs["lat"], kwargs["lon"], kwargs.get("sector"))

    def point(self, lat: float, lon: float, sector: Optional[str] = None) -> dict:
        return point(lat, lon, sector)

    def get_water_risk(self, lat: float, lon: float, year: int = 2030, scenario: str = "bau") -> dict:
        """Headline scores at a coordinate for a year.

        water_stress: baseline water stress, interpolated to `year` along the Aqueduct future projection
        (2030/2050/2080, default business-as-usual). flood_risk: max of riverine and coastal flood risk scores
        (baseline — Aqueduct 4.0 publishes no future flood scores). drought_risk: baseline drought risk.
        """
        rec = point(lat, lon)
        if not rec["found"]:
            return {"water_stress": None, "flood_risk": None, "drought_risk": None, "source": "wri_aqueduct_4.0:none",
                    "note": rec["note"]}
        b = rec["baseline"]
        floods = [v for v in (b["rfr"]["score"], b["cfr"]["score"]) if v is not None]
        return {
            "water_stress": _interp_year(rec, "ws", year, scenario),
            "flood_risk": max(floods) if floods else None,
            "flood_riverine": b["rfr"]["score"], "flood_coastal": b["cfr"]["score"],
            "drought_risk": b["drr"]["score"], "groundwater": b["gtd"]["score"],
            "overall_water_risk": rec["overall_water_risk"]["score"],
            "year": year, "scenario": FUTURE_SCEN.get(scenario),
            "source": "wri_aqueduct_4.0",
        }

    def get_region_risk(self, region_code: str, year: int = 2030) -> dict:
        """Country-level median of Aqueduct units for an ISO 3166-2 code's country (admin-1 names are not
        matched to ISO 3166-2 codes, so the result is explicitly country-level)."""
        iso3 = _iso2_to_iso3((region_code or "")[:2])
        D = _load()
        if not iso3:
            return {"water_stress": None, "flood_risk": None, "drought_risk": None,
                    "source": "wri_aqueduct_4.0:unresolved", "note": f"Unknown region code {region_code!r}"}
        m = D["iso3"] == iso3
        if not m.any():
            return {"water_stress": None, "flood_risk": None, "drought_risk": None,
                    "source": "wri_aqueduct_4.0:none", "note": f"No Aqueduct units for {iso3}"}
        names = list(INDICATORS)

        def med(k):
            s = D["score"][m, names.index(k)]
            s = s[s != 255]
            return round(float(np.median(s)) / 50.0, 2) if s.size else None
        fl = [v for v in (med("rfr"), med("cfr")) if v is not None]
        return {"water_stress": med("bws"), "flood_risk": max(fl) if fl else None, "drought_risk": med("drr"),
                "source": f"wri_aqueduct_4.0:country_median:{iso3}", "year_note": "baseline 1979–2019"}


@lru_cache(maxsize=1)
def _iso_table() -> dict:
    import json
    return json.loads((_DATA.parent / "iso2_iso3.json").read_text())


def _iso2_to_iso3(a2: str) -> Optional[str]:
    return _iso_table().get(a2.upper()) if a2 else None
