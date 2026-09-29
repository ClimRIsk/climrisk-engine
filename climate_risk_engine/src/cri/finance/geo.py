"""
finance/geo.py — where is an asset, for policy and vulnerability purposes.

  admin_at(lat, lon)   first-level admin unit (state / province) from the FLOPROS v1
                       layer: name, country, ISO3 and flood-protection standards
  continent(iso3)      region used by the JRC depth-damage curves
  tc_region(iso3)      Eberenz et al. (2021) tropical-cyclone calibration region
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np

_DATA = Path(__file__).resolve().parent.parent / "data"


@lru_cache(maxsize=1)
def _admin():
    z = np.load(_DATA / "admin1_grid.npz")
    meta = json.loads((_DATA / "admin1_flopros.json").read_text())
    return z["grid"], float(z["res"]), meta["units"], meta["source"]


def admin_at(lat: float, lon: float) -> dict:
    grid, res, units, src = _admin()
    i = int(np.clip((90.0 - lat) / res, 0, grid.shape[0] - 1))
    j = int(np.clip((lon + 180.0) / res, 0, grid.shape[1] - 1))
    u = int(grid[i, j])
    if not u:
        return {"found": False, "source": src}
    d = dict(units[u])
    d.update(found=True, unit_id=u, source=src)
    return d


@lru_cache(maxsize=1)
def _ngfs_regions() -> dict:
    return json.loads((_DATA / "ngfs_phase5.json").read_text())["country_region"]


def gcam_region(iso3: Optional[str]) -> Optional[str]:
    m = _ngfs_regions().get(iso3 or "")
    return m["GCAM 6.0 NGFS"].split("|", 1)[1] if m else None


_AFRICA = {"Africa_Eastern", "Africa_Northern", "Africa_Southern", "Africa_Western", "South Africa"}
_LATAM = {"Argentina", "Brazil", "Colombia", "South America_Northern", "South America_Southern",
          "Central America and Caribbean", "Mexico"}
_NAM = {"USA", "Canada"}
_EUR = {"EU-12", "EU-15", "Europe_Eastern", "Europe_Non_EU", "European Free Trade Association", "Russia"}
_OCE = {"Australia_NZ"}


def continent(iso3: Optional[str]) -> str:
    """JRC (Huizinga et al. 2017) regions: africa, asia, europe, northamerica, southamerica, oceania."""
    g = gcam_region(iso3)
    if g in _AFRICA:
        return "africa"
    if g in _LATAM:
        return "southamerica"          # Huizinga groups Central & South America together
    if g in _NAM:
        return "northamerica"
    if g in _EUR:
        return "europe"
    if g in _OCE or iso3 in {"FJI", "PNG", "SLB", "VUT", "WSM", "NCL"}:
        return "oceania"
    return "asia"


# CLIMADA ImpfSetTropCyclone country groups (Eberenz et al. 2021, NHESS 21:393)
_TC_REGION = {
    "NA1": "ABW AIA ARG ATG BHS BLZ BMU BOL BRB CHL COL CPV CRI CUB CYM DMA DOM ECU FLK GLP GRD GTM GUF GUY "
           "HND HTI JAM KNA LCA MEX MSR MTQ NIC PAN PER PRI PRY SHN SLV SUR SXM TCA TTO URY VCT VEN VGB VIR",
    "NA2": "CAN USA",
    "NI": "AFG ARE ARM AZE BGD BHR BTN DJI ERI ETH GEO IND IRN IRQ ISR JOR KAZ KGZ KWT LBN LKA MDV MMR MNG "
          "NPL OMN PAK QAT SAU SOM SYR TJK TKM UGA UZB YEM",
    "OC": "ASM AUS COK FJI FSM GUM KIR MHL MNP NCL NFK NIU NRU NZL PCN PLW PNG PYF SLB TKL TLS TON TUV VUT WLF WSM",
    "SI": "COD COM MDG MLI MOZ MUS MWI SWZ TZA ZAF ZWE",
    "WP1": "KHM IDN LAO MYS THA VNM", "WP2": "PHL", "WP3": "CHN", "WP4": "HKG JPN KOR MAC TWN",
}
_TC_OF = {iso: r for r, s in _TC_REGION.items() for iso in s.split()}


def tc_region(iso3: Optional[str]) -> str:
    return _TC_OF.get(iso3 or "", "ROW")
