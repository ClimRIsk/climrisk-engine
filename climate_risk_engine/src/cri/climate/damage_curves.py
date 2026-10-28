"""
Flood depth-damage by sector, from the published JRC tables (Huizinga et al. 2017, EUR 28552 EN, JRC105688, doi:10.2760/16510).

Replaces an earlier version that used log-logistic parameters (d50, k) and per-m2 maximum damages labelled as read from the JRC figures. Those numbers were not
the JRC tables and cannot be traced to them, so they are gone. This module now reads the tabulated damage fractions held in data/jrc_flood_curves.json (via
finance.vulnerability) and interpolates between the published depths.

What is and is not here
  * Damage is a FRACTION of replacement value. No per-m2 maximum damage is given: those values (JRC Table 1) have not been transcribed, so no function here converts
    to dollars from floor area.
  * JRC has six classes: residential, commercial, industrial, transport, infrastructure, agriculture. There is no chemical or energy class. Chemical, oil and gas sites
    use the industrial table; the earlier "steeper hazmat curve" was a judgement with no source and is removed. Say so if a chemical site needs more.
  * The sector to class mapping below is a classification judgement, and `basis` on every curve says which table was used.
  * Where JRC publishes no table for a continent, the mean of the published continents is used and `basis` says so. These functions take no location, so unless a
    continent is given they use that mean for every class.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..finance import vulnerability as _V

_SECTOR_TO_CLASS: dict[str, str] = {
    "chemicals": "industrial", "oil_gas": "industrial", "energy": "industrial", "utilities": "infrastructure",
    "industrials": "industrial", "metals_mining": "industrial", "mining": "industrial", "cement": "industrial",
    "food_beverage": "industrial", "automotive": "industrial", "beverages": "industrial",
    "construction": "infrastructure", "transport": "transport", "shipping": "transport", "aviation": "transport", "telecom": "infrastructure",
    "agriculture": "agriculture",
    "technology": "commercial", "financials": "commercial", "healthcare": "commercial", "consumer": "commercial", "real_estate": "commercial", "media": "commercial",
}


@dataclass(frozen=True)
class DamageCurve:
    """JRC tabulated depth-damage function for one asset class (and continent, or the mean of published continents)."""
    asset_class: str
    continent: Optional[str]
    basis: str

    def damage_fraction(self, depth_m: float) -> float:
        """Fraction of replacement value lost at a flood depth in metres (0 when dry). Linear between the published depths; held at the last table value beyond 12 m."""
        if depth_m is None or not depth_m > 0.0:
            return 0.0
        cont = self.continent if self.continent in ("africa", "asia", "europe", "northamerica", "oceania", "southamerica") else "oceania"
        curve, _ = _V.jrc_curve(cont, self.asset_class) if self.continent else (_mean_curve(self.asset_class), None)
        return float(np.interp(min(float(depth_m), 12.0), _V.DEPTH, curve))

    def damage_from_rcv(self, depth_m: float, rcv_usd: float) -> float:
        return self.damage_fraction(depth_m) * rcv_usd


def _mean_curve(asset_class: str):
    tab = {k: v for k, v in _V._JRC[asset_class].items() if not (asset_class == "industrial" and k == "oceania")}
    return np.mean(np.array(list(tab.values())), axis=0)


def asset_class_for(sector_key: str) -> str:
    return _SECTOR_TO_CLASS.get(sector_key, "industrial")


def get_damage_curve(sector_key: str, continent: Optional[str] = None) -> DamageCurve:
    cls = asset_class_for(sector_key)
    if continent:
        _, basis = _V.jrc_curve(continent, cls)
    else:
        n = len([1 for k in _V._JRC[cls] if not (cls == "industrial" and k == "oceania")])
        basis = f"JRC {cls}: mean of the {n} published continents (no continent given)"
    if sector_key not in _SECTOR_TO_CLASS:
        basis += "; sector not recognised, industrial used"
    return DamageCurve(cls, continent, basis)


def flood_damage_fraction(depth_m: float, sector_key: str, continent: Optional[str] = None) -> float:
    return get_damage_curve(sector_key, continent).damage_fraction(depth_m)


def flood_damage_usd(depth_m: float, sector_key: str, rcv_usd: float, continent: Optional[str] = None) -> float:
    return get_damage_curve(sector_key, continent).damage_from_rcv(depth_m, rcv_usd)
