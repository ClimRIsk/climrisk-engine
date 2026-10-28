"""
Aviation fuel and SAF blend arithmetic.

  fossil jet life-cycle baseline = 89 gCO2e/MJ (the value ICAO uses for fossil fuel in the CORSIA co-processing equation, ICAO document 'CORSIA Default Life Cycle Emissions Values
                                   for CORSIA Eligible Fuels', June 2025, eq. 2; quoted there as the LC baseline)
  blend LCA   = 89 x fossil energy share + SAF LCA x SAF energy share      (energy basis; ICAO's own co-processing equation is on a volume basis, so for co-processed fuel use that equation)
  reduction   = 1 - blend LCA / 89
  tank-to-wake CO2 for jet fuel burned: US EPA Hub Table 1/2 (72.22 kg CO2/mmBtu, 9.75 kg/gal for Kerosene-Type Jet Fuel), combustion only.

The SAF LCA value is YOUR certified value (supplier proof of sustainability or a CORSIA/RED actual-value certificate). ICAO's per-feedstock default tables were not transcribed: they are indexed by feedstock,
process and region and carry applicability conditions that a hand copy could silently lose. If you quote a default, give the table row and edition yourself.
Scope: this is fuel-side arithmetic. Non-CO2 climate effects of flying (contrails, NOx) are not included.
"""
from __future__ import annotations

from typing import Optional

from ..carbon import epa_factors as EF

FOSSIL_LCA_G_PER_MJ = 89.0
SOURCE = "ICAO, CORSIA Default Life Cycle Emissions Values for CORSIA Eligible Fuels, June 2025 (eq. 2: 89 gCO2e/MJ for the fossil fraction)"


class AviationError(ValueError):
    pass


def jet_combustion(gallons: float) -> dict:
    """Tank-to-wake CO2 for conventional jet fuel (EPA Hub). Biogenic and SAF accounting is separate: see blend()."""
    r = EF.mobile_co2("Kerosene-Type Jet Fuel", gallons)
    return {"gallons": gallons, "kg_co2": r["kg_co2"], "source": r["source"], "table": r["table"], "note": "CO2 only; CH4/N2O for aircraft are small and not loaded; all carbon is counted as emitted whatever the blend."}


def blend(energy_mj: float, saf_energy_share: float, saf_lca_g_per_mj: float) -> dict:
    """energy_mj of finished fuel, of which saf_energy_share is SAF with a certified life-cycle value saf_lca_g_per_mj (gCO2e/MJ, may be negative for credited pathways)."""
    if energy_mj < 0 or not 0 <= saf_energy_share <= 1:
        raise AviationError("energy must be >= 0 and the SAF share between 0 and 1")
    lca = FOSSIL_LCA_G_PER_MJ * (1 - saf_energy_share) + saf_lca_g_per_mj * saf_energy_share
    return {"blend_lca_g_per_mj": lca, "baseline_g_per_mj": FOSSIL_LCA_G_PER_MJ, "reduction_pct": 100.0 * (1 - lca / FOSSIL_LCA_G_PER_MJ), "life_cycle_t_co2e": energy_mj * lca / 1e6,
            "avoided_vs_fossil_t_co2e": energy_mj * (FOSSIL_LCA_G_PER_MJ - lca) / 1e6, "saf_lca_g_per_mj": saf_lca_g_per_mj, "saf_lca_source": "supplied by you (certified value)",
            "basis_note": "Energy-share basis. Co-processed fuel needs ICAO's volume-based equation. ICAO default tables not loaded.", "source": SOURCE}
