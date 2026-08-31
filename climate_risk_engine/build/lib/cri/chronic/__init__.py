"""
CRI Chronic Physical Risk Module
=================================

Models long-term, progressive climate shifts that erode asset viability
over 5–30 year horizons (CSRD / ESRS E1 time horizons: short ≤2030,
medium ≤2040, long ≤2050).

Hazard parameters modelled
--------------------------
    Temperature     Mean annual temperature rise above 1995–2014 baseline
                    under SSP2-4.5 and SSP5-8.5 (IPCC AR6 regional tables)
    Sea-level rise  Local relative SLR from NASA/IPCC AR6 medium confidence
                    projections; includes vertical land motion (VLM)
    Water stress    Aqueduct 3.0 baseline water depletion ratio (WRI)
    Drought         SPEI-based chronic drought frequency change
    Heat stress     WBGT threshold exceedance days (Wet Bulb Globe Temperature)

Business impact linkages
------------------------
    Temperature rise → OPEX (cooling energy), labour productivity loss
    Sea-level rise   → asset impairment, flood frequency amplification, uninsurability
    Water stress     → input cost, operational constraint, regulatory risk
    Drought          → supply chain disruption, feedstock availability
    Heat stress      → labour hour reduction, safety incident rate increase

IPCC scenario mapping
---------------------
    SSP1-2.6  Low emissions  / 1.5–2°C world
    SSP2-4.5  Intermediate   / ~2.7°C by 2100
    SSP3-7.0  High emissions / ~3.6°C by 2100     ← ESRS E1 material
    SSP5-8.5  Very high      / ~4.4°C by 2100     ← ESRS E1 severe

Entry point
-----------
    from cri.chronic import ChronicRiskEngine

    engine = ChronicRiskEngine()
    result = engine.assess(lat=17.6868, lon=83.2185,
                           asset_name='Vizag Steel Plant',
                           asset_rev_m=4200,
                           book_value_m=1800)
    print(result.summary)
"""

from .engine import ChronicRiskEngine, ChronicRiskResult

__all__ = ["ChronicRiskEngine", "ChronicRiskResult"]
