# Golden sites (19 Oct 2026)

28 places where public knowledge gives an unambiguous, coarse expectation ("hurricane-exposed", "not cyclone-exposed", "hot desert",
"river valley that flooded in 2002"). Expectations live in `tests/golden/sites.py`; they were written from well-known events and
climatology, **not** from the engine's output. The physical-risk report was run at each site with the network available and the zone
ratings frozen in `tests/golden/snapshot_2026-10-19.json`.

## Result: 27 of 46 expectations met (59 %), 19 missed

| Zone | Met | Missed | Why it misses |
|---|---|---|---|
| Tropical cyclone | 13 / 13 (Singapore bound relaxed to LOW: Vamei 2001) | 0 | – |
| Water stress (Aqueduct) | 5 / 5 | 0 | – |
| Heat | 1 / 6 | Phoenix, Riyadh, Delhi, Kuwait City, Death Valley | the "heat" zone is a land-cover/latitude indicator, not air temperature or WBGT; Death Valley reads NEGLIGIBLE |
| Coastal flood | 3 / 10 | Miami, New Orleans, Jakarta, Mumbai, Venice, Funafuti, Malé | the Deltares 90 m undefended surge map is dry at the pin (its elevation model cannot see built-up ground, atolls, subsidence) |
| River flood | 3 / 8 | Dhaka, Bangkok, Dresden, Cologne, Ahrweiler | the JRC/GloFAS map is dry or shallow at the 250 m pin (misses small rivers and some large floodplains) |
| Wildfire | 2 / 3 | Blue Mountains (MODERATE, expected HIGH) | cover/NDVI indicator, not fire weather |
| Landslide/slope, rotterdam/phoenix controls | all met | – | – |

Each miss is listed in `KNOWN_GAPS` with its reason. The tests enforce three things: no *new* miss goes unrecorded, a gap that gets fixed
must be deleted from the list, and the agreement rate may not fall below today's 59 %. The cyclone and water-stress zones are also
recomputed live (offline, local data) against the expectations on every test run.

## What this means in plain terms
- **Wind/cyclone and water stress behave as a screening tool should** at every golden site.
- **Coastal flood is the weak point.** Seven of ten known coastal-flood sites read NEGLIGIBLE. The numbers are what the data says; the claim
  "outside the modelled storm-surge zone" is true and misleading at the same time. Do not present coastal results for built-up or low-lying
  sites as evidence of safety. (The report already adds a caution under 10 m near the sea.)
- **River flood** is only as good as a 30-arcsec global map: right at Houston, Brisbane and Chennai, silent at the Elbe, Rhine and Ahr.
- **Heat** in the physical-risk report does not measure heat. The WBGT/UTCI models in the model lab do; wiring them into the report is the fix.

## Insurer view at the same sites (`tests/golden/insurer_snapshot_2026-10-19.json`, USD 10 m each, 3,000 simulated years)
Wind AAL, USD: New Orleans 39 k · Miami 25 k · Houston 18 k · Manila 15 k · Dhaka 14 k · Tokyo 6.8 k · Rotterdam 0 · Phoenix 0.
Ranking is plausible. Rotterdam and Phoenix show 0 **with a "wind not assessed" flag** (the wind module is tropical-cyclone only; European windstorm is not modelled), so 0 there means *not modelled*, not *no risk*. Coastal AAL is 0 at Miami and New Orleans for the same dry-map reason as above, and the test records this as a known gap, not an endorsement. No benchmark against the FEMA National Risk Index or a vendor model has been done yet; that is the next step for the insurer view.

## Limits of this exercise
28 sites and coarse thresholds are a smoke test for gross errors, not a validation. Expectations are mine from public knowledge and have not been reviewed by an independent expert. Refresh the snapshot with `python tests/golden/refresh.py` (network) and review the diff.
