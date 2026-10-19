# Golden sites (19 Oct 2026)

28 places where public knowledge gives an unambiguous, coarse expectation ("hurricane-exposed", "not cyclone-exposed", "hot desert",
"river valley that flooded in 2002"). Expectations live in `tests/golden/sites.py`; they were written from well-known events and
climatology, **not** from the engine's output. The physical-risk report was run at each site with the network available and the zone
ratings frozen in `tests/golden/snapshot_2026-10-19.json`.

## Result: 51 of 59 expectations met (86 %), 8 missed (was 28 of 46, 61 %, at the start of the day)

Later the same day: open LiDAR terrain (England 1 m, Netherlands 0.5 m) and a floodplain-position screen were added, three English flood towns joined the set (all met), and an earthquake zone was added (USGS hazard model in the US, recorded-seismicity screen elsewhere; 11 new expectations, all met). The English towns are not independent of the LiDAR benchmark that set the thresholds. The remaining eight misses are listed in `KNOWN_GAPS`.

| Zone | Met | Missed | Why it misses / what changed |
|---|---|---|---|
| Tropical cyclone | 14 / 14 (Singapore bound relaxed to LOW after a first run: Typhoon Vamei 2001) | 0 | – |
| Water stress (Aqueduct) | 3 / 3 | 0 | – |
| Heat | 6 / 6 | 0 | **Fixed.** The zone now counts days at or above 35 C and 40 C in the ERA5 daily-maximum record (1994-2023, `finance/heat_hazard.py`). Phoenix, Riyadh, Kuwait City and Death Valley CRITICAL, Delhi HIGH, Reykjavik NEGLIGIBLE. Dry-bulb only: humid-tropics heat stress (wet-bulb) is understated and the zone says so. If the record cannot be read the zone falls back to the old land-cover indicator, labelled as such, or to "Not assessed". |
| Coastal flood | 8 / 10 | Jakarta, Mumbai | **Partly fixed.** A dry surge map on very low ground near the sea is no longer a clean bill: an *elevation screen* holds the rating at MODERATE (ground ≤ 3 m, or ≤ 8 m with a recorded defence standard below 1-in-100) or LOW, only within 15 km (≤ 3 m) or 5 km (≤ 8 m) of the sea and only where the defence standard is below 1-in-1,000. It is labelled as an elevation screen, not a flood depth. Jakarta (ground model 10 m despite subsidence) and Mumbai (6 m, defence 1-in-100) stay below the screen's limits. |
| River flood | 4 / 9 | Dhaka, Bangkok, Dresden, Cologne, Ahrweiler | **Not fixed, caution added.** The JRC/GloFAS map (30 arcsec) is dry or shallow at the pin. No free higher-resolution global river hazard map exists; the terrain HAND layer's own backtest shows little point skill, so it is not used to override a rating. The report now adds a "Verify" caution when permanent water lies within 300 m of a site the river map shows as dry or low. |
| Wildfire | 2 / 3 | Blue Mountains (MODERATE, expected HIGH) | **Not fixed.** Cover/NDVI indicator, not fire weather. The model lab's fire-weather index was tried and looks implausible (Pasadena 230 days a year at FWI ≥ 30), so it was not wired in; it needs its own check first. |
| Landslide/slope | 1 / 1 (Kathmandu) | 0 | – |

Each miss is listed in `KNOWN_GAPS` with its reason. The tests enforce three things: no *new* miss goes unrecorded, a gap that gets fixed
must be deleted from the list, and the agreement rate may not fall below 80 % (today 83 %). The cyclone and water-stress zones are also
recomputed live (offline, local data) against the expectations on every test run.

Side effect to know: the elevation screen also raises Reykjavik's coastal flood to MODERATE (low coast, no recorded defence). That is a screening statement, not a validated one; no golden expectation covers it.

## What this means in plain terms
- **Wind/cyclone, heat and water stress behave as a screening tool should** at every golden site.
- **Coastal flood** is better but still rests on a 90 m map plus a coarse elevation screen. Do not present coastal results for built-up or low-lying
  sites as evidence of safety; the report states when a rating is an elevation floor.
- **River flood** is only as good as a 30-arcsec global map: right at Houston, Brisbane and Chennai, silent at the Elbe, Rhine and Ahr. The report
  cautions when water is close; a local flood study is still needed.
- **Heat** is dry-bulb temperature from a reanalysis cell, not WBGT, and not a projection.

## Insurer view at the same sites (`tests/golden/insurer_snapshot_2026-10-19.json`, USD 10 m each, 3,000 simulated years)
Wind AAL, USD: New Orleans 39 k · Miami 25 k · Houston 18 k · Manila 15 k · Dhaka 14 k · Tokyo 6.8 k · Rotterdam 0 · Phoenix 0.
Ranking is plausible. Rotterdam and Phoenix show 0 **with a "wind not assessed" flag** (the wind module is tropical-cyclone only; European windstorm is not modelled), so 0 there means *not modelled*, not *no risk*. Coastal AAL is still 0 at Miami and New Orleans: the elevation screen above changes the *report rating*, not the insurer simulation, which uses the surge map directly. The insurer dashboard now tags every site whose cover is incomplete as "partial" (hover for the perils not assessed). `tools/benchmark_fema_nri.py` compares engine wind/river/coast loss rates with FEMA's National Risk Index county building losses, but it needs the NRI county CSV, which this environment could not download; run it on a machine with access.

## Limits of this exercise
28 sites and coarse thresholds are a smoke test for gross errors, not a validation. Expectations are mine from public knowledge and have not been reviewed by an independent expert. Refresh the snapshot with `python tests/golden/refresh.py` (network) and review the diff.
