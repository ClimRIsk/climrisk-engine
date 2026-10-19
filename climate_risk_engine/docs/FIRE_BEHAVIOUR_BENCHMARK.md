# Wildfire fuel + spread (Rothermel) — benchmark and decision

**Model:** `finance/fire_behaviour.py`. Rothermel (1972) surface-fire spread with Anderson (1982) fuel models: fuel from ESA WorldCover (tree→FM10, shrub→FM5, grass/crop→FM1), slope from the terrain grid, weather from ERA5 daily maxima, fine dead-fuel moisture from the Canadian FFMC (which remembers rain). Outputs head-fire rate of spread, flame length and Byram intensity on the 95th-percentile warm-season day.

**Physics check:** the spread calculation is tested against published BehavePlus-class values (6 % moisture, 5 mph midflame wind, flat): FM1 93 ch/h vs ~78 published, FM2 33 vs ~35, FM6 34 vs ~32, FM8 2 vs ~1.6 — within ±40 % (the test tolerance); the response to wind, slope and moisture has the right sign; spread stops above the moisture of extinction.

**Test:** `scripts/fire_behaviour_benchmark.py` on the Canadian National Fire Database (fires ≥ 20 ha, May–Sept, 8 boxes of 1°, unseen years 2011–2022, rank-only ROC area, nothing fitted). Question: does the model flag the days large fires are reported?

| Day score | Median ROC area (8 boxes) |
|---|---|
| Hot-day temperature alone | **0.80** |
| Fire Weather Index | 0.71 |
| Rothermel spread, same-day moisture, timber / brush / grass fuel | 0.53 / 0.62 / 0.61 |
| Rothermel spread, FFMC moisture, timber / brush / grass fuel | 0.67 / 0.67 / 0.76 |

(The FWI figure differs slightly from the earlier validation file because three boxes drew weather from a different source on this run.)

## What this says

- A spread model driven by one day's weather has almost no skill at telling fire days from other days (0.5–0.6). Giving it the fuel-moisture memory of the FFMC lifts it to 0.67–0.76, roughly the level of the FWI and still **below plain hot-day temperature**. Grass is the best of three fixed fuel choices, picked after seeing the results, so treat 0.76 as optimistic.
- Large fires in boreal Canada are driven by drought, ignition and blow-up days; surface spread rate on the average windy day does not capture that. The same holds for the destructive fires elsewhere (Fort McMurray 2016, Blue Mountains 2019): the model gives about 1 m flames there on a 95th-percentile day because the fires that matter are crown fires with spotting, which this model does not contain.

## Decision

Shown in the wildfire zone as a **physical fact** ("surface-fire behaviour on a severe warm-season day"), clearly labelled surface fire only. **It does not set or change the rating.** The rating remains the fuel/terrain screen, with the Blue Mountains still a listed known gap. Switch off with `CRI_FIREBEHAVIOUR=0`. Untested outside Canada. To go further needs crown-fire (Van Wagner), spotting, live-fuel moisture from satellite and burned-area truth in other regimes.
