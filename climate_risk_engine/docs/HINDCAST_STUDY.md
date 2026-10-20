# Named-event hindcast: would ClimRisk have pointed at the buildings that flooded?

Run 20 October 2026. Reproduce: `PYTHONPATH=.:src python scripts/hindcast_carlisle.py --town carlisle|cockermouth|york --stage buildings|terrain|score`. Results: `data/backtest/<town>_hindcast.json`. Metric code is unit-tested (`tests/test_stress_regressions.py::test_hindcast_metrics_*`).

## The question

A customer with a list of buildings asks: before the event, which ones does your model say are exposed, and was it right? This tests exactly that for named English floods, building by building.

## Method

- **Truth:** Environment Agency Recorded Flood Outlines for the named event. A building counts as flooded if its footprint touches the outline.
- **Buildings:** every OpenStreetMap footprint in a box of about 5-6 km around the town: Carlisle 12,759, Cockermouth 3,109, York 32,270.
- **Scores from terrain only,** computed without looking at the outline: height above nearest drainage (HAND) from the EA 1 m LiDAR bare earth (the engine's flood-floor input) at each footprint centroid; the same on the global 10-30 m tiles (the engine before LiDAR); and global height above the lowest ground within 1 km.
- **Nothing fitted.** The fixed bands tested are the ones already in the engine (HAND < 1 m HIGH, < 5 m MODERATE, < 10 m LOW), set before this test.
- **Uncertainty:** spatial block bootstrap (500 m blocks, 1,000 resamples); neighbouring buildings flood together, so a plain bootstrap would be too confident.

## Results

| Town and event | Flooded / scored | LiDAR HAND ROC area | Global-tile HAND | Global height-above-low | Top 10 % by LiDAR finds | Top 20 % finds |
|---|---|---|---|---|---|---|
| Carlisle, Storm Desmond Dec 2015 | 571 / 11,614 (4.9 %) | **0.90** (0.83-0.94) | 0.58 (0.44-0.71) | 0.69 | **59 %** of flooded | 81 % |
| Carlisle, Jan 2005 | 627 / 11,614 (5.4 %) | **0.89** (0.82-0.93) | 0.61 (0.48-0.73) | 0.71 | 53 % | 75 % |
| Cockermouth, Nov 2009 | 250 / 3,013 (8.3 %) | **0.92** (0.85-0.97) | 0.87 (0.78-0.93) | **0.95** | 52 % | 86 % |
| York, Dec 2015 | 628 / 23,017 (2.7 %) | **0.81** (0.69-0.89) | 0.50 (0.39-0.61) | 0.51 | **60 %** | 67 % |
| York, Storm Dennis Feb 2020 | 48 / 23,017 (0.2 %) | 0.63 (0.54-0.76) | 0.51 | 0.66 | 25 % | 27 % |

Random flagging would find 10 % and 20 %. Brackets are 95 % block-bootstrap intervals.

The engine's fixed bands, on LiDAR:

| Town and event | HAND < 1 m: buildings flagged, hit rate, false alarms | HAND < 5 m: flagged, hit rate, false alarms | Precision at < 5 m |
|---|---|---|---|
| Carlisle 2015 | 1,345 (12 %), 64 %, 9 % | 4,401 (38 %), 95 %, 35 % | 12 % |
| Cockermouth 2009 | 40 (1 %), **4 %**, 1 % | 543 (18 %), 81 %, 12 % | 37 % |
| York 2015 | 2,546 (11 %), 62 %, 10 % | 16,465 (72 %), 98 %, 71 % | 4 % |

## What we can say, and what we cannot

1. **LiDAR ranking works at building level.** In the four major events across the three towns, flooded buildings sit in the engine's most exposed tenth five to six times as often as chance. Four of the five events clear 0.8 (the fifth, a 48-building local flood, does not).
2. **LiDAR is worth having over the global tiles in flat, built-up places.** At Carlisle and York the global-tile score is no better than chance (0.50-0.61); the difference is large and its interval excludes zero. At Cockermouth, a steep valley town, the global tiles already do well (0.87) and a simple relative-height score on them (0.95) matched or beat LiDAR HAND; the LiDAR gain there is inside the noise.
3. **The fixed height bands are susceptibility, not event prediction.** "HAND < 1 m" caught 62-64 % of flooded buildings at Carlisle and York but only 4 % at Cockermouth, where the 2009 flood stood well above the channel. "HAND < 5 m" catches most flooded buildings (81-98 %) but flags 18-72 % of all buildings, so it is not a statement that a building will flood; in a flat city like York almost everything is low. The report must keep describing these bands as how far the river can reach, never as a flood prediction.
4. **It does not rank small, local floods well.** Storm Dennis at York (48 buildings) scored 0.63, with an interval that reaches 0.54.
5. **Not tested: depth, loss, or insured loss.** The outlines give extent, not depth, and no per-building claims are public, so this study says nothing about the dollar numbers. That remains the largest open validation gap.

## Limits that bias the result

- The terrain is the 2022 LiDAR, after both the 2015 events and after the Carlisle flood-defence scheme; protected buildings that flooded in 2015 can read safe. This costs hits, so the true pre-event skill could be a little higher or lower; it is not corrected.
- Recorded outlines are the mapped extent, not every flooded building; both sample and truth mix river and surface-water causes.
- The two Carlisle events overlap in place and are not independent evidence. Three towns, all in England, all river floods.
- Building footprints are from OpenStreetMap volunteers, and 3 % (Cockermouth), 9 % (Carlisle) and 28 % (York) of footprints fell outside the LiDAR window and were dropped.

## What changed in the product because of this

- The river-flood zone now also states the pin's **position among the surrounding 2.5 km** ("lower than X % of the land around it"), because ranking is the better-supported signal. It is a fact; it changes no rating.
- The bands keep their wording as susceptibility. No threshold was tuned to this data.
