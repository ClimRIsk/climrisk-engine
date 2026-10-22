# Hail, tornado and marine heatwave: what was built and how it was tested

Built 22 October 2026. Code: `src/cri/convective.py`, `src/cri/marine.py`; tests: `tests/test_convective.py`, `tests/test_marine.py`; benchmarks: `scripts/convective_benchmark.py`, `scripts/marine_benchmark.py`. All three appear in the physical report as **context** zones with their evidence grade. They never change the overall rating, the issue list or the measures, and a site outside a model's coverage gets no zone (never "low").

| Hazard | What it does | Data | Tested against | Result | Grade |
|---|---|---|---|---|---|
| Severe hail | Days a year with a report of hail of 1 inch or more (and 2 inches or more) within 40 km | NOAA Storm Prediction Center public storm reports, 1996-2024 (public domain). United States only | Temporal hold-out: 1996-2009 reports against 2010-2024 reports at 600 random US points | Rank correlation 0.93, top-quartile AUC 0.94, median error 0.28 days a year | B |
| Tornado | Days a year with a tornado touching down within 40 km; EF1+ and EF2+ separately; band follows EF1+ | Same source, final F/EF-rated list | Same hold-out | Rank correlation 0.77 (EF1+) to 0.83 (all), AUC 0.90, median error 0.07-0.14 days a year | B |
| Marine heatwave | Hobday et al. (2016) events on daily sea-surface temperature: days a year (1982-2011 baseline against the last ten years), strongest and latest event, category | NOAA OISST v2.1, 0.25 degree, 1982 to within days of today | Agreement with the published reference code; detection of 9 documented events | Same events as the reference on 892 of 895; 7 of 9 documented events found | B |

## Hail and tornado

The measure is days with at least one report within 25 miles (40 km), the standard in the severe-weather climatology literature (Brooks et al. 2003; Gensini and Ashley 2011). One storm that produces fifty reports counts as one day.

Test design: 600 points drawn at random over the contiguous United States (seed 42, fixed before any result was read). The model is the 1996-2009 count; the target is the 2010-2024 count at the same point.

| Measure | Hail 1 in+ | Hail 2 in+ | Tornado (all) | Tornado EF1+ |
|---|---|---|---|---|
| Mean days a year, 1996-2009 | 1.89 | 0.24 | 0.50 | 0.21 |
| Mean days a year, 2010-2024 | 2.04 | 0.28 | 0.49 | 0.25 |
| Rank correlation, early against late | 0.93 | 0.82 | 0.83 | 0.77 |
| AUC for the late top quartile | 0.94 | 0.92 | 0.90 | 0.90 |
| Median absolute error (days a year) | 0.28 | 0.07 | 0.14 | 0.07 |
| Share within a factor of two, where both periods have reports | 88 % | 69 % | 72 % | 66 % |
| Points with no report early but reports late | 4.5 % of all points (10 % had none early) | 9 % (38 % had none early) | 9 % (23 %) | 8 % (38 %) |

Full output: `docs/convective_holdout.json`.

What this shows: past report counts predict later report counts well for common events, and less well for rare ones, where a few reports decide the number. What it does not show: that reports measure the hazard. Reports come from people and spotters, so thinly populated land is under-counted, reported sizes are estimates, and reporting practice changed over the period (the tornado EF0 count in particular tracks how many people are watching, which is why the band follows EF1 and stronger). It also says nothing about damage, nothing about the chance a given building is hit, and nothing about future climates. The bands (hail 0.3 / 1 / 2.5 days a year, tornado EF1+ 0.1 / 0.3 / 0.6) are screening judgement from the spread over random US points, not calibrated to loss.

Coverage: the United States, and only where no other country lies within about 33 km. The source records US reports only; a 40 km circle that takes in Canada or Mexico would read as fewer storms than there are, so the model reports "not covered" there rather than a low number. Outside the US it also reports "not covered". Europe has an equivalent database (ESSL, ESWD) but its data need registration.

## Marine heatwave

Method: for each calendar day the 1982-2011 mean and 90th percentile of sea-surface temperature (11-day window around the day, then smoothed over 31 days). A marine heatwave is at least 5 days above the 90th percentile; events separated by 2 days or fewer are joined. Category 1-4 follows Hobday et al. (2018).

**Agreement with the reference code.** Run on the same 8 cached ocean cells (43 years of daily data each), the published reference implementation (Oliver, `marineHeatWaves.py`, with `climatologyPeriod=[1982, 2011]`) and this module find the same events: 892 of 895 have identical start and end dates, and the same number of events at every cell. The 3 differences are one event each at three cells. All events with a peak anomaly of 2 C or more match except 2 of 477. Output: `docs/marine_crosscheck.json`.

**Documented events.** Written down before any series was read: an event counts as found if a marine heatwave overlapping the documented window has a peak anomaly of at least 2.0 C and lasts 30 days or more.

| Event | Source | Found | Longest/strongest overlapping event |
|---|---|---|---|
| Ningaloo Nino 2011, Western Australia | Pearce and Feng 2013 | yes | 55 days, +5.2 C |
| Northeast Pacific "Blob" 2014-16 | Bond et al. 2015 | yes | 57 days, +3.5 C |
| Northeast Pacific 2019 | Amaya et al. 2020 | yes | 126 days, +3.8 C |
| Tasman Sea 2015/16 | Oliver et al. 2017 | yes | 187 days, +5.1 C |
| Great Barrier Reef 2016 | Hughes et al. 2017 | **no** | 21 days, +2.1 C |
| Mediterranean 2003 | Olita et al. 2007 | yes | 33 days, +4.3 C |
| Mediterranean 2022 | Marba et al. 2022 | yes | 174 days, +4.8 C |
| Northwest Atlantic 2012 | Mills et al. 2013 | yes | 131 days, +3.5 C |
| Peru coastal El Nino 2017 | Garreaud 2018 | **no** | 18 days, +3.8 C |

7 of 9 found. Two reads of the misses: at Peru the surface warming was sharp but short (18 days) and the 30-day rule excludes it; at the Great Barrier Reef the 25 km cell shows a 21-day event of +2.1 C, while the 2016 bleaching was driven by weeks of accumulated heat stress the rule does not capture. The rule was not changed after seeing this.

**This test is weak evidence.** The same criteria are met in an average of 24 % of the *other* years at those cells (range 7-39 %), so a loose criterion finds something in most windows by chance. Seven of nine is clearly above that, but the nine windows were also chosen because the events were famous. What the test supports is "detects the big events the literature agrees on", not "gets the intensity or timing right". Intensity was not tested against an independent source.

What the model is not: it is surface temperature on a 25 km cell, not water at depth, in a harbour or in an estuary. It does not say what heat does to a species, a farm, an outfall or an asset, and it has no forecast. A site on land uses the nearest ocean cell within 0.5 degree and reports the distance.

Cost: the first call for a new site reads about 16,000 daily files from NOAA's public copy on AWS (about 5-10 minutes, needs the `blosc` package; days newer than the AWS copy, which stops in 2025, are read one at a time from NCEI). The series is cached on disk and later calls fetch only new days. For that reason the report zone is **off unless `CRI_MARINE=1`**. Hail and tornado are on by default (`CRI_CONVECTIVE=0` turns them off).

## What was not tested

Hail size against damage; any loss or fragility model for these perils; tornado path width or length; marine heatwave intensity against an independent record; any country other than the United States for hail and tornado; any trend or projection.
