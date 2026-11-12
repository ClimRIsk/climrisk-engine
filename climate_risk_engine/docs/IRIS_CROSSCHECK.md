# IRIS cross-check: status and what must be checked first (12 Nov 2026)

## What IRIS is (from the Figshare record and the Imperial pages)

The Imperial College Storm Model (Sparks and Toumi, Scientific Data 2024, doi 10.1038/s41597-024-03250-y) is a stochastic tropical cyclone wind model. It simulates only the decay after each storm's lifetime-maximum intensity, draws on observed tracks from 1980 to 2021, assigns peak intensity from potential intensity, and produces a 10,000-year global synthetic event set across six basins, with landfall wind return periods validated against observations by the authors. The Figshare record is "IRIS: The Imperial College Storm Model" (Springer Nature collection 6724251). The code is at github.com/njsparks/iris.

## What is built

`uncertainty.tc_sources` compares cyclone wind return levels from two or more independent sources at one site and reports the range as a quantified structural doubt, or "not assessed" with the reason when only one source exists. It takes the numbers as input; it reads nothing from IRIS. `POST /uncertainty/ledger` accepts `tc_winds`.

## What is NOT done, and why

The IRIS files are **not ingested**. From this environment I could not reach the paper's data-availability section or the Figshare file list: the PMC page is behind a reCAPTCHA (not bypassed), and the Nature, GitHub and Figshare API requests timed out. I therefore do not know the file names, structure, size, units convention or coordinate convention, and I will not write a loader against guessed formats.

## Before loading IRIS, check

1. File list, size and format on the Figshare record (the GitHub page says a 100-year sample is shipped in numpy and text formats; the full set may differ).
2. Wind definition: the paper reports 1-minute sustained wind in m/s at landfall. The engine's wind is 1-minute sustained at 10 m in open terrain. Confirm they are the same before comparing.
3. What "return period at a site" means: a landfall maximum wind within a radius of the site, or a wind field at the site. These are different quantities; the radius must be stated.
4. Licence terms for redistribution and commercial use.
5. Period: IRIS represents the 1980 to 2021 climate. Do not compare it with future-climate winds.

## Then

Build `connectors/iris.py` with a site return-level function, compare with the engine's IBTrACS-based `rp100_ms` at a handful of coastal sites (Chennai, Miami, Manila, Okinawa, Brisbane), report the ratios and the sites where they disagree most, and pass both into the ledger.
