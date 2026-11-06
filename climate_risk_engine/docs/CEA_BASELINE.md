# CEA CO2 Baseline Database: what the engine holds and how it stays current (6 Nov 2026)

## What CEA publishes
The Central Electricity Authority (Ministry of Power, India) publishes the **CO2 Baseline Database for the Indian Power Sector** once a year, as an XLSX database and a PDF user guide, listed with a version number at
https://cea.nic.in/cdm-co2-baseline-view?mappingId=234&lang=en. It exists for CDM baselines. It gives **one factor for the single Indian grid** per fiscal year: the weighted average (including renewables), and the operating, build and combined margins.
All regional grids were integrated in December 2013. **There are no state-wise or utility-wise factors.** A figure such as 0.82 for Tamil Nadu or 0.84 for Andhra Pradesh is not a CEA value.
The factors are CO2 only, on net generation, for April-to-March years. No CH4 or N2O, no upstream fuel chain, no transmission and distribution losses.

## What the engine holds (seed = Version 22.0, August 2026)
`src/cri/data/cea_baseline.json`, built by `scripts/build_cea_baseline.py` from the published PDF (Annexure I, Table S, Figures 4-6, revision history). The build fails if the weighted average is not CO2 / generation (within 0.0006) or if CM is not the mean of OM and BM (within 0.0011).
Weighted average, tCO2/MWh: 2013-14 0.774, 2014-15 0.779, 2015-16 0.774, 2016-17 0.770, 2017-18 0.754, 2018-19 0.744, 2019-20 0.713, 2020-21 0.703, 2021-22 0.715, 2022-23 0.716, 2023-24 0.727, 2024-25 0.710, **2025-26 0.675**.
FY2025-26 margins: OM 0.963, BM 0.446, CM 0.705. From FY2022-23 totals include captive injection and cross-border adjustment (the gap between conventional + RE and the total is stored per year); from FY2023-24 renewables enter the build margin. The series therefore has breaks.
Not held: fuel-level emission factors and station data (they are in the XLSX).

## The monitor (`connectors/cea_baseline.py`)
- `sync()` reads the listing at most once a day (the page is server-rendered; links are `/api/cdm-report-version/view-pdf/en/<id>`), finds the newest version and compares it with the version held.
- While the app is live a background thread calls `sync()` every 6 hours (`start_background`, started in the API lifespan); the resident agent's autopilot cycle also calls it and re-indexes the data into the agent's knowledge (`learn()`), so questions are answered from the current version.
- If a newer version is listed, its user guide PDF is downloaded and Annexure I and Table S are parsed. It is **accepted only if every check passes**: every year's factor = CO2 / generation, at least ten years, newest year later than the held one, a cover-page version number equal to the listing entry, Table S average equal to the newest Annexure I year, CM = mean of OM and BM. Then it overlays the seed (`~/.climrisk/cea_overlay.json`, with the PDF's sha256), earlier years are kept, and the agent's inbox says so.
- If any check fails, or no PDF reader is available, the seed stays in use and `status()` says `newer_available` with the reason. The agent will say "CEA has published Version X; my numbers are from Version Y".
- `status()` says `current` only after a successful read of the listing in the last 7 days; otherwise `unverified`. A failed read is shown as `last_error`, never as "current".
- `GET /data/cea` (status + factor for `fy`, `kind`), `POST /data/cea/refresh` (read now), `GET /provenance?file=cea_baseline.json` (adds `monitor` and `newer_edition`).
- Offline mode (`CRI_STORY_OFFLINE`) never touches the network.

## Where it is used
- Scope 2 location-based for **India** (`carbon/factors.electricity_location`, inventory records accept `grid_fy`, default the latest published year, with a note). Replaces the Ember-derived value (609.5 g/kWh, a 2025 calendar-year lifecycle-derived direct share) for India with CEA's 675 g/kWh (FY2025-26). Basis stated as CO2 only.
- Product carbon footprints (ISO 14067, `lca/`) keep Ember **lifecycle** grid factors, because a footprint needs upstream and non-CO2 gases which CEA's factor excludes. A CEA value can be supplied as an override.
- The resident agent: questions about CEA, the Indian grid factor, OM/BM/CM, "is there a newer version" ("check now" reads the listing immediately).

## Limits
- The PDF parser was tested on PDFs of the same table structure generated for the test (pdftotext and pypdf paths), not on a live Version 23.0, which does not exist yet. The first live ingestion should be checked by eye; a layout change fails the checks and leaves the seed in place.
- Only this one CEA publication is monitored. CEA's other publications (regulations, technical standards, monthly generation and capacity reports) are not read.
- The grid factor is annual and national. Plant- or supplier-specific factors (captive and third-party power) must come from the supplier or the plant's own data.
