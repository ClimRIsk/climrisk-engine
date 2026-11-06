# Source monitor: is the engine's data the newest edition? (6 Nov 2026)

`connectors/source_monitor.py` watches the publisher of every external source behind the emission factors. For each source it knows what the engine HOLDS (read from the engine's own data file), reads the publisher's release page, and says whether something newer exists.

| Source | Page read | Engine holds | Newest on 6 Nov 2026 |
|---|---|---|---|
| UK DESNZ conversion factors | gov.uk collection page | 2026 | 2026 (31 Jul 2026) |
| US EPA eGRID | epa.gov/egrid/detailed-data | eGRID2023 rev2 | eGRID2023 rev2 (page updated 22 Sep 2026; eGRID2024 not listed) |
| AIB European Residual Mix | aib-net.org residual-mix page | 2025 (26 May 2026) | 2025 (published 28 May 2026) |
| Ember yearly electricity data | ember-energy.org data page | built 30 Sep 2026 | page updated 3 Nov 2026 (rolling; within the 90-day window) |
| CEA CO2 Baseline Database (India) | cea.nic.in listing | Version 22.0 | Version 22.0 |
| Australia NGA Factors | DCCEEW publication index (one page per edition) | 2026 | 2026 |
| Canada National Inventory Report, Annex 7 | open.canada.ca record (Date Published, Temporal Coverage) | data to 2024, published 14 Apr 2026 | same |
| Germany UBA electricity-mix factors | UBA topic page (publication "1990 - <year>") | 1990-2025 (March 2026) | same |
| China MEE/NBS electricity CO2 factors | none readable: **manual**, always `unverified` | data year 2023 (31 Dec 2025) | not checked automatically |
| Japan SHK electricity supplier list | MOE calc page (r<yy>_denki_coefficient[_rev<n>].pdf) | R8 rev4 | R8 rev4 |
| GHG Protocol corporate standards | update-process page (Standard Development Plan link) | plan of 2026-07-29 | same |
| ISO 14067 | iso.org catalogue page (edition, stage, last confirmed) | 2018, stage 90.92, confirmed 2024 | same |
| IFRS S2 GHG amendments | ifrs.org project page (issue month, effective date) | Dec 2025, effective 1 Jan 2027 | same |
| ESRS simplified delegated act | **manual**: entry into force is an Official Journal event | adopted 3 Jul 2026, pending scrutiny | not checked automatically |
| CBAM certificate prices | Commission price page (quarters published) | 3 of 4 for 2026 | same |
| CBAM legislation and default values | Commission legislation page (default-values date, acts listed) | 10 Aug 2026, 14 acts | same |
| PCAF Part A | carbonaccountingfinancials.com/standard (edition links) | 2nd edition (2022) | **3rd edition (2025)** |

The last seven are the standards watch (docs/STANDARDS_WATCH.md). For them `newer_available` means the publisher's page now reads differently from the recorded status (`compare="changed"`), not that a dataset can be rebuilt.

The last five feed `national_grids.json` (docs/NATIONAL_GRIDS.md). The Australian page was read in Chrome; my fetch tool returned nothing for it, so the first live run on a machine confirms the detector over plain HTTP. China's MEE publishes each year's factors as a separate announcement and the announcement index could not be read as a list, so the monitor says so (`manual`) instead of guessing.

## Rules
- A source is `current` only after a successful read of its release page in the last 7 days. A read that fails, or a page whose layout is not recognised ("layout changed"), makes it `unverified` and shows the error. It never stays `current` on a stale or failed read.
- Newer edition detected: `newer_available`, with the publisher, the new edition, the date, and what to run to rebuild. DESNZ, eGRID, AIB and Ember are **detected, not auto-loaded**: they are large workbooks rebuilt by `scripts/build_carbon_factors.py` (and `build_lca_data.py` for Ember) and checked by a person. CEA is loaded automatically only if every check passes (docs/CEA_BASELINE.md).
- Rolling releases (Ember) count as newer only when the page is more than 90 days past the date of the engine's copy.
- Each source is read at most once a day; one failing source never stops the others. Offline mode (`CRI_STORY_OFFLINE`) makes no requests.

## Where it shows
- `GET /data/sources` (`?source=egrid`), `POST /data/sources/refresh`.
- `GET /provenance?file=carbon_factors.json` adds `monitors` and appends any `newer_edition`.
- The resident agent answers "is the eGRID data up to date", "which residual mix do you use", "has DESNZ published new factors" (the last reads the page immediately). Its autopilot posts an inbox alert when a newer edition appears.
- A background thread re-checks every 6 hours while the app is live.

## Adding a source (countries, standards)
Create a `Source` with `held()` (what the engine holds), `detect(html)` (newest edition on the publisher page; raise `ValueError("layout changed ...")` if not recognised) and call `register()`. Write a test with a page fixture shaped like the real page.

## Limits
- Detectors were written from the real pages as read on 6 Nov 2026. They are confirmed on each machine by the first live run; if a page changes shape the status shows the error.
- Ember changed its CSV layout in July 2026 (one row per area, month and source; metrics as columns). The Ember rebuild scripts have not been re-checked against the new layout; do that before the next rebuild.
- The monitor sees what the publisher's release page says. It does not check the numbers inside a new release.
