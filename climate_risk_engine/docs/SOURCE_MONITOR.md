# Source monitor: is the engine's data the newest edition? (6 Nov 2026)

`connectors/source_monitor.py` watches the publisher of every external source behind the emission factors. For each source it knows what the engine HOLDS (read from the engine's own data file), reads the publisher's release page, and says whether something newer exists.

| Source | Page read | Engine holds | Newest on 6 Nov 2026 |
|---|---|---|---|
| UK DESNZ conversion factors | gov.uk collection page | 2026 | 2026 (31 Jul 2026) |
| US EPA eGRID | epa.gov/egrid/detailed-data | eGRID2023 rev2 | eGRID2023 rev2 (page updated 22 Sep 2026; eGRID2024 not listed) |
| AIB European Residual Mix | aib-net.org residual-mix page | 2025 (26 May 2026) | 2025 (published 28 May 2026) |
| Ember yearly electricity data | ember-energy.org data page | built 30 Sep 2026 | page updated 3 Nov 2026 (rolling; within the 90-day window) |
| CEA CO2 Baseline Database (India) | cea.nic.in listing | Version 22.0 | Version 22.0 |

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
