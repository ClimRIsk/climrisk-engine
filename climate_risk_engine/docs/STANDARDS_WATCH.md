# Standards watch: what the standards say today, and what the engine implements (6 Nov 2026)

`src/cri/data/standards_watch.json` records the status of six standards and rules the engine's methods follow, each read from the publisher's own page on 6 Nov 2026. It is built and checked by `scripts/build_standards_watch.py`, joined to the engine by `connectors/standards_watch.py`, watched by seven monitors in `connectors/source_monitor.py`, and served at `GET /data/standards` (`?item=cbam`). The resident agent answers questions about it ("is Scope 2 being revised", "latest CBAM certificate price", "which PCAF edition do you use").

| Standard | Status on 6 Nov 2026 (publisher's page) | What the engine implements | Engine vs standard |
|---|---|---|---|
| GHG Protocol corporate standards (Corporate 2004, Scope 3 2011, Scope 2 Guidance 2015, AMI) | Under revision. GHG Protocol and ISO are merging them with ISO 14064-1 into one co-branded standard. Scope 2 consultation ran 20 Oct 2025 to 31 Jan 2026 (nearly 1,100 responses, 56 countries; summary 29 Jul 2026). Integrated public consultation Q2 2027, publication Q4 2028 | Scope 2 Guidance 2015: location- and market-based Scope 2 | matches |
| ISO 14067 (product carbon footprint) | ISO 14067:2018, reviewed and confirmed 2024, stage 90.92 "to be revised". GHG Protocol and ISO named a Joint Working Group for a harmonised product standard on 9 Feb 2026 | `lca/cfp.py` follows the 2018 method structure | matches |
| IFRS S2 (ISSB) | GHG-disclosure amendments issued Dec 2025; effective for annual periods beginning on or after 1 Jan 2027, early application permitted | IFRS S2 as issued June 2023 (`outcomes/disclosure.py`) | partly (amendments not reflected; content not read) |
| ESRS (CSRD) | Simplified ESRS delegated act published 3 Jul 2026 and sent to Parliament and Council; applies to financial years beginning on or after 1 Jan 2027 once in force. Entry into force not checked | ESRS E1 checklist from Delegated Regulation 2023/2772 (`regulatory/esrs_e1_gap.py`) | partly |
| CBAM | Definitive regime since 1 Jan 2026. 2026 certificate prices Q1 75.36, Q2 75.28, Q3 82.32 EUR/t; Q4 due 4 Jan 2027; central platform purchases from Feb 2027, weekly prices from 2027. CBAM factor on free allocation: 97.5% (2026), 95, 90, 77.5, 51.5, 39, 26.5, 14% (2033), none from 2034 | `climate/cbam.py` and `finance/policy.py`, corrected today (below) | matches, with limits |
| PCAF | Part A 3rd edition (Dec 2025 edition), Part B Dec 2023, Part C 2nd edition (2025) | Part A 2nd edition (2022) (`regulatory/pcaf.py`) | **behind** |

## Two findings from reading the primary pages

1. **CBAM exposure was overstated about forty-fold for 2026.** `climate/cbam.py` assumed certificates are due on 100% of embedded emissions from 2026 and a EUR 60 price. Directive 2003/87/EC Art. 10a(1a) (EUR-Lex) phases out free allocation with the CBAM factor, and Regulation 2023/956 Art. 31 reduces the certificates to surrender by the free allocation, so only 2.5% of the benchmarked emissions need certificates in 2026, rising to 100% from 2034. The module now uses the same schedule as `finance/policy.py` (which already had it right), the mean of the three published 2026 prices, and flags later prices as a scenario path. The CBAM story report text was updated to match. A defect found while testing: `phase_in_pct` was rounded to whole percents (2.5% showed as 3%); now one decimal.
2. **PCAF Part A is a full edition behind.** The monitor reports `newer_available` on the first live read. What changed in the 3rd edition has not been read, so nothing is claimed about the size of the gap.

## Rules
- Status values come only from primary pages (listed per item). The ISO text is paywalled: edition, stage and dates only; no clause text is stored.
- Anything not read from a primary page is under `not_assessed` for that item, not filled in. Examples: the content of the S2 amendments, whether the ESRS act is in force, the stage of the ISO 14067 revision project, the downstream-goods CBAM proposal.
- Conflicting secondary reports were resolved from the publisher: GHG Protocol's own FAQ gives Q2 2027 and Q4 2028 (not 2027 for the final standard).
- Dates marked done must be on or before the as-of date; planned dates must be after it. The CBAM factor schedule must equal `EU_CBAM_FREE_SHARE`. A bad record fails the build, and the test suite fails if the JSON differs from what the script writes.

## Monitors
`ghgp` (Standard Development Plan link), `iso14067` (edition, stage code, last-confirmed year), `ifrs_s2` (issue month, effective date), `cbam_price` (count of published 2026 quarters), `cbam_law` (default-values date and number of implementing/delegated acts), `pcaf` (Part A edition) read the publisher's page daily. `esrs` is `manual`: the EFRAG news item never changes, and the Official Journal could not be read as a list, so it stays "unverified" and says to check by hand.

For these, `newer_available` means "the page now reads differently from the recorded status" (CBAM prices and PCAF: strictly more recent). The monitor never rewrites the JSON: a person re-reads the page, edits `scripts/build_standards_watch.py`, and rebuilds. The resident agent's autopilot posts an inbox alert when one appears.

## Limits
- Detectors were written from the pages as read on 6 Nov 2026 (through a fetch tool, not a plain HTTP client); the first live run on a machine confirms them and the status shows "layout changed" if a page differs.
- The CBAM price monitor counts the 2026 quarters only; weekly prices from 2027 are not tracked yet.
- The Commission's CBAM default values (Implementing Regulation 2025/2621, corrected by 2026/1740) and free-allocation benchmarks (2025/2620) are not loaded into the calculator, which still uses screening intensities.
- Omnibus I (Directive (EU) 2026/470) is not recorded: it was seen only in secondary sources.
