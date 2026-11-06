# National grid factors: Australia, Canada, Germany, China, Japan (6 Nov 2026)

Location-based Scope 2 factors from each country's own publisher, in `src/cri/data/national_grids.json`, built and cross-checked by `scripts/build_national_grids.py`, resolved by `carbon/national_grids.py` and returned by `electricity_location(country, ..., fy=, region=)`, `GET /carbon/grid-factor?country=AUS&region=VIC`, and the carbon inventory (`grid_region` on an electricity record or the "Electricity" sheet of the assurance workbook).

Every number was read on 6 Nov 2026 from the publisher's document, not from an aggregator.

| Country | Publisher and document | Grain | Default | GWP basis | Year |
|---|---|---|---|---|---|
| Australia | DCCEEW, National Greenhouse Accounts Factors 2026, Table 1 (Scope 2 and 3) and Table 2 (residual mix) | 8 states / grids + national | national 590 g | AR5 | financial years to June |
| Canada | ECCC, National Inventory Report 1990-2024 (Apr 2026), Annex 7 tables A7-1 to A7-14 | 13 provinces/territories + national | national generation intensity 102.17 g (2024, preliminary) | AR5 (see below) | calendar 2024 and 2023 |
| Germany | UBA, CLIMATE CHANGE 16/2026 (Mar 2026), Tables 1-3 | national, 2020-2025 | 2025 GHG 352 g (estimate) | AR5 (stated in UBA s.2.8) | calendar |
| China | MEE + NBS, Announcement 2025 No. 47 (31 Dec 2025) | 30 provinces, 7 regions, national | national 530.6 g | CO2 only | data year 2023 |
| Japan | MOE/METI SHK electricity supplier list, R8 submission rev4 | none: supplier-specific | substitute value 422 g | CO2 only | R6 results |

## What each country returns beyond `g_per_kwh`
- **Australia**: `scope3_g_per_kwh` (upstream and network losses, Table 1) and the national **residual mix** 790 g for the market-based method. "WA" alone means the South West grid; the Pilbara grid is `WA-NWIS`.
- **Canada**: `consumption_g_per_kwh` (adds transmission losses and SF6, and reflects imports). The headline is the generation intensity; a note appears when the two differ a lot (Prince Edward Island: 3 g generation, 265 g consumption).
- **Germany**: `co2_only_g_per_kwh` (UBA's headline "Emissionsfaktor Strommix", 344 g in 2025), `with_upstream_g_per_kwh` (406 g), `inland_consumption_co2_g_per_kwh` (nets the trade balance), and the status of the year (inventory, provisional, estimate). Market-based Scope 2 still uses the AIB residual mix.
- **China**: the national value excluding market-traded non-fossil electricity (609.6 g) as the **residual mix** for the market-based method (MEE: green power bought with traceable certificates counts as zero, the remainder uses this factor), and the fossil-electricity factor (827.3 g). Province codes are ISO 3166-2:CN (HB Hubei, HE Hebei); Chinese names work too.
- **Japan**: only the substitute value. Japan publishes factors per electricity retailer (basic and adjusted, some per menu), not a national average. Enter the retailer's factor as `supplier_g_per_kwh` or `location_g_per_kwh` when you know it.

## Checks in the build script
Australia: all values inside 0-1 kg, national inside the state range, residual mix above the national average. Canada: 14 jurisdictions, national inside the provincial range, consumption intensity never below generation intensity. Germany: CO2 g/kWh equals emissions / consumption within 2 g (rounded inputs), GHG with upstream equals GHG without + upstream within 1 g, CO2 < GHG < GHG with upstream. China: 30 provinces with unique codes and 7 regions, and our figures **reproduce the year-on-year changes MEE states** (-1.1%, +4.1%, -0.6%). A transcription error fails the build; the test suite also fails if the JSON is not exactly what the script writes.

## Honest limits
- Canada's Annex 7 says "CO2 eq" but does not name the GWP set. Canada's inventory uses AR5 values (NIR methodology chapters; not re-read for this file), so the factor is labelled AR5 and the note says so.
- Australia's publication date is not stated on the PDF; it is the 2026 edition and cites a decision of 24 July 2026, so it came out after that.
- Japan: the substitute value is the only country-wide number; the ten regional retailers' own factors were not extracted (the PDF text loses the alignment between names and values).
- China publishes with a two-year lag (2023 data in Dec 2025). A 2024 or 2025 reporting year therefore uses the 2023 factor and the note says so.
- The monitor sees new editions; it does not load them. Rebuild with `python scripts/build_national_grids.py` after updating the tables in the script from the new publication. China's MEE cannot be checked automatically.
- T&D losses and upstream (Scope 3 category 3) for these countries still come from the library's own loss and upstream tables (not from these publishers), applied to the national direct factor; supplier or national data would be better.
