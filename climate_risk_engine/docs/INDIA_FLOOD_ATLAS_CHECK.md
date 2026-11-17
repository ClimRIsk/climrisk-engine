# India observed-flood data: NRSC atlas ingested, and what it says about our river map

17 Nov 2026. Closes gap 1 of `PEER_REFERENCE_ITC_IIRS.md` (no India-specific observed flood layer) for district scale, and adds CHIRPS as a third rain source for a known storm.

## What was found and ingested

**NRSC/ISRO Flood Affected Area Atlas of India, v1, 1998-2022** (prepared for NDMA, March 2023). Cumulative area mapped as flooded per district, from multi-date optical and SAR satellite flood maps and Central Water Commission gauge levels. Source: `ndem.nrsc.gov.in` (technical document and atlas PDFs, both open to download).
- Parsed 439 district rows over 25 states from the PDF tables. For every state the district sum equals the atlas's own TOTAL within rounding (`scripts/build_india_flood_atlas.py` stores the check).
- Matched to 2021 district boundaries (geoBoundaries, ODbL) by name: 401 of 439. The 38 unmatched are districts created after the atlas districts were drawn (for example the 2022 Andhra Pradesh districts, Delhi's reorganised districts) or spelled so differently that I would be guessing. They are kept as unmatched, never forced.
- `analysis/india_flood_atlas.py` gives, for any point, the district, the atlas hectares, the state rank and the national percentile; for no record it says why (outside India, state not in the atlas, or no matching row) and never returns zero. `GET /india/flood-atlas`, site profile `history.flood_atlas`, and the flood concordance all use it.
- Licence of the atlas: not confirmed (government publication). Only district statistics are stored, no maps or imagery. Boundaries are ODbL (attribution and share-alike).

**CHIRPS v3 daily rainfall** (UC Santa Barbara Climate Hazards Center; satellite infrared plus gauge blend, 0.05 degrees). Open, read by HTTP range request, about 3 s per day. `connectors/chirps.py`, `rain_design.event_cross_check`, ledger entry `rain_event_check`.

## Findings

**1. Chennai 2015, rain.** The record's largest day at the Meenambakkam gauge is 345 mm (2 Dec 2015, an 08:30 to 08:30 observation day). ERA5 gives about 202 mm and NASA POWER about 191 mm as 100-year design values, and ERA5 read about 101 mm for that storm. CHIRPS reads 235 mm on 1 Dec and 36 mm on 2 Dec in the site's cell: best day 235 mm, two days 271 mm, which is about 79 % of the gauge. A gauge day straddles two UTC days, which is why both are shown. Three independently built products now bracket the storm between 100 and 345 mm: the point gauge is the high end and is still the one used for design. (Correction to my earlier note: NASA POWER's precipitation is already IMERG-calibrated, so satellite rainfall was partly in the engine; CHIRPS adds a gauge-blended product at higher resolution.)

**2. Chennai 2015, flood.** Chennai district reads 4,892 ha in the atlas, the 35th percentile nationally, rank 18 of 24 in Tamil Nadu. That is not because Chennai did not flood: the atlas's Tamil Nadu event list is 2007-2010, 2012 and 2020, and the 2015 flood is not in it. An observed layer is only as good as the events somebody mapped, so a low value never clears a site. The engine says this in the output.

**3. The river map against the atlas.** For 25 Indian cities I compared the JRC river-flood rating (our `river_flood` source alone) with the atlas percentile of the district.

| City | Atlas status | Atlas ha | Percentile | JRC river rating |
|---|---|---|---|---|
| Patna | listed | 170,333 | 97 % | LOW |
| Gorakhpur | listed | 158,713 | 96 % | NEGLIGIBLE |
| Guwahati | listed | 22,564 | 63 % | HIGH |
| Kolkata | listed | 1,941 | 25 % | NEGLIGIBLE |
| Bhubaneswar | listed | 73,503 | 82 % | NEGLIGIBLE |
| Surat | listed | 212 | 4 % | MODERATE |
| Ahmedabad | listed | 233,388 | 99 % | MODERATE |
| Hyderabad | not_matched | - | - | NEGLIGIBLE |
| Vijayawada | listed | 29,582 | 67 % | NEGLIGIBLE |
| Chennai-Velachery | listed | 4,892 | 35 % | HIGH |
| Madurai | listed | 12,388 | 51 % | LOW |
| Kochi | listed | 7,663 | 42 % | NEGLIGIBLE |
| Bengaluru | not_matched | - | - | NEGLIGIBLE |
| Pune | not_matched | - | - | NEGLIGIBLE |
| Nagpur | listed | 155 | 3 % | NEGLIGIBLE |
| Lucknow | listed | 3,783 | 33 % | NEGLIGIBLE |
| Jaipur | not_matched | - | - | NEGLIGIBLE |
| Dehradun | not_matched | - | - | NEGLIGIBLE |
| Bhopal | listed | 714 | 15 % | NEGLIGIBLE |
| Raipur | listed | 414 | 10 % | NEGLIGIBLE |
| Ranchi | not_matched | - | - | NEGLIGIBLE |
| Muzaffarpur | listed | 206,533 | 98 % | NEGLIGIBLE |
| Darbhanga | listed | 211,066 | 98 % | LOW |
| Cuttack | listed | 120,270 | 92 % | NEGLIGIBLE |
| Varanasi | listed | 8,160 | 43 % | NEGLIGIBLE |

Of the 7 cities whose district is in the top quarter of the atlas, **6 get NEGLIGIBLE or LOW from the river map alone** (Patna, Gorakhpur, Bhubaneswar, Muzaffarpur, Darbhanga, Cuttack; only Ahmedabad is MODERATE). The rank correlation over the 19 listed districts is 0.19 (p = 0.45), so there is no detectable agreement. Read this carefully: a city point is not the district, and the full site profile also has terrain and rain sources that this check did not use. But it supports the known weakness stated in the river-map caveat (small and mid-size rivers and the Ganga plain are poorly resolved by a global map), and it is exactly the case the new concordance caution is for. Do not quote the river map alone as a flood screen for the Indian plains.

## Limits

District scale, cumulative extent only (no depth, frequency or return period), includes rain-water pooling, only mapped events, 38 of 439 districts unmatched, atlas licence unchecked. n = 25 cities, chosen by me, not a random sample.

## Source register, checked 17 Nov 2026

Each line says what was checked, where, what was found, and what the engine does with it. "Ingested" means the numbers are in the repo with provenance; nothing here is claimed beyond that.

| Dataset | Checked | Finding | Status in the engine |
|---|---|---|---|
| NRSC Flood Affected Area Atlas (district ha) | PDF on NDEM, parsed | Per-state sums match the atlas totals; 401 of 439 districts matched | **Ingested** (`india_flood_atlas.json`) |
| NRSC Landslide Atlas of India 2023, Table 3 | [PDF on isro.gov.in](https://www.isro.gov.in/media_isro/pdf/LandslideAtlas_2023.pdf), parsed | 147 of 147 ranks parsed (Rudraprayag 1, Tehri Garhwal 2, Thrissur 3 agree with the atlas text); 145 matched to 2021 districts, Karbi Anglong and Jaintia Hills were split since and stay unmatched | **Ingested** (`india_landslide_exposure.json`), used as a caution on the slope concordance. It is an exposure rank, not hazard; unranked districts are "not assessed" |
| NRSC licence for the atlases | [NRSC licensing page](https://www.nrsc.gov.in/EOS_dissemination_Licensing) fetched | The page describes a single-user licence for IRS images, allows derivative works for the user's own needs and a credited extract up to 1024 x 1024 px for promotion. It says nothing about the atlas PDFs. Third-party sources (OSM wiki, a Bhuvan handbook) say copyright lies with NRSC and credit is "BHOOSAMPADA, NRSC (ISRO)"; the NDSAP/GODL-India status of atlas data is not confirmed | **Unresolved.** We store district numbers only, with attribution and no maps or imagery. Before any paid product relies on them, ask NRSC in writing (data@nrsc.gov.in). Not legal advice |
| NRSC flood hazard zonation tables | Flood atlas PDF, chapters 6, 8, 9 | Text tables exist only at state level (Assam, Bihar). Andhra Pradesh's district table is an image; OCR gave a partial table (names missing) whose total (550,452 ha) does not match the state table (791,104 ha) | **Not ingested.** OCR did not reconcile; the map layers sit on Bhuvan/NDEM behind a portal I did not log in to |
| CWC / India-WRIS gauge levels | [CWC page](https://cwc.gov.in/en/water-resources-information-system-wris) fetched; portal probed | CWC says unclassified data are free to download after registration; recent data are posted after validation. `indiawris.gov.in` did not respond from this environment. No published API specification found. The third-party [HydroSwift](https://github.com/carbform/HydroSwift) (MIT) scrapes India-WRIS and the CWC flood forecasting site; it does not change the portal's terms | **Not ingested.** Needs a registered download by a person; gauge series would then feed the Chennai-type check |
| GUARDIAN sub-daily discharge (210 India stations, 2020 onward) | Found by search; not downloaded | Published in Scientific Data (2024) | **Not ingested.** Candidate for a gauge benchmark; licence not read |
| Bhukosh (GSI) landslide inventory and the national susceptibility product | Searched | No download source found; Bhukosh did not respond earlier | **Not ingested.** Gap 6 stays open |
| GHSL (GHS-POP, GHS-BUILT-S R2023A) | [Use conditions page](https://human-settlement.emergency.copernicus.eu/GHSLhowToCite.php) fetched | **CC BY 4.0, commercial use allowed**, with credit that cites the peer-reviewed GHSL paper (Pesaresi et al. 2024, Int. J. Digital Earth 17(1)) and the product page; a generic website citation is "insufficient" | **Verified open, not ingested.** Download is by a wizard; the JRC ftp did not respond from here, so no point-query path is wired |
| FABDEM (bare-earth 30 m) | [Fathom page](https://www.fathom.global/insight/fabdem-download/) fetched | Commercial licence or free for non-commercial use; CC BY-NC-SA 4.0 per Fathom/Bristol; some mirrors list other terms, so read the licence in the file itself. FathomDEM is also CC BY-NC-SA | **Not used.** We cannot use it in a paid product without a Fathom licence (fabdem@fathom.global) |
| FastFlood API (reference flood model) | [API quick start](https://fastflood.pages.dev/documentation/api_quick) and [pricing](https://fastflood.pages.dev/products.html) fetched | `POST /v1/model/run-and-wait` with a Bearer key; `autohazard` takes bbox, return period, scenario, resolution. Free plan: personal use, 10 simulations. Commercial use needs Premium (5,000 EUR/year) or Professional (9,500 EUR/year) | **Not run.** No key held. A benchmark for a paid product needs the commercial licence; the pricing page does not say whether a free key can call the API |

What would change the picture most, in order: a written NRSC permission (unlocks the atlas for a product), a registered India-WRIS download of the stations near the benchmark cities, then a FastFlood free key for a handful of pins.
