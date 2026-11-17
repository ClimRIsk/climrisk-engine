# How physical-risk vendors validate their models, and where ClimRisk is missing a link  (17 Nov 2026)

Vendor material is marketing. Nothing below about a vendor is independently verified; where a vendor publishes no training or validation detail, that is stated. Sources at the end.

## 1. What vendors say they use and how they check it

| Vendor | Data and models (their claims) | Validation (their claims) | What is not public |
|---|---|---|---|
| Moody's RMS | Hazard x exposure x vulnerability; flood at 30 m (was 90 m) | Vulnerability curves from engineering studies, codes and claims; says hundreds of billions of dollars of claims and post-event surveys | Method documents sit behind a support login |
| Fathom | LISFLOOD-FP hydrodynamic model, about 30 m, on bare-earth terrain (FABDEM) | Peer-reviewed benchmarks against FEMA, USGS and UK outlines and observed urban floods | Commercial model internals |
| Jupiter | "Dozens" of climate models, debiased and downscaled | Model-risk-management pack: 500+ pages, 60+ validation tests; says Tier 1 banks validated it | Data sources, ML detail, metrics |
| First Street | Flood model checked against historic events and claims | Expert panel; an independent 2024 study found high uncertainty at property scale | Full methodology; heavy-rain floods are not recreated |
| Cervest | Curated CMIP6 subset calibrated to ERA5 plus observations | Not found | Training procedure, metrics |
| Climate X | "Physics before AI" digital twin, 11 hazards | Claims 95-99 % accuracy | The test behind the number |

Independent view: GARP (13 vendors, Oct 2025) found wide dispersion between vendors; the Investor Leadership Network (7 vendors, Jun 2026) found they often disagree on which hazard is the main one, and asked for vendor transparency and uncertainty measures.

## 2. Link by link: where ClimRisk stands

| # | Link vendors have | ClimRisk, today | Status |
|---|---|---|---|
| 1 | Loss calibration against claims | Transferred JRC depth-damage curves, shown as a transfer spread; no claims or loss data | **Missing.** Needs a loss dataset from a partner (insurer, lender, public agency); none held |
| 2 | Published skill scores on the hazard model | Surface-water model re-scored 17 Nov 2026: AUC 0.84 / 0.85 / 0.74 in three English towns, beaten by a plain terrain index in two; depth unvalidated (`PLUVIAL_BENCHMARK.md`) | **Done for ranking, open for depth.** Needs observed depths or gauge levels |
| 3 | Independent review / model-risk pack | Ledger, glass-box docs, validation ledger; no external reviewer, no bank-run validation | **Missing** |
| 4 | 30 m bare-earth terrain | Copernicus / EA LiDAR (England) on a 60 m grid; FABDEM is non-commercial | **Behind** outside England |
| 5 | Gauge and observed-event data for the home market | NRSC flood and landslide atlases at district scale; no CWC gauge series | **Partly closed** (`INDIA_FLOOD_ATLAS_CHECK.md`) |
| 6 | Transparent uncertainty | Per-site ledger, multi-source concordance, "not assessed" handling | **Ahead of what vendors publish** (on the evidence above) |

## 3. What the re-score found (link 2)

Re-scoring turned up a defect in our own data path: unflagged NOAA gauge outliers (four 255-258 mm "days" at Heathrow) had set a 1-in-50 design rain of 195 mm for an English site; the corrected value is 59 mm. The quality-flag filter is now in the engine, tested, and reports what it removes. Vendors keep this class of error out of sight; here it is documented. Details in `PLUVIAL_BENCHMARK.md`.

## 4. Order of work from here

1. Re-run the station-versus-grid checks that read GHCN stations before the filter existed (`studies_evt`, offshore wind).
2. Depth validation: obtain gauge levels (CWC via a registered India-WRIS download) or surveyed depths for one event (Chennai 2015) and score modelled depth, not only rank.
3. Loss calibration: find one named-event loss dataset (an insurer or lender partner; public event-level loss databases give totals, not asset claims).
4. External review: a one-off independent reviewer's note on the ledger and benchmark method; vendors sell this as a pack.
5. Bare-earth terrain outside England: only with a commercial licence for FABDEM or a different open source.

## Sources

- [Jupiter: verification and validation](https://www.jupiterintel.com/blog/can-you-trust-your-climate-data-how-verification-and-validation-make-the-difference)
- [Jupiter MRM Accelerator](https://jupiterintel.com/products/mrm-accelerator)
- [First Street: Flood Factor accuracy](https://help.firststreet.org/hc/en-us/articles/360056405714-How-do-I-know-if-my-Flood-Factor-score-is-accurate)
- [Smart Cities Dive on the UC Irvine/AGU comparison](https://www.smartcitiesdive.com/news/flood-climate-risk-data-models-accurate-first-street-UC-irvine/723496/)
- [Cervest combined signals and ratings methodology](https://knowledgebase.earth-scan.com/hubfs/EarthScan_Combined%20Signals%20%26%20Ratings%20Methodology.pdf?hsLang=en)
- [Climate X methodology](https://www.climate-x.com/spectra/methodology)
- [Moody's: validated financial metrics for physical risk](https://climate.moodys.com/climate-blog/more-than-checking-a-box-delivering-real-world-validated-forward-looking-financial-metrics-for-physical-climate-risks)
- [Moody's Climate on Demand v2](https://climate.moodys.com/climate-blog/understand-physical-climate-risk-introducing-moodys-climate-on-demand-version-2)
- [Fathom/Bristol 30 m validation and FABDEM comparison (NHESS 2024)](https://doaj.org/article/a80a0008d35f4bf993d7ba52f7331515)
- [GARP: why vendors disagree](https://www.garp.org/risk-intelligence/sustainability-climate/why-vendors-disagree-260630)
- [GARP benchmarking of 13 vendors](https://altiorem.org/?p=20443)
- [Investor Leadership Network: Why Vendors Disagree (PDF)](https://investorleadershipnetwork.org/wp-content/uploads/ILN-Physical-Risk-Paper-Why-Vendors-Disagree.pdf)
