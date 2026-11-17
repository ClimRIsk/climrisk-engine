# How ITC (University of Twente) and IIRS/NRSC build hazard models, and what ClimRisk is missing

17 Nov 2026. Desk research from public pages and abstracts; I did **not** read the primary papers (for example van den Bout et al. 2023 in full), so method details below are as summarised by their authors' blog, abstracts and course material. "IIRS" is taken to mean the Indian Institute of Remote Sensing (ISRO, Dehradun), ITC's joint-degree partner. Flood atlases and early warning are done by NRSC (ISRO), a sister body, not IIRS; the two are kept apart below.

## 1. ITC, University of Twente

**Risk framework.** Risk = hazard x vulnerability x amount of elements at risk (some versions divided by capacity). Exposure is the GIS overlay of hazard intensity maps on elements-at-risk maps; damage per intensity comes from vulnerability curves; losses are integrated across hazards and return periods. Social vulnerability comes from national census data; wider vulnerability uses indicators combined with hazard indicators by spatial multi-criteria evaluation. Multi-hazard work is expected to study cascading and concatenated hazards. (ITC training and CHARIM material by van Westen's group.)

**Model development style (FastFlood, van den Bout, Jetten, van Westen, Lombardo 2023, Environmental Modelling & Software).** Physics first, then shortcuts, then checked against a slower reference solver: separate routing from pressure effects, compute peak depth only, solve a steady state first and compensate, use the fast-sweeping algorithm. Self-reported agreement with full Saint-Venant simulation: more than 97 % flood extent and 95 % depth, about 1500 times faster; one 2026 conference abstract reports 98.5 % extent similarity to HEC-RAS on a Canadian river. The authors say plainly that the speed depends on assumptions and that results suffer when those fail, and that wide use creates a risk of misuse, so they are adding uncertainty visualisation, ensembles, open training material and links to calibration data. It runs in the browser (WebAssembly) and has been spun out as a company (Fast Hazard).

**Data FastFlood links automatically.** SRTM and Copernicus 30 m elevation (the authors cite a 30 m map with forests and buildings removed as a comparison point), ESA WorldCover 10 m land cover, ECMWF forecast precipitation, OpenStreetMap buildings and roads for exposure. Users add levees, reservoirs, pumps, land-cover and channel edits.

**Other ITC lines of work.** Integrated physically based multi-hazard modelling (van den Bout's PhD: earthquake-triggered landslides, cyclone-driven flood plus debris flow; applied in Tajikistan and Dominica). Deep-learning building damage mapping from UAV, oblique and satellite imagery (Kerle, Nex, Duarte, Vetrivel). A browser-based 3D flood digital twin for small island states built on FastFlood (a 2025 MSc thesis). The Disaster Resilience curriculum pairs data-driven hazard modelling, physically based hazard modelling, multi-hazard risk assessment, weather and climate, and machine learning for geospatial data, with volunteered geo-information.

## 2. IIRS and NRSC (India)

**IIRS** is an ISRO unit (separate since 2011) for training, education and research in geoinformatics; it runs a PG diploma in Natural Hazards and Disaster Risk Management and, since 2002, a joint MSc and PG diploma in Geoinformatics with ITC (agreement renewed Nov 2024). IIRS-linked research that I found: multi-temporal, multi-frequency, multi-polarisation SAR flood mapping with optimised delineation thresholds, with the known pitfalls stated (wind or rain roughening of water can make flooded area be under-estimated; arid sand can make it over-estimated); a proposal to fuse satellite data, social-media imagery and topography for rapid flood mapping; and landslide susceptibility in the Himalaya by statistical methods (logistic regression, certainty factor, information value, frequency ratio; AUC about 0.85 to 0.88 in one Siwalik study) and machine learning (random forest, ANN, MLP, deep learning, 70/30 splits), using SRTM DEM, Sentinel-2, IMD rainfall and inventories from the Bhukosh portal. I could not confirm DEM choice, inventory construction or Cartosat use for IIRS itself.

**NRSC** (not IIRS): Flood Affected Area Atlas of India built from multi-date optical and SAR satellite inundation maps, 1998 to 2022, plus Central Water Commission gauge levels; district and state statistics on the NDEM geoportal; flood hazard zonation atlases for six states (Assam, Bihar, UP, West Bengal, Andhra Pradesh, Odisha). NRSC notes that mapped "flood" can include rain-water accumulation in low land. Operational spatial flood early warning for the Godavari and Tapi under the National Hydrology Project, and a pan-India runoff forecast being operationalised. I did not find the hydrodynamic model details.

**Style.** Observation first: decades of satellite archives become hazard layers; statistical or ML susceptibility is trained on inventories and scored with success and prediction rate curves; products are delivered as atlases and operational early warning; people are trained to run them.

## 3. What ClimRisk is missing, in order of importance

| # | Gap | What they do | What we have | Honest status |
|---|---|---|---|---|
| 1 | **India-specific observed flood layer** | NRSC atlas 1998-2022 plus CWC gauge levels | CEMS events (72, 2023-2026), JRC surface water, a Sentinel-1 water-history entry | No Indian cumulative inundation or gauge data. For BRSR and Indian clients this is the biggest gap. Access and licence of the NDEM atlas data are unchecked. |
| 2 | **Validation against a reference hydraulic model** | FastFlood checked against full Saint-Venant (and HEC-RAS) | Own local-inertial solver; new rain-on-grid setup not scored against any outline | Needs a benchmark: our solver vs a reference model and vs FastFlood (free web tool) on the same sites, as a cross-source check. |
| 3 | **Bare-earth terrain** | Compares forest-and-building-removed elevation models | SRTM / Copernicus / Terrarium tiles (surface models, built-up areas read as raised) | Not done. FABDEM-type data is licensed non-commercial; check before use. |
| 4 | **Compound and cascading hazards** | Integrated multi-hazard physics (rain to landslide to debris flow, cyclone rain plus surge) | Peril-by-peril; the concordance compares sources per peril; disclosure text assumes independent events | Not modelled. Cyclone rain, river and surge are not combined at a site. |
| 5 | **Elements at risk and vulnerability** | Building footprints, census, typology, vulnerability curves, deep-learning damage mapping from imagery | User-supplied assets, OSM footprints, JRC depth-damage curves | No building typology, no local curves, no social vulnerability, no capacity term. |
| 6 | **Inventory-trained landslide susceptibility** | Frequency ratio, information value, ML on inventories, AUC with success and prediction rate curves | Infinite-slope physics; check against NASA catalog: mean AUC about 0.70 at 5 sites | No Indian inventory (Bhukosh), no trained model, no rate curves. |
| 7 | **Spatial multi-criteria evaluation with AHP** | Indicator layers weighted by pairwise judgment | An AHP module with consistency ratio, not connected to any spatial composite | The method exists but is not used on a map. |
| 8 | **SAR flood mapping detail** | Multi-temporal change detection, optimised thresholds, stated pitfalls | Sentinel-1 water history entry; MODIS and SAR miss dense-urban flooding (stated in the caveat) | Threshold handling and wind-roughening checks not documented. |
| 9 | Operational forecasting, citizen and social-media data | NRSC early warning, IIRS social-media fusion | None | Out of scope for a screening engine; do not claim it. |

**Where we are level or ahead (say it, with evidence only):** a per-site uncertainty ledger, multi-source concordance by peril and explicit not-assessed handling, gauge-anchored design rain with source comparison, a second climate-model pipeline check, and a 3D twin with protection what-ifs. FastFlood's own authors list uncertainty visualisation and ensembles as future work.

## 4. Suggested order of work

1. Benchmark our solver against FastFlood and, where available, EA outlines, for the same pins (closes gap 2 and gives a third flood source).
2. Check access to NRSC/NDEM atlas layers and CWC gauge data; ingest if the licence allows (gap 1).
3. Wire AHP to a spatial indicator composite with sensitivity (gap 7).
4. Compound event: cyclone rain plus surge plus river at one site (gap 4).
5. Landslide: train frequency-ratio or ML on an inventory where one is open, report success and prediction rate curves (gap 6).

## Sources

- [FastFlood blog by van den Bout (EGU, 2024)](https://blogs.egu.eu/divisions/gd/2024/01/10/can-we-predict-floods-accurately-fast-and-everywhere)
- [University of Twente: FastFlood Global spin-off](https://www.utwente.nl/en/news/2025/11/676280/dutch-spinoff-launches-groundbreaking-global-flood-forecasting-tool)
- [ITC: agreement with IIRS to extend joint education (Nov 2024)](https://www.utwente.nl/en/itc/news/2024/11/1865489/faculty-itc-and-iirs-sign-agreement-to-extend-joint-education-programme)
- [ITC Disaster Resilience curriculum](https://www.utwente.nl/en/education/master/programmes/geo-information-science-earth-observation/specialisations/disaster-resilience/courses-research/)
- [ITC risk analysis guidebook, van Westen (session 6)](https://filetransfer.itc.nl/pub/westen/Multi_hazard_risk_course/Guidebook/Session%2006%20Risk%20Analysis.pdf)
- [CHARIM use case book: current risk](https://cdema.org/virtuallibrary/index.php/charim-hbook/use-case-book/4-planning-alternatives/4-2-analyze-current-risk) (found via search; not read in full)
- [NRSC Flood Affected Area Atlas of India](https://www.nrsc.gov.in/sites/default/files/pdf/DMSP/Flood%20Affected%20Area%20%20Atlas%20of%20India%20-Satellite%20based%20study.pdf)
- [NRSC presentation to NIDM](https://nidm.gov.in/pdf/acrc26/Preparatory%20Session/4.%20NRSC.pdf)
- [IIRS-linked SAR backscatter project (India S&T portal)](https://www.indiascienceandtechnology.gov.in/research/backscatter-analysis-using-multi-temporal-multifrequency-and-multi-polarization-sar-data-context)
- [ITC UAV and deep-learning damage mapping (Remote Sensing 2019)](https://mdpi.com/2072-4292/11/3/287)
- [GeoTwin: browser 3D flood digital twin thesis (ITC, 2025)](https://essay.utwente.nl/fileshare/file/108659/GeoTwin-s3245780-Az%20Ali.pdf)
