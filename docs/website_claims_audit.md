# climrisk.io — claim-by-claim audit against the engine (30 Sep 2026)

Legend: ✅ true in the engine · 🔧 built/fixed in this pass · ✏️ website copy corrected · ⚠️ still open

## API surface (engine page, section 04)
| Claim | Finding | Action |
|---|---|---|
| POST /regulatory/sfdr-pai, /eu-taxonomy, /pcaf | Modules existed; routes missing from the local engine (only in the old Render build) | 🔧 ported + tested |
| POST /stress/event, GET /stress/event/catalogue | same | 🔧 ported + tested |
| POST /physical/slr, /physical/biodiversity | same | 🔧 ported + tested |
| POST /portfolio/benchmark | Read a non-existent field (always crashed); "benchmark" values were invented MSCI premia and active VaR was hard-coded to 0 | 🔧 rebuilt: portfolio and index (user-supplied factsheet sector weights) both run through the engine's portfolio model; active = sector allocation + regional exposure |
| POST /portfolio/counterparty | Used the same invented MSCI premia | 🔧 now uses the engine's own EAL and CVaR99 contributions |
| /compare/scenarios | Called a function that never existed | 🔧 rewritten on NGFS Phase 5 carbon prices |
| /regulatory/sfdr-pai/export | Wrong arguments + non-existent fields | 🔧 rewritten (xlsx + csv) |
| POST /companies / GET /companies | `Commodity.OTHER` crash / 500 on the local engine | 🔧 fixed |
| /portfolio/risk, /credit, /stream, /assets/batch | Missing on the local engine | 🔧 ported + tested |

## Science & data claims
| Claim | Finding | Action |
|---|---|---|
| "Dynamic Life Cycle Assessments" | No LCA code existed; page described physical risk | 🔧 `cri.lca`: ISO 14067-aligned cradle-to-gate (A1–A3) for cement/clinker, steel, aluminium, ammonia and custom inventories; IPCC 2006, GCCA, Ember, DESNZ 2025, AR6 GWP100; Monte Carlo P5–P95; exclusions listed; NGFS grid pathways to 2050; EU CBAM view. API `/lca/*`, workbench `/lca/dashboard`, app sidebar, agent tool, CLI. ✏️ copy rewritten |
| "XGBoost trained on 14,000+ verified CDP disclosures" | False — estimator is trained on EPA GHGRP observed emissions matched to SEC revenue | ✏️ corrected; held-out accuracy published on /validation |
| "GradientBoosting trajectory trained on NGFS + decades of disaster-loss data" | Surrogate trained on 50,000 simulated company-years calibrated to NGFS/EM-DAT | ✏️ corrected |
| "WRI Aqueduct … 13 indicator layers", live API | The "live API" host does not exist; values came from a hand-typed regional table | 🔧 real Aqueduct 4.0 dataset (68,506 units, 0.05° grid, 13 indicators, industry weightings, 2030/2050/2080 projections) shipped with the engine; `/physical/water-risk`, `/agent/water-risk` lookup, Risk Analyst water context, CLI `cri water`. ✏️ copy |
| "26 hazards on a 0.1° grid" | 25 parametric regional functions; site-level models are separate | ✏️ corrected everywhere (home, engine, methodology, use-cases, showcase) |
| "CMIP6 34 GCMs", "NASA NEX-GDDP", "GRACE", "ERA5" | Not used (ERA5 only with an optional CDS key) | ✏️ corrected to CMIP6 HighResMIP via Open-Meteo, NASA POWER/FIRMS, Sentinel, WorldCover |
| "NGFS Phase 4" | Risk Analyst uses NGFS Phase 5 (7 scenarios, 3 models) | ✏️ corrected on all pages and the footer (MACC chart still cites the Phase 4 prices it actually uses) |
| Validation: 91% vs MSCI, ±8% vs Moody's, r=0.87 vs S&P CSA | Fabricated — no such benchmark exists | ✏️ replaced with the real, reproducible tests (TC winds vs 148 stations; flood vs FEMA claims, 3 counties; emissions vs EPA, 163 companies; world QA), weak results included |
| "Model confidence, audited 92%" (home) | No audit exists | ✏️ replaced with verifiable counts |
| TNFD "KBA / protected-area proximity", "WDPA" | ~20 hand-drawn bounding boxes | 🔧 live OpenStreetMap protected areas (containing / 5 km / 25 km, IUCN class where tagged; ODbL), bounding boxes only as a labelled offline fallback. ✏️ copy says KBA/WDPA need a licence |
| "Black Summer, Texas freeze and nine others" (13 events) | 8 events in the replay catalogue | ✏️ lists the 8 |
| SBTi "deviation alerts vs target" | Only recommendation text | 🔧 `/sbti/alignment`: SBTi ACA minimums (4.2 %/yr S1+2, 2.5 %/yr S3, 90 % by 2050), base-year/timeframe checks, deviation alerts |
| SEBI/BRSR report | No endpoint; INR intensity always empty | 🔧 `/reports/brsr` with tCO2e per INR crore (company turnover or supplied FX rate) and optional PPP-adjusted intensity |
| TCFD / ISSB / CSRD / BRSR report numbers | Unit bugs: emissions (MtCO2e) labelled tCO2e; USD m amounts divided by 1e6 again, so carbon cost, capex and losses showed 0.0 | 🔧 fixed in `outcomes/disclosure.py` |
| "Export … into a single consolidated CSV" | /runs/export returned Excel for one company | 🔧 `POST /portfolio/export.csv` and `cri ingest` — one row per asset through the Risk Analyst |
| `cri ingest portfolio.xlsx` terminal demo | No CLI existed | 🔧 `cri` command (ingest, run, water, lca, serve); platform terminal now shows real output |
| Cases: "Real portfolios … every number is reproducible" | Mining and bank cases are not clients; Heineken/Shell figures came from an older model | ✏️ Heineken/Shell re-run on the current engine (`cri run --company …`), others labelled illustrative, CRI scores removed |
| "Map 10,000 assets in under 5 minutes" | Measured: screening (`/assets/batch`) 10,000 assets in 0.4 s; full Risk Analyst median 3.4 s/asset cold, 0.03 s warm | ✏️ "screen 10,000 assets in seconds" |
| Company page: LCA "accredited" | No accreditation exists | ✏️ removed |
| ClimateBERT (4 classifiers), AI agent with email, PCAF 6 asset classes, SFDR 18 PAI, EU Taxonomy, DCF/WACC/MACC, VaR 95/99, 18 commodities, PD/LGD, SLR | Present in code | ✅ |

## One-stop-shop gaps built in the second pass (30 Sep 2026)
| Gap | What was there | Action |
|---|---|---|
| Carbon auditing (Practice 01) | Only revenue-based estimates (~3.5× median error) or reported figures; no inventory from client data | 🔧 `cri.carbon`: GHG Protocol / ISO 14064-1 inventory from activity data — Scope 1, Scope 2 location + market (instruments → supplier → residual mix → location, disclosed), 15 Scope 3 categories, cat. 3 derived automatically; operational/financial control or equity share; base-year recalculation test; completeness rule (every S3 category reported or excluded with a reason); assurance pack (.xlsx audit trail) and ISO 14064-1 report content; intake template upload. Factors: DESNZ 2026 (AR6-restated), eGRID2023, Green-e 2025, AIB 2025, Ember-derived fallback, World Bank T&D, USEEIO v1.3 + CPI. `/carbon/*`, workbench, app sidebar, 14 tests. ✏️ capabilities/home/frameworks/methodology copy |
| Placeholder numbers in `/agent/assess` | ITR hard-coded 2.4 °C for every company; loss bands = 1/2.5/6 % of EV; decision always WATCH; disclosure scores 0; unknown companies silently replaced by the first registry company; first-word name matching | 🔧 all removed/replaced with engine outputs; no silent substitution (502); ITR withheld (see limitations); EV impacts vs Current Policies from the same model |
| Legacy parametric builder bugs | Sector hints not recognised ("materials" → technology); zone hazard *severity* used as production-loss fraction (90 % of output lost by 2050); emissions per unit divided by price instead of multiplied (~25× error) | 🔧 GICS hint mapping + whole-word aliases; severity × sector sensitivity; unit fix; synthetic Scope 1–3 now from the GHGRP-trained estimator |
| Homepage "live" widget | Ran the legacy parametric engine with hard-coded US jurisdiction and $5bn revenue for every company | 🔧 `/public/company-risk`: Risk Analyst on the company's mapped assets (asset register), honest "not mapped" answer otherwise; ✏️ widget + copy rewritten |
| Risk Analyst totals | Physical + capped transition loss could exceed asset value (e.g. BHP 123 % of value) | 🔧 total loss per asset capped at its value (closure) |
| Transition plan | MACC and SBTi check existed separately | 🔧 `cri.transition_plan` (targets → levers → capex/opex with reinvestment → path, gap, MACC, NGFS avoided carbon cost, SBTi check, ESRS E1-1 / TPT fields), can seed from the inventory; `/transition-plan`, workbench, 8 tests |
| Real estate | None (pitched on use-cases page) | 🔧 `cri.real_estate`: EUI and operational carbon per building, grid decarbonisation + retrofits to 2050, stranding year vs a **user-supplied** pathway (CRREM is licence-restricted for paid software, so not embedded), NL/UK/FR/EU regulatory checks; `/real-estate/assess`, workbench, 6 tests |
| Double materiality | Impact-side scoring module only | 🔧 `cri.materiality` IRO register (ESRS 1 ch.3 criteria, stated formula, stakeholder input, checks) + E1 starter IROs from engine results; `/materiality/*`, workbench |
| Carbon markets | None | 🔧 `cri.carbon_markets`: EU ETS balance and CBAM free-allocation phase-out projection; carbon-credit screen (CCP label, removal/durability, vintage, SBTi and EU 2024/825 claim rules); `/carbon-markets/*`, workbench; 6 tests |
| Operator response and earnings (transition-risk gap analysis, 2 Oct 2026) | Emissions held flat to 2050 (cost of doing nothing only); no link between carbon/energy prices and earnings; credit module needed typed-in scores | 🔧 `finance/abatement.py`: sector levers (cement, steel, coal/gas power, coal mining, oil & gas, refining, ammonia) with sourced cost ranges, lead times, availability; a lever starts when the net carbon price the plant faces reaches its levelised cost, then persists; unabated cost kept as "cost of inaction". `finance/earnings.py` (optional financials): EBITDA channels (volume, carbon, abatement opex, power, fuel, commodity price — NGFS price differences to Current Policies), operating closure, net-debt roll-forward, EBIT coverage → Damodaran synthetic rating → S&P class PD, ECL, covenant breach probability, IAS 36-style climate-attributable impairment. NGFS coal/gas/oil prices added to the data. API (`financials`, `abatement`, `levers`), upload columns, Risk Analyst cards 5–6, 25 tests |

## Known limitations left as they are (documented, not advertised)
| Item | Note |
|---|---|
| Portfolio loss model (`/portfolio/risk`) | Sector × region model; company physical/transition scores do not change EAL/VaR (schema now says so). Asset-level risk is in the Risk Analyst |
| Legacy CRI rating / company engine | Still internally inconsistent for many companies (e.g. rating A with a large NZE EV loss; negative EVs). Kept for the app's existing assessment flow and labelled `model_status: legacy`; the Risk Analyst is the recommended path. Recommendation: rebuild the app's assessment flow on the Risk Analyst |
| Implied temperature rise (`cri.climate.paris_alignment`) | Sector budget shares far too small — rated 12 %/yr reducers above 2 °C. Withheld from `/agent/assess`; still called by the batch job runner (`agent/job_runner.py`) — needs recalibration before any client use |
| `data_acquisition/eutl_client.py` | Calls an EUTL API host that does not resolve (like the old Aqueduct "live API"); fails silently. Verified emissions must come from the client or the Union Registry downloads |
| Public widget on Render | Needs the Render service redeployed with this engine version; until then the homepage widget cannot reach `/public/company-risk` |
| Flood skill | Area model shows moderate skill in 3 US counties; no evidence yet outside the US |
| Abatement and earnings layers | Lever costs are published ranges (some analyst judgement, flagged per lever), flat in real terms, mixing source years; decision price is net of pass-through (conservative — no windfall credit); myopic operator; no financing or permitting limits; levelised costs charged only to 2050. Earnings: financials must match the emissions perimeter; rating/PD are a coverage-based proxy; no refinancing or waivers; impairment without terminal value (increment vs baseline only). Not yet modelled: plant age and legal phase-out dates, CBAM exposure by export share, demand elasticity beyond NGFS sector output, business interruption |
