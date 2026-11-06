# Constants and shallow-logic audit (28 Oct 2026)

Tool: `tools/audit_constants.py` (AST scan of `src/cri`). First run: 126 module-level numeric constants with no source nearby, 128 silent numeric defaults (`.get(k, 0.4)`, `x or 0.18`), 376 comments or strings that admit a guess ("rough", "proxy", "assume", "illustrative"), across 168 of 328 files. The tool finds places to look; it does not prove a number wrong. Data files are covered separately by `cri/provenance.py` and `GET /provenance` (38 files, all registered, `tests/test_provenance.py` fails if a data file has no entry).

## Found and fixed
| Where | Problem | Now |
|---|---|---|
| `climate/physical_risk._MODEL_AGREEMENT` | "fraction of 34 CMIP6 GCMs agreeing", values with no source; used to build P10/P90 bands and printed as "x% of 34 GCMs agree" | removed; P10/P90 fields are `None` |
| `climate/physical_risk._TIPPING_POINT_MULTIPLIER` | 1.20 to 1.90 "95th percentile / median", not reproducible | removed; tipping-stress fields are `None` |
| `data_sources` of that result | listed NASA NEX-GDDP-CMIP6 and NGFS Phase 4, neither used | removed; placeholder inputs are named instead |
| `assess_physical_risk` inputs | country tables for flood depth, heat days, water stress; sector asset value per revenue; sector elevation. A test run gave a USD 904M annual loss for a USD 1B steel company | raises `PhysicalRiskInputError` unless the caller supplies the values or passes `allow_placeholders=True`; then tier is `PLACEHOLDER` and the inputs are listed |
| `ml/risk_predictor` | GradientBoosting fitted to synthetic company-years whose labels are a hand-written formula; docstring claimed EM-DAT, CLIMADA and World Bank training data; "80% bootstrap bands" | `predict` raises `ModelWithdrawn`; the three API/agent call sites no longer call it |
| `agent/job_runner._run_extended_analyses` | default revenue USD 1,000M when missing; sector default Aqueduct score; assumed 18% EBITDA margin into physical risk; `revenue * 0.01` as EAL into the ML layer | no revenue: modules return `not_assessed`; water stress uses the real Aqueduct 4.0 dataset at the company's coordinates or is not assessed; physical risk not assessed; EV and carrying-value proxies are listed under `assumptions` |
| `agent/job_runner` ML block | `ticker` was an undefined name, so the whole ML block raised and never ran | fixed |

## Found, not changed (judgement calls that need a decision or a source)
- `climate/physical_risk`: compound-event matrix (cites "IPCC AR6 Ch.11 Table 11.1", which does not tabulate co-occurrence probabilities), synergy weight 0.12, `_COUNTRY_ADAPTIVE_CAPACITY` (labelled ND-GAIN 2023 but hand-typed), climate credit spread, insurance withdrawal schedule. These still act on the result; the result now says so.
- `climate/portfolio`: `_SECTOR_PHI`, `_TRANSITION_FACTOR_WEIGHTS`, `_SECTOR_MACRO_SENSITIVITY`, `_HAZARD_DAMAGE_FRAC`, flat 0.22 / 0.15 fallbacks.
- `climate/hazard_layers`: regional baselines with `.get(region, 1.5)` fallbacks for ETC, permafrost, dust, hail, subsidence, avalanche, heatwave, GLOF.
- `/assess` path (`api/main`, `ai/tools`): hard-coded 2030 carbon prices 130 / 45 / 18 labelled NGFS Phase 4 (the repo holds Phase 5), default revenue USD 5,000M in `ai/tools`, `scope1 = intensity * revenue * 0.6`.
- `connectors/hf_carbontlens` and `data/carbontlens_tickers.json`: third-party model scores, licence unchecked; `ndgain_extended.json` is partly curated. Registered as `third_party` in the provenance registry.
- `climate/wildfire_risk`, `climate/gis/resolver` (hand-typed elevation, Koeppen, winter temperature grids), `asset_twin` layouts.
- 376 admission strings: not triaged line by line.

## Reading this audit
"Fixed" means the invented number no longer reaches an output. "Not changed" means it still does, usually labelled parametric. None of the unchanged items has been validated against data.

## Addendum 6 Nov 2026: AR6 basis check
- AR6 GWP100 values in `carbon/factors.py` and `lca_factors.json` (CH4 fossil 29.8, non-fossil 27.0, N2O 273) match IPCC AR6 WG1 Table 7.15. The non-fossil CH4 value is never applied in the DESNZ path, because DESNZ gives bioenergy factors as a single CO2e number (flagged as AR5, not restated).
- Fixed: grid factors from Ember (AR5-era lifecycle medians) were not flagged for AR6 inventories; they now are, per factor basis. UK DESNZ and US eGRID values are restated to AR6; AIB and CEA are CO2 only.
- Fixed: DESNZ well-to-tank factors have no per-gas split and stay on AR5; the inventory now says so on every line that adds them when the basis is not AR5.
- Fixed: the product footprint claimed "IPCC AR6 GWP100 applied to every flow". Combustion CH4/N2O and process gases are AR6; well-to-tank (DESNZ 2025) and grid lifecycle (Ember) factors are used as published and are not restated. The result now carries `gwp_note`.
- Unchanged and labelled elsewhere: shipping FuelEU tank-to-wake uses AR4 by regulation; EPA Hub factors and oil-and-gas methane use AR5.
