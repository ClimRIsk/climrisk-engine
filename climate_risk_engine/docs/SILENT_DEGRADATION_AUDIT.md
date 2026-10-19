# Silent-degradation audit (19 Oct 2026)

Question: where can the engine turn a *failure* or *missing data* into something that reads like an *answer*?
Method: `tools/audit_fallbacks.py` (AST scan of every broad `except`) and `tools/audit_imports.py` (every `from cri… import name`, including
lazy imports on fallback paths, resolved against the real module). Re-run both after changes; `tests/test_import_integrity.py` guards the second.

## Census
450 broad exception handlers in 298 files (98k lines). By what the handler does:
VALUE 100 (returns/assigns a concrete value) · SWALLOW 130 (pass/continue) · LOGONLY 118 · NONE 47 · RERAISE/error-structure 55.
All 100 VALUE handlers and the SWALLOW/LOGONLY handlers in compute code were read (cache read/write, progress callbacks and file cleanup were
not individually reviewed beyond a glance — 60+ of the SWALLOW class are those).

## Fixed
| Where | Was | Now |
|---|---|---|
| `/portfolio/stream` live feed | on any price-fetch failure, a **random ±2 % move** was fed into the live VaR/exposure numbers | no price → exposure unchanged, each position reports `price_source` (`yfinance` or `unavailable`) |
| `analysis/portfolio._composite_score` | failure → **50.0 "Moderate"** blended into the portfolio average | raises; the holding cannot be averaged |
| `climate/spatial_downscaling` WRI step | fallback imported a name that does not exist → whole downscaling raised whenever the web API gave nothing | uses the local Aqueduct 4.0 dataset first, then the web API; otherwise labelled `UNAVAILABLE … placeholders, not data` |
| `climate/spatial_downscaling` GIS and NASA POWER | failure → elevation 0 m / 999 km coast / 28 °C etc. labelled as the real source | labelled `UNAVAILABLE … placeholders` in `data_sources` |
| `climate/physical_risk` wildfire and sea-level-rise stages | failed stage → loss 0.0, silent | added to `model_agreement_notes`: "not included … not the same as zero" |
| `climate_scenario._wind` | cyclone lookup failure → "not exposed" | `tc_status: unavailable` and a note that it is NOT evidence of no cyclone exposure |
| `/batch/assess-assets` | country lookup failure → "GL" silently | result flagged ESTIMATED |
| `engine/pipeline`, `orchestrator`, `operations/company` provenance swallows | silent | log a warning |
| `engine/pipeline._get_company_from_run` | imported `COMPANY_REGISTRY` from a module that does not have it, so registry companies always fell to the proxy rating | uses `companies_seed.all_seed()` |
| (earlier today) Climate Monitor fallback, unknown sector → technology, default revenue "user-supplied", open-sea sites, polar OOM | see `STRESS_TEST_RESULTS.md` | fixed |

## Found, deliberately NOT changed (needs a decision)
- **`climate/hazard_financial._structural_damage_fraction`**: the intended JRC depth-damage path imported two names that do not exist, so it has **never run**; every flood hazard in this older company-level model has used a flat 5 % of replacement value. Wiring it up would give 27 %–92 % at the 0.8 m median depth, and the "chemical" (d50 0.4 m) and "critical_infrastructure" curves are not credible against the JRC publication. Left as the stated flat default with a comment; the curves need review before anyone enables them.
- **`climate/physical_risk` SLR stage** passes `asset_lat=0, asset_lon=0` to the sea-level-rise function, so its regional multiplier is always the default 1.0. The company-level model is parametric by design, but the SLR "regional" claim is not location-specific.
- `agent/job_runner` shallow fallback imports `cri.chronic.flood`, which does not exist; it reports "Flood assessment unavailable" honestly, so left.
- Other labelled fallbacks that are acceptable: landslide/flood design rainfall 150 mm (flagged in the output text), insurance hazard fetch failures (flow into "NOT ASSESSED, never zero"), `asset_twin` gaps lists, `siteprofile` nulls.

## Not covered
Resident-agent and LLM-training code paths were skimmed, not audited. The audit finds *places where this can happen*; it does not prove every
output is honest (that needs the golden-site and back-test work).
