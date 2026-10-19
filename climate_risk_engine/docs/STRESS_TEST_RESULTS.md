# Engine stress test — results (19 Oct 2026)

What was run, what broke, what was fixed, and what was **not** tested. Screening-grade engine; nothing here is a validation of
the physics against observations (see `LANDSLIDE_VALIDATION.md`, `RAG_RECCE.md` for those).

## Campaign

| Layer | Tool | Scope | Result |
|---|---|---|---|
| L1 API fuzz, offline | `tools/stress_api.py` | 163 routes, 2,369 requests: schema-derived baseline + missing fields, wrong types, extreme numbers (±1e308, ±90/180 limits), 20 kB strings, script/SQL/path/NUL/emoji strings, nested junk; network disabled | before fixes: 29 HTTP-500s on 6 routes. After: **0 crashes, 0 traceback leaks, 0 NaN/Infinity bodies, 0 hangs**; 34 clean 502/503 "dependency unavailable" |
| L1 API fuzz, online | same, `--mode online` | 102 routes, 561 requests | 0 crashes (one route is slow: `/climate-scenario/compose` 30–40 s per call) |
| L2 Property tests | `tests/test_stress_properties.py` (hypothesis) | SCS runoff bounds + monotonicity, wind/EVT return levels monotone in return period, Bruun retreat, UTCI finite-or-NaN, flood solver mass conservation and non-negative depth on random terrain, insurance run deterministic / finite / linear in value / deductible ladder non-increasing | 10 properties, all pass (no counter-example found) |
| L3 Global-site sweep | `tools/stress_sites.py` | 39 awkward coordinates (poles, antimeridian, open ocean, Dead Sea, Death Valley, Rotterdam, atolls, ice sheets, Mariana Trench…) through location-hazard, water-risk and the physical-risk report | found 5 real defects (below); water-risk handled ocean/poles honestly |
| Concurrency / determinism | `tools/stress_load.py` | 32 parallel requests over 4 routes | 0 errors; identical requests gave identical bodies |
| Regression suite | pytest, all 69 files | after the fixes | all pass (one test, `test_hyper_arid_site_gets_no_fire_model`, failed once in a long combined run and passed alone and in every subset — it makes real network calls it should not; flaky, not yet made hermetic) |

## Defects found and fixed (each has a test in `tests/test_stress_regressions.py`)

1. **Out-of-memory kill at high latitude.** A site at 89.9 °N asked for ~100,000 terrain tiles and the process was killed. `get_elevation_grid` now refuses |lat| beyond Web-Mercator coverage (85.05°, including the search radius), non-finite coordinates, and any grid over 256 tiles, with a clear message.
2. **Terrain outage cost >60 s per site.** After the parallel prefetch failed, the stitch loop re-fetched every failed tile one at a time with retry sleeps. It now reuses the prefetch result (20 s end-to-end offline, from more than 60 s).
3. **Open ocean rated as a hazardous site.** The terrain model includes the sea floor, so a pin in the Pacific came back as "a valley 4,992 m below sea level" with CRITICAL surface-water flooding, and (0,0) as CRITICAL; poles came back "overall NEGLIGIBLE". The physical-risk report now refuses a site with no administrative unit in the boundary layer (open water, polar caps) with a 422 saying why. Every real island nation tested (Tuvalu, Kiribati, Maldives, Marshall Is., Nauru, Tonga, Seychelles…) is covered and still assessed.
4. **Unavailable data presented as a neutral reading.** When the external Climate Monitor was unreachable, `/analyze/location-hazard` returned `hazard_score 40, "Moderate", anomalies 0.0`. It now returns nulls, `available:false`, and no score (the AI-chat tool that uses the same client gets the same honest null).
5. **Unknown sector silently scored as the lowest-risk sector.** `sector_hint:"other"` (or any unrecognised text) became "technology". The assessment now adds a prominent `SECTOR NOT RECOGNISED … scored with the technology profile, the model's lowest-risk sector` data-gap. Separately, a default revenue of USD 5,000 m was labelled `"user-supplied"` when the user supplied nothing; it is now `assumed (placeholder default, not company data)`.
6. **Bare 500s on malformed free-form fields.** Dict/list fields the schema cannot validate (asset-twin, EU taxonomy overrides, hazard3d polygons) raised KeyError/TypeError and returned a plain 500. The API now maps those to 422 `unprocessable_request` and network/dependency failures to 503 `dependency_unavailable`, both with a short message and no traceback.

## Known limitations found but **not** fixed (stated, not hidden)

- **Low coasts and atolls are mis-modelled.** The 30 m terrain model and 90 m coastal-flood maps read Funafuti (Tuvalu) at 6 m, Malé at 7 m, Tarawa/Kiritimati region at −4 m with a "HIGH slope" artefact, and rate coastal flood NEGLIGIBLE for Tuvalu and Malé. The report now carries a caution on every site under 10 m near the sea (and under 5 m with a HIGH slope rating) that the rating is unconfirmed. The underlying data cannot resolve these places; surveyed levels are needed. Rotterdam reads +6 m (actual ground is below sea level behind dikes); the existing dike caution applies.
- **Pins on water** (e.g. Dead Sea) are still rated as if they were land; a caution is now added when land cover says water.
- **Latency**: `/climate-scenario/compose` takes 30–40 s per call (downloads a projection file each time); a full physical-risk report is 20–40 s per site online.
- **Latitude/longitude on `/climate-scenario/compose` is not range-validated** (1e308 yields a 502 instead of a 422).
- **Dateline**: sites whose search box crosses ±180° are clamped at the edge rather than wrapped (sites right on the line are now refused as open water, but an island near ±180° is untested).
- Online sweep covered 102 of ~180 routes (heavy network routes — reports, asset, analyst, resident, hazard3d — were covered by the site sweep and the unit tests instead); `/runs/file` (multipart) and the LLM / publish / delete / licence routes were deliberately not fuzzed.
- Not tested: Windows build, Google photoreal layer (needs the user's key), true multi-user load, long-run memory growth, adversarial security testing (this was robustness testing, not a penetration test).

## Reproduce

```
python tools/stress_api.py --mode offline --resume --skip hazard3d/study-area,reports,runs/file --out out.jsonl   # repeat until "done 0 routes"
python tools/stress_sites.py --route location-hazard|water-risk|asset|report --out sites.jsonl
python tools/stress_load.py 32
pytest tests/test_stress_properties.py tests/test_stress_regressions.py
```

## Follow-up fixes (19 Oct 2026, after the golden-site run)
- Heat zone now reads real ERA5 daily maxima (`finance/heat_hazard.py`); golden heat sites 6/6.
- Coastal flood: an elevation screen stops a dry 90 m surge map on very low ground near the sea from reading as safe (golden coastal 8/10; Jakarta and Mumbai remain).
- River flood: not fixed (no usable free higher-resolution global map); the report now warns when water is within 300 m of a site the map shows as dry.
- `/climate-scenario/compose`: coordinates are range-checked (1e308 returned 502) and results are cached per place, pathway and year (a cold call takes 30-40 s).
- Terrain tiles now wrap across the antimeridian instead of clamping.
- `assess_physical_risk` accepts `asset_lat`/`asset_lon`; without them it states that the sea-level-rise regional multiplier is the global default.
- Insurer dashboard tags sites with incomplete cover as "partial".
- `test_hyper_arid_site_gets_no_fire_model` made hermetic (network blocked).
- Open: the model lab's fire-weather index gives implausible values (Pasadena 230 days a year at FWI ≥ 30, London 9), so it was not wired into the wildfire zone; FEMA NRI benchmark script exists but needs the county CSV, which this environment could not download; flat 5 % flood damage in the legacy `hazard_financial` model is unchanged.
