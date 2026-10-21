---
title: ClimRisk Engine
emoji: 🌍
colorFrom: yellow
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
---

# ClimRisk: climate risk to finance, with the evidence shown

An open-for-evaluation engine that turns a location or a company into physical climate-risk findings and financial numbers, plus a local AI agent that can run it and explain the result with citations. Built to be checked: every hazard module has a written benchmark, and the places where it does not beat a simple baseline are published next to the places where it does.

> Screening grade. Not an underwriting model. Claims here are limited to what the benchmarks in `climate_risk_engine/docs/` support.

## What it does

- **Physical hazards for a site** (`cri.physical_report`): river flood, coastal flood, surface-water flood, tropical cyclone, landslide, wildfire, heat, earthquake, water stress. Each zone returns a rating, the facts behind it and its source; a layer that could not be read says "Not assessed", never "low".
- **Open LiDAR terrain** where it exists (England 1 m, Netherlands 0.5 m, US 3DEP 1 m), global 10-30 m tiles elsewhere; local-inertial shallow-water solver for rain-on-grid flooding; HAND floodplain screen; Rothermel surface-fire behaviour; USGS seismic hazard.
- **Insurer and finance layers**: correlated annual loss simulation with policy terms, exceedance curves, transition and carbon scenarios, covenants, LCA.
- **3D hazard viewer**: real satellite or aerial imagery draped on a LiDAR terrain mesh with extruded buildings (heights measured from the LiDAR surface model in England and the Netherlands) and hazard layers on top. No API key. See `docs/3D_HAZARD_VIEWER.md`.

## The AI parts

| Component | Where | What it is |
|---|---|---|
| Resident agent | `src/cri/resident/` | Runs inside the engine with no cloud key: intent model trained in the repo (`nlu.py`), plans a chain of engine tools, remembers (SQLite), answers free-form questions with citations from a case library of recorded climate events and a public-source knowledge index (offline dense retrieval in `semantic.py`), drafts research and approval-gated posts. |
| LLM orchestrator | `src/cri/ai/` | Tool registry and orchestrator for an optional LLM, and a narrator that turns engine data into readable text. Facts and numbers come from the engine and retrieved sources, not from the model. |
| Fine-tuning pipeline | `src/cri/llmtrain/` | Builds the fine-tuning set for a local model from open sources with each source's licence and whether it may be used for training or only retrieval, then scores a model on held-out data for citation validity, refusal and number faithfulness. The model is taught how to answer; it is not asked to remember facts. |
| Local model | not in this repo | `climrisk-llm-v8`: Llama-3.2-3B-Instruct base, fine-tuned, quantised Q4_K_M, about 2 GB. Weights are distributed separately (the Llama 3.2 community licence applies to them). |
| Assessment job runner | `src/cri/agent/` | Async job layer: company name in, entity resolution, data acquisition, risk assessment, trajectory out, with polling. |
| ML layer | `src/cri/ml/` | Disclosure scanner, risk trajectory predictor, emissions estimator; each degrades gracefully when its optional ML packages are missing. |

`src/cri/maintainer/` holds private maintainer tooling used on the author's own machine; it is excluded from the built package.

## Evidence, including the unflattering parts

Start with `climate_risk_engine/docs/VALIDATION_LEDGER.md` (graded claims) and `STRESS_TEST_RESULTS.md`. Examples:

- **Building-level hindcast** (`docs/HINDCAST_STUDY.md`): on Carlisle 2005 and 2015, Cockermouth 2009 and York 2015, LiDAR floodplain position ranks the buildings that flooded with ROC area 0.81-0.92 against 0.50-0.87 for the same method on global terrain. Depth and loss are not tested.
- **A model that lost to a simple rule** (`docs/RIVER_DEPTH_BENCHMARK.md`): HAND plus a rating curve scored a critical success index of 0.37 against 0.76 for a plain height band, so it was not shipped.
- **Surface-water model** (`docs/PLUVIAL_BENCHMARK.md`): modest skill (0.64-0.71), can only raise a rating.
- **Wildfire**: spread model matches published values but does not beat hot-day temperature at picking large-fire days (`docs/FIRE_BEHAVIOUR_BENCHMARK.md`).
- 776 automated tests, golden-site regression set, a silent-degradation audit.

## Run it

```bash
pip install "./climate_risk_engine[api]"
uvicorn cri.api.main:app --port 8000        # interactive docs at /docs, health at /health
cd climate_risk_engine && PYTHONPATH=.:src python -m pytest -q
```

Needs network for live data (ERA5, elevation tiles, LiDAR services, Sentinel-2); results are cached under `~/.cache/climrisk`. Python 3.10+. Docker: see `Dockerfile` (also runs as a Hugging Face Space on port 7860).

## Known limits

Hazard frequencies are historical with no climate uplift. Flood depth-damage and the insurer correlation lengths are not calibrated to claims. No external reviewer has audited the model. England is the only place with a building-level flood test. See `docs/VALIDATION_DOSSIER.md` section 4 and `GAP_REGISTER.md`.

## Licence

PolyForm Noncommercial 1.0.0 (see `LICENSE`): free to read, run, modify and evaluate for non-commercial purposes; commercial use needs written permission. Data sources keep their own licences (EA OGL v3, Copernicus, PDOK, USGS, OpenStreetMap ODbL and others).

## Deployment note

This repository also builds the Hugging Face Space: `pip install "./climate_risk_engine[api]"`, port 7860. No secrets or external database are required.
