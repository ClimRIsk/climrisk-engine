---
title: ClimRisk Engine
emoji: 🌍
colorFrom: yellow
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
---

# ClimRisk Engine API

FastAPI backend for the ClimRisk Climate Risk Intelligence (CRI) engine —
scenario-driven climate-to-finance translation. Same `cri` package and
`/health`-checked service previously deployed on Render.com; this Space
mirrors that exact build (`pip install "./climate_risk_engine[api]"`) and
start command, just listening on port 7860 as Hugging Face Spaces requires.

- Interactive API docs: `/docs`
- Health check: `/health`

No secrets or external database are required — the engine runs
self-contained off its own seed data and public climate data connectors.
