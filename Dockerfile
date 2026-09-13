# ClimRisk Engine — Hugging Face Spaces (Docker SDK, CPU Basic / free)
#
# Mirrors the exact install/start commands already proven working on Render:
#   build:  pip install "./climate_risk_engine[api]"
#   start:  uvicorn cri.api.main:app --host 0.0.0.0 --port $PORT
# only the port changes, since HF Spaces requires the app to listen on 7860.
#
# JRC hydraulic flood depth (flood_depth.py) requires rasterio, installed here
# as a separate step AFTER the main package so a rasterio wheel failure is
# isolated and doesn't prevent the core API from starting.
# If rasterio is unavailable at runtime, the engine falls back to phi×ELF
# automatically (see flood_depth.py: _RASTERIO_AVAILABLE guard).

FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY climate_risk_engine/ /app/climate_risk_engine/

RUN pip install --no-cache-dir "./climate_risk_engine[api]"

# rasterio: binary manylinux wheel — bundles GDAL, no system GDAL needed.
# Installed separately so a wheel failure doesn't break the core API build.
RUN pip install --no-cache-dir "rasterio>=1.3" || \
    echo "WARNING: rasterio install failed — JRC flood depth disabled, phi×ELF fallback active"

ENV CRI_CACHE_DIR=/tmp/cri_cache

EXPOSE 7860

CMD ["uvicorn", "cri.api.main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
