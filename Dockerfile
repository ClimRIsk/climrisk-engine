# ClimRisk Engine — Hugging Face Spaces (Docker SDK, CPU Basic / free)
#
# Mirrors the exact install/start commands already proven working on Render:
#   build:  pip install "./climate_risk_engine[api,gis]"
#   start:  uvicorn cri.api.main:app --host 0.0.0.0 --port $PORT
# only the port changes, since HF Spaces requires the app to listen on 7860.
#
# [gis] extras add rasterio for JRC Global Flood Hazard Map COG reads.
# GDAL system libraries are required for rasterio to build/run.

FROM python:3.11-slim

WORKDIR /app

# System deps: build tools + GDAL for rasterio (JRC hydraulic flood depth)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gdal-bin \
    libgdal-dev \
    python3-gdal \
    && rm -rf /var/lib/apt/lists/*

# Pin GDAL Python bindings to match the system GDAL version
RUN GDAL_VERSION=$(gdal-config --version) && \
    pip install --no-cache-dir "GDAL==${GDAL_VERSION}"

COPY climate_risk_engine/ /app/climate_risk_engine/

RUN pip install --no-cache-dir "./climate_risk_engine[api,gis]"

ENV CRI_CACHE_DIR=/tmp/cri_cache

EXPOSE 7860

CMD ["uvicorn", "cri.api.main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
