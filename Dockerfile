# ClimRisk Engine — Render / Hugging Face Spaces deployment
#
# Base: ghcr.io/osgeo/gdal (Ubuntu 22.04 + GDAL 3.8 + Python 3.10 pre-installed)
# This image is purpose-built for geospatial workloads and includes all system
# libraries rasterio needs — no manual GDAL install, no manylinux wheel conflicts.
#
# rasterio, shapely, pyproj are available as conda packages on this image,
# but we install via pip to stay aligned with pyproject.toml version pins.
#
# Port: HF Spaces requires 7860. Render uses $PORT (default 10000).
# Set PORT=7860 in Render env vars if deploying there too.

FROM ghcr.io/osgeo/gdal:ubuntu-small-3.8.5

WORKDIR /app

# Install pip + build tools (OSGeo image has Python but not pip by default)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY climate_risk_engine/ /app/climate_risk_engine/

# Install core API deps
RUN pip3 install --no-cache-dir "./climate_risk_engine[api]"

# Install rasterio — works cleanly against the system GDAL already on this image
RUN pip3 install --no-cache-dir "rasterio>=1.3"

ENV CRI_CACHE_DIR=/tmp/cri_cache

EXPOSE 7860

CMD ["uvicorn", "cri.api.main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
