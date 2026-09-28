"""
ml/prithvi_flood.py — Client for IBM/NASA Prithvi-EO-2.0 flood segmentation.

Model  : ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11
         300M-parameter geospatial foundation model fine-tuned on Sen1Floods11.
Input  : 6 Sentinel-2 bands — BLUE, GREEN, RED, NIR_NARROW (B8A), SWIR_1 (B11),
         SWIR_2 (B12) — as surface reflectance in 0–1, shape (6, H, W).
Output : per-pixel class, 0 = no water, 1 = water / flood.

The model needs terratorch + torch, which is too heavy for the API server, so
it runs on a dedicated Hugging Face Inference Endpoint (GPU or CPU) using the
handler in deploy/hf_prithvi_flood/. The engine calls it over HTTPS.

Environment
───────────
  CLIMRISK_PRITHVI_ENDPOINT   https URL of the deployed endpoint (unset → disabled)
  HF_TOKEN                    token with access to that endpoint

Wire format (JSON)
──────────────────
  request : {"inputs": {"array_npy_b64": <base64 of np.save(float32 (6,H,W))>}}
  response: {"mask_npy_b64": <base64 of np.save(uint8 (H,W))>, "model": "..."}

When the endpoint is not configured or fails, callers fall back to the
spectral MNDWI water index — the assessment never blocks on the model.
"""
from __future__ import annotations

import base64
import io
import json
import logging
import os
import urllib.request
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

PRITHVI_BANDS = ["blue", "green", "red", "nir08", "swir16", "swir22"]  # Earth Search asset keys
MODEL_ID = "ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11"


def endpoint_configured() -> bool:
    return bool(os.environ.get("CLIMRISK_PRITHVI_ENDPOINT"))


def _npy_b64(arr: np.ndarray) -> str:
    buf = io.BytesIO()
    np.save(buf, arr, allow_pickle=False)
    return base64.b64encode(buf.getvalue()).decode()


def _from_npy_b64(s: str) -> np.ndarray:
    return np.load(io.BytesIO(base64.b64decode(s)), allow_pickle=False)


def segment_water(chip: np.ndarray, timeout_s: int = 90) -> Optional[np.ndarray]:
    """
    Run Prithvi flood segmentation on a (6, H, W) reflectance chip.
    Returns a uint8 (H, W) mask (1 = water) or None if unavailable.
    """
    url = os.environ.get("CLIMRISK_PRITHVI_ENDPOINT")
    if not url:
        return None
    if chip.ndim != 3 or chip.shape[0] != 6:
        raise ValueError(f"expected (6, H, W) chip, got {chip.shape}")

    chip = np.nan_to_num(chip.astype(np.float32), nan=0.0)
    body = json.dumps({"inputs": {"array_npy_b64": _npy_b64(chip)}}).encode()
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    token = os.environ.get("HF_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"

    try:
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=timeout_s) as r:
            resp = json.loads(r.read())
        mask = _from_npy_b64(resp["mask_npy_b64"]).astype(np.uint8)
        if mask.shape != chip.shape[1:]:
            log.warning("Prithvi mask shape %s != chip %s", mask.shape, chip.shape[1:])
            return None
        return mask
    except Exception as exc:
        log.warning("Prithvi endpoint call failed: %s", exc)
        return None
