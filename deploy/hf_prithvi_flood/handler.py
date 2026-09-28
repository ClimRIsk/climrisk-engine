"""
Hugging Face Inference Endpoint handler — Prithvi-EO-2.0 flood segmentation.

Deploy
──────
1. Duplicate ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11 into your
   own HF org (it ships config.yaml, the .pt checkpoint and inference.py).
2. Add this handler.py and the requirements.txt from this folder to that repo.
3. Create an Inference Endpoint from the repo (Custom task). A small GPU
   (e.g. T4) keeps latency ~1–3 s per 512×512 chip; CPU works but is slower.
4. Set on Render:  CLIMRISK_PRITHVI_ENDPOINT=<endpoint URL>  HF_TOKEN=<token>

The pre/post-processing reuses the model repo's own inference.run_model, so
normalisation and 512-px windowing match exactly what the model was trained on.

Wire format — see climate_risk_engine/src/cri/ml/prithvi_flood.py

NOTE: this file runs inside the endpoint container, not the API server. It has
not been executed in the ClimRisk CI (no torch/terratorch there) — smoke-test
it once after deployment with the repo's examples/*.tif.
"""
import base64
import io
import os
import sys

import numpy as np


class EndpointHandler:
    def __init__(self, path: str = ""):
        sys.path.insert(0, path)
        from inference import run_model  # from the Prithvi model repo
        from terratorch.cli_tools import LightningInferenceModel

        self._run_model = run_model
        config = os.path.join(path, "config.yaml")
        ckpt = os.path.join(path, "Prithvi-EO-V2-300M-TL-Sen1Floods11.pt")
        self.lm = LightningInferenceModel.from_config(config, ckpt)
        self.lm.model.eval()
        self.img_size = 512

    def __call__(self, data: dict) -> dict:
        payload = data.get("inputs", data)
        arr = np.load(io.BytesIO(base64.b64decode(payload["array_npy_b64"])), allow_pickle=False)
        arr = arr.astype(np.float32)
        if arr.ndim != 3 or arr.shape[0] != 6:
            return {"error": f"expected (6, H, W), got {list(arr.shape)}"}
        if arr.mean() > 1:           # accept 0–10000 DN too
            arr = arr / 10000.0

        x = arr[np.newaxis, :, np.newaxis, :, :]   # (B=1, C=6, T=1, H, W)
        pred = self._run_model(x, None, None, self.lm.model, self.lm.datamodule, self.img_size)
        mask = np.asarray(pred).squeeze().astype(np.uint8)

        buf = io.BytesIO()
        np.save(buf, mask, allow_pickle=False)
        return {
            "mask_npy_b64": base64.b64encode(buf.getvalue()).decode(),
            "shape": list(mask.shape),
            "classes": {"0": "no_water", "1": "water"},
            "model": "ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11",
        }
