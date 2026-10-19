"""
cri/resident/semantic.py — offline dense retrieval for the case library and the knowledge index.

Passages are embedded once with BAAI/bge-small-en-v1.5 (384 dimensions, ONNX, ~65 MB, run on the CPU through fastembed) and the vectors are shipped next to the
index (data/case_embeddings.npy). At question time only the question is embedded; scores are cosine similarity, blended with BM25 in casebase.retrieve().
Nothing leaves the machine. If the vectors, the model or fastembed are missing, every function returns None and BM25 carries on alone — the app never fails because of this layer.

  Dense.load(n_items)        -> Dense | None   (vectors must match the index row count)
  Dense.scores(question)     -> numpy array of cosine similarities, one per passage, or None
  embed_texts(texts)         -> float32 matrix (used to build or extend an index; the model is the same, passages carry no prefix)
  model_dir()                where the embedding model is looked up (CRI_EMBED_DIR, ~/.climrisk/models/embed, or data/embed-model next to this file)
"""
from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Optional

MODEL = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
_HERE = Path(__file__).parent
_LOCK = threading.Lock()
_MODEL = {"m": None, "tried": False}


def model_dir() -> Path:
    env = os.environ.get("CRI_EMBED_DIR")
    if env:
        return Path(env)
    shipped = _HERE / "data" / "embed-model"
    if shipped.exists():
        return shipped
    return Path(os.environ.get("CRI_HOME") or (Path.home() / ".climrisk")) / "models" / "embed"


def _model():
    with _LOCK:
        if _MODEL["tried"]:
            return _MODEL["m"]
        _MODEL["tried"] = True
        if os.environ.get("CRI_RAG_SEMANTIC", "1") == "0":
            return None
        try:
            from fastembed import TextEmbedding
            _MODEL["m"] = TextEmbedding(MODEL, cache_dir=str(model_dir()), local_files_only=not os.environ.get("CRI_EMBED_ALLOW_DOWNLOAD"))
        except Exception:                                            # noqa: BLE001 - fail soft: BM25 only
            _MODEL["m"] = None
        return _MODEL["m"]


def embed_texts(texts: list, query: bool = False):
    import numpy as np
    m = _model()
    if m is None:
        return None
    pre = QUERY_PREFIX if query else ""
    return np.array(list(m.embed([pre + t[:1200] for t in texts], batch_size=64)), dtype=np.float32)


class Dense:
    def __init__(self, mat):
        self.mat = mat

    @classmethod
    def load(cls, n_items: int, path: Optional[Path] = None) -> Optional["Dense"]:
        try:
            import numpy as np
            p = path or (_HERE / "data" / "case_embeddings.npy")
            if not p.exists():
                return None
            mat = np.load(p).astype(np.float32)
            if mat.shape[0] != n_items:                              # stale vectors for a different index: never mix them
                return None
            n = np.linalg.norm(mat, axis=1, keepdims=True)
            n[n == 0] = 1.0
            return cls(mat / n)
        except Exception:                                            # noqa: BLE001
            return None

    def scores(self, q: str):
        import numpy as np
        v = embed_texts([q], query=True)
        if v is None:
            return None
        v = v[0]
        n = float(np.linalg.norm(v))
        return None if n == 0 else self.mat @ (v / n)
