"""
cri/resident/nlu.py — the resident agent's language understanding, trained inside the engine.

Model     TF-IDF over word 1–2-grams and character 3–5-grams (hashed), multinomial logistic regression trained by Adam in
          numpy. No external ML library and no network. Trains on the corpus in seconds.
Honesty   Reports accuracy on HELD-OUT TEMPLATES (phrasings never seen in training) and a grouped cross-validation score.
          Predictions below a confidence floor return "unknown" and the agent asks instead of guessing.
Learning  Corrections and confirmations from the user are stored (resident.memory) and added to the training set on retrain;
          the retrain report compares held-out accuracy before and after.
Entities  Coordinates, return period, year, sector, company (matched against the engine's asset register), numbers with units.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Optional

import numpy as np

NF = 1 << 14
CONF_FLOOR = 0.30      # lowered from 0.35 on 29 Oct 2026: on 306 independent utterances 0.30 gave the same number of wrong picks as 0.35 and 3 more correct ones; below it the agent asks
FIT = dict(lr=0.5, epochs=300, l2=1e-5)

_WORD = re.compile(r"[a-z0-9\.\-']+")


def _norm(t: str) -> str:
    t = t.lower().strip()
    t = re.sub(r"(-?\d+(?:\.\d+)?)\s*[,;]\s*(-?\d+(?:\.\d+)?)", " <coord> ", t)
    t = re.sub(r"\b1[\s\-]*in[\s\-]*\d+\b|\b\d+[\s\-]*(?:year|yr)s?\b", " <rp> ", t)
    t = re.sub(r"\b(19|20)\d\d\b", " <year> ", t)
    t = re.sub(r"\b\d+(?:\.\d+)?\s*(?:m/s|metres?|meters?|m\b|percent|%)", " <num> ", t)
    return t


def _feats(t: str) -> list:
    t = _norm(t)
    w = _WORD.findall(t)
    f = ["w:" + x for x in w] + ["b:" + a + "_" + b for a, b in zip(w, w[1:])]
    s = " " + " ".join(w) + " "
    for n in (3, 4, 5):
        f += ["c:" + s[i:i + n] for i in range(len(s) - n + 1)]
    return f


def _h(s: str) -> int:
    h = 2166136261
    for ch in s.encode():
        h = ((h ^ ch) * 16777619) & 0xFFFFFFFF
    return h % NF


def _vec(texts: list, idf: Optional[np.ndarray] = None) -> tuple:
    X = np.zeros((len(texts), NF), dtype=np.float32)
    for i, t in enumerate(texts):
        for f in _feats(t):
            X[i, _h(f)] += 1.0
    X = np.log1p(X)
    return X, idf


class Model:
    def __init__(self):
        self.classes: list = []
        self.W = None
        self.b = None
        self.idf = None
        self.meta: dict = {}

    def _tfidf(self, X: np.ndarray) -> np.ndarray:
        Z = X * self.idf[None, :]
        n = np.linalg.norm(Z, axis=1, keepdims=True)
        return Z / np.maximum(n, 1e-9)

    def fit(self, rows: list, epochs: int = 300, lr: float = 0.5, l2: float = 1e-5, seed: int = 0) -> "Model":
        texts = [r[0] for r in rows]
        self.classes = sorted({r[1] for r in rows})
        ci = {c: i for i, c in enumerate(self.classes)}
        y = np.array([ci[r[1]] for r in rows])
        X, _ = _vec(texts)
        df = (X > 0).sum(axis=0)
        self.idf = (np.log((1 + len(texts)) / (1 + df)) + 1.0).astype(np.float32)
        Z = self._tfidf(X)
        K = len(self.classes)
        rng = np.random.default_rng(seed)
        W = np.zeros((NF, K), np.float32); b = np.zeros(K, np.float32)
        mb, vb = np.zeros_like(b), np.zeros_like(b)
        Y = np.eye(K, dtype=np.float32)[y]
        # restrict updates to columns that occur in training (sparse in practice)
        cols = np.where((Z != 0).any(axis=0))[0]
        Zc = Z[:, cols]; Wc = W[cols]
        mWc, vWc = np.zeros_like(Wc), np.zeros_like(Wc)
        for ep in range(1, epochs + 1):
            S = Zc @ Wc + b
            S -= S.max(axis=1, keepdims=True)
            P = np.exp(S); P /= P.sum(axis=1, keepdims=True)
            G = (P - Y) / len(texts)
            gW = Zc.T @ G + l2 * Wc; gb = G.sum(axis=0)
            mWc = 0.9 * mWc + 0.1 * gW; vWc = 0.999 * vWc + 0.001 * gW * gW
            mb = 0.9 * mb + 0.1 * gb; vb = 0.999 * vb + 0.001 * gb * gb
            c1, c2 = 1 - 0.9 ** ep, 1 - 0.999 ** ep
            Wc -= lr * (mWc / c1) / (np.sqrt(vWc / c2) + 1e-8)
            b -= lr * (mb / c1) / (np.sqrt(vb / c2) + 1e-8)
        W[cols] = Wc
        self.W, self.b = W, b
        self.meta = {"n_train": len(texts), "n_classes": K, "features": NF, "epochs": epochs}
        return self

    def proba(self, text: str) -> np.ndarray:
        X, _ = _vec([text])
        Z = self._tfidf(X)
        S = Z @ self.W + self.b
        S -= S.max()
        P = np.exp(S[0]); return P / P.sum()

    def predict(self, text: str, hybrid: bool = True) -> dict:
        P = self.proba(text)
        if hybrid:
            P = P * np.array([1.0 + CUES_W * _cue(text).get(c, 0.0) for c in self.classes]); P = P / P.sum()
        if "offtopic" in self.classes and DOMAIN_WORDS.search(text):
            # Climate, hazard, emissions, asset or shipping vocabulary is present: never refuse as "outside what I do" (a wrong refusal of real work is worse than a question).
            # The off-topic class is removed and the rest renormalised, so the best remaining intent still has to clear the confidence floor.
            P = P.copy(); P[self.classes.index("offtopic")] = 0.0; P = P / P.sum()
        o = np.argsort(-P)
        top = [{"intent": self.classes[i], "p": round(float(P[i]), 3)} for i in o[:3]]
        off = top[0]["intent"] == "offtopic"
        out = {"intent": "unknown" if (off or top[0]["p"] < CONF_FLOOR) else top[0]["intent"], "confidence": top[0]["p"], "top": [t for t in top if t["intent"] != "offtopic"] or top}
        if off and top[0]["p"] >= 0.5:
            out["offtopic"] = True                                       # the model is fairly sure this is outside what the agent does; never offer climate tools as guesses
        return out

    # persistence (JSON, sparse): only non-zero weight rows are stored
    def to_json(self) -> dict:
        nz = np.where(np.abs(self.W).max(axis=1) > 0.05)[0]
        return {"classes": self.classes, "b": self.b.round(5).tolist(), "rows": nz.tolist(),
                "W": self.W[nz].round(3).tolist(), "idf": self.idf.round(3).tolist(), "meta": self.meta}

    @classmethod
    def from_json(cls, d: dict) -> "Model":
        m = cls(); m.classes = d["classes"]; K = len(m.classes)
        m.b = np.array(d["b"], np.float32); m.W = np.zeros((NF, K), np.float32); m.idf = np.ones(NF, np.float32)
        rows = np.array(d["rows"], int)
        m.W[rows] = np.array(d["W"], np.float32); m.idf = np.array(d["idf"], np.float32)
        m.meta = d.get("meta", {})
        return m


def evaluate(rows_train: list, rows_test: list, **kw) -> dict:
    m = Model().fit(rows_train, **kw)
    ok, conf = 0, []
    errs = []
    for u, i in rows_test:
        p = m.predict(u)
        hit = p["intent"] == i
        ok += hit; conf.append(p["confidence"])
        if not hit and len(errs) < 12:
            errs.append({"text": u, "true": i, "pred": p["intent"], "p": p["confidence"]})
    return {"n_test": len(rows_test), "accuracy": round(ok / max(len(rows_test), 1), 4), "mean_confidence": round(float(np.mean(conf)), 3), "errors": errs}


def grouped_cv(rows: list, k: int = 5, **kw) -> dict:
    """k-fold by template index: each fold holds out whole templates."""
    from .corpus import utterances
    base = utterances()
    accs = []
    for fold in range(k):
        tr = [(u, i) for u, i, ti in base if ti % k != fold]
        te = [(u, i) for u, i, ti in base if ti % k == fold]
        accs.append(evaluate(tr, te, **kw)["accuracy"])
    return {"folds": k, "accuracies": accs, "mean": round(float(np.mean(accs)), 4), "min": round(float(min(accs)), 4)}


# explicit-term cues: deterministic priors that sharpen the learned model on unambiguous vocabulary
CUES_W = 3.0
_CUES = [
    (r"\bcbam\b|border (carbon|tax|adjustment)", "report_cbam"), (r"iso[\s\-]*14067|\bcfp\b", "report_iso14067"),
    (r"\blca\b|life[\s\-]*cycle", "report_lca"), (r"carbon audit|ghg (audit|inventory)|carbon accounting|scope [123]", "report_carbon"),
    (r"transition (risk|report)|stranded|carbon price", "report_transition"), (r"financial (risk|climate|impact)|ebitda|value at risk|loss distribution", "report_financial"),
    (r"physical (risk|report|hazard)", "report_physical"), (r"linkedin|\bpost\b", "draft_post"), (r"\b(3d|three[\s\-]*dimension\w*|twin|viewer)\b", "twin"),
    (r"protect|defen[cs]e|harden|mitigat|retrofit|barrier|flood wall|resilien|adaptation|how high", "protect"), (r"scenario|what[\s\-]*if|stress test|compare .*cases|different futures", "scenario"),
    (r"\bremember\b|note (this|that)|keep in mind|for future reference", "remember"), (r"what do you (remember|know about)|recall|what is in your memory", "recall"),
    (r"\b(plants?|assets?|sites?|facilities|locations?)\b.*\b(of|owned|own|operate|belong)|where (are|does)", "assets"),
    (r"research|scan|this week|live (hazard|event)|gdacs|news|happening", "research"), (r"\bwhy\b|explain|how (reliable|accurate|was)|assumption|limits|mean", "explain"),
    (r"\b(history|previous|so far|recent (runs|activity|work)|did you do)\b", "history"), (r"what can you|\bhelp\b|capabilit|how (do i use|can you help)", "help"),
    (r"^\s*(hi|hey|hello|thanks|thank you|ok|cool|nice|bye|goodbye|great|awesome|perfect)\b", "smalltalk"),
    (r"\b(ship|vessel|tanker|bulker|container ship|voyage|bunkers?|knots?|nautical)\b", "voyage"),
    (r"\b(build|construct\w*|erect|put up|develop)\b.*\b(storey|stories|floors?|tower|warehouse|factory|hotel|apartments?|hospital|school|data ?cent\w+|mall|car ?park)\b", "feasibility"),
    (r"\b(publish|post it|go ahead and post|approve)\b", "publish_post"), (r"risk analy|assess\b|climate risk for|how risky|run the numbers", "risk_analysis"),
]


def _cue(text: str) -> dict:
    low = text.lower(); out: dict = {}
    for pat, intent in _CUES:
        if re.search(pat, low):
            out[intent] = out.get(intent, 0.0) + 1.0
    return out


# ── entities ─────────────────────────────────────────────────────────────────

_NUM = r"-?\d{1,3}(?:\.\d+)?"
_COORD = re.compile(rf"(?<![\d.])({_NUM})\s*°?\s*([NSns])?\s*[,;/ ]\s*({_NUM})\s*°?\s*([EWew])?(?![\d.])")
_SECT = {"cement": "cement", "steel": "iron-and-steel", "power": "electricity-generation", "power station": "electricity-generation",
         "refinery": "oil-and-gas-refining", "oil": "oil-and-gas-refining", "gas": "oil-and-gas-refining", "chemical": "chemicals", "chemicals": "chemicals",
         "mining": "mining", "mine": "mining", "aluminium": "aluminum", "aluminum": "aluminum", "pulp": "pulp-and-paper", "paper": "pulp-and-paper",
         "data centre": "data centre", "data center": "data centre", "port": "port", "logistics": "port", "real estate": "real estate", "building": "real estate",
         "agriculture": "agriculture", "food": "agriculture"}


def entities(text: str) -> dict:
    out: dict = {}
    t = text
    for m in _COORD.finditer(t):
        a, ns, b, ew = m.groups()
        lat, lon = float(a), float(b)
        if ns and ns.lower() == "s": lat = -abs(lat)
        if ew and ew.lower() == "w": lon = -abs(lon)
        if not ("." in a or "." in b or ns or ew):
            continue                      # two whole numbers with no hemisphere letter are far more often "5,000 m2" or "20,000 sq ft" than a coordinate
        if -90 <= lat <= 90 and -180 <= lon <= 180 and ("." in a or "." in b or ns or ew or "," in m.group(0)):
            out["lat"], out["lon"] = lat, lon
            break
    m = re.search(r"\b1[\s\-]*in[\s\-]*(\d{2,4})\b|\b(\d{2,4})[\s\-]*(?:year|yr)\b", t.lower())
    if m:
        out["rp"] = int(m.group(1) or m.group(2))
    m = re.search(r"\b(20[2-9]\d)\b", t)
    if m:
        out["year"] = int(m.group(1))
    low = t.lower()
    for k in sorted(_SECT, key=len, reverse=True):
        if re.search(rf"\b{re.escape(k)}\b", low):
            out["sector"] = _SECT[k]; break
    for k, v in (("flood", "flood"), ("storm surge", "flood"), ("sea level", "flood"), ("cyclone", "wind"), ("typhoon", "wind"), ("hurricane", "wind"), ("wind", "wind")):
        if k in low:
            out["hazard"] = v; break
    m = re.search(r"([\d.]+)\s*(?:m/s)", low)
    if m: out["wind_ms"] = float(m.group(1))
    m = re.search(r"([\d.]+)\s*(?:m\b|metres?|meters?)(?!/)", low)
    if m: out["metres"] = float(m.group(1))
    return out


def find_company(text: str) -> Optional[dict]:
    """Match the longest owner name in the asset register that appears in the text."""
    try:
        from ..data.asset_register import search
    except Exception:                                          # noqa: BLE001
        return None
    words = re.findall(r"[A-Za-z][A-Za-z&\.\-']+", text)
    cands = []
    for n in (3, 2, 1):
        for i in range(len(words) - n + 1):
            q = " ".join(words[i:i + n])
            if len(q) >= 3 and q.lower() not in _STOP:
                cands.append(q)
    seen = set()
    for q in cands:
        if q.lower() in seen:
            continue
        seen.add(q.lower())
        try:
            r = search(q, limit=3)
        except Exception:                                      # noqa: BLE001
            continue
        if not r:
            continue
        top = r[0]
        if top.get("match") == "exact" or (top.get("match") == "group" and q[0].isupper()):
            pass
        elif top.get("match") == "prefix" and q[0].isupper() and len(q) >= 4 and (top.get("name") or "").lower().split()[:1] == q.lower().split()[:1]:
            pass
        else:
            continue
        return {"query": (top["key"].split(":", 1)[1].title() if top["key"].startswith("group:") else q), "key": top["key"], "name": top.get("name") or top["key"], "match": top["match"]}
    return None


_STOP = set("""the a an of for on in to and or at is are was were this that it my our your show me build run write make give list where what which how why
can you i we please about with from into over under report post twin scenario scenarios flood wind cyclone physical transition financial plant site sites
assets asset carbon audit tell about summarise summarize turn find check scan lca cbam company portfolio linkedin explain remember recall help hello thanks thank research scan model protect""".split())


# Vocabulary of the agent's own domain (plus a few meta phrases about the agent itself). Used only as a veto on the off-topic verdict, never to choose an intent.
DOMAIN_WORDS = re.compile(r"\b(flood\w*|cyclone\w*|hurricane\w*|typhoon\w*|storm\w*|surge|sea[- ]level|climate\w*|carbon|emission\w*|ghg|greenhouse|scopes? ?[123]|cbam|lca|life[- ]cycle|14067|"
                          r"hazard\w*|wildfire\w*|drought\w*|heat ?wave\w*|el nino|la nina|disaster\w*|transformer\w*|substation\w*|refiner\w*|factor(?:y|ies)|warehouse\w*|"
                          r"assets?|twin|3-?d|ngfs|physical risk|transition risk|stranded|voyage|vessel|bunker\w*|cement|steel|exposure|resilien\w*|mitigat\w*|get(?:ting)? started|can i ask|what can you do|what you can do|how (?:do|does) (?:this|it) work|net ?zero|mild|severe|optimistic|pessimistic|tcfd|issb|csrd)\b", re.I)

# ── persistence + singleton ─────────────────────────────────────────────────
MODEL_PATH = Path(__file__).resolve().parent.parent / "data" / "resident_nlu.json"
_M: Optional[Model] = None


def train(extra: Optional[list] = None, save: bool = True) -> dict:
    """Train on the corpus (+ logged feedback rows `extra`), report held-out and grouped-CV accuracy, save the model."""
    from .corpus import utterances, split_by_template, real_rows
    rows = utterances()
    real = real_rows()                                                # real people's out-of-scope questions (CLINC150, CC BY 3.0)
    tr, te = split_by_template(rows)
    ev = evaluate(tr + real, te)
    cv = grouped_cv(rows)
    allrows = [(u, i) for u, i, _ in rows] + real + list(extra or [])
    m = Model().fit(allrows)
    m.meta.update({"heldout_model_only": ev["accuracy"], "grouped_cv_mean": cv["mean"], "n_feedback": len(extra or []), "n_real_oos": len(real)})
    if save:
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        MODEL_PATH.write_text(json.dumps(m.to_json()))
    global _M
    _M = m
    hy = sum(m.predict(u)["intent"] == i for u, i in te) / max(len(te), 1)
    return {"n_train": len(allrows), "heldout_templates": ev["accuracy"], "grouped_cv": cv, "n_feedback": len(extra or []), "n_real_oos": len(real),
            "note": "held-out = phrasings never seen in training; the in-sample figure after adding feedback is not a held-out score",
            "errors_sample": ev["errors"][:6], "hybrid_in_sample": round(hy, 3)}


def get_model() -> Model:
    global _M
    if _M is None:
        if MODEL_PATH.exists():
            _M = Model.from_json(json.loads(MODEL_PATH.read_text()))
        else:
            train()
    return _M
