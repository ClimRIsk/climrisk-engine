"""Analytic Hierarchy Process (Saaty) for ranking hazards, assets or measures against criteria, with the checks that make it defensible.  (12 Nov 2026)

What AHP is: experts compare criteria two at a time on a 1 to 9 scale ("how much more important is A than B?"). The principal eigenvector of the pairwise matrix gives the
weights. The consistency ratio (CR) says whether the judgments contradict each other (A > B, B > C, but C > A); CR above 0.10 means they should be revisited.

What this module will NOT do: invent judgments. The pairwise comparisons are the expert's input; there are no built-in defaults. Without them the answer is "not assessed", not a
guess. Every result carries the judgments it was built from, the consistency ratio, and a rank-stability figure (how often the ranking survives plausible changes to the weights),
so a weight chosen by one person is visible as one person's judgment.

Pure numpy, no data files. References: Saaty (1980), The Analytic Hierarchy Process; random-index values from Saaty (1980) / Alonso & Lamata (2006) for n up to 10.
"""
from __future__ import annotations

import itertools
import math
from typing import Optional, Sequence

import numpy as np

# Random consistency index by matrix size n (Saaty). n=1,2 are always consistent.
RI = {1: 0.0, 2: 0.0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}
SAATY_SCALE = "1 equal, 3 moderate, 5 strong, 7 very strong, 9 extreme; 2, 4, 6, 8 between; reciprocals for the reverse comparison"
CR_LIMIT = 0.10


def matrix_from_judgments(items: Sequence[str], judgments: dict) -> np.ndarray:
    """Reciprocal matrix from {(a, b): v} or {"a|b": v} meaning "a is v times as important as b" (v in [1/9, 9]). Every pair must be given exactly once (either order)."""
    items = list(items)
    n = len(items)
    if n < 2:
        raise ValueError("need at least two items to compare")
    if len(set(items)) != n:
        raise ValueError("item names must be unique")
    idx = {k: i for i, k in enumerate(items)}
    A = np.eye(n)
    seen = set()
    for key, v in judgments.items():
        a, b = key.split("|") if isinstance(key, str) else key
        if a not in idx or b not in idx or a == b:
            raise ValueError(f"unknown or repeated item in judgment {key!r}")
        v = float(v)
        if not (1.0 / 9.0 - 1e-9 <= v <= 9.0 + 1e-9) or not math.isfinite(v):
            raise ValueError(f"judgment {key!r} = {v} is outside the 1/9 to 9 scale")
        pair = frozenset((a, b))
        if pair in seen:
            raise ValueError(f"pair {a!r}, {b!r} given twice")
        seen.add(pair)
        A[idx[a], idx[b]] = v
        A[idx[b], idx[a]] = 1.0 / v
    missing = [(a, b) for a, b in itertools.combinations(items, 2) if frozenset((a, b)) not in seen]
    if missing:
        raise ValueError("missing comparisons: " + ", ".join(f"{a} vs {b}" for a, b in missing[:6]) + (" ..." if len(missing) > 6 else ""))
    return A


def weights(A: np.ndarray) -> dict:
    """Principal-eigenvector weights, lambda_max, consistency index and ratio. CR is None for n < 3 (nothing to be inconsistent about)."""
    A = np.asarray(A, float)
    n = A.shape[0]
    if A.shape != (n, n) or n < 2 or (A <= 0).any() or not np.allclose(A * A.T, 1.0, atol=1e-6):
        raise ValueError("matrix must be square, positive and reciprocal")
    vals, vecs = np.linalg.eig(A)
    k = int(np.argmax(vals.real))
    w = np.abs(vecs[:, k].real)
    w = w / w.sum()
    lam = float(vals[k].real)
    ci = (lam - n) / (n - 1)
    ri = RI.get(n)
    cr = None if n < 3 else (ci / ri if ri else None)
    return {"weights": w, "lambda_max": lam, "ci": float(ci), "cr": None if cr is None else float(cr), "n": n,
            "consistent": True if cr is None else bool(cr <= CR_LIMIT), "ri": ri}


def criteria_weights(criteria: Sequence[str], judgments: Optional[dict]) -> dict:
    """The entry point for a user's judgments. No judgments, no weights: returns status 'not_assessed'."""
    if not judgments:
        return {"status": "not_assessed", "reason": "no pairwise judgments were supplied; AHP weights are an expert's input and the engine does not supply defaults", "scale": SAATY_SCALE}
    try:
        A = matrix_from_judgments(criteria, judgments)
    except ValueError as e:
        return {"status": "invalid", "reason": str(e), "scale": SAATY_SCALE}
    r = weights(A)
    out = {"status": "ok" if r["consistent"] else "inconsistent", "criteria": list(criteria), "weights": {c: round(float(x), 4) for c, x in zip(criteria, r["weights"])},
           "lambda_max": round(r["lambda_max"], 4), "consistency_index": round(r["ci"], 4), "consistency_ratio": None if r["cr"] is None else round(r["cr"], 4),
           "judgments": {(k if isinstance(k, str) else "|".join(k)): float(v) for k, v in judgments.items()}, "scale": SAATY_SCALE}
    if not r["consistent"]:
        out["advice"] = _most_inconsistent(criteria, A, r["weights"])
    return out


def _most_inconsistent(criteria, A, w) -> str:
    """Name the comparison that disagrees most with the weights (largest |a_ij - w_i/w_j| in log space), so the expert knows which judgment to revisit."""
    n = len(criteria)
    best, pair = -1.0, None
    for i, j in itertools.combinations(range(n), 2):
        d = abs(math.log(A[i, j]) - math.log(w[i] / w[j]))
        if d > best:
            best, pair = d, (i, j)
    i, j = pair
    return (f"Consistency ratio is above {CR_LIMIT}. The comparison that disagrees most with the rest is {criteria[i]} vs {criteria[j]} "
            f"(judged {A[i, j]:.2g}, the other judgments imply about {w[i] / w[j]:.2g}). Revisit that one first.")


def group_judgments(sets: Sequence[dict]) -> dict:
    """Combine several experts' judgments by the geometric mean per pair (the aggregation that keeps reciprocity). Pairs must be the same set in every expert's judgments."""
    if not sets:
        return {}
    norm = []
    for s in sets:
        m = {}
        for k, v in s.items():
            a, b = k.split("|") if isinstance(k, str) else k
            m[(a, b)] = float(v)
        norm.append(m)
    pairs = {frozenset(k) for k in norm[0]}
    if any({frozenset(k) for k in m} != pairs for m in norm):
        raise ValueError("every expert must judge the same set of pairs")
    out = {}
    for key in norm[0]:
        a, b = key
        vals = []
        for m in norm:
            vals.append(m[(a, b)] if (a, b) in m else 1.0 / m[(b, a)])
        out[key] = float(math.exp(sum(math.log(v) for v in vals) / len(vals)))
    return out


def rank(alternatives: dict, crit_weights: dict, benefit: Optional[dict] = None, trials: int = 2000, concentration: float = 60.0, seed: int = 7) -> dict:
    """Rank alternatives by the weighted sum of their normalised criterion scores.

    alternatives: {name: {criterion: raw score}}; every alternative must have every criterion. benefit[c] False means lower is better (default True).
    Scores are min-max normalised per criterion to 0..1 (a stated, common choice; the ranking can depend on it, which `stability` partly reveals).
    stability: share of random weight vectors (Dirichlet around the given weights, `concentration` controls the spread) under which each alternative keeps its rank.
    """
    crits = list(crit_weights)
    names = list(alternatives)
    if not names or not crits:
        return {"status": "not_assessed", "reason": "need alternatives and weighted criteria"}
    for a in names:
        miss = [c for c in crits if c not in alternatives[a] or alternatives[a][c] is None or not math.isfinite(float(alternatives[a][c]))]
        if miss:
            return {"status": "not_assessed", "reason": f"{a} has no score for {', '.join(miss)}; a missing score is not treated as zero"}
    benefit = benefit or {}
    X = np.array([[float(alternatives[a][c]) for c in crits] for a in names])
    lo, hi = X.min(axis=0), X.max(axis=0)
    span = np.where(hi > lo, hi - lo, 1.0)
    N = (X - lo) / span
    for j, c in enumerate(crits):
        if not benefit.get(c, True):
            N[:, j] = 1.0 - N[:, j]
        if hi[j] == lo[j]:
            N[:, j] = 0.5
    w = np.array([crit_weights[c] for c in crits], float)
    w = w / w.sum()
    score = N @ w
    order = np.argsort(-score)
    rng = np.random.default_rng(seed)
    W = rng.dirichlet(np.maximum(w * concentration, 1e-3), size=trials)
    S = N @ W.T                                                        # alternatives x trials
    ranks = np.argsort(np.argsort(-S, axis=0), axis=0)
    same = np.array([(ranks[i] == int(np.where(order == i)[0][0])).mean() for i in range(len(names))])
    top = np.array([(ranks[i] == 0).mean() for i in range(len(names))])
    rows = [{"name": names[i], "score": round(float(score[i]), 4), "rank": r + 1, "keeps_rank_share": round(float(same[i]), 3), "is_first_share": round(float(top[i]), 3),
             "by_criterion": {c: round(float(N[i, j] * w[j]), 4) for j, c in enumerate(crits)}} for r, i in enumerate(order)]
    lead = rows[0]
    return {"status": "ok", "ranking": rows, "weights": {c: round(float(x), 4) for c, x in zip(crits, w)}, "normalisation": "min-max per criterion, 0 to 1",
            "stability_note": (f"{lead['name']} is first in {100 * lead['is_first_share']:.0f}% of {trials} weight perturbations; "
                               + ("the top rank is robust to the weights." if lead["is_first_share"] >= 0.8 else "the top rank depends on the weights chosen: say so when you present it."))}
