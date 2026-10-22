"""Hold-out test for cri.convective: fit on 1996-2009, predict 2010-2024 at random US land points.
  python scripts/convective_benchmark.py out.json [n_points]
No network needed (reads the bundled SPC table)."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cri import convective as V  # noqa: E402


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def auc(score, label):
    score, label = np.asarray(score, float), np.asarray(label, bool)
    pos, neg = score[label], score[~label]
    if len(pos) == 0 or len(neg) == 0:
        return None
    allv = np.concatenate([pos, neg])
    r = np.argsort(np.argsort(allv)).astype(float) + 1
    for v in np.unique(allv):
        m = allv == v
        if m.sum() > 1:
            r[m] = r[m].mean()
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def main(out, n=600):
    rng = np.random.default_rng(42)
    pts = []
    while len(pts) < n:
        la, lo = rng.uniform(25.0, 49.0), rng.uniform(-124.5, -67.0)
        if V.in_united_states(la, lo):
            pts.append((la, lo))
    res = {"n_points": len(pts), "early": [1996, 2009], "late": [2010, 2024], "radius_km": 40.0, "perils": {}}
    for name, kind, mv in (("hail_1in", "hail", 1.0), ("hail_2in", "hail", 2.0), ("tornado_all", "tor", -1e9), ("tornado_ef1plus", "tor", 1)):
        e = np.array([V.days_with_report(kind, la, lo, 40.0, 1996, 2009, mv)["days_per_year"] for la, lo in pts])
        l = np.array([V.days_with_report(kind, la, lo, 40.0, 2010, 2024, mv)["days_per_year"] for la, lo in pts])
        top = l >= np.percentile(l, 75)
        res["perils"][name] = {
            "mean_early_days_per_year": round(float(e.mean()), 3), "mean_late_days_per_year": round(float(l.mean()), 3),
            "late_over_early_ratio": round(float(l.mean() / e.mean()), 3), "spearman_early_vs_late": round(spearman(e, l), 3),
            "auc_early_ranks_late_top_quartile": round(auc(e, top), 3),
            "points_never_reported_early_but_reported_late": round(float(((e == 0) & (l > 0)).mean()), 3),
            "points_never_reported_early": round(float((e == 0).mean()), 3),
            "median_abs_error_days": round(float(np.median(np.abs(e - l))), 3),
            "within_factor_2_where_both_positive": round(float(np.mean((e[(e > 0) & (l > 0)] / l[(e > 0) & (l > 0)] < 2) & (e[(e > 0) & (l > 0)] / l[(e > 0) & (l > 0)] > 0.5))), 3),
            "percentiles_late_days_per_year": {str(q): round(float(np.percentile(l, q)), 3) for q in (10, 25, 50, 75, 90, 99)}}
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 600)
