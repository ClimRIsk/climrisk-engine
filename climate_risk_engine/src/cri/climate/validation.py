"""
Model Validation Framework — Task 57
======================================
Implements three standard model performance metrics for credit risk models:

1. Gini coefficient (discriminatory power)
   — Measures how well the model separates defaulters from non-defaulters.
   — Gini = 2 × AUROC − 1.
   — Source: Engelmann, Hayden, Tasche (2003) "Measuring the Discriminative
             Power of Rating Systems", Bundesbank Discussion Paper 01/2003.

2. Brier score (probability calibration)
   — Mean squared error between predicted PD and realised binary default.
   — BS = (1/N) Σ (PD_i − D_i)²   where D_i ∈ {0, 1}.
   — BS < 0.25 indicates reasonable calibration.
   — Source: Brier (1950) "Verification of Forecasts Expressed in Terms of
             Probability", Monthly Weather Review 78(1):1–3.

3. Kupiec Proportion-of-Failures (VaR coverage test)
   — Tests whether the realised violation rate matches the nominal VaR level.
   — LR_PoF = −2 ln[(1−p)^(T−x) p^x] + 2 ln[(1−x/T)^(T−x) (x/T)^x]
   — Distributed χ²(1) under H₀ (correct coverage).
   — Source: Kupiec (1995) "Techniques for Verifying the Accuracy of Risk
             Measurement Models", Journal of Derivatives 3(2):73–84.
   — Basel III §MAR99 mandates this test for internal VaR model approval.

4. Hosmer-Lemeshow calibration test (supplementary)
   — Groups observations into 10 risk buckets; tests whether predicted PD
     matches realised default rates across buckets.
   — HL stat ~ χ²(8) under H₀.
   — Source: Hosmer & Lemeshow (2013) Applied Logistic Regression §5.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional


# ── Helper ────────────────────────────────────────────────────────────────────

def _chi2_pvalue_approx(stat: float, df: int) -> float:
    """Approximate p-value for chi-squared statistic (Wilson-Hilferty).

    Accurate to ~2% for df ≥ 1, stat ≥ 0.
    Returns 0 if stat is very large (p-value effectively 0).
    """
    if stat <= 0.0:
        return 1.0
    # Wilson-Hilferty normal approximation
    z = (stat / df) ** (1 / 3) - (1 - 2 / (9 * df))
    z /= math.sqrt(2 / (9 * df))
    # Normal CDF survival (1 - Φ(z))
    return 0.5 * math.erfc(z / math.sqrt(2))


# ── 1. Gini coefficient ───────────────────────────────────────────────────────

def compute_gini(
    predicted_pd: List[float],
    realised_default: List[int],
) -> Optional[float]:
    """Gini coefficient = 2 × AUROC − 1.

    predicted_pd      : Ordered list of model PD scores.
    realised_default  : Corresponding 0/1 default indicators.
    Returns Gini in [−1, 1]; values > 0.6 considered strong discriminatory power.
    """
    n = len(predicted_pd)
    if n < 2:
        return None

    n_defaults = sum(realised_default)
    if n_defaults == 0 or n_defaults == n:
        return None  # Undefined if all same class

    # Sort by predicted PD descending (higher PD = predicted riskier)
    paired = sorted(zip(predicted_pd, realised_default), key=lambda x: -x[0])

    # Compute AUROC via trapezoidal rule on the ROC curve
    auroc = 0.0
    tp = 0
    fp = 0
    n_pos = n_defaults
    n_neg = n - n_defaults
    prev_tp = 0
    prev_fp = 0

    for _, d in paired:
        if d == 1:
            tp += 1
        else:
            fp += 1
        # Add trapezoid area
        tpr = tp / n_pos
        fpr = fp / n_neg
        prev_tpr = prev_tp / n_pos
        prev_fpr = prev_fp / n_neg
        auroc += 0.5 * (tpr + prev_tpr) * (fpr - prev_fpr)
        prev_tp, prev_fp = tp, fp

    gini = 2.0 * auroc - 1.0
    return round(gini, 4)


# ── 2. Brier score ────────────────────────────────────────────────────────────

def compute_brier_score(
    predicted_pd: List[float],
    realised_default: List[int],
) -> Optional[float]:
    """Brier score = (1/N) Σ (PD_i − D_i)².

    Returns Brier score in [0, 1]; < 0.25 is reasonable, < 0.10 is strong.
    """
    n = len(predicted_pd)
    if n == 0:
        return None
    bs = sum((p - d) ** 2 for p, d in zip(predicted_pd, realised_default)) / n
    return round(bs, 6)


# ── 3. Kupiec PoF test ────────────────────────────────────────────────────────

def compute_kupiec_pof(
    violations: int,      # number of days VaR was breached
    n_periods:  int,      # total number of observation periods
    var_level:  float = 0.99,  # VaR confidence level (99%)
) -> Dict[str, Any]:
    """Kupiec Proportion-of-Failures VaR coverage test.

    H₀: true violation rate = (1 − var_level)
    H₁: true violation rate ≠ (1 − var_level)

    Returns dict with: LR_stat, p_value, critical_value (95%), reject_h0, message.
    """
    if n_periods < 1:
        return {"error": "n_periods must be ≥ 1"}

    p = 1.0 - var_level         # nominal violation probability (e.g. 0.01 for 99% VaR)
    T = n_periods
    x = violations

    # Avoid log(0)
    if x == 0:
        # Constrained log-likelihood = log(1-p)^T only
        lr = -2.0 * T * math.log(1.0 - p)
    elif x == T:
        lr = -2.0 * T * math.log(p)
    else:
        hat_p = x / T    # empirical violation rate
        lr = -2.0 * (
            x * math.log(p / hat_p) + (T - x) * math.log((1 - p) / (1 - hat_p))
        )

    lr = max(0.0, lr)
    pval = _chi2_pvalue_approx(lr, df=1)
    critical_95 = 3.841   # χ²(1, 0.05)
    reject = lr > critical_95

    violation_rate = x / T if T > 0 else 0.0
    return {
        "lr_statistic":       round(lr, 4),
        "p_value":            round(pval, 4),
        "critical_value_95":  critical_95,
        "reject_h0":          reject,
        "violations":         x,
        "n_periods":          T,
        "empirical_rate":     round(violation_rate, 6),
        "nominal_rate":       round(p, 4),
        "message": (
            f"VaR₉₉ coverage {'REJECTED' if reject else 'NOT REJECTED'} "
            f"at 95% confidence (LR={lr:.3f}, χ²(1) critical=3.841). "
            f"Empirical violation rate: {violation_rate:.1%} vs nominal {p:.1%}."
        ),
    }


# ── 4. Hosmer-Lemeshow test ───────────────────────────────────────────────────

def compute_hosmer_lemeshow(
    predicted_pd: List[float],
    realised_default: List[int],
    n_groups: int = 10,
) -> Dict[str, Any]:
    """Hosmer-Lemeshow χ² calibration test (10 decile bins by default).

    Tests whether predicted PD matches observed default rates across risk buckets.
    HL ~ χ²(n_groups − 2) under H₀ (good calibration).
    """
    n = len(predicted_pd)
    if n < n_groups:
        return {"error": f"Need at least {n_groups} observations for HL test."}

    # Sort by predicted PD
    pairs = sorted(zip(predicted_pd, realised_default), key=lambda x: x[0])
    group_size = n // n_groups

    hl_stat = 0.0
    groups_out = []
    for g in range(n_groups):
        start = g * group_size
        end = (g + 1) * group_size if g < n_groups - 1 else n
        grp = pairs[start:end]
        n_g  = len(grp)
        obs_d = sum(d for _, d in grp)
        exp_d = sum(p for p, _ in grp)

        # Protect against exp_d=0 or exp_d=n_g
        if exp_d > 1e-9 and (n_g - exp_d) > 1e-9:
            hl_stat += (obs_d - exp_d) ** 2 / (exp_d * (1 - exp_d / n_g))

        groups_out.append({
            "group":         g + 1,
            "n":             n_g,
            "expected_defaults": round(exp_d, 2),
            "observed_defaults": obs_d,
            "mean_pd":       round(sum(p for p, _ in grp) / n_g, 4),
        })

    df = n_groups - 2
    pval = _chi2_pvalue_approx(hl_stat, df=max(1, df))
    critical_95 = 15.507   # χ²(8, 0.05)
    reject = hl_stat > critical_95

    return {
        "hl_statistic":      round(hl_stat, 4),
        "p_value":           round(pval, 4),
        "df":                df,
        "critical_value_95": critical_95,
        "reject_h0":         reject,
        "n_groups":          n_groups,
        "groups":            groups_out,
        "message": (
            f"Calibration {'POOR' if reject else 'ACCEPTABLE'}: "
            f"HL={hl_stat:.2f}, χ²({df}) critical={critical_95:.3f}, p={pval:.3f}."
        ),
    }


# ── Top-level orchestrator ────────────────────────────────────────────────────

def run_validation(
    observations: List[Dict[str, Any]],
    metrics: List[str] = ("gini", "brier", "kupiec"),
) -> Dict[str, Any]:
    """Run all requested validation metrics on a set of observations.

    Each observation dict should have:
      predicted_eal   : float  — model's expected annual loss prediction (USD M)
      realised_loss   : float  — actual observed loss for the period (USD M)
      predicted_pd    : float  — model's PD estimate (0–1)
      realised_default: int    — 1 if the obligor defaulted in this period, else 0
    """
    n = len(observations)
    if n == 0:
        return {"error": "No observations provided."}

    pds  = [float(o.get("predicted_pd", 0.0)) for o in observations]
    defs = [int(o.get("realised_default", 0))  for o in observations]

    # Kupiec needs VaR violation data — use EAL as proxy for VaR99 if available
    eal_preds  = [float(o.get("predicted_eal", 0.0))  for o in observations]
    real_losses= [float(o.get("realised_loss", 0.0))  for o in observations]
    violations = sum(1 for pred, real in zip(eal_preds, real_losses) if real > pred)

    result: Dict[str, Any] = {"n_observations": n}

    if "gini" in metrics:
        result["gini_coefficient"] = compute_gini(pds, defs)

    if "brier" in metrics:
        result["brier_score"] = compute_brier_score(pds, defs)

    if "kupiec" in metrics:
        result["kupiec_pof"] = compute_kupiec_pof(violations, n, var_level=0.99)

    if "hl" in metrics and n >= 10:
        result["hl_test"] = compute_hosmer_lemeshow(pds, defs)

    # Interpretation
    msgs = []
    g = result.get("gini_coefficient")
    if g is not None:
        if g >= 0.60: msgs.append(f"Strong discrimination (Gini={g:.2f})")
        elif g >= 0.40: msgs.append(f"Moderate discrimination (Gini={g:.2f})")
        else: msgs.append(f"Weak discrimination (Gini={g:.2f}) — model may not separate risk well")

    b = result.get("brier_score")
    if b is not None:
        if b < 0.10:  msgs.append(f"Strong PD calibration (Brier={b:.4f})")
        elif b < 0.25: msgs.append(f"Acceptable PD calibration (Brier={b:.4f})")
        else: msgs.append(f"Poor PD calibration (Brier={b:.4f}) — recalibration recommended")

    k = result.get("kupiec_pof")
    if k and isinstance(k, dict) and "message" in k:
        msgs.append(k["message"])

    result["summary"] = " | ".join(msgs) if msgs else "No metrics computed."
    result["methodology"] = (
        "Gini: AUROC×2-1 (Engelmann 2003). "
        "Brier: MSE of PD vs default indicator. "
        "Kupiec: PoF test χ²(1) (Basel III MAR99). "
        "HL: Hosmer-Lemeshow χ²(8) calibration."
    )
    return result
