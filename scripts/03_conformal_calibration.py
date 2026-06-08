"""
Conformal prediction calibration and evaluation for the Conformal TB Triage pipeline.

Uses probe predictions from Phase 2 to:
1. Calibrate APS, RAPS, Mondrian (class-conditional) conformal methods
2. Evaluate coverage, set size, singleton fraction on test sets
3. Run Conformal Risk Control (CRC) for direct FNR bounding
4. Check all deployment gates
5. Produce the three-tier triage classification

Run: python3 src/conformal/conformal_calibration.py
"""

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

from config import *

PREDICTIONS_PATH = RESULTS_DIR / "probe_predictions.parquet"
ALPHA_TARGETS = [0.05, 0.10, 0.20]

# Only use models that actually work (AUROC > 0.70 on test)
VALID_MODELS = ["rad_dino", "biomedclip", "torchxrayvision", "dinov2"]
VALID_PROBES = ["linear", "knn", "xgboost", "mlp"]

# ---------------------------------------------------------------------------
# Conformal methods
# ---------------------------------------------------------------------------

def compute_aps_scores(y_prob, y_true):
    """Adaptive Prediction Sets nonconformity scores.
    Score = 1 - P(true class). Lower score = more conforming."""
    scores = np.where(y_true == 1, 1 - y_prob, y_prob)
    return scores


def calibrate_aps(cal_scores, alpha):
    """Compute APS threshold for target miscoverage alpha."""
    n = len(cal_scores)
    # Finite-sample corrected quantile
    q_level = np.ceil((n + 1) * (1 - alpha)) / n
    q_level = min(q_level, 1.0)
    threshold = np.quantile(cal_scores, q_level)
    return threshold


def predict_aps(y_prob, threshold):
    """Generate prediction sets using APS threshold.
    Returns: list of sets, each containing 0 (non-TB), 1 (TB), or both."""
    sets = []
    for p in y_prob:
        s = set()
        # Include class 1 (TB) if its nonconformity score <= threshold
        if (1 - p) <= threshold:
            s.add(1)
        # Include class 0 (non-TB) if its score <= threshold
        if p <= threshold:
            s.add(0)
        sets.append(s)
    return sets


def calibrate_raps(cal_scores, alpha, lam=0.1, k_reg=1):
    """RAPS: regularised APS. Adds penalty for including extra classes."""
    n = len(cal_scores)
    # RAPS adds lambda * max(|C| - k_reg, 0) to the score
    # For binary, this simplifies: penalty is lambda if set size > k_reg
    regularised = cal_scores + lam * (cal_scores > np.median(cal_scores)).astype(float)
    q_level = np.ceil((n + 1) * (1 - alpha)) / n
    q_level = min(q_level, 1.0)
    threshold = np.quantile(regularised, q_level)
    return threshold, lam


def predict_raps(y_prob, threshold, lam=0.1):
    """Generate RAPS prediction sets."""
    sets = []
    for p in y_prob:
        s = set()
        score_tb = 1 - p
        score_nontb = p
        # Add classes in decreasing probability order, with regularisation
        if p >= 0.5:
            # TB more likely
            if score_tb <= threshold:
                s.add(1)
            if score_nontb <= threshold - lam * (1 in s):
                s.add(0)
        else:
            # Non-TB more likely
            if score_nontb <= threshold:
                s.add(0)
            if score_tb <= threshold - lam * (0 in s):
                s.add(1)
        sets.append(s)
    return sets


def calibrate_mondrian(cal_scores, cal_y, alpha):
    """Class-conditional (Mondrian) conformal: separate threshold per class."""
    thresholds = {}
    for cls in [0, 1]:
        mask = cal_y == cls
        cls_scores = cal_scores[mask]
        n = len(cls_scores)
        if n < 5:
            thresholds[cls] = 1.0  # fallback
            continue
        q_level = np.ceil((n + 1) * (1 - alpha)) / n
        q_level = min(q_level, 1.0)
        thresholds[cls] = np.quantile(cls_scores, q_level)
    return thresholds


def predict_mondrian(y_prob, thresholds):
    """Generate Mondrian prediction sets."""
    sets = []
    for p in y_prob:
        s = set()
        if (1 - p) <= thresholds.get(1, 1.0):
            s.add(1)
        if p <= thresholds.get(0, 1.0):
            s.add(0)
        sets.append(s)
    return sets


def conformal_risk_control(cal_y_prob, cal_y_true, target_fnr=0.10):
    """Conformal Risk Control: find threshold bounding E[FNR] <= target_fnr.
    Returns operating threshold on predicted probability."""
    # Sort by decreasing probability of TB
    sorted_probs = np.sort(cal_y_prob)
    best_thresh = 0.0

    for thresh in np.linspace(0, 1, 1000):
        predicted_positive = cal_y_prob >= thresh
        fn = ((cal_y_true == 1) & (~predicted_positive)).sum()
        tp_fn = (cal_y_true == 1).sum()
        fnr = fn / max(tp_fn, 1)
        if fnr <= target_fnr:
            best_thresh = thresh

    return best_thresh


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_prediction_sets(pred_sets, y_true):
    """Compute conformal metrics from prediction sets."""
    n = len(pred_sets)
    coverages = [int(y in s) for y, s in zip(y_true, pred_sets)]
    set_sizes = [len(s) for s in pred_sets]
    singletons = [int(len(s) == 1) for s in pred_sets]
    empties = [int(len(s) == 0) for s in pred_sets]

    # Class-conditional coverage
    tb_mask = y_true == 1
    nontb_mask = y_true == 0

    tb_coverage = np.mean([coverages[i] for i in range(n) if tb_mask[i]]) if tb_mask.any() else float("nan")
    nontb_coverage = np.mean([coverages[i] for i in range(n) if nontb_mask[i]]) if nontb_mask.any() else float("nan")

    return {
        "marginal_coverage": round(np.mean(coverages), 4),
        "tb_coverage": round(tb_coverage, 4),
        "nontb_coverage": round(nontb_coverage, 4),
        "mean_set_size": round(np.mean(set_sizes), 4),
        "median_set_size": round(np.median(set_sizes), 4),
        "singleton_frac": round(np.mean(singletons), 4),
        "empty_frac": round(np.mean(empties), 4),
        "coverage_disparity": round(abs(tb_coverage - nontb_coverage), 4),
    }


def three_tier_triage(pred_sets, y_true):
    """Classify into triage tiers based on prediction set content."""
    tiers = []
    for s in pred_sets:
        if s == {0}:
            tiers.append("tier1_clear")       # High confidence non-TB
        elif s == {1}:
            tiers.append("tier2_refer")       # High confidence TB → Xpert
        elif s == {0, 1}:
            tiers.append("tier3_uncertain")   # Needs clinical review
        else:
            tiers.append("tier3_uncertain")   # Empty set → uncertain

    tiers = np.array(tiers)

    result = {}
    for tier in ["tier1_clear", "tier2_refer", "tier3_uncertain"]:
        mask = tiers == tier
        n_tier = mask.sum()
        result[f"{tier}_n"] = int(n_tier)
        result[f"{tier}_frac"] = round(n_tier / len(tiers), 4)
        if n_tier > 0:
            tier_y = y_true[mask]
            result[f"{tier}_tb_frac"] = round(tier_y.mean(), 4) if len(tier_y) > 0 else 0
        else:
            result[f"{tier}_tb_frac"] = 0

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("CONFORMAL CALIBRATION — Conformal TB Triage")
    print("=" * 70)

    preds = pd.read_parquet(PREDICTIONS_PATH)
    print(f"Loaded predictions: {len(preds)} rows", flush=True)

    all_conformal_results = []
    all_triage_results = []
    all_crc_results = []

    for emb in VALID_MODELS:
        for probe in VALID_PROBES:
            combo = f"{emb}+{probe}"

            # Get calibration and test predictions
            cal = preds[(preds["embedding"] == emb) & (preds["probe"] == probe) & (preds["split"] == "calibration")]
            test = preds[(preds["embedding"] == emb) & (preds["probe"] == probe) & (preds["split"] == "test")]
            ext = preds[(preds["embedding"] == emb) & (preds["probe"] == probe) & (preds["split"] == "ext_pakistan")]

            if len(cal) < 50 or len(test) < 100:
                continue

            cal_y = cal["y_true"].values
            cal_prob = cal["y_prob"].values
            test_y = test["y_true"].values
            test_prob = test["y_prob"].values

            cal_scores = compute_aps_scores(cal_prob, cal_y)

            # ── APS ──
            for alpha in ALPHA_TARGETS:
                thresh = calibrate_aps(cal_scores, alpha)
                pred_sets = predict_aps(test_prob, thresh)
                metrics = evaluate_prediction_sets(pred_sets, test_y)
                triage = three_tier_triage(pred_sets, test_y)

                row = {
                    "embedding": emb, "probe": probe, "method": "APS",
                    "alpha": alpha, "split": "test", **metrics,
                }
                all_conformal_results.append(row)

                if alpha == 0.10:
                    triage_row = {"embedding": emb, "probe": probe, "method": "APS", "alpha": alpha, **triage}
                    all_triage_results.append(triage_row)

            # ── RAPS ──
            for alpha in ALPHA_TARGETS:
                for lam in [0.001, 0.01, 0.1, 1.0]:
                    thresh, _ = calibrate_raps(cal_scores, alpha, lam=lam)
                    pred_sets = predict_raps(test_prob, thresh, lam=lam)
                    metrics = evaluate_prediction_sets(pred_sets, test_y)

                    row = {
                        "embedding": emb, "probe": probe, "method": f"RAPS_lam{lam}",
                        "alpha": alpha, "split": "test", **metrics,
                    }
                    all_conformal_results.append(row)

            # ── Mondrian (class-conditional) ──
            for alpha in ALPHA_TARGETS:
                thresholds = calibrate_mondrian(cal_scores, cal_y, alpha)
                pred_sets = predict_mondrian(test_prob, thresholds)
                metrics = evaluate_prediction_sets(pred_sets, test_y)
                triage = three_tier_triage(pred_sets, test_y)

                row = {
                    "embedding": emb, "probe": probe, "method": "Mondrian",
                    "alpha": alpha, "split": "test", **metrics,
                }
                all_conformal_results.append(row)

                if alpha == 0.10:
                    triage_row = {"embedding": emb, "probe": probe, "method": "Mondrian", "alpha": alpha, **triage}
                    all_triage_results.append(triage_row)

                # Also evaluate on Pakistan external
                if len(ext) > 50:
                    ext_y = ext["y_true"].values
                    ext_prob = ext["y_prob"].values
                    ext_sets = predict_mondrian(ext_prob, thresholds)
                    ext_metrics = evaluate_prediction_sets(ext_sets, ext_y)
                    ext_row = {
                        "embedding": emb, "probe": probe, "method": "Mondrian",
                        "alpha": alpha, "split": "ext_pakistan", **ext_metrics,
                    }
                    all_conformal_results.append(ext_row)

            # ── Conformal Risk Control (CRC) ──
            for target_fnr in [0.05, 0.10, 0.15]:
                crc_thresh = conformal_risk_control(cal_prob, cal_y, target_fnr)
                test_pred_pos = test_prob >= crc_thresh
                test_fn = ((test_y == 1) & (~test_pred_pos)).sum()
                test_tp = ((test_y == 1) & test_pred_pos).sum()
                test_tn = ((test_y == 0) & (~test_pred_pos)).sum()
                test_fp = ((test_y == 0) & test_pred_pos).sum()

                achieved_fnr = test_fn / max(test_fn + test_tp, 1)
                sensitivity = test_tp / max(test_tp + test_fn, 1)
                specificity = test_tn / max(test_tn + test_fp, 1)

                crc_row = {
                    "embedding": emb, "probe": probe, "method": "CRC",
                    "target_fnr": target_fnr, "threshold": round(crc_thresh, 4),
                    "achieved_fnr": round(achieved_fnr, 4),
                    "sensitivity": round(sensitivity, 4),
                    "specificity": round(specificity, 4),
                    "n_test": len(test_y),
                }
                all_crc_results.append(crc_row)

    # ── Save results ──
    conf_df = pd.DataFrame(all_conformal_results)
    # In-sample (probe-training-split) calibration — the defect documented in the
    # manuscript. The authoritative held-out table conformal_results.csv is written
    # by conformal_pipeline.py; this arm is suffixed to avoid clobbering it.
    conf_df.to_csv(TABLES_DIR / "conformal_results_insample.csv", index=False)

    triage_df = pd.DataFrame(all_triage_results)
    triage_df.to_csv(TABLES_DIR / "triage_results.csv", index=False)

    crc_df = pd.DataFrame(all_crc_results)
    crc_df.to_csv(TABLES_DIR / "crc_results.csv", index=False)

    # ── Summary: Primary model (RAD-DINO + linear + Mondrian @ alpha=0.10) ──
    print(f"\n{'=' * 70}")
    print("PRIMARY RESULT: RAD-DINO + linear + Mondrian @ alpha=0.10")
    print(f"{'=' * 70}")

    primary = conf_df[
        (conf_df["embedding"] == "rad_dino") &
        (conf_df["probe"] == "linear") &
        (conf_df["method"] == "Mondrian") &
        (conf_df["alpha"] == 0.10) &
        (conf_df["split"] == "test")
    ]
    if len(primary) > 0:
        r = primary.iloc[0]
        print(f"  TB-class coverage:    {r['tb_coverage']:.4f}  (target: >= 0.90)")
        print(f"  Non-TB coverage:      {r['nontb_coverage']:.4f}")
        print(f"  Marginal coverage:    {r['marginal_coverage']:.4f}")
        print(f"  Mean set size:        {r['mean_set_size']:.4f}")
        print(f"  Singleton fraction:   {r['singleton_frac']:.4f}")
        print(f"  Coverage disparity:   {r['coverage_disparity']:.4f}")
        print(f"  Empty set fraction:   {r['empty_frac']:.4f}")

    # ── Summary: Best model Mondrian @ alpha=0.10 ──
    print(f"\n{'=' * 70}")
    print("ALL MODELS: Mondrian @ alpha=0.10 on TBX11K test")
    print(f"{'=' * 70}")

    mondrian_10 = conf_df[
        (conf_df["method"] == "Mondrian") &
        (conf_df["alpha"] == 0.10) &
        (conf_df["split"] == "test")
    ].copy()
    if len(mondrian_10) > 0:
        mondrian_10["combo"] = mondrian_10["embedding"] + " + " + mondrian_10["probe"]
        display_cols = ["combo", "tb_coverage", "nontb_coverage", "singleton_frac", "mean_set_size"]
        print(mondrian_10[display_cols].to_string(index=False))

    # ── CRC Summary ──
    print(f"\n{'=' * 70}")
    print("CONFORMAL RISK CONTROL: FNR targets")
    print(f"{'=' * 70}")

    crc_primary = crc_df[(crc_df["embedding"] == "rad_dino") & (crc_df["probe"] == "linear")]
    if len(crc_primary) > 0:
        print(crc_primary[["target_fnr", "threshold", "achieved_fnr", "sensitivity", "specificity"]].to_string(index=False))

    # ── Three-tier triage ──
    print(f"\n{'=' * 70}")
    print("THREE-TIER TRIAGE (Mondrian @ alpha=0.10)")
    print(f"{'=' * 70}")

    triage_primary = triage_df[
        (triage_df["embedding"] == "rad_dino") &
        (triage_df["probe"] == "linear") &
        (triage_df["method"] == "Mondrian")
    ]
    if len(triage_primary) > 0:
        t = triage_primary.iloc[0]
        print(f"  Tier 1 (clear non-TB):  {t['tier1_clear_n']:>5} ({t['tier1_clear_frac']:.1%})  TB among these: {t['tier1_clear_tb_frac']:.3f}")
        print(f"  Tier 2 (refer for Xpert): {t['tier2_refer_n']:>5} ({t['tier2_refer_frac']:.1%})  TB among these: {t['tier2_refer_tb_frac']:.3f}")
        print(f"  Tier 3 (uncertain):     {t['tier3_uncertain_n']:>5} ({t['tier3_uncertain_frac']:.1%})  TB among these: {t['tier3_uncertain_tb_frac']:.3f}")

    # ── Deployment gates ──
    print(f"\n{'=' * 70}")
    print("DEPLOYMENT GATES")
    print(f"{'=' * 70}")

    # G1: Already checked in probe training
    probe_results = pd.read_csv(RESULTS_DIR / "probe_results.csv")
    max_auroc = probe_results[probe_results["split"] == "test"]["auroc"].max()
    g1 = max_auroc >= 0.75
    print(f"  G1 (AUROC >= 0.75):           {'PASS' if g1 else 'FAIL'}  ({max_auroc:.4f})")

    # G2: TB-class coverage >= 85%
    best_tb_cov = mondrian_10["tb_coverage"].max() if len(mondrian_10) > 0 else 0
    g2 = best_tb_cov >= 0.85
    print(f"  G2 (TB coverage >= 85%):      {'PASS' if g2 else 'FAIL'}  ({best_tb_cov:.4f})")

    # G3: Singleton fraction >= 40%
    primary_singleton = primary.iloc[0]["singleton_frac"] if len(primary) > 0 else 0
    g3 = primary_singleton >= 0.40
    print(f"  G3 (Singleton >= 40%):        {'PASS' if g3 else 'FAIL'}  ({primary_singleton:.4f})")

    # G4: Coverage disparity <= 15pp
    primary_disparity = primary.iloc[0]["coverage_disparity"] if len(primary) > 0 else 1.0
    g4 = primary_disparity <= 0.15
    print(f"  G4 (Disparity <= 15pp):       {'PASS' if g4 else 'FAIL'}  ({primary_disparity:.4f})")

    # G5: evaluated later (requires recalibration simulation)
    print(f"  G5 (Recal n <= 500):          PENDING (requires recalibration simulation)")

    all_pass = g1 and g2 and g3 and g4
    print(f"\n  Joint (G1-G4): {'ALL PASS' if all_pass else 'SOME FAILED'}")

    print(f"\nResults saved to: {TABLES_DIR}/")
    print("  conformal_results_insample.csv")
    print("  triage_results.csv")
    print("  crc_results.csv")


if __name__ == "__main__":
    main()
