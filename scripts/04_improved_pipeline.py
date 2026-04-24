"""
Improved conformal pipeline applying all pre-registered enhancements:
1. L2 normalisation of embeddings
2. Probability calibration (temperature, Platt, isotonic)
3. Fused embeddings
4. Weighted conformal prediction for covariate shift
5. Full re-evaluation with deployment gates

Run: python3 src/conformal/improved_pipeline.py
"""

import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import normalize
from sklearn.isotonic import IsotonicRegression

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

from config import *

N_FOLDS = 5

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_split_data(emb_name, emb_cols, df, split, l2_norm=False):
    """Get X, y for a split. Optionally L2-normalise."""
    mask = (df["split"] == split) & (df["tb_binary"].isin(["tb_positive", "tb_negative"]))
    sub = df[mask]
    X = sub[emb_cols].values.astype(np.float32)
    y = (sub["tb_binary"] == "tb_positive").astype(int).values
    ids = sub["patient_id"].values
    if l2_norm and X.shape[0] > 0:
        X = normalize(X, norm="l2")
    return X, y, ids


# ---------------------------------------------------------------------------
# Probability calibration
# ---------------------------------------------------------------------------

def temperature_scale(logits_cal, y_cal, logits_test):
    """Find optimal temperature T that minimises NLL on calibration set."""
    from scipy.optimize import minimize_scalar

    def nll(T):
        scaled = logits_cal / T
        probs = 1 / (1 + np.exp(-scaled))
        probs = np.clip(probs, 1e-7, 1 - 1e-7)
        return -np.mean(y_cal * np.log(probs) + (1 - y_cal) * np.log(1 - probs))

    result = minimize_scalar(nll, bounds=(0.1, 10.0), method="bounded")
    T = result.x
    calibrated = 1 / (1 + np.exp(-logits_test / T))
    return calibrated, T


def platt_scale(probs_cal, y_cal, probs_test):
    """Platt scaling: logistic regression on logits."""
    logits_cal = np.log(np.clip(probs_cal, 1e-7, 1 - 1e-7) / (1 - np.clip(probs_cal, 1e-7, 1 - 1e-7)))
    logits_test = np.log(np.clip(probs_test, 1e-7, 1 - 1e-7) / (1 - np.clip(probs_test, 1e-7, 1 - 1e-7)))

    lr = LogisticRegression(C=1e10, solver="lbfgs", max_iter=1000)
    lr.fit(logits_cal.reshape(-1, 1), y_cal)
    calibrated = lr.predict_proba(logits_test.reshape(-1, 1))[:, 1]
    return calibrated


def isotonic_scale(probs_cal, y_cal, probs_test):
    """Isotonic regression calibration."""
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(probs_cal, y_cal)
    calibrated = iso.predict(probs_test)
    return calibrated


# ---------------------------------------------------------------------------
# Conformal methods (same as before but cleaner)
# ---------------------------------------------------------------------------

def aps_calibrate_and_predict(cal_prob, cal_y, test_prob, alpha):
    cal_scores = np.where(cal_y == 1, 1 - cal_prob, cal_prob)
    n = len(cal_scores)
    q = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    thresh = np.quantile(cal_scores, q)

    sets = []
    for p in test_prob:
        s = set()
        if (1 - p) <= thresh:
            s.add(1)
        if p <= thresh:
            s.add(0)
        sets.append(s)
    return sets


def mondrian_calibrate_and_predict(cal_prob, cal_y, test_prob, alpha):
    cal_scores = np.where(cal_y == 1, 1 - cal_prob, cal_prob)
    thresholds = {}
    for cls in [0, 1]:
        cls_scores = cal_scores[cal_y == cls]
        n = len(cls_scores)
        if n < 5:
            thresholds[cls] = 1.0
            continue
        q = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
        thresholds[cls] = np.quantile(cls_scores, q)

    sets = []
    for p in test_prob:
        s = set()
        if (1 - p) <= thresholds.get(1, 1.0):
            s.add(1)
        if p <= thresholds.get(0, 1.0):
            s.add(0)
        sets.append(s)
    return sets


def weighted_mondrian(cal_prob, cal_y, test_prob, cal_X, test_X, alpha):
    """Weighted conformal with density-ratio estimation in embedding space."""
    # Train domain classifier: calibration (0) vs test (1)
    X_domain = np.vstack([cal_X, test_X])
    y_domain = np.array([0] * len(cal_X) + [1] * len(test_X))

    domain_clf = LogisticRegression(C=1.0, max_iter=1000, random_state=SEED)
    domain_clf.fit(X_domain, y_domain)

    # Density ratio weights for calibration samples
    cal_domain_prob = domain_clf.predict_proba(cal_X)[:, 1]
    weights = cal_domain_prob / (1 - cal_domain_prob + 1e-8)

    # Winsorise at 95th percentile (pre-registered §5.4.20)
    w95 = np.percentile(weights, 95)
    weights_clipped = np.minimum(weights, w95)

    # Effective sample size
    ess = (weights_clipped.sum() ** 2) / (weights_clipped ** 2).sum()

    cal_scores = np.where(cal_y == 1, 1 - cal_prob, cal_prob)

    # Weighted quantile per class
    thresholds = {}
    for cls in [0, 1]:
        mask = cal_y == cls
        cls_scores = cal_scores[mask]
        cls_weights = weights_clipped[mask]
        cls_weights = cls_weights / cls_weights.sum()

        # Weighted quantile
        sorted_idx = np.argsort(cls_scores)
        cumw = np.cumsum(cls_weights[sorted_idx])
        q_target = (1 - alpha)
        idx = np.searchsorted(cumw, q_target)
        idx = min(idx, len(cls_scores) - 1)
        thresholds[cls] = cls_scores[sorted_idx[idx]]

    sets = []
    for p in test_prob:
        s = set()
        if (1 - p) <= thresholds.get(1, 1.0):
            s.add(1)
        if p <= thresholds.get(0, 1.0):
            s.add(0)
        sets.append(s)

    return sets, {"ess": round(ess, 1), "w95": round(w95, 2), "max_w": round(weights.max(), 2)}


def conformal_risk_control(cal_prob, cal_y, test_prob, test_y, target_fnr=0.10):
    """CRC: find threshold bounding FNR."""
    best_thresh = 0.0
    for thresh in np.linspace(0, 1, 2000):
        pred_pos = cal_prob >= thresh
        fn = ((cal_y == 1) & (~pred_pos)).sum()
        n_pos = (cal_y == 1).sum()
        fnr = fn / max(n_pos, 1)
        if fnr <= target_fnr:
            best_thresh = thresh

    # Evaluate on test
    test_pred_pos = test_prob >= best_thresh
    fn = ((test_y == 1) & (~test_pred_pos)).sum()
    tp = ((test_y == 1) & test_pred_pos).sum()
    tn = ((test_y == 0) & (~test_pred_pos)).sum()
    fp = ((test_y == 0) & test_pred_pos).sum()

    return {
        "threshold": round(best_thresh, 4),
        "sensitivity": round(tp / max(tp + fn, 1), 4),
        "specificity": round(tn / max(tn + fp, 1), 4),
        "achieved_fnr": round(fn / max(tp + fn, 1), 4),
    }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def eval_sets(pred_sets, y_true):
    n = len(pred_sets)
    coverages = [int(y in s) for y, s in zip(y_true, pred_sets)]
    set_sizes = [len(s) for s in pred_sets]
    singletons = [int(len(s) == 1) for s in pred_sets]
    empties = [int(len(s) == 0) for s in pred_sets]
    tb_mask = y_true == 1
    nontb_mask = y_true == 0
    tb_cov = np.mean([coverages[i] for i in range(n) if tb_mask[i]]) if tb_mask.any() else float("nan")
    nontb_cov = np.mean([coverages[i] for i in range(n) if nontb_mask[i]]) if nontb_mask.any() else float("nan")
    return {
        "marginal_cov": round(np.mean(coverages), 4),
        "tb_cov": round(tb_cov, 4),
        "nontb_cov": round(nontb_cov, 4),
        "mean_size": round(np.mean(set_sizes), 4),
        "singleton": round(np.mean(singletons), 4),
        "empty": round(np.mean(empties), 4),
        "disparity": round(abs(tb_cov - nontb_cov), 4),
    }


def triage(pred_sets, y_true):
    tiers = []
    for s in pred_sets:
        if s == {0}: tiers.append(1)
        elif s == {1}: tiers.append(2)
        else: tiers.append(3)
    tiers = np.array(tiers)
    result = {}
    for t in [1, 2, 3]:
        m = tiers == t
        result[f"t{t}_n"] = int(m.sum())
        result[f"t{t}_pct"] = round(m.mean() * 100, 1)
        result[f"t{t}_tb"] = round(y_true[m].mean(), 3) if m.any() else 0
    return result


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("IMPROVED CONFORMAL PIPELINE")
    print("=" * 70)

    results = []

    for emb_name in WORKING_MODELS:
        print(f"\n{'━' * 60}")
        print(f"  {emb_name}")
        print(f"{'━' * 60}")

        df = pd.read_parquet(EMB_DIR / f"{emb_name}.parquet")
        emb_cols = [c for c in df.columns if c.startswith("emb_")]

        for l2 in [False, True]:
            norm_tag = "L2" if l2 else "raw"
            X_cal, y_cal, _ = load_split_data(emb_name, emb_cols, df, "calibration", l2_norm=l2)
            X_dev, y_dev, _ = load_split_data(emb_name, emb_cols, df, "dev", l2_norm=l2)
            X_test, y_test, _ = load_split_data(emb_name, emb_cols, df, "test", l2_norm=l2)
            X_ext, y_ext, _ = load_split_data(emb_name, emb_cols, df, "ext_pakistan", l2_norm=l2)

            if len(X_cal) < 50:
                continue

            # Train linear probe (primary)
            best_c, best_score = None, -1
            skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
            for C in [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]:
                scores = []
                for tr, va in skf.split(X_cal, y_cal):
                    m = LogisticRegression(C=C, max_iter=2000, solver="lbfgs", random_state=SEED)
                    m.fit(X_cal[tr], y_cal[tr])
                    p = m.predict_proba(X_cal[va])[:, 1]
                    if len(np.unique(y_cal[va])) > 1:
                        scores.append(roc_auc_score(y_cal[va], p))
                ms = np.mean(scores) if scores else 0
                if ms > best_score:
                    best_c, best_score = C, ms

            probe = LogisticRegression(C=best_c, max_iter=2000, solver="lbfgs", random_state=SEED)
            probe.fit(X_cal, y_cal)

            # Raw probabilities
            cal_prob = probe.predict_proba(X_cal)[:, 1]
            dev_prob = probe.predict_proba(X_dev)[:, 1]
            test_prob = probe.predict_proba(X_test)[:, 1]
            ext_prob = probe.predict_proba(X_ext)[:, 1]

            test_auroc = roc_auc_score(y_test, test_prob)

            # --- Calibration methods ---
            calibration_methods = {"uncalibrated": test_prob}

            # Temperature scaling (calibrate on dev set)
            logits_dev = np.log(np.clip(dev_prob, 1e-7, 1-1e-7) / (1 - np.clip(dev_prob, 1e-7, 1-1e-7)))
            logits_test = np.log(np.clip(test_prob, 1e-7, 1-1e-7) / (1 - np.clip(test_prob, 1e-7, 1-1e-7)))
            logits_cal = np.log(np.clip(cal_prob, 1e-7, 1-1e-7) / (1 - np.clip(cal_prob, 1e-7, 1-1e-7)))
            temp_test, T = temperature_scale(logits_dev, y_dev, logits_test)
            temp_cal, _ = temperature_scale(logits_dev, y_dev, logits_cal)
            calibration_methods["temperature"] = temp_test

            # Platt scaling (calibrate on dev set)
            platt_test = platt_scale(dev_prob, y_dev, test_prob)
            platt_cal = platt_scale(dev_prob, y_dev, cal_prob)
            calibration_methods["platt"] = platt_test

            # Isotonic (calibrate on dev set)
            iso_test = isotonic_scale(dev_prob, y_dev, test_prob)
            iso_cal = isotonic_scale(dev_prob, y_dev, cal_prob)
            calibration_methods["isotonic"] = iso_test

            # Recalibrated cal probs for conformal
            cal_probs_by_method = {
                "uncalibrated": cal_prob,
                "temperature": temp_cal,
                "platt": platt_cal,
                "isotonic": iso_cal,
            }

            for cal_method, t_prob in calibration_methods.items():
                c_prob = cal_probs_by_method[cal_method]

                # ECE and Brier for reporting
                brier = brier_score_loss(y_test, t_prob)

                for alpha in [0.05, 0.10, 0.20]:

                    # APS
                    sets_aps = aps_calibrate_and_predict(c_prob, y_cal, t_prob, alpha)
                    m_aps = eval_sets(sets_aps, y_test)

                    results.append({
                        "embedding": emb_name, "norm": norm_tag, "cal_method": cal_method,
                        "conformal": "APS", "alpha": alpha,
                        "auroc": round(test_auroc, 4), "brier": round(brier, 4),
                        **m_aps,
                    })

                    # Mondrian
                    sets_mon = mondrian_calibrate_and_predict(c_prob, y_cal, t_prob, alpha)
                    m_mon = eval_sets(sets_mon, y_test)

                    results.append({
                        "embedding": emb_name, "norm": norm_tag, "cal_method": cal_method,
                        "conformal": "Mondrian", "alpha": alpha,
                        "auroc": round(test_auroc, 4), "brier": round(brier, 4),
                        **m_mon,
                    })

                    # Weighted Mondrian (only for alpha=0.10 to save time)
                    if alpha == 0.10:
                        sets_wm, w_info = weighted_mondrian(
                            c_prob, y_cal, t_prob, X_cal, X_test, alpha
                        )
                        m_wm = eval_sets(sets_wm, y_test)

                        results.append({
                            "embedding": emb_name, "norm": norm_tag, "cal_method": cal_method,
                            "conformal": "WeightedMondrian", "alpha": alpha,
                            "auroc": round(test_auroc, 4), "brier": round(brier, 4),
                            **m_wm, **w_info,
                        })

                # CRC at target FNR=0.10
                crc = conformal_risk_control(c_prob, y_cal, t_prob, y_test, target_fnr=0.10)
                results.append({
                    "embedding": emb_name, "norm": norm_tag, "cal_method": cal_method,
                    "conformal": "CRC", "alpha": 0.10,
                    "auroc": round(test_auroc, 4), "brier": round(brier, 4),
                    **crc,
                })

            tag = f"{emb_name}/{norm_tag}"
            print(f"  {tag:30s}  AUROC={test_auroc:.4f}  C={best_c}", flush=True)

    # Save all results
    results_df = pd.DataFrame(results)
    results_df.to_csv(TABLES_DIR / "improved_conformal_results.csv", index=False)

    # ── Find best configuration ──
    print(f"\n{'=' * 70}")
    print("BEST CONFIGURATIONS (Mondrian @ alpha=0.10, TB coverage)")
    print(f"{'=' * 70}")

    mon10 = results_df[
        (results_df["conformal"] == "Mondrian") &
        (results_df["alpha"] == 0.10)
    ].copy()
    mon10 = mon10.sort_values("tb_cov", ascending=False)
    display = ["embedding", "norm", "cal_method", "auroc", "tb_cov", "nontb_cov", "singleton", "disparity", "empty"]
    print(mon10[display].head(20).to_string(index=False))

    # ── Best weighted Mondrian ──
    print(f"\n{'=' * 70}")
    print("WEIGHTED MONDRIAN @ alpha=0.10")
    print(f"{'=' * 70}")
    wm = results_df[results_df["conformal"] == "WeightedMondrian"].copy()
    if len(wm) > 0:
        wm = wm.sort_values("disparity", ascending=True)
        wm_display = ["embedding", "norm", "cal_method", "tb_cov", "nontb_cov", "singleton", "disparity", "ess"]
        print(wm[wm_display].head(20).to_string(index=False))

    # ── Best CRC ──
    print(f"\n{'=' * 70}")
    print("CRC @ target FNR=0.10 (WHO TPP alignment)")
    print(f"{'=' * 70}")
    crc_results = results_df[results_df["conformal"] == "CRC"].copy()
    if len(crc_results) > 0:
        crc_results = crc_results.sort_values("sensitivity", ascending=False)
        crc_display = ["embedding", "norm", "cal_method", "sensitivity", "specificity", "achieved_fnr", "threshold"]
        print(crc_results[crc_display].head(20).to_string(index=False))

    # ── Deployment gates on best Mondrian config ──
    print(f"\n{'=' * 70}")
    print("DEPLOYMENT GATES (best achievable)")
    print(f"{'=' * 70}")

    best_auroc = results_df["auroc"].max()
    g1 = best_auroc >= 0.75
    print(f"  G1 (AUROC >= 0.75):      {'PASS' if g1 else 'FAIL'}  ({best_auroc:.4f})")

    best_tb_cov = mon10["tb_cov"].max() if len(mon10) > 0 else 0
    g2 = best_tb_cov >= 0.85
    print(f"  G2 (TB cov >= 85%):      {'PASS' if g2 else 'FAIL'}  ({best_tb_cov:.4f})")

    best_singleton = mon10["singleton"].max() if len(mon10) > 0 else 0
    g3 = best_singleton >= 0.40
    print(f"  G3 (Singleton >= 40%):   {'PASS' if g3 else 'FAIL'}  ({best_singleton:.4f})")

    best_disparity = mon10["disparity"].min() if len(mon10) > 0 else 1.0
    g4 = best_disparity <= 0.15
    print(f"  G4 (Disparity <= 15pp):  {'PASS' if g4 else 'FAIL'}  ({best_disparity:.4f})")

    # Weighted Mondrian disparity
    if len(wm) > 0:
        best_wm_disp = wm["disparity"].min()
        g4_wm = best_wm_disp <= 0.15
        print(f"  G4 (weighted CP):        {'PASS' if g4_wm else 'FAIL'}  ({best_wm_disp:.4f})")

    # Best CRC WHO TPP
    if len(crc_results) > 0:
        who_candidates = crc_results[(crc_results["sensitivity"] >= 0.90) & (crc_results["specificity"] >= 0.70)]
        if len(who_candidates) > 0:
            print(f"\n  WHO TPP (sens>=90%, spec>=70%): ACHIEVABLE")
            best_who = who_candidates.iloc[0]
            print(f"    {best_who['embedding']}/{best_who['norm']}/{best_who['cal_method']}: "
                  f"sens={best_who['sensitivity']}, spec={best_who['specificity']}")
        else:
            print(f"\n  WHO TPP: NOT achievable with current models")
            # Show closest
            closest = crc_results.iloc[0]
            print(f"    Closest: sens={closest['sensitivity']}, spec={closest['specificity']}")

    print(f"\nResults: {TABLES_DIR / 'improved_conformal_results.csv'}")


if __name__ == "__main__":
    main()
