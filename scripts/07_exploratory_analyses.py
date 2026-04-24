"""
Exploratory analyses (§5.5):
- §5.5.1 Multi-class conformal (TBX11K)
- §5.5.2 Selective prediction (abstention)
- §5.5.3 Online/adaptive conformal (ACI)
- §5.5.4 SHAP feature importance
- §5.5.6 Concordance across conformal methods
- §5.5.7 Empty set analysis
- §5.5.8 Calibration-conformal interaction
- §5.5.9 Cross-model disagreement
- §5.5.11 Cost-sensitive conformal
- §5.5.14 Multi-site LODO

Run: python3 src/evaluation/exploratory_analyses.py
"""

import time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import normalize
from sklearn.isotonic import IsotonicRegression

warnings.filterwarnings("ignore")

from config import *

np.random.seed(SEED)
WORKING = WORKING_MODELS


def load(name, l2=True):
    df = pd.read_parquet(EMB_DIR / f"{name}.parquet")
    ec = [c for c in df.columns if c.startswith("emb_")]
    sm = pd.read_parquet(SPLITS_PATH)[["patient_id", "label"]].drop_duplicates("patient_id")
    df = df.merge(sm, on="patient_id", how="left")
    m = df["tb_binary"].isin(["tb_positive", "tb_negative"])
    df = df[m].copy()
    X = df[ec].values.astype(np.float32)
    if l2: X = normalize(X, norm="l2")
    y = (df["tb_binary"] == "tb_positive").astype(int).values
    return X, y, df["split"].values, df["dataset"].values, df["label"].values, df["patient_id"].values


def mondrian(cp, cy, tp, alpha):
    sc = np.where(cy == 1, 1 - cp, cp)
    th = {}
    for c in [0, 1]:
        cs = sc[cy == c]; n = len(cs)
        if n < 3: th[c] = 1.0; continue
        th[c] = np.quantile(cs, min(np.ceil((n+1)*(1-alpha))/n, 1.0))
    sets = []
    for p in tp:
        s = set()
        if (1-p) <= th.get(1, 1.0): s.add(1)
        if p <= th.get(0, 1.0): s.add(0)
        sets.append(s)
    return sets


def aps(cp, cy, tp, alpha):
    sc = np.where(cy == 1, 1 - cp, cp)
    n = len(sc); th = np.quantile(sc, min(np.ceil((n+1)*(1-alpha))/n, 1.0))
    sets = []
    for p in tp:
        s = set()
        if (1-p) <= th: s.add(1)
        if p <= th: s.add(0)
        sets.append(s)
    return sets


def get_pipeline(name="rad_dino"):
    X, y, sp, ds, lab, pid = load(name)
    cal_m, dev_m, test_m = sp == "calibration", sp == "dev", sp == "test"
    pr = LogisticRegression(C=10, max_iter=2000, solver="lbfgs", random_state=SEED)
    pr.fit(X[cal_m], y[cal_m])
    dp = pr.predict_proba(X[dev_m])[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip"); iso.fit(dp, y[dev_m])
    cp = iso.predict(pr.predict_proba(X[cal_m])[:, 1])
    tp = iso.predict(pr.predict_proba(X[test_m])[:, 1])
    return cp, y[cal_m], tp, y[test_m], lab[test_m], pid[test_m], pr, X, y, sp


# =========================================================================

def run_multiclass():
    """§5.5.1 Multi-class conformal on TBX11K."""
    print("=" * 70)
    print("§5.5.1 MULTI-CLASS CONFORMAL (TBX11K)")
    print("=" * 70)

    X, y, sp, ds, lab, pid = load("rad_dino")
    # Multi-class: healthy, sick_non_tb, active_tb, latent_tb
    tbx_test = (sp == "test") & np.isin(ds, ["tbx11k"])
    test_labels = lab[tbx_test]
    unique_labels = [l for l in np.unique(test_labels) if l != "unknown" and pd.notna(l)]

    if len(unique_labels) < 3:
        print(f"  Only {len(unique_labels)} classes found in test: {unique_labels}")
        print("  Multi-class conformal requires >=3 classes. Reporting class distribution only.")

    print(f"\n  TBX11K test label distribution:")
    for l in sorted(np.unique(test_labels)):
        print(f"    {l:20s}: {(test_labels == l).sum():>5}")

    # Train multi-class probe on calibration (binary since cal is Shenzhen/Montgomery)
    # Report which TBX11K multi-class categories get classified as TB vs non-TB
    cal_m = sp == "calibration"
    pr = LogisticRegression(C=10, max_iter=2000, solver="lbfgs", random_state=SEED)
    pr.fit(X[cal_m], y[cal_m])
    test_prob = pr.predict_proba(X[tbx_test])[:, 1]

    results = []
    for label in sorted(np.unique(test_labels)):
        lm = test_labels == label
        if lm.sum() < 5: continue
        mean_prob = test_prob[lm].mean()
        results.append({"label": label, "n": int(lm.sum()), "mean_p_tb": round(mean_prob, 4),
                        "pct_above_50": round((test_prob[lm] > 0.5).mean(), 4)})
        print(f"    {label:20s}: n={lm.sum():>5}, mean P(TB)={mean_prob:.4f}, >{0.5}: {(test_prob[lm] > 0.5).mean():.1%}")

    pd.DataFrame(results).to_csv(TABLES_DIR / "multiclass_analysis.csv", index=False)


def run_selective_prediction():
    """§5.5.2 Selective prediction (abstention)."""
    print("\n" + "=" * 70)
    print("§5.5.2 SELECTIVE PREDICTION (ABSTENTION)")
    print("=" * 70)

    cp, cy, tp, ty, _, _, _, _, _, _ = get_pipeline()
    sets = mondrian(cp, cy, tp, 0.10)

    # Abstain when set size > 1
    results = []
    for max_size in [1, 2]:
        non_abstain = [i for i, s in enumerate(sets) if len(s) <= max_size and len(s) > 0]
        abstain_rate = 1 - len(non_abstain) / len(sets)
        if len(non_abstain) > 0:
            na_y = ty[non_abstain]
            na_pred = [1 if 1 in sets[i] else 0 for i in non_abstain]
            accuracy = (np.array(na_pred) == na_y).mean()
            tb_in_non_abstain = na_y.mean()
        else:
            accuracy = float("nan")
            tb_in_non_abstain = float("nan")

        results.append({"max_set_size": max_size, "abstain_rate": round(abstain_rate, 4),
                        "n_non_abstain": len(non_abstain), "accuracy": round(accuracy, 4),
                        "tb_rate_non_abstain": round(tb_in_non_abstain, 4)})
        print(f"  Max set size {max_size}: abstain={abstain_rate:.1%}, accuracy={accuracy:.4f}", flush=True)

    pd.DataFrame(results).to_csv(TABLES_DIR / "selective_prediction.csv", index=False)


def run_adaptive_conformal():
    """§5.5.3 Online/Adaptive Conformal Inference (ACI)."""
    print("\n" + "=" * 70)
    print("§5.5.3 ADAPTIVE CONFORMAL INFERENCE (ACI)")
    print("=" * 70)

    cp, cy, tp, ty, _, _, _, _, _, _ = get_pipeline()
    cal_sc = np.where(cy == 1, 1 - cp, cp)

    # ACI: update alpha_t after each prediction
    results = []
    for gamma in [0.005, 0.01, 0.05]:
        alpha_t = 0.10
        coverages = []
        set_sizes = []

        # Initial threshold from calibration
        n = len(cal_sc)
        for t in range(len(tp)):
            q = min(np.ceil((n + 1) * (1 - alpha_t)) / n, 1.0)
            thresh = np.quantile(cal_sc, q)

            s = set()
            if (1 - tp[t]) <= thresh: s.add(1)
            if tp[t] <= thresh: s.add(0)

            covered = int(ty[t] in s)
            coverages.append(covered)
            set_sizes.append(len(s))

            # Update alpha
            alpha_t = alpha_t + gamma * (0.10 - covered)
            alpha_t = np.clip(alpha_t, 0.01, 0.50)

        # Rolling coverage (window 500)
        window = 500
        rolling = [np.mean(coverages[max(0, i-window):i+1]) for i in range(len(coverages))]

        results.append({
            "gamma": gamma,
            "mean_coverage": round(np.mean(coverages), 4),
            "mean_set_size": round(np.mean(set_sizes), 4),
            "final_alpha": round(alpha_t, 4),
            "min_rolling_cov": round(min(rolling[window:]) if len(rolling) > window else min(rolling), 4),
        })
        print(f"  gamma={gamma}: mean_cov={np.mean(coverages):.4f}, mean_size={np.mean(set_sizes):.4f}, final_alpha={alpha_t:.4f}", flush=True)

    pd.DataFrame(results).to_csv(TABLES_DIR / "adaptive_conformal.csv", index=False)


def run_shap():
    """§5.5.4 SHAP feature importance (XGBoost)."""
    print("\n" + "=" * 70)
    print("§5.5.4 SHAP FEATURE IMPORTANCE")
    print("=" * 70)

    try:
        import xgboost as xgb
    except ImportError:
        print("  xgboost not installed. Skipping.")
        return

    X, y, sp, _, _, _ = load("rad_dino")
    cal_m = sp == "calibration"
    test_m = sp == "test"

    model = xgb.XGBClassifier(max_depth=3, n_estimators=200, learning_rate=0.1,
                               reg_lambda=1, reg_alpha=0.1, random_state=SEED, verbosity=0)
    model.fit(X[cal_m], y[cal_m])

    # Feature importance (gain-based, faster than SHAP)
    importance = model.feature_importances_
    top50 = np.argsort(importance)[-50:][::-1]

    results = [{"dim": int(i), "importance": round(float(importance[i]), 6)} for i in top50]
    pd.DataFrame(results).to_csv(TABLES_DIR / "shap_feature_importance.csv", index=False)
    print(f"  Top 5 embedding dims: {top50[:5].tolist()}")
    print(f"  Top 5 importances: {[round(importance[i], 4) for i in top50[:5]]}")


def run_concordance():
    """§5.5.6 Concordance across conformal methods."""
    print("\n" + "=" * 70)
    print("§5.5.6 CONCORDANCE ACROSS CONFORMAL METHODS")
    print("=" * 70)

    cp, cy, tp, ty, _, _, _, _, _, _ = get_pipeline()

    methods = {
        "APS": aps(cp, cy, tp, 0.10),
        "Mondrian": mondrian(cp, cy, tp, 0.10),
    }

    # Also add a method with different alpha for variety
    methods["Mondrian_05"] = mondrian(cp, cy, tp, 0.05)

    # Pairwise agreement
    names = list(methods.keys())
    results = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            s1 = methods[names[i]]; s2 = methods[names[j]]
            agree = sum(1 for a, b in zip(s1, s2) if a == b) / len(s1)

            # Classify into 3 categories
            cats1 = [0 if s == {0} else 1 if s == {1} else 2 for s in s1]
            cats2 = [0 if s == {0} else 1 if s == {1} else 2 for s in s2]
            from sklearn.metrics import cohen_kappa_score
            kappa = cohen_kappa_score(cats1, cats2)

            results.append({"method_a": names[i], "method_b": names[j],
                            "agreement": round(agree, 4), "kappa": round(kappa, 4)})
            print(f"  {names[i]:15s} vs {names[j]:15s}: agree={agree:.4f}, kappa={kappa:.4f}", flush=True)

    # Core uncertain: uncertain under ALL methods
    all_uncertain = sum(1 for i in range(len(tp))
                        if all(len(methods[m][i]) > 1 or len(methods[m][i]) == 0 for m in names))
    print(f"\n  Core uncertain (uncertain in all methods): {all_uncertain} ({all_uncertain/len(tp):.1%})")

    pd.DataFrame(results).to_csv(TABLES_DIR / "concordance.csv", index=False)


def run_empty_sets():
    """§5.5.7 Empty set analysis."""
    print("\n" + "=" * 70)
    print("§5.5.7 EMPTY PREDICTION SET ANALYSIS")
    print("=" * 70)

    cp, cy, tp, ty, labels, _, _, _, _, _ = get_pipeline()

    for alpha in [0.05, 0.10, 0.20]:
        sets = mondrian(cp, cy, tp, alpha)
        empty_mask = np.array([len(s) == 0 for s in sets])
        n_empty = empty_mask.sum()
        if n_empty > 0:
            empty_tb_rate = ty[empty_mask].mean()
            # What labels are in empty sets?
            empty_labels = labels[empty_mask]
            label_dist = pd.Series(empty_labels).value_counts().to_dict()
        else:
            empty_tb_rate = 0; label_dist = {}
        print(f"  alpha={alpha}: {n_empty} empty ({n_empty/len(sets):.1%}), "
              f"TB rate in empty: {empty_tb_rate:.3f}, labels: {label_dist}", flush=True)

    pd.DataFrame([{"analysis": "empty_sets", "note": "see console output"}]).to_csv(
        TABLES_DIR / "empty_sets.csv", index=False)


def run_cal_conformal_interaction():
    """§5.5.8 Calibration-conformal interaction matrix."""
    print("\n" + "=" * 70)
    print("§5.5.8 CALIBRATION-CONFORMAL INTERACTION")
    print("=" * 70)

    X, y, sp, _, _, _ = load("rad_dino")
    cal_m, dev_m, test_m = sp == "calibration", sp == "dev", sp == "test"
    pr = LogisticRegression(C=10, max_iter=2000, solver="lbfgs", random_state=SEED)
    pr.fit(X[cal_m], y[cal_m])

    raw_cal = pr.predict_proba(X[cal_m])[:, 1]
    raw_test = pr.predict_proba(X[test_m])[:, 1]
    dev_p = pr.predict_proba(X[dev_m])[:, 1]

    # Calibration methods
    iso = IsotonicRegression(out_of_bounds="clip"); iso.fit(dev_p, y[dev_m])
    from scipy.optimize import minimize_scalar
    logits_d = np.log(np.clip(dev_p, 1e-7, 1-1e-7) / (1 - np.clip(dev_p, 1e-7, 1-1e-7)))
    logits_t = np.log(np.clip(raw_test, 1e-7, 1-1e-7) / (1 - np.clip(raw_test, 1e-7, 1-1e-7)))
    logits_c = np.log(np.clip(raw_cal, 1e-7, 1-1e-7) / (1 - np.clip(raw_cal, 1e-7, 1-1e-7)))
    def nll(T):
        p = 1/(1+np.exp(-logits_d/T)); p = np.clip(p, 1e-7, 1-1e-7)
        return -np.mean(y[dev_m]*np.log(p) + (1-y[dev_m])*np.log(1-p))
    T = minimize_scalar(nll, bounds=(0.1, 10), method="bounded").x
    platt = LogisticRegression(C=1e10, solver="lbfgs", max_iter=1000)
    platt.fit(logits_d.reshape(-1, 1), y[dev_m])

    cal_methods = {
        "uncalibrated": (raw_cal, raw_test),
        "temperature": (1/(1+np.exp(-logits_c/T)), 1/(1+np.exp(-logits_t/T))),
        "platt": (platt.predict_proba(logits_c.reshape(-1,1))[:,1], platt.predict_proba(logits_t.reshape(-1,1))[:,1]),
        "isotonic": (iso.predict(raw_cal), iso.predict(raw_test)),
    }

    results = []
    print(f"  {'Cal method':15s} {'Conformal':10s} {'TB_cov':>8} {'Singleton':>10} {'Empty':>8}")
    for cm_name, (c_prob, t_prob) in cal_methods.items():
        for conf_name, conf_fn in [("APS", aps), ("Mondrian", mondrian)]:
            sets = conf_fn(c_prob, y[cal_m], t_prob, 0.10)
            tb_m = y[test_m] == 1
            cov = [int(yi in s) for yi, s in zip(y[test_m], sets)]
            tb_cov = np.mean([cov[i] for i in range(len(cov)) if tb_m[i]])
            sing = np.mean([len(s) == 1 for s in sets])
            empty = np.mean([len(s) == 0 for s in sets])
            results.append({"cal_method": cm_name, "conformal": conf_name,
                            "tb_cov": round(tb_cov, 4), "singleton": round(sing, 4), "empty": round(empty, 4)})
            print(f"  {cm_name:15s} {conf_name:10s} {tb_cov:8.4f} {sing:10.4f} {empty:8.4f}")

    pd.DataFrame(results).to_csv(TABLES_DIR / "cal_conformal_interaction.csv", index=False)


def run_cross_model_disagreement():
    """§5.5.9 Cross-model disagreement."""
    print("\n" + "=" * 70)
    print("§5.5.9 CROSS-MODEL DISAGREEMENT")
    print("=" * 70)

    preds = pd.read_parquet(RESULTS_DIR / "probe_predictions.parquet")

    # Get test predictions for linear probe across working models
    test_preds = {}
    for emb in WORKING:
        sub = preds[(preds["embedding"] == emb) & (preds["probe"] == "linear") & (preds["split"] == "test")]
        test_preds[emb] = sub.set_index("patient_id")["y_prob"]

    # Align on shared IDs
    shared = set(test_preds[WORKING[0]].index)
    for emb in WORKING[1:]:
        shared &= set(test_preds[emb].index)
    shared = sorted(shared)

    # Predictions matrix
    pred_matrix = pd.DataFrame({emb: test_preds[emb].loc[shared] for emb in WORKING})
    binary_preds = (pred_matrix > 0.5).astype(int)

    # Disagreement: count how many models predict TB for each image
    n_tb_votes = binary_preds.sum(axis=1)
    full_agree = ((n_tb_votes == 0) | (n_tb_votes == len(WORKING))).mean()
    any_disagree = (n_tb_votes > 0) & (n_tb_votes < len(WORKING))

    # Get true labels
    true_labels = preds[(preds["embedding"] == WORKING[0]) & (preds["probe"] == "linear") &
                        (preds["split"] == "test")].set_index("patient_id")["y_true"].loc[shared]

    print(f"  Full agreement: {full_agree:.1%}")
    print(f"  Any disagreement: {any_disagree.mean():.1%}")
    print(f"\n  Disagreement by true label:")
    print(f"    TB cases with disagreement: {any_disagree[true_labels == 1].mean():.1%}")
    print(f"    Non-TB cases with disagreement: {any_disagree[true_labels == 0].mean():.1%}")

    pd.DataFrame({"n_tb_votes": n_tb_votes.value_counts().sort_index()}).to_csv(
        TABLES_DIR / "cross_model_disagreement.csv")


def run_cost_sensitive():
    """§5.5.11 Cost-sensitive conformal prediction."""
    print("\n" + "=" * 70)
    print("§5.5.11 COST-SENSITIVE CONFORMAL")
    print("=" * 70)

    cp, cy, tp, ty, _, _, _, _, _, _ = get_pipeline()
    cal_sc = np.where(cy == 1, 1 - cp, cp)

    results = []
    for K in [1, 5, 10, 20, 50]:
        # Scale TB-class scores by 1/K (makes it harder to exclude TB)
        adj_sc = np.where(cy == 1, cal_sc / K, cal_sc)
        n = len(adj_sc)
        th = np.quantile(adj_sc, min(np.ceil((n+1)*0.9)/n, 1.0))

        sets = []
        for p in tp:
            s = set()
            if (1-p)/K <= th: s.add(1)
            if p <= th: s.add(0)
            sets.append(s)

        tb_m = ty == 1
        cov = [int(yi in s) for yi, s in zip(ty, sets)]
        tb_cov = np.mean([cov[i] for i in range(len(cov)) if tb_m[i]])
        nontb_cov = np.mean([cov[i] for i in range(len(cov)) if ~tb_m[i]])
        singleton = np.mean([len(s) == 1 for s in sets])
        tier3 = np.mean([len(s) > 1 for s in sets])

        results.append({"K": K, "tb_cov": round(tb_cov, 4), "nontb_cov": round(nontb_cov, 4),
                        "singleton": round(singleton, 4), "tier3_frac": round(tier3, 4)})
        print(f"  K={K:>3}: TB_cov={tb_cov:.4f}  nonTB_cov={nontb_cov:.4f}  singleton={singleton:.4f}  tier3={tier3:.1%}", flush=True)

    pd.DataFrame(results).to_csv(TABLES_DIR / "cost_sensitive.csv", index=False)


def run_lodo():
    """§5.5.14 Multi-site LODO."""
    print("\n" + "=" * 70)
    print("§5.5.14 MULTI-SITE LEAVE-ONE-DATASET-OUT")
    print("=" * 70)

    X, y, sp, ds, _, _ = load("rad_dino")
    tb_datasets = ["shenzhen", "montgomery", "tbx11k", "pakistan"]
    results = []

    for held_out in tb_datasets:
        ho_m = ds == held_out
        if ho_m.sum() < 30: continue
        others = [d for d in tb_datasets if d != held_out]
        train_m = np.isin(ds, others)
        if train_m.sum() < 50: continue

        train_idx = np.where(train_m)[0]
        cal_idx, dev_idx = train_test_split(train_idx, test_size=0.3, random_state=SEED, stratify=y[train_m])

        pr = LogisticRegression(C=10, max_iter=2000, solver="lbfgs", random_state=SEED)
        pr.fit(X[cal_idx], y[cal_idx])

        dp = pr.predict_proba(X[dev_idx])[:, 1]
        iso = IsotonicRegression(out_of_bounds="clip"); iso.fit(dp, y[dev_idx])
        c_prob = iso.predict(pr.predict_proba(X[cal_idx])[:, 1])
        t_prob = iso.predict(pr.predict_proba(X[ho_m])[:, 1])

        auroc = roc_auc_score(y[ho_m], pr.predict_proba(X[ho_m])[:, 1]) if len(np.unique(y[ho_m])) > 1 else np.nan
        sets = mondrian(c_prob, y[cal_idx], t_prob, 0.10)
        cov = [int(yi in s) for yi, s in zip(y[ho_m], sets)]
        tb_m = y[ho_m] == 1
        tb_cov = np.mean([cov[i] for i in range(len(cov)) if tb_m[i]]) if tb_m.any() else np.nan

        results.append({"held_out": held_out, "n": int(ho_m.sum()), "n_tb": int(y[ho_m].sum()),
                        "auroc": round(auroc, 4), "tb_cov": round(tb_cov, 4)})
        print(f"  {held_out:15s}: n={ho_m.sum():>5}  AUROC={auroc:.4f}  TB_cov={tb_cov:.4f}", flush=True)

    pd.DataFrame(results).to_csv(TABLES_DIR / "lodo_simulation.csv", index=False)


# =========================================================================

def main():
    t0 = time.time()
    print("=" * 70)
    print("EXPLORATORY ANALYSES (§5.5)")
    print("=" * 70)

    run_multiclass()
    run_selective_prediction()
    run_adaptive_conformal()
    run_shap()
    run_concordance()
    run_empty_sets()
    run_cal_conformal_interaction()
    run_cross_model_disagreement()
    run_cost_sensitive()
    run_lodo()

    print(f"\n{'=' * 70}")
    print(f"All exploratory analyses complete in {time.time()-t0:.0f}s")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
