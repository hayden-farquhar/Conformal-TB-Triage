"""
Core analyses for manuscript tables and figures.
Runs: bootstrap CIs, fused embeddings, WHO TPP alignment, referral cascade,
conformal efficiency curves, embedding space analysis, exchangeability diagnostics,
calibration set size sensitivity, shortcut detection.

Run: python3 src/evaluation/core_analyses.py
"""

import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import normalize
from sklearn.isotonic import IsotonicRegression
from sklearn.manifold import TSNE
from scipy.stats import permutation_test

warnings.filterwarnings("ignore")

from config import *

np.random.seed(SEED)


def load_emb(name, l2=False):
    df = pd.read_parquet(EMB_DIR / f"{name}.parquet")
    ec = [c for c in df.columns if c.startswith("emb_")]
    mask = df["tb_binary"].isin(["tb_positive", "tb_negative"])
    df = df[mask].copy()
    X = df[ec].values.astype(np.float32)
    if l2:
        X = normalize(X, norm="l2")
    y = (df["tb_binary"] == "tb_positive").astype(int).values
    return X, y, df["split"].values, df["dataset"].values, df["patient_id"].values, ec


def train_linear(X, y, C=0.1):
    m = LogisticRegression(C=C, max_iter=2000, solver="lbfgs", random_state=SEED)
    m.fit(X, y)
    return m


def mondrian_sets(cal_prob, cal_y, test_prob, alpha):
    scores = np.where(cal_y == 1, 1 - cal_prob, cal_prob)
    thresholds = {}
    for cls in [0, 1]:
        cs = scores[cal_y == cls]
        n = len(cs)
        q = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
        thresholds[cls] = np.quantile(cs, q)
    sets = []
    for p in test_prob:
        s = set()
        if (1 - p) <= thresholds.get(1, 1.0): s.add(1)
        if p <= thresholds.get(0, 1.0): s.add(0)
        sets.append(s)
    return sets


def eval_conformal(sets, y):
    cov = [int(yi in s) for yi, s in zip(y, sets)]
    tb_m = y == 1
    nt_m = y == 0
    return {
        "tb_cov": np.mean([cov[i] for i in range(len(cov)) if tb_m[i]]),
        "nontb_cov": np.mean([cov[i] for i in range(len(cov)) if nt_m[i]]),
        "singleton": np.mean([len(s) == 1 for s in sets]),
        "mean_size": np.mean([len(s) for s in sets]),
    }


# =========================================================================
# 1. BOOTSTRAP CIs
# =========================================================================

def run_bootstrap_cis():
    print("\n" + "=" * 70)
    print("1. BOOTSTRAP 95% CIs (2000 resamples)")
    print("=" * 70)

    X, y, splits, datasets, pids, ec = load_emb("rad_dino", l2=True)
    cal_mask = splits == "calibration"
    test_mask = splits == "test"

    probe = train_linear(X[cal_mask], y[cal_mask], C=10)

    test_X, test_y = X[test_mask], y[test_mask]
    cal_X, cal_y = X[cal_mask], y[cal_mask]
    cal_prob = probe.predict_proba(cal_X)[:, 1]

    # Isotonic calibration on dev
    dev_mask = splits == "dev"
    dev_prob = probe.predict_proba(X[dev_mask])[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(dev_prob, y[dev_mask])
    cal_prob_iso = iso.predict(cal_prob)

    n_boot = 2000
    aurocs, auprcs, tb_covs, singletons, sensitivities, specificities = [], [], [], [], [], []

    for b in range(n_boot):
        idx = np.random.choice(len(test_y), len(test_y), replace=True)
        by, bX = test_y[idx], test_X[idx]
        if len(np.unique(by)) < 2:
            continue
        bp = probe.predict_proba(bX)[:, 1]
        bp_iso = iso.predict(bp)

        aurocs.append(roc_auc_score(by, bp))
        auprcs.append(average_precision_score(by, bp))

        # Conformal on bootstrap test
        sets = mondrian_sets(cal_prob_iso, cal_y, bp_iso, 0.10)
        m = eval_conformal(sets, by)
        tb_covs.append(m["tb_cov"])
        singletons.append(m["singleton"])

        # CRC sensitivity/specificity
        thresh = 0.032  # from best CRC result
        pred_pos = bp_iso >= thresh
        tp = ((by == 1) & pred_pos).sum()
        fn = ((by == 1) & ~pred_pos).sum()
        tn = ((by == 0) & ~pred_pos).sum()
        fp = ((by == 0) & pred_pos).sum()
        sensitivities.append(tp / max(tp + fn, 1))
        specificities.append(tn / max(tn + fp, 1))

        if (b + 1) % 500 == 0:
            print(f"  {b+1}/{n_boot} resamples done...", flush=True)

    def ci(arr):
        return f"{np.mean(arr):.4f} [{np.percentile(arr, 2.5):.4f}, {np.percentile(arr, 97.5):.4f}]"

    print(f"\n  RAD-DINO + linear + L2 + isotonic:")
    print(f"    AUROC:       {ci(aurocs)}")
    print(f"    AUPRC:       {ci(auprcs)}")
    print(f"    TB coverage: {ci(tb_covs)}")
    print(f"    Singleton:   {ci(singletons)}")
    print(f"    CRC sens:    {ci(sensitivities)}")
    print(f"    CRC spec:    {ci(specificities)}")

    ci_df = pd.DataFrame({
        "metric": ["AUROC", "AUPRC", "TB_coverage", "Singleton", "CRC_sensitivity", "CRC_specificity"],
        "mean": [np.mean(x) for x in [aurocs, auprcs, tb_covs, singletons, sensitivities, specificities]],
        "ci_lower": [np.percentile(x, 2.5) for x in [aurocs, auprcs, tb_covs, singletons, sensitivities, specificities]],
        "ci_upper": [np.percentile(x, 97.5) for x in [aurocs, auprcs, tb_covs, singletons, sensitivities, specificities]],
    })
    ci_df.to_csv(TABLES_DIR / "bootstrap_cis.csv", index=False)
    return ci_df


# =========================================================================
# 2. FUSED EMBEDDINGS
# =========================================================================

def run_fused_embeddings():
    print("\n" + "=" * 70)
    print("2. FUSED EMBEDDINGS")
    print("=" * 70)

    # Load all working models
    dfs = {}
    for name in WORKING_MODELS:
        df = pd.read_parquet(EMB_DIR / f"{name}.parquet")
        ec = [c for c in df.columns if c.startswith("emb_")]
        mask = df["tb_binary"].isin(["tb_positive", "tb_negative"])
        dfs[name] = df[mask].set_index("patient_id")[ec]

    # Align on shared patient IDs
    shared_ids = set(dfs[WORKING_MODELS[0]].index)
    for name in WORKING_MODELS[1:]:
        shared_ids &= set(dfs[name].index)
    shared_ids = sorted(shared_ids)
    print(f"  Shared patient IDs: {len(shared_ids)}", flush=True)

    # Get metadata from first model
    meta_df = pd.read_parquet(EMB_DIR / f"{WORKING_MODELS[0]}.parquet")
    meta_df = meta_df[meta_df["tb_binary"].isin(["tb_positive", "tb_negative"])]
    meta = meta_df.set_index("patient_id")[["split", "dataset", "tb_binary"]].loc[shared_ids]

    # Fuse: RAD-DINO + BiomedCLIP + torchxrayvision (top 3)
    fusions = {
        "rad_dino+biomedclip+txrv": ["rad_dino", "biomedclip", "torchxrayvision"],
        "rad_dino+biomedclip": ["rad_dino", "biomedclip"],
        "all4": WORKING_MODELS,
    }

    results = []
    for fuse_name, models in fusions.items():
        X_fused = np.hstack([dfs[m].loc[shared_ids].values for m in models])
        X_fused = normalize(X_fused, norm="l2")
        y = (meta["tb_binary"] == "tb_positive").astype(int).values
        splits = meta["split"].values

        cal_m = splits == "calibration"
        dev_m = splits == "dev"
        test_m = splits == "test"
        ext_m = splits == "ext_pakistan"

        # Train probe
        best_c, best_s = None, -1
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        for C in [1e-3, 1e-2, 0.1, 1, 10]:
            ss = []
            for tr, va in skf.split(X_fused[cal_m], y[cal_m]):
                m = LogisticRegression(C=C, max_iter=2000, solver="lbfgs", random_state=SEED)
                m.fit(X_fused[cal_m][tr], y[cal_m][tr])
                p = m.predict_proba(X_fused[cal_m][va])[:, 1]
                if len(np.unique(y[cal_m][va])) > 1:
                    ss.append(roc_auc_score(y[cal_m][va], p))
            ms = np.mean(ss) if ss else 0
            if ms > best_s:
                best_c, best_s = C, ms

        probe = LogisticRegression(C=best_c, max_iter=2000, solver="lbfgs", random_state=SEED)
        probe.fit(X_fused[cal_m], y[cal_m])

        for split_name, mask in [("test", test_m), ("ext_pakistan", ext_m)]:
            if mask.sum() == 0:
                continue
            prob = probe.predict_proba(X_fused[mask])[:, 1]
            auroc = roc_auc_score(y[mask], prob)
            auprc = average_precision_score(y[mask], prob)

            # Conformal (isotonic calibration via dev)
            dev_prob = probe.predict_proba(X_fused[dev_m])[:, 1]
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(dev_prob, y[dev_m])
            cal_prob_iso = iso.predict(probe.predict_proba(X_fused[cal_m])[:, 1])
            test_prob_iso = iso.predict(prob)
            sets = mondrian_sets(cal_prob_iso, y[cal_m], test_prob_iso, 0.10)
            cm = eval_conformal(sets, y[mask])

            results.append({
                "fusion": fuse_name, "dims": X_fused.shape[1], "split": split_name,
                "auroc": round(auroc, 4), "auprc": round(auprc, 4),
                "cv_auroc": round(best_s, 4), "C": best_c,
                **{k: round(v, 4) for k, v in cm.items()},
            })
            print(f"  {fuse_name:35s}  {split_name:15s}  AUROC={auroc:.4f}  TB_cov={cm['tb_cov']:.4f}", flush=True)

    fused_df = pd.DataFrame(results)
    fused_df.to_csv(TABLES_DIR / "fused_embeddings.csv", index=False)
    return fused_df


# =========================================================================
# 3. WHO TPP ALIGNMENT
# =========================================================================

def run_who_tpp():
    print("\n" + "=" * 70)
    print("3. WHO TPP ALIGNMENT (prevalence-adjusted PPV/NPV)")
    print("=" * 70)

    X, y, splits, _, _, _ = load_emb("rad_dino", l2=True)
    cal_m, dev_m, test_m = splits == "calibration", splits == "dev", splits == "test"

    probe = train_linear(X[cal_m], y[cal_m], C=10)
    test_prob = probe.predict_proba(X[test_m])[:, 1]
    test_y = y[test_m]

    # Find operating points
    auroc = roc_auc_score(test_y, test_prob)

    # At fixed sensitivity = 90%
    thresholds = np.sort(test_prob)
    results = []
    for prev in [0.001, 0.005, 0.01, 0.05, 0.10, 0.20]:
        # Find threshold for 90% sensitivity
        best_t_sens = 0
        for t in np.linspace(0, 1, 2000):
            tp = ((test_y == 1) & (test_prob >= t)).sum()
            fn = ((test_y == 1) & (test_prob < t)).sum()
            sens = tp / max(tp + fn, 1)
            if sens >= 0.90:
                best_t_sens = t

        tp = ((test_y == 1) & (test_prob >= best_t_sens)).sum()
        fn = ((test_y == 1) & (test_prob < best_t_sens)).sum()
        tn = ((test_y == 0) & (test_prob < best_t_sens)).sum()
        fp = ((test_y == 0) & (test_prob >= best_t_sens)).sum()
        sens = tp / max(tp + fn, 1)
        spec = tn / max(tn + fp, 1)

        # Prevalence-adjusted PPV and NPV
        ppv = (sens * prev) / (sens * prev + (1 - spec) * (1 - prev))
        npv = (spec * (1 - prev)) / (spec * (1 - prev) + (1 - sens) * prev)
        nns = 1 / (prev * sens) if prev * sens > 0 else float("inf")

        results.append({
            "prevalence": prev, "threshold": round(best_t_sens, 4),
            "sensitivity": round(sens, 4), "specificity": round(spec, 4),
            "ppv": round(ppv, 4), "npv": round(npv, 4),
            "nns": round(nns, 1),
        })

    who_df = pd.DataFrame(results)
    who_df.to_csv(TABLES_DIR / "who_tpp_alignment.csv", index=False)
    print(who_df.to_string(index=False))
    return who_df


# =========================================================================
# 4. REFERRAL CASCADE
# =========================================================================

def run_referral_cascade():
    print("\n" + "=" * 70)
    print("4. REFERRAL CASCADE MODEL (per 10,000 screened)")
    print("=" * 70)

    # Best CRC operating point: RAD-DINO/L2/isotonic
    # sensitivity=0.941, specificity=0.584
    # Also compare to no-AI and symptom-based

    N = 10000
    xpert_sens, xpert_spec = 0.987, 0.984

    scenarios = {
        "A: No AI (universal Xpert)": {"ai_sens": 1.0, "ai_spec": 0.0},
        "B: Symptom screen": {"ai_sens": 0.60, "ai_spec": 0.70},
        "C: AI triage (conformal)": {"ai_sens": 0.941, "ai_spec": 0.584},
        "D: AI at WHO TPP minimum": {"ai_sens": 0.90, "ai_spec": 0.70},
    }

    results = []
    for prev in [0.001, 0.005, 0.01, 0.05, 0.10, 0.20]:
        n_tb = int(N * prev)
        n_nontb = N - n_tb

        for scenario_name, params in scenarios.items():
            ai_sens = params["ai_sens"]
            ai_spec = params["ai_spec"]

            # AI triage stage
            ai_pos = int(n_tb * ai_sens) + int(n_nontb * (1 - ai_spec))
            ai_tp = int(n_tb * ai_sens)
            ai_fn = n_tb - ai_tp

            # Xpert stage (only AI-positives get Xpert)
            xpert_tp = int(ai_tp * xpert_sens)
            xpert_fp = int((ai_pos - ai_tp) * (1 - xpert_spec))
            tb_detected = xpert_tp
            tb_missed = n_tb - tb_detected
            false_referrals = ai_pos - ai_tp
            cartridges = ai_pos
            cartridges_per_case = cartridges / max(tb_detected, 1)
            cartridges_saved = N - cartridges  # vs universal

            results.append({
                "scenario": scenario_name, "prevalence": prev,
                "ai_referrals": ai_pos, "tb_detected": tb_detected,
                "tb_missed": tb_missed, "false_referrals": false_referrals,
                "xpert_cartridges": cartridges,
                "cartridges_per_case": round(cartridges_per_case, 1),
                "cartridges_saved": cartridges_saved,
            })

    cascade_df = pd.DataFrame(results)
    cascade_df.to_csv(TABLES_DIR / "referral_cascade.csv", index=False)

    # Print summary at 1% prevalence
    print("\n  At 1% prevalence (per 10,000 screened):")
    sub = cascade_df[cascade_df["prevalence"] == 0.01]
    print(sub[["scenario", "ai_referrals", "tb_detected", "tb_missed", "xpert_cartridges", "cartridges_per_case"]].to_string(index=False))
    return cascade_df


# =========================================================================
# 5. CONFORMAL EFFICIENCY CURVES
# =========================================================================

def run_efficiency_curves():
    print("\n" + "=" * 70)
    print("5. CONFORMAL EFFICIENCY CURVES")
    print("=" * 70)

    X, y, splits, _, _, _ = load_emb("rad_dino", l2=True)
    cal_m, dev_m, test_m = splits == "calibration", splits == "dev", splits == "test"

    probe = train_linear(X[cal_m], y[cal_m], C=10)
    dev_prob = probe.predict_proba(X[dev_m])[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(dev_prob, y[dev_m])

    cal_prob = iso.predict(probe.predict_proba(X[cal_m])[:, 1])
    test_prob = iso.predict(probe.predict_proba(X[test_m])[:, 1])

    results = []
    alphas = np.arange(0.01, 0.51, 0.01)
    for alpha in alphas:
        sets = mondrian_sets(cal_prob, y[cal_m], test_prob, alpha)
        m = eval_conformal(sets, y[test_m])
        results.append({"alpha": round(alpha, 2), **{k: round(v, 4) for k, v in m.items()}})

    eff_df = pd.DataFrame(results)
    eff_df.to_csv(TABLES_DIR / "efficiency_curves.csv", index=False)

    # Find clinical utility threshold (singleton > 80%)
    above_80 = eff_df[eff_df["singleton"] >= 0.80]
    if len(above_80) > 0:
        util_alpha = above_80.iloc[0]["alpha"]
        util_cov = above_80.iloc[0]["tb_cov"]
        print(f"  Clinical utility threshold: alpha={util_alpha} (TB_cov={util_cov:.3f}, singleton={above_80.iloc[0]['singleton']:.3f})")
    else:
        print("  Clinical utility threshold: singleton never reaches 80%")

    print(f"  At alpha=0.10: TB_cov={eff_df[eff_df['alpha']==0.10].iloc[0]['tb_cov']:.4f}, "
          f"singleton={eff_df[eff_df['alpha']==0.10].iloc[0]['singleton']:.4f}")
    print(f"  At alpha=0.05: TB_cov={eff_df[eff_df['alpha']==0.05].iloc[0]['tb_cov']:.4f}, "
          f"singleton={eff_df[eff_df['alpha']==0.05].iloc[0]['singleton']:.4f}")
    return eff_df


# =========================================================================
# 6. SHORTCUT DETECTION
# =========================================================================

def run_shortcut_detection():
    print("\n" + "=" * 70)
    print("6. SHORTCUT DETECTION (dataset vs TB classification)")
    print("=" * 70)

    results = []
    for emb_name in WORKING_MODELS:
        df = pd.read_parquet(EMB_DIR / f"{emb_name}.parquet")
        ec = [c for c in df.columns if c.startswith("emb_")]
        X = normalize(df[ec].values.astype(np.float32), norm="l2")

        # Dataset classification AUROC (one-vs-rest for multi-class)
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        dataset_labels = le.fit_transform(df["dataset"].values)
        n_classes = len(le.classes_)

        # Use logistic regression with CV
        from sklearn.multiclass import OneVsRestClassifier
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
        dataset_aurocs = []
        for tr, va in skf.split(X, dataset_labels):
            clf = OneVsRestClassifier(LogisticRegression(C=1, max_iter=1000, random_state=SEED))
            clf.fit(X[tr], dataset_labels[tr])
            pred = clf.predict_proba(X[va])
            try:
                from sklearn.metrics import roc_auc_score
                a = roc_auc_score(dataset_labels[va], pred, multi_class="ovr", average="macro")
                dataset_aurocs.append(a)
            except:
                pass
        dataset_auroc = np.mean(dataset_aurocs) if dataset_aurocs else 0

        # TB classification AUROC (on labelled subset)
        mask = df["tb_binary"].isin(["tb_positive", "tb_negative"])
        X_tb = X[mask]
        y_tb = (df[mask]["tb_binary"] == "tb_positive").astype(int).values
        tb_aurocs = []
        for tr, va in skf.split(X_tb, y_tb):
            clf = LogisticRegression(C=1, max_iter=1000, random_state=SEED)
            clf.fit(X_tb[tr], y_tb[tr])
            p = clf.predict_proba(X_tb[va])[:, 1]
            if len(np.unique(y_tb[va])) > 1:
                tb_aurocs.append(roc_auc_score(y_tb[va], p))
        tb_auroc = np.mean(tb_aurocs) if tb_aurocs else 0

        ratio = dataset_auroc / max(tb_auroc, 0.01)
        flag = "CRITICAL" if ratio > 1.0 else "OK"

        results.append({
            "embedding": emb_name,
            "dataset_auroc": round(dataset_auroc, 4),
            "tb_auroc": round(tb_auroc, 4),
            "shortcut_ratio": round(ratio, 4),
            "flag": flag,
        })
        print(f"  {emb_name:20s}  dataset={dataset_auroc:.4f}  TB={tb_auroc:.4f}  ratio={ratio:.4f}  {flag}", flush=True)

    shortcut_df = pd.DataFrame(results)
    shortcut_df.to_csv(TABLES_DIR / "shortcut_detection.csv", index=False)
    return shortcut_df


# =========================================================================
# 7. EXCHANGEABILITY DIAGNOSTICS
# =========================================================================

def run_exchangeability():
    print("\n" + "=" * 70)
    print("7. EXCHANGEABILITY DIAGNOSTICS")
    print("=" * 70)

    X, y, splits, datasets, _, _ = load_emb("rad_dino", l2=True)
    cal_m = splits == "calibration"
    test_m = splits == "test"

    probe = train_linear(X[cal_m], y[cal_m], C=10)

    cal_scores = np.where(y[cal_m] == 1, 1 - probe.predict_proba(X[cal_m])[:, 1],
                          probe.predict_proba(X[cal_m])[:, 1])
    test_scores = np.where(y[test_m] == 1, 1 - probe.predict_proba(X[test_m])[:, 1],
                           probe.predict_proba(X[test_m])[:, 1])

    # Two-sample KS test
    from scipy.stats import ks_2samp
    ks_stat, ks_p = ks_2samp(cal_scores, test_scores)
    print(f"  KS test (cal vs test nonconformity scores):")
    print(f"    Statistic: {ks_stat:.4f}, p-value: {ks_p:.6f}")
    if ks_p < 0.05:
        print(f"    Exchangeability REJECTED (p < 0.05) → weighted CP motivated")
    else:
        print(f"    Exchangeability not rejected (p >= 0.05)")

    # Embedding space: KS test on L2 norms
    cal_norms = np.linalg.norm(X[cal_m], axis=1)
    test_norms = np.linalg.norm(X[test_m], axis=1)
    ks2_stat, ks2_p = ks_2samp(cal_norms, test_norms)
    print(f"\n  KS test (embedding L2 norms):")
    print(f"    Cal mean={cal_norms.mean():.4f}, Test mean={test_norms.mean():.4f}")
    print(f"    Statistic: {ks2_stat:.4f}, p-value: {ks2_p:.6f}")

    return {"ks_scores_stat": ks_stat, "ks_scores_p": ks_p,
            "ks_norms_stat": ks2_stat, "ks_norms_p": ks2_p}


# =========================================================================
# 8. CALIBRATION SET SIZE SENSITIVITY
# =========================================================================

def run_calset_sensitivity():
    print("\n" + "=" * 70)
    print("8. CALIBRATION SET SIZE SENSITIVITY")
    print("=" * 70)

    X, y, splits, _, _, _ = load_emb("rad_dino", l2=True)
    cal_m = splits == "calibration"
    dev_m = splits == "dev"
    test_m = splits == "test"

    X_cal, y_cal = X[cal_m], y[cal_m]
    X_dev, y_dev = X[dev_m], y[dev_m]
    X_test, y_test = X[test_m], y[test_m]

    results = []
    fractions = [0.10, 0.20, 0.30, 0.50, 0.70, 1.00]

    for frac in fractions:
        n_sub = max(int(len(X_cal) * frac), 20)
        # Repeat 10 times with different subsamples
        tb_covs, singletons = [], []
        for rep in range(10):
            rng = np.random.RandomState(SEED + rep)
            # Stratified subsample
            idx_0 = np.where(y_cal == 0)[0]
            idx_1 = np.where(y_cal == 1)[0]
            n0 = max(int(n_sub * len(idx_0) / len(y_cal)), 5)
            n1 = max(int(n_sub * len(idx_1) / len(y_cal)), 5)
            sub_idx = np.concatenate([
                rng.choice(idx_0, min(n0, len(idx_0)), replace=False),
                rng.choice(idx_1, min(n1, len(idx_1)), replace=False),
            ])

            probe = train_linear(X_cal[sub_idx], y_cal[sub_idx], C=10)
            dev_prob = probe.predict_proba(X_dev)[:, 1]
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(dev_prob, y_dev)

            cal_prob = iso.predict(probe.predict_proba(X_cal[sub_idx])[:, 1])
            test_prob = iso.predict(probe.predict_proba(X_test)[:, 1])
            sets = mondrian_sets(cal_prob, y_cal[sub_idx], test_prob, 0.10)
            m = eval_conformal(sets, y_test)
            tb_covs.append(m["tb_cov"])
            singletons.append(m["singleton"])

        results.append({
            "fraction": frac, "n_cal": n_sub,
            "tb_cov_mean": round(np.mean(tb_covs), 4),
            "tb_cov_std": round(np.std(tb_covs), 4),
            "singleton_mean": round(np.mean(singletons), 4),
            "singleton_std": round(np.std(singletons), 4),
        })
        print(f"  {frac:.0%} ({n_sub:>4} images): TB_cov={np.mean(tb_covs):.4f} ± {np.std(tb_covs):.4f}, "
              f"singleton={np.mean(singletons):.4f} ± {np.std(singletons):.4f}", flush=True)

    calset_df = pd.DataFrame(results)
    # In-sample arm (subsamples the probe-training split). The authoritative
    # held-out calset_sensitivity.csv is written by conformal_sensitivity.py.
    calset_df.to_csv(TABLES_DIR / "calset_sensitivity_insample.csv", index=False)

    # Find minimum n for stable coverage
    stable = calset_df[calset_df["tb_cov_std"] < 0.02]
    if len(stable) > 0:
        min_n = stable.iloc[0]["n_cal"]
        print(f"\n  Minimum n for stable coverage (SD < 0.02): {min_n}")
    return calset_df


# =========================================================================
# MAIN
# =========================================================================

def main():
    t0 = time.time()
    print("=" * 70)
    print("CORE ANALYSES — Conformal TB Triage")
    print("=" * 70)

    run_bootstrap_cis()
    run_fused_embeddings()
    run_who_tpp()
    run_referral_cascade()
    run_efficiency_curves()
    run_shortcut_detection()
    run_exchangeability()
    run_calset_sensitivity()

    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"All core analyses complete in {elapsed:.0f}s")
    print(f"Results in: {TABLES_DIR}/")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
