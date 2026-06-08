"""
Tier 2 sensitivity analyses (§5.4):
- Seed stability (5 seeds)
- Meta-coverage (200 resplits)
- Label noise sensitivity
- Latent TB exclusion
- Prevalence mismatch
- Non-TB confusion matrix

Run: python3 src/evaluation/sensitivity_analyses.py
"""

import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import normalize
from sklearn.isotonic import IsotonicRegression

warnings.filterwarnings("ignore")

from config import *


def load_primary():
    """Load RAD-DINO L2-normalised embeddings with labels from splits."""
    df = pd.read_parquet(EMB_DIR / "rad_dino.parquet")
    ec = [c for c in df.columns if c.startswith("emb_")]

    # Join multi-class label from splits.parquet
    splits_meta = pd.read_parquet(SPLITS_PATH)[["patient_id", "label"]].drop_duplicates("patient_id")
    df = df.merge(splits_meta, on="patient_id", how="left")

    mask = df["tb_binary"].isin(["tb_positive", "tb_negative"])
    df = df[mask].copy()
    X = normalize(df[ec].values.astype(np.float32), norm="l2")
    y = (df["tb_binary"] == "tb_positive").astype(int).values
    return X, y, df["split"].values, df["label"].values, df["patient_id"].values


def train_and_evaluate(X_cal, y_cal, X_dev, y_dev, X_test, y_test, alpha=0.10):
    """Full pipeline: train probe, isotonic calibrate, Mondrian conformal."""
    probe = LogisticRegression(C=10, max_iter=2000, solver="lbfgs", random_state=SEED)
    probe.fit(X_cal, y_cal)

    dev_prob = probe.predict_proba(X_dev)[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(dev_prob, y_dev)

    cal_prob = iso.predict(probe.predict_proba(X_cal)[:, 1])
    test_prob_raw = probe.predict_proba(X_test)[:, 1]
    test_prob = iso.predict(test_prob_raw)

    auroc = roc_auc_score(y_test, test_prob_raw) if len(np.unique(y_test)) > 1 else float("nan")

    # Mondrian conformal
    scores = np.where(y_cal == 1, 1 - cal_prob, cal_prob)
    thresholds = {}
    for cls in [0, 1]:
        cs = scores[y_cal == cls]
        n = len(cs)
        if n < 3:
            thresholds[cls] = 1.0
            continue
        q = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
        thresholds[cls] = np.quantile(cs, q)

    sets = []
    for p in test_prob:
        s = set()
        if (1 - p) <= thresholds.get(1, 1.0): s.add(1)
        if p <= thresholds.get(0, 1.0): s.add(0)
        sets.append(s)

    cov = [int(yi in s) for yi, s in zip(y_test, sets)]
    tb_m = y_test == 1
    tb_cov = np.mean([cov[i] for i in range(len(cov)) if tb_m[i]]) if tb_m.any() else float("nan")
    singleton = np.mean([len(s) == 1 for s in sets])

    return {"auroc": auroc, "tb_cov": tb_cov, "singleton": singleton}


# =========================================================================
# 1. SEED STABILITY (§5.4.10)
# =========================================================================

def run_seed_stability():
    print("=" * 70)
    print("1. SEED STABILITY (§5.4.10) — 5 seeds")
    print("=" * 70)

    X, y, splits, labels, pids = load_primary()

    # For seed stability, we re-split the TBX11K portion with different seeds
    cal_m = splits == "calibration"
    tbx_m = (splits == "dev") | (splits == "test")

    X_cal, y_cal = X[cal_m], y[cal_m]
    X_tbx, y_tbx = X[tbx_m], y[tbx_m]

    seeds = [42, 123, 456, 789, 1024]
    results = []

    for seed in seeds:
        # Re-split TBX11K into dev/test with this seed
        idx_dev, idx_test = train_test_split(
            np.arange(len(y_tbx)), test_size=0.7, random_state=seed, stratify=y_tbx
        )
        m = train_and_evaluate(X_cal, y_cal, X_tbx[idx_dev], y_tbx[idx_dev],
                               X_tbx[idx_test], y_tbx[idx_test])
        results.append({"seed": seed, **m})
        print(f"  Seed {seed}: AUROC={m['auroc']:.4f}  TB_cov={m['tb_cov']:.4f}  singleton={m['singleton']:.4f}", flush=True)

    df = pd.DataFrame(results)
    print(f"\n  Mean ± SD:")
    print(f"    AUROC:     {df['auroc'].mean():.4f} ± {df['auroc'].std():.4f}")
    print(f"    TB_cov:    {df['tb_cov'].mean():.4f} ± {df['tb_cov'].std():.4f}")
    print(f"    Singleton: {df['singleton'].mean():.4f} ± {df['singleton'].std():.4f}")

    auroc_pass = df["auroc"].std() < 0.01
    cov_pass = df["tb_cov"].std() < 0.02
    print(f"\n  AUROC SD < 0.01: {'PASS' if auroc_pass else 'FAIL'} ({df['auroc'].std():.4f})")
    print(f"  Coverage SD < 0.02: {'PASS' if cov_pass else 'FAIL'} ({df['tb_cov'].std():.4f})")

    df.to_csv(TABLES_DIR / "seed_stability.csv", index=False)
    return df


# =========================================================================
# 2. META-COVERAGE (§5.4.11) — 200 resplits
# =========================================================================

def run_meta_coverage():
    print("\n" + "=" * 70)
    print("2. META-COVERAGE (§5.4.11) — 200 resplits")
    print("=" * 70)

    X, y, splits, _, _ = load_primary()
    cal_m = splits == "calibration"
    dev_m = splits == "dev"
    test_m = splits == "test"

    # Pool calibration + test for resplitting
    pool_m = cal_m | test_m
    X_pool, y_pool = X[pool_m], y[pool_m]
    X_dev, y_dev = X[dev_m], y[dev_m]
    n_cal = cal_m.sum()

    coverages = []
    n_resplits = 200

    for i in range(n_resplits):
        rng = np.random.RandomState(SEED + i)
        # Stratified resplit maintaining same cal size
        idx = np.arange(len(y_pool))
        idx_cal, idx_test = train_test_split(
            idx, train_size=n_cal, random_state=SEED + i, stratify=y_pool
        )

        m = train_and_evaluate(
            X_pool[idx_cal], y_pool[idx_cal],
            X_dev, y_dev,
            X_pool[idx_test], y_pool[idx_test],
        )
        coverages.append(m["tb_cov"])

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{n_resplits} resplits done (mean TB_cov so far: {np.mean(coverages):.4f})", flush=True)

    coverages = np.array(coverages)
    meta_cov_90 = (coverages >= 0.90).mean()
    meta_cov_85 = (coverages >= 0.85).mean()

    print(f"\n  Meta-coverage results:")
    print(f"    Mean TB coverage: {coverages.mean():.4f} ± {coverages.std():.4f}")
    print(f"    Fraction achieving >=90%: {meta_cov_90:.4f} (target: >=0.90)")
    print(f"    Fraction achieving >=85%: {meta_cov_85:.4f}")
    print(f"    Min: {coverages.min():.4f}, Max: {coverages.max():.4f}")

    if meta_cov_90 < 0.85:
        print(f"    WARNING: Meta-coverage below 0.85 — systematic validity concern")

    pd.DataFrame({"resplit": range(n_resplits), "tb_coverage": coverages}).to_csv(
        TABLES_DIR / "meta_coverage.csv", index=False
    )
    return coverages


# =========================================================================
# 3. LABEL NOISE SENSITIVITY (§5.4.12)
# =========================================================================

def run_label_noise():
    print("\n" + "=" * 70)
    print("3. LABEL NOISE SENSITIVITY (§5.4.12)")
    print("=" * 70)

    X, y, splits, _, _ = load_primary()
    cal_m = splits == "calibration"
    dev_m = splits == "dev"
    test_m = splits == "test"

    X_cal, y_cal = X[cal_m], y[cal_m]
    X_dev, y_dev = X[dev_m], y[dev_m]
    X_test, y_test = X[test_m], y[test_m]

    noise_fractions = [0.00, 0.05, 0.10, 0.15, 0.20]
    n_reps = 10
    results = []

    for frac in noise_fractions:
        aurocs, covs, singletons = [], [], []
        for rep in range(n_reps):
            rng = np.random.RandomState(SEED + rep)
            y_noisy = y_cal.copy()
            n_flip = int(len(y_noisy) * frac)
            if n_flip > 0:
                flip_idx = rng.choice(len(y_noisy), n_flip, replace=False)
                y_noisy[flip_idx] = 1 - y_noisy[flip_idx]

            m = train_and_evaluate(X_cal, y_noisy, X_dev, y_dev, X_test, y_test)
            aurocs.append(m["auroc"])
            covs.append(m["tb_cov"])
            singletons.append(m["singleton"])

        results.append({
            "noise_frac": frac,
            "auroc_mean": round(np.mean(aurocs), 4), "auroc_std": round(np.std(aurocs), 4),
            "tb_cov_mean": round(np.mean(covs), 4), "tb_cov_std": round(np.std(covs), 4),
            "singleton_mean": round(np.mean(singletons), 4),
        })
        print(f"  {frac:.0%} noise: AUROC={np.mean(aurocs):.4f}±{np.std(aurocs):.4f}  "
              f"TB_cov={np.mean(covs):.4f}±{np.std(covs):.4f}", flush=True)

    df = pd.DataFrame(results)
    df.to_csv(TABLES_DIR / "label_noise.csv", index=False)

    # Find tolerance threshold
    baseline_cov = df[df["noise_frac"] == 0.00]["tb_cov_mean"].values[0]
    below_85 = df[df["tb_cov_mean"] < 0.85]
    if len(below_85) > 0:
        thresh = below_85.iloc[0]["noise_frac"]
        print(f"\n  Coverage drops below 85% at {thresh:.0%} noise")
    else:
        print(f"\n  Coverage stays above 85% even at 20% noise")
    return df


# =========================================================================
# 4. LATENT TB EXCLUSION (§5.4.13)
# =========================================================================

def run_latent_tb():
    print("\n" + "=" * 70)
    print("4. LATENT TB EXCLUSION SENSITIVITY (§5.4.13)")
    print("=" * 70)

    df_full = pd.read_parquet(EMB_DIR / "rad_dino.parquet")
    ec = [c for c in df_full.columns if c.startswith("emb_")]

    # Join multi-class label from splits.parquet
    splits_meta = pd.read_parquet(SPLITS_PATH)[["patient_id", "label"]].drop_duplicates("patient_id")
    df_full = df_full.merge(splits_meta, on="patient_id", how="left")

    # Three definitions:
    # A: primary (active + latent = TB positive)
    # B: active only; latent excluded entirely
    # C: active only; latent reclassified as TB-negative

    definitions = {
        "A (primary: active+latent=TB+)": {
            "include": lambda r: r["tb_binary"] in ["tb_positive", "tb_negative"],
            "label": lambda r: 1 if r["tb_binary"] == "tb_positive" else 0,
        },
        "B (active only; latent excluded)": {
            "include": lambda r: r["label"] != "latent_tb" and r["tb_binary"] in ["tb_positive", "tb_negative"],
            "label": lambda r: 1 if r["label"] in ["active_tb", "tb"] else 0,
        },
        "C (active only; latent=TB-)": {
            "include": lambda r: r["tb_binary"] in ["tb_positive", "tb_negative"],
            "label": lambda r: 1 if r["label"] in ["active_tb", "tb"] and r["label"] != "latent_tb" else 0,
        },
    }

    results = []
    for def_name, funcs in definitions.items():
        mask = df_full.apply(funcs["include"], axis=1)
        sub = df_full[mask].copy()
        sub["y"] = sub.apply(funcs["label"], axis=1)
        X = normalize(sub[ec].values.astype(np.float32), norm="l2")
        y = sub["y"].values
        splits = sub["split"].values

        cal_m = splits == "calibration"
        dev_m = splits == "dev"
        test_m = splits == "test"

        if cal_m.sum() < 50 or test_m.sum() < 100:
            print(f"  {def_name}: insufficient data (cal={cal_m.sum()}, test={test_m.sum()})")
            continue

        m = train_and_evaluate(X[cal_m], y[cal_m], X[dev_m], y[dev_m], X[test_m], y[test_m])

        n_latent_test = ((sub["label"] == "latent_tb") & (splits == "test")).sum()
        n_tb_test = (y[test_m] == 1).sum()

        results.append({
            "definition": def_name,
            "n_cal": int(cal_m.sum()), "n_test": int(test_m.sum()),
            "n_tb_test": int(n_tb_test),
            "n_latent_test": int(n_latent_test),
            **{k: round(v, 4) for k, v in m.items()},
        })
        print(f"  {def_name}:")
        print(f"    n_test={test_m.sum()}, n_TB+={n_tb_test}, n_latent={n_latent_test}")
        print(f"    AUROC={m['auroc']:.4f}  TB_cov={m['tb_cov']:.4f}  singleton={m['singleton']:.4f}", flush=True)

    df = pd.DataFrame(results)
    df.to_csv(TABLES_DIR / "latent_tb_exclusion.csv", index=False)

    # Check if Definition B/C substantially improves AUROC
    if len(df) >= 2:
        a_auroc = df[df["definition"].str.startswith("A")]["auroc"].values[0]
        for _, row in df.iterrows():
            if not row["definition"].startswith("A"):
                delta = row["auroc"] - a_auroc
                print(f"\n  {row['definition']}: AUROC delta = {'+' if delta >= 0 else ''}{delta:.4f} vs primary")
    return df


# =========================================================================
# 5. PREVALENCE MISMATCH (§5.4.14)
# =========================================================================

def run_prevalence_mismatch():
    print("\n" + "=" * 70)
    print("5. CALIBRATION PREVALENCE MISMATCH (§5.4.14)")
    print("=" * 70)

    X, y, splits, _, _ = load_primary()
    cal_m = splits == "calibration"
    dev_m = splits == "dev"
    test_m = splits == "test"

    X_cal, y_cal = X[cal_m], y[cal_m]
    X_dev, y_dev = X[dev_m], y[dev_m]
    X_test, y_test = X[test_m], y[test_m]

    # Current cal prevalence is ~49% (394/800). Subsample to lower prevalences.
    target_prevs = [0.05, 0.10, 0.25, 0.49]
    results = []

    idx_tb = np.where(y_cal == 1)[0]
    idx_nontb = np.where(y_cal == 0)[0]

    for target_prev in target_prevs:
        # Keep all non-TB, subsample TB to achieve target prevalence
        # target_prev = n_tb / (n_tb + n_nontb)
        # n_tb = target_prev * n_nontb / (1 - target_prev)
        n_nontb = len(idx_nontb)
        n_tb_needed = int(target_prev * n_nontb / (1 - target_prev))
        n_tb_needed = min(n_tb_needed, len(idx_tb))

        rng = np.random.RandomState(SEED)
        tb_sub = rng.choice(idx_tb, n_tb_needed, replace=False) if n_tb_needed < len(idx_tb) else idx_tb
        sub_idx = np.concatenate([idx_nontb, tb_sub])

        actual_prev = y_cal[sub_idx].mean()
        m = train_and_evaluate(X_cal[sub_idx], y_cal[sub_idx], X_dev, y_dev, X_test, y_test)

        results.append({
            "target_prev": target_prev, "actual_prev": round(actual_prev, 3),
            "n_cal": len(sub_idx), "n_tb": int(y_cal[sub_idx].sum()),
            **{k: round(v, 4) for k, v in m.items()},
        })
        print(f"  prev={actual_prev:.1%} (n={len(sub_idx)}, TB+={y_cal[sub_idx].sum()}): "
              f"AUROC={m['auroc']:.4f}  TB_cov={m['tb_cov']:.4f}  singleton={m['singleton']:.4f}", flush=True)

    df = pd.DataFrame(results)
    df.to_csv(TABLES_DIR / "prevalence_mismatch.csv", index=False)
    return df


# =========================================================================
# 6. NON-TB CONFUSION MATRIX (§5.4.19)
# =========================================================================

def run_nontb_confusion():
    print("\n" + "=" * 70)
    print("6. NON-TB FALSE POSITIVE ANALYSIS (§5.4.19)")
    print("=" * 70)

    df_full = pd.read_parquet(EMB_DIR / "rad_dino.parquet")
    ec = [c for c in df_full.columns if c.startswith("emb_")]

    # Join multi-class label from splits.parquet
    splits_meta = pd.read_parquet(SPLITS_PATH)[["patient_id", "label"]].drop_duplicates("patient_id")
    df_full = df_full.merge(splits_meta, on="patient_id", how="left")

    # Train probe on calibration
    mask = df_full["tb_binary"].isin(["tb_positive", "tb_negative"])
    df_lab = df_full[mask].copy()
    X = normalize(df_lab[ec].values.astype(np.float32), norm="l2")
    y = (df_lab["tb_binary"] == "tb_positive").astype(int).values
    splits = df_lab["split"].values

    cal_m = splits == "calibration"
    test_m = splits == "test"

    probe = LogisticRegression(C=10, max_iter=2000, solver="lbfgs", random_state=SEED)
    probe.fit(X[cal_m], y[cal_m])
    test_prob = probe.predict_proba(X[test_m])[:, 1]
    test_y = y[test_m]
    test_labels = df_lab[mask].iloc[np.where(test_m)[0]]["label"].values

    # Find false positives at 90% sensitivity threshold
    sorted_idx = np.argsort(-test_prob)
    tp = 0
    total_pos = test_y.sum()
    thresh_90 = 0.5
    for i in sorted_idx:
        if test_y[i] == 1:
            tp += 1
        if tp / max(total_pos, 1) >= 0.90:
            thresh_90 = test_prob[i]
            break

    fp_mask = (test_prob >= thresh_90) & (test_y == 0)
    fp_labels = test_labels[fp_mask]
    all_nontb_labels = test_labels[test_y == 0]

    print(f"  Threshold for 90% sensitivity: {thresh_90:.4f}")
    print(f"  False positives: {fp_mask.sum()} / {(test_y == 0).sum()} non-TB images")
    print(f"\n  False positive pathology distribution:")

    fp_counts = pd.Series(fp_labels).value_counts()
    all_counts = pd.Series(all_nontb_labels).value_counts()

    for label in fp_counts.index[:10]:
        fp_n = fp_counts.get(label, 0)
        all_n = all_counts.get(label, 0)
        fp_rate = fp_n / max(all_n, 1)
        print(f"    {label:20s}  FP={fp_n:>5}  total={all_n:>5}  FP_rate={fp_rate:.3f}")

    confusion_df = pd.DataFrame({
        "label": fp_counts.index,
        "n_false_positive": fp_counts.values,
        "n_total": [all_counts.get(l, 0) for l in fp_counts.index],
        "fp_rate": [fp_counts.get(l, 0) / max(all_counts.get(l, 0), 1) for l in fp_counts.index],
    })
    confusion_df.to_csv(TABLES_DIR / "nontb_confusion.csv", index=False)
    return confusion_df


# =========================================================================
# MAIN
# =========================================================================

def main():
    t0 = time.time()
    print("=" * 70)
    print("TIER 2 SENSITIVITY ANALYSES — Conformal TB Triage")
    print("=" * 70)

    run_seed_stability()
    run_meta_coverage()
    run_label_noise()
    run_latent_tb()
    run_prevalence_mismatch()
    run_nontb_confusion()

    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"All sensitivity analyses complete in {elapsed:.0f}s")
    print(f"Results in: {TABLES_DIR}/")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
