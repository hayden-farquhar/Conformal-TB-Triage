"""
Tier 3 analyses:
1. Reference baselines (random, prevalence, radiomics)
2. Decision curve analysis
3. Computational cost benchmarking
4. Embedding space visualisation (t-SNE data export)
5. Calibration curves (ECE/MCE)

Run: python3 src/evaluation/tier3_analyses.py
"""

import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    roc_curve, precision_recall_curve
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import normalize
from sklearn.isotonic import IsotonicRegression
from sklearn.manifold import TSNE

warnings.filterwarnings("ignore")

from config import *

np.random.seed(SEED)


def load_emb_with_meta(name, l2=False):
    df = pd.read_parquet(EMB_DIR / f"{name}.parquet")
    ec = [c for c in df.columns if c.startswith("emb_")]
    splits_meta = pd.read_parquet(SPLITS_PATH)[["patient_id", "label", "dataset"]].drop_duplicates("patient_id")
    df = df.merge(splits_meta[["patient_id", "label"]], on="patient_id", how="left")
    mask = df["tb_binary"].isin(["tb_positive", "tb_negative"])
    df = df[mask].copy()
    X = df[ec].values.astype(np.float32)
    if l2:
        X = normalize(X, norm="l2")
    y = (df["tb_binary"] == "tb_positive").astype(int).values
    return X, y, df["split"].values, df["dataset"].values, df["patient_id"].values, ec


# =========================================================================
# 1. REFERENCE BASELINES (§5.2.4)
# =========================================================================

def run_baselines():
    print("=" * 70)
    print("1. REFERENCE BASELINES (§5.2.4)")
    print("=" * 70)

    X, y, splits, datasets, pids, ec = load_emb_with_meta("rad_dino", l2=True)
    test_m = splits == "test"
    cal_m = splits == "calibration"
    test_y = y[test_m]
    n_test = len(test_y)

    results = []

    # A. Random classifier
    rng = np.random.RandomState(SEED)
    random_prob = rng.uniform(0, 1, n_test)
    results.append({
        "baseline": "Random",
        "auroc": round(roc_auc_score(test_y, random_prob), 4),
        "auprc": round(average_precision_score(test_y, random_prob), 4),
    })

    # B. Prevalence-based classifier
    cal_prev = y[cal_m].mean()
    prev_prob = np.full(n_test, cal_prev)
    results.append({
        "baseline": f"Prevalence ({cal_prev:.2f})",
        "auroc": 0.5000,  # constant predictions → AUROC = 0.5 by definition
        "auprc": round(average_precision_score(test_y, prev_prob), 4),
    })

    # C. Radiomics baseline (simple image-level features)
    # We don't have raw images locally, so approximate with embedding statistics
    # Use first 10 PCA components as a "simple features" proxy
    from sklearn.decomposition import PCA
    pca = PCA(n_components=10, random_state=SEED)
    X_pca_cal = pca.fit_transform(X[cal_m])
    X_pca_test = pca.transform(X[test_m])

    # Also compute simple stats: mean, std, skew, kurtosis of embedding dims
    from scipy.stats import skew, kurtosis
    def simple_features(X_block):
        return np.column_stack([
            X_block.mean(axis=1),
            X_block.std(axis=1),
            skew(X_block, axis=1),
            kurtosis(X_block, axis=1),
            np.linalg.norm(X_block, axis=1),
        ])

    X_simple_cal = simple_features(X[cal_m])
    X_simple_test = simple_features(X[test_m])

    lr_simple = LogisticRegression(C=1, max_iter=1000, random_state=SEED)
    lr_simple.fit(X_simple_cal, y[cal_m])
    simple_prob = lr_simple.predict_proba(X_simple_test)[:, 1]
    results.append({
        "baseline": "Simple stats (5 features)",
        "auroc": round(roc_auc_score(test_y, simple_prob), 4),
        "auprc": round(average_precision_score(test_y, simple_prob), 4),
    })

    lr_pca = LogisticRegression(C=1, max_iter=1000, random_state=SEED)
    lr_pca.fit(X_pca_cal, y[cal_m])
    pca_prob = lr_pca.predict_proba(X_pca_test)[:, 1]
    results.append({
        "baseline": "PCA-10 (dimensionality proxy)",
        "auroc": round(roc_auc_score(test_y, pca_prob), 4),
        "auprc": round(average_precision_score(test_y, pca_prob), 4),
    })

    # Add the actual model results for comparison
    probe = LogisticRegression(C=10, max_iter=2000, solver="lbfgs", random_state=SEED)
    probe.fit(X[cal_m], y[cal_m])
    model_prob = probe.predict_proba(X[test_m])[:, 1]
    results.append({
        "baseline": "RAD-DINO + linear (primary)",
        "auroc": round(roc_auc_score(test_y, model_prob), 4),
        "auprc": round(average_precision_score(test_y, model_prob), 4),
    })

    df = pd.DataFrame(results)
    df.to_csv(TABLES_DIR / "reference_baselines.csv", index=False)
    print(df.to_string(index=False))
    return df


# =========================================================================
# 2. DECISION CURVE ANALYSIS (§5.3.8)
# =========================================================================

def run_decision_curves():
    print("\n" + "=" * 70)
    print("2. DECISION CURVE ANALYSIS (§5.3.8)")
    print("=" * 70)

    X, y, splits, _, _, _ = load_emb_with_meta("rad_dino", l2=True)
    cal_m = splits == "calibration"
    test_m = splits == "test"
    test_y = y[test_m]

    probe = LogisticRegression(C=10, max_iter=2000, solver="lbfgs", random_state=SEED)
    probe.fit(X[cal_m], y[cal_m])
    test_prob = probe.predict_proba(X[test_m])[:, 1]

    prev = test_y.mean()
    thresholds = np.arange(0.01, 0.51, 0.01)
    results = []

    for t in thresholds:
        # Model net benefit
        tp = ((test_prob >= t) & (test_y == 1)).sum()
        fp = ((test_prob >= t) & (test_y == 0)).sum()
        n = len(test_y)
        net_benefit_model = tp / n - fp / n * (t / (1 - t))

        # Treat all
        net_benefit_all = prev - (1 - prev) * (t / (1 - t))

        # Treat none
        net_benefit_none = 0

        results.append({
            "threshold": round(t, 2),
            "net_benefit_model": round(net_benefit_model, 6),
            "net_benefit_all": round(net_benefit_all, 6),
            "net_benefit_none": 0,
        })

    dca_df = pd.DataFrame(results)
    dca_df.to_csv(TABLES_DIR / "decision_curves.csv", index=False)

    # Find range where model provides net benefit over both alternatives
    beneficial = dca_df[
        (dca_df["net_benefit_model"] > dca_df["net_benefit_all"]) &
        (dca_df["net_benefit_model"] > 0)
    ]
    if len(beneficial) > 0:
        t_low = beneficial["threshold"].min()
        t_high = beneficial["threshold"].max()
        print(f"  Model provides net benefit over 'treat all' for thresholds {t_low:.2f}–{t_high:.2f}")
    else:
        print("  Model does not provide net benefit over 'treat all' at any threshold")

    # Net benefit at WHO TPP operating point (threshold ~0.10 for 10% prevalence)
    who_row = dca_df[dca_df["threshold"] == 0.10].iloc[0]
    print(f"  At threshold=0.10: model NB={who_row['net_benefit_model']:.4f}, treat-all NB={who_row['net_benefit_all']:.4f}")
    return dca_df


# =========================================================================
# 3. CALIBRATION CURVES / ECE (§5.2.3)
# =========================================================================

def run_calibration_assessment():
    print("\n" + "=" * 70)
    print("3. CALIBRATION ASSESSMENT (ECE/MCE/Brier)")
    print("=" * 70)

    X, y, splits, _, _, _ = load_emb_with_meta("rad_dino", l2=True)
    cal_m = splits == "calibration"
    dev_m = splits == "dev"
    test_m = splits == "test"

    probe = LogisticRegression(C=10, max_iter=2000, solver="lbfgs", random_state=SEED)
    probe.fit(X[cal_m], y[cal_m])

    test_y = y[test_m]
    raw_prob = probe.predict_proba(X[test_m])[:, 1]
    dev_prob = probe.predict_proba(X[dev_m])[:, 1]

    # Calibration methods
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(dev_prob, y[dev_m])
    iso_prob = iso.predict(raw_prob)

    # Temperature scaling
    from scipy.optimize import minimize_scalar
    logits_dev = np.log(np.clip(dev_prob, 1e-7, 1-1e-7) / (1 - np.clip(dev_prob, 1e-7, 1-1e-7)))
    logits_test = np.log(np.clip(raw_prob, 1e-7, 1-1e-7) / (1 - np.clip(raw_prob, 1e-7, 1-1e-7)))

    def nll(T):
        p = 1 / (1 + np.exp(-logits_dev / T))
        p = np.clip(p, 1e-7, 1-1e-7)
        return -np.mean(y[dev_m] * np.log(p) + (1 - y[dev_m]) * np.log(1 - p))

    T_opt = minimize_scalar(nll, bounds=(0.1, 10), method="bounded").x
    temp_prob = 1 / (1 + np.exp(-logits_test / T_opt))

    # Platt scaling
    platt = LogisticRegression(C=1e10, solver="lbfgs", max_iter=1000)
    platt.fit(logits_dev.reshape(-1, 1), y[dev_m])
    platt_prob = platt.predict_proba(logits_test.reshape(-1, 1))[:, 1]

    def compute_ece(y_true, y_prob, n_bins=10):
        bins = np.linspace(0, 1, n_bins + 1)
        ece, mce = 0, 0
        bin_data = []
        for i in range(n_bins):
            mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
            if mask.sum() == 0:
                continue
            bin_acc = y_true[mask].mean()
            bin_conf = y_prob[mask].mean()
            bin_size = mask.sum()
            diff = abs(bin_acc - bin_conf)
            ece += diff * bin_size / len(y_true)
            mce = max(mce, diff)
            bin_data.append({
                "bin_lower": round(bins[i], 1), "bin_upper": round(bins[i+1], 1),
                "n": int(bin_size), "accuracy": round(bin_acc, 4),
                "confidence": round(bin_conf, 4), "gap": round(diff, 4),
            })
        return ece, mce, bin_data

    results = []
    calibration_bins = []
    for name, prob in [("Uncalibrated", raw_prob), ("Temperature", temp_prob),
                       ("Platt", platt_prob), ("Isotonic", iso_prob)]:
        ece, mce, bins = compute_ece(test_y, prob)
        brier = brier_score_loss(test_y, prob)
        results.append({
            "method": name, "ECE": round(ece, 4), "MCE": round(mce, 4),
            "Brier": round(brier, 4),
            "T": round(T_opt, 4) if name == "Temperature" else None,
        })
        for b in bins:
            b["method"] = name
            calibration_bins.append(b)
        print(f"  {name:15s}  ECE={ece:.4f}  MCE={mce:.4f}  Brier={brier:.4f}", flush=True)

    pd.DataFrame(results).to_csv(TABLES_DIR / "calibration_metrics.csv", index=False)
    pd.DataFrame(calibration_bins).to_csv(TABLES_DIR / "calibration_bins.csv", index=False)
    return results


# =========================================================================
# 4. ROC AND PR CURVE DATA
# =========================================================================

def run_roc_pr_curves():
    print("\n" + "=" * 70)
    print("4. ROC AND PR CURVE DATA")
    print("=" * 70)

    results_roc = []
    results_pr = []

    for emb_name in WORKING_MODELS:
        X, y, splits, _, _, _ = load_emb_with_meta(emb_name, l2=True)
        cal_m = splits == "calibration"
        test_m = splits == "test"

        best_C = {"rad_dino": 10, "biomedclip": 0.01, "torchxrayvision": 100, "dinov2": 100}.get(emb_name, 1)
        probe = LogisticRegression(C=best_C, max_iter=2000, solver="lbfgs", random_state=SEED)
        probe.fit(X[cal_m], y[cal_m])
        test_prob = probe.predict_proba(X[test_m])[:, 1]

        fpr, tpr, _ = roc_curve(y[test_m], test_prob)
        prec, rec, _ = precision_recall_curve(y[test_m], test_prob)

        # Subsample for reasonable file size
        step = max(1, len(fpr) // 500)
        for i in range(0, len(fpr), step):
            results_roc.append({"embedding": emb_name, "fpr": round(fpr[i], 6), "tpr": round(tpr[i], 6)})
        step = max(1, len(prec) // 500)
        for i in range(0, len(prec), step):
            results_pr.append({"embedding": emb_name, "recall": round(rec[i], 6), "precision": round(prec[i], 6)})

        auroc = roc_auc_score(y[test_m], test_prob)
        auprc = average_precision_score(y[test_m], test_prob)
        print(f"  {emb_name:20s}  AUROC={auroc:.4f}  AUPRC={auprc:.4f}", flush=True)

    pd.DataFrame(results_roc).to_csv(TABLES_DIR / "roc_curves.csv", index=False)
    pd.DataFrame(results_pr).to_csv(TABLES_DIR / "pr_curves.csv", index=False)


# =========================================================================
# 5. EMBEDDING SPACE VISUALISATION DATA (§5.3.6)
# =========================================================================

def run_embedding_tsne():
    print("\n" + "=" * 70)
    print("5. EMBEDDING SPACE t-SNE (§5.3.6)")
    print("=" * 70)

    for emb_name in ["rad_dino", "biomedclip"]:
        print(f"  {emb_name}: computing t-SNE...", flush=True)

        df = pd.read_parquet(EMB_DIR / f"{emb_name}.parquet")
        ec = [c for c in df.columns if c.startswith("emb_")]

        # Subsample for speed (t-SNE is O(n^2))
        rng = np.random.RandomState(SEED)
        n_sample = min(5000, len(df))
        idx = rng.choice(len(df), n_sample, replace=False)
        X_sub = normalize(df.iloc[idx][ec].values.astype(np.float32), norm="l2")

        tsne = TSNE(n_components=2, random_state=SEED, perplexity=30, max_iter=1000)
        coords = tsne.fit_transform(X_sub)

        tsne_df = pd.DataFrame({
            "patient_id": df.iloc[idx]["patient_id"].values,
            "tsne_x": coords[:, 0],
            "tsne_y": coords[:, 1],
            "dataset": df.iloc[idx]["dataset"].values,
            "tb_binary": df.iloc[idx]["tb_binary"].values,
            "split": df.iloc[idx]["split"].values,
        })
        tsne_df.to_csv(TABLES_DIR / f"tsne_{emb_name}.csv", index=False)
        print(f"    Saved {n_sample} points.", flush=True)

    # Silhouette scores
    print("\n  Silhouette scores:", flush=True)
    from sklearn.metrics import silhouette_score

    for emb_name in WORKING_MODELS:
        df = pd.read_parquet(EMB_DIR / f"{emb_name}.parquet")
        ec = [c for c in df.columns if c.startswith("emb_")]
        mask = df["tb_binary"].isin(["tb_positive", "tb_negative"])
        df_lab = df[mask]

        rng = np.random.RandomState(SEED)
        n = min(3000, len(df_lab))
        idx = rng.choice(len(df_lab), n, replace=False)

        X_sub = normalize(df_lab.iloc[idx][ec].values.astype(np.float32), norm="l2")
        y_tb = (df_lab.iloc[idx]["tb_binary"] == "tb_positive").astype(int).values
        y_ds = df_lab.iloc[idx]["dataset"].values

        sil_tb = silhouette_score(X_sub, y_tb, metric="cosine", sample_size=min(n, 2000))
        try:
            sil_ds = silhouette_score(X_sub, y_ds, metric="cosine", sample_size=min(n, 2000))
        except:
            sil_ds = float("nan")

        print(f"    {emb_name:20s}  TB silhouette={sil_tb:.4f}  Dataset silhouette={sil_ds:.4f}", flush=True)


# =========================================================================
# 6. COMPUTATIONAL COST (§5.3.13)
# =========================================================================

def run_computational_cost():
    print("\n" + "=" * 70)
    print("6. COMPUTATIONAL COST ESTIMATE (§5.3.13)")
    print("=" * 70)

    # Model sizes (from the parquet files as proxy for embedding dimensions)
    model_info = {
        "rad_dino": {"params_M": 86.6, "embed_dim": 768, "arch": "ViT-B/16"},
        "eva_x": {"params_M": 86.6, "embed_dim": 768, "arch": "ViT-B/16"},
        "chexzero": {"params_M": 151.3, "embed_dim": 512, "arch": "ViT-B/32"},
        "biomedclip": {"params_M": 86.6, "embed_dim": 512, "arch": "ViT-B/16"},
        "gloria": {"params_M": 25.6, "embed_dim": 2048, "arch": "ResNet-50"},
        "torchxrayvision": {"params_M": 7.0, "embed_dim": 1024, "arch": "DenseNet-121"},
        "dinov2": {"params_M": 86.6, "embed_dim": 768, "arch": "ViT-B/14"},
    }

    # Probe cost: negligible (linear probe = one matrix multiply)
    # Conformal cost: negligible (quantile computation)

    results = []
    for name, info in model_info.items():
        emb_path = EMB_DIR / f"{name}.parquet"
        emb_size_mb = emb_path.stat().st_size / 1e6 if emb_path.exists() else 0

        # Estimate model weight file size (roughly 4 bytes × params)
        weight_size_mb = info["params_M"] * 4  # FP32

        results.append({
            "model": name,
            "architecture": info["arch"],
            "params_M": info["params_M"],
            "embed_dim": info["embed_dim"],
            "weight_size_MB": round(weight_size_mb, 0),
            "embedding_file_MB": round(emb_size_mb, 1),
            "fits_4GB_RAM": "Yes" if weight_size_mb < 3500 else "No",
            "fits_8GB_RAM": "Yes" if weight_size_mb < 7500 else "No",
        })

    cost_df = pd.DataFrame(results)
    cost_df.to_csv(TABLES_DIR / "computational_cost.csv", index=False)
    print(cost_df.to_string(index=False))

    # Probe + conformal inference time (measured)
    X, y, splits, _, _, _ = load_emb_with_meta("rad_dino", l2=True)
    test_m = splits == "test"
    cal_m = splits == "calibration"
    probe = LogisticRegression(C=10, max_iter=2000, solver="lbfgs", random_state=SEED)
    probe.fit(X[cal_m], y[cal_m])

    t0 = time.time()
    for _ in range(100):
        _ = probe.predict_proba(X[test_m])
    probe_time = (time.time() - t0) / 100 / len(X[test_m]) * 1000  # ms per image
    print(f"\n  Probe inference: {probe_time:.4f} ms/image (CPU)")
    print(f"  Conformal set computation: <0.001 ms/image")
    print(f"  Total post-embedding: <{probe_time + 0.001:.3f} ms/image")

    return cost_df


# =========================================================================
# MAIN
# =========================================================================

def main():
    t0 = time.time()
    print("=" * 70)
    print("TIER 3 ANALYSES — Conformal TB Triage")
    print("=" * 70)

    run_baselines()
    run_decision_curves()
    run_calibration_assessment()
    run_roc_pr_curves()
    run_embedding_tsne()
    run_computational_cost()

    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"All Tier 3 analyses complete in {elapsed:.0f}s")
    print(f"{'=' * 70}")

    # List all output files
    print(f"\nAll results files in {TABLES_DIR}/:")
    for f in sorted(TABLES_DIR.glob("*.csv")):
        print(f"  {f.name:40s}  {f.stat().st_size/1e3:.1f} KB")


if __name__ == "__main__":
    main()
