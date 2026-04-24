"""
Probe training for Project 68: Conformal TB Triage.

Trains 4 probe types on each of 7 embedding models using the calibration set.
Evaluates on dev, test, and external sets. Saves predictions and metrics.

Run: python3 src/probes/train_probes.py
"""

import sys
import time
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

from config import *
N_FOLDS = 5

EMBEDDING_MODELS = [
    "rad_dino", "eva_x", "chexzero", "biomedclip",
    "gloria", "torchxrayvision", "dinov2",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_embedding(model_name: str) -> pd.DataFrame:
    """Load embedding parquet and return with split/label info."""
    path = EMB_DIR / f"{model_name}.parquet"
    df = pd.read_parquet(path)
    emb_cols = [c for c in df.columns if c.startswith("emb_")]
    return df, emb_cols


def get_xy(df, emb_cols, split_name):
    """Extract X matrix and binary y vector for a given split.
    Only includes labelled images (tb_positive or tb_negative)."""
    mask = (df["split"] == split_name) & (df["tb_binary"].isin(["tb_positive", "tb_negative"]))
    subset = df[mask].copy()
    X = subset[emb_cols].values.astype(np.float32)
    y = (subset["tb_binary"] == "tb_positive").astype(int).values
    return X, y, subset["patient_id"].values


def evaluate(y_true, y_prob):
    """Compute AUROC and AUPRC."""
    if len(np.unique(y_true)) < 2:
        return {"auroc": float("nan"), "auprc": float("nan")}
    return {
        "auroc": round(roc_auc_score(y_true, y_prob), 4),
        "auprc": round(average_precision_score(y_true, y_prob), 4),
    }


# ---------------------------------------------------------------------------
# Probe definitions
# ---------------------------------------------------------------------------

def train_linear_probe(X_cal, y_cal):
    """L2-regularised logistic regression with CV for C selection."""
    C_values = [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    best_c, best_score = None, -1
    for C in C_values:
        fold_scores = []
        for train_idx, val_idx in skf.split(X_cal, y_cal):
            model = LogisticRegression(C=C, max_iter=2000, solver="lbfgs", random_state=SEED)
            model.fit(X_cal[train_idx], y_cal[train_idx])
            prob = model.predict_proba(X_cal[val_idx])[:, 1]
            if len(np.unique(y_cal[val_idx])) > 1:
                fold_scores.append(roc_auc_score(y_cal[val_idx], prob))
        mean_score = np.mean(fold_scores) if fold_scores else 0
        if mean_score > best_score:
            best_c, best_score = C, mean_score

    # Refit on full calibration set
    final = LogisticRegression(C=best_c, max_iter=2000, solver="lbfgs", random_state=SEED)
    final.fit(X_cal, y_cal)
    return final, {"probe": "linear", "best_C": best_c, "cv_auroc": round(best_score, 4)}


def train_knn_probe(X_cal, y_cal):
    """k-NN with CV for k selection. Cosine distance."""
    k_values = [5, 10, 25, 50]
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    best_k, best_score = None, -1
    for k in k_values:
        if k >= len(X_cal):
            continue
        fold_scores = []
        for train_idx, val_idx in skf.split(X_cal, y_cal):
            model = KNeighborsClassifier(n_neighbors=k, metric="cosine", n_jobs=-1)
            model.fit(X_cal[train_idx], y_cal[train_idx])
            prob = model.predict_proba(X_cal[val_idx])[:, 1]
            if len(np.unique(y_cal[val_idx])) > 1:
                fold_scores.append(roc_auc_score(y_cal[val_idx], prob))
        mean_score = np.mean(fold_scores) if fold_scores else 0
        if mean_score > best_score:
            best_k, best_score = k, mean_score

    final = KNeighborsClassifier(n_neighbors=best_k, metric="cosine", n_jobs=-1)
    final.fit(X_cal, y_cal)
    return final, {"probe": "knn", "best_k": best_k, "cv_auroc": round(best_score, 4)}


def train_xgboost_probe(X_cal, y_cal):
    """XGBoost with early stopping via CV."""
    try:
        import xgboost as xgb
    except ImportError:
        print("  xgboost not installed, skipping.", flush=True)
        return None, {"probe": "xgboost", "error": "not installed"}

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_scores = []
    best_n = 200

    for train_idx, val_idx in skf.split(X_cal, y_cal):
        model = xgb.XGBClassifier(
            max_depth=3, n_estimators=200, learning_rate=0.1,
            reg_lambda=1, reg_alpha=0.1,
            eval_metric="logloss", random_state=SEED,
            early_stopping_rounds=20, verbosity=0,
        )
        model.fit(
            X_cal[train_idx], y_cal[train_idx],
            eval_set=[(X_cal[val_idx], y_cal[val_idx])],
            verbose=False,
        )
        prob = model.predict_proba(X_cal[val_idx])[:, 1]
        if len(np.unique(y_cal[val_idx])) > 1:
            fold_scores.append(roc_auc_score(y_cal[val_idx], prob))
        best_n = max(best_n, model.best_iteration + 1) if hasattr(model, "best_iteration") else best_n

    cv_score = np.mean(fold_scores) if fold_scores else 0

    # Refit on full calibration set
    final = xgb.XGBClassifier(
        max_depth=3, n_estimators=min(best_n + 20, 200), learning_rate=0.1,
        reg_lambda=1, reg_alpha=0.1,
        eval_metric="logloss", random_state=SEED, verbosity=0,
    )
    final.fit(X_cal, y_cal)
    return final, {"probe": "xgboost", "best_n_estimators": best_n, "cv_auroc": round(cv_score, 4)}


def train_mlp_probe(X_cal, y_cal):
    """MLP probe using sklearn MLPClassifier (no torch dependency needed)."""
    from sklearn.neural_network import MLPClassifier

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_scores = []

    for train_idx, val_idx in skf.split(X_cal, y_cal):
        model = MLPClassifier(
            hidden_layer_sizes=(256, 128),
            activation="relu", solver="adam",
            alpha=1e-4,  # L2 penalty (weight_decay equivalent)
            learning_rate_init=1e-3,
            max_iter=100, early_stopping=True,
            validation_fraction=0.15, n_iter_no_change=10,
            random_state=SEED,
        )
        model.fit(X_cal[train_idx], y_cal[train_idx])
        prob = model.predict_proba(X_cal[val_idx])[:, 1]
        if len(np.unique(y_cal[val_idx])) > 1:
            fold_scores.append(roc_auc_score(y_cal[val_idx], prob))

    cv_score = np.mean(fold_scores) if fold_scores else 0

    # Refit on full calibration set
    final = MLPClassifier(
        hidden_layer_sizes=(256, 128),
        activation="relu", solver="adam",
        alpha=1e-4, learning_rate_init=1e-3,
        max_iter=100, early_stopping=True,
        validation_fraction=0.15, n_iter_no_change=10,
        random_state=SEED,
    )
    final.fit(X_cal, y_cal)
    return final, {"probe": "mlp", "cv_auroc": round(cv_score, 4)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

PROBE_TRAINERS = {
    "linear": train_linear_probe,
    "knn": train_knn_probe,
    "xgboost": train_xgboost_probe,
    "mlp": train_mlp_probe,
}

EVAL_SPLITS = ["calibration", "dev", "test", "ext_pakistan"]


def main():
    print("=" * 70)
    print("PROBE TRAINING — Project 68: Conformal TB Triage")
    print("=" * 70)

    all_results = []
    all_predictions = []

    for emb_name in EMBEDDING_MODELS:
        print(f"\n{'─' * 60}")
        print(f"Embedding: {emb_name}")
        print(f"{'─' * 60}")

        t0 = time.time()
        df, emb_cols = load_embedding(emb_name)
        print(f"  Loaded {len(df)} rows x {len(emb_cols)}d in {time.time()-t0:.1f}s", flush=True)

        # Get calibration data
        X_cal, y_cal, ids_cal = get_xy(df, emb_cols, "calibration")
        print(f"  Calibration: {len(X_cal)} images ({y_cal.sum()} TB+, {(1-y_cal).sum()} TB-)", flush=True)

        if len(X_cal) < 50:
            print(f"  SKIP: Too few calibration samples ({len(X_cal)})", flush=True)
            continue

        for probe_name, trainer in PROBE_TRAINERS.items():
            print(f"\n  {probe_name}:", flush=True)
            t1 = time.time()

            model, info = trainer(X_cal, y_cal)
            print(f"    Trained in {time.time()-t1:.1f}s. {info}", flush=True)

            if model is None:
                continue

            # Evaluate on all splits
            for split_name in EVAL_SPLITS:
                X_eval, y_eval, ids_eval = get_xy(df, emb_cols, split_name)
                if len(X_eval) == 0:
                    continue

                y_prob = model.predict_proba(X_eval)[:, 1]
                metrics = evaluate(y_eval, y_prob)

                result = {
                    "embedding": emb_name,
                    "probe": probe_name,
                    "split": split_name,
                    "n": len(X_eval),
                    "n_tb": int(y_eval.sum()),
                    "prevalence": round(y_eval.mean(), 3),
                    **info,
                    **metrics,
                }
                all_results.append(result)

                # Save predictions for conformal calibration later
                pred_df = pd.DataFrame({
                    "patient_id": ids_eval,
                    "y_true": y_eval,
                    "y_prob": y_prob,
                    "split": split_name,
                    "embedding": emb_name,
                    "probe": probe_name,
                })
                all_predictions.append(pred_df)

                marker = "***" if split_name == "test" else "   "
                print(f"    {marker} {split_name:15s}  n={len(X_eval):>6}  "
                      f"AUROC={metrics['auroc']:.4f}  AUPRC={metrics['auprc']:.4f}", flush=True)

    # Save results
    results_df = pd.DataFrame(all_results)
    results_path = RESULTS_DIR / "probe_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\n\nResults saved: {results_path}", flush=True)

    # Save predictions
    predictions_df = pd.concat(all_predictions, ignore_index=True)
    predictions_path = RESULTS_DIR / "probe_predictions.parquet"
    predictions_df.to_parquet(predictions_path, index=False)
    print(f"Predictions saved: {predictions_path}", flush=True)

    # Summary table
    print(f"\n{'=' * 70}")
    print("SUMMARY: AUROC on TBX11K test set")
    print(f"{'=' * 70}")
    test_results = results_df[results_df["split"] == "test"].copy()
    if len(test_results) > 0:
        pivot = test_results.pivot_table(
            index="embedding", columns="probe", values="auroc"
        )
        # Reorder
        probe_order = [p for p in ["linear", "knn", "xgboost", "mlp"] if p in pivot.columns]
        emb_order = [e for e in EMBEDDING_MODELS if e in pivot.index]
        pivot = pivot.loc[emb_order, probe_order]
        print(pivot.to_string(float_format="%.4f"))

        # Best overall
        best_idx = test_results["auroc"].idxmax()
        best = test_results.loc[best_idx]
        print(f"\nBest: {best['embedding']} + {best['probe']} = {best['auroc']:.4f} AUROC")

        # Deployment gate G1 check
        max_auroc = test_results["auroc"].max()
        g1_pass = max_auroc >= 0.75
        print(f"\nDeployment Gate G1 (AUROC >= 0.75): {'PASS' if g1_pass else 'FAIL'} ({max_auroc:.4f})")

    # External validation summary
    print(f"\n{'=' * 70}")
    print("EXTERNAL VALIDATION: Pakistan")
    print(f"{'=' * 70}")
    ext_results = results_df[results_df["split"] == "ext_pakistan"].copy()
    if len(ext_results) > 0:
        pivot_ext = ext_results.pivot_table(
            index="embedding", columns="probe", values="auroc"
        )
        probe_order = [p for p in ["linear", "knn", "xgboost", "mlp"] if p in pivot_ext.columns]
        emb_order = [e for e in EMBEDDING_MODELS if e in pivot_ext.index]
        pivot_ext = pivot_ext.loc[emb_order, probe_order]
        print(pivot_ext.to_string(float_format="%.4f"))


if __name__ == "__main__":
    main()
