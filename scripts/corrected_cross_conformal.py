"""
Corrected (held-out, in-distribution) cross-conformal CV+.

The original implementation pooled the Shenzhen+Montgomery `calibration` split
(NLM; the probe-training distribution, out-of-distribution w.r.t. the TBX11K
test set) WITH the held-out TBX11K `dev` split for its CV+ folds, and compared
against the in-sample split-conformal number (TB 94.1%). Both references are
superseded by this revision.

This version runs 5-fold CV+ purely over the held-out TBX11K `dev` pool
(in-distribution with test), using the corrected probe recipe (RAW embeddings,
CV-selected C), and compares against the valid held-out split-conformal headline
(TB-class 92.5%). The result shows that CV+'s single marginal threshold
under-covers the minority TB class and emits empty sets, the failure the
class-conditional (Mondrian) split-conformal headline avoids.

CPU only. Reads rad_dino embeddings from `$DATA_ROOT/embeddings/rad_dino.parquet`
(set DATA_ROOT; defaults to ./data relative to the repository root).
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("DATA_ROOT", str(REPO_ROOT / "data")))
EMB_DIR = DATA_ROOT / "embeddings"
TABLES_DIR = REPO_ROOT / "outputs" / "tables"
SEED = 42
ALPHA = 0.10
SPLIT_REF_TB = 0.925  # valid held-out split-conformal TB-class coverage (Table 3)


def train_linear_probe(Xc, yc):
    """Corrected recipe: raw embeddings, 5-fold CV-selected C."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    best_c, best = None, -1.0
    for C in [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]:
        fold = []
        for tr, va in skf.split(Xc, yc):
            m = LogisticRegression(C=C, max_iter=2000, solver="lbfgs",
                                   random_state=SEED).fit(Xc[tr], yc[tr])
            if len(np.unique(yc[va])) > 1:
                fold.append(roc_auc_score(yc[va], m.predict_proba(Xc[va])[:, 1]))
        if fold and np.mean(fold) > best:
            best, best_c = float(np.mean(fold)), C
    final = LogisticRegression(C=best_c, max_iter=2000, solver="lbfgs",
                               random_state=SEED).fit(Xc, yc)
    return final, best_c


def main():
    df = pd.read_parquet(EMB_DIR / "rad_dino.parquet")
    ec = [c for c in df.columns if c.startswith("emb_")]
    df = df[df["tb_binary"].isin(["tb_positive", "tb_negative"])].copy()
    y_all = (df["tb_binary"] == "tb_positive").astype(int).values
    X_all = df[ec].values.astype(np.float32)  # RAW (no L2)
    split = df["split"].values

    X_pool, y_pool = X_all[split == "dev"], y_all[split == "dev"]
    X_test, y_test = X_all[split == "test"], y_all[split == "test"]
    print(f"CV+ over held-out dev pool: n_dev={len(y_pool)}, n_test={len(y_test)}", flush=True)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    test_scores_per_fold = np.zeros((len(y_test), 5))
    oof_scores = []
    for fi, (tr, cal) in enumerate(skf.split(X_pool, y_pool)):
        probe, c = train_linear_probe(X_pool[tr], y_pool[tr])
        cal_prob = probe.predict_proba(X_pool[cal])[:, 1]
        cal_y = y_pool[cal]
        oof_scores.extend(np.where(cal_y == 1, 1 - cal_prob, cal_prob))
        test_scores_per_fold[:, fi] = probe.predict_proba(X_test)[:, 1]
        print(f"  fold {fi+1}: cal_n={len(cal)} C={c} "
              f"AUROC_test={roc_auc_score(y_test, test_scores_per_fold[:, fi]):.4f}", flush=True)

    avg_test = test_scores_per_fold.mean(axis=1)
    oof = np.asarray(oof_scores)
    n = len(oof)
    thr = float(np.quantile(oof, min(np.ceil((n + 1) * (1 - ALPHA)) / n, 1.0)))

    inc1 = (1 - avg_test) <= thr
    inc0 = avg_test <= thr
    size = inc0.astype(int) + inc1.astype(int)
    covered = np.where(y_test == 1, inc1, inc0)
    tb_m = y_test == 1
    res = {
        "method": "CV+ (held-out dev, in-distribution)",
        "alpha": ALPHA,
        "auroc": round(float(roc_auc_score(y_test, avg_test)), 4),
        "marginal_cov": round(float(covered.mean()), 4),
        "tb_cov": round(float(covered[tb_m].mean()), 4),
        "nontb_cov": round(float(covered[~tb_m].mean()), 4),
        "singleton": round(float((size == 1).mean()), 4),
        "empty": round(float((size == 0).mean()), 4),
        "mean_size": round(float(size.mean()), 4),
        "n_cal_scores": n,
        "split_conformal_tb_ref": SPLIT_REF_TB,
    }
    print("\nCorrected CV+ results (alpha=0.10):")
    for k, v in res.items():
        print(f"  {k}: {v}")
    pd.DataFrame([res]).to_csv(TABLES_DIR / "cross_conformal_corrected.csv", index=False)
    print("Wrote cross_conformal_corrected.csv.")


if __name__ == "__main__":
    main()
