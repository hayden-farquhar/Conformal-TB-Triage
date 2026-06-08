"""
Conformal pipeline with held-out conformal calibration.

Addresses the in-sample-calibration defect identified in a split-provenance audit:
an earlier version of this pipeline trained the probes on the NLM `calibration`
split AND computed the conformal nonconformity scores on that same split, so the
conformal calibration scores were in-sample. That violates split-conformal
data-independence and produced a degenerate result (TB_cov 0.941 but
marginal_cov 0.475, 14.7% empty sets).

Held-out design (split-conformal, valid):
  - probe:            trained on NLM `calibration` (unchanged; AUROC 0.9138 on test)
  - conformal calib:  HELD-OUT TBX11K `dev` (out-of-sample for the probe;
                      in-distribution with the test set)
  - test:             TBX11K `test`
  - external shift:   Pakistan `ext_pakistan`

Probabilistic recalibration (isotonic) is fit on a DISJOINT half of `dev` so the
conformal calibration half is never reused. Everything here is reproducible from
results/probe_predictions.parquet -- no GPU / embedding re-run required, because the
dev/test/ext predictions are already out-of-sample for the NLM-trained probe.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import train_test_split

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "outputs"
PRED_PATH = RESULTS_DIR / "probe_predictions.parquet"
OUT_PATH = RESULTS_DIR / "tables" / "conformal_results.csv"

SEED = 42
WORKING_MODELS = ["rad_dino", "biomedclip", "torchxrayvision", "dinov2"]
PROBES = ["linear", "mlp"]
ALPHAS = [0.05, 0.10, 0.20]
N_BOOT = 1000


# ---------------------------------------------------------------------------
# Conformal (identical maths to src/conformal/improved_pipeline.py)
# ---------------------------------------------------------------------------
def mondrian_thresholds(cal_prob, cal_y, alpha):
    cal_scores = np.where(cal_y == 1, 1 - cal_prob, cal_prob)
    thr = {}
    for cls in (0, 1):
        s = cal_scores[cal_y == cls]
        n = len(s)
        if n < 5:
            thr[cls] = 1.0
            continue
        q = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
        thr[cls] = np.quantile(s, q)
    return thr


def aps_threshold(cal_prob, cal_y, alpha):
    cal_scores = np.where(cal_y == 1, 1 - cal_prob, cal_prob)
    n = len(cal_scores)
    q = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    return np.quantile(cal_scores, q)


def sets_from_mondrian(thr, test_prob):
    inc1 = (1 - test_prob) <= thr.get(1, 1.0)
    inc0 = test_prob <= thr.get(0, 1.0)
    return inc0, inc1


def sets_from_aps(thr, test_prob):
    inc1 = (1 - test_prob) <= thr
    inc0 = test_prob <= thr
    return inc0, inc1


def eval_sets(inc0, inc1, y):
    covered = np.where(y == 1, inc1, inc0)
    size = inc0.astype(int) + inc1.astype(int)
    tb = y == 1
    ntb = y == 0
    tb_cov = covered[tb].mean() if tb.any() else np.nan
    ntb_cov = covered[ntb].mean() if ntb.any() else np.nan
    return {
        "marginal_cov": round(float(covered.mean()), 4),
        "tb_cov": round(float(tb_cov), 4),
        "nontb_cov": round(float(ntb_cov), 4),
        "mean_size": round(float(size.mean()), 4),
        "singleton": round(float((size == 1).mean()), 4),
        "empty": round(float((size == 0).mean()), 4),
        "disparity": round(float(abs(tb_cov - ntb_cov)), 4),
    }


def boot_marginal_ci(cal_prob, cal_y, test_prob, test_y, alpha, conformal, n_boot=N_BOOT):
    """Bootstrap the conformal calibration set; report marginal + TB coverage CIs
    on the FIXED test set. This is the valid replacement for the original
    'resplit fragility' diagnostic."""
    rng = np.random.default_rng(SEED)
    margs, tbs = [], []
    n = len(cal_prob)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        cp, cy = cal_prob[idx], cal_y[idx]
        if conformal == "Mondrian":
            thr = mondrian_thresholds(cp, cy, alpha)
            inc0, inc1 = sets_from_mondrian(thr, test_prob)
        else:
            thr = aps_threshold(cp, cy, alpha)
            inc0, inc1 = sets_from_aps(thr, test_prob)
        covered = np.where(test_y == 1, inc1, inc0)
        margs.append(covered.mean())
        tbs.append(covered[test_y == 1].mean())
    return (
        round(float(np.percentile(margs, 2.5)), 4),
        round(float(np.percentile(margs, 97.5)), 4),
        round(float(np.percentile(tbs, 2.5)), 4),
        round(float(np.percentile(tbs, 97.5)), 4),
    )


def main():
    df = pd.read_parquet(PRED_PATH)
    rows = []

    for emb in WORKING_MODELS:
        for probe in PROBES:
            d = df[(df.embedding == emb) & (df.probe == probe)]
            if d.empty:
                continue

            def split(sp):
                s = d[d.split == sp]
                return s["y_prob"].to_numpy(), s["y_true"].to_numpy().astype(int)

            dev_p, dev_y = split("dev")
            test_p, test_y = split("test")
            ext_p, ext_y = split("ext_pakistan")
            if len(dev_p) == 0 or len(test_p) == 0:
                continue

            # Disjoint dev halves: recal (isotonic) vs conformal calibration
            idx = np.arange(len(dev_p))
            recal_idx, conf_idx = train_test_split(
                idx, test_size=0.5, random_state=SEED, stratify=dev_y
            )
            dev_recal_p, dev_recal_y = dev_p[recal_idx], dev_y[recal_idx]
            dev_conf_p, dev_conf_y = dev_p[conf_idx], dev_y[conf_idx]

            # raw and isotonic score variants
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(dev_recal_p, dev_recal_y)
            variants = {
                "raw": (dev_conf_p, test_p, ext_p),
                "isotonic": (
                    iso.predict(dev_conf_p),
                    iso.predict(test_p),
                    iso.predict(ext_p),
                ),
            }

            for score_name, (cal_s, test_s, ext_s) in variants.items():
                for alpha in ALPHAS:
                    for conformal in ("APS", "Mondrian"):
                        if conformal == "Mondrian":
                            thr = mondrian_thresholds(cal_s, dev_conf_y, alpha)
                            ti0, ti1 = sets_from_mondrian(thr, test_s)
                            ei0, ei1 = sets_from_mondrian(thr, ext_s)
                        else:
                            thr = aps_threshold(cal_s, dev_conf_y, alpha)
                            ti0, ti1 = sets_from_aps(thr, test_s)
                            ei0, ei1 = sets_from_aps(thr, ext_s)

                        m_test = eval_sets(ti0, ti1, test_y)
                        lo_m, hi_m, lo_tb, hi_tb = boot_marginal_ci(
                            cal_s, dev_conf_y, test_s, test_y, alpha, conformal
                        )
                        rows.append({
                            "embedding": emb, "probe": probe, "score": score_name,
                            "conformal": conformal, "alpha": alpha, "eval": "tbx11k_test",
                            "n_cal": len(cal_s), "n_eval": len(test_y),
                            **m_test,
                            "marg_lo95": lo_m, "marg_hi95": hi_m,
                            "tb_lo95": lo_tb, "tb_hi95": hi_tb,
                        })

                        if len(ext_y):
                            m_ext = eval_sets(ei0, ei1, ext_y)
                            rows.append({
                                "embedding": emb, "probe": probe, "score": score_name,
                                "conformal": conformal, "alpha": alpha,
                                "eval": "pakistan_ext",
                                "n_cal": len(cal_s), "n_eval": len(ext_y),
                                **m_ext,
                                "marg_lo95": np.nan, "marg_hi95": np.nan,
                                "tb_lo95": np.nan, "tb_hi95": np.nan,
                            })

    out = pd.DataFrame(rows)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)
    print(f"Saved {len(out)} rows -> {OUT_PATH}\n")

    # Headline: primary model, Mondrian, alpha 0.10
    head = out[(out.conformal == "Mondrian") & (out.alpha == 0.10)
               & (out.embedding == "rad_dino")]
    cols = ["probe", "score", "eval", "n_cal", "n_eval", "marginal_cov", "tb_cov",
            "nontb_cov", "mean_size", "empty", "disparity",
            "marg_lo95", "marg_hi95"]
    print("HELD-OUT HEADLINE (rad_dino, Mondrian, alpha=0.10, target marginal 90%):")
    print(head[cols].to_string(index=False))


if __name__ == "__main__":
    main()
