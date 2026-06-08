"""
Round-2 recompute: emit the REAL numbers needed to reconcile the manuscript and
supplement after the held-out conformal-calibration correction.

Reuses the exact conformal maths from corrected_pipeline.py (no re-implementation).
Everything here runs from saved out-of-sample probe predictions -- no GPU /
embedding re-run -- because dev/test/ext predictions are already out-of-sample for
the NLM-trained probe.

Outputs (printed; nothing is written to manuscript files by this script):
  (a) Table 2 AUROC: 4 working models x {linear, mlp} on the TBX11K test split
  (b) Multiclass mean P(TB) per clinical group (rad_dino linear, test)
  (c) Subgroup coverage + bootstrap CIs per clinical group
      (raw Mondrian alpha=0.10, held-out design)
  (d) Figure 3 L2-RAD-DINO-linear: in-sample vs held-out coverage (raw + isotonic)
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import train_test_split

import sys
SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
from corrected_pipeline import (
    mondrian_thresholds, sets_from_mondrian, eval_sets, SEED,
)

PROJ = SCRIPTS.parents[1]                      # project root (68 Conformal TB Triage)
PRED = PROJ / "results" / "probe_predictions.parquet"
PRED_L2 = PROJ / "results" / "probe_predictions_l2_rad_dino_linear.parquet"
SPLITS = PROJ / "repository" / "data" / "processed" / "splits.parquet"

WORKING_MODELS = ["rad_dino", "biomedclip", "torchxrayvision", "dinov2"]
ALPHA = 0.10
N_BOOT = 1000


def auroc(y, p):
    return roc_auc_score(y, p)


def boot_auroc_ci(y, p, n_boot=N_BOOT):
    rng = np.random.default_rng(SEED)
    n = len(y)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yy, pp = y[idx], p[idx]
        if len(np.unique(yy)) < 2:
            continue
        vals.append(roc_auc_score(yy, pp))
    return round(float(np.percentile(vals, 2.5)), 4), round(float(np.percentile(vals, 97.5)), 4)


def dev_halves(dev_p, dev_y):
    idx = np.arange(len(dev_p))
    recal_idx, conf_idx = train_test_split(
        idx, test_size=0.5, random_state=SEED, stratify=dev_y)
    return recal_idx, conf_idx


def main():
    df = pd.read_parquet(PRED)
    sp = pd.read_parquet(SPLITS)[["patient_id", "label", "tb_binary"]]

    # ---------------------------------------------------------------
    # (a) Table 2 AUROC -- discrimination on the test split
    # ---------------------------------------------------------------
    print("=" * 72)
    print("(a) TABLE 2 AUROC  (TBX11K test split, y_true in {0,1})")
    print("=" * 72)
    print(f"{'model':<18}{'probe':<8}{'n':>7}{'AUROC':>9}  95% CI")
    for emb in WORKING_MODELS:
        for probe in ["linear", "mlp"]:
            d = df[(df.embedding == emb) & (df.probe == probe) & (df.split == "test")]
            if d.empty:
                continue
            y = d.y_true.to_numpy().astype(int)
            p = d.y_prob.to_numpy()
            a = auroc(y, p)
            lo, hi = boot_auroc_ci(y, p)
            print(f"{emb:<18}{probe:<8}{len(y):>7}{a:>9.4f}  [{lo:.4f}, {hi:.4f}]")

    # ---------------------------------------------------------------
    # (b) Multiclass mean P(TB) per clinical group (rad_dino linear, test)
    # ---------------------------------------------------------------
    print("\n" + "=" * 72)
    print("(b) MULTICLASS mean predicted P(TB) per clinical group")
    print("    (rad_dino linear, raw probabilities, test split)")
    print("=" * 72)
    rl = df[(df.embedding == "rad_dino") & (df.probe == "linear") & (df.split == "test")]
    rl = rl.merge(sp, on="patient_id", how="left")
    for grp in ["active_tb", "latent_tb", "sick_non_tb", "healthy", "normal"]:
        g = rl[rl.label == grp]
        if g.empty:
            continue
        print(f"  {grp:<14} n={len(g):>5}  mean P(TB)={g.y_prob.mean():.4f}  "
              f"median={g.y_prob.median():.4f}")

    # ---------------------------------------------------------------
    # Build the held-out conformal calibration (rad_dino linear) once
    # ---------------------------------------------------------------
    rl_dev = df[(df.embedding == "rad_dino") & (df.probe == "linear") & (df.split == "dev")]
    dev_p = rl_dev.y_prob.to_numpy()
    dev_y = rl_dev.y_true.to_numpy().astype(int)
    recal_idx, conf_idx = dev_halves(dev_p, dev_y)
    conf_p_raw, conf_y = dev_p[conf_idx], dev_y[conf_idx]

    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(dev_p[recal_idx], dev_y[recal_idx])

    test_p = rl.y_prob.to_numpy()
    test_y = rl.y_true.to_numpy().astype(int)
    test_label = rl.label.to_numpy()

    thr_raw = mondrian_thresholds(conf_p_raw, conf_y, ALPHA)
    inc0_raw, inc1_raw = sets_from_mondrian(thr_raw, test_p)
    covered_raw = np.where(test_y == 1, inc1_raw, inc0_raw)

    # ---------------------------------------------------------------
    # (c) Subgroup coverage + bootstrap CIs (raw Mondrian, held-out)
    #     CI method A: bootstrap the test subgroup members (fixed thresholds)
    #     CI method B: bootstrap the conformal calibration set (fixed test) --
    #                  matches the headline marginal-CI diagnostic
    # ---------------------------------------------------------------
    print("\n" + "=" * 72)
    print("(c) SUBGROUP coverage  (rad_dino linear, raw Mondrian, alpha=0.10, held-out)")
    print("    point = coverage on test subgroup; CI_test = bootstrap subgroup members")
    print("=" * 72)
    rng = np.random.default_rng(SEED)
    print(f"{'group':<14}{'n':>6}{'coverage':>10}  CI_test(95%)")
    for grp in ["active_tb", "latent_tb", "sick_non_tb", "healthy"]:
        mask = test_label == grp
        cov_g = covered_raw[mask]
        n = int(mask.sum())
        if n == 0:
            continue
        point = cov_g.mean()
        boots = [cov_g[rng.integers(0, n, n)].mean() for _ in range(N_BOOT)]
        lo, hi = np.percentile(boots, 2.5), np.percentile(boots, 97.5)
        print(f"{grp:<14}{n:>6}{point:>10.4f}  [{lo:.4f}, {hi:.4f}]")

    # CI method B (calibration bootstrap) for the marginal -- sanity check vs headline
    print("\n  [sanity] marginal coverage via calibration bootstrap (fixed test):")
    margs = []
    nC = len(conf_p_raw)
    rngB = np.random.default_rng(SEED)
    for _ in range(N_BOOT):
        bi = rngB.integers(0, nC, nC)
        thr = mondrian_thresholds(conf_p_raw[bi], conf_y[bi], ALPHA)
        i0, i1 = sets_from_mondrian(thr, test_p)
        margs.append(np.where(test_y == 1, i1, i0).mean())
    print(f"    marginal point={covered_raw.mean():.4f}  "
          f"calib-boot CI=[{np.percentile(margs,2.5):.4f}, {np.percentile(margs,97.5):.4f}]")

    # ---------------------------------------------------------------
    # (d) Figure 3 L2-RAD-DINO-linear: in-sample vs held-out coverage
    # ---------------------------------------------------------------
    print("\n" + "=" * 72)
    print("(d) FIGURE 3  L2-normalised RAD-DINO linear: in-sample vs held-out")
    print("=" * 72)
    l2 = pd.read_parquet(PRED_L2)
    l2cal = l2[l2.split == "calibration"]
    l2dev = l2[l2.split == "dev"]
    l2test = l2[l2.split == "test"]
    cal_p = l2cal.y_prob.to_numpy(); cal_y = l2cal.y_true.to_numpy().astype(int)
    l2dev_p = l2dev.y_prob.to_numpy(); l2dev_y = l2dev.y_true.to_numpy().astype(int)
    l2test_p = l2test.y_prob.to_numpy(); l2test_y = l2test.y_true.to_numpy().astype(int)

    # in-sample design: conformal calib == probe-training calibration split
    thr_in = mondrian_thresholds(cal_p, cal_y, ALPHA)
    i0, i1 = sets_from_mondrian(thr_in, l2test_p)
    print("  IN-SAMPLE (conformal calib = NLM calibration split, n=%d):" % len(cal_y))
    print("   ", eval_sets(i0, i1, l2test_y))

    # held-out design: conformal calib = held-out dev half (raw + isotonic)
    ridx, cidx = dev_halves(l2dev_p, l2dev_y)
    confp, confy = l2dev_p[cidx], l2dev_y[cidx]
    thr_ho = mondrian_thresholds(confp, confy, ALPHA)
    h0, h1 = sets_from_mondrian(thr_ho, l2test_p)
    print("  HELD-OUT raw (conformal calib = dev half, n=%d):" % len(confy))
    print("   ", eval_sets(h0, h1, l2test_y))

    iso2 = IsotonicRegression(out_of_bounds="clip")
    iso2.fit(l2dev_p[ridx], l2dev_y[ridx])
    thr_ho_iso = mondrian_thresholds(iso2.predict(confp), confy, ALPHA)
    hi0, hi1 = sets_from_mondrian(thr_ho_iso, iso2.predict(l2test_p))
    print("  HELD-OUT isotonic:")
    print("   ", eval_sets(hi0, hi1, l2test_y))


if __name__ == "__main__":
    main()
