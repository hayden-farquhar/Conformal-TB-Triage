"""
Held-out-calibration CPU sensitivity analyses.
Supplementary tables S-noise / S-seed / S-resplit.

These three analyses were originally computed under the in-sample-calibration
pipeline (src/evaluation/sensitivity_analyses.py), which set Mondrian thresholds
on the probe-TRAINING split -- the defect this revision addresses -- and required
the Drive embeddings to retrain probes. The held-out versions below operate
purely on the saved held-out predictions (results/probe_predictions.parquet,
no GPU / no embedding re-run) and perturb the single operative source of
randomness in the held-out pipeline: the conformal-calibration draw out of the
held-out TBX11K `dev` pool, and the quality of its labels. The TBX11K `test`
evaluation set is held FIXED throughout.

Primary config (matches manuscript Table 3 headline): RAD-DINO, linear probe,
RAW scores, Mondrian, alpha = 0.10. The dev pool (n = 2,520) is split 50/50
(stratified) into a recalibration half and a conformal-calibration half; under
raw scoring only the conformal-calibration half is used to set thresholds.

GPU-dependent analyses (image-degradation, TTA) are NOT recomputed here; they
require re-extracting embeddings on perturbed images in a Colab GPU session.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from conformal_pipeline import (
    mondrian_thresholds,
    sets_from_mondrian,
    eval_sets,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
TABLES_DIR = REPO_ROOT / "outputs" / "tables"
PRED_PATH = REPO_ROOT / "outputs" / "probe_predictions.parquet"

SEED = 42
EMB = "rad_dino"
PROBE = "linear"
ALPHA = 0.10
TARGET = 1 - ALPHA  # 0.90


def load_primary():
    """Return (dev_prob, dev_y, test_prob, test_y) for the primary config,
    raw scores, from saved out-of-sample predictions."""
    d = pd.read_parquet(PRED_PATH)
    d = d[(d.embedding == EMB) & (d.probe == PROBE)]
    dev = d[d.split == "dev"]
    test = d[d.split == "test"]
    return (
        dev.y_prob.to_numpy(),
        dev.y_true.to_numpy().astype(int),
        test.y_prob.to_numpy(),
        test.y_true.to_numpy().astype(int),
    )


def conf_half(dev_p, dev_y, seed):
    """The conformal-calibration half of the held-out dev pool, under a given
    split seed (identical procedure to conformal_pipeline; SEED=42 reproduces
    the headline)."""
    idx = np.arange(len(dev_p))
    _, conf_idx = train_test_split(
        idx, test_size=0.5, random_state=seed, stratify=dev_y
    )
    return dev_p[conf_idx], dev_y[conf_idx]


def evaluate(cal_p, cal_y, test_p, test_y):
    thr = mondrian_thresholds(cal_p, cal_y, ALPHA)
    inc0, inc1 = sets_from_mondrian(thr, test_p)
    return eval_sets(inc0, inc1, test_y)


# ---------------------------------------------------------------- 1. Label noise
def run_label_noise(dev_p, dev_y, test_p, test_y):
    """Flip a fraction of the CONFORMAL-CALIBRATION labels (the labels that set
    the Mondrian thresholds) and measure coverage degradation on the fixed test
    set. This probes how robust conformal validity is to mislabeled calibration
    data -- the on-thesis label-noise question for a calibration study."""
    cal_p, cal_y = conf_half(dev_p, dev_y, SEED)
    fracs = [0.00, 0.05, 0.10, 0.15, 0.20]
    n_reps = 50
    rows = []
    for frac in fracs:
        margs, tbs, ntbs, empties, singles = [], [], [], [], []
        for rep in range(n_reps):
            rng = np.random.default_rng(SEED + rep)
            yy = cal_y.copy()
            n_flip = int(round(len(yy) * frac))
            if n_flip > 0:
                flip = rng.choice(len(yy), n_flip, replace=False)
                yy[flip] = 1 - yy[flip]
            m = evaluate(cal_p, yy, test_p, test_y)
            margs.append(m["marginal_cov"]); tbs.append(m["tb_cov"])
            ntbs.append(m["nontb_cov"]); empties.append(m["empty"])
            singles.append(m["singleton"])
        rows.append({
            "noise_frac": frac, "n_cal": len(cal_y), "n_reps": n_reps,
            "marginal_mean": round(np.mean(margs), 4), "marginal_std": round(np.std(margs), 4),
            "tb_cov_mean": round(np.mean(tbs), 4), "tb_cov_std": round(np.std(tbs), 4),
            "nontb_cov_mean": round(np.mean(ntbs), 4),
            "empty_mean": round(np.mean(empties), 4),
            "singleton_mean": round(np.mean(singles), 4),
        })
        print(f"  noise {frac:.0%}: marginal={np.mean(margs):.3f}±{np.std(margs):.3f} "
              f"TB={np.mean(tbs):.3f} empty={np.mean(empties):.3f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(TABLES_DIR / "label_noise.csv", index=False)
    base = df.loc[df.noise_frac == 0.0, "marginal_mean"].iloc[0]
    drop20 = base - df.loc[df.noise_frac == 0.20, "marginal_mean"].iloc[0]
    print(f"  -> marginal coverage drops {drop20:.3f} from 0% to 20% calibration-label noise")
    return df


# ---------------------------------------------------------------- 2. Seed stability
def run_seed_stability(dev_p, dev_y, test_p, test_y):
    """Vary the dev 50/50 conformal-split seed across a fixed panel of seeds and
    measure the headline operating point on the fixed test set. Answers: is the
    91.4% marginal-coverage headline contingent on SEED=42?"""
    seeds = [42, 123, 456, 789, 1024, 2025, 7, 31337]
    rows = []
    for s in seeds:
        cal_p, cal_y = conf_half(dev_p, dev_y, s)
        m = evaluate(cal_p, cal_y, test_p, test_y)
        rows.append({"seed": s, "n_cal": len(cal_y), **m})
        print(f"  seed {s}: marginal={m['marginal_cov']:.3f} TB={m['tb_cov']:.3f} "
              f"empty={m['empty']:.3f} singleton={m['singleton']:.3f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(TABLES_DIR / "seed_stability.csv", index=False)
    msd = df.marginal_cov.std()
    tsd = df.tb_cov.std()
    all_marg_target = bool((df.marginal_cov >= 0.85).all())
    all_empty_zero = bool((df["empty"] <= 0.005).all())
    print(f"  -> marginal SD={msd:.4f}, TB SD={tsd:.4f}, "
          f"all marginal>=0.85: {all_marg_target}, all empty~0: {all_empty_zero}")
    return df


# ---------------------------------------------------------------- 3. Meta-coverage
def run_meta_coverage(dev_p, dev_y, test_p, test_y):
    """200 random redraws of the dev 50/50 conformal split (resampling which
    held-out points calibrate the thresholds), evaluated on the FIXED test set.
    Held-out replacement for the original 'mean ~75%, 0/200 reach >=90%'
    resplit-fragility finding, which was an artefact of the in-sample defect."""
    n_resplits = 200
    margs, tbs, empties = [], [], []
    for i in range(n_resplits):
        cal_p, cal_y = conf_half(dev_p, dev_y, 10_000 + i)
        m = evaluate(cal_p, cal_y, test_p, test_y)
        margs.append(m["marginal_cov"]); tbs.append(m["tb_cov"]); empties.append(m["empty"])
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{n_resplits} resplits (mean marginal {np.mean(margs):.3f})", flush=True)
    margs, tbs, empties = map(np.array, (margs, tbs, empties))
    summary = pd.DataFrame([{
        "n_resplits": n_resplits,
        "marginal_mean": round(margs.mean(), 4), "marginal_std": round(margs.std(), 4),
        "marginal_min": round(margs.min(), 4), "marginal_max": round(margs.max(), 4),
        "tb_mean": round(tbs.mean(), 4), "tb_std": round(tbs.std(), 4),
        "frac_marginal_ge90": round(float((margs >= 0.90).mean()), 4),
        "frac_tb_ge90": round(float((tbs >= 0.90).mean()), 4),
        "empty_mean": round(empties.mean(), 4),
    }])
    summary.to_csv(TABLES_DIR / "meta_coverage.csv", index=False)
    pd.DataFrame({"resplit": np.arange(n_resplits), "marginal_cov": margs,
                  "tb_cov": tbs, "empty": empties}).to_csv(
        TABLES_DIR / "meta_coverage_draws.csv", index=False)
    print(f"  -> marginal {margs.mean():.3f}±{margs.std():.3f}; "
          f"{(margs >= 0.90).mean():.0%} of resplits reach >=90% marginal "
          f"(original in-sample design: 0/200)")
    return summary


# ---------------------------------------------------------------- 4. Calibration-set size
def run_calset_sensitivity(dev_p, dev_y, test_p, test_y):
    """Held-out calibration-set-size sensitivity. Under the held-out design the
    operative 'calibration set' is the conformal-calibration pool, NOT the
    probe-training data. Subsample that held-out pool (stratified) at increasing
    fractions, keeping the probe and the test set FIXED, and measure how many
    held-out exchangeable points are needed for stable coverage. The original
    S5 analysis (src/evaluation/core_analyses.py:run_calset_sensitivity) instead
    scored the probe on its own training subsample to build the calibration
    scores -- the in-sample defect -- so its '89.3% +/- 4.2%' is not a valid
    estimate of the held-out calibration-size requirement."""
    cal_p_full, cal_y_full = conf_half(dev_p, dev_y, SEED)
    fracs = [0.10, 0.20, 0.30, 0.50, 0.70, 1.00]
    n_reps = 10
    idx0 = np.where(cal_y_full == 0)[0]
    idx1 = np.where(cal_y_full == 1)[0]
    rows = []
    for frac in fracs:
        n_sub = max(int(len(cal_y_full) * frac), 20)
        margs, tbs, singles, empties = [], [], [], []
        for rep in range(n_reps):
            rng = np.random.default_rng(SEED + rep)
            n0 = max(int(round(n_sub * len(idx0) / len(cal_y_full))), 5)
            n1 = max(int(round(n_sub * len(idx1) / len(cal_y_full))), 5)
            sub = np.concatenate([
                rng.choice(idx0, min(n0, len(idx0)), replace=False),
                rng.choice(idx1, min(n1, len(idx1)), replace=False),
            ])
            m = evaluate(cal_p_full[sub], cal_y_full[sub], test_p, test_y)
            margs.append(m["marginal_cov"]); tbs.append(m["tb_cov"])
            singles.append(m["singleton"]); empties.append(m["empty"])
        rows.append({
            "fraction": frac, "n_cal": n_sub, "n_reps": n_reps,
            "marginal_mean": round(np.mean(margs), 4), "marginal_std": round(np.std(margs), 4),
            "tb_cov_mean": round(np.mean(tbs), 4), "tb_cov_std": round(np.std(tbs), 4),
            "singleton_mean": round(np.mean(singles), 4), "singleton_std": round(np.std(singles), 4),
            "empty_mean": round(np.mean(empties), 4),
        })
        print(f"  {frac:.0%} ({n_sub:>4} cal): marginal={np.mean(margs):.3f}±{np.std(margs):.3f} "
              f"TB={np.mean(tbs):.3f}±{np.std(tbs):.3f} empty={np.mean(empties):.3f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(TABLES_DIR / "calset_sensitivity.csv", index=False)
    stable = df[df.tb_cov_std < 0.02]
    if len(stable):
        print(f"  -> TB coverage stable (SD<0.02) from n_cal >= {int(stable.iloc[0].n_cal)}")
    return df


FIGS = REPO_ROOT / "outputs" / "figures"


def make_figures():
    """Regenerate the two CPU-reproducible supplementary figures from the saved
    held-out-calibration CSVs, so the supplement is internally
    consistent with the held-out text."""
    ln = pd.read_csv(TABLES_DIR / "label_noise.csv")
    sd = pd.read_csv(TABLES_DIR / "seed_stability.csv")
    draws = pd.read_csv(TABLES_DIR / "meta_coverage_draws.csv")
    cs = pd.read_csv(TABLES_DIR / "calset_sensitivity.csv")

    # --- Label-noise figure ---
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    x = ln.noise_frac * 100
    ax.plot(x, ln.marginal_mean, "-o", color="#1D3557", lw=2, label="Marginal coverage")
    ax.fill_between(x, ln.marginal_mean - ln.marginal_std,
                    ln.marginal_mean + ln.marginal_std, color="#1D3557", alpha=0.15)
    ax.plot(x, ln.tb_cov_mean, "-s", color="#457B9D", lw=2, label="TB-class coverage")
    ax.plot(x, ln.empty_mean, "-^", color="#E63946", lw=2, label="Empty-set fraction")
    ax.axhline(TARGET, ls="--", color="grey", lw=1)
    ax.text(20, TARGET + 0.01, "90% target", ha="right", fontsize=8, color="grey")
    ax.set_xlabel("Conformal-calibration label corruption (%)")
    ax.set_ylabel("Proportion")
    ax.set_ylim(0, 1.05)
    ax.set_title("Coverage robustness to calibration-label noise\n"
                 "(held-out calibration; probe fixed)")
    ax.legend(loc="center right", fontsize=9, frameon=False)
    fig.tight_layout()
    fig.savefig(FIGS / "sfig_label_noise.pdf")
    fig.savefig(FIGS / "sfig_label_noise.png", dpi=200)
    plt.close(fig)

    # --- Seed + resplit stability figure (two panels) ---
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(10.4, 4.6))
    xs = np.arange(len(sd))
    w = 0.4
    axL.bar(xs - w / 2, sd.marginal_cov, w, color="#1D3557", label="Marginal")
    axL.bar(xs + w / 2, sd.tb_cov, w, color="#457B9D", label="TB-class")
    axL.axhline(TARGET, ls="--", color="grey", lw=1)
    axL.set_xticks(xs)
    axL.set_xticklabels([str(s) for s in sd.seed], rotation=45, fontsize=8)
    axL.set_ylim(0.80, 1.0)
    axL.set_xlabel("Conformal-split seed")
    axL.set_ylabel("Coverage")
    axL.set_title(f"Seed stability (SD: marginal {sd.marginal_cov.std():.3f}, "
                  f"TB {sd.tb_cov.std():.3f})")
    axL.legend(loc="lower right", fontsize=9, frameon=False)

    axR.hist(draws.marginal_cov, bins=24, color="#457B9D", alpha=0.85,
             edgecolor="white")
    axR.axvline(TARGET, ls="--", color="#E63946", lw=1.4, label="90% target")
    axR.axvline(draws.marginal_cov.mean(), ls="-", color="#1D3557", lw=1.6,
                label=f"Mean {draws.marginal_cov.mean():.3f}")
    frac_ge90 = (draws.marginal_cov >= 0.90).mean()
    axR.set_xlabel("Marginal coverage (fixed test set)")
    axR.set_ylabel("Resplits")
    axR.set_title(f"200 conformal resplits — {frac_ge90:.0%} reach ≥90%\n"
                  "(in-sample design: 0/200)")
    axR.legend(loc="upper left", fontsize=9, frameon=False)
    fig.tight_layout()
    fig.savefig(FIGS / "sfig_seed_resplit.pdf")
    fig.savefig(FIGS / "sfig_seed_resplit.png", dpi=200)
    plt.close(fig)

    # --- Calibration-set-size figure ---
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.errorbar(cs.n_cal, cs.marginal_mean, yerr=cs.marginal_std, fmt="-o",
                color="#1D3557", lw=2, capsize=3, label="Marginal coverage")
    ax.errorbar(cs.n_cal, cs.tb_cov_mean, yerr=cs.tb_cov_std, fmt="-s",
                color="#457B9D", lw=2, capsize=3, label="TB-class coverage")
    ax.axhline(TARGET, ls="--", color="grey", lw=1)
    ax.text(cs.n_cal.max(), TARGET + 0.005, "90% target", ha="right",
            fontsize=8, color="grey")
    ax.set_xlabel("Held-out conformal-calibration samples (n)")
    ax.set_ylabel("Coverage")
    ax.set_ylim(0.80, 1.0)
    ax.set_title("Coverage vs held-out calibration-set size\n"
                 "(held-out design; probe and test fixed)")
    ax.legend(loc="lower right", fontsize=9, frameon=False)
    fig.tight_layout()
    fig.savefig(FIGS / "sfig_calset_sensitivity.pdf")
    fig.savefig(FIGS / "sfig_calset_sensitivity.png", dpi=200)
    plt.close(fig)
    print("Figures written: sfig_label_noise, sfig_seed_resplit, "
          "sfig_calset_sensitivity.")


def main():
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)
    dev_p, dev_y, test_p, test_y = load_primary()
    print(f"Loaded held-out dev (n={len(dev_p)}) and fixed test (n={len(test_p)}); "
          f"primary config {EMB}/{PROBE}/raw/Mondrian/alpha={ALPHA}")
    print("=== 1. Label-noise tolerance (conformal-calibration labels) ===")
    run_label_noise(dev_p, dev_y, test_p, test_y)
    print("=== 2. Seed stability (conformal-split seed) ===")
    run_seed_stability(dev_p, dev_y, test_p, test_y)
    print("=== 3. Meta-coverage (200 conformal resplits) ===")
    run_meta_coverage(dev_p, dev_y, test_p, test_y)
    print("=== 4. Calibration-set-size sensitivity (held-out pool) ===")
    run_calset_sensitivity(dev_p, dev_y, test_p, test_y)
    print("=== 5. Figures ===")
    make_figures()
    print("Done. Wrote label_noise.csv, seed_stability.csv, "
          "meta_coverage.csv (+_draws), calset_sensitivity.csv "
          "and 3 supplementary figures.")


if __name__ == "__main__":
    main()
