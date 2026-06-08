"""
Held-out-calibration figures for the GPU-derived sensitivity
analyses (image degradation = S4, TTA, lung-segmentation, perturbation).

These analyses are recomputed in a Colab GPU session (perturbed images are
re-embedded through the frozen backbone); the resulting CSVs land in
the held-out-calibration tables in results/tables/. This module only renders them, so the
supplement is internally consistent with the held-out pipeline.

The original S4 generator (supplementary_figures.sfig5_image_degradation)
plotted AUROC and TB-class coverage ONLY. Under the held-out design that
readout is misleading: TB-class coverage can stay high (or even climb) while
MARGINAL coverage and the empty/singleton structure collapse. This generator
therefore plots marginal coverage alongside TB-class coverage so the
coverage-decoupling under degradation is visible rather than hidden.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
TABLES_DIR = REPO_ROOT / "outputs" / "tables"
FIGS = REPO_ROOT / "outputs" / "figures"

TARGET = 0.90
C_MARGINAL = "#1D3557"
C_TBCLASS = "#457B9D"
C_AUROC = "#2A9D8F"
C_EMPTY = "#E63946"

FAMILY_LABEL = {
    "none": "Baseline",
    "resolution": "Resolution",
    "jpeg": "JPEG",
    "noise": "Gaussian noise",
    "brightness": "Brightness",
}
FAMILY_ORDER = ["none", "resolution", "jpeg", "noise", "brightness"]


def make_degradation_figure():
    df = pd.read_csv(TABLES_DIR / "image_degradation.csv")
    df = df.assign(family=pd.Categorical(df.degradation, categories=FAMILY_ORDER,
                                         ordered=True))
    df = df.sort_values(["family"]).reset_index(drop=True)

    labels = [f"{FAMILY_LABEL[d]}\n{lv}" if d != "none" else "Baseline"
              for d, lv in zip(df.degradation, df.level)]
    x = np.arange(len(df))

    # family boundaries for light vertical separators / shading
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.0, 5.0))

    # --- Left: discrimination (AUROC) ---
    axL.plot(x, df.auroc, "-o", color=C_AUROC, lw=2)
    axL.axhline(0.90, ls="--", color="grey", lw=1)
    axL.text(len(df) - 0.5, 0.905, "0.90", ha="right", fontsize=8, color="grey")
    axL.set_ylim(0.70, 0.95)
    axL.set_ylabel("AUROC")
    axL.set_title("Discrimination under image-quality degradation")
    axL.set_xticks(x)
    axL.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)

    # --- Right: coverage decoupling (marginal vs TB-class) ---
    axR.plot(x, df.marginal_cov, "-o", color=C_MARGINAL, lw=2, label="Marginal coverage")
    axR.plot(x, df.tb_cov, "-s", color=C_TBCLASS, lw=2, label="TB-class coverage")
    axR.plot(x, df["empty"], "-^", color=C_EMPTY, lw=1.6, label="Empty-set fraction")
    axR.axhline(TARGET, ls="--", color="grey", lw=1)
    axR.text(len(df) - 0.5, TARGET + 0.015, "90% target", ha="right", fontsize=8, color="grey")
    axR.set_ylim(0.0, 1.05)
    axR.set_ylabel("Proportion")
    axR.set_title("Coverage decoupling under degradation\n"
                  "(TB-class can diverge from marginal in either direction)")
    axR.set_xticks(x)
    axR.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    axR.legend(loc="lower left", fontsize=9, frameon=False)

    # annotate the two diagnostic failure points
    i32 = df.index[(df.degradation == "resolution") & (df.level == "32x32")]
    if len(i32):
        k = int(i32[0])
        axR.annotate("TB-class collapses\n(0.11) while marginal\nstays ~0.92",
                     xy=(k, df.tb_cov.iloc[k]), xytext=(k - 0.3, 0.45),
                     fontsize=7, color=C_TBCLASS, ha="center",
                     arrowprops=dict(arrowstyle="->", color=C_TBCLASS, lw=1))
    iq10 = df.index[(df.degradation == "jpeg") & (df.level == "q10")]
    if len(iq10):
        k = int(iq10[0])
        axR.annotate("marginal collapses\n(0.745) while TB-class\nstays ~0.98",
                     xy=(k, df.marginal_cov.iloc[k]), xytext=(k + 0.2, 0.60),
                     fontsize=7, color=C_MARGINAL, ha="center",
                     arrowprops=dict(arrowstyle="->", color=C_MARGINAL, lw=1))

    for ax in (axL, axR):
        for b in (2.5, 6.5, 9.5):  # family separators after baseline/res/jpeg/noise
            ax.axvline(b, color="0.85", lw=0.8, zorder=0)

    fig.suptitle("Supplementary Figure S4. Image-quality degradation "
                 "(held-out calibration)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(FIGS / "sfig_image_degradation.pdf")
    fig.savefig(FIGS / "sfig_image_degradation.png", dpi=200)
    plt.close(fig)
    print("Figure written: sfig_image_degradation.{png,pdf}")


def main():
    FIGS.mkdir(parents=True, exist_ok=True)
    make_degradation_figure()


if __name__ == "__main__":
    main()
