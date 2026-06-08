"""
Figure 3 - In-sample vs held-out conformal calibration (provenance comparison).

Controlled comparison: the configuration is held FIXED at the paper's primary
config (RAD-DINO, linear probe, raw scores, Mondrian class-conditional, alpha=0.10)
and ONLY the conformal-calibration provenance is varied:

  - In-sample (defect): conformal calibration scores computed on the probe-training
                        (NLM calibration) split -- violates split-conformal
                        data-independence.
  - Held-out (valid):   conformal calibration scores computed on the held-out
                        TBX11K development split (out-of-sample for the probe).

Both arms are recomputed live from results/probe_predictions.parquet, so every
number on the figure is reproducible from the shipped artifact. No GPU / embedding
re-run is required.

Run: python3 fig3_provenance.py
"""
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
from corrected_pipeline import mondrian_thresholds, sets_from_mondrian, eval_sets, SEED
from sklearn.model_selection import train_test_split

PROJ = SCRIPTS.parents[1]
PRED = PROJ / "results" / "probe_predictions.parquet"

# Output surfaces (kept in sync). The legacy filename is retained because the
# manuscript-submission packaging and repo both reference it.
OUT_RESULTS = PROJ / "results" / "figures" / "fig3_efficiency_curves"
OUT_REPO = PROJ / "repository" / "outputs" / "figures" / "fig3_efficiency_curves"
OUT_SUBMISSION_PDF = PROJ / "manuscript" / "submission" / "Figure_3.pdf"

ALPHA = 0.10
C_IN = "#E63946"   # in-sample (defect) - red
C_HO = "#1D3557"   # held-out (valid) - navy
TARGET = 0.90


def compute():
    df = pd.read_parquet(PRED)
    d = df[(df.embedding == "rad_dino") & (df.probe == "linear")]
    cal = d[d.split == "calibration"]
    dev = d[d.split == "dev"]
    test = d[d.split == "test"]
    cp, cy = cal.y_prob.to_numpy(), cal.y_true.to_numpy().astype(int)
    dp, dy = dev.y_prob.to_numpy(), dev.y_true.to_numpy().astype(int)
    tp, ty = test.y_prob.to_numpy(), test.y_true.to_numpy().astype(int)

    # In-sample: conformal calibration on the probe-training split
    thr_in = mondrian_thresholds(cp, cy, ALPHA)
    i0, i1 = sets_from_mondrian(thr_in, tp)
    m_in = eval_sets(i0, i1, ty)

    # Held-out: conformal calibration on the held-out dev conformal half
    idx = np.arange(len(dp))
    _, conf_idx = train_test_split(idx, test_size=0.5, random_state=SEED, stratify=dy)
    thr_ho = mondrian_thresholds(dp[conf_idx], dy[conf_idx], ALPHA)
    h0, h1 = sets_from_mondrian(thr_ho, tp)
    m_ho = eval_sets(h0, h1, ty)
    return m_in, m_ho


def make_figure(m_in, m_ho):
    metrics = ["marginal_cov", "tb_cov", "nontb_cov", "empty"]
    labels = ["Marginal\ncoverage", "TB-class\ncoverage", "Non-TB\ncoverage", "Empty-set\nfraction"]
    in_vals = [m_in[k] for k in metrics]
    ho_vals = [m_ho[k] for k in metrics]

    x = np.arange(len(metrics))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8, 5))
    b1 = ax.bar(x - w/2, in_vals, w, color=C_IN, label="In-sample calibration (defect)")
    b2 = ax.bar(x + w/2, ho_vals, w, color=C_HO, label="Held-out calibration (valid)")

    ax.axhline(TARGET, ls="--", color="grey", lw=1.2)
    ax.text(len(metrics) - 0.5, TARGET + 0.015, "90% target", ha="right",
            fontsize=9, color="grey")

    for bars, vals in ((b1, in_vals), (b2, ho_vals)):
        for rect, v in zip(bars, vals):
            ax.text(rect.get_x() + rect.get_width()/2, v + 0.012, f"{v*100:.0f}%",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10, fontweight="bold")
    ax.set_ylabel("Proportion", fontsize=11)
    ax.set_ylim(0, 1.08)
    ax.set_title("RAD-DINO linear, raw Mondrian (α=0.10);\n"
                 "configuration fixed — only the calibration provenance differs",
                 fontsize=11, fontweight="bold")
    ax.legend(loc="upper center", fontsize=9, frameon=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    return fig


def main():
    m_in, m_ho = compute()
    print("Figure 3 (reproducible) numbers:")
    print(f"  In-sample (defect): marginal {m_in['marginal_cov']:.3f}  TB {m_in['tb_cov']:.3f}  "
          f"non-TB {m_in['nontb_cov']:.3f}  empty {m_in['empty']:.3f}  disparity {m_in['disparity']:.3f}")
    print(f"  Held-out (valid):   marginal {m_ho['marginal_cov']:.3f}  TB {m_ho['tb_cov']:.3f}  "
          f"non-TB {m_ho['nontb_cov']:.3f}  empty {m_ho['empty']:.3f}  disparity {m_ho['disparity']:.3f}")

    fig = make_figure(m_in, m_ho)
    for base in (OUT_RESULTS, OUT_REPO):
        base.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(f"{base}.png", dpi=300, bbox_inches="tight")
        fig.savefig(f"{base}.pdf", bbox_inches="tight")
    plt.close(fig)
    shutil.copyfile(f"{OUT_RESULTS}.pdf", OUT_SUBMISSION_PDF)
    print(f"Saved fig3_efficiency_curves.png/.pdf to results + repository/outputs; "
          f"copied PDF to {OUT_SUBMISSION_PDF.name}")


if __name__ == "__main__":
    main()
