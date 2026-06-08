"""
Generate missing supplementary figures (sFig 5, 8–15).
Matches the style of figures_and_final.py.

Run: python3 src/evaluation/supplementary_figures.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 10, "figure.dpi": 300, "savefig.bbox": "tight",
                      "axes.spines.top": False, "axes.spines.right": False})

from config import *


def save(fig, name):
    fig.savefig(FIGURES_DIR / f"{name}.png")
    fig.savefig(FIGURES_DIR / f"{name}.pdf")
    plt.close(fig)
    print(f"    Saved {name}.png + .pdf")


# ─────────────────────────────────────────────────────────────────────
# sFig 5: Image degradation sensitivity
# ─────────────────────────────────────────────────────────────────────
def sfig5_image_degradation():
    print("  sFig 5: Image degradation...", flush=True)
    df = pd.read_csv(TABLES_DIR / "image_degradation.csv")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Colour by degradation type
    colors = {"none": "#264653", "resolution": "#E63946", "jpeg": "#457B9D",
              "noise": "#2A9D8F", "brightness": "#E9C46A"}

    # Each row gets a sequential x position
    x = np.arange(len(df))
    for deg_type in df["degradation"].unique():
        mask = df["degradation"] == deg_type
        idx = np.where(mask)[0]
        ax1.plot(idx, df.loc[mask, "auroc"], "o-", color=colors.get(deg_type, "gray"),
                 label=deg_type, lw=1.5, ms=6)
        ax2.plot(idx, df.loc[mask, "tb_cov"], "s-", color=colors.get(deg_type, "gray"),
                 label=deg_type, lw=1.5, ms=6)

    labels = df["level"].tolist()
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)

    ax1.axhline(df.iloc[0]["auroc"], ls="--", color="gray", alpha=0.5, lw=0.8)
    ax2.axhline(0.90, ls="--", color="gray", alpha=0.5, lw=0.8, label="WHO TPP target")
    ax2.axhline(0.85, ls=":", color="red", alpha=0.5, lw=0.8, label="85% threshold")

    ax1.set_ylabel("AUROC")
    ax1.set_title("Discrimination Under Degradation")
    ax1.legend(fontsize=7, loc="lower left")

    ax2.set_ylabel("TB-Class Coverage")
    ax2.set_title("Conformal Coverage Under Degradation")
    ax2.legend(fontsize=7, loc="lower left")

    fig.suptitle("Image Quality Degradation Sensitivity", fontsize=11)
    fig.tight_layout()
    save(fig, "sfig5_image_degradation")


# ─────────────────────────────────────────────────────────────────────
# sFig 8: Shortcut risk ratio and dataset-clustering
# ─────────────────────────────────────────────────────────────────────
def sfig8_shortcut():
    print("  sFig 8: Shortcut detection...", flush=True)
    df = pd.read_csv(TABLES_DIR / "shortcut_detection.csv")

    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(df))
    w = 0.35

    bars1 = ax.bar(x - w/2, df["dataset_auroc"], w, label="Dataset-origin AUROC",
                   color="#457B9D", alpha=0.85)
    bars2 = ax.bar(x + w/2, df["tb_auroc"], w, label="TB-classification AUROC",
                   color="#E63946", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS.get(e, e) for e in df["embedding"]], fontsize=9)
    ax.set_ylabel("AUROC")
    ax.set_ylim(0.90, 1.01)
    ax.axhline(1.0, ls="--", color="gray", alpha=0.4, lw=0.8)
    ax.legend(fontsize=8)

    # Annotate shortcut ratio
    for i, row in df.iterrows():
        ax.annotate(f"ratio={row['shortcut_ratio']:.3f}",
                    (i, max(row["dataset_auroc"], row["tb_auroc"]) + 0.002),
                    ha="center", fontsize=7, color="black")

    ax.set_title("Shortcut Risk — Dataset-Origin vs TB Classification AUROC")
    fig.tight_layout()
    save(fig, "sfig8_shortcut")


# ─────────────────────────────────────────────────────────────────────
# sFig 9: Drift monitoring simulation
# ─────────────────────────────────────────────────────────────────────
def sfig9_drift():
    print("  sFig 9: Drift monitoring...", flush=True)
    df = pd.read_csv(TABLES_DIR / "drift_monitoring.csv")

    fig, ax = plt.subplots(figsize=(10, 5))

    dataset_colors = {"shenzhen": "#E63946", "montgomery": "#457B9D",
                      "tbx11k": "#2A9D8F", "pakistan": "#E9C46A"}

    # Plot MMD over sequential windows, colour by dataset
    cumulative_idx = 0
    boundaries = []
    prev_ds = None
    for _, row in df.iterrows():
        color = dataset_colors.get(row["dataset"], "gray")
        marker = "^" if row["drift_detected"] else "o"
        ax.plot(cumulative_idx, row["mmd2"], marker, color=color, ms=3, alpha=0.7)
        if row["dataset"] != prev_ds:
            if prev_ds is not None:
                boundaries.append(cumulative_idx)
            prev_ds = row["dataset"]
        cumulative_idx += 1

    # Mark dataset transitions
    for b in boundaries:
        ax.axvline(b, ls="--", color="gray", alpha=0.5, lw=0.8)

    # Legend for datasets
    for ds, color in dataset_colors.items():
        ax.plot([], [], "o", color=color, ms=5, label=ds.capitalize())
    ax.plot([], [], "^", color="black", ms=5, label="Drift detected")

    # Annotate dataset regions
    regions = df.groupby("dataset", sort=False).size()
    start = 0
    for ds, count in regions.items():
        mid = start + count / 2
        ax.text(mid, ax.get_ylim()[1] * 0.95, ds.capitalize(),
                ha="center", fontsize=8, style="italic", alpha=0.7)
        start += count

    ax.set_xlabel("Sequential Window Index")
    ax.set_ylabel("MMD² Statistic")
    ax.set_title("Embedding Drift Monitoring Simulation")
    ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    save(fig, "sfig9_drift")


# ─────────────────────────────────────────────────────────────────────
# sFig 10: SHAP summary (top-50 embedding dimensions)
# ─────────────────────────────────────────────────────────────────────
def sfig10_shap():
    print("  sFig 10: SHAP feature importance...", flush=True)
    df = pd.read_csv(TABLES_DIR / "shap_feature_importance.csv")
    df = df.sort_values("importance", ascending=True)  # ascending for horizontal bar

    fig, ax = plt.subplots(figsize=(7, 10))
    ax.barh(range(len(df)), df["importance"], color="#E63946", alpha=0.8)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels([f"dim {int(d)}" for d in df["dim"]], fontsize=6)
    ax.set_xlabel("Mean |SHAP Value|")
    ax.set_title("RAD-DINO Embedding Dimension Importance\n(XGBoost Probe, Top 50)")
    fig.tight_layout()
    save(fig, "sfig10_shap")


# ─────────────────────────────────────────────────────────────────────
# sFig 11: Venn-ABERS probability interval widths
# ─────────────────────────────────────────────────────────────────────
def sfig11_venn_abers():
    print("  sFig 11: Venn-ABERS intervals...", flush=True)
    df = pd.read_csv(TABLES_DIR / "venn_abers.csv")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Panel A: width distribution by true label
    for label, color, name in [(1, "#E63946", "TB+"), (0, "#457B9D", "TB−")]:
        sub = df[df["y_true"] == label]
        ax1.hist(sub["width"], bins=40, alpha=0.6, color=color, label=f"{name} (n={len(sub)})",
                 edgecolor="white", linewidth=0.3)

    ax1.set_xlabel("Probability Interval Width [p_upper − p_lower]")
    ax1.set_ylabel("Count")
    ax1.set_title("Interval Width Distribution by TB Status")
    ax1.legend(fontsize=8)

    # Panel B: p_lower vs p_upper scatter
    tb = df[df["y_true"] == 1]
    ntb = df[df["y_true"] == 0]
    ax2.scatter(ntb["p_lower"], ntb["p_upper"], c="#457B9D", s=2, alpha=0.2, label="TB−")
    ax2.scatter(tb["p_lower"], tb["p_upper"], c="#E63946", s=2, alpha=0.4, label="TB+")
    ax2.plot([0, 1], [0, 1], "--", color="gray", alpha=0.4, lw=0.8)
    ax2.set_xlabel("p_lower (P(TB))")
    ax2.set_ylabel("p_upper (P(TB))")
    ax2.set_title("Venn-ABERS Probability Intervals")
    ax2.legend(fontsize=8, markerscale=4)
    ax2.set_xlim(-0.02, 1.02)
    ax2.set_ylim(-0.02, 1.02)

    fig.suptitle("Venn-ABERS Conformal Probability Intervals", fontsize=11)
    fig.tight_layout()
    save(fig, "sfig11_venn_abers")


# ─────────────────────────────────────────────────────────────────────
# sFig 12: Non-TB pathology confusion (false positive analysis)
# ─────────────────────────────────────────────────────────────────────
def sfig12_nontb_confusion():
    print("  sFig 12: Non-TB confusion...", flush=True)
    df = pd.read_csv(TABLES_DIR / "nontb_confusion.csv")

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["#E63946", "#457B9D"]
    bars = ax.bar(range(len(df)), df["fp_rate"], color=colors[:len(df)], alpha=0.85)
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df["label"].str.replace("_", " ").str.title(), fontsize=10)
    ax.set_ylabel("False Positive Rate")
    ax.set_ylim(0, max(df["fp_rate"]) * 1.2)

    # Annotate bars
    for i, row in df.iterrows():
        ax.annotate(f"{row['n_false_positive']}/{row['n_total']}\n({row['fp_rate']:.1%})",
                    (i, row["fp_rate"] + 0.01), ha="center", fontsize=9)

    ax.set_title("False Positive Rate by Non-TB Category\n(RAD-DINO + Linear Probe on TBX11K Test)")
    fig.tight_layout()
    save(fig, "sfig12_nontb_confusion")


# ─────────────────────────────────────────────────────────────────────
# sFig 13: Seed stability
# ─────────────────────────────────────────────────────────────────────
def sfig13_seed_stability():
    print("  sFig 13: Seed stability...", flush=True)
    df = pd.read_csv(TABLES_DIR / "seed_stability.csv")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    metrics = [("auroc", "AUROC"), ("tb_cov", "TB Coverage"), ("singleton", "Singleton Fraction")]
    colors = ["#E63946", "#457B9D", "#2A9D8F"]

    for ax, (col, label), color in zip(axes, metrics, colors):
        vals = df[col].values
        seeds = df["seed"].astype(str).values
        ax.bar(range(len(vals)), vals, color=color, alpha=0.8)
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels([f"seed={s}" for s in seeds], fontsize=7, rotation=30)
        ax.set_ylabel(label)

        # Mean ± SD annotation
        mu, sd = vals.mean(), vals.std()
        ax.axhline(mu, ls="--", color="gray", alpha=0.6, lw=0.8)
        ax.set_title(f"{label}\n(mean={mu:.4f}, SD={sd:.4f})")

        # Set y-range to highlight variation
        margin = max(sd * 4, 0.01)
        ax.set_ylim(mu - margin, mu + margin)

    fig.suptitle("Seed Stability (RAD-DINO + Linear Probe + Mondrian)", fontsize=11)
    fig.tight_layout()
    save(fig, "sfig13_seed_stability")


# ─────────────────────────────────────────────────────────────────────
# sFig 14: Geographic representation gap
# ─────────────────────────────────────────────────────────────────────
def sfig14_geographic_gap():
    print("  sFig 14: Geographic gap...", flush=True)
    df = pd.read_csv(TABLES_DIR / "geographic_gap.csv")
    df["represented"] = df["represented"].astype(str).str.strip().str.lower().isin(["yes", "true", "1"])
    df = df.sort_values("tb_incidence_2024", ascending=True)

    fig, ax = plt.subplots(figsize=(8, 10))
    colors = ["#2A9D8F" if r else "#E63946" for r in df["represented"]]
    ax.barh(range(len(df)), df["tb_incidence_2024"] / 1000, color=colors, alpha=0.8)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df["country"], fontsize=8)
    ax.set_xlabel("Estimated TB Incidence 2024 (thousands)")

    # Legend
    ax.barh([], [], color="#2A9D8F", alpha=0.8, label="Represented")
    ax.barh([], [], color="#E63946", alpha=0.8, label="Not represented")
    ax.legend(fontsize=9, loc="lower right")

    # Summary stats
    total = df["tb_incidence_2024"].sum()
    rep = df[df["represented"] == True]["tb_incidence_2024"].sum()
    n_rep = df["represented"].sum()
    ax.text(0.97, 0.05,
            f"Represented: {n_rep}/{len(df)} countries\n"
            f"TB incidence covered: {rep/total:.1%}",
            transform=ax.transAxes, ha="right", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    ax.set_title("Geographic Representation of WHO High-Burden TB Countries", fontsize=10)
    fig.tight_layout()
    save(fig, "sfig14_geographic_gap")


# ─────────────────────────────────────────────────────────────────────
# sFig 15: Computational cost — Latency vs AUROC Pareto
# ─────────────────────────────────────────────────────────────────────
def sfig15_computational_cost():
    print("  sFig 15: Computational cost...", flush=True)
    df = pd.read_csv(TABLES_DIR / "computational_cost.csv")

    # Only include the 4 working models (exclude eva_x, chexzero, gloria)
    working = ["rad_dino", "biomedclip", "torchxrayvision", "dinov2"]
    df_w = df[df["model"].isin(working)].copy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Panel A: model size comparison
    x = np.arange(len(df_w))
    ax1.bar(x - 0.2, df_w["params_M"], 0.35, label="Parameters (M)", color="#E63946", alpha=0.8)
    ax1.bar(x + 0.2, df_w["weight_size_MB"], 0.35, label="Weight File (MB)", color="#457B9D", alpha=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels([MODEL_LABELS.get(m, m) for m in df_w["model"]], fontsize=9)
    ax1.set_ylabel("Size")
    ax1.set_title("Model Size Comparison")
    ax1.legend(fontsize=8)

    # Panel B: embedding dim vs file size scatter
    for _, row in df.iterrows():
        color = MODEL_COLORS.get(row["model"], "#AAAAAA")
        label = MODEL_LABELS.get(row["model"], row["model"])
        alpha = 1.0 if row["model"] in working else 0.3
        ax2.scatter(row["embed_dim"], row["embedding_file_MB"], c=color, s=100,
                    alpha=alpha, zorder=5, edgecolors="black", linewidth=0.5)
        ax2.annotate(label, (row["embed_dim"], row["embedding_file_MB"]),
                     textcoords="offset points", xytext=(5, 5), fontsize=7, alpha=alpha)

    ax2.set_xlabel("Embedding Dimensionality")
    ax2.set_ylabel("Embedding File Size (MB)")
    ax2.set_title("Embedding Dimensionality vs Storage")

    fig.suptitle("Computational Cost and Deployment Footprint", fontsize=11)
    fig.tight_layout()
    save(fig, "sfig15_computational_cost")


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating supplementary figures...", flush=True)
    # Image-degradation and seed/resplit supplementary figures are owned by the
    # held-out generators (gpu_sensitivity_figures.py, conformal_sensitivity.py).
    sfig8_shortcut()
    sfig9_drift()
    sfig10_shap()
    sfig11_venn_abers()
    sfig12_nontb_confusion()
    sfig14_geographic_gap()
    sfig15_computational_cost()
    print("\nDone. All supplementary figures saved to results/figures/")
