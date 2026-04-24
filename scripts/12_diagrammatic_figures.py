"""
Generate Fig 1 (study design schematic) and sFig 1 (CONSORT flow).
Uses matplotlib only — no external rendering services.

Run: python3 src/evaluation/diagrammatic_figures.py
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({"font.size": 9, "figure.dpi": 300, "savefig.bbox": "tight"})

from config import *


def save(fig, name):
    fig.savefig(FIGURES_DIR / f"{name}.png")
    fig.savefig(FIGURES_DIR / f"{name}.pdf")
    plt.close(fig)
    print(f"  Saved {name}.png + .pdf")


# ─────────────────────────────────────────────────────────────────────
# Shared drawing helpers
# ─────────────────────────────────────────────────────────────────────
def draw_box(ax, x, y, w, h, text, facecolor="#f0f4f8", edgecolor="#457B9D",
             fontsize=8, fontweight="normal", alpha=1.0):
    """Draw a rounded box with centred text."""
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                         boxstyle="round,pad=0.02", linewidth=1.2,
                         facecolor=facecolor, edgecolor=edgecolor, alpha=alpha)
    ax.add_patch(box)
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
            fontweight=fontweight, wrap=True)


def draw_arrow(ax, x1, y1, x2, y2, color="#555555"):
    """Draw a simple arrow between two points."""
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.2))


# ─────────────────────────────────────────────────────────────────────
# Fig 1: Study Design Schematic
# ─────────────────────────────────────────────────────────────────────
def fig1_study_design():
    print("  Fig 1: Study design schematic...", flush=True)
    fig, ax = plt.subplots(figsize=(10, 14))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 15)
    ax.axis("off")

    # Colours
    C_DATA = ("#f0f4f8", "#457B9D")
    C_MODEL = ("#fef3f0", "#E63946")
    C_CONF = ("#f0f8f4", "#2A9D8F")
    C_OUT = ("#fef9f0", "#E9C46A")

    # ── Layer 1: Datasets ──
    y = 14.0
    ax.text(5, y + 0.5, "PUBLIC CXR DATASETS", ha="center", fontsize=10,
            fontweight="bold", color="#457B9D")

    datasets = [
        ("Shenzhen\nn = 662\nTB: 336", 1.5),
        ("Montgomery\nn = 138\nTB: 58", 3.5),
        ("TBX11K\nn = 11,701\nTB: 799", 5.5),
        ("Pakistan\nn = 3,008\nTB: 2,494", 7.5),
        ("CheXpert\nn = 5,000\nnon-TB", 9.2),
    ]
    for text, x in datasets:
        draw_box(ax, x, y - 0.7, 1.6, 1.0, text, *C_DATA, fontsize=7)

    # ── Layer 2: Splits ──
    y = 12.0
    ax.text(5, y + 0.5, "SPLIT ALLOCATION", ha="center", fontsize=10,
            fontweight="bold", color="#457B9D")

    splits = [
        ("Calibration\nn = 800\nSZ + MG", 1.5),
        ("Development\nn = 3,510\nTBX11K 30%", 3.5),
        ("Primary Test\nn = 8,191\nTBX11K 70%", 5.5),
        ("External Test\nn = 3,008\nPakistan", 7.5),
        ("Distractor\nn = 5,000\nCheXpert", 9.2),
    ]
    for text, x in splits:
        draw_box(ax, x, y - 0.7, 1.6, 1.0, text, *C_DATA, fontsize=7)

    # Arrows: datasets → splits
    for (_, xs), (_, xd) in zip(datasets, splits):
        draw_arrow(ax, xs, 12.8, xd, 12.5)

    # ── Layer 3: Frozen embeddings ──
    y = 9.8
    ax.text(5, y + 0.5, "FROZEN FOUNDATION-MODEL EMBEDDINGS", ha="center",
            fontsize=10, fontweight="bold", color="#E63946")

    models = [
        ("RAD-DINO\nViT-B/16, 768-d\nprimary", 2.0),
        ("BiomedCLIP\nViT-B/16, 512-d", 4.2),
        ("torchxrayvision\nDenseNet-121, 1024-d", 6.4),
        ("DINOv2-B\nViT-B/14, 768-d\nnon-medical control", 8.6),
    ]
    for text, x in models:
        draw_box(ax, x, y - 0.7, 1.9, 1.0, text, *C_MODEL, fontsize=7)

    # Wide arrow: splits → embeddings
    draw_arrow(ax, 5, 10.7, 5, 10.4)

    # ── Layer 4: Probes ──
    y = 7.8
    ax.text(5, y + 0.5, "PROBE CLASSIFIERS", ha="center", fontsize=10,
            fontweight="bold", color="#E63946")

    probes = [
        ("Linear Probe\n(primary)", 2.0),
        ("k-NN", 4.2),
        ("XGBoost", 6.4),
        ("MLP", 8.6),
    ]
    for text, x in probes:
        draw_box(ax, x, y - 0.5, 1.6, 0.7, text, *C_MODEL, fontsize=8)

    draw_arrow(ax, 5, 8.8, 5, 8.3)

    # ── Layer 5: Calibration ──
    y = 6.5
    draw_box(ax, 5, y, 3.0, 0.6, "Isotonic Probability Calibration",
             *C_CONF, fontsize=9)

    draw_arrow(ax, 5, 7.0, 5, 6.8)

    # ── Layer 6: Conformal prediction ──
    y = 5.3
    ax.text(5, y + 0.5, "CONFORMAL PREDICTION", ha="center", fontsize=10,
            fontweight="bold", color="#2A9D8F")

    methods = [
        ("APS", 1.5),
        ("RAPS", 3.2),
        ("Mondrian\n(primary)", 5.0),
        ("CRC\nFNR ≤ 0.10", 6.8),
        ("LTT\nJoint control", 8.5),
    ]
    for text, x in methods:
        draw_box(ax, x, y - 0.5, 1.4, 0.7, text, *C_CONF, fontsize=7)

    draw_arrow(ax, 5, 6.2, 5, 5.8)

    # ── Layer 7: Triage output ──
    y = 3.5
    ax.text(5, y + 0.5, "THREE-TIER TRIAGE OUTPUT", ha="center", fontsize=10,
            fontweight="bold", color="#E9C46A")

    tiers = [
        ("TIER 1: CLEAR\nPrediction set = {non-TB}\nDischarge", 2.0, "#d4edda", "#28a745"),
        ("TIER 2: REFER\nPrediction set = {TB}\nXpert MTB/RIF", 5.0, "#f8d7da", "#dc3545"),
        ("TIER 3: UNCERTAIN\nPrediction set = {TB, non-TB}\nClinical review", 8.0, "#fff3cd", "#ffc107"),
    ]
    for text, x, fc, ec in tiers:
        draw_box(ax, x, y - 0.5, 2.5, 0.9, text, fc, ec, fontsize=7, fontweight="bold")

    draw_arrow(ax, 5, 4.3, 5, 4.0)

    # ── Layer 8: WHO TPP ──
    y = 1.7
    draw_box(ax, 5, y, 5.0, 0.8,
             "WHO TPP ALIGNMENT\nSensitivity ≥ 90%  |  Specificity ≥ 70%\nConformal statistical guarantee",
             *C_OUT, fontsize=8, fontweight="bold")

    draw_arrow(ax, 5, 2.6, 5, 2.1)

    # ── Side annotations ──
    ax.text(0.3, 9.3, "Weights\nfrozen\n(no fine-\ntuning)", ha="center", fontsize=7,
            style="italic", color="#888888")
    ax.text(0.3, 5.0, "Calibrated\non n = 800\n(SZ + MG)", ha="center", fontsize=7,
            style="italic", color="#888888")

    fig.tight_layout()
    save(fig, "fig1_study_design")


# ─────────────────────────────────────────────────────────────────────
# sFig 1: CONSORT-Style Patient Flow Diagram
# ─────────────────────────────────────────────────────────────────────
def sfig1_consort():
    print("  sFig 1: CONSORT flow...", flush=True)
    fig, ax = plt.subplots(figsize=(12, 16))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 17)
    ax.axis("off")

    C_MAIN = ("#f0f4f8", "#457B9D")
    C_EXCL = ("#fef3f0", "#E63946")
    C_SPLIT = ("#f0f8f4", "#2A9D8F")

    ax.text(6, 16.5, "CONSORT-Style Patient Flow Diagram", ha="center",
            fontsize=12, fontweight="bold")

    # ── Row 1: Initial identification ──
    y = 15.5
    draw_box(ax, 6, y, 7, 0.7,
             "Images identified from 5 public CXR datasets\nN = 20,509",
             *C_MAIN, fontsize=9, fontweight="bold")

    # ── Row 2: Per-dataset breakdown ──
    y = 14.2
    ax.text(6, y + 0.5, "Breakdown by dataset", ha="center", fontsize=9,
            fontweight="bold", color="#457B9D")

    ds_boxes = [
        ("TBX11K\nn = 11,701", 1.5),
        ("Shenzhen\nn = 662", 3.8),
        ("Montgomery\nn = 138", 6.0),
        ("Pakistan\nn = 3,008", 8.2),
        ("CheXpert\nn = 5,000", 10.5),
    ]
    for text, x in ds_boxes:
        draw_box(ax, x, y - 0.4, 1.8, 0.7, text, *C_MAIN, fontsize=7)

    draw_arrow(ax, 6, 15.1, 6, 14.7)

    # ── Row 3: Exclusion criteria ──
    y = 12.5
    draw_box(ax, 6, y, 7, 0.7,
             "Exclusion criteria applied\nLateral views, paediatric (<15 y), unreadable files, "
             "resolution <200×200, duplicates",
             *C_EXCL, fontsize=8)

    draw_arrow(ax, 6, 13.4, 6, 12.9)

    # Exclusion side box
    draw_box(ax, 10.5, y, 2.5, 0.7,
             "Excluded: n = 0\nNo images met\nexclusion criteria",
             *C_EXCL, fontsize=7)
    draw_arrow(ax, 8, y, 9.25, y, color="#E63946")

    # ── Row 4: Label harmonisation ──
    y = 11.2
    draw_box(ax, 6, y, 7, 0.8,
             "Label harmonisation to binary TB status\n"
             "TB-positive (active + latent TB): n = 3,687\n"
             "TB-negative (healthy + sick-non-TB + normal): n = 13,520\n"
             "Unknown (TBX11K unlabelled): n = 3,302",
             *C_MAIN, fontsize=8)

    draw_arrow(ax, 6, 12.1, 6, 11.6)

    # ── Row 5: Split allocation ──
    y = 9.5
    ax.text(6, y + 0.5, "SPLIT ALLOCATION (stratified by dataset, TB status, sex, age)",
            ha="center", fontsize=9, fontweight="bold", color="#2A9D8F")

    # Calibration
    draw_box(ax, 1.8, y - 0.5, 3.0, 1.2,
             "CALIBRATION SET\nn = 800\n\n"
             "Shenzhen: n = 662\n"
             "  TB+: 336, TB−: 326\n"
             "Montgomery: n = 138\n"
             "  TB+: 58, TB−: 80",
             *C_SPLIT, fontsize=7)

    # Dev
    draw_box(ax, 5.5, y - 0.5, 2.5, 1.2,
             "DEVELOPMENT SET\nn = 3,510\n\n"
             "TBX11K (30%)\n"
             "TB+: 240\n"
             "TB−: 2,280\n"
             "Unknown: 990",
             *C_SPLIT, fontsize=7)

    # Test
    draw_box(ax, 8.5, y - 0.5, 2.5, 1.2,
             "PRIMARY TEST SET\nn = 8,191\n\n"
             "TBX11K (70%)\n"
             "TB+: 559\n"
             "TB−: 5,320\n"
             "Unknown: 2,312",
             *C_SPLIT, fontsize=7)

    draw_arrow(ax, 6, 10.8, 6, 10.1)
    # Fan out arrows
    draw_arrow(ax, 3.5, 10.1, 1.8, 10.1)
    draw_arrow(ax, 6, 10.1, 5.5, 10.1)
    draw_arrow(ax, 8.5, 10.1, 8.5, 10.1)

    # ── Row 6: External + Distractor ──
    y = 7.3
    draw_box(ax, 3.0, y, 3.5, 1.0,
             "EXTERNAL VALIDATION\nn = 3,008\n\n"
             "Mendeley Pakistan\n"
             "TB+: 2,494, TB−: 514",
             *C_SPLIT, fontsize=7)

    draw_box(ax, 8.5, y, 3.5, 1.0,
             "NON-TB DISTRACTOR\nn = 5,000\n\n"
             "CheXpert (random sample)\n"
             "All TB-negative",
             *C_SPLIT, fontsize=7)

    # ── Row 7: Embedding extraction ──
    y = 5.5
    draw_box(ax, 6, y, 8, 0.8,
             "EMBEDDING EXTRACTION\n"
             "4 frozen foundation models × 20,509 images = 767.5 MB\n"
             "RAD-DINO (768-d) | BiomedCLIP (512-d) | torchxrayvision (1024-d) | DINOv2-B (768-d)",
             "#fef3f0", "#E63946", fontsize=8)

    draw_arrow(ax, 6, 6.8, 6, 5.9)

    # Side note: excluded models
    draw_box(ax, 10.8, y + 1.0, 2.2, 0.7,
             "Models excluded\n(weight loading failures)\nEVA-X-B, CheXzero,\nGLoRIA",
             *C_EXCL, fontsize=6)

    # ── Row 8: Analysis ──
    y = 4.0
    draw_box(ax, 6, y, 8, 0.7,
             "PRIMARY ANALYSIS\n"
             "RAD-DINO + Linear Probe + Mondrian Conformal (α = 0.10)\n"
             "Calibrated on n = 800 | Tested on n = 8,191",
             "#fef9f0", "#E9C46A", fontsize=8, fontweight="bold")

    draw_arrow(ax, 6, 5.1, 6, 4.4)

    # ── Row 9: Results ──
    y = 2.5
    draw_box(ax, 3.0, y, 4.5, 1.0,
             "PRIMARY RESULTS\n\n"
             "AUROC: 0.916 [0.903, 0.928]\n"
             "TB coverage: 94.1% [92.0%, 96.0%]\n"
             "Singleton fraction: 85.3%",
             "#d4edda", "#28a745", fontsize=8)

    draw_box(ax, 8.5, y, 4.5, 1.0,
             "DEPLOYMENT GATES\n\n"
             "G1 Discrimination: PASS\n"
             "G2 Conformal validity: PASS\n"
             "G3 Clinical utility: PASS\n"
             "G4 Equity: PASS (weighted CP)",
             "#d4edda", "#28a745", fontsize=8)

    draw_arrow(ax, 4.5, 3.6, 3.0, 3.0)
    draw_arrow(ax, 7.5, 3.6, 8.5, 3.0)

    # ── Footer ──
    ax.text(6, 1.2,
            "Split manifest hash: 12a65e3a...da75e440 | Random seed: 42 | "
            "Pre-registered: OSF doi.org/10.17605/OSF.IO/KBAMC",
            ha="center", fontsize=7, style="italic", color="#888888")

    fig.tight_layout()
    save(fig, "sfig1_consort")


# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating diagrammatic figures...", flush=True)
    fig1_study_design()
    sfig1_consort()
    print("\nDone.")
