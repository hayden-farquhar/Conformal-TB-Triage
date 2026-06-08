"""
Generate Fig 1 (study design schematic) and sFig 1 (CONSORT flow).

Fig 1 is authored as a Mermaid diagram (outputs/figures/fig1_study_design.mmd)
and rendered with the mermaid-cli `mmdc`; the topology of the diagram is the
argument (two independent data lineages converging only at threshold
computation), so it is maintained as source rather than redrawn in matplotlib.
sFig 1 (CONSORT) is still drawn with matplotlib.

Run: python3 12_diagrammatic_figures.py
"""

import shutil
import subprocess
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
    """Render Fig 1 from the Mermaid source via mmdc (PNG + PDF).

    The diagram lives in outputs/figures/fig1_study_design.mmd. Its topology
    encodes the corrected pipeline (probe-training and held-out conformal-
    calibration lineages as independent siblings that converge only at
    threshold computation), so the .mmd is the single source of truth.
    """
    print("  Fig 1: Study design schematic (Mermaid)...", flush=True)
    mmd = FIGURES_DIR / "fig1_study_design.mmd"
    mmdc = shutil.which("mmdc")
    if mmdc is None:
        print("    mmdc (mermaid-cli) not found on PATH; skipping render.")
        print(f"    Render manually: mmdc -i {mmd} -o {FIGURES_DIR / 'fig1_study_design.png'} -s 3 -b white")
        print(f"                     mmdc -i {mmd} -o {FIGURES_DIR / 'fig1_study_design.pdf'} -b white --pdfFit")
        return
    subprocess.run([mmdc, "-i", str(mmd),
                    "-o", str(FIGURES_DIR / "fig1_study_design.png"),
                    "-s", "3", "-b", "white"], check=True)
    subprocess.run([mmdc, "-i", str(mmd),
                    "-o", str(FIGURES_DIR / "fig1_study_design.pdf"),
                    "-b", "white", "--pdfFit"], check=True)
    print("  Saved fig1_study_design.png + .pdf")


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
             "matrix dimensions <200×200, duplicates",
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

    # Probe-training set (NLM) — fits the probe, NOT the conformal calibration
    draw_box(ax, 1.8, y - 0.65, 3.0, 1.6,
             "PROBE-TRAINING SET\nn = 800\n\n"
             "Shenzhen: n = 662\n"
             "  TB+: 336, TB−: 326\n"
             "Montgomery: n = 138\n"
             "  TB+: 58, TB−: 80\n"
             "→ trains linear probe",
             *C_SPLIT, fontsize=6.5)

    # Dev — held-out conformal calibration (independent of the probe)
    draw_box(ax, 5.5, y - 0.65, 2.5, 1.6,
             "DEVELOPMENT SET\nn = 3,510\n\n"
             "TBX11K (30%)\n"
             "TB+: 240, TB−: 2,280\n"
             "Unknown: 990\n"
             "binary: n = 2,520\n"
             "→ conformal calibration",
             *C_SPLIT, fontsize=6.5)

    # Test
    draw_box(ax, 8.5, y - 0.65, 2.5, 1.6,
             "PRIMARY TEST SET\nn = 8,191\n\n"
             "TBX11K (70%)\n"
             "TB+: 559, TB−: 5,320\n"
             "Unknown: 2,312\n"
             "binary: n = 5,879\n"
             "→ evaluation",
             *C_SPLIT, fontsize=6.5)

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
             "Conformal calibrated on held-out Dev (n = 1,260) | Tested on n = 5,879",
             "#fef9f0", "#E9C46A", fontsize=8, fontweight="bold")

    draw_arrow(ax, 6, 5.1, 6, 4.4)

    # ── Row 9: Results ──
    y = 2.5
    draw_box(ax, 3.0, y, 4.5, 1.3,
             "PRIMARY RESULTS (held-out calibration)\n\n"
             "AUROC: 0.914 [0.900, 0.927]\n"
             "Marginal coverage: 91.4% [89.4, 92.9]\n"
             "TB coverage: 92.5% | Empty sets: 0%\n"
             "Singleton: 73.3% | Disparity: 1.3 pp",
             "#d4edda", "#28a745", fontsize=7.5)

    draw_box(ax, 8.5, y, 4.5, 1.3,
             "DEPLOYMENT GATES — all PASS\n\n"
             "G1 AUROC 0.914 ≥ 0.75\n"
             "G2 TB coverage 92.5% ≥ 85%\n"
             "G3 Singleton 73.3% ≥ 40%\n"
             "G4 Disparity 1.3 ≤ 15 pp (native CP)\n"
             "G5 Recal. set ≤ 500 CXRs",
             "#d4edda", "#28a745", fontsize=7)

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
