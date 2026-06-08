# Conformal TB Triage

Code repository for: *Valid and recalibratable conformal prediction sets on frozen foundation-model embeddings for tuberculosis chest radiograph triage: a correction and evaluation study*

Hayden Farquhar MBBS MPHTM

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19718656.svg)](https://doi.org/10.5281/zenodo.19718656)

Pre-registration: [OSF doi.org/10.17605/OSF.IO/KBAMC](https://doi.org/10.17605/OSF.IO/KBAMC)

> **Method note.** A split-provenance audit of an earlier version of this
> pipeline found that the conformal nonconformity scores were computed on the
> same split used to train the probe, violating the independence condition
> that split-conformal coverage depends on. That in-sample design certifies a
> *degenerate* predictor (94.1% TB-class coverage but only 47.5% marginal
> coverage with 14.7% empty sets). The analysis here computes the conformal
> calibration on a held-out development split, and every result, figure, and
> table in this repository reflects that held-out design.

> **Scope of the shipped outputs.** Every analysis is computed under the
> held-out design: discrimination, conformal coverage, recalibration, the CPU
> sensitivity suite (label noise, seed/resplit stability, meta-coverage,
> calibration-set size), and the four image-based GPU sensitivity analyses
> (image-quality degradation, test-time augmentation, lung segmentation, and
> prediction-set stability), produced via
> `notebooks/03_gpu_sensitivity_analyses.ipynb`. The in-sample design is retained
> only as a documented contrast: the scripts and notebooks that reproduce it
> write an `_insample` suffix (e.g. `conformal_results_insample.csv`) and should
> not be cited as results.

## Overview

This repository contains the analysis code for a pre-registered evaluation of
class-conditional conformal prediction for TB chest radiograph (CXR) triage on
frozen foundation-model embeddings. Rather than a deployment claim, the study is
a method-and-lesson contribution with three explicit findings: (1) a
documented, reproducible conformal failure mode produced when the
probe-training split is reused for calibration, and the diagnostic that detects
it (decompose coverage by class *and* report empty- and singleton-set rates);
(2) the recalibration economics of restoring coverage under geographic shift;
and (3) a contamination control (DINOv2, natural-image pretraining) showing that
validity is a property of the calibration design, not of CXR pretraining leakage.

The pipeline extracts frozen embeddings from 4 foundation models (RAD-DINO,
BiomedCLIP, torchxrayvision, DINOv2-B), trains lightweight probes, and applies
conformal prediction methods (APS, RAPS, Mondrian, Conformal Risk Control,
Learn Then Test). A WHO-TPP–aligned triage rule (sensitivity ≥90%, specificity
≥70%) is retained as a worked example and its true operating envelope reported
honestly — including where it falls short of the TPP under prospectively-valid
threshold transfer.

An interactive [Streamlit demo](app/) is included.

## Data Sources

| Dataset | Role | n | URL | Access | License |
|---------|------|---|-----|--------|---------|
| TBX11K | Primary test | 11,701 | [Kaggle](https://www.kaggle.com/datasets/usmanshams/tbx-11) | Free (Kaggle account) | CC-BY |
| Shenzhen (NLM) | Calibration | 662 | [NLM](https://lhncbc.nlm.nih.gov/LHC-downloads/dataset.html) | Free | Public domain |
| Montgomery (NLM) | Calibration | 138 | [NLM](https://lhncbc.nlm.nih.gov/LHC-downloads/dataset.html) | Free | Public domain |
| Mendeley Pakistan | External validation | 3,008 | [Mendeley](https://data.mendeley.com/datasets/jctsfj2sfn/1) | Free (Mendeley account) | CC-BY 4.0 |
| CheXpert v1.0 | Non-TB distractor | 5,000 | [Kaggle](https://www.kaggle.com/datasets/ashery/chexpert) | Free (Kaggle account) | Stanford Research License |

Raw images are not redistributed. See [`data/raw/README.md`](data/raw/README.md) for download instructions.

## Repository Structure

```
├── README.md
├── LICENSE                          MIT
├── requirements.txt                 Python dependencies
├── data_dictionary.md               Variable definitions for all data files
├── .zenodo.json                     Zenodo metadata for DOI minting
│
├── data/
│   ├── raw/README.md                Instructions to obtain raw CXR images
│   └── processed/
│       └── splits.parquet           Split manifest (20,509 images, SHA-256 verified)
│
├── notebooks/                       Colab/Kaggle notebooks (GPU required)
│   ├── 01_download_datasets.ipynb   Dataset download and split creation
│   ├── 02_extract_embeddings.ipynb  Frozen embedding extraction (7 models)
│   ├── 03_gpu_sensitivity_analyses.ipynb   Image degradation, TTA, stability
│   └── 03b_lung_segmentation_sensitivity.ipynb   Lung segmentation (classical CV)
│
├── scripts/                         Analysis scripts (CPU, run in order)
│   ├── config.py                    Shared path configuration
│   ├── 01_download_datasets.py      Dataset download (CLI version)
│   ├── 02_train_probes.py           Linear, kNN, XGBoost, MLP probes
│   ├── 03_conformal_calibration.py  APS, RAPS, Mondrian, CRC calibration
│   ├── 04_improved_pipeline.py      Weighted CP, probability calibration, fusion
│   ├── 05_core_analyses.py          Bootstrap CIs, WHO TPP, referral cascade
│   ├── 06_sensitivity_analyses.py   Seed stability, meta-coverage, label noise
│   ├── 07_exploratory_analyses.py   Multi-class, abstention, adaptive, SHAP
│   ├── 08_remaining_secondary.py    LTT, LODO, commercial comparison
│   ├── 09_tier3_analyses.py         Baselines, DCA, computational cost, t-SNE
│   ├── 10_figures_and_final.py      Main figures + drift monitoring, Venn-ABERS
│   ├── 11_supplementary_figures.py  Supplementary figures (sFig 5, 8-15)
│   ├── 12_diagrammatic_figures.py   Fig 1 (study design), sFig 1 (CONSORT)
│   ├── conformal_pipeline.py        Held-out conformal calibration (authoritative headline)
│   ├── conformal_sensitivity.py     Held-out CPU sensitivity (label noise, seed/resplit, meta-coverage, cal-set size)
│   ├── gpu_sensitivity_figures.py   Held-out GPU-sensitivity figure (image-quality degradation)
│   ├── venn_abers_intervals.py      Held-out Venn-ABERS probability intervals (held-out dev)
│   ├── cross_conformal_cvplus.py    Held-out CV+ cross-conformal (held-out dev)
│   ├── fig3_provenance.py           Figure 3: in-sample vs held-out calibration provenance
│   └── recompute_headline_numbers.py Recompute headline coverage numbers from saved predictions
│
├── app/                             Interactive Streamlit demo
│   ├── app.py                       Demo application
│   ├── probe.pkl                    Trained linear probe (RAD-DINO)
│   ├── isotonic.pkl                 Isotonic calibrator
│   └── conformal_thresholds.json    Mondrian conformal thresholds
│
└── outputs/                         Pre-computed results
    ├── tables/                      68 CSV result files
    └── figures/                     26 publication figures (PNG/PDF)
```

## Requirements

Python ≥ 3.11. Install dependencies:

```bash
pip install -r requirements.txt
```

GPU is required only for the Colab notebooks (embedding extraction and
GPU-intensive sensitivity analyses). All numbered scripts (02-12) run on CPU
using pre-extracted embeddings.

## Reproduction

### Quick start (from pre-extracted embeddings)

If you have the embedding parquet files (available from the Zenodo deposit):

```bash
# Set the path to your embedding data
export DATA_ROOT=/path/to/embedding/data

# Run scripts in order (each takes 1-30 minutes on CPU)
cd scripts
python 02_train_probes.py
python 03_conformal_calibration.py
python 04_improved_pipeline.py
python 05_core_analyses.py
python 06_sensitivity_analyses.py
python 07_exploratory_analyses.py
python 08_remaining_secondary.py
python 09_tier3_analyses.py
python 10_figures_and_final.py
python 11_supplementary_figures.py
python 12_diagrammatic_figures.py
```

Results are written to `outputs/tables/` and `outputs/figures/`.

### Full reproduction (from raw images)

To reproduce from scratch including embedding extraction:

1. **Download datasets** using notebook `01_download_datasets.ipynb` on Google Colab (or `scripts/01_download_datasets.py` locally)
2. **Extract embeddings** using notebook `02_extract_embeddings.ipynb` on a GPU runtime (Kaggle T4 or Colab). This produces the parquet files needed by subsequent scripts.
3. **Run GPU sensitivity analyses** using notebooks `03_gpu_sensitivity_analyses.ipynb` and `03b_lung_segmentation_sensitivity.ipynb` on a GPU runtime.
4. **Run CPU scripts 02-12** as shown above.

Estimated total runtime: ~4 hours GPU (embedding extraction) + ~2 hours CPU (all scripts).

### Interactive demo

```bash
streamlit run app/app.py
```

Upload a PA/AP chest X-ray to see the conformal prediction set, triage tier,
and calibrated probability. Requires a GPU for real-time RAD-DINO embedding
extraction, or runs on CPU (~30 seconds per image).

## Script Descriptions

| Script | Description | Key Outputs |
|--------|-------------|-------------|
| `01_download_datasets.py` | Downloads and verifies all CXR datasets | Raw images, `splits.parquet` |
| `02_train_probes.py` | Trains 4 probe types on 4 embedding models | Probe models, predictions |
| `03_conformal_calibration.py` | Calibrates APS, RAPS, Mondrian, CRC | `conformal_results.csv`, `crc_results.csv`, `triage_results.csv` |
| `04_improved_pipeline.py` | Weighted CP, isotonic calibration, fusion | `improved_conformal_results.csv`, `fused_embeddings.csv` |
| `05_core_analyses.py` | Bootstrap CIs, WHO TPP, referral cascade, shortcuts | `bootstrap_cis.csv`, `who_tpp_alignment.csv`, `referral_cascade.csv` |
| `06_sensitivity_analyses.py` | Seed stability, meta-coverage, label noise, latent TB | `seed_stability.csv`, `meta_coverage.csv`, `label_noise.csv` |
| `07_exploratory_analyses.py` | Multi-class, abstention, adaptive CP, SHAP | `multiclass_analysis.csv`, `shap_feature_importance.csv` |
| `08_remaining_secondary.py` | LTT, LODO recalibration, commercial comparison | `ltt_results.csv`, `recalibration_lodo.csv` |
| `09_tier3_analyses.py` | Reference baselines, DCA, computational cost, t-SNE | `reference_baselines.csv`, `decision_curves.csv` |
| `10_figures_and_final.py` | Main figures + drift monitoring (in-sample Venn-ABERS / CV+ retained as `_insample`) | Fig 2, 4, 5, 7, sFig 2-3 |
| `11_supplementary_figures.py` | Remaining supplementary figures | sFig 8-12, 14-15 |
| `12_diagrammatic_figures.py` | Study design schematic and CONSORT flow | Fig 1, sFig 1 |
| `conformal_pipeline.py` | Held-out conformal calibration (split-valid; authoritative headline) | `conformal_results.csv` |
| `conformal_sensitivity.py` | Held-out CPU sensitivity (label noise, seed/resplit, meta-coverage, cal-set size, subgroups) | `label_noise.csv`, `seed_stability.csv`, `meta_coverage.csv`, `calset_sensitivity.csv` |
| `gpu_sensitivity_figures.py` | Held-out GPU-sensitivity figure: image-quality degradation, marginal + TB-class coverage | `sfig_image_degradation.{png,pdf}` |
| `venn_abers_intervals.py` | Held-out Venn-ABERS probability intervals (isotonic calibrators on held-out dev, not the probe-training split) | `venn_abers.csv` (+`_summary`) |
| `cross_conformal_cvplus.py` | Held-out CV+ over the in-distribution held-out dev pool (shows marginal CV+ under-covers the TB class + emits empty sets) | `cross_conformal.csv` |
| `fig3_provenance.py` | Figure 3: in-sample vs held-out calibration provenance (recomputed live from `probe_predictions.parquet`) | `fig3_efficiency_curves.{png,pdf}` |

> **Note.** `conformal_pipeline.py`, `conformal_sensitivity.py`,
> `gpu_sensitivity_figures.py`, `venn_abers_intervals.py`, and
> `cross_conformal_cvplus.py` own the held-out-calibrated headline and
> sensitivity outputs under their canonical names. Where a numbered script (`03`–`06`,
> `10`, `11`) recomputes one of these analyses under the original in-sample design,
> its output is written with an `_insample` suffix and retained only to document
> the superseded design and audit trail; do not cite the `_insample` outputs.

See [`data_dictionary.md`](data_dictionary.md) for detailed descriptions of the output CSV files.

> **Figure file naming.** The figure files in `outputs/figures/` are reproducibility
> outputs whose filenames follow an internal scheme and do not map one-to-one onto
> the manuscript supplement's figure numbers (S1–S11). The authoritative, numbered
> figures are those in the paper and its supplement; treat the files here as the
> regenerated sources behind them, identified by descriptive name rather than by
> supplement number.

## Key Results (held-out calibration)

Primary model: RAD-DINO, linear probe, Mondrian, α=0.10, TBX11K test (n=5,879),
conformal calibration on the held-out development split (n_cal=1,260).

| Metric | Value | 95% CI |
|--------|-------|--------|
| AUROC (RAD-DINO, TBX11K test) | 0.914 | [0.900, 0.927] |
| Marginal coverage (Mondrian, α=0.10) | 91.4% | [89.4%, 92.9%] |
| TB-class coverage | 92.5% | [85.3%, 98.6%] |
| Empty-set fraction | 0.0% | — |
| Singleton fraction | 73.3% | — |
| Class-coverage disparity | 1.3 pts | — |

Isotonic recalibration raises marginal coverage to 94.0%. All five pre-registered
gates pass under the held-out design.

**In-sample defect, for contrast** (probe-training split reused for calibration):
94.1% TB-class coverage but only 47.5% marginal coverage with 14.7% empty sets —
a degenerate predictor that passes a class-conditional check while failing globally.

**WHO Type 2 TPP.** The discriminative frontier reaches the TPP corner (90.5%
sensitivity, 71.5% specificity) only under in-sample threshold selection. Under
prospectively-valid threshold transfer the operating point meets the 90%
sensitivity floor (92.5%) but not the 70% specificity target (64.1%); the TPP
is therefore not met under a valid transfer.

Split manifest SHA-256: `12a65e3a500d9f473ecbfd3f1787c43eb2537c2db742a7bb52538a58da75e440`

## Citation

If you use this code, please cite:

```
Farquhar H. Valid and recalibratable conformal prediction sets on frozen
foundation-model embeddings for tuberculosis chest radiograph triage: a
correction and evaluation study. 2026. Code: https://doi.org/10.5281/zenodo.19718656
```

## License

Code: [MIT License](LICENSE)
