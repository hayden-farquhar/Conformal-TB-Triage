# Conformal TB Triage

Code repository for: **Conformal prediction sets on frozen foundation-model embeddings for WHO-TPP-aligned tuberculosis chest X-ray triage: a multi-cohort derivation and external validation study**

Hayden Farquhar MBBS MPHTM

Preprint: to be posted

Pre-registration: [OSF doi.org/10.17605/OSF.IO/KBAMC](https://doi.org/10.17605/OSF.IO/KBAMC)

## Overview

This repository contains all analysis code to reproduce the findings of the accompanying manuscript. The study evaluates whether frozen medical-imaging foundation-model embeddings combined with class-conditional conformal prediction can produce a TB chest X-ray triage rule that meets the WHO Target Product Profile (sensitivity ≥90%, specificity ≥70%) with formal statistical coverage guarantees.

The pipeline extracts frozen embeddings from 4 foundation models (RAD-DINO, BiomedCLIP, torchxrayvision, DINOv2-B), trains lightweight probes, and applies conformal prediction methods (APS, RAPS, Mondrian, Conformal Risk Control, Learn Then Test) with comprehensive robustness evaluation across 46 pre-registered analyses.

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
│   └── 12_diagrammatic_figures.py   Fig 1 (study design), sFig 1 (CONSORT)
│
├── app/                             Interactive Streamlit demo
│   ├── app.py                       Demo application
│   ├── probe.pkl                    Trained linear probe (RAD-DINO)
│   ├── isotonic.pkl                 Isotonic calibrator
│   └── conformal_thresholds.json    Mondrian conformal thresholds
│
└── outputs/                         Pre-computed results
    ├── tables/                      53 CSV result files
    └── figures/                     21 publication figures (PNG)
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
| `10_figures_and_final.py` | Main figures + drift monitoring, Venn-ABERS, cross-conformal | Fig 2-7, sFig 2-4, 6-7 |
| `11_supplementary_figures.py` | Remaining supplementary figures | sFig 5, 8-15 |
| `12_diagrammatic_figures.py` | Study design schematic and CONSORT flow | Fig 1, sFig 1 |

See [`data_dictionary.md`](data_dictionary.md) for detailed descriptions of all 53 output CSV files.

## Key Results

| Metric | Value | 95% CI |
|--------|-------|--------|
| AUROC (RAD-DINO, TBX11K test) | 0.916 | [0.903, 0.928] |
| TB-class coverage (Mondrian, α=0.10) | 94.1% | [92.0%, 96.0%] |
| Singleton fraction | 85.3% | — |
| WHO TPP sensitivity | 90.2% | — |
| WHO TPP specificity | 72.1% | — |

All 5 pre-registered deployment gates passed (G1-G4 PASS; G4 with weighted conformal prediction).

Split manifest SHA-256: `12a65e3a500d9f473ecbfd3f1787c43eb2537c2db742a7bb52538a58da75e440`

## Citation

If you use this code, please cite:

```
Farquhar H. Conformal prediction sets on frozen foundation-model embeddings
for WHO-TPP-aligned tuberculosis chest X-ray triage: a multi-cohort derivation
and external validation study. 2026.
```

## License

Code: [MIT License](LICENSE)
