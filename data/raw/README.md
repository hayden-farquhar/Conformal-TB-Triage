# Raw Data Acquisition

Raw chest X-ray images are not redistributed in this repository due to size
and licensing constraints. Follow the instructions below to obtain each dataset.

## Datasets

### TBX11K (primary test set)
- **Source:** Kaggle
- **URL:** https://www.kaggle.com/datasets/usmanshams/tbx-11
- **Access:** Free (Kaggle account required)
- **License:** CC-BY
- **Approx. size:** 11,701 images

### Shenzhen (calibration set)
- **Source:** U.S. National Library of Medicine
- **URL:** https://lhncbc.nlm.nih.gov/LHC-downloads/dataset.html
- **Access:** Free (no registration)
- **License:** Public domain (U.S. government)
- **Approx. size:** 662 images

### Montgomery County (calibration set)
- **Source:** U.S. National Library of Medicine
- **URL:** https://lhncbc.nlm.nih.gov/LHC-downloads/dataset.html
- **Access:** Free (no registration)
- **License:** Public domain (U.S. government)
- **Approx. size:** 138 images

### Mendeley Pakistan TB (external validation)
- **Source:** Mendeley Data
- **URL:** https://data.mendeley.com/datasets/jctsfj2sfn/1
- **Access:** Free (Mendeley account required)
- **License:** CC-BY 4.0
- **Approx. size:** 3,008 images

### CheXpert v1.0 (non-TB distractor)
- **Source:** Kaggle
- **URL:** https://www.kaggle.com/datasets/ashery/chexpert
- **Access:** Free (Kaggle account required)
- **License:** Stanford Research License
- **Approx. size:** 5,000 images (random sample from full dataset)

## Embedding Data

Pre-extracted embeddings (767.5 MB total) are available from the accompanying
Zenodo deposit. These parquet files contain the frozen foundation-model
embeddings for all 20,509 images across 4 models (RAD-DINO, BiomedCLIP,
torchxrayvision, DINOv2-B), and are required to run the analysis scripts
(steps 02-12) without re-extracting from raw images.

Place embedding parquet files in the directory pointed to by the `DATA_ROOT`
environment variable (see `scripts/config.py`).
