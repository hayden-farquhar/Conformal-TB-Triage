"""
Shared path configuration for all analysis scripts.

Data layout expected:
    DATA_ROOT/
    ├── embeddings/          # Parquet files: rad_dino.parquet, biomedclip.parquet, etc.
    └── splits.parquet       # Split manifest with patient IDs, labels, file paths

    REPO_ROOT/
    ├── outputs/
    │   ├── tables/          # CSV result files
    │   └── figures/         # Publication figures
    └── models/              # Trained probe models (auto-created)

Set the DATA_ROOT environment variable to point to your local copy of the
embedding data. If unset, defaults to ./data (relative to repository root).

    export DATA_ROOT=/path/to/your/embedding/data
"""

import os
from pathlib import Path

# Repository root (one level up from scripts/)
REPO_ROOT = Path(__file__).resolve().parent.parent

# Data root — embeddings and splits
DATA_ROOT = Path(os.environ.get("DATA_ROOT", str(REPO_ROOT / "data")))
EMB_DIR = DATA_ROOT / "embeddings"
SPLITS_PATH = DATA_ROOT / "splits.parquet"

# Output directories
RESULTS_DIR = REPO_ROOT / "outputs"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"
MODELS_DIR = REPO_ROOT / "models"

# Create directories
for d in [TABLES_DIR, FIGURES_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Constants
SEED = 42
WORKING_MODELS = ["rad_dino", "biomedclip", "torchxrayvision", "dinov2"]
MODEL_LABELS = {
    "rad_dino": "RAD-DINO",
    "biomedclip": "BiomedCLIP",
    "torchxrayvision": "torchxrayvision",
    "dinov2": "DINOv2-B",
}
MODEL_COLORS = {
    "rad_dino": "#E63946",
    "biomedclip": "#457B9D",
    "torchxrayvision": "#2A9D8F",
    "dinov2": "#E9C46A",
}
