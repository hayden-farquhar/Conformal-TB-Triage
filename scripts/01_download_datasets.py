"""
Dataset download and verification for Project 68: Conformal TB Triage.

Usage:
    # On Google Colab (recommended for GPU embedding extraction):
    #   1. Mount Google Drive
    #   2. Run this script to download datasets to Colab disk or Drive
    #   3. Run embedding extraction (separate notebook)
    #   4. Embeddings saved to Drive persist across sessions

    # On local machine (for CPU-only work after embeddings extracted):
    #   python src/ingest/download_datasets.py --target local

    # To download a single dataset:
    #   python src/ingest/download_datasets.py --dataset tbx11k

    # To verify already-downloaded datasets:
    #   python src/ingest/download_datasets.py --verify-only
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlretrieve

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

from config import *

RAW_DATA_DIR = REPO_ROOT / "data" / "raw"
SNAPSHOT_LOG = REPO_ROOT / "osf" / "data_snapshot_log.md"

# Detect Colab environment
IN_COLAB = "COLAB_GPU" in os.environ or Path("/content").exists()

DATASETS = {
    "tbx11k": {
        "name": "TBX11K",
        "source": "kaggle",
        "kaggle_slug": "vbookshelf/tbx11k-simplified",
        "dest_dir": "tbx11k",
        "license": "CC-BY",
        "expected_images_approx": 11200,
        "notes": "Simplified version with PNG images at 512x512.",
    },
    "shenzhen": {
        "name": "Shenzhen (NLM)",
        "source": "url",
        "url": "https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets/Shenzhen-Hospital-CXR-Set/ChinaSet_AllFiles.zip",
        "dest_dir": "shenzhen_montgomery",
        "license": "Public domain (US government)",
        "expected_images_approx": 662,
        "notes": "Chinese CXR dataset from Shenzhen No.3 Hospital.",
    },
    "montgomery": {
        "name": "Montgomery (NLM)",
        "source": "url",
        "url": "https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets/Montgomery-County-CXR-Set/MontgomerySet.zip",
        "dest_dir": "shenzhen_montgomery",
        "license": "Public domain (US government)",
        "expected_images_approx": 138,
        "notes": "US CXR dataset from Montgomery County, Maryland.",
    },
    "mendeley_pakistan": {
        "name": "Mendeley Pakistan TB",
        "source": "manual",
        "url": "https://data.mendeley.com/datasets/8j2g3csprk/2",
        "dest_dir": "mendeley_pakistan",
        "license": "CC-BY 4.0",
        "expected_images_approx": 3008,
        "notes": "Download manually from Mendeley Data. Direct download requires API or browser.",
    },
    "jsrt": {
        "name": "JSRT",
        "source": "manual",
        "url": "http://db.jsrt.or.jp/eng.php",
        "dest_dir": "jsrt",
        "license": "Public (academic registration)",
        "expected_images_approx": 247,
        "notes": "Requires registration at db.jsrt.or.jp. Download link sent by email.",
    },
    "chexpert": {
        "name": "CheXpert v1.0-small",
        "source": "kaggle",
        "kaggle_slug": "ashery/chexpert",
        "dest_dir": "chexpert",
        "license": "Stanford Research License",
        "expected_images_approx": 224316,
        "notes": "Small version (~11 GB). We subsample 5,000 for the distractor split.",
    },
    "nih_cxr14": {
        "name": "NIH ChestX-ray14",
        "source": "kaggle",
        "kaggle_slug": "nih-chest-xrays/data",
        "dest_dir": "nih_cxr14",
        "license": "Public (attribution required)",
        "expected_images_approx": 112120,
        "notes": "~42 GB full. We subsample 5,000 for the distractor split. Consider partial download.",
    },
    "padchest": {
        "name": "PadChest",
        "source": "manual",
        "url": "https://bimcv.cipf.es/bimcv-projects/padchest/",
        "dest_dir": "padchest_subset",
        "license": "Free (academic registration)",
        "expected_images_approx": 160000,
        "notes": "~1 TB full. Requires registration at BIMCV. We subsample 5,000 for distractor split.",
    },
    "tb_portals": {
        "name": "NIAID TB Portals",
        "source": "manual",
        "url": "https://tbportals.niaid.nih.gov/download-data",
        "dest_dir": "tb_portals",
        "license": "DUA (non-commercial research)",
        "expected_images_approx": 9000,
        "notes": "Requires DUA via accessclinicaldata@NIAID.nih.gov. CXR imaging data requested separately.",
    },
}

# Datasets that can be auto-downloaded (Kaggle API or direct URL)
AUTO_DOWNLOADABLE = {"tbx11k", "shenzhen", "montgomery", "chexpert", "nih_cxr14"}

# Datasets requiring manual steps
MANUAL_DATASETS = {"mendeley_pakistan", "jsrt", "padchest", "tb_portals"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sha256_file(filepath: Path) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def count_images(directory: Path, extensions: tuple = (".png", ".jpg", ".jpeg", ".dcm", ".dicom")) -> int:
    """Count image files recursively in a directory."""
    count = 0
    if directory.exists():
        for ext in extensions:
            count += len(list(directory.rglob(f"*{ext}")))
    return count


def ensure_kaggle_api():
    """Check that kaggle CLI is available."""
    try:
        subprocess.run(["kaggle", "--version"], capture_output=True, check=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def download_kaggle_dataset(slug: str, dest: Path):
    """Download a Kaggle dataset using the kaggle CLI."""
    dest.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading from Kaggle: {slug}")
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", slug, "-p", str(dest), "--unzip"],
        check=True,
    )
    print(f"  Done. Files in {dest}")


def download_url(url: str, dest_dir: Path, filename: str = None):
    """Download a file from a URL."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    if filename is None:
        filename = url.split("/")[-1]
    dest_path = dest_dir / filename

    if dest_path.exists():
        print(f"  Already exists: {dest_path}")
        return dest_path

    print(f"  Downloading: {url}")
    print(f"  To: {dest_path}")
    urlretrieve(url, dest_path)
    print(f"  Done. Size: {dest_path.stat().st_size / 1e6:.1f} MB")
    return dest_path


def extract_zip(zip_path: Path, dest_dir: Path):
    """Extract a zip file."""
    print(f"  Extracting: {zip_path}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)
    print(f"  Extracted to: {dest_dir}")


# ---------------------------------------------------------------------------
# Download functions per dataset
# ---------------------------------------------------------------------------

def download_tbx11k(dest_root: Path):
    dest = dest_root / "tbx11k"
    if count_images(dest) > 1000:
        print(f"  TBX11K already downloaded ({count_images(dest)} images)")
        return
    download_kaggle_dataset("vbookshelf/tbx11k-simplified", dest)


def download_shenzhen(dest_root: Path):
    dest = dest_root / "shenzhen_montgomery"
    shenzhen_count = count_images(dest / "ChinaSet_AllFiles") if (dest / "ChinaSet_AllFiles").exists() else count_images(dest)
    if shenzhen_count > 600:
        print(f"  Shenzhen already downloaded (~{shenzhen_count} images)")
        return
    zip_path = download_url(
        "https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets/Shenzhen-Hospital-CXR-Set/ChinaSet_AllFiles.zip",
        dest,
    )
    if zip_path.suffix == ".zip":
        extract_zip(zip_path, dest)


def download_montgomery(dest_root: Path):
    dest = dest_root / "shenzhen_montgomery"
    mont_count = count_images(dest / "MontgomerySet") if (dest / "MontgomerySet").exists() else 0
    if mont_count > 100:
        print(f"  Montgomery already downloaded (~{mont_count} images)")
        return
    zip_path = download_url(
        "https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets/Montgomery-County-CXR-Set/MontgomerySet.zip",
        dest,
    )
    if zip_path.suffix == ".zip":
        extract_zip(zip_path, dest)


def download_chexpert(dest_root: Path):
    dest = dest_root / "chexpert"
    if count_images(dest) > 1000:
        print(f"  CheXpert already downloaded ({count_images(dest)} images)")
        return
    download_kaggle_dataset("ashery/chexpert", dest)


def download_nih_cxr14(dest_root: Path):
    dest = dest_root / "nih_cxr14"
    if count_images(dest) > 1000:
        print(f"  NIH CXR14 already downloaded ({count_images(dest)} images)")
        return
    print("  NOTE: NIH CXR14 is ~42 GB. Downloading full dataset.")
    print("  For subset-only, consider downloading via Kaggle Notebook with output.")
    download_kaggle_dataset("nih-chest-xrays/data", dest)


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify_dataset(name: str, dest_root: Path) -> dict:
    """Verify a downloaded dataset and return metadata."""
    config = DATASETS[name]
    dest = dest_root / config["dest_dir"]
    n_images = count_images(dest)
    expected = config["expected_images_approx"]

    status = "OK" if n_images > expected * 0.5 else "MISSING" if n_images == 0 else "PARTIAL"

    return {
        "name": config["name"],
        "directory": str(dest),
        "images_found": n_images,
        "images_expected": expected,
        "status": status,
        "license": config["license"],
        "verified_at": datetime.now(timezone.utc).isoformat(),
    }


def print_verification_table(results: list[dict]):
    """Print a summary verification table."""
    print("\n" + "=" * 80)
    print("DATASET VERIFICATION SUMMARY")
    print("=" * 80)
    print(f"{'Dataset':<25} {'Status':<10} {'Found':<10} {'Expected':<10} {'License'}")
    print("-" * 80)
    for r in results:
        status_marker = "OK" if r["status"] == "OK" else "!!" if r["status"] == "MISSING" else "~"
        print(f"{r['name']:<25} {status_marker:<10} {r['images_found']:<10} {r['images_expected']:<10} {r['license']}")
    print("=" * 80)

    missing = [r for r in results if r["status"] == "MISSING"]
    if missing:
        print(f"\n{len(missing)} dataset(s) not yet downloaded:")
        for r in missing:
            config_key = [k for k, v in DATASETS.items() if v["name"] == r["name"]][0]
            config = DATASETS[config_key]
            if config_key in MANUAL_DATASETS:
                print(f"  - {r['name']}: Manual download required. URL: {config.get('url', 'N/A')}")
                print(f"    Notes: {config['notes']}")
            else:
                print(f"  - {r['name']}: Run with --dataset {config_key}")


# ---------------------------------------------------------------------------
# Colab/Drive integration
# ---------------------------------------------------------------------------

COLAB_SETUP_INSTRUCTIONS = """
# ============================================================
# Google Colab Setup for Project 68: Conformal TB Triage
# ============================================================
#
# Run this in a Colab notebook cell before using the download script:
#
# Cell 1: Mount Google Drive
#   from google.colab import drive
#   drive.mount('/content/drive')
#
# Cell 2: Set up Kaggle credentials
#   # Upload your kaggle.json API key
#   from google.colab import files
#   files.upload()  # upload kaggle.json
#   !mkdir -p ~/.kaggle && mv kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
#
# Cell 3: Clone project repo (or copy from Drive)
#   # Option A: Clone from GitHub
#   !git clone https://github.com/YOUR_REPO/conformal-tb-triage.git /content/project
#
#   # Option B: Copy from Drive
#   !cp -r "/content/drive/MyDrive/your-project-folder" /content/project
#
# Cell 4: Download datasets to Colab disk (fast, but ephemeral)
#   !cd /content/project && python src/ingest/download_datasets.py --target colab
#
# Cell 5: After embedding extraction, save embeddings to Drive
#   !cp -r /content/project/data/interim/embeddings "/content/drive/MyDrive/embeddings-backup/"
#
# ============================================================
"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Download and verify datasets for P68")
    parser.add_argument(
        "--dataset",
        choices=list(DATASETS.keys()) + ["all", "auto"],
        default="all",
        help="Which dataset to download. 'auto' = only auto-downloadable. 'all' = attempt all.",
    )
    parser.add_argument(
        "--target",
        choices=["local", "colab"],
        default="colab" if IN_COLAB else "local",
        help="Download target. 'colab' puts large datasets on /content (ephemeral); 'local' uses data/raw/.",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify existing downloads, don't download anything new.",
    )
    parser.add_argument(
        "--colab-instructions",
        action="store_true",
        help="Print Colab setup instructions and exit.",
    )
    args = parser.parse_args()

    if args.colab_instructions:
        print(COLAB_SETUP_INSTRUCTIONS)
        return

    # Determine data root
    if args.target == "colab" and IN_COLAB:
        data_root = Path("/content/data/raw")
    else:
        data_root = RAW_DATA_DIR

    data_root.mkdir(parents=True, exist_ok=True)
    print(f"Data root: {data_root}")
    print(f"Target: {args.target}")
    print(f"Colab detected: {IN_COLAB}")
    print()

    # Verification only
    if args.verify_only:
        results = [verify_dataset(name, data_root) for name in DATASETS]
        print_verification_table(results)
        return

    # Determine which datasets to download
    if args.dataset == "all":
        to_download = list(DATASETS.keys())
    elif args.dataset == "auto":
        to_download = list(AUTO_DOWNLOADABLE)
    else:
        to_download = [args.dataset]

    # Check Kaggle API for Kaggle datasets
    kaggle_needed = any(DATASETS[d]["source"] == "kaggle" for d in to_download)
    if kaggle_needed and not ensure_kaggle_api():
        print("WARNING: Kaggle CLI not found. Install with: pip install kaggle")
        print("         Then place kaggle.json in ~/.kaggle/")
        print("         Skipping Kaggle datasets.\n")
        to_download = [d for d in to_download if DATASETS[d]["source"] != "kaggle"]

    # Download each dataset
    download_funcs = {
        "tbx11k": download_tbx11k,
        "shenzhen": download_shenzhen,
        "montgomery": download_montgomery,
        "chexpert": download_chexpert,
        "nih_cxr14": download_nih_cxr14,
    }

    for name in to_download:
        config = DATASETS[name]
        print(f"\n--- {config['name']} ---")

        if name in download_funcs:
            try:
                download_funcs[name](data_root)
            except Exception as e:
                print(f"  ERROR: {e}")
        elif name in MANUAL_DATASETS:
            print(f"  Manual download required.")
            print(f"  URL: {config.get('url', 'N/A')}")
            print(f"  Notes: {config['notes']}")
            print(f"  Place files in: {data_root / config['dest_dir']}")
        else:
            print(f"  No download function for {name}")

    # Post-download verification
    print("\n\nRunning post-download verification...")
    results = [verify_dataset(name, data_root) for name in DATASETS]
    print_verification_table(results)

    # Save verification log
    log_path = data_root / "download_verification.json"
    with open(log_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nVerification log saved to: {log_path}")

    # Print manual download instructions for remaining datasets
    manual_remaining = [
        name for name in MANUAL_DATASETS
        if verify_dataset(name, data_root)["status"] == "MISSING"
    ]
    if manual_remaining:
        print("\n" + "=" * 60)
        print("MANUAL DOWNLOADS STILL NEEDED:")
        print("=" * 60)
        for name in manual_remaining:
            config = DATASETS[name]
            print(f"\n  {config['name']}:")
            print(f"    URL: {config.get('url', 'N/A')}")
            print(f"    Destination: {data_root / config['dest_dir']}")
            print(f"    Notes: {config['notes']}")

        if "tb_portals" in manual_remaining:
            print("\n  --- NIAID TB Portals DUA Application ---")
            print("  1. Email accessclinicaldata@NIAID.nih.gov requesting imaging data access")
            print("  2. Reference: NIAID TB Portals Program, CXR images for TB triage research")
            print("  3. Describe: non-commercial academic research, conformal prediction study")
            print("  4. PI: Hayden Farquhar MBBS MPHTM (independent researcher)")
            print("  5. Expected processing time: 2-4 weeks")


if __name__ == "__main__":
    main()
