"""Download the songs dataset CSV from Kaggle into ./data (run once).

Usage:
    python download_assets.py

The Kaggle dataset slug can be set below or via the KAGGLE_DATASET
environment variable, e.g. KAGGLE_DATASET=username/dataset-name
"""
import os
import shutil
import sys
from pathlib import Path

KAGGLE_DATASET = os.getenv("KAGGLE_DATASET", "rehammayea/60k-songs-with-description")
CSV_NAME = "songs_with_descriptions_V3.csv"
DEST_DIR = Path("data")


def main() -> None:
    target = DEST_DIR / CSV_NAME
    if target.exists():
        print(f"{target} already exists - nothing to download.")
        return

    if KAGGLE_DATASET.startswith("USERNAME/"):
        sys.exit(
            "Set the Kaggle dataset slug first: edit KAGGLE_DATASET in "
            "download_assets.py or run with KAGGLE_DATASET=username/dataset-name"
        )

    import kagglehub  # imported here so the message above shows even if not installed

    print(f"Downloading {KAGGLE_DATASET} from Kaggle ...")
    src_dir = Path(kagglehub.dataset_download(KAGGLE_DATASET))

    matches = list(src_dir.rglob(CSV_NAME))
    if not matches:
        sys.exit(f"{CSV_NAME} was not found inside the Kaggle dataset ({src_dir}).")

    DEST_DIR.mkdir(exist_ok=True)
    shutil.copy2(matches[0], target)
    print(f"Done: {target}")


if __name__ == "__main__":
    main()
