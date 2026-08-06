#!/usr/bin/env python3
"""Fetches the PlantVillage dataset into data/PlantVillage/<Crop>___<Disease>/*.jpg
— the layout src/data/plantvillage.py expects — using TensorFlow Datasets'
built-in 'plant_village' catalog entry (https://www.tensorflow.org/datasets/catalog/plant_village).

Why TFDS instead of Kaggle: it needs no account or API token, just a
`pip install tensorflow tensorflow-datasets`. It's still a real ~800MB
transfer the first time you run this — TFDS just automates the download,
caching, and checksum verification instead of you hunting down a zip file.

TFDS caches its own copy in TFRecord format under ~/tensorflow_datasets/
(or $TFDS_DATA_DIR) — a packed binary format, not individual browsable
images. This script does one extra pass over that cache and writes each
image back out as a real, human-viewable .jpg, so the end result is
identical to a manual/Kaggle download: plain image files, one folder per
class, openable in any file browser.

Run:
    python scripts/download_plantvillage.py

This can take several minutes (downloading ~800MB, then re-encoding
~54,000 images to disk) — it is meant to be run once, ahead of training.
"""

from __future__ import annotations

import sys
from pathlib import Path

DEST = Path(__file__).resolve().parent.parent / "data" / "PlantVillage"

# TFDS's plant_village builder fetches from this Mendeley Data URL, which
# 302-redirects to a plain public S3 object. Mendeley's Cloudflare protection
# blocks that initial request from many cloud/datacenter IPs (Colab included)
# with a 403 — but the S3 object itself has no such restriction. Verified
# manually: this S3 URL returns 200 OK with the identical file Mendeley
# redirects to.
_MENDELEY_S3_FALLBACK_URL = (
    "https://prod-dcd-datasets-public-files-eu-west-1.s3.eu-west-1.amazonaws.com/"
    "d29ed9b2-8a5d-4663-8a82-c9174f2c7066"
)


def _load_plant_village(tfds):
    """Tries the canonical Mendeley URL first (works fine off cloud IPs);
    falls back to the direct S3 object only if that's actually blocked.
    """
    try:
        return tfds.load("plant_village", split="train", with_info=True, shuffle_files=False)
    except Exception as e:
        if "403" not in str(e):
            raise
        print(
            "Mendeley's download endpoint returned 403 (its Cloudflare protection blocking "
            "this IP — common on Colab/cloud runners). Retrying via the direct S3 object it "
            "redirects to instead...",
            file=sys.stderr,
        )
        from tensorflow_datasets.datasets.plant_village import plant_village_dataset_builder as pv

        pv._URL = _MENDELEY_S3_FALLBACK_URL
        return tfds.load("plant_village", split="train", with_info=True, shuffle_files=False)


def main() -> int:
    if DEST.exists() and any(DEST.iterdir()):
        print(f"{DEST} already has data — nothing to do.")
        return 0

    try:
        import tensorflow_datasets as tfds
    except ImportError:
        print(
            "tensorflow_datasets is not installed. Run:\n"
            "    pip install tensorflow tensorflow-datasets\n"
            "and re-run this script.",
            file=sys.stderr,
        )
        return 1

    from PIL import Image
    from tqdm import tqdm

    print("Loading 'plant_village' via TensorFlow Datasets (downloads on first run)...")
    dataset, info = _load_plant_village(tfds)
    class_names = info.features["label"].names
    total = info.splits["train"].num_examples

    DEST.mkdir(parents=True, exist_ok=True)
    for name in class_names:
        (DEST / name).mkdir(exist_ok=True)

    counts = {name: 0 for name in class_names}
    for example in tqdm(tfds.as_numpy(dataset), total=total, desc="Writing images"):
        class_name = class_names[int(example["label"])]
        idx = counts[class_name]
        counts[class_name] += 1
        Image.fromarray(example["image"]).save(DEST / class_name / f"img_{idx}.jpg")

    n_classes = sum(1 for c in counts.values() if c > 0)
    n_images = sum(counts.values())
    print(f"Done. {n_images} images across {n_classes} classes written to {DEST}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
