#!/usr/bin/env python3
"""Fetches the PlantWild v1 dataset into data/PlantWild/<class name>/*.jpg —
the layout src/data/plantwild.py expects — from its Hugging Face Hub release
(https://huggingface.co/datasets/uqtwei2/PlantWild).

PlantWild ships as a single zip (plantwild.zip) with images already
organised as plantwild/images/<class name>/*.jpg — one folder per class,
lowercase names with spaces (e.g. "tomato early blight"). This script just
downloads and unpacks that folder into data/PlantWild/, the same flat
one-folder-per-class layout as PlantVillage and PlantDoc.

Note: the release also ships plantwild_v2.zip (an expanded, disease-only
revision with no "healthy" classes) — not used here, since
PLANTWILD_TO_PLANTVILLAGE in src/data/plantwild.py was built against v1's
89 class names.

This is a ~2.7GB download (18,542 images across 89 classes).

Run:
    python scripts/download_plantwild.py
"""

from __future__ import annotations

import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

DEST = Path(__file__).resolve().parent.parent / "data" / "PlantWild"

_ARCHIVE_URL = "https://huggingface.co/datasets/uqtwei2/PlantWild/resolve/main/plantwild.zip"


def main() -> int:
    if DEST.exists() and any(DEST.iterdir()):
        print(f"{DEST} already has data — nothing to do.")
        return 0

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        archive = tmp_path / "plantwild.zip"

        print(f"Downloading {_ARCHIVE_URL} (~2.7GB, this will take a while)...")
        urllib.request.urlretrieve(_ARCHIVE_URL, archive)

        print("Extracting...")
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(tmp_path)

        images_root = tmp_path / "plantwild" / "images"
        if not images_root.exists():
            print(
                f"Expected {images_root} in the extracted archive but it's missing — "
                f"PlantWild's release layout may have changed.",
                file=sys.stderr,
            )
            return 1

        DEST.mkdir(parents=True, exist_ok=True)
        counts: dict[str, int] = {}
        for class_dir in sorted(images_root.iterdir()):
            if not class_dir.is_dir():
                continue
            out_dir = DEST / class_dir.name
            out_dir.mkdir(exist_ok=True)
            for img in class_dir.iterdir():
                if not img.is_file():
                    continue
                shutil.copy2(img, out_dir / img.name)
                counts[class_dir.name] = counts.get(class_dir.name, 0) + 1

    n_classes = len(counts)
    n_images = sum(counts.values())
    if n_images == 0:
        print("No images were found in the downloaded archive — PlantWild's repo layout may have changed.", file=sys.stderr)
        return 1
    print(f"Done. {n_images} images across {n_classes} classes written to {DEST}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
