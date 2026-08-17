#!/usr/bin/env python3
"""Fetches the PlantDoc dataset into data/PlantDoc/<raw class name>/*.jpg —
the layout src/data/plantdoc.py expects — from its GitHub release
(https://github.com/pratikkayal/PlantDoc-Dataset).

PlantDoc ships as separate train/ and test/ folders with the same class
names; this script merges both into one flat pool per class (our own
train/test splitting happens downstream, same as PlantVillage), prefixing
filenames by split to avoid collisions.

Run:
    python scripts/download_plantdoc.py
"""

from __future__ import annotations

import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

DEST = Path(__file__).resolve().parent.parent / "data" / "PlantDoc"

_ARCHIVE_URLS = [
    "https://github.com/pratikkayal/PlantDoc-Dataset/archive/refs/heads/master.zip",
    "https://github.com/pratikkayal/PlantDoc-Dataset/archive/refs/heads/main.zip",
]


def _download_archive(dest_zip: Path) -> None:
    last_err: Exception | None = None
    for url in _ARCHIVE_URLS:
        try:
            print(f"Downloading {url} ...")
            urllib.request.urlretrieve(url, dest_zip)
            return
        except Exception as e:  # try the next branch name
            last_err = e
    raise RuntimeError(f"Could not download PlantDoc-Dataset from any known branch: {last_err}")


def main() -> int:
    if DEST.exists() and any(DEST.iterdir()):
        print(f"{DEST} already has data — nothing to do.")
        return 0

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        archive = tmp_path / "plantdoc.zip"
        _download_archive(archive)

        print("Extracting...")
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(tmp_path)

        extracted_roots = [p for p in tmp_path.iterdir() if p.is_dir()]
        if len(extracted_roots) != 1:
            print(f"Expected exactly one extracted top-level folder, found: {extracted_roots}", file=sys.stderr)
            return 1
        repo_root = extracted_roots[0]

        DEST.mkdir(parents=True, exist_ok=True)
        counts: dict[str, int] = {}
        for split in ("train", "test"):
            split_dir = repo_root / split
            if not split_dir.exists():
                continue
            for class_dir in sorted(split_dir.iterdir()):
                if not class_dir.is_dir():
                    continue
                out_dir = DEST / class_dir.name
                out_dir.mkdir(exist_ok=True)
                for img in class_dir.iterdir():
                    if not img.is_file():
                        continue
                    shutil.copy2(img, out_dir / f"{split}_{img.name}")
                    counts[class_dir.name] = counts.get(class_dir.name, 0) + 1

    n_classes = len(counts)
    n_images = sum(counts.values())
    if n_images == 0:
        print("No images were found in the downloaded archive — PlantDoc's repo layout may have changed.", file=sys.stderr)
        return 1
    print(f"Done. {n_images} images across {n_classes} classes written to {DEST}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
