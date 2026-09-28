#!/usr/bin/env python3
"""Fetches the PlantDoc OBJECT-DETECTION dataset (VOC-annotated leaf boxes,
distinct from scripts/download_plantdoc.py's classification-only folders)
into data/PlantDoc-OD/TRAIN and data/PlantDoc-OD/TEST — the layout
src/detection/train_detector.py expects — from its GitHub release
(https://github.com/pratikkayal/PlantDoc-Object-Detection-Dataset).

Run:
    python scripts/download_plantdoc_od.py
"""

from __future__ import annotations

import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

DEST = Path(__file__).resolve().parent.parent / "data" / "PlantDoc-OD"

_ARCHIVE_URLS = [
    "https://github.com/pratikkayal/PlantDoc-Object-Detection-Dataset/archive/refs/heads/master.zip",
    "https://github.com/pratikkayal/PlantDoc-Object-Detection-Dataset/archive/refs/heads/main.zip",
]


def _download_archive(dest_zip: Path) -> None:
    last_err: Exception | None = None
    for url in _ARCHIVE_URLS:
        try:
            print(f"Downloading {url} ...")
            urllib.request.urlretrieve(url, dest_zip)
            return
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Could not download PlantDoc-Object-Detection-Dataset from any known branch: {last_err}")


def main() -> int:
    if DEST.exists() and any(DEST.iterdir()):
        print(f"{DEST} already has data — nothing to do.")
        return 0

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        archive = tmp_path / "plantdoc_od.zip"
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
        for split in ("TRAIN", "TEST"):
            split_dir = repo_root / split
            if not split_dir.exists():
                continue
            out_dir = DEST / split
            out_dir.mkdir(exist_ok=True)
            n = 0
            for f in split_dir.iterdir():
                if f.is_file():
                    shutil.copy2(f, out_dir / f.name)
                    n += 1
            counts[split] = n

    n_images = sum(1 for _ in DEST.rglob("*.jpg"))
    if n_images == 0:
        print("No images were found in the downloaded archive — the repo layout may have changed.", file=sys.stderr)
        return 1
    print(f"Done. {counts} files written to {DEST} ({n_images} .jpg images).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
