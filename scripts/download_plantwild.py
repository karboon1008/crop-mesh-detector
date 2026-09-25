#!/usr/bin/env python3
"""Fetches the PlantWild dataset into data/PlantWild/<class name>/*.jpg — the
layout src/data/plantwild.py expects — from its Hugging Face dataset release
(https://huggingface.co/datasets/uqtwei2/PlantWild).

PlantWild ships as a single zip whose reference loader
(https://github.com/tqwei05/MVPDR/blob/main/datasets/plantwild.py) reads
images from an `images/<class name>/*.jpg` directory inside the archive; this
script extracts that directory (searched for by name rather than a hardcoded
path, since the exact nesting wasn't independently verified against the real
archive) directly into data/PlantWild/.

PlantWild is released under CC-BY-NC-ND-4.0 (non-commercial, no derivatives) —
fine for this research/education simulation, but do not redistribute
repackaged copies.

Run:
    python scripts/download_plantwild.py
"""

from __future__ import annotations

import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

DEST = Path(__file__).resolve().parent.parent / "data" / "PlantWild"

_ARCHIVE_URL = "https://huggingface.co/datasets/uqtwei2/PlantWild/resolve/main/plantwild.zip"


def _download_archive(dest_zip: Path) -> None:
    print(f"Downloading {_ARCHIVE_URL} ...")
    urllib.request.urlretrieve(_ARCHIVE_URL, dest_zip)


def _find_images_dir(root: Path) -> Path | None:
    """Depth-first search for a directory literally named 'images' whose
    immediate children are themselves directories (i.e. per-class folders).
    """
    for candidate in root.rglob("images"):
        if candidate.is_dir() and any(c.is_dir() for c in candidate.iterdir()):
            return candidate
    return None


def main() -> int:
    if DEST.exists() and any(DEST.iterdir()):
        print(f"{DEST} already has data — nothing to do.")
        return 0

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        archive = tmp_path / "plantwild.zip"
        _download_archive(archive)

        print("Extracting (this is a large archive, may take a while)...")
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(tmp_path)

        images_dir = _find_images_dir(tmp_path)
        if images_dir is None:
            print(
                "Could not find an 'images/<class>/*' directory inside the extracted "
                "archive — PlantWild's release layout may have changed.",
                file=sys.stderr,
            )
            return 1

        DEST.mkdir(parents=True, exist_ok=True)
        counts: dict[str, int] = {}
        for class_dir in sorted(images_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            out_dir = DEST / class_dir.name
            out_dir.mkdir(exist_ok=True)
            n = 0
            for img in class_dir.iterdir():
                if not img.is_file():
                    continue
                img.rename(out_dir / img.name)
                n += 1
            counts[class_dir.name] = n

    n_classes = len(counts)
    n_images = sum(counts.values())
    if n_images == 0:
        print("No images were found in the downloaded archive — PlantWild's layout may have changed.", file=sys.stderr)
        return 1
    print(f"Done. {n_images} images across {n_classes} classes written to {DEST}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
