#!/usr/bin/env python3
"""Replaces PlantVillage with the Kaggle "Leaves: Healthy or Diseased" dataset
(https://www.kaggle.com/datasets/prasanshasatpathy/leaves-healthy-or-diseased).

The whole pipeline reads PlantVillage-style class folders, `<Crop>___<Disease>/`
(src/data/plantvillage.py). This script rewrites the Kaggle download into that layout,
with only two "diseases", so nothing else has to change:

    data/LeavesHealthyDiseased/
        Mango___healthy/  Mango___diseased/  Guava___healthy/  ...

The crop head then learns the plant and the disease head learns healthy vs diseased.

The Kaggle layout isn't fixed here, so each image's plant and condition are read from
its path. All of these work, with train/test/valid split folders merged (the pipeline
makes its own splits):
    Mango/healthy/x.jpg            Mango/diseased/x.jpg
    Mango healthy (P0b)/x.jpg      train/Mango diseased (P0a)/x.jpg
    Mango_healthy/x.jpg            healthy/x.jpg  (no plant: crop "Leaf")
If some images' condition can't be worked out, the script stops and shows examples,
rather than guessing.

    # download with kagglehub (pip install kagglehub; may need a Kaggle API token), then convert
    python scripts/prepare_leaves_healthy_diseased.py --download
    # or: download the zip from the Kaggle page yourself, unzip it, then
    python scripts/prepare_leaves_healthy_diseased.py --source ~/Downloads/leaves-healthy-or-diseased
"""

from __future__ import annotations

import argparse
import hashlib
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

KAGGLE_DATASET = "prasanshasatpathy/leaves-healthy-or-diseased"
DEFAULT_OUT = Path(__file__).resolve().parent.parent / "data" / "LeavesHealthyDiseased"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

HEALTHY_WORDS = {"healthy", "normal", "fresh"}
DISEASED_WORDS = {"diseased", "disease", "unhealthy", "infected", "sick", "damaged"}
# Folder names that are neither a plant nor a condition.
IGNORED_FOLDERS = {"train", "training", "test", "testing", "valid", "val", "validation", "images", "image",
                   "data", "dataset", "leaves", "leaf", "plants", "plant", "all", "leaves healthy or diseased"}


def _words(name: str) -> list[str]:
    return [w for w in re.split(r"[\s_\-.,]+", name.lower()) if w]


def _clean_plant(name: str) -> str:
    """'Mango healthy (P0b)' -> 'Mango'; 'alstonia_scholaris' -> 'Alstonia_scholaris'."""
    name = re.sub(r"\(.*?\)", " ", name)  # "(P0b)"-style codes
    kept = [w for w in re.split(r"[\s_\-]+", name) if w and w.lower() not in HEALTHY_WORDS | DISEASED_WORDS]
    plant = "_".join(kept).strip("_.,")
    return plant[:1].upper() + plant[1:] if plant else ""


def classify_path(relative: Path) -> tuple[str, str] | None:
    """(plant, 'healthy' | 'diseased') from an image path relative to the dataset root, or
    None when no folder says healthy or diseased. The nearest folder that names a condition
    wins; the plant is that folder with the condition word removed, else the nearest plain
    folder above it that isn't a split/container name, else "Leaf".
    """
    folders = list(relative.parent.parts)
    for i in range(len(folders) - 1, -1, -1):
        words = set(_words(folders[i]))
        condition = ("healthy" if words & HEALTHY_WORDS else "diseased" if words & DISEASED_WORDS else None)
        if condition is None:
            continue
        plant = _clean_plant(folders[i])
        j = i - 1
        while not plant and j >= 0:
            if folders[j].lower().strip() not in IGNORED_FOLDERS:
                plant = _clean_plant(folders[j])
            j -= 1
        return (plant or "Leaf"), condition
    return None


def _file_hash(path: Path) -> str:
    return hashlib.sha1(path.read_bytes()).hexdigest()


def convert(source: Path, out: Path, overwrite: bool = False) -> Counter:
    source, out = Path(source), Path(out)
    images = sorted(p for p in source.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    if not images:
        raise SystemExit(f"No images found under {source}")

    labelled, unknown = [], []
    for path in images:
        label = classify_path(path.relative_to(source))
        (labelled if label else unknown).append((path, label))
    if unknown:
        examples = "\n  ".join(str(p.relative_to(source)) for p, _ in unknown[:8])
        raise SystemExit(
            f"Couldn't tell healthy from diseased for {len(unknown)} of {len(images)} images, e.g.:\n  {examples}\n"
            f"Their folders need a 'healthy' or 'diseased' word; send this output so the rules can be extended.")

    if out.exists():
        if not overwrite:
            raise SystemExit(f"{out} already exists; pass --overwrite to rebuild it")
        shutil.rmtree(out)

    counts: Counter = Counter()
    seen: dict[str, str] = {}  # content hash -> class: the same photo in train/ and test/ is kept once
    conflicts = 0
    for path, (plant, condition) in labelled:
        digest = _file_hash(path)
        class_name = f"{plant}___{condition}"
        if digest in seen:
            conflicts += seen[digest] != class_name
            continue
        seen[digest] = class_name
        target_dir = out / class_name
        target_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target_dir / f"{digest[:16]}{path.suffix.lower()}")
        counts[class_name] += 1

    duplicates = len(labelled) - sum(counts.values())
    print(f"Wrote {sum(counts.values())} images in {len(counts)} classes to {out}/"
          + (f" ({duplicates} duplicate files skipped" + (f", {conflicts} with conflicting labels" if conflicts else "")
             + ")" if duplicates else ""))
    for class_name, n in sorted(counts.items()):
        print(f"  {class_name:45s} {n:6d}")
    return counts


def download() -> Path:
    try:
        import kagglehub
    except ImportError:
        raise SystemExit("pip install kagglehub  (or download the zip from the Kaggle page and use --source)")
    return Path(kagglehub.dataset_download(KAGGLE_DATASET))


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--source", help="Unzipped Kaggle download")
    group.add_argument("--download", action="store_true", help=f"Download {KAGGLE_DATASET} with kagglehub first")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="Output root (config.yaml data.root)")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output folder")
    args = parser.parse_args()

    source = download() if args.download else Path(args.source).expanduser()
    print(f"Converting {source}")
    convert(source, Path(args.out), overwrite=args.overwrite)


if __name__ == "__main__":
    sys.exit(main())
