"""Merges PlantVillage with PlantDoc and PlantWild into ONE dataset, then
(optionally) keeps only the crops with the most images.

Every PlantDoc/PlantWild class folder is mapped onto a PlantVillage
"Crop___disease" class, so the merged dataset keeps PlantVillageDataset's
exact shape (`.base.samples`/`.targets`/`.labels`) and everything downstream
— the continual stream, partition_nodes, filter_dataset_by_crop — works on
it unchanged. Folders with no PlantVillage equivalent (PlantWild's banana,
rice, coffee, ...) are dropped and listed.

Source folders are found by walking each configured root recursively: any
directory that directly holds images is one class, named after the
directory. That covers PlantDoc's train/<class> + test/<class> and
PlantWild's v1/v2 images/<class> layouts without hardcoding their nesting.

config.yaml (data.*):
  root:              PlantVillage
  extra_sources:     {plantdoc: [roots...], plantwild: [roots...]}
  included_crops:    a list of crop names, or "auto" (top auto_crop_count
                     crops by merged image count, among crops with at least
                     auto_crop_min_diseases disease classes)
  auto_crop_node_groups: with "auto", [[nodes for the largest crop],
                     [nodes for the 2nd], ...] — fills manual_node_crops
"""

from __future__ import annotations

import os
import re
from collections import Counter
from pathlib import Path

from src.data.plantvillage import (
    PlantVillageDataset,
    _parse_crop_disease,
    filter_dataset_by_crop,
    load_full_dataset,
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# Folder names that don't normalize to a PlantVillage class name on their
# own (keys are normalized, see _normalize). Folders already named like a
# PlantVillage class (e.g. PlantWild's "Apple_Apple_scab", PlantDoc test's
# "Apple_healthy") match without an entry here.
PLANTDOC_TO_PLANTVILLAGE: dict[str, tuple[str, str]] = {
    "apple scab leaf": ("Apple", "Apple_scab"),
    "apple leaf": ("Apple", "healthy"),
    "apple rust leaf": ("Apple", "Cedar_apple_rust"),
    "bell pepper leaf spot": ("Pepper,_bell", "Bacterial_spot"),
    "bell pepper leaf": ("Pepper,_bell", "healthy"),
    "blueberry leaf": ("Blueberry", "healthy"),
    "cherry leaf": ("Cherry", "healthy"),
    "corn gray leaf spot": ("Corn", "Cercospora_leaf_spot Gray_leaf_spot"),
    "corn leaf blight": ("Corn", "Northern_Leaf_Blight"),
    "corn rust leaf": ("Corn", "Common_rust"),
    "peach leaf": ("Peach", "healthy"),
    "potato leaf early blight": ("Potato", "Early_blight"),
    "potato leaf late blight": ("Potato", "Late_blight"),
    "raspberry leaf": ("Raspberry", "healthy"),
    "soyabean leaf": ("Soybean", "healthy"),
    "squash powdery mildew leaf": ("Squash", "Powdery_mildew"),
    "strawberry leaf": ("Strawberry", "healthy"),
    "tomato early blight leaf": ("Tomato", "Early_blight"),
    "tomato septoria leaf spot": ("Tomato", "Septoria_leaf_spot"),
    "tomato leaf bacterial spot": ("Tomato", "Bacterial_spot"),
    "tomato leaf late blight": ("Tomato", "Late_blight"),
    "tomato leaf mosaic virus": ("Tomato", "Tomato_mosaic_virus"),
    "tomato leaf yellow virus": ("Tomato", "Tomato_Yellow_Leaf_Curl_Virus"),
    "tomato leaf": ("Tomato", "healthy"),
    "tomato mold leaf": ("Tomato", "Leaf_Mold"),
    "tomato two spotted spider mites leaf": ("Tomato", "Spider_mites Two-spotted_spider_mite"),
    "grape leaf black rot": ("Grape", "Black_rot"),
    "grape leaf": ("Grape", "healthy"),
}

PLANTWILD_TO_PLANTVILLAGE: dict[str, tuple[str, str]] = {
    "apple black rot": ("Apple", "Black_rot"),
    "apple leaf": ("Apple", "healthy"),
    "apple rust": ("Apple", "Cedar_apple_rust"),
    "apple scab": ("Apple", "Apple_scab"),
    "bell pepper leaf": ("Pepper,_bell", "healthy"),
    "bell pepper leaf spot": ("Pepper,_bell", "Bacterial_spot"),
    "blueberry leaf": ("Blueberry", "healthy"),
    "cherry leaf": ("Cherry", "healthy"),
    "cherry powdery mildew": ("Cherry", "Powdery_mildew"),
    "citrus greening disease": ("Orange", "Haunglongbing_(Citrus_greening)"),
    "corn gray leaf spot": ("Corn", "Cercospora_leaf_spot Gray_leaf_spot"),
    "corn leaf": ("Corn", "healthy"),
    "corn northern leaf blight": ("Corn", "Northern_Leaf_Blight"),
    "corn rust": ("Corn", "Common_rust"),
    "grape black rot": ("Grape", "Black_rot"),
    "grape leaf": ("Grape", "healthy"),
    "grape leaf spot": ("Grape", "Leaf_blight_(Isariopsis_Leaf_Spot)"),
    "peach leaf": ("Peach", "healthy"),
    "potato early blight": ("Potato", "Early_blight"),
    "potato late blight": ("Potato", "Late_blight"),
    "potato leaf": ("Potato", "healthy"),
    "raspberry leaf": ("Raspberry", "healthy"),
    "soybean leaf": ("Soybean", "healthy"),
    "squash powdery mildew": ("Squash", "Powdery_mildew"),
    "strawberry leaf": ("Strawberry", "healthy"),
    "strawberry leaf scorch": ("Strawberry", "Leaf_scorch"),
    "tomato bacterial leaf spot": ("Tomato", "Bacterial_spot"),
    "tomato early blight": ("Tomato", "Early_blight"),
    "tomato late blight": ("Tomato", "Late_blight"),
    "tomato leaf": ("Tomato", "healthy"),
    "tomato leaf mold": ("Tomato", "Leaf_Mold"),
    "tomato mosaic virus": ("Tomato", "Tomato_mosaic_virus"),
    "tomato septoria leaf spot": ("Tomato", "Septoria_leaf_spot"),
    "tomato yellow leaf curl virus": ("Tomato", "Tomato_Yellow_Leaf_Curl_Virus"),
}

SOURCE_MAPS = {"plantdoc": PLANTDOC_TO_PLANTVILLAGE, "plantwild": PLANTWILD_TO_PLANTVILLAGE}


def _normalize(name: str) -> str:
    """Lowercase, '+'/'_'/'-' -> space, whitespace collapsed."""
    return re.sub(r"\s+", " ", re.sub(r"[+_\-]", " ", name.strip().lower())).strip()


def _image_dirs(root: Path):
    """(class folder name, sorted image paths) for every directory under
    `root` that directly holds images."""
    for dirpath, dirnames, filenames in os.walk(root):
        # skip zip-extraction junk (__MACOSX mirrors every class folder) and hidden dirs
        dirnames[:] = sorted(d for d in dirnames if not d.startswith((".", "__MACOSX")))
        images = sorted(f for f in filenames if Path(f).suffix.lower() in IMAGE_EXTENSIONS)
        if images:
            yield Path(dirpath).name, [os.path.join(dirpath, f) for f in images]


def merge_extra_sources(dataset: PlantVillageDataset, extra_sources: dict[str, list[str]]) -> dict[str, Counter]:
    """Appends every mappable PlantDoc/PlantWild image to `dataset` in place
    (as extra samples of the matching PlantVillage class). Returns
    {source: Counter(PlantVillage class name -> images added)}.
    """
    by_pv_name = {_normalize(name): name for name in dataset.base.classes}
    added: dict[str, Counter] = {"plantvillage": Counter(dataset.base.classes[t] for t in dataset.base.targets)}
    for source, roots in (extra_sources or {}).items():
        if source not in SOURCE_MAPS:
            raise ValueError(f"data.extra_sources: unknown source {source!r} (expected one of {sorted(SOURCE_MAPS)})")
        mapping = SOURCE_MAPS[source]
        counts: Counter = Counter()
        dropped: Counter = Counter()
        for root in roots if isinstance(roots, list) else [roots]:
            root = Path(root)
            if not root.exists():
                raise FileNotFoundError(f"data.extra_sources.{source}: {root} does not exist")
            for folder, paths in _image_dirs(root):
                key = _normalize(folder)
                if key in mapping:
                    crop, disease = mapping[key]
                    pv_name = f"{crop}___{disease}"
                else:
                    pv_name = by_pv_name.get(key)
                cls_idx = dataset.base.class_to_idx.get(pv_name) if pv_name else None
                if cls_idx is None:
                    dropped[folder] += len(paths)
                    continue
                dataset.base.samples.extend((p, cls_idx) for p in paths)
                counts[pv_name] += len(paths)
        if dropped:
            print(f"{source}: dropped {sum(dropped.values())} images in {len(dropped)} folder(s) with no "
                  f"PlantVillage class: {sorted(dropped)}")
        added[source] = counts
    dataset.base.imgs = dataset.base.samples
    dataset.base.targets = [cls_idx for _, cls_idx in dataset.base.samples]
    return added


def select_top_crops(counts_by_source: dict[str, Counter], n: int, min_diseases: int) -> list[str]:
    """The `n` crops with the most merged images, among crops with at least
    `min_diseases` distinct disease classes (healthy counts as one). Prints
    the per-crop, per-source table the choice was made from."""
    per_crop: dict[str, Counter] = {}
    diseases: dict[str, set[str]] = {}
    for source, counts in counts_by_source.items():
        for pv_name, count in counts.items():
            crop, disease = _parse_crop_disease(pv_name)
            per_crop.setdefault(crop, Counter())[source] += count
            diseases.setdefault(crop, set()).add(disease)
    ranked = sorted(per_crop, key=lambda c: sum(per_crop[c].values()), reverse=True)
    eligible = [c for c in ranked if len(diseases[c]) >= min_diseases]
    selected = eligible[:n]
    if len(selected) < n:
        raise ValueError(f"only {len(selected)} crop(s) have >= {min_diseases} disease classes; asked for {n}")

    sources = list(counts_by_source)
    print("Merged images per crop (" + " + ".join(sources) + "):")
    for crop in ranked:
        total = sum(per_crop[crop].values())
        parts = ", ".join(f"{s} {per_crop[crop][s]}" for s in sources)
        mark = f"  <- selected #{selected.index(crop) + 1}" if crop in selected else (
            f"  (skipped: only {len(diseases[crop])} class)" if crop not in eligible else "")
        print(f"  {crop:14s} {total:6d}  [{parts}]  {len(diseases[crop])} classes{mark}")
    return selected


def load_dataset(cfg) -> PlantVillageDataset:
    """The project dataset per config.yaml's data.* keys: PlantVillage,
    merged with data.extra_sources, restricted to data.included_crops. With
    included_crops "auto", also fills data.manual_node_crops from
    data.auto_crop_node_groups (so the stream/partition read the chosen
    crops like any hand-written assignment)."""
    dataset = load_full_dataset(cfg.get("data.root"), cfg.get("data.image_size", 224))
    counts = merge_extra_sources(dataset, cfg.get("data.extra_sources", None))

    included = cfg.get("data.included_crops", None)
    if included == "auto":
        groups = cfg.get("data.auto_crop_node_groups", None)
        n = cfg.get("data.auto_crop_count", len(groups) if groups else 2)
        included = select_top_crops(counts, n, cfg.get("data.auto_crop_min_diseases", 2))
        if groups:
            if len(groups) != len(included):
                raise ValueError(f"data.auto_crop_node_groups has {len(groups)} group(s) for {len(included)} crop(s)")
            cfg.set("data.manual_node_crops", {
                str(node): [crop] for crop, nodes in zip(included, groups) for node in nodes
            })
            print("Node crops: " + ", ".join(f"{k}={v[0]}" for k, v in cfg.get("data.manual_node_crops").items()))
    if included:
        filter_dataset_by_crop(dataset, included_crops=list(included))
    return dataset
