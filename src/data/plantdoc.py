"""PlantDoc dataset helpers.

Two independent uses of PlantDoc live here:
  - load_plantdoc_tomato_paths / load_plantdoc_apple_paths: loaders for
    PlantDoc's Tomato- and Apple-labeled images, folded into the
    Tomato/Apple Dirichlet mesh pipelines' canonical label spaces (see
    src/validation/tomato_mesh_dataset.py and apple_mesh_dataset.py).
    PlantDoc ships its own train/test split; both subfolders are pooled
    here since the mesh pipelines define their own global split downstream.
  - PLANTDOC_TO_PLANTVILLAGE / overlapping_plantvillage_classes: the
    PlantDoc -> PlantVillage crop/disease label mapping, and the set of
    PlantVillage classes that mapping covers. Used by src/data/merged.py to
    combine the two datasets 1:1 per class into one project dataset.
"""

from __future__ import annotations

from pathlib import Path

PLANTDOC_CANONICAL_MAP: dict[str, str] = {
    "Tomato leaf": "healthy",
    "Tomato Early blight leaf": "Early_blight",
    "Tomato leaf bacterial spot": "Bacterial_spot",
    "Tomato leaf late blight": "Late_blight",
    "Tomato leaf mosaic virus": "Tomato_mosaic_virus",
    "Tomato leaf yellow virus": "Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato mold leaf": "Leaf_Mold",
    "Tomato Septoria leaf spot": "Septoria_leaf_spot",
    "Tomato two spotted spider mites leaf": "Spider_mites Two-spotted_spider_mite",
}

# Apple's PlantDoc folders use TWO different naming conventions between
# train and test (confirmed against the real download, not assumed) --
# train uses free-text names ("Apple leaf"), test uses underscore-joined
# PlantVillage-style names ("Apple_healthy"). There is no PlantDoc folder
# (train or test) for Black_rot at all.
PLANTDOC_APPLE_TRAIN_MAP: dict[str, str] = {
    "Apple leaf": "healthy",
    "Apple rust leaf": "Cedar_apple_rust",
    "Apple Scab Leaf": "Apple_scab",
}
PLANTDOC_APPLE_TEST_MAP: dict[str, str] = {
    "Apple_Apple_scab": "Apple_scab",
    "Apple_Cedar_apple_rust": "Cedar_apple_rust",
    "Apple_healthy": "healthy",
}

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")


def load_plantdoc_tomato_paths(root: str | Path) -> list[tuple[str, str]]:
    """Returns (path, canonical_disease_name) pairs for every image under
    root/train and root/test whose immediate folder name is a known
    PlantDoc Tomato class (see PLANTDOC_CANONICAL_MAP). Folders not in the
    map (other crops, or unrecognized names) are ignored.
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"PlantDoc root not found: {root}")

    results: list[tuple[str, str]] = []
    for split in ("train", "test"):
        split_dir = root / split
        if not split_dir.is_dir():
            continue
        for folder_name, canonical_name in PLANTDOC_CANONICAL_MAP.items():
            class_dir = split_dir / folder_name
            if not class_dir.is_dir():
                continue
            for path in sorted(class_dir.iterdir()):
                if path.suffix in IMAGE_EXTENSIONS:
                    results.append((str(path), canonical_name))
    return results


def load_plantdoc_apple_paths(root: str | Path) -> list[tuple[str, str]]:
    """Same idea as load_plantdoc_tomato_paths, but Apple's train and test
    splits use different folder-naming conventions (see
    PLANTDOC_APPLE_TRAIN_MAP / PLANTDOC_APPLE_TEST_MAP above), so each
    split is walked against its own map rather than one shared map.
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"PlantDoc root not found: {root}")

    results: list[tuple[str, str]] = []
    for split, canonical_map in (("train", PLANTDOC_APPLE_TRAIN_MAP), ("test", PLANTDOC_APPLE_TEST_MAP)):
        split_dir = root / split
        if not split_dir.is_dir():
            continue
        for folder_name, canonical_name in canonical_map.items():
            class_dir = split_dir / folder_name
            if not class_dir.is_dir():
                continue
            for path in sorted(class_dir.iterdir()):
                if path.suffix in IMAGE_EXTENSIONS:
                    results.append((str(path), canonical_name))
    return results


# PlantDoc folder name -> (PlantVillage crop, PlantVillage disease). Every
# key must match a raw PlantDoc class folder exactly; every value must match
# an entry already present in the PlantVillage dataset's crop_classes /
# disease_classes (checked at load time, not just here).
PLANTDOC_TO_PLANTVILLAGE: dict[str, tuple[str, str]] = {
    "Apple Scab Leaf": ("Apple", "Apple_scab"),
    "Apple leaf": ("Apple", "healthy"),
    "Apple rust leaf": ("Apple", "Cedar_apple_rust"),
    "Bell_pepper leaf spot": ("Pepper,_bell", "Bacterial_spot"),
    "Bell_pepper leaf": ("Pepper,_bell", "healthy"),
    "Corn Gray leaf spot": ("Corn", "Cercospora_leaf_spot Gray_leaf_spot"),
    "Corn leaf blight": ("Corn", "Northern_Leaf_Blight"),
    "Corn rust leaf": ("Corn", "Common_rust"),
    "Potato leaf early blight": ("Potato", "Early_blight"),
    "Potato leaf late blight": ("Potato", "Late_blight"),
    "Tomato Early blight leaf": ("Tomato", "Early_blight"),
    "Tomato Septoria leaf spot": ("Tomato", "Septoria_leaf_spot"),
    "Tomato leaf bacterial spot": ("Tomato", "Bacterial_spot"),
    "Tomato leaf late blight": ("Tomato", "Late_blight"),
    "Tomato leaf mosaic virus": ("Tomato", "Tomato_mosaic_virus"),
    "Tomato leaf yellow virus": ("Tomato", "Tomato_Yellow_Leaf_Curl_Virus"),
    "Tomato leaf": ("Tomato", "healthy"),
    "Tomato mold leaf": ("Tomato", "Leaf_Mold"),
    "grape leaf black rot": ("Grape", "Black_rot"),
    "grape leaf": ("Grape", "healthy"),
    # "Tomato two spotted spider mites leaf" is deliberately omitted: PlantDoc
    # has only 2 images for it, not enough to count as real domain coverage.
}


def overlapping_plantvillage_classes() -> set[str]:
    return {f"{crop}___{disease}" for crop, disease in PLANTDOC_TO_PLANTVILLAGE.values()}
