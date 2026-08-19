"""Loader for PlantWild's Tomato-labeled images (v1 and v2 builds), folded
into the Tomato Dirichlet mesh pipeline's canonical label space (see
src/validation/tomato_mesh_dataset.py). Both versions are pooled by the
caller; global perceptual-hash dedup (tomato_mesh_dataset.py) is
responsible for catching any near-duplicate images between the two.
"""

from __future__ import annotations

from pathlib import Path

PLANTWILD_V1_CANONICAL_MAP: dict[str, str] = {
    "tomato leaf": "healthy",
    "tomato bacterial leaf spot": "Bacterial_spot",
    "tomato early blight": "Early_blight",
    "tomato late blight": "Late_blight",
    "tomato leaf mold": "Leaf_Mold",
    "tomato mosaic virus": "Tomato_mosaic_virus",
    "tomato septoria leaf spot": "Septoria_leaf_spot",
    "tomato yellow leaf curl virus": "Tomato_Yellow_Leaf_Curl_Virus",
}

# v2 has no healthy/leaf equivalent for Tomato -- confirmed against the
# real download, not assumed.
PLANTWILD_V2_CANONICAL_MAP: dict[str, str] = {
    k: v for k, v in PLANTWILD_V1_CANONICAL_MAP.items() if k != "tomato leaf"
}

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")


def _load_from_root(root: str | Path, canonical_map: dict[str, str]) -> list[tuple[str, str]]:
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"PlantWild root not found: {root}")

    results: list[tuple[str, str]] = []
    for folder_name, canonical_name in canonical_map.items():
        class_dir = root / folder_name
        if not class_dir.is_dir():
            continue
        for path in sorted(class_dir.iterdir()):
            if path.suffix in IMAGE_EXTENSIONS:
                results.append((str(path), canonical_name))
    return results


def load_plantwild_v1_tomato_paths(root: str | Path) -> list[tuple[str, str]]:
    return _load_from_root(root, PLANTWILD_V1_CANONICAL_MAP)


def load_plantwild_v2_tomato_paths(root: str | Path) -> list[tuple[str, str]]:
    return _load_from_root(root, PLANTWILD_V2_CANONICAL_MAP)
