"""Shared logic for remapping a domain-shift dataset's own raw class folders
onto PlantVillage's (crop, disease) label space — used by both PlantDoc and
PlantWild, which otherwise duplicate the exact same drop/validate/warn shape.
"""

from __future__ import annotations
from typing import Callable


def build_class_remap(
    raw_classes: list[str],
    mapping: dict[str, tuple[str, str]],
    crop_classes: list[str],
    disease_classes: list[str],
    pv_class_to_crop_disease: dict[int, tuple[int, int]],
    dataset_name: str,
    normalize: Callable[[str], str] = lambda s: s,
) -> dict[int, tuple[int, int]]:
    """Builds raw ImageFolder class id -> (crop_idx, disease_idx), remapped
    onto an already-built PlantVillage crop/disease label space.

    Raw classes with no `mapping` entry (after `normalize`) are dropped with
    a printed warning. A mapped (crop, disease) pair that ISN'T in
    `crop_classes`/`disease_classes` raises immediately, since that would
    otherwise silently point at the wrong label index. PlantVillage classes
    with no coverage among the retained raw classes are also printed as a
    warning (they simply train on other datasets/nodes only).
    """
    crop_to_idx = {c: i for i, c in enumerate(crop_classes)}
    disease_to_idx = {d: i for i, d in enumerate(disease_classes)}

    raw_class_to_pair: dict[int, tuple[int, int]] = {}
    dropped = []
    for raw_idx, raw_name in enumerate(raw_classes):
        mapped = mapping.get(normalize(raw_name))
        if mapped is None:
            dropped.append(raw_name)
            continue
        crop, disease = mapped
        if crop not in crop_to_idx or disease not in disease_to_idx:
            raise ValueError(
                f"{dataset_name} class {raw_name!r} maps to (crop={crop!r}, disease={disease!r}), "
                f"which is not in the PlantVillage label space — fix the mapping table"
            )
        raw_class_to_pair[raw_idx] = (crop_to_idx[crop], disease_to_idx[disease])
    if dropped:
        print(f"{dataset_name}: dropping {len(dropped)} class(es) with no PlantVillage match: {dropped}")

    # "class" here means the actual PlantVillage folder (a specific
    # crop+disease pair, e.g. "Corn___healthy") — checking crop and disease
    # coverage separately would miss e.g. Corn___healthy being uncovered even
    # though "healthy" itself is covered via some other crop, and "Corn" is
    # covered via some other Corn disease.
    covered_pairs = set(raw_class_to_pair.values())
    uncovered = sorted(
        f"{crop_classes[c]}___{disease_classes[d]}"
        for c, d in set(pv_class_to_crop_disease.values())
        if (c, d) not in covered_pairs
    )
    if uncovered:
        print(
            f"{dataset_name}: {len(uncovered)} PlantVillage class(es) have no {dataset_name} coverage, "
            f"so they train on other datasets/nodes only: {uncovered}"
        )

    return raw_class_to_pair
