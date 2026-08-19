"""The PlantDoc -> PlantVillage crop/disease label mapping, and the set of
PlantVillage classes that mapping covers. Used by src/data/merged.py to
combine the two datasets 1:1 per class into one project dataset.
"""

from __future__ import annotations

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
