"""PlantWild loading and remapping onto PlantVillage's (crop, disease) label
space, so a large-scale, in-the-wild image source can be mixed into training
the same way PlantDoc's are (see src/data/plantdoc.py for the sibling
loader, and src/data/mixing.py for the batch sampler that combines domains).

Unlike PlantDoc (internet-scraped single-image search results), PlantWild
is a purpose-built in-the-wild benchmark whose disease taxonomy already
overlaps heavily with PlantVillage's — see PLANTWILD_TO_PLANTVILLAGE below.
"""

from __future__ import annotations
from pathlib import Path
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder

from src.data.plantvillage import build_eval_transform, build_train_transform

# PlantWild folder name -> (PlantVillage crop, PlantVillage disease). Every
# key must match a raw PlantWild class folder exactly; every value must
# match an entry already present in the PlantVillage dataset's
# crop_classes / disease_classes (checked at load time, not just here).
# The other 55 of PlantWild's 89 classes have no PlantVillage equivalent —
# mostly crops PlantVillage doesn't cover at all (banana, basil, bean,
# broccoli, cabbage, carrot, cauliflower, celery, coffee, cucumber,
# eggplant, garlic, ginger, lettuce, maple, plum, rice, tobacco, zucchini)
# — and are left unmapped, dropped at load time with a printed warning.
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
    "peach leaf": ("Peach", "healthy"),
    "potato early blight": ("Potato", "Early_blight"),
    "potato late blight": ("Potato", "Late_blight"),
    "potato leaf": ("Potato", "healthy"),
    "raspberry leaf": ("Raspberry", "healthy"),
    "soybean leaf": ("Soybean", "healthy"),
    "squash leaf": ("Squash", "healthy"),
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


class PlantWildDataset(Dataset):
    """Wraps torchvision's ImageFolder, remapped onto an already-built
    PlantVillage crop/disease label space (so it can share a model head
    with PlantVillage and PlantDoc). PlantWild folders with no PlantVillage
    crop/disease equivalent are dropped with a printed warning. A mapped
    (crop, disease) pair that ISN'T in `crop_classes`/`disease_classes`
    raises immediately, since that would otherwise silently point at the
    wrong label index.

    PlantWild is intended purely as extra, domain-shifted training signal
    (see src/data/mixing.py), so __getitem__ always applies the training
    augmentation transform. `eval_transform` is exposed separately (see
    `_lookup`) for callers that need a deterministic view of this dataset —
    e.g. carving a reproducible probe-set/global-test-set slice out of it
    (see EvalView in src/data/mixing.py).
    """

    def __init__(
        self,
        root: str | Path,
        crop_classes: list[str],
        disease_classes: list[str],
        pv_class_to_crop_disease: dict[int, tuple[int, int]],
        image_size: int = 160,
    ):
        self.base = ImageFolder(str(root))
        self.train_transform = build_train_transform(image_size)
        self.eval_transform = build_eval_transform(image_size)
        crop_to_idx = {c: i for i, c in enumerate(crop_classes)}
        disease_to_idx = {d: i for i, d in enumerate(disease_classes)}

        raw_class_to_pair: dict[int, tuple[int, int]] = {}
        dropped = []
        for raw_idx, raw_name in enumerate(self.base.classes):
            mapped = PLANTWILD_TO_PLANTVILLAGE.get(raw_name)
            if mapped is None:
                dropped.append(raw_name)
                continue
            crop, disease = mapped
            if crop not in crop_to_idx or disease not in disease_to_idx:
                raise ValueError(
                    f"PlantWild class {raw_name!r} maps to (crop={crop!r}, disease={disease!r}), "
                    f"which is not in the PlantVillage label space — fix PLANTWILD_TO_PLANTVILLAGE"
                )
            raw_class_to_pair[raw_idx] = (crop_to_idx[crop], disease_to_idx[disease])
        if dropped:
            print(f"PlantWild: dropping {len(dropped)} class(es) with no PlantVillage match: {dropped}")

        # "class" here means the actual PlantVillage folder (a specific
        # crop+disease pair, e.g. "Corn___healthy") — checking crop and
        # disease coverage separately would miss e.g. Corn___healthy being
        # uncovered even though "healthy" itself is covered via some other
        # crop, and "Corn" is covered via some other Corn disease.
        covered_pairs = set(raw_class_to_pair.values())
        uncovered = sorted(
            f"{crop_classes[c]}___{disease_classes[d]}"
            for c, d in set(pv_class_to_crop_disease.values())
            if (c, d) not in covered_pairs
        )
        if uncovered:
            print(
                f"PlantWild: {len(uncovered)} PlantVillage class(es) have no PlantWild coverage, "
                f"so they train on PlantVillage only: {uncovered}"
            )

        self._indices = [i for i, (_, raw_cls) in enumerate(self.base.samples) if raw_cls in raw_class_to_pair]
        self._raw_class_to_pair = raw_class_to_pair

    def __len__(self) -> int:
        return len(self._indices)

    def _lookup(self, i: int):
        """Untransformed (image, crop_idx, disease_idx) for retained sample
        `i` — shared by __getitem__ (applies train_transform) and EvalView
        (applies eval_transform instead), so probe/global-test carves stay
        reproducible without duplicating the remap lookup.
        """
        idx = self._indices[i]
        image, raw_class = self.base[idx]
        crop_idx, disease_idx = self._raw_class_to_pair[raw_class]
        return image, crop_idx, disease_idx

    def __getitem__(self, i: int):
        image, crop_idx, disease_idx = self._lookup(i)
        return self.train_transform(image), crop_idx, disease_idx

    @property
    def raw_labels(self) -> list[int]:
        """Original ImageFolder class id (pre-PlantVillage-remap) of each
        retained sample — a finer stratum than crop/disease alone, used by
        MixedDomainBatchSampler to class-balance within PlantWild.
        """
        return [self.base.samples[idx][1] for idx in self._indices]

    @property
    def pairs(self) -> list[tuple[int, int]]:
        """(crop_idx, disease_idx) for each retained sample, in the same
        order as __getitem__ — used to compute inverse-frequency class
        weights when this dataset is a node's sole training data (see
        compute_crop_class_weights_from_pairs /
        compute_disease_class_weights_from_pairs in
        src/data/plantvillage.py), without loading any images.
        """
        return [self._raw_class_to_pair[raw_cls] for _, raw_cls in (self.base.samples[idx] for idx in self._indices)]


def load_plantwild_dataset(
    root: str | Path | None,
    crop_classes: list[str],
    disease_classes: list[str],
    pv_class_to_crop_disease: dict[int, tuple[int, int]],
    image_size: int = 160,
) -> PlantWildDataset | None:
    """Returns None (domain mixing disabled) if `root` is unset or doesn't
    exist yet, so PlantWild stays fully opt-in — leaving data.plantwild_root
    unset in config.yaml reproduces today's behaviour.
    """
    if not root:
        return None
    root = Path(root)
    if not root.exists():
        return None
    return PlantWildDataset(root, crop_classes, disease_classes, pv_class_to_crop_disease, image_size=image_size)
