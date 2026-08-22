"""PlantWild loading and remapping onto PlantVillage's (crop, disease) label
space, so this larger in-the-wild dataset can serve as its own node (see
src/data/multi_dataset.py) alongside PlantVillage and PlantDoc.

PlantWild's own reference loader
(https://github.com/tqwei05/MVPDR/blob/main/datasets/plantwild.py) reads
images from an `images/<class name>/*.jpg` directory, with class folder names
lowercase and word-separated by '+' (e.g. 'apple+black+rot'). Raw class
names are normalized (lowercase, '+'/'_' -> space, whitespace collapsed)
before matching PLANTWILD_TO_PLANTVILLAGE, defensively, in case the actual
on-disk convention differs slightly.
"""

from __future__ import annotations
import re
from pathlib import Path
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder

from src.data.plantvillage import build_eval_transform, build_train_transform
from src.data.remap import build_class_remap

# PlantWild folder name (normalized) -> (PlantVillage crop, PlantVillage
# disease). Every key must be the normalized form (see `_normalize`) of a raw
# PlantWild class folder; every value must match an entry already present in
# the PlantVillage dataset's crop_classes / disease_classes (checked at load
# time, not just here). PlantWild covers many more species than PlantVillage
# (89 classes total) — everything with no PlantVillage crop/disease
# equivalent (banana, basil, bean, broccoli, cabbage, carrot, cauliflower,
# celery, coffee, cucumber, eggplant, garlic, ginger, lettuce, maple, plum,
# rice, tobacco, zucchini, ...) is intentionally left unmapped and dropped.
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


def _normalize(raw_name: str) -> str:
    normalized = re.sub(r"[+_]", " ", raw_name.strip().lower())
    return re.sub(r"\s+", " ", normalized).strip()


class PlantWildDataset(Dataset):
    """Wraps torchvision's ImageFolder, remapped onto an already-built
    PlantVillage crop/disease label space (so all three datasets can share
    one model head). Exposes both an eval transform and a training
    augmentation transform (see `make_subset`), since PlantWild is a
    first-class node with its own private train/validation split, not just
    extra mixed-in signal.
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
        self.transform = build_train_transform(image_size)
        self.eval_transform = build_eval_transform(image_size)

        raw_class_to_pair = build_class_remap(
            self.base.classes, PLANTWILD_TO_PLANTVILLAGE,
            crop_classes, disease_classes, pv_class_to_crop_disease, "PlantWild",
            normalize=_normalize,
        )

        self._indices = [i for i, (_, raw_cls) in enumerate(self.base.samples) if raw_cls in raw_class_to_pair]
        self._raw_class_to_pair = raw_class_to_pair

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, i: int):
        idx = self._indices[i]
        image, raw_class = self.base[idx]
        image = self.eval_transform(image)
        crop_idx, disease_idx = self._raw_class_to_pair[raw_class]
        return image, crop_idx, disease_idx

    @property
    def pair_labels(self) -> list[tuple[int, int]]:
        """(crop_idx, disease_idx) per retained sample, in logical (0..len-1)
        order — the same shape PlantVillageDataset/PlantDocDataset expose,
        for cross-dataset stratified carving and class weighting.
        """
        return [self._raw_class_to_pair[self.base.samples[idx][1]] for idx in self._indices]


def load_plantwild_dataset(
    root: str | Path | None,
    crop_classes: list[str],
    disease_classes: list[str],
    pv_class_to_crop_disease: dict[int, tuple[int, int]],
    image_size: int = 160,
) -> PlantWildDataset | None:
    """Returns None if `root` is unset or doesn't exist yet — callers that
    need PlantWild as a first-class node (src/data/multi_dataset.py) should
    treat None as a hard error, since a node can't exist with zero data;
    this mirrors load_plantdoc_dataset's opt-out shape only for symmetry.
    """
    if not root:
        return None
    root = Path(root)
    if not root.exists():
        return None
    return PlantWildDataset(root, crop_classes, disease_classes, pv_class_to_crop_disease, image_size=image_size)


class _TransformedSubset(Dataset):
    def __init__(self, dataset: PlantWildDataset, indices: list[int], transform):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        idx = self.dataset._indices[self.indices[i]]
        image, raw_class = self.dataset.base[idx]
        image = self.transform(image)
        crop_idx, disease_idx = self.dataset._raw_class_to_pair[raw_class]
        return image, crop_idx, disease_idx


def make_subset(dataset: PlantWildDataset, indices: list[int], train: bool = False) -> Dataset:
    """train=True applies the augmentation transform (this node's own
    private train split); train=False (default) applies the plain eval
    transform (probe/global-test/validation splits). `indices` are logical
    indices (0..len(dataset)-1), same convention as `dataset[i]`.
    """
    transform = dataset.transform if train else dataset.eval_transform
    return _TransformedSubset(dataset, indices, transform)
