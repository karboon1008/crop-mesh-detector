"""PlantDoc loading and remapping onto PlantVillage's (crop, disease) label
space, so real-world field images can be mixed into training batches that
would otherwise be pure lab-photographed PlantVillage (see
src/data/mixing.py for the batch sampler that combines the two).
"""

from __future__ import annotations
from pathlib import Path
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder

from src.data.plantvillage import build_eval_transform, build_train_transform
from src.data.remap import build_class_remap

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
    "Blueberry leaf": ("Blueberry", "healthy"),
    "Cherry leaf": ("Cherry", "healthy"),
    "Corn Gray leaf spot": ("Corn", "Cercospora_leaf_spot Gray_leaf_spot"),
    "Corn leaf blight": ("Corn", "Northern_Leaf_Blight"),
    "Corn rust leaf": ("Corn", "Common_rust"),
    "Peach leaf": ("Peach", "healthy"),
    "Potato leaf early blight": ("Potato", "Early_blight"),
    "Potato leaf late blight": ("Potato", "Late_blight"),
    "Raspberry leaf": ("Raspberry", "healthy"),
    "Soyabean leaf": ("Soybean", "healthy"),
    "Squash Powdery mildew leaf": ("Squash", "Powdery_mildew"),
    "Strawberry leaf": ("Strawberry", "healthy"),
    "Tomato Early blight leaf": ("Tomato", "Early_blight"),
    "Tomato Septoria leaf spot": ("Tomato", "Septoria_leaf_spot"),
    "Tomato leaf bacterial spot": ("Tomato", "Bacterial_spot"),
    "Tomato leaf late blight": ("Tomato", "Late_blight"),
    "Tomato leaf mosaic virus": ("Tomato", "Tomato_mosaic_virus"),
    "Tomato leaf yellow virus": ("Tomato", "Tomato_Yellow_Leaf_Curl_Virus"),
    "Tomato leaf": ("Tomato", "healthy"),
    "Tomato mold leaf": ("Tomato", "Leaf_Mold"),
    "Tomato two spotted spider mites leaf": ("Tomato", "Spider_mites Two-spotted_spider_mite"),
    "grape leaf black rot": ("Grape", "Black_rot"),
    "grape leaf": ("Grape", "healthy"),
}


class PlantDocDataset(Dataset):
    """Wraps torchvision's ImageFolder, remapped onto an already-built
    PlantVillage crop/disease label space (so the two datasets can share one
    model head). PlantDoc folders with no PlantVillage crop/disease
    equivalent are dropped with a printed warning — none exist for the
    current 13-species/28-folder release, but the check guards future
    PlantDoc revisions. A mapped (crop, disease) pair that ISN'T in
    `crop_classes`/`disease_classes` raises immediately, since that would
    otherwise silently point at the wrong label index.

    PlantDoc can be used two ways: as extra, domain-shifted training signal
    mixed into another dataset's batches (see src/data/mixing.py), in which
    case __getitem__ always applies the training augmentation transform —
    or as a first-class node of its own (see src/data/multi_dataset.py),
    in which case `make_subset` below builds eval-transformed views for its
    probe/global-test/validation contributions and train-transformed views
    for its own private train split.
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
            self.base.classes, PLANTDOC_TO_PLANTVILLAGE,
            crop_classes, disease_classes, pv_class_to_crop_disease, "PlantDoc",
        )

        self._indices = [i for i, (_, raw_cls) in enumerate(self.base.samples) if raw_cls in raw_class_to_pair]
        self._raw_class_to_pair = raw_class_to_pair

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, i: int):
        idx = self._indices[i]
        image, raw_class = self.base[idx]
        image = self.transform(image)
        crop_idx, disease_idx = self._raw_class_to_pair[raw_class]
        return image, crop_idx, disease_idx

    @property
    def raw_labels(self) -> list[int]:
        """Original ImageFolder class id (pre-PlantVillage-remap) of each
        retained sample — a finer stratum than crop/disease alone, used by
        MixedDomainBatchSampler to class-balance within PlantDoc.
        """
        return [self.base.samples[idx][1] for idx in self._indices]

    @property
    def pair_labels(self) -> list[tuple[int, int]]:
        """(crop_idx, disease_idx) per retained sample, in logical (0..len-1)
        order — the same shape PlantVillageDataset/PlantWildDataset expose,
        for cross-dataset stratified carving and class weighting.
        """
        return [self._raw_class_to_pair[self.base.samples[idx][1]] for idx in self._indices]


def load_plantdoc_dataset(
    root: str | Path | None,
    crop_classes: list[str],
    disease_classes: list[str],
    pv_class_to_crop_disease: dict[int, tuple[int, int]],
    image_size: int = 160,
) -> PlantDocDataset | None:
    """Returns None (domain mixing disabled) if `root` is unset or doesn't
    exist yet, so PlantDoc stays fully opt-in — leaving data.plantdoc_root
    unset in config.yaml reproduces today's PlantVillage-only behaviour.
    """
    if not root:
        return None
    root = Path(root)
    if not root.exists():
        return None
    return PlantDocDataset(root, crop_classes, disease_classes, pv_class_to_crop_disease, image_size=image_size)


class _TransformedSubset(Dataset):
    def __init__(self, dataset: PlantDocDataset, indices: list[int], transform):
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


def make_subset(dataset: PlantDocDataset, indices: list[int], train: bool = False) -> Dataset:
    """train=True applies the augmentation transform (a node's own private
    train split); train=False (default) applies the plain eval transform
    (probe/global-test/validation splits). `indices` are logical indices
    (0..len(dataset)-1), same convention as `dataset[i]`.
    """
    transform = dataset.transform if train else dataset.eval_transform
    return _TransformedSubset(dataset, indices, transform)
