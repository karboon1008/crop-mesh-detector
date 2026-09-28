"""Leaf-box dataset for fine-tuning the stage-1 leaf detector.

Reads Pascal VOC annotations: a folder of images, each with a sibling
`.xml` of the same stem (the layout of the PlantDoc object-detection
dataset's TRAIN/ and TEST/ folders, and what LabelImg / CVAT export).
Every annotated object becomes class 1 ("leaf") whatever its name — the
detector only has to find leaves; crop type and disease are the mesh
classifier's job.

Some public VOC exports record a <size> that doesn't match the image file
on disk (PlantDoc has several). Boxes are rescaled from the annotated
size to the real one so they still line up.
"""

from __future__ import annotations

import random
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

from src.detection.leaf_detector import LEAF_LABEL

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


@dataclass
class AnnotatedImage:
    image_path: Path
    boxes: list[tuple[float, float, float, float]]  # x1, y1, x2, y2, already in on-disk pixel coordinates


def parse_voc_xml(xml_path: Path, image_size: tuple[int, int]) -> list[tuple[float, float, float, float]]:
    """Boxes from one VOC file, rescaled to `image_size` (width, height)
    when the file's <size> disagrees, clamped to the image and with
    degenerate (zero-area) boxes dropped.
    """
    root = ET.parse(xml_path).getroot()
    width, height = image_size
    scale_x = scale_y = 1.0
    size = root.find("size")
    if size is not None:
        ann_w = float(size.findtext("width", "0") or 0)
        ann_h = float(size.findtext("height", "0") or 0)
        if ann_w > 0 and ann_h > 0:
            scale_x, scale_y = width / ann_w, height / ann_h

    boxes = []
    for obj in root.iter("object"):
        bnd = obj.find("bndbox")
        if bnd is None:
            continue
        x1 = float(bnd.findtext("xmin")) * scale_x
        y1 = float(bnd.findtext("ymin")) * scale_y
        x2 = float(bnd.findtext("xmax")) * scale_x
        y2 = float(bnd.findtext("ymax")) * scale_y
        x1, x2 = sorted((min(max(x1, 0.0), width), min(max(x2, 0.0), width)))
        y1, y2 = sorted((min(max(y1, 0.0), height), min(max(y2, 0.0), height)))
        if x2 - x1 >= 1 and y2 - y1 >= 1:
            boxes.append((x1, y1, x2, y2))
    return boxes


def load_voc_folder(folder: Path) -> list[AnnotatedImage]:
    """Every image in `folder` that has a sibling .xml with at least one
    usable box. Images without annotations are skipped, not treated as
    "no leaves": in these datasets a missing .xml means unlabelled.
    """
    folder = Path(folder)
    items = []
    for image_path in sorted(folder.iterdir()):
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        xml_path = image_path.with_suffix(".xml")
        if not xml_path.exists():
            continue
        with Image.open(image_path) as img:
            size = img.size
        boxes = parse_voc_xml(xml_path, size)
        if boxes:
            items.append(AnnotatedImage(image_path, boxes))
    if not items:
        raise FileNotFoundError(f"No images with VOC .xml annotations found in {folder}")
    return items


class LeafBoxDataset(Dataset):
    """Yields (image tensor in [0, 1], target dict) in torchvision's
    detection format. Normalisation and resizing to 320x320 happen inside
    the SSDLite model itself, so only a horizontal flip is applied here
    (the one augmentation that is trivially box-safe).
    """

    def __init__(self, items: list[AnnotatedImage], train: bool = False, flip_prob: float = 0.5):
        self.items = items
        self.train = train
        self.flip_prob = flip_prob

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        item = self.items[idx]
        image = Image.open(item.image_path).convert("RGB")
        boxes = torch.tensor(item.boxes, dtype=torch.float32)
        if self.train and random.random() < self.flip_prob:
            image = TF.hflip(image)
            width = image.size[0]
            boxes = torch.stack([width - boxes[:, 2], boxes[:, 1], width - boxes[:, 0], boxes[:, 3]], dim=1)
        target = {"boxes": boxes, "labels": torch.full((len(boxes),), LEAF_LABEL, dtype=torch.int64)}
        return TF.to_tensor(image), target


def detection_collate(batch):
    """Detection models take a list of variable-size images, not a stacked batch."""
    images, targets = zip(*batch)
    return list(images), list(targets)
