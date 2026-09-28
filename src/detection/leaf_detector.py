"""Stage 1 of the two-stage pipeline: find leaves in a field photo.

PlantVillage photos are one centred leaf on a plain background, which is
what the mesh-trained crop/disease classifiers (src/models/factory.py)
expect. A real field photo has several leaves, soil, stems and sky. This
module finds each leaf with a single-class ("leaf") SSDLite-MobileNetV3
detector and cuts it out, so every crop can be handed to the unchanged
classifier as if it were a PlantVillage image.

The detector starts from torchvision's COCO-pretrained
`ssdlite320_mobilenet_v3_large`. COCO has no leaf class, so only the
backbone and box-regression head are reused; the classification head is
replaced with a 2-class (background, leaf) head and fine-tuned on a
leaf-box dataset (src/detection/train_detector.py).

The detector never takes part in the mesh: "where is a leaf" is not
private farm knowledge, so it is trained once and shipped to every node.
The mesh, its knowledge payloads and their byte counts are unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchvision.models.detection import _utils as det_utils
from torchvision.models.detection.ssdlite import SSDLiteClassificationHead
from torchvision.transforms import functional as TF

LEAF_LABEL = 1  # 0 is background in torchvision detection models
DETECTOR_IMAGE_SIZE = 320  # ssdlite320's native input size


@dataclass
class LeafDetection:
    box: tuple[float, float, float, float]  # x1, y1, x2, y2 in original-image pixels
    score: float


def build_leaf_detector(pretrained: bool = True) -> nn.Module:
    """SSDLite320-MobileNetV3-Large with its 91-class COCO classification
    head swapped for a (background, leaf) head. `pretrained=True` loads the
    COCO detection weights (backbone + box regression); `pretrained=False`
    builds it with random weights and downloads nothing (tests, or loading
    a fine-tuned checkpoint over it).
    """
    if pretrained:
        from torchvision.models.detection import SSDLite320_MobileNet_V3_Large_Weights

        model = ssdlite320_mobilenet_v3_large(weights=SSDLite320_MobileNet_V3_Large_Weights.COCO_V1)
    else:
        model = ssdlite320_mobilenet_v3_large(weights=None, weights_backbone=None)

    size = (DETECTOR_IMAGE_SIZE, DETECTOR_IMAGE_SIZE)
    in_channels = det_utils.retrieve_out_channels(model.backbone, size)
    num_anchors = model.anchor_generator.num_anchors_per_location()
    norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.03)  # same as torchvision's own ssdlite builder
    model.head.classification_head = SSDLiteClassificationHead(in_channels, num_anchors, 2, norm_layer)
    return model


def save_detector(model: nn.Module, path: Path, metadata: dict | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "metadata": metadata or {}}, path)


def load_detector(path: Path, device: str = "cpu") -> nn.Module:
    checkpoint = torch.load(Path(path), map_location=device)
    model = build_leaf_detector(pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval()


class LeafDetector:
    """Inference wrapper: PIL image in, leaf boxes out, plus the square
    crops the classifier consumes.
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = "cpu",
        score_threshold: float = 0.5,
        max_leaves: int = 10,
        min_box_fraction: float = 0.02,
    ):
        self.model = model.to(device).eval()
        self.device = device
        self.score_threshold = score_threshold
        self.max_leaves = max_leaves
        # Boxes whose shorter side is below this share of the image's
        # shorter side are dropped: too small to classify reliably once
        # upscaled to the classifier's input size.
        self.min_box_fraction = min_box_fraction

    @classmethod
    def from_checkpoint(cls, path: Path, device: str = "cpu", **kwargs) -> "LeafDetector":
        return cls(load_detector(path, device), device=device, **kwargs)

    @torch.no_grad()
    def detect(self, image: Image.Image) -> list[LeafDetection]:
        image = image.convert("RGB")
        # torchvision detection models resize + normalise internally and
        # return boxes in the input image's own pixel coordinates.
        tensor = TF.to_tensor(image).to(self.device)
        output = self.model([tensor])[0]
        min_side = self.min_box_fraction * min(image.size)
        detections = []
        for box, score, label in zip(output["boxes"], output["scores"], output["labels"]):
            if label.item() != LEAF_LABEL or score.item() < self.score_threshold:
                continue
            x1, y1, x2, y2 = box.tolist()
            if min(x2 - x1, y2 - y1) < min_side:
                continue
            detections.append(LeafDetection((x1, y1, x2, y2), score.item()))
        detections.sort(key=lambda d: d.score, reverse=True)
        return detections[: self.max_leaves]


def square_crop_box(
    box: tuple[float, float, float, float], image_size: tuple[int, int], pad_fraction: float = 0.1
) -> tuple[int, int, int, int]:
    """Expands a leaf box to a padded square, clamped to the image.

    The classifier resizes its input to a square without keeping the
    aspect ratio (src/predict.py:build_transform), and PlantVillage leaves
    sit in a square frame with a margin around them. A padded square crop
    matches that; a tight rectangular crop would get stretched.
    """
    width, height = image_size
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    side = max(x2 - x1, y2 - y1) * (1 + 2 * pad_fraction)
    side = min(side, width, height)
    left = min(max(cx - side / 2, 0), width - side)
    top = min(max(cy - side / 2, 0), height - side)
    return int(round(left)), int(round(top)), int(round(left + side)), int(round(top + side))


def crop_leaves(
    image: Image.Image, detections: list[LeafDetection], pad_fraction: float = 0.1
) -> list[tuple[Image.Image, LeafDetection | None]]:
    """One (crop, detection) pair per detected leaf. With no detections,
    returns the whole frame paired with None, so a photo that already
    looks like PlantVillage (one leaf filling the frame) still gets a
    prediction instead of nothing.
    """
    image = image.convert("RGB")
    if not detections:
        return [(image, None)]
    return [(image.crop(square_crop_box(d.box, image.size, pad_fraction)), d) for d in detections]
