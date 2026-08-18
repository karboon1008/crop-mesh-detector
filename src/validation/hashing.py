"""64-bit average-hash perceptual hashing, used to group near-duplicate
PlantVillage images before splitting train/test (see node1_dataset.py) —
PlantVillage is known to contain near-identical shots of the same physical
leaf, which a plain random split would happily place on both sides.
"""

from __future__ import annotations

import numpy as np
from PIL import Image


def average_hash(image: Image.Image, hash_size: int = 8) -> int:
    gray = image.convert("L").resize((hash_size, hash_size), Image.BILINEAR)
    pixels = np.asarray(gray).flatten()
    mean = pixels.mean()
    bits = 0
    for i, pixel in enumerate(pixels):
        if pixel > mean:
            bits |= 1 << i
    return int(bits)


def hamming_distance(a: int, b: int) -> int:
    return bin(a ^ b).count("1")
