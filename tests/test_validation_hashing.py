from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.validation.hashing import average_hash, hamming_distance


def _image_from_gray_array(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(arr, mode="L").convert("RGB")


def test_identical_images_have_zero_hamming_distance():
    rng = np.random.RandomState(0)
    arr = rng.randint(0, 255, size=(64, 64), dtype=np.uint8)
    img1 = _image_from_gray_array(arr)
    img2 = _image_from_gray_array(arr.copy())

    assert hamming_distance(average_hash(img1), average_hash(img2)) == 0


def test_slightly_brightened_image_is_a_near_duplicate():
    rng = np.random.RandomState(0)
    arr = rng.randint(0, 255, size=(64, 64), dtype=np.uint8)
    brightened = np.clip(arr.astype(int) + 10, 0, 255).astype(np.uint8)
    img1 = _image_from_gray_array(arr)
    img2 = _image_from_gray_array(brightened)

    assert hamming_distance(average_hash(img1), average_hash(img2)) <= 5


def test_unrelated_images_have_large_hamming_distance():
    rng1 = np.random.RandomState(1)
    rng2 = np.random.RandomState(2)
    arr1 = rng1.randint(0, 255, size=(64, 64), dtype=np.uint8)
    arr2 = rng2.randint(0, 255, size=(64, 64), dtype=np.uint8)
    img1 = _image_from_gray_array(arr1)
    img2 = _image_from_gray_array(arr2)

    assert hamming_distance(average_hash(img1), average_hash(img2)) > 5


def test_average_hash_is_64_bits_for_default_hash_size():
    rng = np.random.RandomState(0)
    arr = rng.randint(0, 255, size=(64, 64), dtype=np.uint8)
    img = _image_from_gray_array(arr)

    assert 0 <= average_hash(img) < (1 << 64)
