"""Shared micrograph read helpers for detect, star export, and GUI preview."""

from __future__ import annotations

import os
from typing import Tuple

import cv2
import mrcfile
import numpy as np

from partinet.process_utils.guided_denoiser import transform

_RASTER_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


def _mrc_array_shape(data: np.ndarray) -> Tuple[int, int]:
    """Return (width, height) from a 2D or (1, H, W) MRC array."""
    if data.ndim == 2:
        h, w = data.shape
    elif data.ndim == 3 and data.shape[0] == 1:
        _, h, w = data.shape
    else:
        raise ValueError(f"Unsupported MRC array shape: {data.shape}")
    return int(w), int(h)


def micrograph_dimensions(path: str) -> Tuple[int, int]:
    """Return pixel (width, height) for STAR coordinate scaling."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".mrc":
        with mrcfile.open(path, permissive=True) as mrc:
            return _mrc_array_shape(mrc.data)
    from PIL import Image as PILImage
    with PILImage.open(path) as img:
        return img.size


def load_micrograph_for_detect(path: str) -> np.ndarray:
    """Load a micrograph as BGR uint8 using the same rules as detect LoadImages."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".mrc":
        img_mrc = mrcfile.read(path)
        img_mrc = np.asarray(img_mrc, dtype=np.float32)
        img0 = transform(img_mrc).astype(np.uint8)
        if img0.ndim == 2:
            img0 = cv2.cvtColor(img0, cv2.COLOR_GRAY2BGR)
        return img0
    img0 = cv2.imread(path)
    if img0 is None:
        raise ValueError(f"Could not read image: {path}")
    return img0


def is_micrograph_file(filename: str) -> bool:
    ext = os.path.splitext(filename)[1].lower()
    return ext in _RASTER_EXTS or ext == ".mrc"
