"""Small, deterministic classical segmentation baseline (OpenCV)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class ClassicalConfig:
    """Parameters for the color-distance segmentation candidate."""

    method: str = "otsu"
    polarity: str = "bright"
    blur_kernel: int = 5
    morph_kernel: int = 5
    morph_open_iterations: int = 1
    morph_close_iterations: int = 2
    border_fraction: float = 0.04
    color_distance_threshold: float = 28.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _odd_kernel(value: int) -> int:
    value = max(1, int(value))
    return value if value % 2 else value + 1


def segment_classical(image: np.ndarray, config: ClassicalConfig | None = None) -> np.ndarray:
    """Segment likely foreground and return a 0/255 uint8 mask.

    ``otsu`` is the baseline: grayscale Otsu thresholding with explicit bright
    or dark foreground polarity. ``border_distance`` estimates the background
    color from the image border and thresholds per-pixel Lab distance. Both
    candidates receive the same light morphology to suppress speckle and close
    small gaps. This is a general-purpose baseline, not an egg detector; select
    the method/parameters using only the development partition.
    """
    cfg = config or ClassicalConfig()
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError("image must be a non-empty NumPy array")
    if image.ndim == 2:
        bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.ndim == 3 and image.shape[2] == 3:
        bgr = image
    elif image.ndim == 3 and image.shape[2] == 4:
        bgr = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    else:
        raise ValueError(f"expected grayscale, BGR, or BGRA image; got shape {image.shape}")

    blur = _odd_kernel(cfg.blur_kernel)
    smooth = cv2.GaussianBlur(bgr, (blur, blur), 0) if blur > 1 else bgr
    method = cfg.method.lower()
    if method == "otsu":
        gray = cv2.cvtColor(smooth, cv2.COLOR_BGR2GRAY)
        threshold_type = cv2.THRESH_BINARY if cfg.polarity == "bright" else cv2.THRESH_BINARY_INV
        if cfg.polarity not in {"bright", "dark"}:
            raise ValueError("otsu polarity must be 'bright' or 'dark'")
        _, raw = cv2.threshold(gray, 0, 255, threshold_type | cv2.THRESH_OTSU)
    elif method == "border_distance":
        lab = cv2.cvtColor(smooth, cv2.COLOR_BGR2LAB).astype(np.float32)
        h, w = lab.shape[:2]
        band = max(1, int(round(min(h, w) * cfg.border_fraction)))
        border = np.concatenate(
            (lab[:band].reshape(-1, 3), lab[-band:].reshape(-1, 3),
             lab[:, :band].reshape(-1, 3), lab[:, -band:].reshape(-1, 3)), axis=0
        )
        background = np.median(border, axis=0)
        distance = np.linalg.norm(lab - background[None, None, :], axis=2)
        raw = (distance >= float(cfg.color_distance_threshold)).astype(np.uint8) * 255
    else:
        raise ValueError(f"unknown segmentation method: {cfg.method!r}")

    kernel_size = _odd_kernel(cfg.morph_kernel)
    if kernel_size > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        if cfg.morph_open_iterations > 0:
            raw = cv2.morphologyEx(raw, cv2.MORPH_OPEN, kernel, iterations=cfg.morph_open_iterations)
        if cfg.morph_close_iterations > 0:
            raw = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, kernel, iterations=cfg.morph_close_iterations)
    return np.where(raw > 0, 255, 0).astype(np.uint8)
