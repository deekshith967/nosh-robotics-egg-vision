import cv2
import numpy as np
import pytest

from eggvision.classical import ClassicalConfig, segment_classical


def test_otsu_finds_synthetic_bright_foreground():
    image = np.zeros((100, 120, 3), np.uint8)
    truth = np.zeros((100, 120), np.uint8)
    cv2.ellipse(image, (60, 50), (24, 16), 0, 0, 360, (230, 230, 230), -1)
    cv2.ellipse(truth, (60, 50), (24, 16), 0, 0, 360, 255, -1)
    pred = segment_classical(image, ClassicalConfig(morph_kernel=3)) > 0
    expected = truth > 0
    intersection = np.logical_and(expected, pred).sum()
    union = np.logical_or(expected, pred).sum()
    assert intersection / union > 0.9


def test_border_distance_handles_color_image():
    image = np.full((80, 80, 3), (30, 40, 60), np.uint8)
    cv2.circle(image, (40, 40), 14, (220, 210, 195), -1)
    mask = segment_classical(image, ClassicalConfig(method="border_distance", morph_kernel=3))
    assert mask.dtype == np.uint8
    assert set(np.unique(mask)).issubset({0, 255})
    assert mask[40, 40] == 255
    assert mask[0, 0] == 0


def test_rejects_empty_input():
    with pytest.raises(ValueError):
        segment_classical(np.array([]))


def test_rejects_unknown_method():
    with pytest.raises(ValueError, match="unknown segmentation method"):
        segment_classical(np.zeros((8, 8, 3), np.uint8), ClassicalConfig(method="missing"))
