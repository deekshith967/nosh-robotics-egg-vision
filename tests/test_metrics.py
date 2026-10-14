import numpy as np
import pytest

from eggvision.metrics import binary_metrics, error_visualization, summarize_metrics


def test_binary_metrics_perfect_and_explicit_empty_conventions():
    mask = np.array([[0, 255], [255, 0]], dtype=np.uint8)
    assert binary_metrics(mask, mask)["iou"] == 1.0
    empty = np.zeros((3, 3), dtype=np.uint8)
    empty_metrics = binary_metrics(empty, empty)
    assert all(empty_metrics[name] == 1.0 for name in ("iou", "dice", "precision", "recall"))
    same_size_empty = np.zeros(mask.shape, dtype=np.uint8)
    assert binary_metrics(mask, same_size_empty)["recall"] == 0.0


def test_metric_counts_and_macro_pooled_summaries():
    truth = np.array([[1, 1], [0, 0]], dtype=np.uint8)
    prediction = np.array([[1, 0], [1, 0]], dtype=np.uint8)
    result = binary_metrics(truth, prediction)
    assert (result["tp_pixels"], result["fp_pixels"], result["fn_pixels"]) == (1, 1, 1)
    summary = summarize_metrics([result, result])
    assert summary["image_count"] == 2
    assert summary["macro"]["iou"] == pytest.approx(1 / 3)
    assert summary["pooled"]["dice"] == pytest.approx(0.5)


def test_error_visualization_has_four_panels():
    image = np.zeros((12, 20, 3), dtype=np.uint8)
    mask = np.zeros((12, 20), dtype=np.uint8)
    visual = error_visualization(image, mask, mask, "fixture")
    assert visual.shape == (56, 40, 3)
