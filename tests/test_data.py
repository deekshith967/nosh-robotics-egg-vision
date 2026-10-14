import json

import cv2
import numpy as np
import pytest

from eggvision.data import discover_dataset, split_records
from eggvision.sam_inference import _make_prompt_boxes


def write_png(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), image)


def test_yolo_polygon_mask_is_decoded(tmp_path):
    image_path = tmp_path / "train" / "images" / "egg.png"
    write_png(image_path, np.zeros((100, 120, 3), np.uint8))
    label_path = tmp_path / "train" / "labels" / "egg.txt"
    label_path.parent.mkdir(parents=True)
    label_path.write_text("0 0.25 0.25 0.75 0.25 0.75 0.75 0.25 0.75\n", encoding="utf-8")

    index = discover_dataset(tmp_path)
    assert len(index.records) == 1
    record = index.records[0]
    assert record.has_mask
    mask = record.load_mask((100, 120, 3))
    assert mask[50, 60] == 255
    assert mask[2, 2] == 0
    assert 3000 < int((mask > 0).sum()) < 4000


def test_yolo_detection_box_is_prompt_only_not_pixel_ground_truth(tmp_path):
    image_path = tmp_path / "train" / "images" / "egg.png"
    write_png(image_path, np.zeros((100, 120, 3), np.uint8))
    label_path = tmp_path / "train" / "labels" / "egg.txt"
    label_path.parent.mkdir(parents=True)
    label_path.write_text("0 0.5 0.5 0.5 0.6\n", encoding="utf-8")

    record = discover_dataset(tmp_path).records[0]
    assert not record.has_mask
    assert len(record.prompt_boxes) == 1
    box = record.prompt_boxes[0]
    assert box.normalized is True
    assert box.bbox_xyxy == pytest.approx((0.25, 0.2, 0.75, 0.8))
    sam_prompts = _make_prompt_boxes(record, (100, 120, 3))
    assert sam_prompts[0][0] == [30, 20, 90, 80]
    with pytest.raises(ValueError, match="no pixel segmentation"):
        record.load_mask()


def test_coco_bbox_only_is_prompt_not_pixel_ground_truth(tmp_path):
    write_png(tmp_path / "images" / "egg.png", np.zeros((64, 80, 3), np.uint8))
    payload = {
        "images": [{"id": 1, "file_name": "images/egg.png", "height": 64, "width": 80}],
        "categories": [{"id": 3, "name": "egg"}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 3, "bbox": [10, 12, 30, 20]}],
    }
    (tmp_path / "instances.json").write_text(json.dumps(payload), encoding="utf-8")

    record = discover_dataset(tmp_path).records[0]
    assert not record.has_mask
    assert record.prompt_boxes[0].bbox_xyxy == (10.0, 12.0, 40.0, 32.0)
    assert not record.prompt_boxes[0].normalized
    assert "not evaluation ground truth" in record.annotation_note
    assert _make_prompt_boxes(record, (64, 80, 3))[0][0] == [10, 12, 40, 32]


def test_coco_polygon_and_egg_category_are_decoded(tmp_path):
    image_path = tmp_path / "images" / "egg.png"
    write_png(image_path, np.zeros((64, 80, 3), np.uint8))
    annotations = {
        "images": [{"id": 1, "file_name": "images/egg.png", "height": 64, "width": 80}],
        "categories": [{"id": 3, "name": "egg"}, {"id": 4, "name": "background"}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 3, "segmentation": [[10, 10, 50, 10, 50, 40, 10, 40]], "bbox": [10, 10, 40, 30]},
            {"id": 2, "image_id": 1, "category_id": 4, "segmentation": [[55, 10, 70, 10, 70, 20, 55, 20]]},
        ],
    }
    annotation_dir = tmp_path / "annotations"
    annotation_dir.mkdir()
    (annotation_dir / "instances.json").write_text(json.dumps(annotations), encoding="utf-8")
    record = discover_dataset(tmp_path).records[0]
    mask = record.load_mask((64, 80, 3))
    assert mask[20, 20] == 255
    assert mask[15, 60] == 0
    assert len(record.prompt_boxes) == 1


def test_paired_mask_png_is_found(tmp_path):
    write_png(tmp_path / "egg.jpg", np.zeros((40, 50, 3), np.uint8))
    mask = np.zeros((40, 50), np.uint8)
    cv2.circle(mask, (25, 20), 8, 255, -1)
    write_png(tmp_path / "masks" / "egg.png", mask)
    record = discover_dataset(tmp_path).records[0]
    assert record.has_mask
    assert record.load_mask()[20, 25] == 255


def test_split_is_deterministic_and_capture_groups_never_cross(tmp_path):
    for video, x_offset in (("clipA", 10), ("clipB", 100)):
        for frame in range(3):
            image = np.zeros((32, 40, 3), np.uint8)
            cv2.circle(image, (x_offset + frame, 16), 3, (255, 255, 255), -1)
            write_png(tmp_path / f"{video}_frame{frame}.png", image)
    records = discover_dataset(tmp_path).records
    dev1, eval1, info1 = split_records(records, group_regex=r"(clip[A-Z])", seed=42)
    dev2, eval2, info2 = split_records(records, group_regex=r"(clip[A-Z])", seed=42)
    assert [r.key for r in dev1] == [r.key for r in dev2]
    assert [r.key for r in eval1] == [r.key for r in eval2]
    assert info1 == info2
    dev_groups = {r.key.split("_")[0] for r in dev1}
    eval_groups = {r.key.split("_")[0] for r in eval1}
    assert dev_groups.isdisjoint(eval_groups)
    # Entire 3-frame captures are indivisible, so a 50/50 split is the closest
    # possible outcome for two capture groups, despite the requested 80/20 ratio.
    assert info1["actual_evaluation_fraction"] == 0.5


def test_exact_duplicates_are_not_split(tmp_path):
    first = np.zeros((24, 24, 3), np.uint8)
    cv2.circle(first, (12, 12), 5, (210, 210, 210), -1)
    other = np.full((24, 24, 3), 42, np.uint8)
    write_png(tmp_path / "copy-a.png", first)
    write_png(tmp_path / "copy-b.png", first.copy())
    write_png(tmp_path / "different.png", other)
    records = discover_dataset(tmp_path).records
    dev, evaluation, _ = split_records(records, seed=2)
    partition = {record.key: "dev" for record in dev}
    partition.update({record.key: "eval" for record in evaluation})
    assert partition["copy-a.png"] == partition["copy-b.png"]


def test_single_group_cannot_be_leakage_safely_split(tmp_path):
    image = np.zeros((20, 20, 3), np.uint8)
    write_png(tmp_path / "one.png", image)
    write_png(tmp_path / "same.png", image.copy())
    records = discover_dataset(tmp_path).records
    with pytest.raises(ValueError, match="at least two independent"):
        split_records(records)
