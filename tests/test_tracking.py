import json

import cv2
import numpy as np
import pytest

from eggvision.classical import ClassicalConfig
from eggvision.tracking import Detection, GreedyCentroidTracker, detect_foreground_objects, track_video


def det(x, y=20):
    return Detection((x - 8, y - 8, x + 8, y + 8), (float(x), float(y)), 180)


def test_tracker_assigns_persistent_ids_and_survives_short_occlusion():
    tracker = GreedyCentroidTracker(max_distance=30, max_age=3)
    first = tracker.update([det(20), det(80)], 0)
    assert [item["track_id"] for item in first] == [1, 2]
    moved = tracker.update([det(24), det(76)], 1)
    assert [item["track_id"] for item in moved] == [1, 2]
    tracker.update([], 2)
    returned = tracker.update([det(32), det(68)], 3)
    assert [item["track_id"] for item in returned] == [1, 2]


def test_classical_contours_find_simple_foreground():
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    cv2.ellipse(image, (70, 60), (23, 15), 15, 0, 360, (240, 240, 240), -1)
    detections = detect_foreground_objects(
        image, ClassicalConfig(morph_kernel=3), min_area_fraction=0.001, max_area_fraction=0.2
    )
    assert len(detections) == 1
    assert abs(detections[0].centroid_xy[0] - 70) < 2


def test_synthetic_video_to_annotated_mp4(tmp_path):
    source = tmp_path / "synthetic_input.mp4"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 8.0, (160, 120))
    if not writer.isOpened():
        pytest.skip("OpenCV MP4 writer unavailable in this environment")
    for frame_index in range(8):
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        cv2.circle(frame, (35 + frame_index * 3, 38), 13, (235, 235, 235), -1)
        cv2.circle(frame, (115 - frame_index * 2, 80), 12, (210, 210, 210), -1)
        writer.write(frame)
    writer.release()

    output = tmp_path / "annotated.mp4"
    result = track_video(source, output, max_distance=30, min_area_fraction=0.001, max_area_fraction=0.2)
    assert result["status"] == "completed"
    assert result["frames_processed"] == 8
    assert output.is_file() and output.stat().st_size > 0
    sidecar = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
    assert sidecar["max_persistent_track_id"] >= 2
    assert sidecar["frames"][0]["detections"]
