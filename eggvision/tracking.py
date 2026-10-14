"""Classical foreground detections and persistent-ID centroid tracking."""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .classical import ClassicalConfig, segment_classical


@dataclass(frozen=True)
class Detection:
    bbox_xyxy: tuple[int, int, int, int]
    centroid_xy: tuple[float, float]
    area_pixels: int


@dataclass
class Track:
    track_id: int
    bbox_xyxy: tuple[int, int, int, int]
    centroid_xy: tuple[float, float]
    velocity_xy: tuple[float, float]
    last_frame: int
    age: int = 1
    missed_frames: int = 0


class GreedyCentroidTracker:
    """One-to-one distance/IoU association with constant-velocity prediction.

    This light implementation avoids a heavy tracking dependency and gives each
    track a monotonically increasing ID. It is suitable for modest egg counts;
    crowded scenes, long occlusions, or touching objects need a stronger model.
    """

    def __init__(self, max_distance: float = 90.0, max_age: int = 8, iou_weight: float = 0.35):
        if max_distance <= 0 or max_age < 0 or not 0 <= iou_weight <= 1:
            raise ValueError("invalid tracker settings")
        self.max_distance = float(max_distance)
        self.max_age = int(max_age)
        self.iou_weight = float(iou_weight)
        self.next_id = 1
        self.tracks: dict[int, Track] = {}

    @staticmethod
    def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        iw, ih = max(0, min(ax2, bx2) - max(ax1, bx1)), max(0, min(ay2, by2) - max(ay1, by1))
        intersection = iw * ih
        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
        union = area_a + area_b - intersection
        return intersection / union if union else 0.0

    def update(self, detections: list[Detection], frame_index: int) -> list[dict[str, Any]]:
        active = list(self.tracks.values())
        candidates: list[tuple[float, int, int]] = []
        for track in active:
            dt = max(1, frame_index - track.last_frame)
            predicted = (
                track.centroid_xy[0] + track.velocity_xy[0] * dt,
                track.centroid_xy[1] + track.velocity_xy[1] * dt,
            )
            for detection_index, detection in enumerate(detections):
                distance = math.dist(predicted, detection.centroid_xy)
                overlap = self._iou(track.bbox_xyxy, detection.bbox_xyxy)
                if distance > self.max_distance * min(dt, self.max_age + 1) and overlap < 0.05:
                    continue
                distance_cost = min(1.0, distance / (self.max_distance * min(dt, self.max_age + 1)))
                cost = (1 - self.iou_weight) * distance_cost + self.iou_weight * (1 - overlap)
                candidates.append((cost, track.track_id, detection_index))

        matched_tracks: set[int] = set()
        matched_detections: set[int] = set()
        assignments: dict[int, int] = {}
        for _, track_id, detection_index in sorted(candidates):
            if track_id in matched_tracks or detection_index in matched_detections:
                continue
            matched_tracks.add(track_id)
            matched_detections.add(detection_index)
            assignments[detection_index] = track_id

        for track in active:
            if track.track_id not in matched_tracks:
                track.missed_frames += max(1, frame_index - track.last_frame)

        output: list[dict[str, Any]] = []
        for detection_index, detection in enumerate(detections):
            if detection_index in assignments:
                track = self.tracks[assignments[detection_index]]
                dt = max(1, frame_index - track.last_frame)
                measured_velocity = (
                    (detection.centroid_xy[0] - track.centroid_xy[0]) / dt,
                    (detection.centroid_xy[1] - track.centroid_xy[1]) / dt,
                )
                track.velocity_xy = (
                    0.65 * track.velocity_xy[0] + 0.35 * measured_velocity[0],
                    0.65 * track.velocity_xy[1] + 0.35 * measured_velocity[1],
                )
                track.bbox_xyxy = detection.bbox_xyxy
                track.centroid_xy = detection.centroid_xy
                track.last_frame = frame_index
                track.age += dt
                track.missed_frames = 0
            else:
                track = Track(
                    track_id=self.next_id,
                    bbox_xyxy=detection.bbox_xyxy,
                    centroid_xy=detection.centroid_xy,
                    velocity_xy=(0.0, 0.0),
                    last_frame=frame_index,
                )
                self.tracks[self.next_id] = track
                self.next_id += 1
            output.append({
                "track_id": track.track_id,
                "bbox_xyxy": list(detection.bbox_xyxy),
                "centroid_xy": [round(detection.centroid_xy[0], 2), round(detection.centroid_xy[1], 2)],
                "area_pixels": detection.area_pixels,
                "track_age_frames": track.age,
            })

        for track_id in [key for key, value in self.tracks.items() if value.missed_frames > self.max_age]:
            del self.tracks[track_id]
        return output


def detect_foreground_objects(
    frame_bgr: np.ndarray,
    config: ClassicalConfig,
    min_area_fraction: float = 0.0006,
    max_area_fraction: float = 0.30,
    min_solidity: float = 0.45,
) -> list[Detection]:
    """Extract connected components from a classical segmentation mask."""
    if not 0 <= min_area_fraction < max_area_fraction <= 1:
        raise ValueError("expected 0 <= min_area_fraction < max_area_fraction <= 1")
    mask = segment_classical(frame_bgr, config)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_area = frame_bgr.shape[0] * frame_bgr.shape[1]
    detections: list[Detection] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < max(4.0, image_area * min_area_fraction) or area > image_area * max_area_fraction:
            continue
        x, y, width, height = cv2.boundingRect(contour)
        if width < 3 or height < 3:
            continue
        aspect_ratio = width / max(1, height)
        if not 0.25 <= aspect_ratio <= 4.0:
            continue
        hull_area = float(cv2.contourArea(cv2.convexHull(contour)))
        solidity = area / hull_area if hull_area else 0.0
        if solidity < min_solidity:
            continue
        moments = cv2.moments(contour)
        if moments["m00"]:
            centroid = (moments["m10"] / moments["m00"], moments["m01"] / moments["m00"])
        else:
            centroid = (x + width / 2.0, y + height / 2.0)
        detections.append(Detection((x, y, x + width, y + height), centroid, int(round(area))))
    return sorted(detections, key=lambda item: (item.centroid_xy[0], item.centroid_xy[1]))


def _read_classical_config(path: str | Path | None) -> ClassicalConfig:
    if path is None:
        return ClassicalConfig()
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    config = payload.get("segmentation_config", payload.get("selected_config", payload))
    allowed = set(ClassicalConfig.__dataclass_fields__)
    return ClassicalConfig(**{key: value for key, value in config.items() if key in allowed})


def _track_color(track_id: int) -> tuple[int, int, int]:
    hue = (track_id * 37) % 180
    hsv = np.uint8([[[hue, 210, 255]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return tuple(int(v) for v in bgr)


def track_video(
    video_path: str | Path,
    output_path: str | Path,
    segmentation_config_path: str | Path | None = None,
    max_distance: float = 90.0,
    max_age: int = 8,
    min_area_fraction: float = 0.0006,
    max_area_fraction: float = 0.30,
) -> dict[str, Any]:
    """Track per-frame classical detections, writing MP4 and JSON sidecar."""
    video_path = Path(video_path).expanduser()
    output_path = Path(output_path).expanduser()
    if not video_path.is_file():
        raise FileNotFoundError(
            f"local video not found: {video_path}. Download the supplied video to a local file first; remote fetching is not performed."
        )
    if video_path.resolve() == output_path.resolve():
        raise ValueError("output path must not overwrite the input video")
    config = _read_classical_config(segmentation_config_path)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"OpenCV could not open video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    fps_fallback = not np.isfinite(fps) or fps <= 0
    if fps_fallback:
        fps = 25.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        capture.release()
        raise ValueError("video reports invalid frame dimensions")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError("could not open MP4 writer; install an FFmpeg-enabled OpenCV build")

    tracker = GreedyCentroidTracker(max_distance=max_distance, max_age=max_age)
    frame_records: list[dict[str, Any]] = []
    frame_index = 0
    started = time.time()
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame.shape[1] != width or frame.shape[0] != height:
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            detections = detect_foreground_objects(
                frame, config,
                min_area_fraction=min_area_fraction,
                max_area_fraction=max_area_fraction,
            )
            tracked = tracker.update(detections, frame_index)
            annotated = frame.copy()
            for item in tracked:
                x1, y1, x2, y2 = item["bbox_xyxy"]
                color = _track_color(item["track_id"])
                cv2.rectangle(annotated, (x1, y1), (max(x1, x2 - 1), max(y1, y2 - 1)), color, 2)
                label = f"Egg {item['track_id']}"
                cv2.putText(annotated, label, (x1, max(18, y1 - 7)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2, cv2.LINE_AA)
                cx, cy = [int(round(v)) for v in item["centroid_xy"]]
                cv2.circle(annotated, (cx, cy), 3, color, -1, cv2.LINE_AA)
            cv2.putText(annotated, f"Frame {frame_index} | tracks {len(tracked)}", (12, height - 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
            writer.write(annotated)
            frame_records.append({"frame": frame_index, "detections": tracked})
            frame_index += 1
    finally:
        capture.release()
        writer.release()

    if frame_index == 0:
        raise ValueError("video contained no decodable frames")
    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise RuntimeError("annotated video file was not produced")
    summary = {
        "status": "completed",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": round(time.time() - started, 3),
        "input_filename": video_path.name,
        "output_filename": output_path.name,
        "output_bytes": output_path.stat().st_size,
        "fps": fps,
        "fps_fallback_used": fps_fallback,
        "frame_width": width,
        "frame_height": height,
        "frames_processed": frame_index,
        "max_persistent_track_id": tracker.next_id - 1,
        "detections_total": sum(len(item["detections"]) for item in frame_records),
        "tracker": {"type": "greedy centroid + constant-velocity prediction + bbox IoU", "max_distance_px": max_distance, "max_age_frames": max_age},
        "segmentation_config": asdict(config),
        "limitations": "IDs can switch during crossings/occlusion; touching eggs may merge in the classical mask. Validate and tune against the actual video.",
        "frames": frame_records,
    }
    sidecar = output_path.with_suffix(".json")
    sidecar.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary
