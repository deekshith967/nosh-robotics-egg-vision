"""Dataset discovery, mask decoding, and leakage-aware group splitting.

Supported labels: COCO instance polygons/RLE, YOLO segmentation polygons, and
paired binary/semantic PNG masks. Bounding-box-only labels are deliberately not
silently treated as pixel ground truth.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Pattern

import cv2
import numpy as np

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
MASK_DIR_NAMES = {"mask", "masks", "annotations", "annotation", "labels"}
SKIP_DIR_NAMES = {".git", ".venv", "venv", "runs", "outputs", "predictions", "__pycache__"}


@dataclass
class MaskSpec:
    """A lazy, source-aware description of one image's pixel mask."""

    kind: str
    value: Any
    source: str
    class_ids: set[int] | None = None

    def decode(self, height: int, width: int) -> np.ndarray:
        mask = np.zeros((height, width), dtype=np.uint8)
        if self.kind == "coco":
            for segmentation in self.value:
                if isinstance(segmentation, dict):
                    decoded = _decode_coco_rle(segmentation, height, width)
                    mask = np.maximum(mask, decoded)
                    continue
                polygons = segmentation if segmentation and isinstance(segmentation[0], (list, tuple)) else [segmentation]
                for polygon in polygons:
                    coords = np.asarray(polygon, dtype=np.float64).reshape(-1)
                    if coords.size < 6 or coords.size % 2:
                        continue
                    points = np.rint(coords.reshape(-1, 2)).astype(np.int32)
                    points[:, 0] = np.clip(points[:, 0], 0, width - 1)
                    points[:, 1] = np.clip(points[:, 1], 0, height - 1)
                    cv2.fillPoly(mask, [points], 255)
        elif self.kind == "yolo":
            for line in self.value.read_text(encoding="utf-8").splitlines():
                parts = line.strip().split()
                if len(parts) < 7:
                    continue  # A detection box is not a segmentation polygon.
                try:
                    category = int(float(parts[0]))
                    coords = np.asarray([float(v) for v in parts[1:]], dtype=np.float64)
                except ValueError:
                    continue
                if self.class_ids is not None and category not in self.class_ids:
                    continue
                if coords.size < 6 or coords.size % 2:
                    continue
                points = coords.reshape(-1, 2)
                if np.nanmax(np.abs(points)) <= 1.000001:
                    points = points * np.array([width, height], dtype=np.float64)
                points = np.rint(points).astype(np.int32)
                points[:, 0] = np.clip(points[:, 0], 0, width - 1)
                points[:, 1] = np.clip(points[:, 1], 0, height - 1)
                cv2.fillPoly(mask, [points], 255)
        elif self.kind == "png":
            raw = cv2.imread(str(self.value), cv2.IMREAD_UNCHANGED)
            if raw is None:
                raise ValueError(f"could not read mask file: {self.value}")
            if raw.ndim == 3:
                # Any nonzero color is foreground; an alpha channel is ignored.
                raw = np.max(raw[:, :, :3], axis=2)
            if raw.shape[:2] != (height, width):
                raw = cv2.resize(raw, (width, height), interpolation=cv2.INTER_NEAREST)
            mask = np.where(raw > 0, 255, 0).astype(np.uint8)
        else:
            raise ValueError(f"unsupported mask specification: {self.kind}")
        return np.where(mask > 0, 255, 0).astype(np.uint8)


@dataclass(frozen=True)
class BoxPrompt:
    """Annotation bounding box for SAM prompting; never a pixel mask label."""

    bbox_xyxy: tuple[float, float, float, float]
    normalized: bool = False
    source: str = "annotation bounding box"


def _decode_coco_rle(rle: dict[str, Any], height: int, width: int) -> np.ndarray:
    size = rle.get("size", [height, width])
    rh, rw = int(size[0]), int(size[1])
    counts = rle.get("counts")
    if isinstance(counts, list):
        flat = np.zeros(rh * rw, dtype=np.uint8)
        offset, value = 0, 0
        for count in counts:
            count = int(count)
            if count < 0 or offset + count > flat.size:
                raise ValueError("malformed uncompressed COCO RLE counts")
            if value:
                flat[offset : offset + count] = 255
            offset += count
            value = 1 - value
        decoded = flat.reshape((rw, rh)).T
    elif isinstance(counts, (str, bytes)):
        try:
            from pycocotools import mask as mask_utils  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "compressed COCO RLE requires optional pycocotools; install it with `pip install pycocotools`"
            ) from exc
        item = dict(rle)
        if isinstance(item["counts"], str):
            item["counts"] = item["counts"].encode("ascii")
        decoded = mask_utils.decode(item)
    else:
        raise ValueError("COCO RLE annotation has no valid counts")
    if decoded.shape[:2] != (height, width):
        decoded = cv2.resize(decoded.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
    return np.where(decoded > 0, 255, 0).astype(np.uint8)


@dataclass
class DatasetRecord:
    key: str
    image_path: Path
    mask_spec: MaskSpec | None = None
    annotation_note: str | None = None
    prompt_boxes: list[BoxPrompt] = field(default_factory=list)

    def load_image(self) -> np.ndarray:
        image = cv2.imread(str(self.image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"could not read image: {self.image_path}")
        return image

    def load_mask(self, image_shape: tuple[int, ...] | None = None) -> np.ndarray:
        if self.mask_spec is None:
            raise ValueError(f"no pixel segmentation annotation for {self.key}")
        if image_shape is None:
            image = self.load_image()
            height, width = image.shape[:2]
        else:
            height, width = image_shape[:2]
        return self.mask_spec.decode(height, width)

    @property
    def has_mask(self) -> bool:
        return self.mask_spec is not None


@dataclass
class DatasetIndex:
    root: Path
    records: list[DatasetRecord]
    warnings: list[str] = field(default_factory=list)

    @property
    def annotated_records(self) -> list[DatasetRecord]:
        return [record for record in self.records if record.has_mask]


def _safe_images(root: Path) -> list[Path]:
    paths: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        relative_parts = path.relative_to(root).parts
        if any(part.startswith(".") or part.lower() in SKIP_DIR_NAMES for part in relative_parts[:-1]):
            continue
        if any(part.lower() in MASK_DIR_NAMES for part in relative_parts[:-1]):
            continue
        stem = path.stem.lower()
        if stem.endswith(("_mask", "_masks", "_segmentation", "_label", "_labels", "_gt")):
            continue
        paths.append(path)
    return sorted(paths, key=lambda p: p.as_posix().casefold())


def _resolve_coco_image(file_name: str, annotation_dir: Path, root: Path, by_name: dict[str, list[Path]]) -> Path | None:
    supplied = Path(file_name)
    candidates = [
        annotation_dir / supplied,
        annotation_dir / "images" / supplied.name,
        root / supplied,
        root / "images" / supplied.name,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    matches = by_name.get(supplied.name.casefold(), [])
    if len(matches) == 1:
        return matches[0].resolve()
    if len(matches) > 1:
        local = [candidate for candidate in matches if annotation_dir.resolve() in candidate.resolve().parents]
        if len(local) == 1:
            return local[0].resolve()
        # Match the deepest common dataset directory, without guessing between partitions.
        suffix_parts = [part.casefold() for part in supplied.parts]
        path_matches = [
            candidate for candidate in matches
            if len(suffix_parts) > 1 and [p.casefold() for p in candidate.parts[-len(suffix_parts):]] == suffix_parts
        ]
        if len(path_matches) == 1:
            return path_matches[0].resolve()
    return None


def _candidate_mask_files(image_path: Path, root: Path) -> list[Path]:
    stem = image_path.stem
    candidates: list[Path] = []
    if image_path.parent.name.lower() == "images":
        partition = image_path.parent.parent
        candidates.extend([
            partition / "masks" / f"{stem}.png",
            partition / "labels" / f"{stem}.png",
        ])
    candidates.extend([
        image_path.parent / "masks" / f"{stem}.png",
        image_path.parent / f"{stem}_mask.png",
        image_path.parent / f"{stem}_gt.png",
        root / "masks" / f"{stem}.png",
    ])
    return [p for p in candidates if p.is_file()]


def discover_dataset(root: str | Path, class_ids: set[int] | None = None) -> DatasetIndex:
    """Index images and any recognized per-pixel annotations below ``root``.

    COCO category names matching ``egg`` are selected when present. If the COCO
    file has exactly one category it is selected; otherwise all categories are
    foreground unless ``class_ids`` is given. This rule and mask provenance are
    included in the returned records/logs.
    """
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"dataset directory does not exist: {root}")
    image_paths = _safe_images(root)
    if not image_paths:
        raise ValueError(f"no image files found below dataset directory: {root}")

    records_by_path = {
        path.resolve(): DatasetRecord(key=path.relative_to(root).as_posix(), image_path=path.resolve())
        for path in image_paths
    }
    warnings_list: list[str] = []
    by_name: dict[str, list[Path]] = {}
    for path in image_paths:
        by_name.setdefault(path.name.casefold(), []).append(path)

    # COCO annotations have precedence over auto-discovered paired masks.
    for annotation_path in sorted(root.rglob("*.json")):
        if any(part.lower() in SKIP_DIR_NAMES for part in annotation_path.relative_to(root).parts):
            continue
        try:
            payload = json.loads(annotation_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict) or not isinstance(payload.get("images"), list) or not isinstance(payload.get("annotations"), list):
            continue
        categories = payload.get("categories", [])
        category_names = {int(item["id"]): str(item.get("name", "")) for item in categories if "id" in item}
        all_category_ids = set(category_names) or {int(a.get("category_id", 0)) for a in payload["annotations"]}
        egg_ids = {category_id for category_id, name in category_names.items() if re.search(r"\begg(s)?\b", name, re.I)}
        selected_ids = class_ids if class_ids is not None else (egg_ids or (all_category_ids if len(all_category_ids) == 1 else all_category_ids))
        annotations_by_image: dict[Any, list[dict[str, Any]]] = {}
        for annotation in payload["annotations"]:
            if int(annotation.get("category_id", 0)) in selected_ids:
                annotations_by_image.setdefault(annotation.get("image_id"), []).append(annotation)
        for image_info in payload["images"]:
            if not image_info.get("file_name"):
                continue
            image_path = _resolve_coco_image(image_info["file_name"], annotation_path.parent, root, by_name)
            if image_path is None or image_path not in records_by_path:
                continue
            selected_annotations = annotations_by_image.get(image_info.get("id"), [])
            segmentations = [annotation["segmentation"] for annotation in selected_annotations if annotation.get("segmentation")]
            record = records_by_path[image_path]
            if segmentations:
                record.mask_spec = MaskSpec("coco", segmentations, str(annotation_path.relative_to(root)))
            boxes = []
            for annotation in selected_annotations:
                bbox = annotation.get("bbox")
                if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                    x, y, box_width, box_height = [float(value) for value in bbox]
                    if box_width > 0 and box_height > 0:
                        boxes.append(BoxPrompt(
                            (x, y, x + box_width, y + box_height),
                            normalized=False,
                            source=f"COCO bbox: {annotation_path.relative_to(root).as_posix()}",
                        ))
            if boxes:
                record.prompt_boxes = boxes
            if selected_annotations and not segmentations:
                record.annotation_note = "COCO bounding boxes found without pixel segmentation; usable as SAM prompts, not evaluation ground truth"

    # YOLOv8/Roboflow polygon labels: images/<split>/x.jpg -> labels/<split>/x.txt,
    # or <split>/images/x.jpg -> <split>/labels/x.txt.
    for record in records_by_path.values():
        if record.mask_spec is not None:
            continue
        image_path = record.image_path
        candidates: list[Path] = []
        for index, part in enumerate(image_path.parts):
            if part.lower() == "images":
                replacement = list(image_path.parts)
                replacement[index] = "labels"
                candidates.append(Path(*replacement).with_suffix(".txt"))
        candidates.extend([
            image_path.parent / "labels" / f"{image_path.stem}.txt",
            image_path.parent.parent / "labels" / f"{image_path.stem}.txt",
        ])
        label_path = next((candidate for candidate in candidates if candidate.is_file()), None)
        if label_path is not None:
            label_lines = [line.strip().split() for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            selected = set(class_ids) if class_ids is not None else None
            has_polygon = False
            for parts in label_lines:
                if len(parts) < 7:
                    continue
                try:
                    category = int(float(parts[0]))
                except ValueError:
                    continue
                if selected is None or category in selected:
                    has_polygon = True
                    break
            if has_polygon:
                record.mask_spec = MaskSpec("yolo", label_path, label_path.relative_to(root).as_posix(), selected)
                record.annotation_note = f"YOLO polygon segmentation: {label_path.relative_to(root).as_posix()}"
            else:
                boxes: list[BoxPrompt] = []
                for parts in label_lines:
                    if len(parts) != 5:
                        continue
                    try:
                        category = int(float(parts[0]))
                        center_x, center_y, box_width, box_height = [float(value) for value in parts[1:]]
                    except ValueError:
                        continue
                    if selected is not None and category not in selected:
                        continue
                    if not np.all(np.isfinite([center_x, center_y, box_width, box_height])) or box_width <= 0 or box_height <= 0:
                        continue
                    normalized = max(abs(center_x), abs(center_y), abs(box_width), abs(box_height)) <= 1.000001
                    boxes.append(BoxPrompt(
                        (center_x - box_width / 2, center_y - box_height / 2,
                         center_x + box_width / 2, center_y + box_height / 2),
                        normalized=normalized,
                        source=f"YOLO box: {label_path.relative_to(root).as_posix()}",
                    ))
                if boxes and not record.prompt_boxes:
                    record.prompt_boxes = boxes
                if boxes:
                    record.annotation_note = f"YOLO boxes only (SAM prompts, not pixel ground truth): {label_path.relative_to(root).as_posix()}"
                elif not label_lines:
                    record.annotation_note = f"empty YOLO label file: {label_path.relative_to(root).as_posix()}"
                else:
                    record.annotation_note = f"no selected-class YOLO polygon or box prompt: {label_path.relative_to(root).as_posix()}"

    for record in records_by_path.values():
        if record.mask_spec is not None:
            continue
        matches = _candidate_mask_files(record.image_path, root)
        if matches:
            if len(matches) > 1:
                warnings_list.append(f"multiple paired masks for {record.key}; using {matches[0]}")
            record.mask_spec = MaskSpec("png", matches[0], str(matches[0].relative_to(root)))

    annotated_count = sum(record.has_mask for record in records_by_path.values())
    prompt_count = sum(bool(record.prompt_boxes) for record in records_by_path.values())
    if annotated_count == 0:
        if prompt_count:
            warnings_list.append("Bounding-box prompts are available for SAM, but no pixel masks are available for quantitative segmentation evaluation")
        else:
            warnings_list.append("No pixel masks found; only unlabeled-image inference is possible until segmentation annotations are supplied")
    elif annotated_count < len(records_by_path):
        warnings_list.append(
            f"Pixel masks are present for {annotated_count}/{len(records_by_path)} images; unannotated images are excluded from classical metrics but remain available for SAM inference"
        )
    return DatasetIndex(root=root, records=list(records_by_path.values()), warnings=warnings_list)


def _image_content_hash(path: Path) -> str:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        payload = path.read_bytes()
    else:
        payload = f"{image.shape}:{image.dtype}".encode("ascii") + image.tobytes()
    return hashlib.sha256(payload).hexdigest()


def group_records(records: list[DatasetRecord], group_regex: str | None = None) -> dict[str, list[DatasetRecord]]:
    """Group exact decoded-image duplicates and optional sequence-name matches."""
    if not records:
        return {}
    pattern: Pattern[str] | None = re.compile(group_regex) if group_regex else None
    parents = list(range(len(records)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        a, b = find(left), find(right)
        if a != b:
            parents[max(a, b)] = min(a, b)

    content_owner: dict[str, int] = {}
    capture_owner: dict[str, int] = {}
    for index, record in enumerate(records):
        content_hash = _image_content_hash(record.image_path)
        if content_hash in content_owner:
            union(index, content_owner[content_hash])
        else:
            content_owner[content_hash] = index
        if pattern:
            match = pattern.search(record.key)
            if match:
                capture = match.group(1) if match.groups() else match.group(0)
                if capture in capture_owner:
                    union(index, capture_owner[capture])
                else:
                    capture_owner[capture] = index

    grouped: dict[str, list[DatasetRecord]] = {}
    for index, record in enumerate(records):
        group = f"group-{find(index):06d}"
        grouped.setdefault(group, []).append(record)
    return grouped


def split_records(
    records: list[DatasetRecord],
    development_fraction: float = 0.8,
    seed: int = 42,
    group_regex: str | None = None,
) -> tuple[list[DatasetRecord], list[DatasetRecord], dict[str, Any]]:
    """Make a deterministic group-aware split as close as possible to 4:1.

    Exact duplicate image bytes/pixels and optional regex-captured sequence IDs
    are indivisible. A subset-sum pass targets the requested *image* count in
    the evaluation split while never putting any group in both partitions.
    """
    if not 0.0 < development_fraction < 1.0:
        raise ValueError("development_fraction must be strictly between 0 and 1")
    grouped = group_records(records, group_regex)
    if len(grouped) < 2:
        raise ValueError("at least two independent image groups are required for a leakage-safe split")
    target_eval = int(round(len(records) * (1.0 - development_fraction)))
    target_eval = min(max(target_eval, 1), len(records) - 1)
    rng = random.Random(seed)
    group_items = list(grouped.items())
    rng.shuffle(group_items)

    # Reachable sums and predecessor links yield a near-exact holdout without
    # splitting duplicate/sequence groups. For unusually large corpora use a
    # seeded greedy pass to bound memory and runtime.
    if len(group_items) * target_eval <= 4_000_000:
        reachable: dict[int, tuple[int, str] | None] = {0: None}
        for group_id, members in group_items:
            size = len(members)
            for current in sorted(list(reachable), reverse=True):
                new_total = current + size
                if new_total <= len(records) - 1 and new_total not in reachable:
                    reachable[new_total] = (current, group_id)
        candidates = [total for total in reachable if total not in (0, len(records))]
        eval_size = min(candidates, key=lambda total: (abs(total - target_eval), total))
        chosen: set[str] = set()
        total = eval_size
        while total:
            previous, group_id = reachable[total]  # type: ignore[misc]
            chosen.add(group_id)
            total = previous
    else:
        chosen = set()
        eval_size = 0
        for group_id, members in group_items:
            size = len(members)
            if eval_size == 0 or (
                eval_size + size < len(records)
                and abs(eval_size + size - target_eval) < abs(eval_size - target_eval)
            ):
                chosen.add(group_id)
                eval_size += size

    evaluation = [record for group_id, members in grouped.items() if group_id in chosen for record in members]
    development = [record for group_id, members in grouped.items() if group_id not in chosen for record in members]
    if not development or not evaluation:
        raise ValueError("could not produce non-empty development and evaluation partitions")
    development_ids = {id(record) for record in development}
    evaluation_ids = {id(record) for record in evaluation}
    assert development_ids.isdisjoint(evaluation_ids)
    actual = {
        "requested_development_fraction": development_fraction,
        "development_images": len(development),
        "evaluation_images": len(evaluation),
        "actual_development_fraction": len(development) / len(records),
        "actual_evaluation_fraction": len(evaluation) / len(records),
        "group_count": len(grouped),
        "seed": seed,
        "group_regex": group_regex,
        "group_overlap": 0,
        "note": "Group indivisibility can make the image ratio differ slightly from exactly 4:1.",
    }
    return development, evaluation, actual
