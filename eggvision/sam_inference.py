"""Optional Segment Anything Model (SAM) inference for dataset images.

SAM's masks are proposals/pseudo-labels, not verified annotations. Annotation-
prompt mode uses provided boxes (or derives boxes from provided masks); automatic
mode saves model proposals without asserting semantic egg identity.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .data import DatasetRecord, discover_dataset


def _make_slug(key: str) -> str:
    return key.replace("/", "__").replace("\\", "__")


def _boxes_from_mask(mask: np.ndarray) -> list[list[int]]:
    binary = (mask > 0).astype(np.uint8)
    count, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    boxes: list[list[int]] = []
    for label in range(1, count):
        x, y, width, height, area = [int(v) for v in stats[label]]
        if area < 4 or width < 2 or height < 2:
            continue
        boxes.append([x, y, x + width, y + height])
    return boxes


def _make_prompt_boxes(record: DatasetRecord, image_shape: tuple[int, ...]) -> list[tuple[list[int], str]]:
    height, width = image_shape[:2]
    prompts: list[tuple[list[int], str]] = []
    if record.prompt_boxes:
        for prompt in record.prompt_boxes:
            x1, y1, x2, y2 = prompt.bbox_xyxy
            if prompt.normalized:
                x1, x2 = x1 * width, x2 * width
                y1, y2 = y1 * height, y2 * height
            box = [
                int(np.clip(round(x1), 0, width - 1)),
                int(np.clip(round(y1), 0, height - 1)),
                int(np.clip(round(x2), 1, width)),
                int(np.clip(round(y2), 1, height)),
            ]
            if box[2] > box[0] and box[3] > box[1]:
                prompts.append((box, prompt.source))
    elif record.has_mask:
        image_bgr = record.load_image()
        ground_truth = record.load_mask(image_bgr.shape)
        prompts.extend((box, "bounding box derived from provided pixel mask") for box in _boxes_from_mask(ground_truth))
    return prompts


def _sam_model(checkpoint: Path, model_type: str, device: str):
    try:
        import torch
        from segment_anything import sam_model_registry
    except ImportError as exc:
        raise RuntimeError(
            "SAM inference needs PyTorch and the official segment-anything package. "
            "See requirements-sam.txt and README setup instructions."
        ) from exc
    chosen_device = device
    if chosen_device == "auto":
        chosen_device = "cuda" if torch.cuda.is_available() else "cpu"
    if chosen_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"requested device {chosen_device}, but CUDA is unavailable")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"SAM checkpoint not found: {checkpoint}")
    if model_type not in sam_model_registry:
        raise ValueError(f"unknown SAM model type {model_type!r}; choose {sorted(sam_model_registry)}")
    model = sam_model_registry[model_type](checkpoint=str(checkpoint))
    model.to(device=chosen_device)
    model.eval()
    return torch, model, chosen_device


def _annotation_box_masks(
    predictor: Any,
    image_rgb: np.ndarray,
    record: DatasetRecord,
) -> tuple[list[dict[str, Any]], str]:
    prompts = _make_prompt_boxes(record, image_rgb.shape)
    if not prompts:
        return [], "no_annotation_boxes"
    predictor.set_image(image_rgb)
    instances: list[dict[str, Any]] = []
    for instance_index, (box, prompt_source) in enumerate(prompts):
        masks, scores, _ = predictor.predict(
            box=np.asarray(box, dtype=np.float32),
            multimask_output=True,
        )
        best = int(np.argmax(scores))
        mask = (masks[best] > 0).astype(np.uint8) * 255
        instances.append({
            "mask": mask,
            "bbox_xyxy": [int(v) for v in box],
            "sam_score": float(scores[best]),
            "instance_index": instance_index,
            "prompt": f"{prompt_source}; SAM result is a pseudo-label, not verified ground truth",
        })
    return instances, "annotation_box_prompts"


def _automatic_masks(generator: Any, image_rgb: np.ndarray) -> list[dict[str, Any]]:
    proposals = generator.generate(image_rgb)
    instances: list[dict[str, Any]] = []
    for index, item in enumerate(proposals):
        mask = np.asarray(item["segmentation"], dtype=np.uint8) * 255
        bbox = item.get("bbox", [0, 0, 0, 0])  # AMG order is x, y, width, height.
        instances.append({
            "mask": mask,
            "bbox_xyxy": [int(bbox[0]), int(bbox[1]), int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])],
            "sam_score": float(item.get("predicted_iou", 0.0)),
            "stability_score": float(item.get("stability_score", 0.0)),
            "instance_index": index,
            "prompt": "automatic SAM proposal; semantic egg identity is not guaranteed",
        })
    return instances


def run_sam_inference(
    dataset_root: str | Path,
    checkpoint: str | Path,
    output_dir: str | Path,
    model_type: str = "vit_b",
    prompt_mode: str = "annotations",
    device: str = "auto",
) -> dict[str, Any]:
    """Run SAM over every discovered dataset image and save masks/overlays."""
    if prompt_mode not in {"annotations", "automatic"}:
        raise ValueError("prompt_mode must be 'annotations' or 'automatic'")
    started = time.time()
    index = discover_dataset(dataset_root)
    checkpoint = Path(checkpoint).expanduser()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _, model, chosen_device = _sam_model(checkpoint, model_type, device)
    from segment_anything import SamAutomaticMaskGenerator, SamPredictor

    predictor = SamPredictor(model)
    generator = SamAutomaticMaskGenerator(model) if prompt_mode == "automatic" else None
    image_outputs: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    mask_root = output_dir / "masks"
    overlay_root = output_dir / "overlays"
    for record in index.records:
        try:
            bgr = record.load_image()
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            if prompt_mode == "annotations" and (record.prompt_boxes or record.has_mask):
                instances, mode_used = _annotation_box_masks(predictor, rgb, record)
                if not instances:
                    if generator is None:
                        generator = SamAutomaticMaskGenerator(model)
                    instances = _automatic_masks(generator, rgb)
                    mode_used = "automatic_fallback_no_annotation_components"
            else:
                if generator is None:
                    generator = SamAutomaticMaskGenerator(model)
                instances = _automatic_masks(generator, rgb)
                mode_used = "automatic_proposals"

            slug = _make_slug(record.key)
            image_mask_dir = mask_root / slug
            image_mask_dir.mkdir(parents=True, exist_ok=True)
            union = np.zeros(bgr.shape[:2], dtype=np.uint8)
            serialized_instances: list[dict[str, Any]] = []
            for instance in instances:
                instance_mask = instance.pop("mask")
                union = np.maximum(union, instance_mask)
                mask_name = f"instance_{int(instance['instance_index']):04d}.png"
                if not cv2.imwrite(str(image_mask_dir / mask_name), instance_mask):
                    raise OSError(f"failed writing SAM mask {image_mask_dir / mask_name}")
                serialized_instances.append({
                    **instance,
                    "mask_file": f"masks/{slug}/{mask_name}",
                    "area_pixels": int((instance_mask > 0).sum()),
                })
            overlay = bgr.copy()
            color_layer = np.zeros_like(bgr)
            color_layer[union > 0] = (10, 200, 255)
            overlay = cv2.addWeighted(overlay, 0.72, color_layer, 0.48, 0)
            cv2.putText(overlay, f"SAM {mode_used}: {len(instances)} proposals", (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (20, 20, 20), 3, cv2.LINE_AA)
            cv2.putText(overlay, f"SAM {mode_used}: {len(instances)} proposals", (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
            overlay_path = overlay_root / f"{slug}.jpg"
            overlay_path.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(overlay_path), overlay):
                raise OSError(f"failed writing SAM overlay {overlay_path}")
            image_outputs.append({
                "image": record.key,
                "mode": mode_used,
                "has_pixel_annotation": record.has_mask,
                "annotation_prompt_count": len(instances) if mode_used == "annotation_box_prompts" else 0,
                "instance_count": len(instances),
                "union_mask_file": f"masks/{slug}/union.png",
                "overlay_file": f"overlays/{slug}.jpg",
                "instances": serialized_instances,
            })
            if not cv2.imwrite(str(image_mask_dir / "union.png"), union):
                raise OSError(f"failed writing SAM union mask for {record.key}")
        except Exception as exc:
            skipped.append({"image": record.key, "error": f"{type(exc).__name__}: {exc}"})

    manifest = {
        "status": "completed" if not skipped else "completed_with_skips",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": round(time.time() - started, 3),
        "dataset_root": str(Path(dataset_root).expanduser().resolve()),
        "image_count": len(index.records),
        "processed_count": len(image_outputs),
        "skipped_count": len(skipped),
        "model": {"name": "Segment Anything (SAM)", "type": model_type, "checkpoint_filename": checkpoint.name, "device": chosen_device},
        "requested_prompt_mode": prompt_mode,
        "pseudo_label_warning": "Every SAM mask is a model output/pseudo-label, not verified ground truth. Automatic proposals do not guarantee semantic egg identity.",
        "image_outputs": image_outputs,
        "skipped_images": skipped,
        "dataset_warnings": index.warnings,
    }
    (output_dir / "sam_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest
