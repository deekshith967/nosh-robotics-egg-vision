# Experiment protocol and limitations

## Input/annotation contract

The repository supports a local Roboflow-style export with JPEG/PNG images and one of:

1. COCO JSON instance `segmentation` polygons (uncompressed or compressed RLE; compressed RLE requires optional `pycocotools`),
2. YOLO polygon-segmentation text files paired through `images/` and `labels/`, or
3. paired binary/semantic PNG masks in a `masks/` directory or named `<stem>_mask.png`.

Bounding-box-only labels are **not** converted into pixel ground truth. COCO/YOLO boxes can provide SAM prompts, but remain insufficient for pixel-level evaluation. COCO categories named `egg` are selected when available; a sole category is selected automatically; otherwise all foreground categories are included unless `--class-ids` is passed. If the supplied export differs, adapt the parser explicitly and record that change before running.

## Classical segmentation and parameter selection

`eggvision.classical.segment_classical` has two explainable candidates:

- grayscale Otsu thresholding, with bright and dark foreground polarities;
- Lab color distance from the median image-border color.

All candidates use Gaussian smoothing and elliptical open/close morphology. The checked-in `configs/experiment_grid.json` predeclares nine settings and a method rationale before evaluation: three bright-Otsu settings, two dark-Otsu settings, and four border-distance settings (thresholds 16, 24, 36, 48, paired with documented blur/morphology values). Each candidate log records its parameters, rationale, development metrics, keep/discard decision, and selection rationale. The run JSON stores the grid path and SHA-256 digest. The selection criterion is **macro development-set Dice**, then development IoU, with candidate order as a deterministic final tie-break.

The classical experiment uses only images with pixel annotations. A seeded 80/20 development/evaluation split groups exact decoded-image duplicates; `--group-regex` can additionally group video/capture sequences by a regex capture group. No group crosses the split. The held-out evaluation set is consulted only once, after development-set parameter selection. With fewer than two independent groups, the run is blocked rather than leaking samples. Group indivisibility may make the image-count ratio slightly different from exactly 4:1; the actual counts are logged.

Exact pixel-identical images are detected. Near-duplicates are not automatically detected, so use `--group-regex` if filenames or folder structure encode acquisition sequences. The split seed, image counts, group count, config grid, selected config, dependency versions, and complete per-image evaluation metrics are saved with each run.

## Metrics, visualizations, and failure analysis

For each held-out annotated image the pipeline records IoU, Dice, precision, recall, TP/FP/FN pixel counts, annotation source, and a failure category (false-negative-dominant, false-positive-dominant, or mixed). It reports both unweighted macro-per-image averages and pooled pixel-level metrics. Empty-mask conventions are implemented and documented in `eggvision/metrics.py`.

For each evaluation image a four-panel PNG is generated (input, ground truth, prediction, and an error map: TP green, FP red, FN blue). The ten lowest-IoU images are called out in the JSON summary. Do not cite the synthetic smoke-test IoU/Dice as assignment results.

After both the selected classical run and SAM inference exist, `scripts/compare_sam_classical.py` compares the frozen classical config with SAM union masks on the held-out evaluation image list. It emits per-image/macro/pixel metrics and error images labeled `SAM pseudo-label (NOT GT)`. These are agreement scores against model outputs, not verified accuracy. The normal ground-truth metrics remain separate and require dataset pixel annotations.

## Pretrained segmentation

The optional adapter uses Meta's Segment Anything Model (SAM), a local official checkpoint, and either:

- `annotations` mode: use provided COCO/YOLO bounding boxes when present; otherwise derive box prompts from provided segmentation masks, then retain the top-scoring SAM output per prompt. Images with no prompt fall back to automatic proposals; or
- `automatic` mode: save all SAM automatic-mask-generator proposals.

SAM outputs are **pseudo-labels, not verified ground truth**. Automatic proposals do not guarantee semantic egg identity. They are never substituted for dataset ground-truth masks in the classical metric calculation.

## Video tracking

The video pipeline segments each frame with the selected classical configuration, filters connected components by configurable area/shape, and associates detections using greedy one-to-one distance/IoU matching with constant-velocity prediction. It exports an MP4 with persistent IDs and a per-frame JSON sidecar. This is an interpretable baseline, not a learned multi-object tracker: occlusion, crossings, touching eggs, camera cuts, or poor threshold separation can break IDs. Validate thresholds and visually review the actual video.

## Current evidence and blockers

The repository checkout did not contain the assignment PDF, a dataset export, a SAM checkpoint, or a local copy of the supplied video. Runtime egress restrictions exclude the supplied Roboflow, Google Drive, and YouTube hosts and the official checkpoint host; no HTTP requests were sent, so there are no HTTP/redirect/archive-integrity results. The permitted GitHub API checks found only the starter README upstream and no exact `egg-vzhtr` repository mirror. See `experiments/logs/source_access.json`.

The official pinned SAM GitHub README identifies the ViT-B checkpoint source and states the model is Apache-2.0; the weight file and a trusted checksum were not obtained here. This workspace has two CPU cores, about 3.6 GiB available RAM and 19 GiB free disk at audit time, no confirmed GPU, and no PyTorch installation. Thus this checkout can verify code paths and controlled synthetic tests, but it cannot report real dataset metrics, claim pretrained SAM inference on the dataset, or produce the requested annotated video. The generated PDF report calls these items out explicitly. Do not present synthetic-fixture test results as real egg-vision performance.
