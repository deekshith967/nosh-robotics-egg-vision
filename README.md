# Nosh Robotics — Egg Vision Assignment

An explainable, reproducible computer-vision baseline for egg segmentation, pixel-level evaluation, and persistent-ID video tracking. **The original assignment PDF, dataset, SAM checkpoint, and source video were not present in the inspected checkout.** Consequently, the four workstreams below are implemented from the requirements in the task prompt, but real-data metrics, SAM dataset predictions, and a video of the supplied clip are not claimed as completed. See the [`completion checklist`](docs/completion_checklist.md), [`assignment PDF status`](docs/assignment_pdf_status.md), and generated [PDF status report](reports/submission_report.pdf).

## What is implemented

1. **Pretrained segmentation:** an optional official Segment Anything Model (SAM) adapter that visits every supported image in a local dataset export. It can prompt SAM from COCO/YOLO bounding boxes, derive boxes from supplied egg masks, or save automatic SAM proposals. Outputs are explicitly marked as **pseudo-labels, not verified ground truth**; automatic proposals do not guarantee semantic egg identity.
2. **Classical segmentation:** OpenCV Otsu bright/dark thresholding and Lab border-color-distance candidates with smoothing and morphology. A predeclared nine-setting parameter sweep selects the configuration using **development-set macro Dice only**. The labeled images are split 80/20 at exact-image duplicate/capture-group level; the seed and actual split counts are recorded.
3. **Evaluation:** per-image IoU, Dice, precision, recall, TP/FP/FN counts, macro and pooled summaries, lowest-IoU cases, failure-mode counts, and TP/FP/FN visualizations. Only supplied pixel annotations are treated as ground truth. SAM outputs are never passed off as verified labels.
4. **Video tracking:** a classical connected-component detector plus greedy centroid/IoU association and short-term constant-velocity prediction; writes annotated MP4 with persistent IDs and a frame-by-frame JSON sidecar. This is an interpretable baseline; it must be reviewed and tuned on the actual video.

The detailed assumptions, split policy, experiment grid, scoring conventions, and limitations are documented in [`docs/experiment_protocol.md`](docs/experiment_protocol.md).

## Environment and setup

Tested in this workspace with Python 3.11.2, Linux, 2 CPU cores, OpenCV 4.14.0, NumPy 2.4.6, and no available CUDA/PyTorch installation. The following commands keep dependencies in a local virtual environment:

```bash
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For pretrained SAM, see the [official Segment Anything reference](https://github.com/facebookresearch/segment-anything). First install a PyTorch build appropriate to the machine (choose the CPU/CUDA command from the [official PyTorch installer](https://pytorch.org/get-started/locally/)), then install the optional reference implementation:

```bash
python -m pip install -r requirements-sam.txt
```

Download an official SAM checkpoint separately, place it under `models/` (ignored by Git), and pass its local path. **No checkpoint is bundled.** Checkpoint names, dependency versions, and machine setup can affect runtime and output.

## Dataset layout and preparation

Export/download the [supplied Roboflow egg dataset](https://universe.roboflow.com/fire-bf7qe/egg-vzhtr/dataset/1) locally in a segmentation-compatible format and unpack it under `data/egg-dataset/`. The [alternative ZIP](https://drive.google.com/file/d/1a58rACcCFnaj0ZiDoNyGbDWXunRyWziv/view?usp=sharing) may be used if it includes images plus usable segmentation annotations. Local files under `data/` are ignored by Git. Supported annotations:

- COCO instance polygon or RLE segmentation JSON;
- YOLO polygon-segmentation text labels paired with `images/` and `labels/`;
- paired binary/semantic PNG masks under `masks/` or named `<image-stem>_mask.png`.

Bounding-box-only labels are not silently treated as pixel ground truth; COCO/YOLO boxes can still prompt SAM, but cannot support pixel-level evaluation. If the source export has another schema, add and test an explicit adapter before reporting results. `--class-ids` can select COCO/YOLO category IDs. When an egg-named COCO category exists it is selected automatically; a sole category is also selected by default.

For capture/video-related images, pass a regex whose capture group identifies a source sequence, e.g. `--group-regex '(clip_[A-Za-z0-9]+)_frame'`. Exact pixel-identical images are grouped automatically. Near-duplicates are not automatically discovered; use source/capture grouping when appropriate.

## Input retrieval status and manual handoff

On 2026-10-14, the runtime's outbound egress policy excluded `universe.roboflow.com`, `drive.google.com`, `youtube.com`, and the SAM checkpoint host `dl.fbaipublicfiles.com`. No HTTP requests were sent to those hosts: HTTP status, redirects, ZIP integrity, and file contents therefore remain **unverified**, and no input files were obtained. A GitHub API check of the upstream repository returned HTTP 200 but found only the starter README; a GitHub repository search for `egg-vzhtr` returned zero results. No replacement dataset/video was substituted. See [`experiments/logs/source_access.json`](experiments/logs/source_access.json) for the exact source URLs and checks.

To unblock execution, download/upload these exact inputs to the workspace (not Git):

- Roboflow export from the supplied link, or the exact Google Drive ZIP, with pixel segmentation annotations, unpacked under `data/egg-dataset/`.
- The exact supplied YouTube clip saved as `data/egg-video.mp4`.
- The original assignment PDF saved as `assignment.pdf` (it is absent locally and from the upstream GitHub tree; exact grading requirements cannot otherwise be checked).
- Optional official SAM ViT-B checkpoint saved as `models/sam_vit_b_01ec64.pth`. The pinned [official SAM README](https://github.com/facebookresearch/segment-anything/blob/dca509fe793f601edb92606367a655c15ac00fdf/README.md#model-checkpoints) lists its download as [`sam_vit_b_01ec64.pth`](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth) and states the model is Apache-2.0. The checkpoint host was not reachable here; no checkpoint was downloaded or hash-verified, and that README does not publish a SHA-256 checksum. After you provide the file, record `sha256sum models/sam_vit_b_01ec64.pth` in the run log.

## Exact execution commands

Run the fast controlled synthetic baseline and test suite (these do **not** measure assignment-dataset performance):

```bash
python scripts/smoke_baseline.py
python -m pytest -q
```

Run classical development/evaluation parameter selection and final held-out evaluation:

```bash
python scripts/run_classical.py \
  --data data/egg-dataset \
  --output runs/classical \
  --seed 42
```

If the images are frames or captures that should stay together, add the appropriate grouping expression, e.g. `--group-regex '(clip_[A-Za-z0-9]+)_frame'`. The run writes `runs/classical/experiment.json`, `development_candidates.csv` (parameters, rationale, development metrics, keep/discard decision), `evaluation_per_image.csv`, `selected_config.json`, and `qualitative/`. Evaluation data is not used for selection.

Run pretrained SAM inference over the local export (requires the optional dependencies and local checkpoint):

```bash
python scripts/run_sam.py \
  --data data/egg-dataset \
  --checkpoint models/sam_vit_b_01ec64.pth \
  --model-type vit_b \
  --prompt-mode annotations \
  --device auto \
  --output runs/sam
```

Use `--prompt-mode automatic` to export all SAM proposals instead of using dataset masks as box prompts. Review `sam_manifest.json`, per-instance masks, union masks, and overlays. An inference manifest can report skipped images; confirm `processed_count == image_count` before calling the task complete.

Compare the frozen classical configuration with SAM union masks on the held-out evaluation images (these are **pseudo-label agreement scores**, not ground-truth accuracy):

```bash
python scripts/compare_sam_classical.py \
  --data data/egg-dataset \
  --classical runs/classical/experiment.json \
  --sam runs/sam/sam_manifest.json \
  --output runs/pseudo-agreement
```

Track the [supplied YouTube video](https://www.youtube.com/watch?v=Y4JyV94y2_A) after saving a local copy as `data/egg-video.mp4`:

```bash
python scripts/track_video.py \
  --video data/egg-video.mp4 \
  --config runs/classical/selected_config.json \
  --output runs/tracking/annotated.mp4
```

This command writes an MP4 and adjacent JSON sidecar. The supplied YouTube URL is not downloaded by the program; remote video access is not configured here. Review the result frame-by-frame for ID switches and missed/merged eggs.

Build or refresh the report after running experiments:

```bash
python scripts/build_report.py \
  --classical runs/classical/experiment.json \
  --sam runs/sam/sam_manifest.json \
  --tracking runs/tracking/annotated.json \
  --agreement runs/pseudo-agreement/pseudo_agreement.json \
  --smoke experiments/logs/baseline_smoke.json \
  --output reports/submission_report.pdf
```

Omit any `--sam` or `--tracking` path that does not exist; the report will mark that workstream as not run/blocked. Each successful build writes `experiments/logs/pdf_validation.json` with page count, byte size, and PDF-content checks. Do not invent metrics to fill a missing result.

## Logs, outputs, and Git hygiene

- `configs/experiment_grid.json` is the checked-in, predeclared tuning configuration; each successful run records its SHA-256, selected config, seed, and split details.
- `experiments/logs/baseline_smoke.json`, `input_audit.json`, `source_access.json`, `verification.json`, `pdf_validation.json`, and the `*_attempt.json` files are checked-in logs; the baseline fixture is **synthetic**, not a dataset experiment, and the attempt logs record the exact missing-input blockers.
- Real experiment artifacts, downloaded inputs/checkpoints, and MP4 outputs are under ignored `runs/`, `data/`, and `models/` paths. Keep checkpoints, videos, secrets, and unnecessary datasets out of Git.
- Configuration, code, protocol, test suite, logs, and this report are small, reviewable source deliverables.
- Before submission, verify report values against run JSON/CSV, confirm all dataset images were processed by SAM, confirm the 80/20 split is group-safe, and play the annotated MP4.

## Tests and current blockers

`python -m pytest -q` covers segmentation, label decoding, duplicate/group splitting, metrics, visualizations, tracking ID persistence, and a generated temporary test video when the local OpenCV build supports MP4 writing. These tests validate code behavior only; they are not evidence of performance on the missing assignment dataset/video.

At initial inspection, the repository had only the starter README. No assignment PDF was found in the checkout or `/home/user`; no dataset/video/checkpoint was locally available. The PDF report records these blockers and leaves assignment performance metrics as unavailable. Provide the exact assignment PDF and local data/checkpoint/video before making a complete-submission claim.

## AI assistance disclosure

AI assistance was used to scaffold and edit the implementation, tests, documentation, experiment protocol, and report. The code and outputs require human review and validation against the actual assignment brief and data. Synthetic fixtures are identified as synthetic; no dataset performance, pretrained-inference result, or supplied-video output is claimed unless its corresponding real run artifact exists. No credentials or model checkpoints are included.
