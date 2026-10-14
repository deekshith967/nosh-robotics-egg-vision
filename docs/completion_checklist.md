# Submission completion checklist (current workspace)

## Deliverables already implemented and independently verified

- [x] Pretrained SAM runner for local dataset images, including COCO/YOLO box prompts, mask-derived prompts, automatic proposals, per-image masks/overlays, and a processing manifest.
- [x] Classical OpenCV segmentation methods and a predeclared nine-candidate grid.
- [x] Deterministic, group-aware 80/20 split; candidate metrics and keep/discard/selection rationales are recorded. Evaluation images are not used for selection.
- [x] Ground-truth metric and qualitative-error pipeline (IoU, Dice, precision, recall, per-image counts, pooled/macro summaries, failure modes, visualizations).
- [x] Separate frozen classical-vs-SAM pseudo-label agreement workflow; its output is explicitly not ground-truth accuracy.
- [x] Persistent-ID detector/tracker, MP4 writer, JSON sidecar, and frame-level test coverage.
- [x] Automated PDF report generator and PDF validity test.
- [x] AI-assistance disclosure, setup commands, ignored input/output paths, and source-access audit.
- [x] Latest local test log reports 21 passed, 0 failed; all fixtures are synthetic/test data.

## Required real-assignment inputs still missing

- [ ] Original assignment brief PDF at `assignment.pdf` (required to validate exact wording and grading criteria).
- [ ] Roboflow segmentation export or supplied Google Drive ZIP under `data/egg-dataset/` (no real images were obtained; local image count is 0).
- [ ] Official SAM ViT-B checkpoint at `models/sam_vit_b_01ec64.pth` (no weights or checksum verified).
- [ ] Authorized local copy of the exact supplied YouTube clip at `data/egg-video.mp4` (no real annotated MP4 exists).

Source hosts are blocked by this runtime's egress restrictions. No requests were sent to them; HTTP codes, redirects, ZIP integrity, and media contents were therefore not measured. GitHub checks found no exact dataset mirror and no assignment PDF. See `experiments/logs/source_access.json`.

## Completion procedure once inputs are transferred

1. Put the original PDF at `assignment.pdf`; verify the exact assignment instructions before claiming full compliance.
2. Export/download the provided dataset in COCO polygon/RLE, YOLO polygon-segmentation, or paired-mask format and place its images and annotations under `data/egg-dataset/`. If given the original ZIP, run `unzip -t <downloaded-zip>` before extraction. Run `python scripts/audit_inputs.py` and confirm the image/mask counts.
3. Provide the official SAM ViT-B checkpoint from the URL recorded in `README.md`. Run `sha256sum models/sam_vit_b_01ec64.pth` and save the digest; install compatible PyTorch and `requirements-sam.txt`.
4. Run `python scripts/run_classical.py --data data/egg-dataset --output runs/classical --seed 42`. Review actual group split counts, all nine development candidates, selection rationales, held-out metrics, and qualitative error images.
5. Run `python scripts/run_sam.py --data data/egg-dataset --checkpoint models/sam_vit_b_01ec64.pth --model-type vit_b --prompt-mode annotations --device auto --output runs/sam`. Confirm the manifest has no skipped images. These outputs remain pseudo-labels, not verified ground truth.
6. Run `python scripts/compare_sam_classical.py --data data/egg-dataset --classical runs/classical/experiment.json --sam runs/sam/sam_manifest.json --output runs/pseudo-agreement` only as agreement analysis; report it separately from ground-truth scores.
7. Place the exact supplied clip at `data/egg-video.mp4`. Run `python scripts/track_video.py --video data/egg-video.mp4 --config runs/classical/selected_config.json --output runs/tracking/annotated.mp4`; watch the entire MP4 and review ID continuity and JSON metadata.
8. Run `python scripts/verify.py` and `python scripts/build_report.py`. Review the regenerated PDF and every result against the saved logs; confirm `experiments/logs/pdf_validation.json` reports `passed` before publication.

Until those four missing inputs are supplied, the assignment remains **partially implemented but not complete**. No real-data scores or supplied-video output should be stated.
