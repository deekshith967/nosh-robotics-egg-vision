#!/usr/bin/env python3
"""Generate a factual PDF report from available run JSON/logs only."""
from __future__ import annotations

import argparse
import html
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from pypdf import PdfReader
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
NAVY = colors.HexColor("#12233F")
TEAL = colors.HexColor("#087E8B")
PALE = colors.HexColor("#EEF4F8")
AMBER = colors.HexColor("#FFF2D4")
MUTED = colors.HexColor("#5D6B7A")
RED = colors.HexColor("#A1372A")


def read_json(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def nice(value: Any) -> str:
    if value is None:
        return "Not available"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def metric_table(summary: dict[str, Any] | None, styles: dict[str, ParagraphStyle]) -> Table:
    if not summary or not summary.get("macro"):
        rows = [
            ["Metric", "Macro / image", "Pooled pixels", "Meaning"],
            ["IoU", "N/A", "N/A", "Not evaluated"],
            ["Dice", "N/A", "N/A", "Not evaluated"],
            ["Precision", "N/A", "N/A", "Not evaluated"],
            ["Recall", "N/A", "N/A", "Not evaluated"],
        ]
    else:
        macro = summary["macro"]
        pooled = summary.get("pooled") or {}
        rows = [
            ["Metric", "Macro / image", "Pooled pixels", "Meaning"],
            ["IoU", nice(macro.get("iou")), nice(pooled.get("iou")), "Intersection over union"],
            ["Dice", nice(macro.get("dice")), nice(pooled.get("dice")), "F1 overlap"],
            ["Precision", nice(macro.get("precision")), nice(pooled.get("precision")), "Predicted foreground correctness"],
            ["Recall", nice(macro.get("recall")), nice(pooled.get("recall")), "Ground-truth foreground coverage"],
        ]
    table = Table(rows, colWidths=[30 * mm, 32 * mm, 32 * mm, 80 * mm], repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("LEADING", (0, 0), (-1, -1), 11),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#D1DCE5")),
        ("BACKGROUND", (0, 1), (-1, -1), colors.white),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, PALE]),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    return table


def _validate_and_log_pdf(pdf_path: Path, validation_log: Path) -> dict[str, Any]:
    raw = pdf_path.read_bytes()
    reader = PdfReader(str(pdf_path))
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    checks = {
        "pdf_header": raw.startswith(b"%PDF-"),
        "has_pages": len(reader.pages) > 0,
        "has_title": "Egg Vision Assignment" in text,
        "has_limitations": "SUBMISSION LIMITATION" in text,
        "has_ai_disclosure": "AI assistance disclosure" in text,
    }
    result = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if all(checks.values()) else "failed",
        "pdf_path": str(pdf_path.resolve()),
        "page_count": len(reader.pages),
        "size_bytes": len(raw),
        "checks": checks,
    }
    validation_log.parent.mkdir(parents=True, exist_ok=True)
    validation_log.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if result["status"] != "passed":
        raise RuntimeError(f"PDF validation failed; see {validation_log}: {checks}")
    return result


def build_pdf(args: argparse.Namespace) -> None:
    classical = read_json(args.classical) or read_json(args.classical_attempt)
    sam = read_json(args.sam) or read_json(args.sam_attempt)
    tracking = read_json(args.tracking) or read_json(args.tracking_attempt)
    agreement = read_json(args.agreement) or read_json(args.agreement_attempt)
    smoke = read_json(args.smoke)
    verification = read_json(args.verification)
    audit = read_json(args.audit)

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="TitleCustom", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=25, leading=29, textColor=NAVY, alignment=TA_LEFT, spaceAfter=8))
    styles.add(ParagraphStyle(name="Subtitle", parent=styles["Normal"], fontName="Helvetica", fontSize=12, leading=17, textColor=MUTED, spaceAfter=14))
    styles.add(ParagraphStyle(name="H1Custom", parent=styles["Heading1"], fontName="Helvetica-Bold", fontSize=17, leading=21, textColor=NAVY, spaceBefore=4, spaceAfter=9))
    styles.add(ParagraphStyle(name="H2Custom", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=11.5, leading=14, textColor=TEAL, spaceBefore=8, spaceAfter=4))
    styles.add(ParagraphStyle(name="body", parent=styles["BodyText"], fontName="Helvetica", fontSize=9, leading=13, textColor=colors.HexColor("#263442"), spaceAfter=6))
    styles.add(ParagraphStyle(name="small", parent=styles["BodyText"], fontName="Helvetica", fontSize=7.8, leading=10, textColor=MUTED))
    styles.add(ParagraphStyle(name="alert", parent=styles["BodyText"], fontName="Helvetica-Bold", fontSize=10, leading=14, textColor=RED))
    styles.add(ParagraphStyle(name="card", parent=styles["BodyText"], fontName="Helvetica-Bold", fontSize=11, leading=14, textColor=NAVY, alignment=TA_CENTER))
    styles.add(ParagraphStyle(name="cardlabel", parent=styles["BodyText"], fontName="Helvetica", fontSize=7.5, leading=10, textColor=MUTED, alignment=TA_CENTER))
    styles.add(ParagraphStyle(name="center", parent=styles["BodyText"], alignment=TA_CENTER, fontSize=9, leading=12))

    page_path = args.output
    page_path.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(page_path), pagesize=A4, rightMargin=16 * mm, leftMargin=16 * mm,
        topMargin=18 * mm, bottomMargin=17 * mm,
        title="Nosh Robotics Egg Vision Assignment - Execution Report",
        author="Nosh Robotics Computer Vision Intern Assignment",
        subject="Reproducible egg segmentation, evaluation, and tracking status",
    )

    def on_page(canvas, document):
        canvas.saveState()
        width, height = A4
        canvas.setFillColor(NAVY)
        canvas.rect(0, height - 7 * mm, width, 7 * mm, stroke=0, fill=1)
        canvas.setStrokeColor(colors.HexColor("#DCE5EC"))
        canvas.line(16 * mm, 13 * mm, width - 16 * mm, 13 * mm)
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(MUTED)
        canvas.drawString(16 * mm, 8 * mm, "Nosh Robotics | Egg Vision | Evidence-based status")
        canvas.drawRightString(width - 16 * mm, 8 * mm, f"Page {document.page}")
        canvas.restoreState()

    story: list[Any] = []
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    story.append(Spacer(1, 11 * mm))
    story.append(Paragraph("NOSH ROBOTICS / COMPUTER VISION INTERN", styles["H2Custom"]))
    story.append(Paragraph("Egg Vision Assignment", styles["TitleCustom"]))
    story.append(Paragraph("Segmentation, quantitative evaluation, and persistent-ID video tracking", styles["Subtitle"]))
    story.append(Paragraph(f"Execution report | Generated {generated}", styles["small"]))
    story.append(Spacer(1, 7 * mm))

    alert_text = (
        "SUBMISSION LIMITATION: The original assignment PDF, dataset export, SAM checkpoint, and local source video "
        "were absent. Runtime egress blocked the supplied data/video hosts, so no source HTTP response or archive check "
        "is available and no files were obtained. This report does not claim real-data metrics, dataset-wide SAM results, "
        "or an annotated copy of the supplied video."
    )
    alert = Table([[Paragraph(alert_text, styles["alert"])]], colWidths=[178 * mm])
    alert.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), AMBER),
        ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor("#E8BD66")),
        ("LEFTPADDING", (0, 0), (-1, -1), 10), ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 9), ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
    ]))
    story.append(alert)
    story.append(Spacer(1, 6 * mm))

    classical_status = classical.get("status") if classical else "not run"
    sam_status = sam.get("status") if sam else "not run"
    tracking_status = tracking.get("status") if tracking else "not run"
    verification_status = verification.get("status") if verification else "not recorded"
    cards = [
        [Paragraph("CLASSICAL RUN", styles["cardlabel"]), Paragraph("SAM INFERENCE", styles["cardlabel"]), Paragraph("VIDEO EXPORT", styles["cardlabel"]), Paragraph("CODE CHECKS", styles["cardlabel"])],
        [Paragraph(html.escape(str(classical_status)).upper(), styles["card"]), Paragraph(html.escape(str(sam_status)).upper(), styles["card"]), Paragraph(html.escape(str(tracking_status)).upper(), styles["card"]), Paragraph(html.escape(str(verification_status)).upper(), styles["card"])],
    ]
    card_table = Table(cards, colWidths=[44.5 * mm] * 4, rowHeights=[9 * mm, 14 * mm])
    card_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), PALE),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#D1DCE5")),
        ("INNERGRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#D1DCE5")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(card_table)
    story.append(Spacer(1, 7 * mm))

    story.append(Paragraph("What the available run actually shows", styles["H1Custom"]))
    if smoke and smoke.get("kind") == "synthetic_smoke_test_not_dataset_evaluation" and smoke.get("status") == "passed":
        smoke_text = (
            f"The deterministic, generated {smoke.get('image_size_wh', ['?', '?'])[0]} x "
            f"{smoke.get('image_size_wh', ['?', '?'])[1]} synthetic fixture passed the classical baseline smoke test: "
            f"IoU {nice(smoke.get('synthetic_iou'))}, Dice {nice(smoke.get('synthetic_dice'))}. "
            "This validates basic code execution only; it is explicitly not dataset performance."
        )
    else:
        smoke_text = "No valid synthetic smoke-test log was supplied to the report builder."
    story.append(Paragraph(html.escape(smoke_text), styles["body"]))

    if verification and verification.get("test_summary"):
        counts = verification["test_summary"].get("counts", {})
        verification_text = (
            f"Automated verification log: {counts.get('passed', 0)} passed, {counts.get('failed', 0)} failed, "
            f"{counts.get('skipped', 0)} skipped, {counts.get('errors', 0)} errors; "
            f"overall status: {verification.get('status', 'unknown')}."
        )
    else:
        verification_text = "No automated test summary was supplied to the report builder."
    story.append(Paragraph(html.escape(verification_text), styles["body"]))

    story.append(Paragraph("Assignment data performance", styles["H2Custom"]))
    if classical and classical.get("status") == "completed":
        evaluation = classical.get("evaluation", {})
        summary = evaluation.get("summary")
        story.append(metric_table(summary, styles))
        story.append(Spacer(1, 3 * mm))
        story.append(Paragraph(
            f"Evaluation images: {summary.get('image_count', 'unknown') if summary else 'unknown'}; "
            f"selected configuration: <font name='Courier'>{html.escape(json.dumps(classical.get('selected_config', {}), sort_keys=True))}</font>.", styles["small"]
        ))
    else:
        story.append(metric_table(None, styles))
        blocker = classical.get("blocker") if classical else "No classical run JSON was supplied."
        story.append(Paragraph(f"Blocker recorded: {html.escape(str(blocker))}", styles["small"]))
        story.append(Paragraph("No evaluation scores are stated because there are no verified image annotations to score.", styles["alert"]))

    story.append(PageBreak())
    story.append(Paragraph("Methods and safeguards", styles["H1Custom"]))
    story.append(Paragraph("1. Classical segmentation and model selection", styles["H2Custom"]))
    story.append(Paragraph(
        "A deterministic OpenCV baseline compares grayscale Otsu thresholding (bright/dark polarity) with Lab "
        "distance from the median image-border color. Gaussian blur and elliptical opening/closing suppress noise "
        "and fill small gaps. Nine predeclared parameter settings are ranked by macro development Dice, then "
        "development IoU; candidate order breaks any remaining tie. The predeclared grid records each candidate's "
        "design rationale, and the candidate log records development scores plus keep/discard and selection rationales. "
        "The held-out evaluation partition is read only after that selection.", styles["body"]
    ))
    story.append(Paragraph("2. Split and leakage safeguards", styles["H2Custom"]))
    story.append(Paragraph(
        "Annotated images are divided into development/evaluation at a requested 80/20 ratio (4:1). Exact decoded-image "
        "duplicates and optional regex-captured source sequences are indivisible groups; a seeded subset-sum split seeks "
        "the closest possible image ratio without group overlap. Actual counts and the seed are logged. Near-duplicate "
        "frames require suitable source grouping; no claim of automatic near-duplicate detection is made.", styles["body"]
    ))
    story.append(Paragraph("3. Metrics and qualitative error review", styles["H2Custom"]))
    story.append(Paragraph(
        "For each held-out image the runner records IoU, Dice, precision, recall, TP/FP/FN pixel counts, and an error "
        "category. Both macro per-image means and pooled pixel metrics are emitted, together with per-image CSV/JSON and "
        "four-panel input/ground-truth/prediction/error images. False negatives are blue, false positives red, and true "
        "positives green. Lowest-IoU cases are surfaced for review. A separate held-out comparison to SAM masks, when run, "
        "is explicitly reported as pseudo-label agreement and never as ground-truth accuracy.", styles["body"]
    ))
    story.append(Paragraph("4. Pretrained SAM outputs", styles["H2Custom"]))
    story.append(Paragraph(
        "The optional adapter uses official Segment Anything with a separately supplied local checkpoint. Annotation-prompt "
        "mode uses provided COCO/YOLO boxes or boxes derived from pixel masks; SAM's returned masks remain pseudo-labels. "
        "Automatic proposals are not guaranteed to be eggs. The pinned upstream README identifies the official ViT-B weight "
        "URL and Apache-2.0 model license, but its checkpoint host was not reachable and no weight or checksum was obtained. "
        "Any comparison against SAM is agreement only, not verified ground-truth accuracy.", styles["body"]
    ))
    story.append(Paragraph("5. Tracking", styles["H2Custom"]))
    story.append(Paragraph(
        "The transparent video baseline segments each frame, filters connected components, predicts centroid motion, and "
        "greedily associates boxes by distance and overlap. It writes frame labels/IDs to MP4 and a machine-readable sidecar. "
        "Crossings, occlusion, touching eggs, or poor foreground separation can switch IDs; visual review on the actual video "
        "is required.", styles["body"]
    ))

    story.append(PageBreak())
    story.append(Paragraph("Execution evidence and remaining work", styles["H1Custom"]))
    rows = [["Workstream", "Status", "Evidence / blocker"]]
    if classical and classical.get("status") == "completed":
        ev = classical.get("evaluation", {}).get("summary", {})
        detail = f"{ev.get('image_count', '?')} labeled evaluation images; run JSON, per-image CSV, candidate log, and visualizations available."
        status = "Executed"
    else:
        detail = str(classical.get("blocker") if classical else "No completed run log; dataset annotations were not supplied.")
        status = "Blocked"
    rows.append(["Classical segmentation + evaluation", status, Paragraph(html.escape(detail), styles["small"])])
    rows.append([
        "Dataset/video/checkpoint retrieval",
        "Blocked",
        Paragraph("No requests sent: runtime egress excludes the supplied Roboflow, Drive, YouTube, and SAM checkpoint hosts. No files obtained; HTTP status, redirects, MIME, and ZIP integrity are unavailable. See experiments/logs/source_access.json.", styles["small"]),
    ])
    if sam and int(sam.get("processed_count", 0)) >= int(sam.get("image_count", 1)):
        detail, status = f"Processed {sam.get('processed_count')} / {sam.get('image_count')} images; see SAM manifest.", "Executed"
    else:
        detail = sam.get("blocker") if sam and sam.get("status") == "blocked" else "No complete SAM manifest. Dataset images, optional PyTorch/SAM dependencies, and a local pretrained checkpoint are required."
        status = "Blocked" if sam and sam.get("status") == "blocked" else "Not executed"
    rows.append(["Pretrained SAM", status, Paragraph(html.escape(str(detail)), styles["small"])])
    if tracking and tracking.get("status") == "completed":
        detail, status = f"{tracking.get('frames_processed')} frames; output {tracking.get('output_filename')} ({tracking.get('output_bytes')} bytes).", "Executed"
    else:
        detail = tracking.get("blocker") if tracking and tracking.get("status") == "blocked" else "No tracking sidecar/MP4 from the supplied clip; a local video copy was not present."
        status = "Blocked" if tracking and tracking.get("status") == "blocked" else "Not executed"
    rows.append(["Video tracking", status, Paragraph(html.escape(str(detail)), styles["small"])])
    if agreement and agreement.get("status") in {"completed", "completed_with_skips"}:
        macro = (agreement.get("summary") or {}).get("macro") or {}
        detail = (
            f"{agreement.get('compared_images', 0)} held-out images; pseudo-label agreement IoU={nice(macro.get('iou'))}, "
            f"Dice={nice(macro.get('dice'))}. This is not ground-truth accuracy."
        )
        agreement_status = str(agreement.get("status"))
    elif agreement and agreement.get("status") == "blocked":
        detail = agreement.get("blocker", "Inputs required for agreement were unavailable.")
        agreement_status = "Blocked"
    else:
        detail = "Not run; requires a completed classical holdout and SAM union masks."
        agreement_status = "Not executed"
    rows.append(["Classical vs SAM pseudo-label agreement", agreement_status, Paragraph(html.escape(str(detail)), styles["small"])])
    rows.append(["Synthetic smoke test", "Passed" if smoke and smoke.get("status") == "passed" else "Not verified", Paragraph("Recorded controlled fixture only; not assignment-data evaluation.", styles["small"])])
    rows.append(["Automated tests", verification.get("status", "Not recorded") if verification else "Not recorded", Paragraph("See experiments/logs/verification.json; tests include data-free fixtures and a temporary synthetic MP4.", styles["small"])])
    rows.append(["Assignment PDF review", "Blocked", Paragraph("No original assignment PDF was present; exact brief/grading criteria remain unchecked.", styles["small"])])
    evidence = Table(rows, colWidths=[42 * mm, 25 * mm, 107 * mm], repeatRows=1)
    evidence.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"), ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#D1DCE5")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, PALE]),
        ("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6), ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(evidence)
    story.append(Spacer(1, 5 * mm))

    if audit:
        inputs = audit.get("inputs", {})
        resources = audit.get("resources", {})
        memory = resources.get("memory", {})
        disk = resources.get("disk", {})
        gib = lambda value: f"{value / (1024 ** 3):.1f} GiB" if isinstance(value, (int, float)) else "unknown"
        env_text = (
            f"Audited environment: Python {audit.get('python', 'unknown')}; {audit.get('cpu_cores', 'unknown')} CPU cores; "
            f"RAM available {gib(memory.get('available_bytes'))} / {gib(memory.get('total_bytes'))}; "
            f"disk free {gib(disk.get('free_bytes'))} / {gib(disk.get('total_bytes'))}; "
            f"GPU probe: {audit.get('gpu_probe', 'not recorded')}. Local dataset available: "
            f"{inputs.get('dataset', {}).get('available', False)} ({inputs.get('dataset', {}).get('image_count', 0)} image files); "
            f"video: {inputs.get('video', {}).get('available', False)}; SAM checkpoint: {inputs.get('sam_checkpoint', {}).get('available', False)}; "
            f"assignment brief PDF found: {inputs.get('assignment_pdf', {}).get('available', False)}."
        )
        story.append(Paragraph(html.escape(env_text), styles["small"]))

    story.append(Paragraph("What is needed to finish the real-data submission", styles["H2Custom"]))
    for line in [
        "Provide the complete assignment PDF and check for any additional requirements beyond the four tasks in the prompt.",
        "Place the segmentation dataset and its pixel annotations under data/egg-dataset; run the classical experiment and review the saved holdout visualizations.",
        "Install PyTorch/SAM, provide an official local checkpoint, run SAM over all images, and verify the manifest has no skipped images.",
        "Place a local copy of the specified video under data/egg-video.mp4, run tracking, inspect ID continuity, and play the exported MP4.",
        "Rebuild this PDF from the real logs and include only actual results; rerun the test suite before submission.",
    ]:
        story.append(Paragraph("- " + html.escape(line), styles["body"]))

    story.append(PageBreak())
    story.append(Paragraph("Reproducibility and disclosure", styles["H1Custom"]))
    story.append(Paragraph("Source and output map", styles["H2Custom"]))
    files = [
        ("configs/experiment_grid.json", "Versioned, predeclared nine-candidate tuning grid with rationales"),
        ("docs/completion_checklist.md", "Real-input blockers and exact completion steps"),
        ("eggvision/classical.py", "OpenCV segmentation candidates"),
        ("eggvision/data.py", "Dataset discovery, annotation decoding, duplicate grouping, 80/20 split"),
        ("eggvision/experiment.py", "Development-only parameter selection and held-out evaluation"),
        ("eggvision/metrics.py", "IoU, Dice, precision, recall, and error visualizations"),
        ("eggvision/pseudo_agreement.py", "Frozen holdout comparison against explicitly unverified SAM pseudo-labels"),
        ("scripts/compare_sam_classical.py", "Separate pseudo-label agreement CLI; not accuracy scoring"),
        ("eggvision/sam_inference.py", "Optional pretrained SAM image inference"),
        ("eggvision/tracking.py", "Classical detections, persistent IDs, MP4 and JSON output"),
        ("tests/", "Unit, data-format, split, evaluation, and synthetic-video tests"),
        ("experiments/logs/", "Input audit, test/smoke verification, and PDF validation logs"),
        ("runs/", "Ignored local experiment outputs, SAM masks, and video artifacts"),
    ]
    file_rows = [["Path", "Purpose"]] + [[Paragraph(f"<font name='Courier'>{html.escape(path)}</font>", styles["small"]), Paragraph(html.escape(desc), styles["small"])] for path, desc in files]
    file_table = Table(file_rows, colWidths=[67 * mm, 107 * mm], repeatRows=1)
    file_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"), ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#D1DCE5")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, PALE]),
        ("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6), ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(file_table)
    story.append(Spacer(1, 6 * mm))
    story.append(Paragraph("AI assistance disclosure", styles["H2Custom"]))
    story.append(Paragraph(
        "AI assistance was used to scaffold and edit the implementation, tests, documentation, experiment protocol, and "
        "report. A human must review and validate the code against the original assignment brief and actual data. The "
        "controlled synthetic fixture is clearly identified and is not represented as dataset performance. No real-data "
        "metric, SAM dataset output, or supplied-video output is claimed unless supported by its corresponding run artifact. "
        "No credentials or pretrained model checkpoint are included.", styles["body"]
    ))
    story.append(Paragraph("Reproduce the verified checks", styles["H2Custom"]))
    story.append(Paragraph(
        "python -m pip install -r requirements.txt<br/>"
        "python scripts/verify.py<br/>"
        "python scripts/run_classical.py --data data/egg-dataset --output runs/classical --seed 42<br/>"
        "python scripts/run_sam.py --data data/egg-dataset --checkpoint models/sam_vit_b_01ec64.pth --output runs/sam<br/>"
        "python scripts/track_video.py --video data/egg-video.mp4 --config runs/classical/selected_config.json --output runs/tracking/annotated.mp4",
        styles["small"]
    ))

    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
    _validate_and_log_pdf(page_path, args.validation_log)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classical", type=Path, default=ROOT / "runs" / "classical" / "experiment.json")
    parser.add_argument("--classical-attempt", type=Path, default=ROOT / "experiments" / "logs" / "classical_attempt.json")
    parser.add_argument("--sam", type=Path, default=ROOT / "runs" / "sam" / "sam_manifest.json")
    parser.add_argument("--sam-attempt", type=Path, default=ROOT / "experiments" / "logs" / "sam_attempt.json")
    parser.add_argument("--tracking", type=Path, default=ROOT / "runs" / "tracking" / "annotated.json")
    parser.add_argument("--tracking-attempt", type=Path, default=ROOT / "experiments" / "logs" / "tracking_attempt.json")
    parser.add_argument("--agreement", type=Path, default=ROOT / "runs" / "pseudo-agreement" / "pseudo_agreement.json")
    parser.add_argument("--agreement-attempt", type=Path, default=ROOT / "experiments" / "logs" / "pseudo_agreement_attempt.json")
    parser.add_argument("--smoke", type=Path, default=ROOT / "experiments" / "logs" / "baseline_smoke.json")
    parser.add_argument("--verification", type=Path, default=ROOT / "experiments" / "logs" / "verification.json")
    parser.add_argument("--audit", type=Path, default=ROOT / "experiments" / "logs" / "input_audit.json")
    parser.add_argument("--output", type=Path, default=ROOT / "reports" / "submission_report.pdf")
    parser.add_argument("--validation-log", type=Path, default=ROOT / "experiments" / "logs" / "pdf_validation.json")
    args = parser.parse_args()
    build_pdf(args)
    print(f"Wrote report: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
