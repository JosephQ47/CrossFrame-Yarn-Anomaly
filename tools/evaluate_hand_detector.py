#!/usr/bin/env python3
"""Evaluate the production person/face/hand ONNX model on labelled images.

@spec docs/spec.md §4.5
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
CLASS_NAMES = {0: "person", 1: "face", 2: "hand"}


@dataclass
class Detection:
    class_id: int
    confidence: float
    xyxy: tuple[float, float, float, float]


def letterbox(image: np.ndarray, size: int = 640) -> tuple[np.ndarray, float, tuple[float, float]]:
    height, width = image.shape[:2]
    ratio = min(size / height, size / width)
    resized = (round(width * ratio), round(height * ratio))
    pad_w = (size - resized[0]) / 2
    pad_h = (size - resized[1]) / 2
    if (width, height) != resized:
        image = cv2.resize(image, resized, interpolation=cv2.INTER_LINEAR)
    top, bottom = round(pad_h - 0.1), round(pad_h + 0.1)
    left, right = round(pad_w - 0.1), round(pad_w + 0.1)
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return image, ratio, (pad_w, pad_h)


def infer(
    session: ort.InferenceSession,
    path: Path,
    confidence: float,
    iou: float,
    class_ids_allowed: set[int],
    roi: tuple[float, float, float, float] | None,
) -> list[Detection]:
    # cv2.imread cannot reliably open non-ASCII Windows paths.
    original = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if original is None:
        raise ValueError(f"cannot read image: {path}")
    offset_x = offset_y = 0
    inference_image = original
    if roi is not None:
        height, width = original.shape[:2]
        x1, y1, x2, y2 = roi
        px1, py1 = int(round(x1 * width)), int(round(y1 * height))
        px2, py2 = int(round(x2 * width)), int(round(y2 * height))
        inference_image = original[py1:py2, px1:px2]
        offset_x, offset_y = px1, py1
    image, ratio, (pad_w, pad_h) = letterbox(inference_image)
    tensor = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
    tensor = np.ascontiguousarray(tensor, dtype=np.float32)[None] / 255.0
    output = session.run(None, {session.get_inputs()[0].name: tensor})[0][0]

    class_scores = output[:, 5:]
    class_ids = class_scores.argmax(axis=1)
    scores = output[:, 4] * class_scores[np.arange(len(output)), class_ids]
    keep = (scores >= confidence) & np.isin(class_ids, list(class_ids_allowed))
    boxes = output[keep, :4]
    scores = scores[keep]
    class_ids = class_ids[keep]
    if len(boxes) == 0:
        return []

    boxes_xywh = np.column_stack((boxes[:, 0] - boxes[:, 2] / 2, boxes[:, 1] - boxes[:, 3] / 2, boxes[:, 2], boxes[:, 3]))
    selected: list[int] = []
    for class_id in np.unique(class_ids):
        indices = np.flatnonzero(class_ids == class_id)
        local = cv2.dnn.NMSBoxes(boxes_xywh[indices].tolist(), scores[indices].tolist(), confidence, iou)
        if len(local):
            selected.extend(indices[np.asarray(local).reshape(-1)].tolist())

    detections: list[Detection] = []
    height, width = original.shape[:2]
    for index in sorted(selected, key=lambda item: float(scores[item]), reverse=True):
        x, y, w, h = boxes_xywh[index]
        x1 = np.clip((x - pad_w) / ratio + offset_x, 0, width)
        y1 = np.clip((y - pad_h) / ratio + offset_y, 0, height)
        x2 = np.clip((x + w - pad_w) / ratio + offset_x, 0, width)
        y2 = np.clip((y + h - pad_h) / ratio + offset_y, 0, height)
        detections.append(Detection(int(class_ids[index]), float(scores[index]), (float(x1), float(y1), float(x2), float(y2))))
    return detections


def load_labels(path: Path | None) -> dict[str, bool]:
    if path is None:
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = csv.DictReader(handle)
        required = {"file", "hand_intrusion"}
        if not required.issubset(rows.fieldnames or []):
            raise ValueError("labels CSV requires columns: file,hand_intrusion")
        return {row["file"].replace("\\", "/"): row["hand_intrusion"].strip().lower() in {"1", "true", "yes", "y"} for row in rows}


def collect_images(roots: list[Path]) -> list[Path]:
    images: list[Path] = []
    for root in roots:
        if root.is_file() and root.suffix.lower() in IMAGE_SUFFIXES:
            images.append(root)
        elif root.is_dir():
            images.extend(path for path in root.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)
    return sorted(set(path.resolve() for path in images), key=lambda path: str(path).lower())


def metrics(rows: list[dict[str, object]]) -> dict[str, object]:
    labelled = [row for row in rows if row["ground_truth"] != ""]
    if not labelled:
        return {"labelled_images": 0, "note": "No labels supplied; prediction inventory only."}
    tp = sum(row["ground_truth"] is True and row["predicted"] is True for row in labelled)
    fp = sum(row["ground_truth"] is False and row["predicted"] is True for row in labelled)
    tn = sum(row["ground_truth"] is False and row["predicted"] is False for row in labelled)
    fn = sum(row["ground_truth"] is True and row["predicted"] is False for row in labelled)
    safe_div = lambda num, den: num / den if den else None
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    return {
        "labelled_images": len(labelled), "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "recall": recall, "false_positive_rate": safe_div(fp, fp + tn),
        "precision": precision,
        "f1": safe_div(2 * precision * recall, precision + recall) if precision is not None and recall is not None else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--images", type=Path, nargs="+", required=True)
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--classes", nargs="+", choices=sorted(CLASS_NAMES.values()), default=["hand"])
    parser.add_argument("--roi", type=float, nargs=4, metavar=("X1", "Y1", "X2", "Y2"), help="normalized inference ROI")
    args = parser.parse_args()

    labels = load_labels(args.labels)
    images = collect_images(args.images)
    session = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])
    allowed = {class_id for class_id, name in CLASS_NAMES.items() if name in args.classes}
    roi = tuple(args.roi) if args.roi else None
    if roi is not None and not (0 <= roi[0] < roi[2] <= 1 and 0 <= roi[1] < roi[3] <= 1):
        raise ValueError("ROI must satisfy 0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1")
    rows: list[dict[str, object]] = []
    for path in images:
        detections = infer(session, path, args.confidence, args.iou, allowed, roi)
        key_candidates = [str(path).replace("\\", "/"), path.name]
        truth = next((labels[key] for key in key_candidates if key in labels), "")
        best = detections[0] if detections else None
        rows.append({
            "file": str(path), "ground_truth": truth, "predicted": bool(detections),
            "max_confidence": best.confidence if best else 0.0,
            "best_class": CLASS_NAMES.get(best.class_id, str(best.class_id)) if best else "",
            "detections": json.dumps([
                {"class": CLASS_NAMES.get(det.class_id, str(det.class_id)), "confidence": det.confidence, "xyxy": det.xyxy}
                for det in detections
            ], ensure_ascii=False),
        })

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["file", "ground_truth", "predicted", "max_confidence", "best_class", "detections"])
        writer.writeheader()
        writer.writerows(rows)
    summary = {"model": str(args.model.resolve()), "classes": args.classes, "roi": roi, "confidence": args.confidence, "iou": args.iou, "images": len(rows), **metrics(rows)}
    args.output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
