"""Replay the archived C# selftest frames with the same ONNX P0 pipeline."""

import argparse
import csv
import hashlib
import json
import sys
from collections import deque
from pathlib import Path

import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import verify_onnx_pipeline as P  # noqa: E402


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(csv_path, image_dir, onnx_path):
    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("empty C# CSV")
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    buffers, thresholds = {}, {}
    comparisons = []
    for row in rows:
        image_path = image_dir / row["file"]
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        image = P.read_bgr(image_path)
        if image is None:
            raise ValueError(f"unreadable image: {image_path}")
        feat, y0, y1 = P.band_feat(session, image)
        cam = row["cam"]
        buf = buffers.setdefault(cam, deque(maxlen=8))
        ready = len(buf) == 8
        score, box = 0.0, None
        threshold = thresholds.get(cam)
        if ready:
            refs = list(buf)
            anomaly = P.anom_map(feat, refs)
            score, _, box = P.score_and_locate(anomaly, image.shape[:2], y0, y1)
            if threshold is None:
                loo = [P.score_and_locate(P.anom_map(refs[i], refs[:i] + refs[i+1:]),
                                          image.shape[:2], y0, y1)[0]
                       for i in range(8)]
                threshold = P.robust_thr(loo)
                thresholds[cam] = threshold
        buf.append(feat)
        expected_box = None
        if row["box"].strip():
            x, y, width, height = (int(v) for v in row["box"].split())
            expected_box = [x, y, x + width, y + height]
        score_error = abs(score - float(row["score"])) if ready else 0.0
        threshold_error = abs(threshold - float(row["threshold"])) if ready else 0.0
        record = {"file": row["file"], "image_sha256": digest(image_path),
                  "ready_match": ready == bool(int(row["ready"])),
                  "score_error": score_error, "threshold_error": threshold_error,
                  "alarm_match": (ready and score >= threshold) == bool(int(row["alarm"])),
                  "box_match": box == expected_box,
                  "python_box": box, "csharp_box": expected_box}
        comparisons.append(record)
    return {"scope": "archived Cam1 P0 C#/Python ONNX parity; not causal accuracy",
            "onnx_sha256": digest(onnx_path), "csharp_csv_sha256": digest(csv_path),
            "frames": len(comparisons), "ready_frames": sum(r["ready_match"] and
                                                    r["csharp_box"] is not None for r in comparisons),
            "max_score_error": max(r["score_error"] for r in comparisons),
            "max_threshold_error": max(r["threshold_error"] for r in comparisons),
            "ready_all_match": all(r["ready_match"] for r in comparisons),
            "alarm_all_match": all(r["alarm_match"] for r in comparisons),
            "box_all_match": all(r["box_match"] for r in comparisons),
            "comparisons": comparisons}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.csv, args.image_dir, args.onnx)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "comparisons"},
                     ensure_ascii=False, indent=2))
    if not all((result["ready_all_match"], result["alarm_all_match"],
                result["box_all_match"], result["max_score_error"] <= .01,
                result["max_threshold_error"] <= .01)):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
