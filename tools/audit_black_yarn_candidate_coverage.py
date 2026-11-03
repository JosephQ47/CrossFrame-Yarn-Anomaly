"""Audit whether an out-of-fold DINO rank heatmap proposes the A defect box.

This is a retrospective single-event development audit. No candidate comes
from the ground-truth box at inference; boxes are used only for hit scoring.
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import black_hand_detail_server as REVIEW  # noqa: E402
import fewshot_semantic_heads as S  # noqa: E402
import train_black_reviewer_a as A  # noqa: E402

RANK = ROOT / "outputs/black_reviewer_a_rank/results.json"
OUT = ROOT / "outputs/black_yarn_candidate_audit/candidate_coverage.json"
KS = (1, 3, 5, 10)
NMS_RADIUS = 2


def candidates(prob, row, count=max(KS)):
    """Greedy score peaks, suppressing a fixed 2-patch radius."""
    active = prob.copy()
    gh, gw = active.shape
    h, w = row["hw"]
    points = []
    for _ in range(count):
        i, j = np.unravel_index(np.argmax(active), active.shape)
        if not np.isfinite(active[i, j]):
            break
        points.append({"patch": [int(i), int(j)],
                       "point": [float((j + .5) * w / gw),
                                 float(row["y0"] + (i + .5) * (row["y1"] - row["y0"]) / gh)],
                       "score": float(active[i, j])})
        active[max(0, i-NMS_RADIUS):min(gh, i+NMS_RADIUS+1),
               max(0, j-NMS_RADIUS):min(gw, j+NMS_RADIUS+1)] = -np.inf
    return points


def hit(point, boxes):
    x, y = point
    return any(x0 <= x <= x1 and y0 <= y <= y1 for x0, y0, x1, y1 in boxes)


def main():
    rows, excluded, digest = A.reviewed_rows()
    detail_source = REVIEW.roster()
    detail = REVIEW.current_reviews(detail_source)
    if len(detail) != 7:
        raise RuntimeError(f"expected seven saved optional detail reviews, got {len(detail)}")
    detail_audit = []
    for frame_id, entry in detail.items():
        source = detail_source[frame_id]
        if entry["source_sha256"] != source["sha256"]:
            raise RuntimeError(f"detail source hash mismatch: {frame_id}")
        cleaned = REVIEW.validate(entry, source)
        if cleaned["yarn_centerline"] != entry["yarn_centerline"]:
            raise RuntimeError(f"detail coordinates changed: {frame_id}")
        boxes = json.loads(source["box_or_centerline_json"] or "[]")
        line = entry["yarn_centerline"]
        detail_audit.append({"id": frame_id, "A_status": source["defect_status"],
                             "line_points": len(line), "points_inside_A_box":
                             sum(hit(point, boxes) for point in line),
                             "note": entry["notes"]})
    bundle = json.loads(RANK.read_text(encoding="utf-8"))["methods"]["mil_rank"]
    by_cam = {f["test_cam"]: f for f in bundle["folds"]}
    prediction = {r["id"]: r for r in bundle["rows"]}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    records = []
    for row in rows:
        fold = by_cam[row["cam"]]
        checkpoint_path = Path(fold["checkpoint"])
        if hashlib.sha256(checkpoint_path.read_bytes()).hexdigest() != fold["checkpoint_sha256"]:
            raise RuntimeError(f"checkpoint hash mismatch: {row['cam']}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if row["id"] not in checkpoint["test_ids"]:
            raise RuntimeError(f"not out-of-fold: {row['id']}")
        model = S.Head("M4").to(device)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        x, _ = S.load_row(row, "M4", checkpoint["mean"], checkpoint["std"])
        with torch.no_grad():
            prob = torch.sigmoid(model(x[None].to(device)))[0, 0].cpu().numpy()
        peaks = candidates(prob, row)
        records.append({"id": row["id"], "cam": row["cam"],
                        "defect": bool(row["gt"]),
                        "alarm": bool(prediction[row["id"]]["a_alarm"]),
                        "candidate_hits": [hit(p["point"], row["gt"]) for p in peaks],
                        "candidates": peaks})
    pos = [r for r in records if r["defect"]]
    summary = {"positive": len(pos), "normal": len(records)-len(pos),
               "positive_candidate_coverage": {str(k): sum(any(r["candidate_hits"][:k]) for r in pos)
                                                for k in KS},
               "alarm_and_candidate_coverage": {str(k): sum(r["alarm"] and any(r["candidate_hits"][:k]) for r in pos)
                                                for k in KS},
               "missed_at_10": [r["id"] for r in pos if not any(r["candidate_hits"][:10])],
               "detail_line_count": sum(bool(r["line_points"]) for r in detail_audit),
               "detail_within_A_box": sum(r["points_inside_A_box"] for r in detail_audit),
               "detail_total_points": sum(r["line_points"] for r in detail_audit)}
    output = {"scope": "retrospective, same July fault event; out-of-fold head, noncausal feature cache",
              "A_label_sha256": digest, "excluded_A_uncertain": excluded,
              "nms_radius_patches": NMS_RADIUS, "summary": summary,
              "detail_audit": detail_audit, "rows": records}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
