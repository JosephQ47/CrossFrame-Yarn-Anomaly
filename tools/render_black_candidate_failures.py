"""Render original-resolution crops around remaining A-label failures."""

import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
import black_yarn_detector as D  # noqa: E402
import train_black_reviewer_a as A  # noqa: E402

AUDIT = ROOT / "outputs/black_yarn_candidate_audit"
OUT = AUDIT / "failure_crops.jpg"
IDS = (
    "DI2_Cam4_2026-07-18_10-14-15-405",
    "DI2_Cam4_2026-07-18_10-14-40-475",
    "DI2_Cam3_2026-07-18_10-18-03-405",
    "DI2_Cam6_2026-07-18_10-11-55-178",
)
SIZE = 420


def crop(image, center, boxes, candidates, selected, yolo, title):
    h, w = image.shape[:2]
    cx, cy = [int(round(v)) for v in center]
    x0 = min(max(cx-SIZE//2, 0), max(w-SIZE, 0))
    y0 = min(max(cy-SIZE//2, 0), max(h-SIZE, 0))
    result = image[y0:y0+SIZE, x0:x0+SIZE].copy()
    if result.shape[:2] != (SIZE, SIZE):
        result = cv2.copyMakeBorder(result, 0, SIZE-result.shape[0], 0,
                                    SIZE-result.shape[1], cv2.BORDER_CONSTANT)
    for box in boxes:
        cv2.rectangle(result, (int(box[0]-x0), int(box[1]-y0)),
                      (int(box[2]-x0), int(box[3]-y0)), (0, 220, 0), 3)
    for index, candidate in enumerate(candidates):
        x, y = [int(round(v)) for v in candidate["point"]]
        cv2.circle(result, (x-x0, y-y0), 5, (0, 220, 220), -1)
        cv2.putText(result, str(index+1), (x-x0+6, y-y0-6),
                    cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 220, 220), 1)
    if yolo:
        box = yolo["box"]
        x, y = int((box[0]+box[2])/2-x0), int((box[1]+box[3])/2-y0)
        cv2.drawMarker(result, (x, y), (255, 220, 0), cv2.MARKER_CROSS, 22, 3)
    if selected:
        x, y = [int(round(v)) for v in selected["point"]]
        cv2.circle(result, (x-x0, y-y0), 11, (0, 0, 255), 3)
    card = np.full((SIZE+36, SIZE, 3), (26, 34, 36), np.uint8)
    card[36:] = result
    cv2.putText(card, title[:39], (8, 24), cv2.FONT_HERSHEY_SIMPLEX,
                .55, (255, 255, 255), 1, cv2.LINE_AA)
    return card


def main():
    rows, _, _ = A.reviewed_rows()
    by_id = {r["id"]: r for r in rows}
    audit = {r["id"]: r for r in json.loads((AUDIT / "candidate_coverage.json")
                                           .read_text(encoding="utf-8"))["rows"]}
    fc = {r["id"]: r for r in json.loads((AUDIT / "fc_rerank.json")
                                        .read_text(encoding="utf-8"))["rows"]}
    yolo = json.loads((AUDIT / "yolo_fold_top1.json").read_text(encoding="utf-8"))["proposals"]
    cards = []
    for image_id in IDS:
        row = by_id[image_id]
        path = Path(row["file"])
        if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            raise RuntimeError(f"source changed: {image_id}")
        image = D.read_bgr(path)
        cand = audit[image_id]["candidates"]
        chosen = cand[fc[image_id]["fc_selected_rank"]-1]
        fold = sorted({r["cam"] for r in rows}).index(row["cam"])+1
        yp = yolo[f"{fold}:{image_id}"]
        if row["gt"]:
            box = row["gt"][0]
            gt_center = [(box[0]+box[2])/2, (box[1]+box[3])/2]
            cards.append(crop(image, gt_center, row["gt"], cand, chosen, yp,
                              row["cam"]+" GT / top10 / FC / YOLO"))
        cards.append(crop(image, chosen["point"], row["gt"], cand, chosen, yp,
                          row["cam"]+" selected "+image_id.split("_")[-1]))
    sheet = np.full((len(cards)//2*(SIZE+36), 2*SIZE, 3), (26, 34, 36), np.uint8)
    for i, card in enumerate(cards):
        sheet[(i//2)*(SIZE+36):(i//2+1)*(SIZE+36),
              (i%2)*SIZE:(i%2+1)*SIZE] = card
    ok, data = cv2.imencode(".jpg", sheet, [cv2.IMWRITE_JPEG_QUALITY, 94])
    if not ok:
        raise RuntimeError("cannot encode contact")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    data.tofile(str(OUT))
    print(OUT)


if __name__ == "__main__":
    main()
