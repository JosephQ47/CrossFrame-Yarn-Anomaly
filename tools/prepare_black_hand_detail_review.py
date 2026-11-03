"""Prepare image-first review of the seven A-labeled hand frames.

Red is the existing LAB proposal, green is the immutable A defect box. The
proposal is untrusted and must not become training truth without review.
"""

import csv
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
import fewshot_semantic_heads as S  # noqa: E402

REVIEW = ROOT / "outputs/black_july_blind_review/july_reviewer_A_locked.csv"
OUT = ROOT / "outputs/black_hand_detail_review"


def read_image(row):
    path = Path(row["source_path"])
    if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
        raise RuntimeError(f"source image changed: {row['frame_id']}")
    img = D.read_bgr(path)
    if img is None:
        raise RuntimeError(f"cannot read: {path}")
    return img


def main():
    with REVIEW.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    hand = [r for r in rows if r["hand_or_occlusion"] == "hand"]
    if len(hand) != 7:
        raise RuntimeError(f"expected 7 hand frames, got {len(hand)}")
    index = {r["id"]: r for r in json.loads(S.INDEX.read_text(encoding="utf-8"))["data"]}
    OUT.mkdir(parents=True, exist_ok=True)
    manifest, cards = [], []
    for row in hand:
        source = index[row["frame_id"]]
        img = read_image(row)
        h, w = img.shape[:2]
        prior = [r for r in rows if r["cam"] == row["cam"]
                 and r["frame_id"] < row["frame_id"] and r["defect_status"] == "normal"]
        refs = prior[:3] if len(prior) >= 3 else []
        mask = None
        if refs:
            gh, gw = S.array_for(source, "M4")[0].shape[:2]
            labs = [D.band_lab(read_image(r), source["y0"], source["y1"], gh, gw)
                    for r in refs]
            lab_cur = D.band_lab(img, source["y0"], source["y1"], gh, gw)
            mask = D.intruder_mask(lab_cur, labs, gh, gw)
        overlay = img.copy()
        boxes = json.loads(row["box_or_centerline_json"] or "[]")
        overlap = None
        if mask is not None:
            large = cv2.resize(mask.astype(np.uint8),
                               (w, source["y1"] - source["y0"]),
                               interpolation=cv2.INTER_NEAREST).astype(bool)
            band = overlay[source["y0"]:source["y1"]]
            band[large] = (0, 0, 255)
            overlay = cv2.addWeighted(img, .68, overlay, .32, 0)
            gt = S.M1.box_grid(boxes, source["hw"], source["y0"], source["y1"],
                               *mask.shape).astype(bool)
            overlap = {"gt_patches": int(gt.sum()),
                       "proposal_over_gt": int((gt & mask).sum()),
                       "proposal_patches": int(mask.sum())}
        for x0, y0, x1, y1 in boxes:
            cv2.rectangle(overlay, (int(x0), int(y0)), (int(x1), int(y1)),
                          (0, 255, 0), 12)
        title = f"{row['cam']} {row['frame_id'].split('_')[-1]}  {row['defect_status']}"
        thumb = cv2.resize(overlay, (960, round(h * 960 / w)))
        card = np.full((thumb.shape[0] + 58, 960, 3), (34, 28, 23), np.uint8)
        card[58:] = thumb
        cv2.putText(card, title, (14, 38), cv2.FONT_HERSHEY_SIMPLEX,
                    .72, (255, 255, 255), 2, cv2.LINE_AA)
        cards.append(card)
        manifest.append({"frame_id": row["frame_id"], "cam": row["cam"],
                         "status": row["defect_status"], "sha256": row["sha256"],
                         "prior_normal_ids": [r["frame_id"] for r in refs],
                         "proposal_available": mask is not None,
                         "overlap": overlap})
    ch = max(c.shape[0] for c in cards)
    sheet = np.full((ch * 4, 1920, 3), (34, 28, 23), np.uint8)
    for i, card in enumerate(cards):
        sheet[(i // 2) * ch:(i // 2) * ch + card.shape[0],
              (i % 2) * 960:(i % 2 + 1) * 960] = card
    encoded, data = cv2.imencode(".jpg", sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not encoded:
        raise RuntimeError("cannot encode contact sheet")
    data.tofile(str(OUT / "hand_review_contact.jpg"))
    (OUT / "manifest.json").write_text(json.dumps({"scope": "draft visual review",
                                                     "items": manifest}, ensure_ascii=False,
                                                    indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"items": len(manifest),
                      "with_proposals": sum(r["proposal_available"] for r in manifest),
                      "without_proposal": [r["frame_id"] for r in manifest
                                           if not r["proposal_available"]]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
