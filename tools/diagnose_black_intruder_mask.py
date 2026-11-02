"""Check whether the existing LAB intrusion mask spares A black-yarn boxes.

Fixed reference normals precede the two target images by filename time. This
is a targeted diagnostic, not a full chronological stream replay.
"""

import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
import black_yarn_detector as D  # noqa: E402
import fewshot_semantic_heads as S  # noqa: E402

REVIEW = ROOT / "outputs/black_july_blind_review/july_reviewer_A_locked.csv"
RANK = ROOT / "outputs/black_reviewer_a_rank/results.json"
OUT = ROOT / "outputs/black_roi_resolution_diag/intruder_mask.json"
TARGETS = (
    "DI2_Cam3_2026-07-18_10-14-15-409",
    "DI2_Cam4_2026-07-18_10-14-15-405",
)


def image(row):
    path = Path(row["source_path"])
    if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
        raise RuntimeError(f"changed image: {row['frame_id']}")
    img = D.read_bgr(path)
    if img is None:
        raise RuntimeError(f"unreadable image: {path}")
    return img


def mask_at(mask, row, point):
    x, y = point
    gh, gw = mask.shape
    h, w = row["hw"]
    ix = min(max(int(x / w * gw), 0), gw - 1)
    iy = min(max(int((y - row["y0"]) / (row["y1"] - row["y0"]) * gh), 0), gh - 1)
    return bool(mask[iy, ix])


def main():
    with REVIEW.open(encoding="utf-8-sig", newline="") as handle:
        review = {r["frame_id"]: r for r in csv.DictReader(handle)}
    index = {r["id"]: r for r in json.loads(S.INDEX.read_text(encoding="utf-8"))["data"]}
    rank_bundle = json.loads(RANK.read_text(encoding="utf-8"))["methods"]["mil_rank"]
    pred = {r["id"]: r for r in rank_bundle["rows"]}
    fold_by_cam = {f["test_cam"]: f for f in rank_bundle["folds"]}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output = {"mask_rule": {"delta_e": D.MASK_DE_T, "area_frac": D.MASK_AREA_FRAC,
                             "dilation": D.MASK_DIL}, "rows": []}
    for target_id in TARGETS:
        row = index[target_id]
        target = review[target_id]
        cam = row["cam"]
        refs = [r for r in review.values()
                if r["cam"] == cam and r["defect_status"] == "normal"
                and r["frame_id"] < target_id]
        if len(refs) != 3:
            raise RuntimeError(f"expected three prior normal references: {cam}")
        img = image(target)
        gh, gw = S.array_for(row, "M4")[0].shape[:2]
        current_lab = D.band_lab(img, row["y0"], row["y1"], gh, gw)
        ref_labs = [D.band_lab(image(r), row["y0"], row["y1"], gh, gw) for r in refs]
        mask = D.intruder_mask(current_lab, ref_labs, gh, gw)
        original_dilation = D.MASK_DIL
        try:
            D.MASK_DIL = 0
            mask_no_dilation = D.intruder_mask(current_lab, ref_labs, gh, gw)
        finally:
            D.MASK_DIL = original_dilation
        fold = fold_by_cam[cam]
        checkpoint = torch.load(fold["checkpoint"], map_location=device, weights_only=False)
        model = S.Head("M4").to(device)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        x, _ = S.load_row(row, "M4", checkpoint["mean"], checkpoint["std"])
        with torch.no_grad():
            prob = torch.sigmoid(model(x[None].to(device)))[0, 0].cpu().numpy()
        boxes = json.loads(target["box_or_centerline_json"])
        positive = S.M1.box_grid(boxes, row["hw"], row["y0"], row["y1"], gh, gw).astype(bool)
        wrong = pred[target_id]["point"]
        item = {"id": target_id, "reference_ids": [r["frame_id"] for r in refs],
                "masked_patches": int(mask.sum()), "total_patches": int(mask.size),
                "wrong_point_masked": mask_at(mask, row, wrong),
                "gt_patches": int(positive.sum()),
                "gt_masked_patches": int((mask & positive).sum()),
                "no_dilation_wrong_point_masked": mask_at(mask_no_dilation, row, wrong),
                "no_dilation_gt_masked_patches": int((mask_no_dilation & positive).sum()),
                "gt_center_masked": mask_at(mask, row, [(boxes[0][0] + boxes[0][2]) / 2,
                                                         (boxes[0][1] + boxes[0][3]) / 2])}
        item["localization"] = {}
        for name, valid, cap in (("raw_cap55", None, 55),
                                 ("raw_uncapped", None, mask.size),
                                 ("masked_cap55", ~mask, 55),
                                 ("masked_uncapped", ~mask, mask.size),
                                 ("masked_no_dilation_uncapped", ~mask_no_dilation, mask.size)):
            area, point, _ = D.score_and_locate(prob, row["hw"], row["y0"], row["y1"],
                                                 cap=cap, valid=valid)
            hit = bool(point and any(b[0] <= point[0] <= b[2] and
                                     b[1] <= point[1] <= b[3] for b in boxes))
            item["localization"][name] = {"area": area, "point": point, "hit": hit}
        output["rows"].append(item)
        print(item, flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
