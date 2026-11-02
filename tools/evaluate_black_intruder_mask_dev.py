"""Retrospective A-label audit of the existing LAB mask on the frozen rank head.

Reference choice uses prior reviewer-A normal labels, so this is an oracle
development diagnostic, not an online or independent evaluation.
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
OUT = ROOT / "outputs/black_roi_resolution_diag/mask_development_audit.json"


def image(row):
    path = Path(row["source_path"])
    if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
        raise RuntimeError(f"image changed: {row['frame_id']}")
    img = D.read_bgr(path)
    if img is None:
        raise RuntimeError(f"image unreadable: {path}")
    return img


def hit(point, boxes):
    return bool(point and any(x0 <= point[0] <= x1 and y0 <= point[1] <= y1
                              for x0, y0, x1, y1 in boxes))


def summarize(rows, key):
    positive = [r for r in rows if r["defect"]]
    negative = [r for r in rows if not r["defect"]]
    return {"positive": len(positive), "normal": len(negative),
            "point_hits": sum(r[key]["hit"] for r in positive),
            "alarm_and_point_hits": sum(r["alarm"] and r[key]["hit"] for r in positive),
            "false_alarms": sum(r["alarm"] for r in negative)}


def main():
    with REVIEW.open(encoding="utf-8-sig", newline="") as handle:
        review = {r["frame_id"]: r for r in csv.DictReader(handle)}
    index = {r["id"]: r for r in json.loads(S.INDEX.read_text(encoding="utf-8"))["data"]}
    bundle = json.loads(RANK.read_text(encoding="utf-8"))["methods"]["mil_rank"]
    prediction = {r["id"]: r for r in bundle["rows"]}
    folds = {f["test_cam"]: f for f in bundle["folds"]}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output = {"scope": "oracle prior-A-normal reference diagnostic; retrospective cache",
              "minimum_prior_normals": 3, "rows": []}
    for cam, fold in folds.items():
        checkpoint = torch.load(fold["checkpoint"], map_location=device, weights_only=False)
        model = S.Head("M4").to(device)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        prior_normal = []
        for row in sorted((r for r in review.values() if r["cam"] == cam),
                          key=lambda r: r["frame_id"]):
            frame_id = row["frame_id"]
            if frame_id not in prediction:  # uncertain frame, never a reference
                continue
            if len(prior_normal) >= 3:
                source = index[frame_id]
                img = image(row)
                x, _ = S.load_row(source, "M4", checkpoint["mean"], checkpoint["std"])
                with torch.no_grad():
                    prob = torch.sigmoid(model(x[None].to(device)))[0, 0].cpu().numpy()
                gh, gw = prob.shape
                current_lab = D.band_lab(img, source["y0"], source["y1"], gh, gw)
                ref_labs = [D.band_lab(image(ref), source["y0"], source["y1"], gh, gw)
                            for ref in prior_normal[:3]]
                mask = D.intruder_mask(current_lab, ref_labs, gh, gw)
                boxes = json.loads(row["box_or_centerline_json"] or "[]")
                variants = {}
                for name, valid in (("raw", None), ("masked", ~mask)):
                    area, point, _ = D.score_and_locate(prob, source["hw"], source["y0"],
                                                        source["y1"], cap=mask.size,
                                                        valid=valid)
                    variants[name] = {"area": area, "point": point,
                                      "hit": hit(point, boxes)}
                gt_grid = S.M1.box_grid(boxes, source["hw"], source["y0"], source["y1"],
                                        gh, gw).astype(bool)
                output["rows"].append({"id": frame_id, "cam": cam, "defect": bool(boxes),
                                       "alarm": prediction[frame_id]["a_alarm"],
                                       "reference_ids": [r["frame_id"] for r in prior_normal[:3]],
                                       "masked_patches": int(mask.sum()),
                                       "gt_patches": int(gt_grid.sum()),
                                       "gt_masked_patches": int((mask & gt_grid).sum()),
                                       **variants})
            if row["defect_status"] == "normal":
                prior_normal.append(row)
    output["summary"] = {name: summarize(output["rows"], name) for name in ("raw", "masked")}
    output["summary"]["eligible_frames"] = len(output["rows"])
    output["summary"]["changed_positive"] = [
        {"id": r["id"], "raw_hit": r["raw"]["hit"], "masked_hit": r["masked"]["hit"],
         "gt_masked_patches": r["gt_masked_patches"], "gt_patches": r["gt_patches"]}
        for r in output["rows"] if r["defect"] and r["raw"]["hit"] != r["masked"]["hit"]]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output["summary"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
