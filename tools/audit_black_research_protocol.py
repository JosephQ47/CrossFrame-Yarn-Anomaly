"""Read-only model/data audit; no training or label changes.

Regenerates candidates from archived fold heads and measures supervision and
coordinate issues. All measurements concern the repeatedly used July dev set.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import eval_black_rank_candidate_fc as E
import fewshot_semantic_heads as S
import train_black_reviewer_a as A


def main():
    rows, excluded, digest = A.reviewed_rows()
    rank = json.loads(E.RANK.read_text(encoding="utf-8"))["methods"]["mil_rank"]
    yolo = json.loads(E.YOLO_ALL.read_text(encoding="utf-8"))
    if yolo["A_label_sha256"] != digest:
        raise RuntimeError("YOLO cache label mismatch")
    torch.set_num_threads(4)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    result = {"scope": "July retrospective development audit; not independent accuracy",
              "A_label_sha256": digest, "excluded": excluded,
              "by_camera": {}, "folds": [], "archived_metrics": {}}
    for cam in sorted({r["cam"] for r in rows}):
        group = [r for r in rows if r["cam"] == cam]
        result["by_camera"][cam] = {"positive": sum(bool(r["gt"]) for r in group),
                                     "normal": sum(not r["gt"] for r in group)}
    shapes = []
    for row in rows:
        x, _ = S.array_for(row, "M4")
        width = row["hw"][1]
        starts = list(range(0, max(1, width-S.D.TILE_W+1), S.D.TILE_W-S.D.OVERLAP))
        if starts[-1]+S.D.TILE_W < width:
            starts.append(width-S.D.TILE_W)
        item = {"hw": row["hw"], "grid": list(x.shape),
                "patch_original_xy": [width/x.shape[1], (row["y1"]-row["y0"])/x.shape[0]],
                "tile_offsets_mod14": sorted(set(v % 14 for v in starts))}
        if item not in shapes:
            shapes.append(item)
    result["feature_geometry"] = shapes
    for fi, fold in enumerate(rank["folds"], 1):
        path = ROOT / Path(fold["checkpoint"])
        if hashlib.sha256(path.read_bytes()).hexdigest() != fold["checkpoint_sha256"]:
            raise RuntimeError("checkpoint mismatch")
        ck = torch.load(path, map_location=device, weights_only=False)
        model = S.Head("M4").to(device)
        model.load_state_dict(ck["state_dict"])
        model.eval()
        report = {"test_cam": fold["test_cam"], "val_cam": fold["val_cam"], "groups": {}}
        for split, key in (("train", "train_ids"), ("val", "val_ids"), ("test", "test_ids")):
            group = [r for r in rows if r["id"] in ck[key]]
            if {r["id"] for r in group} != set(ck[key]):
                raise RuntimeError("fold id mismatch")
            stats = {"frames": len(group), "positive_frames": 0, "normal_frames": 0,
                     "positive_without_candidate": [], "ignored_candidates": 0,
                     "yolo_coincident_metadata_lost": 0,
                     "negative_center_crop_intersects_gt_box": 0,
                     "negative_center_candidates": 0,
                     "no_positive_loss_branch_ignored": 0}
            for row in group:
                yp = yolo["proposals"][f"{fi}:{row['id']}"]
                item = E.prepare(row, model, ck, device, yp, source_score=True)
                stats["positive_frames" if item["defect"] else "normal_frames"] += 1
                if item["defect"] and not item["hit"].any():
                    stats["positive_without_candidate"].append(row["id"])
                stats["ignored_candidates"] += int((~item["usable"]).sum())
                if not item["hit"].any():
                    stats["no_positive_loss_branch_ignored"] += int((~item["usable"]).sum())
                if yp and (yp.get("box") or yp.get("detections")) and not any(
                        p.get("source") == "yolo_top1" for p in item["peaks"]):
                    stats["yolo_coincident_metadata_lost"] += 1
                for peak, usable, hit in zip(item["peaks"], item["usable"], item["hit"]):
                    if usable and not hit:
                        stats["negative_center_candidates"] += 1
                        cx, cy = peak["point"]
                        if any(min(cx+96, b[2]) > max(cx-96, b[0]) and
                               min(cy+96, b[3]) > max(cy-96, b[1]) for b in row["gt"]):
                            stats["negative_center_crop_intersects_gt_box"] += 1
            report["groups"][split] = stats
        result["folds"].append(report)
        print(f"audited fold {fi}/6", flush=True)
    for name in ("fc_rerank", "cnn_rerank", "fc_yolo_union", "fc_yolo_trained_union",
                 "fc_yolo_source_score", "rgb_fusion", "pairwise_rank"):
        path = ROOT / f"outputs/black_yarn_candidate_audit/{name}.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        result["archived_metrics"][name] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                             "summary": data.get("summary", data.get("summaries"))}
        prefix = "cnn_" if name == "cnn_rerank" else "fc_" if name.startswith("fc_") else ""
        if name != "rgb_fusion":
            pos = [r for r in data["rows"] if r["defect"]]
            neg = [r for r in data["rows"] if not r["defect"]]
            result["archived_metrics"][name]["recomputed"] = {
                "positive": len(pos), "normal": len(neg),
                "image_alarm_hits": sum(r[prefix+"alarm"] for r in pos),
                "alarm_and_point_hits": sum(r[prefix+"alarm"] and r[prefix+"point_hit"] for r in pos),
                "false_alarms": sum(r[prefix+"alarm"] for r in neg)}
    result["input_sha256"] = {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (E.RANK, E.YOLO_ALL, S.INDEX, A.LOCK,
                  ROOT / "tools/fewshot_semantic_heads.py",
                  ROOT / "tools/eval_black_rank_candidate_fc.py",
                  ROOT / "tools/eval_black_candidate_pairwise.py",
                  ROOT / "tools/eval_black_rgb_candidate_fusion.py",
                  ROOT / "src/black_yarn_detector.py")}
    path = ROOT / "data/black_research_protocol_audit_2026-11-05.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
