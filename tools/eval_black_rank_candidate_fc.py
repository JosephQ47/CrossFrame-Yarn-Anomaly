"""Rerank rank-head candidate peaks with a train-camera-only FC classifier.

Six camera-held-out folds, validation-camera normal P95 threshold. This is a
retrospective July single-event development experiment, not deployable recall.
The first-stage DINO cache uses noncausal same-camera leave-one-out reference.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import audit_black_yarn_candidate_coverage as C  # noqa: E402
import fewshot_semantic_heads as S  # noqa: E402
import train_black_reviewer_a as A  # noqa: E402

RANK = ROOT / "outputs/black_reviewer_a_rank/results.json"
OUT = ROOT / "outputs/black_yarn_candidate_audit/fc_rerank.json"
UNION_OUT = ROOT / "outputs/black_yarn_candidate_audit/fc_yolo_union.json"
TRAINED_UNION_OUT = ROOT / "outputs/black_yarn_candidate_audit/fc_yolo_trained_union.json"
YOLO = ROOT / "outputs/black_fusion_diagnostic/yolo_black_results.json"
YOLO_ALL = ROOT / "outputs/black_yarn_candidate_audit/yolo_fold_top1.json"
SEED = 20261103
K = 10


def feature(m4, prob, patch):
    i, j = patch
    h, w, _ = m4.shape
    local = m4[max(0, i-2):min(h, i+3), max(0, j-2):min(w, j+3)]
    return np.concatenate([m4[i, j], local.mean(axis=(0, 1)),
                           np.array([float(prob[i, j]), float(i/h), float(j/w)], np.float32)])


def near_box(point, boxes, dx, dy):
    x, y = point
    return any(x0-dx <= x <= x1+dx and y0-dy <= y <= y1+dy
               for x0, y0, x1, y1 in boxes)


def prepare(row, model, checkpoint, device, yolo=None):
    m4, _ = S.array_for(row, "M4")
    x = np.clip((m4 - checkpoint["mean"]) / checkpoint["std"], -5, 5)
    with torch.no_grad():
        prob = torch.sigmoid(model(torch.from_numpy(x.transpose(2, 0, 1).copy())[None]
                                   .to(device)))[0, 0].cpu().numpy()
    peaks = C.candidates(prob, row, K)
    gh, gw = prob.shape
    if yolo and (yolo.get("box") or yolo.get("detections")):
        box = yolo["box"] if "box" in yolo else yolo["detections"][0]["box"]
        cx, cy = (box[0]+box[2])/2, (box[1]+box[3])/2
        i = int(np.clip((cy-row["y0"]) / (row["y1"]-row["y0"]) * gh, 0, gh-1))
        j = int(np.clip(cx / row["hw"][1] * gw, 0, gw-1))
        if not any(p["patch"] == [i, j] for p in peaks):
            peaks.append({"patch": [i, j],
                          "point": [float((j+.5)*row["hw"][1]/gw),
                                    float(row["y0"]+(i+.5)*(row["y1"]-row["y0"])/gh)],
                          "score": float(prob[i, j]), "source": "yolo_top1"})
    dx = row["hw"][1] / gw
    dy = (row["y1"] - row["y0"]) / gh
    return {"id": row["id"], "cam": row["cam"], "defect": bool(row["gt"]),
            "x": np.stack([feature(x, prob, p["patch"]) for p in peaks]),
            "hit": np.array([C.hit(p["point"], row["gt"]) for p in peaks], bool),
            "usable": np.array([C.hit(p["point"], row["gt"]) or
                                not near_box(p["point"], row["gt"], dx, dy)
                                for p in peaks], bool),
            "peaks": peaks}


def main(union_yolo=False, trained_union=False):
    if union_yolo and trained_union:
        raise ValueError("choose one YOLO union mode")
    rows, excluded, digest = A.reviewed_rows()
    bundle = json.loads(RANK.read_text(encoding="utf-8"))["methods"]["mil_rank"]
    by_cam = {f["test_cam"]: f for f in bundle["folds"]}
    rank_pred = {r["id"]: r for r in bundle["rows"]}
    yolo_rows = {}
    if union_yolo:
        yolo_rows = {(r["fold"], r["split"], r["id"]): r for r in
                     json.loads(YOLO.read_text(encoding="utf-8"))["rows"]}
    yolo_all = {}
    if trained_union:
        yolo_all = json.loads(YOLO_ALL.read_text(encoding="utf-8"))["proposals"]
    cams = sorted(by_cam)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    report = []
    folds = []
    for index, test_cam in enumerate(cams):
        val_cam = cams[(index + 1) % len(cams)]
        fold = by_cam[test_cam]
        checkpoint_path = Path(fold["checkpoint"])
        if hashlib.sha256(checkpoint_path.read_bytes()).hexdigest() != fold["checkpoint_sha256"]:
            raise RuntimeError(f"checkpoint hash mismatch: {test_cam}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model = S.Head("M4").to(device)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        tr = [r for r in rows if r["cam"] not in (test_cam, val_cam)]
        va = [r for r in rows if r["cam"] == val_cam]
        te = [r for r in rows if r["cam"] == test_cam]
        if union_yolo and any((index+1, split, r["id"]) not in yolo_rows
                              for split, group in (("val", va), ("test", te)) for r in group):
            raise RuntimeError(f"missing YOLO candidate archive: {test_cam}")
        if trained_union and any(f"{index+1}:{r['id']}" not in yolo_all
                                 for r in tr+va+te):
            raise RuntimeError(f"missing all-fold YOLO candidate: {test_cam}")
        if set(checkpoint["train_ids"]) != {r["id"] for r in tr} or \
           set(checkpoint["val_ids"]) != {r["id"] for r in va} or \
           set(checkpoint["test_ids"]) != {r["id"] for r in te}:
            raise RuntimeError(f"split mismatch: {test_cam}")
        train = [prepare(r, model, checkpoint, device,
                         yolo_all.get(f"{index+1}:{r['id']}") if trained_union else None)
                 for r in tr]
        val = [prepare(r, model, checkpoint, device,
                       yolo_all.get(f"{index+1}:{r['id']}") if trained_union else
                       yolo_rows.get((index+1, "val", r["id"])) if union_yolo else None)
               for r in va]
        test = [prepare(r, model, checkpoint, device,
                        yolo_all.get(f"{index+1}:{r['id']}") if trained_union else
                        yolo_rows.get((index+1, "test", r["id"])) if union_yolo else None)
                for r in te]
        xx = np.concatenate([r["x"][r["usable"]] for r in train])
        yy = np.concatenate([r["hit"][r["usable"]] for r in train]).astype(int)
        if len(np.unique(yy)) != 2:
            raise RuntimeError(f"training candidates lack one class: {test_cam}")
        scaler = StandardScaler().fit(xx)
        clf = LogisticRegression(C=.1, class_weight="balanced", max_iter=2000,
                                 random_state=SEED).fit(scaler.transform(xx), yy)

        def score(item):
            p = clf.predict_proba(scaler.transform(item["x"]))[:, 1]
            selected = int(np.argmax(p))
            return float(p[selected]), selected

        threshold = float(np.quantile([score(r)[0] for r in val if not r["defect"]], .95))
        for item in test:
            value, selected = score(item)
            old = rank_pred[item["id"]]
            report.append({"id": item["id"], "cam": test_cam,
                           "defect": item["defect"], "fc_score": value,
                           "fc_threshold": threshold, "fc_alarm": value >= threshold,
                           "fc_selected_rank": selected + 1,
                           "fc_point_hit": bool(item["hit"][selected]),
                           "rank_alarm": bool(old["a_alarm"]),
                           "rank_point_hit": bool(old["a_point_hit"]),
                           "candidate_reachable": bool(item["hit"].any())})
        folds.append({"test_cam": test_cam, "val_cam": val_cam,
                      "train_candidates": int(len(yy)),
                      "train_positive_candidates": int(yy.sum()),
                      "fc_threshold": threshold})
        print(f"FC fold {index+1}/6 {test_cam}: train {len(yy)} candidates, "
              f"positive {yy.sum()}", flush=True)
    pos = [r for r in report if r["defect"]]
    neg = [r for r in report if not r["defect"]]
    summary = {"positive": len(pos), "normal": len(neg),
               "rank_alarm_and_point": sum(r["rank_alarm"] and r["rank_point_hit"] for r in pos),
               "rank_false_alarms": sum(r["rank_alarm"] for r in neg),
               "candidate_reachable": sum(r["candidate_reachable"] for r in pos),
               "fc_alarm_and_point": sum(r["fc_alarm"] and r["fc_point_hit"] for r in pos),
               "fc_point_hits": sum(r["fc_point_hit"] for r in pos),
               "fc_false_alarms": sum(r["fc_alarm"] for r in neg),
               "fc_selected_rank_gt1": sum(r["fc_selected_rank"] > 1 for r in report),
               "new_hits": [r["id"] for r in pos if r["fc_alarm"] and r["fc_point_hit"]
                            and not (r["rank_alarm"] and r["rank_point_hit"])],
               "lost_hits": [r["id"] for r in pos if r["rank_alarm"] and r["rank_point_hit"]
                             and not (r["fc_alarm"] and r["fc_point_hit"])]}
    target = TRAINED_UNION_OUT if trained_union else UNION_OUT if union_yolo else OUT
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({"scope": "retrospective July one-event; noncausal DINO cache",
                               "A_label_sha256": digest, "excluded_uncertain": excluded,
                               "candidate_top_k": K, "nms_radius_patches": C.NMS_RADIUS,
                               "yolo_top1_union": union_yolo or trained_union,
                               "verifier_trained_with_yolo_candidates": trained_union,
                               "yolo_training_labels": "archived older labels; test camera held out" if (union_yolo or trained_union) else None,
                               "classifier": "L2 logistic FC on rank peak M4 and 5x5 context; C=0.1",
                               "summary": summary, "folds": folds, "rows": report},
                              ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--union-yolo", action="store_true")
    parser.add_argument("--union-yolo-trained", action="store_true")
    args = parser.parse_args()
    main(args.union_yolo, args.union_yolo_trained)
