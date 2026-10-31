"""Train frozen-DINOv2 M4 heads with the locked, single-reviewer July labels.

This is a retrospective development experiment on one fault event. The two
uncertain hand-occluded frames are excluded from training, validation and test.
"""

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import fewshot_semantic_heads as S  # noqa: E402
import train_black_box_mil as MIL  # noqa: E402

LOCK = ROOT / "outputs/black_july_blind_review/july_reviewer_A_locked.csv"
MANIFEST = ROOT / "outputs/black_july_blind_review/july_reviewer_A_lock.json"
ARCHIVE = ROOT / "data/black_fewshot_semantic_2026-10-25/M4_results.json"
DEFAULT_OUT = ROOT / "outputs/black_reviewer_a_supervised"
SEED = 20261024


def reviewed_rows():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if manifest["reviewer"] != "A" or manifest["rows"] != 69:
        raise RuntimeError("expected the completed, locked A review")
    digest = hashlib.sha256(LOCK.read_bytes()).hexdigest()
    if digest != manifest["csv_sha256"]:
        raise RuntimeError("A review lock hash mismatch")
    with LOCK.open(newline="", encoding="utf-8-sig") as handle:
        review = list(csv.DictReader(handle))
    index = json.loads(S.INDEX.read_text(encoding="utf-8"))["data"]
    by_id = {r["frame_id"]: r for r in review}
    if len(review) != 69 or set(by_id) != {r["id"] for r in index}:
        raise RuntimeError("review/index IDs differ")
    rows, excluded = [], []
    for old in index:
        r = by_id[old["id"]]
        if old["sha256"] != r["sha256"] or r["sha256"] != manifest["source_image_sha256"][old["id"]]:
            raise RuntimeError(f"image hash mismatch: {old['id']}")
        if not (S.CACHE / (old["id"] + ".npz")).is_file():
            raise RuntimeError(f"missing frozen feature cache: {old['id']}")
        status = r["defect_status"]
        if status == "uncertain":
            excluded.append(old["id"])
            continue
        if status not in ("normal", "black_yarn"):
            raise RuntimeError(f"unsupported label: {old['id']} {status}")
        boxes = json.loads(r["box_or_centerline_json"] or "[]")
        if status == "black_yarn" and not boxes or status == "normal" and boxes:
            raise RuntimeError(f"inconsistent boxes: {old['id']}")
        clean = []
        h, w = old["hw"]
        for b in boxes:
            if len(b) != 4 or not all(isinstance(v, (int, float)) for v in b):
                raise RuntimeError(f"invalid box: {old['id']}")
            x0, y0, x1, y1 = b
            if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
                raise RuntimeError(f"out-of-image box: {old['id']} {b}")
            clean.append([float(v) for v in b])
        rows.append({**old, "gt": clean, "review_status": status})
    if len(rows) != 67 or sum(bool(r["gt"]) for r in rows) != 19 or len(excluded) != 2:
        raise RuntimeError("unexpected A label counts")
    return rows, excluded, digest


def load(row, mean, std):
    x, _ = S.array_for(row, "M4")
    y = S.M1.box_grid(row["gt"], row["hw"], row["y0"], row["y1"], *x.shape[:2])
    x = np.clip((x - mean) / std, -5, 5).transpose(2, 0, 1).copy()
    return torch.from_numpy(x), torch.from_numpy(y[None])


def loss_for(logits, y, row, method):
    if method == "mil":
        return MIL.box_mil_loss(logits, row)
    if method == "mil_hard":
        grid = logits[0, 0]
        if row["gt"]:
            boxes, far_normal = MIL.box_masks(row, *grid.shape, grid.device)
            positive = torch.stack([
                F.softplus(-grid[mask].topk(max(1, math.ceil(mask.sum().item() * .1))).values.mean())
                for mask in boxes]).mean()
            negatives = grid[far_normal]
        else:
            positive = None
            negatives = grid.flatten()
        hard_k = max(1, math.ceil(negatives.numel() * .01))
        hard_negative = F.softplus(negatives.topk(hard_k).values).mean()
        return hard_negative if positive is None else positive + hard_negative
    if method == "mil_normalhard":
        if row["gt"]:
            return MIL.box_mil_loss(logits, row)
        negatives = logits.flatten()
        hard_k = max(1, math.ceil(negatives.numel() * .01))
        return F.softplus(negatives.topk(hard_k).values).mean()
    if method == "mil_rank":
        if not row["gt"]:
            negatives = logits.flatten()
            hard_k = max(1, math.ceil(negatives.numel() * .01))
            return F.softplus(negatives.topk(hard_k).values).mean()
        grid = logits[0, 0]
        boxes, far_normal = MIL.box_masks(row, *grid.shape, grid.device)
        positive = torch.stack([
            grid[mask].topk(max(1, math.ceil(mask.sum().item() * .1))).values.mean()
            for mask in boxes]).mean()
        far = grid[far_normal]
        far_top = far.topk(max(1, math.ceil(far.numel() * .01))).values.mean()
        return (MIL.box_mil_loss(logits, row)
                + F.softplus(far_top - positive))
    positives = float(y.sum())
    weight = min(max((y.numel() - positives) / max(positives, 1), 1), 100)
    return F.binary_cross_entropy_with_logits(logits, y,
                                              pos_weight=torch.tensor(weight, device=y.device))


@torch.no_grad()
def predict(model, rows, mean, std, device):
    model.eval()
    output = []
    for row in rows:
        x, _ = load(row, mean, std)
        prob = torch.sigmoid(model(x[None].to(device)))[0, 0].cpu().numpy()
        k = max(1, round(prob.size * .01))
        score = float(np.partition(prob.ravel(), -k)[-k:].mean())
        _, point, box = S.D.score_and_locate(prob, row["hw"], row["y0"], row["y1"])
        point_hit = bool(point and any(x0 <= point[0] <= x1 and y0 <= point[1] <= y1
                                       for x0, y0, x1, y1 in row["gt"]))
        output.append({"id": row["id"], "cam": row["cam"], "defect": bool(row["gt"]),
                       "score": score, "point": point, "pred_box": box,
                       "point_hit": point_hit})
    return output


def metrics(rows, name):
    pos = [r for r in rows if r["defect"]]
    neg = [r for r in rows if not r["defect"]]
    return {"alarm_hits": sum(r[f"{name}_alarm"] for r in pos),
            "alarm_and_point_hits": sum(r[f"{name}_alarm"] and r[f"{name}_point_hit"] for r in pos),
            "point_hits": sum(r[f"{name}_point_hit"] for r in pos),
            "false_alarms": sum(r[f"{name}_alarm"] for r in neg)}


def run(methods, epochs, out):
    if epochs < 1 or not methods:
        raise ValueError("positive epochs and at least one method required")
    rows, excluded, digest = reviewed_rows()
    archive = {r["id"]: r for r in json.loads(ARCHIVE.read_text(encoding="utf-8"))["rows"]}
    if {r["id"] for r in rows} - set(archive):
        raise RuntimeError("archived M4 prediction missing")
    out.mkdir(parents=True, exist_ok=True)
    (out / "weights").mkdir(exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    cams = sorted({r["cam"] for r in rows})
    result = {"protocol": {"label_source": "locked reviewer A, single reviewer",
                           "label_sha256": digest, "frames": len(rows),
                           "positive_frames": 19, "normal_frames": 48,
                           "excluded_uncertain_ids": excluded,
                           "feature_cache": str(S.CACHE),
                           "feature_index_sha256": hashlib.sha256(S.INDEX.read_bytes()).hexdigest(),
                           "reference": "retrospective same-camera leave-one-out; includes excluded frames",
                           "split": "six test-camera folds, next camera validates; same fault event",
                           "threshold": "95th percentile of validation camera normal scores",
                           "independent_fault_test": False, "epochs": epochs,
                           "seed": SEED, "device": device}, "methods": {}}
    started = time.perf_counter()
    for method in methods:
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        folds, predictions = [], []
        for fold, test_cam in enumerate(cams):
            val_cam = cams[(fold + 1) % len(cams)]
            train = [r for r in rows if r["cam"] not in (test_cam, val_cam)]
            val = [r for r in rows if r["cam"] == val_cam]
            test = [r for r in rows if r["cam"] == test_cam]
            mean, std = S.train_stats(train, "M4")
            model = S.Head("M4").to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            best, best_epoch, state = math.inf, 0, None
            for epoch in range(epochs):
                model.train()
                for idx in np.random.permutation(len(train)):
                    row = train[idx]
                    x, y = load(row, mean, std)
                    optimizer.zero_grad()
                    logits = model(x[None].to(device))
                    loss = loss_for(logits, y[None].to(device), row, method)
                    loss.backward()
                    optimizer.step()
                model.eval()
                with torch.no_grad():
                    values = []
                    for row in val:
                        x, y = load(row, mean, std)
                        logits = model(x[None].to(device))
                        values.append(float(loss_for(logits, y[None].to(device), row, method)))
                    val_loss = float(np.mean(values))
                if val_loss < best:
                    best, best_epoch = val_loss, epoch + 1
                    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            model.load_state_dict(state)
            checkpoint = out / "weights" / f"{method}_fold{fold+1}_{test_cam}.pt"
            torch.save({"state_dict": state, "mean": mean, "std": std,
                        "train_ids": [r["id"] for r in train],
                        "val_ids": [r["id"] for r in val], "test_ids": [r["id"] for r in test],
                        "label_sha256": digest, "best_epoch": best_epoch, "seed": SEED}, checkpoint)
            val_pred = predict(model, val, mean, std, device)
            threshold = float(np.quantile([r["score"] for r in val_pred if not r["defect"]], .95))
            for pred in predict(model, test, mean, std, device):
                pred.update({"fold": fold + 1, "threshold": threshold,
                             "alarm": pred["score"] >= threshold})
                predictions.append(pred)
            folds.append({"test_cam": test_cam, "val_cam": val_cam,
                          "train_ids": [r["id"] for r in train],
                          "val_ids": [r["id"] for r in val], "test_ids": [r["id"] for r in test],
                          "best_epoch": best_epoch, "best_val_loss": best,
                          "threshold": threshold, "checkpoint": str(checkpoint),
                          "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()})
            print(f"{method} fold {fold+1}/6 {test_cam}: epoch={best_epoch} val={best:.4f}", flush=True)
        packed = [{**r, "a_alarm": r.pop("alarm"), "a_point_hit": r.pop("point_hit")}
                  for r in predictions]
        result["methods"][method] = {"metrics": metrics(packed, "a"), "folds": folds,
                                      "rows": packed}
        print(method, result["methods"][method]["metrics"], flush=True)
    baseline = []
    for row in rows:
        pred = archive[row["id"]]
        box = pred.get("pred_box")
        point = ((box[0]+box[2])/2, (box[1]+box[3])/2) if box else None
        hit = bool(point and any(x0 <= point[0] <= x1 and y0 <= point[1] <= y1
                                 for x0, y0, x1, y1 in row["gt"]))
        baseline.append({"id": row["id"], "defect": bool(row["gt"]),
                         "alarm": bool(pred["alarm"]), "point_hit_approx": hit})
    result["archived_m4_context"] = {
        "note": "Old-label-trained M4 and its original thresholds; point is approximated by archived box center. Descriptive only, not matched retraining.",
        "alarm_hits": sum(r["alarm"] for r in baseline if r["defect"]),
        "false_alarms": sum(r["alarm"] for r in baseline if not r["defect"]),
        "alarm_and_approx_point_hits": sum(r["alarm"] and r["point_hit_approx"] for r in baseline if r["defect"])}
    result["elapsed_seconds"] = round(time.perf_counter() - started, 2)
    (out / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                                      encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--methods", nargs="+", choices=("bce", "mil", "mil_hard", "mil_normalhard", "mil_rank"), default=["bce", "mil"])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    result = run(args.methods, args.epochs, args.out)
    print(json.dumps({"protocol": result["protocol"],
                      "metrics": {k: v["metrics"] for k, v in result["methods"].items()},
                      "archived_m4_context": result["archived_m4_context"],
                      "elapsed_seconds": result["elapsed_seconds"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
