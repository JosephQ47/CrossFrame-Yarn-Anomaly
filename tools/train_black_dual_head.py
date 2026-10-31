"""G1 diagnostic: M4 input/trunk plus a separately supervised image alarm head.

Retrospective single-fault July development data only. Never use for causal or
independent-fault accuracy claims; F2 is not read.
"""

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import fewshot_semantic_heads as S  # noqa: E402
import fewshot_m1 as M1  # noqa: E402

OLD = ROOT / "data/black_fewshot_semantic_2026-10-25/M4_results.json"
DEFAULT_OUT = ROOT / "data/black_dual_head_2026-10-31"
SEED = 20261024


class DualHead(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.trunk = torch.nn.Sequential(
            torch.nn.Conv2d(193, 32, 1), torch.nn.ReLU(),
            torch.nn.Conv2d(32, 16, 3, padding=1), torch.nn.ReLU())
        self.mask = torch.nn.Conv2d(16, 1, 1)
        self.alarm = torch.nn.Sequential(
            torch.nn.Linear(33, 16), torch.nn.ReLU(), torch.nn.Linear(16, 1))

    def forward(self, x):
        z = self.trunk(x)
        mask = self.mask(z)
        k = max(1, round(mask.shape[-2] * mask.shape[-1] * .01))
        top = mask.flatten(2).topk(k, dim=-1).values.mean(dim=-1)
        pooled = torch.cat([z.mean(dim=(-2, -1)), z.amax(dim=(-2, -1)), top], dim=1)
        return mask, self.alarm(pooled).flatten()


def source_rows():
    index = json.loads(S.INDEX.read_text(encoding="utf-8"))["data"]
    current = {r["id"]: r for r in M1.records()}
    if (len(index) != 69 or set(r["id"] for r in index) != set(current)
            or any(r["sha256"] != current[r["id"]]["sha256"] for r in index)):
        raise RuntimeError("July source/index mismatch; no training")
    if sum(bool(r["gt"]) for r in index) != 16:
        raise RuntimeError("expected 16 July positive frames")
    for row in index:
        if not (S.CACHE / (row["id"] + ".npz")).is_file():
            raise RuntimeError(f"missing frozen M4 cache: {row['id']}")
    return index


def losses(mask, alarm, y, positive, image_pos_weight):
    n_pos = float(y.sum())
    patch_weight = min(max((y.numel() - n_pos) / max(n_pos, 1), 1), 100)
    mask_loss = F.binary_cross_entropy_with_logits(
        mask, y, pos_weight=torch.tensor(patch_weight, device=y.device))
    image_loss = F.binary_cross_entropy_with_logits(
        alarm, torch.tensor([float(positive)], device=y.device),
        pos_weight=torch.tensor(image_pos_weight, device=y.device))
    return mask_loss + image_loss, mask_loss, image_loss


@torch.no_grad()
def predict(model, rows, mean, std, device):
    model.eval()
    output = []
    for row in rows:
        x, _ = S.load_row(row, "M4", mean, std)
        mask, alarm = model(x[None].to(device))
        prob = torch.sigmoid(mask)[0, 0].cpu().numpy()
        score = float(torch.sigmoid(alarm)[0].cpu())
        _, point, box = S.D.score_and_locate(prob, row["hw"], row["y0"], row["y1"])
        hit = bool(point and any(b[0] <= point[0] <= b[2] and b[1] <= point[1] <= b[3]
                                 for b in row["gt"]))
        output.append({"id": row["id"], "cam": row["cam"], "defect": bool(row["gt"]),
                       "score": score, "point_hit": hit, "pred_box": box})
    return output


def run(epochs, out):
    if epochs < 1:
        raise ValueError("epochs must be positive")
    rows = source_rows()
    archived = json.loads(OLD.read_text(encoding="utf-8"))["rows"]
    archived_by_id = {r["id"]: r for r in archived}
    if set(archived_by_id) != {r["id"] for r in rows}:
        raise RuntimeError("archived M4 IDs differ from July index")
    out.mkdir(parents=True, exist_ok=True)
    weights = out / "weights"
    weights.mkdir(exist_ok=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cams = sorted({r["cam"] for r in rows})
    predictions, folds = [], []
    started = time.perf_counter()
    for fold, test_cam in enumerate(cams):
        val_cam = cams[(fold + 1) % len(cams)]
        train = [r for r in rows if r["cam"] not in (test_cam, val_cam)]
        val = [r for r in rows if r["cam"] == val_cam]
        test = [r for r in rows if r["cam"] == test_cam]
        mean, std = S.train_stats(train, "M4")
        pos = sum(bool(r["gt"]) for r in train)
        image_pos_weight = (len(train) - pos) / max(pos, 1)
        model = DualHead().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        best_loss, best_epoch, best_state = float("inf"), -1, None
        for epoch in range(epochs):
            model.train()
            for idx in np.random.permutation(len(train)):
                row = train[idx]
                x, y = S.load_row(row, "M4", mean, std)
                x, y = x[None].to(device), y[None].to(device)
                optimizer.zero_grad()
                mask, alarm = model(x)
                loss, _, _ = losses(mask, alarm, y, bool(row["gt"]), image_pos_weight)
                loss.backward()
                optimizer.step()
            model.eval()
            with torch.no_grad():
                values = []
                for row in val:
                    x, y = S.load_row(row, "M4", mean, std)
                    mask, alarm = model(x[None].to(device))
                    loss, _, _ = losses(mask, alarm, y[None].to(device),
                                        bool(row["gt"]), image_pos_weight)
                    values.append(float(loss))
                val_loss = float(np.mean(values))
            if val_loss < best_loss:
                best_loss, best_epoch = val_loss, epoch + 1
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(best_state)
        checkpoint = weights / f"dual_fold{fold+1}_{test_cam}.pt"
        torch.save({"state_dict": best_state, "mean": mean, "std": std,
                    "train_ids": [r["id"] for r in train], "val_ids": [r["id"] for r in val],
                    "test_ids": [r["id"] for r in test], "best_epoch": best_epoch,
                    "seed": SEED, "image_pos_weight": image_pos_weight}, checkpoint)
        val_pred = predict(model, val, mean, std, device)
        test_pred = predict(model, test, mean, std, device)
        normal_scores = [r["score"] for r in val_pred if not r["defect"]]
        threshold = float(np.quantile(normal_scores, .95, method="linear"))
        for pred in test_pred:
            pred.update({"fold": fold + 1, "val_cam": val_cam,
                         "threshold": threshold, "alarm": pred["score"] >= threshold})
        predictions.extend(test_pred)
        folds.append({"fold": fold + 1, "test_cam": test_cam, "val_cam": val_cam,
                      "best_epoch": best_epoch, "best_val_loss": best_loss,
                      "threshold": threshold, "checkpoint": str(checkpoint),
                      "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                      "train_ids": [r["id"] for r in train],
                      "val_ids": [r["id"] for r in val], "test_ids": [r["id"] for r in test]})
        print(f"fold {fold+1}/6 {test_cam}: epoch={best_epoch} val={best_loss:.4f} "
              f"threshold={threshold:.4f}", flush=True)
    old_by_cam, new_by_cam = {}, {}
    comparison = []
    for pred in predictions:
        old = archived_by_id[pred["id"]]
        comparison.append({"id": pred["id"], "cam": pred["cam"], "defect": pred["defect"],
                           "m4_score": old["score"], "m4_threshold": old["threshold"],
                           "m4_alarm": old["alarm"], "m4_point_hit": old["point_hit"],
                           "dual_score": pred["score"], "dual_threshold": pred["threshold"],
                           "dual_alarm": pred["alarm"], "dual_point_hit": pred["point_hit"],
                           "dual_box": pred["pred_box"]})
    def metrics(method, subset):
        positives = [r for r in subset if r["defect"]]
        negatives = [r for r in subset if not r["defect"]]
        return {"alarm_hits": sum(r[f"{method}_alarm"] for r in positives),
                "false_alarms": sum(r[f"{method}_alarm"] for r in negatives),
                "point_hits": sum(r[f"{method}_point_hit"] for r in positives),
                "alarm_and_point_hits": sum(r[f"{method}_alarm"] and
                                            r[f"{method}_point_hit"] for r in positives)}
    for cam in cams:
        group = [r for r in comparison if r["cam"] == cam]
        old_by_cam[cam] = metrics("m4", group)
        new_by_cam[cam] = metrics("dual", group)
    labels = np.array([int(r["defect"]) for r in comparison])
    scores = np.array([r["dual_score"] for r in comparison])
    summary = {"scope": "single July fault, retrospective references, internal six-camera CV only",
               "independent_fault_test": False, "causal_references": False,
               "frames": len(comparison), "positive_frames": int(labels.sum()),
               "normal_frames": int((1 - labels).sum()), "seed": SEED,
               "epochs": epochs, "device": device,
               "elapsed_seconds": round(time.perf_counter() - started, 2),
               "source_index_sha256": hashlib.sha256(S.INDEX.read_bytes()).hexdigest(),
               "archived_m4_sha256": hashlib.sha256(OLD.read_bytes()).hexdigest(),
               "dual_image_auroc": float(roc_auc_score(labels, scores)),
               "dual_image_ap": float(average_precision_score(labels, scores)),
               "m4": metrics("m4", comparison), "dual": metrics("dual", comparison),
               "m4_by_camera": old_by_cam, "dual_by_camera": new_by_cam}
    (out / "results.json").write_text(json.dumps({"summary": summary, "folds": folds,
                                                  "rows": comparison}, ensure_ascii=False,
                                                 indent=2) + "\n", encoding="utf-8")
    with (out / "scores.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparison[0]))
        writer.writeheader()
        writer.writerows(comparison)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    run(args.epochs, args.out)


if __name__ == "__main__":
    main()
