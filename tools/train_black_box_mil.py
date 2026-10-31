"""Box-supervision ablation on the archived M4 feature cache.

July single-fault retrospective diagnostic only. Original labels stay unchanged;
F2 and causal/independent-fault data are never read.
"""

import argparse
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
from train_black_dual_head import OLD, SEED, source_rows  # noqa: E402

DEFAULT_OUT = ROOT / "data/black_box_mil_2026-10-31"


def box_masks(row, gh, gw, device):
    """Patch centers in each box, plus a one-patch ignore halo around all boxes."""
    h, w = row["hw"]
    x = (torch.arange(gw, device=device) + .5) * (w / gw)
    y = row["y0"] + (torch.arange(gh, device=device) + .5) * ((row["y1"] - row["y0"]) / gh)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    dx, dy = w / gw, (row["y1"] - row["y0"]) / gh
    inside, near = [], torch.zeros((gh, gw), dtype=torch.bool, device=device)
    for x0, y0, x1, y1 in row["gt"]:
        mask = (xx >= x0) & (xx <= x1) & (yy >= y0) & (yy <= y1)
        if not bool(mask.any()):
            # The box is thinner than one DINO patch: supervise its nearest patch.
            distance = ((xx - (x0+x1)/2) / dx).square() + ((yy - (y0+y1)/2) / dy).square()
            mask = torch.zeros_like(near)
            mask.view(-1)[distance.argmin()] = True
        inside.append(mask)
        near |= (xx >= x0-dx) & (xx <= x1+dx) & (yy >= y0-dy) & (yy <= y1+dy)
    return inside, ~near


def box_mil_loss(logits, row):
    logits = logits[0, 0]
    if not row["gt"]:
        return F.binary_cross_entropy_with_logits(logits, torch.zeros_like(logits))
    boxes, far_normal = box_masks(row, *logits.shape, logits.device)
    positive_losses = []
    for inside in boxes:
        scores = logits[inside]
        k = max(1, math.ceil(scores.numel() * .1))
        selected = scores.topk(k).values.mean()
        positive_losses.append(F.softplus(-selected))
    positive = torch.stack(positive_losses).mean()
    negative = F.softplus(logits[far_normal]).mean()
    return positive + negative


def metrics(rows, prefix):
    positive = [r for r in rows if r["defect"]]
    negative = [r for r in rows if not r["defect"]]
    return {"alarm_hits": sum(r[f"{prefix}_alarm"] for r in positive),
            "false_alarms": sum(r[f"{prefix}_alarm"] for r in negative),
            "point_hits": sum(r[f"{prefix}_point_hit"] for r in positive),
            "alarm_and_point_hits": sum(r[f"{prefix}_alarm"] and
                                        r[f"{prefix}_point_hit"] for r in positive)}


def run(epochs, out):
    if epochs < 1:
        raise ValueError("epochs must be positive")
    rows = source_rows()
    archive = {r["id"]: r for r in json.loads(OLD.read_text(encoding="utf-8"))["rows"]}
    if set(archive) != {r["id"] for r in rows}:
        raise RuntimeError("archived M4/source IDs differ")
    out.mkdir(parents=True, exist_ok=True)
    weights = out / "weights"
    weights.mkdir(exist_ok=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cams = sorted({r["cam"] for r in rows})
    comparison, folds = [], []
    started = time.perf_counter()
    for fold, test_cam in enumerate(cams):
        val_cam = cams[(fold + 1) % len(cams)]
        train = [r for r in rows if r["cam"] not in (test_cam, val_cam)]
        val = [r for r in rows if r["cam"] == val_cam]
        test = [r for r in rows if r["cam"] == test_cam]
        mean, std = S.train_stats(train, "M4")
        model = S.Head("M4").to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        best_loss, best_epoch, best_state = float("inf"), -1, None
        for epoch in range(epochs):
            model.train()
            for idx in np.random.permutation(len(train)):
                row = train[idx]
                x, _ = S.load_row(row, "M4", mean, std)
                optimizer.zero_grad()
                logits = model(x[None].to(device))
                loss = box_mil_loss(logits, row)
                loss.backward()
                optimizer.step()
            model.eval()
            with torch.no_grad():
                values = []
                for row in val:
                    x, y = S.load_row(row, "M4", mean, std)
                    logits = model(x[None].to(device))
                    # Keep the archived M4 validation-selection rule unchanged.
                    values.append(float(F.binary_cross_entropy_with_logits(
                        logits, y[None].to(device))))
                val_loss = float(np.mean(values))
            if val_loss < best_loss:
                best_loss, best_epoch = val_loss, epoch + 1
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(best_state)
        checkpoint = weights / f"mil_fold{fold+1}_{test_cam}.pt"
        torch.save({"state_dict": best_state, "mean": mean, "std": std,
                    "train_ids": [r["id"] for r in train],
                    "val_ids": [r["id"] for r in val],
                    "test_ids": [r["id"] for r in test],
                    "best_epoch": best_epoch, "seed": SEED}, checkpoint)
        val_pred = S.predict(model, val, "M4", mean, std, device)
        test_pred = S.predict(model, test, "M4", mean, std, device)
        val_normals = [r["score"] for r in val_pred if not r["defect"]]
        threshold = float(np.quantile(val_normals, .95))
        for pred in test_pred:
            old = archive[pred["id"]]
            comparison.append({"id": pred["id"], "cam": pred["cam"],
                               "defect": pred["defect"], "fold": fold+1,
                               "m4_score": old["score"], "m4_threshold": old["threshold"],
                               "m4_alarm": old["alarm"], "m4_point_hit": old["point_hit"],
                               "mil_score": pred["score"], "mil_threshold": threshold,
                               "mil_alarm": pred["score"] >= threshold,
                               "mil_point_hit": pred["point_hit"],
                               "mil_box": pred["pred_box"]})
        folds.append({"fold": fold+1, "test_cam": test_cam, "val_cam": val_cam,
                      "best_epoch": best_epoch, "best_val_loss": best_loss,
                      "val_normal_scores": [{"id": r["id"], "score": r["score"]}
                                            for r in val_pred if not r["defect"]],
                      "threshold": threshold, "checkpoint": str(checkpoint),
                      "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                      "train_ids": [r["id"] for r in train],
                      "val_ids": [r["id"] for r in val],
                      "test_ids": [r["id"] for r in test]})
        print(f"fold {fold+1}/6 {test_cam}: epoch={best_epoch} "
              f"val={best_loss:.4f} threshold={threshold:.4f}", flush=True)
    positive = [r for r in comparison if r["defect"]]
    normal = [r for r in comparison if not r["defect"]]
    summary = {"scope": "single July fault; retrospective leave-one-out references",
               "independent_fault_test": False, "causal_references": False,
               "frames": len(comparison), "positive_frames": len(positive),
               "normal_frames": len(normal), "epochs": epochs, "seed": SEED,
               "device": device, "elapsed_seconds": round(time.perf_counter()-started, 2),
               "source_index_sha256": hashlib.sha256(S.INDEX.read_bytes()).hexdigest(),
               "archived_m4_sha256": hashlib.sha256(OLD.read_bytes()).hexdigest(),
               "m4": metrics(comparison, "m4"), "mil": metrics(comparison, "mil"),
               "by_camera": {cam: {"m4": metrics([r for r in comparison if r["cam"] == cam], "m4"),
                                   "mil": metrics([r for r in comparison if r["cam"] == cam], "mil")}
                             for cam in cams}}
    (out / "results.json").write_text(json.dumps({"summary": summary, "folds": folds,
                                                  "rows": comparison}, ensure_ascii=False,
                                                 indent=2) + "\n", encoding="utf-8")
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
