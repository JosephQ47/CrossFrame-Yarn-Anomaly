"""M6 exploratory single-scale, anchor-free YOLO-style box head on M4 features.

This is not a full YOLO implementation. It tests objectness + box regression
with the same 69-frame six-camera development protocol as M1–M4.
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import fewshot_m1 as M1  # noqa: E402
import fewshot_semantic_heads as S  # noqa: E402

OUT = ROOT / "outputs" / "black_fewshot_box"
SEED = 20261024


class BoxHead(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(S.PROJ_DIM * 3 + 1, 32, 1), torch.nn.ReLU(),
            torch.nn.Conv2d(32, 16, 3, padding=1), torch.nn.ReLU(),
            torch.nn.Conv2d(16, 5, 1))

    def forward(self, x):
        return self.net(x)


def target(row, gh, gw):
    obj = np.zeros((gh, gw), np.float32)
    reg = np.zeros((4,), np.float32)
    cell = None
    if row["gt"]:
        x0, y0, x1, y1 = row["gt"][0]
        h, w = row["hw"]
        band_h = row["y1"] - row["y0"]
        cxg = ((x0 + x1) / 2) * gw / w
        cyg = (((y0 + y1) / 2) - row["y0"]) * gh / band_h
        j, i = int(np.clip(np.floor(cxg), 0, gw-1)), int(np.clip(np.floor(cyg), 0, gh-1))
        obj[i, j] = 1
        reg[:] = [np.clip(cxg - (j + .5), -.49, .49),
                  np.clip(cyg - (i + .5), -.49, .49),
                  np.log(max((x1-x0) * gw / w, .1)),
                  np.log(max((y1-y0) * gh / band_h, .1))]
        cell = (i, j)
    return torch.from_numpy(obj[None, None]), torch.from_numpy(reg), cell


def loss_for(model, row, mean, std, device):
    x, _ = S.load_row(row, "M4", mean, std)
    pred = model(x[None].to(device))
    gh, gw = pred.shape[-2:]
    obj, reg, cell = target(row, gh, gw)
    obj = obj.to(device)
    cls = torch.nn.functional.binary_cross_entropy_with_logits(
        pred[:, :1], obj, pos_weight=torch.tensor(200.0, device=device))
    if cell is None:
        return cls
    i, j = cell
    pr = pred[0, 1:, i, j]
    xy = .5 * torch.tanh(pr[:2])
    box = torch.cat([xy, pr[2:]])
    loc = torch.nn.functional.smooth_l1_loss(box, reg.to(device))
    return cls + loc


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    area_a = max(0, a[2]-a[0]) * max(0, a[3]-a[1])
    area_b = max(0, b[2]-b[0]) * max(0, b[3]-b[1])
    return inter / max(area_a + area_b - inter, 1e-6)


def predict(model, rows, mean, std, device):
    model.eval()
    out = []
    with torch.no_grad():
        for r in rows:
            x, _ = S.load_row(r, "M4", mean, std)
            p = model(x[None].to(device))[0].cpu().numpy()
            prob = torch.sigmoid(torch.from_numpy(p[0])).numpy()
            gh, gw = prob.shape
            i, j = np.unravel_index(np.argmax(prob), prob.shape)
            score = float(prob[i, j])
            reg = p[1:, i, j]
            cxg = j + .5 + .5 * np.tanh(reg[0])
            cyg = i + .5 + .5 * np.tanh(reg[1])
            widthg = float(np.exp(np.clip(reg[2], -4, 5)))
            heightg = float(np.exp(np.clip(reg[3], -4, 5)))
            h, w = r["hw"]
            band_h = r["y1"] - r["y0"]
            cx = cxg * w / gw
            cy = r["y0"] + cyg * band_h / gh
            bw = widthg * w / gw
            bh = heightg * band_h / gh
            box = [float(cx-bw/2), float(cy-bh/2), float(cx+bw/2), float(cy+bh/2)]
            hit = any(b[0] <= cx <= b[2] and b[1] <= cy <= b[3] for b in r["gt"])
            best_iou = max([iou(box, b) for b in r["gt"]], default=0)
            out.append({"id": r["id"], "cam": r["cam"], "defect": bool(r["gt"]),
                        "score": score, "point_hit": bool(hit), "box_iou": best_iou,
                        "pred_box": box})
    return out


def run(epochs):
    OUT.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    recs = json.loads(S.INDEX.read_text(encoding="utf-8"))["data"]
    current = {r["id"]: r for r in M1.records()}
    if len(recs) != 69 or any(r["sha256"] != current[r["id"]]["sha256"] for r in recs):
        raise RuntimeError("source data changed or index incomplete")
    cams = sorted({r["cam"] for r in recs})
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = []
    started = time.perf_counter()
    for fold, test_cam in enumerate(cams):
        val_cam = cams[(fold + 1) % len(cams)]
        tr = [r for r in recs if r["cam"] not in (test_cam, val_cam)]
        va = [r for r in recs if r["cam"] == val_cam]
        te = [r for r in recs if r["cam"] == test_cam]
        mean, std = S.train_stats(tr, "M4")
        model = BoxHead().to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        best_loss, best_state = float("inf"), None
        for _ in range(epochs):
            model.train()
            for idx in np.random.permutation(len(tr)):
                opt.zero_grad()
                loss_for(model, tr[idx], mean, std, device).backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                val_loss = float(np.mean([loss_for(model, r, mean, std, device).item() for r in va]))
            if val_loss < best_loss:
                best_loss = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(best_state)
        torch.save({"state_dict": best_state, "fold": fold, "test_cam": test_cam,
                    "val_cam": val_cam, "train_ids": [r["id"] for r in tr],
                    "val_ids": [r["id"] for r in va], "test_ids": [r["id"] for r in te],
                    "mean": mean, "std": std, "seed": SEED, "epochs": epochs,
                    "best_val_loss": best_loss}, OUT / f"M6_fold{fold+1}_{test_cam}.pt")
        val_pred = predict(model, va, mean, std, device)
        test_pred = predict(model, te, mean, std, device)
        threshold = float(np.quantile([x["score"] for x in val_pred if not x["defect"]], .95))
        for x in test_pred:
            x.update({"fold": fold, "test_cam": test_cam, "val_cam": val_cam,
                      "threshold": threshold, "alarm": x["score"] >= threshold})
        results.extend(test_pred)
        print(f"M6 fold {fold+1}/6 test={test_cam} val_loss={best_loss:.4f}", flush=True)
    pos = [x for x in results if x["defect"]]
    neg = [x for x in results if not x["defect"]]
    summary = {"method": "M6 single-scale anchor-free box head", "frames": len(results),
               "positive_frames": len(pos), "image_auroc": float(roc_auc_score(
                   [int(x["defect"]) for x in results], [x["score"] for x in results])),
               "image_ap": float(average_precision_score(
                   [int(x["defect"]) for x in results], [x["score"] for x in results])),
               "fixed_val_threshold_recall": float(np.mean([x["alarm"] for x in pos])),
               "fixed_val_threshold_fpr": float(np.mean([x["alarm"] for x in neg])),
               "point_hits": sum(x["point_hit"] for x in pos),
               "alarm_and_point_hits": sum(x["alarm"] and x["point_hit"] for x in pos),
               "box_iou_ge_03": sum(x["box_iou"] >= .3 for x in pos),
               "alarm_and_box_iou_ge_03": sum(x["alarm"] and x["box_iou"] >= .3 for x in pos),
               "seed": SEED, "epochs": epochs, "elapsed_seconds": round(time.perf_counter()-started, 2),
               "independent_fault_test": False}
    (OUT / "M6_results.json").write_text(json.dumps(
        {"summary": summary, "rows": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    with (OUT / "M6_scores.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    run(ap.parse_args().epochs)
