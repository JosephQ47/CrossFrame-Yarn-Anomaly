"""M1 探索实验：DINOv2 跨帧异常图 + 小 CNN。

7 月黑纱批已由用户于 2026-10-24 解禁并改作开发集。本脚本按相机留组；
F2、其他冻结数据不参与训练或调参。结果仅说明同一现场故障的内部泛化。

运行：
  python tools/fewshot_m1.py prepare
  python tools/fewshot_m1.py train
"""

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import black_yarn_detector as D  # noqa: E402

OUT = ROOT / "outputs" / "black_fewshot_m1"
CACHE = OUT / "cache"
INDEX = OUT / "index.json"
SEED = 20261024


def records():
    rows = []
    seen = set()
    for jp in sorted(D.BF.glob("*.json")):
        p = jp.with_suffix(".jpg")
        if not p.exists():
            continue
        digest = hashlib.sha256(p.read_bytes()).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        gt = json.loads(jp.read_text(encoding="utf-8"))
        boxes = [D.pbox(s["points"]) for s in gt["shapes"] if s["label"].lower() == "detect"]
        cam = next((f"Cam{i}" for i in range(1, 7) if f"Cam{i}_" in p.name), None)
        if cam is None:
            raise ValueError(f"无法解析相机：{p}")
        rows.append({"id": p.stem, "file": str(p), "sha256": digest, "cam": cam, "gt": boxes})
    if len(rows) != 69 or sum(bool(r["gt"]) for r in rows) != 16:
        raise RuntimeError(f"数据数量与已审计批次不符：{len(rows)} 帧，{sum(bool(r['gt']) for r in rows)} 缺陷")
    return rows


def box_grid(boxes, hw, y0, y1, gh, gw):
    h, w = hw
    x = (np.arange(gw) + .5) * w / gw
    y = y0 + (np.arange(gh) + .5) * (y1 - y0) / gh
    xx, yy = np.meshgrid(x, y)
    out = np.zeros((gh, gw), np.float32)
    for x0, by0, x1, by1 in boxes:
        out = np.maximum(out, ((xx >= x0) & (xx <= x1) & (yy >= by0) & (yy <= by1)).astype(np.float32))
    return out


def prepare():
    import timm

    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    recs = records()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = timm.create_model(D.MODEL, pretrained=True, num_classes=0, dynamic_img_size=True).eval().to(device)
    print(f"device={device}，69 帧／16 缺陷，按相机提取", flush=True)
    for cam in sorted({r["cam"] for r in recs}):
        sub = [r for r in recs if r["cam"] == cam]
        feats = []
        for r in sub:
            img = D.read_bgr(Path(r["file"]))
            if img is None:
                raise RuntimeError(f"图像读取失败：{r['file']}")
            f, y0, y1 = D.band_feat(model, device, img)
            r["hw"], r["y0"], r["y1"] = list(img.shape[:2]), y0, y1
            feats.append(f)
        F = np.stack(feats)
        for k, r in enumerate(sub):
            _, anom = D.anomaly_map(F, k, r["hw"], r["y0"], r["y1"])
            label = box_grid(r["gt"], r["hw"], r["y0"], r["y1"], *anom.shape)
            np.savez_compressed(CACHE / (r["id"] + ".npz"), anom=anom.astype(np.float16), label=label.astype(np.uint8))
        print(f"  {cam}: {len(sub)} 帧，{sum(bool(r['gt']) for r in sub)} 缺陷", flush=True)
        del F, feats
    INDEX.write_text(json.dumps({"data": recs, "protocol": "same-camera leave-one-out retrospective top1",
                                 "source": str(D.BF), "seed": SEED}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"缓存完成：{INDEX}", flush=True)


def load_row(row):
    z = np.load(CACHE / (row["id"] + ".npz"))
    anom = z["anom"].astype(np.float32)
    scale = max(float(np.quantile(anom, .99)), 1e-4)
    x = np.clip(anom / scale, 0, 4)[None]
    y = z["label"].astype(np.float32)[None]
    return torch.from_numpy(x), torch.from_numpy(y)


class Head(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(1, 8, 3, padding=1), torch.nn.ReLU(),
            torch.nn.Conv2d(8, 8, 3, padding=1), torch.nn.ReLU(),
            torch.nn.Conv2d(8, 1, 1),
        )

    def forward(self, x):
        return self.net(x)


def eval_rows(model, rows, device):
    model.eval()
    out = []
    with torch.no_grad():
        for r in rows:
            x, _ = load_row(r)
            prob = torch.sigmoid(model(x[None].to(device)))[0, 0].cpu().numpy()
            k = max(1, int(round(prob.size * .01)))
            score = float(np.partition(prob.ravel(), -k)[-k:].mean())
            _, point, box = D.score_and_locate(prob, r["hw"], r["y0"], r["y1"])
            hit = bool(point and any(b[0] <= point[0] <= b[2] and b[1] <= point[1] <= b[3] for b in r["gt"]))
            out.append({"id": r["id"], "cam": r["cam"], "defect": bool(r["gt"]),
                        "score": score, "point_hit": hit, "pred_box": box})
    return out


def train(epochs):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    index = json.loads(INDEX.read_text(encoding="utf-8"))
    recs = index["data"]
    if len(recs) != 69 or sum(bool(r["gt"]) for r in recs) != 16:
        raise RuntimeError("缓存索引不完整")
    current = {r["id"]: r for r in records()}
    if any(r["sha256"] != current[r["id"]]["sha256"] for r in recs):
        raise RuntimeError("原图哈希已改变，拒绝使用旧缓存")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cams = sorted({r["cam"] for r in recs})
    results = []
    for fold, test_cam in enumerate(cams):
        val_cam = cams[(fold + 1) % len(cams)]
        tr = [r for r in recs if r["cam"] not in (test_cam, val_cam)]
        va = [r for r in recs if r["cam"] == val_cam]
        te = [r for r in recs if r["cam"] == test_cam]
        if not any(r["gt"] for r in tr) or not any(r["gt"] for r in va) or not any(r["gt"] for r in te):
            raise RuntimeError(f"fold {fold} 缺陷分布不足")
        model = Head().to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        best_state, best_loss = None, float("inf")
        for epoch in range(epochs):
            model.train()
            for idx in np.random.permutation(len(tr)):
                x, y = load_row(tr[idx])
                x, y = x[None].to(device), y[None].to(device)
                opt.zero_grad()
                logits = model(x)
                n_pos = float(y.sum())
                weight = min(max((y.numel() - n_pos) / max(n_pos, 1), 1), 100)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits, y, pos_weight=torch.tensor(weight, device=device))
                loss.backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                losses = []
                for r in va:
                    x, y = load_row(r)
                    logits = model(x[None].to(device))
                    losses.append(float(torch.nn.functional.binary_cross_entropy_with_logits(
                        logits, y[None].to(device)).item()))
                val_loss = float(np.mean(losses))
            if val_loss < best_loss:
                best_loss = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(best_state)
        torch.save({"state_dict": best_state, "fold": fold, "test_cam": test_cam,
                    "val_cam": val_cam, "train_ids": [r["id"] for r in tr],
                    "val_ids": [r["id"] for r in va], "test_ids": [r["id"] for r in te],
                    "seed": SEED, "epochs": epochs, "best_val_loss": best_loss},
                   OUT / f"m1_fold{fold+1}_{test_cam}.pt")
        val_pred = eval_rows(model, va, device)
        test_pred = eval_rows(model, te, device)
        # 每 fold 阈值只使用验证相机的正常帧；测试相机标签不进入模型或阈值。
        val_normal = [x["score"] for x in val_pred if not x["defect"]]
        threshold = float(np.quantile(val_normal, .95))
        for x in test_pred:
            x.update({"fold": fold, "test_cam": test_cam, "val_cam": val_cam,
                      "threshold": threshold, "alarm": x["score"] >= threshold})
        results.extend(test_pred)
        print(f"fold {fold+1}/6 test={test_cam} val={val_cam} train={len(tr)} test={len(te)} "
              f"val_loss={best_loss:.4f} 阈值={threshold:.4f}", flush=True)
    labels = np.array([int(x["defect"]) for x in results])
    scores = np.array([x["score"] for x in results])
    positives = [x for x in results if x["defect"]]
    summary = {
        "protocol": "M1 六折留相机，验证相机轮换；异常图使用既有同机位留一参考",
        "limitations": "16 个框来自同一现场故障；7 月批现为开发集，M0 历史结果非独立测试；F2 未参与",
        "frames": len(results), "positive_frames": len(positives),
        "image_auroc": float(roc_auc_score(labels, scores)),
        "image_ap": float(average_precision_score(labels, scores)),
        "fixed_val_threshold_recall": float(np.mean([x["alarm"] for x in positives])),
        "fixed_val_threshold_fpr": float(np.mean([x["alarm"] for x in results if not x["defect"]])),
        "point_hits": int(sum(x["point_hit"] for x in positives)),
        "point_total": len(positives),
        "epochs": epochs, "seed": SEED,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps({"summary": summary, "rows": results},
                                           ensure_ascii=False, indent=2), encoding="utf-8")
    with (OUT / "scores.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["prepare", "train"])
    ap.add_argument("--epochs", type=int, default=20)
    args = ap.parse_args()
    prepare() if args.step == "prepare" else train(args.epochs)


if __name__ == "__main__":
    main()
