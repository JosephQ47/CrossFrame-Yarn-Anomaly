# -*- coding: utf-8 -*-
"""黑纱基准 · 完整指标（按工业异常检测论文通用口径）。

复跑检测器一遍，同时算出：
  图级：AUROC、AP(AUPR)、F1-max、产线定阈(纯正常P95)下 P/R/F1、TPR@FPR5%、FPR@TPR100%
  定位：点命中率、像素级 AUROC、PRO(积分至 FPR<=0.3 归一化)、框 IoU、框级召回@IoU

诚实声明：本基准 GT 是**矩形框**而非像素掩膜。像素级 AUROC 与 PRO 用框掩膜近似
（框内一律算正），因此这两项**偏保守于真实掩膜口径**（框内的正常像素被当成缺陷像素），
不能与 MVTec AD 等有精细掩膜的数据集直接横比，只用于本项目内部方法比较。
像素统计**只在筘齿带内**进行——检测器本身只在带内输出，带外恒为 0，
若把带外像素计入会把 AUROC 无意义地推高。

输出:
  data/scores_per_frame.csv      每帧分数
  data/metrics_黑纱基准.json     全部指标
  data/pro_curve.csv             PRO 曲线原始点
用法: D:\\anaconda3\\envs\\Yolov8\\python.exe metrics_full.py
"""
import os, sys, glob, json, re, csv
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("PYTHONUTF8", "1")
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, cv2, torch, timm
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             roc_curve, precision_recall_curve)
import black_yarn_detector as D

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)
PRO_FPR_MAX = 0.30      # PRO 曲线积分上限，工业异常检测通用取 0.3


def iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    it = (x1 - x0) * (y1 - y0)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - it
    return it / max(ua, 1e-9)


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = timm.create_model(D.MODEL, pretrained=True, num_classes=0,
                              dynamic_img_size=True).eval().to(dev)
    print("device=%s 模型=%s 层=%d topk=%d cap=%d"
          % (dev, D.MODEL, D.LAYERS, D.TOPK, D.AREA_CAP), flush=True)

    recs = []
    for jp in sorted(glob.glob(str(D.BF / "*.json"))):
        d = json.load(open(jp, encoding="utf-8"))
        dets = [D.pbox(s["points"]) for s in d["shapes"] if s["label"].lower() == "detect"]
        ip = D.BF / (Path(jp).stem + ".jpg")
        if not ip.exists():
            continue
        mm = re.search(r"(Cam\d)", ip.name)
        recs.append(dict(path=ip, dets=dets, cam=mm.group(1) if mm else "NA"))

    rows, hit, ndef = [], 0, 0
    px_s, px_l = [], []          # 像素级分数/标签（仅带内，下采样后）
    regions = []                 # PRO 用：每个 GT 区域内的像素分数
    ious = []
    for c in sorted(set(r["cam"] for r in recs)):
        sub = [r for r in recs if r["cam"] == c]
        feats = []
        for r in sub:
            img = D.read_bgr(r["path"])
            f, y0, y1 = D.band_feat(model, dev, img)
            r["_y0"], r["_y1"], r["_hw"] = y0, y1, img.shape[:2]
            feats.append(f)
        F = np.stack(feats)
        for k, r in enumerate(sub):
            m, anom = D.anomaly_map(F, k, r["_hw"], r["_y0"], r["_y1"])
            sc, pk, bx = D.score_and_locate(anom, r["_hw"], r["_y0"], r["_y1"])
            ok = False
            if r["dets"]:
                ndef += 1
                pxy = pk if pk else (-1, -1)
                ok = any(b[0] <= pxy[0] <= b[2] and b[1] <= pxy[1] <= b[3] for b in r["dets"])
                hit += ok
                ious.append(max([iou(bx, b) for b in r["dets"]]) if bx else 0.0)
            rows.append(dict(file=r["path"].stem, cam=c, defect=int(bool(r["dets"])),
                             score=sc, hit=int(ok),
                             box=[int(v) for v in bx] if bx else None,
                             gt=[[int(v) for v in b] for b in r["dets"]]))

            # ---- 像素级：只取筘齿带，8 倍下采样以控内存 ----
            y0, y1 = r["_y0"], r["_y1"]
            band = m[y0:y1]
            gt = np.zeros_like(band, np.uint8)
            for b in r["dets"]:
                gy0, gy1 = int(max(b[1] - y0, 0)), int(min(b[3] - y0, band.shape[0]))
                gx0, gx1 = int(max(b[0], 0)), int(min(b[2], band.shape[1]))
                if gy1 > gy0 and gx1 > gx0:
                    gt[gy0:gy1, gx0:gx1] = 1
            px_s.append(band[::8, ::8].ravel())
            px_l.append(gt[::8, ::8].ravel())
            for b in r["dets"]:
                gy0, gy1 = int(max(b[1] - y0, 0)), int(min(b[3] - y0, band.shape[0]))
                gx0, gx1 = int(max(b[0], 0)), int(min(b[2], band.shape[1]))
                if gy1 > gy0 and gx1 > gx0:
                    regions.append(band[gy0:gy1, gx0:gx1].ravel())
        print("  %s 完成" % c, flush=True)

    # ================= 图级 =================
    s = np.array([r["score"] for r in rows], float)
    l = np.array([r["defect"] for r in rows], int)
    neg, pos = s[l == 0], s[l == 1]
    auroc = roc_auc_score(l, s) * 100
    ap = average_precision_score(l, s) * 100
    prec, rec, _ = precision_recall_curve(l, s)
    f1max = float((2 * prec * rec / np.maximum(prec + rec, 1e-9)).max()) * 100
    fpr, tpr, _ = roc_curve(l, s)
    tpr_at5 = float(np.interp(0.05, fpr, tpr)) * 100
    fpr_at100 = float((neg >= pos.min()).mean()) * 100
    thr95 = float(np.percentile(neg, 95))
    tp = int((pos >= thr95).sum()); fp = int((neg >= thr95).sum())
    R95 = tp / len(pos) * 100
    P95 = tp / max(tp + fp, 1) * 100
    F195 = 2 * P95 * R95 / max(P95 + R95, 1e-9)

    # ================= 像素级 =================
    PS = np.concatenate(px_s); PL = np.concatenate(px_l)
    p_auroc = roc_auc_score(PL, PS) * 100

    # PRO: 阈值扫描，横轴=正常像素 FPR，纵轴=各 GT 区域被覆盖比例的均值
    neg_px = PS[PL == 0]
    xs, ys = [], []
    for t in np.quantile(PS, np.linspace(0.50, 1.0, 120)):
        f_ = float((neg_px >= t).mean())
        if f_ > PRO_FPR_MAX:
            continue
        xs.append(f_)
        ys.append(float(np.mean([float((rg >= t).mean()) for rg in regions])))
    o = np.argsort(xs); xs = np.array(xs)[o]; ys = np.array(ys)[o]
    pro = float(np.trapezoid(ys, xs) / PRO_FPR_MAX) * 100 if len(xs) > 1 else float("nan")

    # ================= 框级 =================
    ious = np.array(ious) if ious else np.zeros(0)
    box_r = {"IoU>=%.1f" % t: float((ious >= t).mean()) * 100 for t in (0.1, 0.3, 0.5)}

    M = {
        "数据": {"总帧": len(rows), "缺陷帧": int(l.sum()), "正常帧": int((1 - l).sum()),
                 "缺陷实例": len(regions), "相机数": len(set(r["cam"] for r in rows))},
        "配置": {"骨干": D.MODEL, "取层": D.LAYERS, "topk": D.TOPK,
                 "筘齿带": [D.BAND_Y0, D.BAND_Y1], "带高": D.TILE_H, "面积上限": D.AREA_CAP},
        "图级": {"AUROC": round(auroc, 1), "AP(AUPR)": round(ap, 1), "F1-max": round(f1max, 1),
                 "TPR@FPR5%": round(tpr_at5, 1), "FPR@TPR100%": round(fpr_at100, 1),
                 "产线定阈": {"阈值(纯正常P95)": round(thr95, 1), "召回": round(R95, 1),
                              "精确率": round(P95, 1), "F1": round(F195, 1)}},
        "定位": {"点命中率": "%d/%d" % (hit, ndef),
                 "点命中率%": round(hit / max(ndef, 1) * 100, 1),
                 "像素AUROC(框掩膜近似)": round(p_auroc, 1),
                 "PRO(FPR<=%.1f)" % PRO_FPR_MAX: round(pro, 1),
                 "框IoU中位数": round(float(np.median(ious)), 3) if ious.size else None,
                 "框级召回%": {k: round(v, 1) for k, v in box_r.items()}},
        "口径声明": "GT 为矩形框非像素掩膜，像素 AUROC/PRO 为框掩膜近似（偏保守），"
                    "仅筘齿带内统计；只有 16 个缺陷实例，单样本波动约 6%。",
    }
    print("\n" + json.dumps(M, ensure_ascii=False, indent=2))
    json.dump(M, open(DATA / "metrics_黑纱基准.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    with open(DATA / "scores_per_frame.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["file", "cam", "defect", "score", "hit", "box", "gt"])
        w.writeheader(); w.writerows(rows)
    np.savetxt(DATA / "pro_curve.csv", np.c_[xs, ys], delimiter=",",
               header="fpr,region_overlap", comments="", fmt="%.6f")
    print("\n已写出 %s / scores_per_frame.csv / pro_curve.csv"
          % (DATA / "metrics_黑纱基准.json"))


if __name__ == "__main__":
    main()
