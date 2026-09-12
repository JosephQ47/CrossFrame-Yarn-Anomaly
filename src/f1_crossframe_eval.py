# -*- coding: utf-8 -*-
"""把黑纱线的 DINOv2 跨帧同位置参考检测器跑在 F1（真实工厂红纱，24 帧）上，用 GT 评分。

为什么：YOLO 一族 8 个臂在 F1 上无一可用（mAP50 ≤0.345）。跨帧参考对颜色天然免疫
（参考来自同机位历史帧，纱色在比对中抵消），黑纱上 15/16、AUROC 98.2。
F1 恰好满足它的条件：单机位(C6)、固定 1280×1920、24 帧、16 缺陷实例 + 9 张确认正常。

BAND 用 (0.26, 0.52)：与基准 (0.34,0.60) **同宽 0.26、只平移**（F1 筘齿更靠上，
实测 GT 框 y 范围 0.28-0.60，中心 0.32-0.46）。放宽会缩小缺陷 patch 面积并破坏与
AREA_CAP=55 的配平，扫参已证伪。

评分与黑纱基准同口径：
  · 定位命中：分数最高连通域重心落在某个 GT 框内
  · 图级 AUROC：有缺陷帧 vs 正常帧
  · 纯正常帧 P95 定阈：召回 / 误报（9 张正常帧，样本极少，只作参考）
参考池为该机位其余全部帧（transductive，与基准一致）。
"""
import os, sys, json
from pathlib import Path
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE"); os.environ.setdefault("PYTHONUTF8", "1")
import cv2, numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import black_yarn_detector as D

RYD = Path(r"C:\Users\quanzhonghui\Documents\色纱课题\outputs\real_yarn_dataset")
OUT = Path(r"C:\Users\quanzhonghui\Documents\色纱课题\outputs\f1_crossframe")
BAND = (0.26, 0.52)


def load():
    recs = []
    for sp in ("test", "finetune"):
        for ip in sorted((RYD / sp / "images").glob("F1__*.jpg")):
            lp = RYD / sp / "labels" / f"{ip.stem}.txt"
            dets = []
            if lp.exists():
                for l in lp.read_text(encoding="utf-8").splitlines():
                    t = l.split()
                    if t and t[0] == "0":
                        v = np.array([float(x) for x in t[1:]]).reshape(-1, 2)
                        dets.append(v)
            recs.append(dict(path=ip, dets=dets))
    recs.sort(key=lambda r: r["path"].stem)
    return recs


def main():
    import torch, timm
    from sklearn.metrics import roc_auc_score
    D.BAND_Y0, D.BAND_Y1 = BAND
    OUT.mkdir(parents=True, exist_ok=True); (OUT / "overlay").mkdir(exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = timm.create_model(D.MODEL, pretrained=True, num_classes=0, dynamic_img_size=True).eval().to(dev)
    recs = load()
    print(f"F1 {len(recs)} 帧，有缺陷 {sum(1 for r in recs if r['dets'])} 帧 / "
          f"{sum(len(r['dets']) for r in recs)} 实例，正常 {sum(1 for r in recs if not r['dets'])} 帧；BAND={BAND}")

    feats = []
    for r in recs:
        img = D.read_bgr(r["path"]); r["hw"] = img.shape[:2]
        f, y0, y1 = D.band_feat(model, dev, img); r["y0"], r["y1"] = y0, y1
        feats.append(f)
    F = np.stack(feats)

    scores, labels, hit, ndef, rows = [], [], 0, 0, []
    for k, r in enumerate(recs):
        m, anom = D.anomaly_map(F, k, r["hw"], r["y0"], r["y1"])
        sc, pk, bx = D.score_and_locate(anom, r["hw"], r["y0"], r["y1"])
        scores.append(sc); labels.append(1 if r["dets"] else 0)
        h, w = r["hw"]; ok_hit = False
        if r["dets"]:
            ndef += 1
            if pk:
                for v in r["dets"]:
                    x0, x1 = v[:, 0].min() * w, v[:, 0].max() * w
                    y0g, y1g = v[:, 1].min() * h, v[:, 1].max() * h
                    if x0 <= pk[0] <= x1 and y0g <= pk[1] <= y1g:
                        ok_hit = True
            hit += ok_hit
        rows.append(dict(file=r["path"].stem, defect=bool(r["dets"]), score=sc, hit=ok_hit,
                         peak=pk, box=bx))
        vis = D.read_bgr(r["path"]).copy()
        hm = cv2.applyColorMap(cv2.normalize(m, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), cv2.COLORMAP_JET)
        vis = cv2.addWeighted(vis, 0.68, hm, 0.32, 0)
        for v in r["dets"]:
            cv2.rectangle(vis, (int(v[:, 0].min() * w), int(v[:, 1].min() * h)),
                          (int(v[:, 0].max() * w), int(v[:, 1].max() * h)), (255, 255, 255), 4)
        if bx: cv2.rectangle(vis, (bx[0], bx[1]), (bx[2], bx[3]), (0, 0, 255), 4)
        cv2.rectangle(vis, (0, r["y0"]), (w - 1, r["y1"]), (0, 200, 255), 2)
        cv2.putText(vis, f"score={sc:.0f} {'DEFECT' if r['dets'] else 'normal'} {'HIT' if ok_hit else ''}",
                    (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (0, 255, 255), 2, cv2.LINE_AA)
        ok, enc = cv2.imencode(".jpg", cv2.resize(vis, (1200, int(1200 * h / w))), [cv2.IMWRITE_JPEG_QUALITY, 88])
        if ok: enc.tofile(str(OUT / "overlay" / f"{r['path'].stem}.jpg"))

    s, l = np.array(scores), np.array(labels)
    auroc = roc_auc_score(l, s) * 100
    normal = s[l == 0]
    thr = np.percentile(normal, 95)
    rec = (s[l == 1] >= thr).mean(); fpr = (normal >= thr).mean()
    order = np.argsort(-s)
    print(f"\n定位命中 {hit}/{ndef}  图级 AUROC {auroc:.1f}")
    print(f"纯正常帧 P95 定阈={thr:.1f}: 召回 {rec:.0%}  误报 {fpr:.0%}  (正常帧仅 9 张)")
    print(f"正常帧分数: {np.sort(normal).astype(int).tolist()}")
    print(f"缺陷帧分数: {np.sort(s[l==1]).astype(int).tolist()}")
    print("\n分数排序（前 12）:")
    for i in order[:12]:
        print(f"  {s[i]:5.0f}  {'缺陷' if l[i] else '正常'}  {'命中' if rows[i]['hit'] else ''}  {rows[i]['file'][-14:]}")
    json.dump(dict(band=BAND, hit=hit, ndef=ndef, auroc=auroc, thr=float(thr), recall=float(rec),
                   fpr=float(fpr), rows=[{**r, "peak": list(r["peak"]) if r["peak"] else None} for r in rows]),
              open(OUT / "f1_crossframe.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
    print(f"\n叠加图 {OUT/'overlay'}  明细 {OUT/'f1_crossframe.json'}")


if __name__ == "__main__":
    main()
