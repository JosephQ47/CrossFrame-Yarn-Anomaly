# -*- coding: utf-8 -*-
"""修复实验 第二轮：外观级入侵物屏蔽（LAB 中值背景差分）。

第一轮结论：干净参考池（R3）把 F1 图级 AUROC 67→93、P95 召回 13%→87%，但定位仍 5/15；
基于异常图连通域面积的屏蔽（R2/R4/B1）无效——K=73 预算下人手块被切成多个 ≤55 的碎块，
面积判据抓不住它，红框照样套在手上。黑纱那个手臂漏检（Cam3）也没翻正。

本轮换判据：入侵物（手、白纸卡）是**外观**上与背景差异巨大的大面积区域，而缺陷是小面积、
与纱线同色系。于是：
  · 参考帧（F1=9 张确认正常；黑纱=同机位其余帧）在筘齿带内取逐像素 LAB **中值**做背景
  · 当前帧与背景的 ΔE（LAB 欧氏距离）> T 的像素 → 形态学开运算 → 只保留面积 > A 的大连通域 → 膨胀
  · 映射到 patch 网格（patch 内被遮比例 > 30% 即视为入侵），从异常图里剔除后再定阈/定位
两个阈值 T=20、A=带面积 1.5% 事先固定，不在 F1 上扫参（F1 只有 15 个缺陷，扫参=过拟合）。

臂：
  F1   R3 干净池(K73)                ← 第一轮最优，对照
       R5 R3 + 外观屏蔽
  黑纱 B0 K73 基线                    ← 15/16
       B2 B0 + 外观屏蔽
"""
import os, sys, json, csv
from pathlib import Path
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE"); os.environ.setdefault("PYTHONUTF8", "1")
import numpy as np, cv2
sys.path.insert(0, str(Path(__file__).resolve().parent))
import black_yarn_detector as D
import f1_repair_exp as E

OUT = E.OUT
DE_T = 20.0        # ΔE 阈值
AREA_FRAC = 0.015  # 大连通域面积下限（占带面积比例）
DIL_PATCH = 1      # 膨胀（patch 单位）
GRID_SCALE = 4     # 背景差分在 patch 网格的 4 倍分辨率上做


def band_lab(img, y0, y1, gh, gw):
    b = cv2.resize(img[y0:y1], (gw * GRID_SCALE, gh * GRID_SCALE), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(b, cv2.COLOR_BGR2LAB).astype(np.float32)


def intruder_mask_appearance(cur_lab, bg_lab, gh, gw):
    """→ patch 级布尔掩膜（True=入侵物）。"""
    de = np.linalg.norm(cur_lab - bg_lab, axis=-1)
    bw = (de > DE_T).astype(np.uint8)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    amin = AREA_FRAC * bw.size
    keep = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= amin]
    if not keep:
        return np.zeros((gh, gw), bool)
    m = np.isin(lab, keep).astype(np.float32)
    frac = cv2.resize(m, (gw, gh), interpolation=cv2.INTER_AREA)   # 每 patch 被遮比例
    pm = (frac > 0.30).astype(np.uint8)
    if DIL_PATCH:
        pm = cv2.dilate(pm, np.ones((2 * DIL_PATCH + 1,) * 2, np.uint8))
    return pm.astype(bool)


def score_with_valid(anom, hw, y0, y1, valid0, k=E.K_ABS, cap=E.CAP):
    vals = anom[valid0]
    if vals.size < 5:
        return 0.0, None, None
    kk = min(k, vals.size)
    thr = np.sort(vals)[-kk]
    bw = ((anom >= thr) & valid0).astype(np.uint8)
    n, lab, stats, cent = cv2.connectedComponentsWithStats(bw, connectivity=8)
    cands = [(stats[i, cv2.CC_STAT_AREA], i) for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] <= cap]
    if not cands:
        return 0.0, None, None
    area, bi = max(cands)
    gh, gw = anom.shape; H, W = hw
    sx, sy = W / gw, (y1 - y0) / gh
    pk = (int(cent[bi][0] * sx), int(y0 + cent[bi][1] * sy))
    ys, xs = np.where(lab == bi)
    box = [int(xs.min() * sx), int(y0 + ys.min() * sy), int((xs.max() + 1) * sx), int(y0 + (ys.max() + 1) * sy)]
    return float(area), pk, box


def run(recs, F, labs, clean_ref, use_mask, tag, save_overlay):
    gh, gw = F.shape[1:3]
    normal_idx = [i for i, r in enumerate(recs) if not r["dets"]]
    scores, labels, hit, ndef, rows = [], [], 0, 0, []
    for k, r in enumerate(recs):
        ref_idx = [i for i in (normal_idx if clean_ref else range(len(recs))) if i != k]
        anom = E.anomaly_from_refs(F[k], F[ref_idx])
        valid = np.ones((gh, gw), bool)
        if use_mask:
            bg = np.median(np.stack([labs[i] for i in ref_idx]), axis=0)
            valid = ~intruder_mask_appearance(labs[k], bg, gh, gw)
        sc, pk, bx = score_with_valid(anom, r["hw"], r["y0"], r["y1"], valid)
        ok = False
        if r["dets"]:
            ndef += 1
            if pk:
                ok = any(b[0] <= pk[0] <= b[2] and b[1] <= pk[1] <= b[3] for b in r["dets"])
            hit += ok
        scores.append(sc); labels.append(1 if r["dets"] else 0)
        rows.append(dict(file=r["path"].stem, defect=bool(r["dets"]), score=sc, hit=ok,
                         masked_patches=int((~valid).sum())))
        if save_overlay:
            img = D.read_bgr(r["path"]); H, W = r["hw"]
            m = np.zeros((H, W), np.float32)
            m[r["y0"]:r["y1"]] = cv2.resize(anom, (W, r["y1"] - r["y0"]), interpolation=cv2.INTER_CUBIC)
            m = cv2.GaussianBlur(m, (0, 0), 9)
            hn = np.clip((m - m.min()) / (m.max() - m.min() + 1e-6), 0, 1)
            vis = cv2.addWeighted(img, 0.62, cv2.applyColorMap((hn * 255).astype(np.uint8), cv2.COLORMAP_JET), 0.38, 0)
            if (~valid).any():
                mk = cv2.resize((~valid).astype(np.uint8), (W, r["y1"] - r["y0"]), interpolation=cv2.INTER_NEAREST)
                full = np.zeros((H, W), np.uint8); full[r["y0"]:r["y1"]] = mk
                grey = vis.copy(); grey[full > 0] = (70, 70, 70)
                vis = cv2.addWeighted(vis, 0.35, grey, 0.65, 0)
            for b in r["dets"]:
                cv2.rectangle(vis, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (255, 255, 255), 4)
            if bx: cv2.rectangle(vis, (bx[0], bx[1]), (bx[2], bx[3]), (0, 0, 255), 4)
            cv2.putText(vis, f"{tag} score={sc:.0f} {'DEFECT' if r['dets'] else 'normal'} {'HIT' if ok else ''}",
                        (12, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)
            od = OUT / f"overlay_{tag}"; od.mkdir(parents=True, exist_ok=True)
            cv2.imencode(".jpg", cv2.resize(vis, (1200, int(1200 * H / W))), [cv2.IMWRITE_JPEG_QUALITY, 86])[1].tofile(
                str(od / f"{r['path'].stem}.jpg"))
    return np.array(scores), np.array(labels), hit, ndef, rows


def main():
    import torch, timm
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = timm.create_model(D.MODEL, pretrained=True, num_classes=0, dynamic_img_size=True).eval().to(dev)
    results = []

    # ---- F1 ----
    D.BAND_Y0, D.BAND_Y1 = E.F1_BAND
    recs = E.load_f1()
    for r in recs:
        H, W = D.read_bgr(r["path"]).shape[:2]
        r["dets"] = [[v[:, 0].min() * W, v[:, 1].min() * H, v[:, 0].max() * W, v[:, 1].max() * H] for v in r["dets"]]
    F = E.featurize(model, dev, recs)
    gh, gw = F.shape[1:3]
    labs = [band_lab(D.read_bgr(r["path"]), r["y0"], r["y1"], gh, gw) for r in recs]
    for tag, clean, mask in (("R3_干净池", True, False), ("R5_干净池+外观屏蔽", True, True)):
        s, l, hit, ndef, rows = run(recs, F, labs, clean, mask, tag, save_overlay=mask)
        auroc, thr, rec, fpr, fpr100 = E.summarize(s, l)
        print(f"F1 {tag:18} 定位 {hit}/{ndef}  AUROC {auroc:5.1f}  P95阈={thr:4.1f} 召回 {rec:3.0f}% 误报 {fpr:3.0f}%  "
              f"100%召回误报 {fpr100:3.0f}%  屏蔽patch均值 {np.mean([r['masked_patches'] for r in rows]):.0f}", flush=True)
        results.append(dict(domain="F1", arm=tag, hit=hit, ndef=ndef, auroc=round(auroc, 1), thr=round(float(thr), 1),
                            recall=round(float(rec), 1), fpr=round(float(fpr), 1), fpr_at_100=round(float(fpr100), 1), rows=rows))
    del F, labs

    # ---- 黑纱：逐机位提特征，两臂共用后即释放（71 帧特征全留会占 ~13 GB）----
    D.BAND_Y0, D.BAND_Y1 = 0.34, 0.60
    brecs = E.load_black()
    cams = sorted(set(r["cam"] for r in brecs))
    arms = (("B0_K73基线", False), ("B2_K73+外观屏蔽", True))
    acc = {t: dict(S=[], L=[], hit=0, ndef=0, rows=[]) for t, _ in arms}
    for c in cams:
        sub = [r for r in brecs if r["cam"] == c]
        F = E.featurize(model, dev, sub)
        gh, gw = F.shape[1:3]
        labs = [band_lab(D.read_bgr(r["path"]), r["y0"], r["y1"], gh, gw) for r in sub]
        for tag, mask in arms:
            s, l, h, n, rows = run(sub, F, labs, False, mask, tag, save_overlay=(mask and c == "Cam3"))
            a = acc[tag]; a["S"].append(s); a["L"].append(l); a["hit"] += h; a["ndef"] += n; a["rows"] += rows
        del F, labs
        print(f"  {c} 完成", flush=True)
    for tag, _ in arms:
        a = acc[tag]
        s, l = np.concatenate(a["S"]), np.concatenate(a["L"])
        auroc, thr, rec, fpr, fpr100 = E.summarize(s, l)
        miss = [r["file"][-22:] for r in a["rows"] if r["defect"] and not r["hit"]]
        print(f"黑纱 {tag:16} 定位 {a['hit']}/{a['ndef']}  AUROC {auroc:5.1f}  P95阈={thr:4.1f} 召回 {rec:3.0f}% 误报 {fpr:3.0f}%  "
              f"100%召回误报 {fpr100:3.0f}%  屏蔽patch均值 {np.mean([r['masked_patches'] for r in a['rows']]):.0f}  漏检 {miss}", flush=True)
        results.append(dict(domain="黑纱", arm=tag, hit=a["hit"], ndef=a["ndef"], auroc=round(auroc, 1), thr=round(float(thr), 1),
                            recall=round(float(rec), 1), fpr=round(float(fpr), 1), fpr_at_100=round(float(fpr100), 1), rows=a["rows"]))

    json.dump(results, open(OUT / "repair_results_round2.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
    with open(OUT / "repair_results_round2.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f); w.writerow(["域", "臂", "定位命中", "缺陷数", "AUROC", "P95阈", "召回%", "误报%", "100%召回误报%"])
        for r in results:
            w.writerow([r["domain"], r["arm"], r["hit"], r["ndef"], r["auroc"], r["thr"], r["recall"], r["fpr"], r["fpr_at_100"]])
    print(f"\n已写出 {OUT/'repair_results_round2.csv'}")


if __name__ == "__main__":
    main()
