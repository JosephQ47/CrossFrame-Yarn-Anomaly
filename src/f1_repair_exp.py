# -*- coding: utf-8 -*-
"""跨帧参考在 F1 红纱上的修复实验：干净参考池 × 入侵物屏蔽 × 绝对 patch 预算。

背景：跨帧参考在黑纱 15/16，搬到 F1（24 帧 / 15 缺陷帧 16 实例 / 9 确认正常）只有 3/15。
诊断出三个原因，本脚本逐个验证、逐个组合：

  (1) 参考池污染：24 帧里 15 帧含缺陷。→ 只用 9 张确认正常帧做参考（=产线"先录正常段标定"流程）
  (2) 入侵物主导：人手/白纸卡在帧间移动，占满异常图前 1%。→ 面积 > cap 的大块连同邻域屏掉，
      在剩余 patch 上重新取阈，最多迭代 3 轮
  (3) **P99 预算随分辨率缩水**：P99 取的是 patch 总数的 1%。
        黑纱基准 3200 宽 → 32×228 patch → 预算 73；F1 1920 宽 → 32×137 → 预算只有 44。
        F1 的 16 个 GT 框换成 patch 面积是 17~227（中位 45），与预算同量级，人手一抢就没了。
      → 预算改为**绝对 patch 数 K=73**（与黑纱基准等价），cap 保持 55
      实测：K=73 单独只把 AUROC 67→69，作用很小；真正的杠杆是 (1) 干净参考池。

实验臂（F1，BAND=(0.26,0.52)）：
  R0 基线         q=99(预算44)  全帧留一参考          ← 复现 3/15
  R1 绝对预算     K=73          全帧留一参考
  R2 R1+屏蔽      K=73 + 入侵屏蔽
  R3 R1+干净池    K=73          9 正常帧做参考
  R4 全部         K=73 + 入侵屏蔽 + 9 正常帧做参考
黑纱基准同口径回归：K=73 基线（应=15/16）vs +入侵屏蔽（看 Cam3 那个手臂漏检能否翻正、其余不掉）。

评分与黑纱基准一致：定位命中(重心落入 GT)、图级 AUROC、纯正常帧 P95 定阈召回/误报。
输出到 D:/研究生课程/课题/色纱研究/F1红纱重标/修复实验/
"""
import os, sys, json, glob, re, csv
from pathlib import Path
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE"); os.environ.setdefault("PYTHONUTF8", "1")
import numpy as np, cv2
sys.path.insert(0, str(Path(__file__).resolve().parent))
import black_yarn_detector as D

RYD = Path(r"C:\Users\quanzhonghui\Documents\色纱课题\outputs\real_yarn_dataset")
OUT = Path(r"D:\研究生课程\课题\色纱研究\F1红纱重标\修复实验")
F1_BAND = (0.26, 0.52)
K_ABS = 73          # 与黑纱基准 P99 等价的绝对 patch 预算
CAP = 55
CONFIRMED_NEG = {
    "F1__2025-08-06-07-48-31-45-5", "F1__2025-08-06-08-20-31-20-5",
    "F1__2025-08-06-08-20-31-88-5", "F1__2025-08-06-08-22-30-78-5",
    "F1__2025-08-06-08-30-01-94-5", "F1__2025-08-06-08-53-11-69-5",
    "F1__2025-08-06-08-53-12-35-5", "F1__2025-08-06-08-59-46-81-5",
    "F1__2025-08-06-08-59-47-45-5",
}


# ---------------- 评分：预算 + 可选入侵屏蔽 ----------------
def score_locate(anom, hw, y0, y1, budget, cap=CAP, mask_intruder=False, rounds=3, dil=2):
    """budget: ("q", 99) 用分位数（原口径）或 ("k", 73) 用绝对 patch 数。
    mask_intruder: 面积 > cap 的连通域及其 dil 邻域屏掉，在剩余 patch 上重新取阈，迭代 rounds 轮。
    返回 (图级分数, 峰值点, 框, 屏蔽掩膜)。"""
    valid = np.ones(anom.shape, bool)
    n = lab = stats = cent = None
    for _ in range(rounds if mask_intruder else 1):
        vals = anom[valid]
        if budget[0] == "q":
            thr = np.percentile(vals, budget[1])
        else:
            k = min(int(budget[1]), vals.size)
            thr = np.sort(vals)[-k]
        bw = ((anom >= thr) & valid).astype(np.uint8)
        n, lab, stats, cent = cv2.connectedComponentsWithStats(bw, connectivity=8)
        big = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] > cap]
        if not (mask_intruder and big):
            break
        m = cv2.dilate(np.isin(lab, big).astype(np.uint8), np.ones((2 * dil + 1,) * 2, np.uint8))
        valid &= ~m.astype(bool)
    cands = [(stats[i, cv2.CC_STAT_AREA], i) for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] <= cap]
    if not cands:
        return 0.0, None, None, ~valid
    area, bi = max(cands)
    gh, gw = anom.shape; H, W = hw
    sx, sy = W / gw, (y1 - y0) / gh
    pk = (int(cent[bi][0] * sx), int(y0 + cent[bi][1] * sy))
    ys, xs = np.where(lab == bi)
    box = [int(xs.min() * sx), int(y0 + ys.min() * sy), int((xs.max() + 1) * sx), int(y0 + (ys.max() + 1) * sy)]
    return float(area), pk, box, ~valid


def anomaly_from_refs(cur, R):
    sims = np.einsum('tijc,ijc->tij', R, cur)
    kk = max(1, min(D.TOPK, sims.shape[0]))
    return 1.0 - np.sort(sims, axis=0)[-kk:].mean(axis=0)


def evaluate(recs, F, budget, mask_intruder, clean_ref, tag, save_overlay=False):
    """recs 同一机位；clean_ref=True 时参考只取无缺陷帧（正常帧自身留一）。"""
    normal_idx = [i for i, r in enumerate(recs) if not r["dets"]]
    scores, labels, hit, ndef, rows = [], [], 0, 0, []
    for k, r in enumerate(recs):
        if clean_ref:
            ref_idx = [i for i in normal_idx if i != k]
        else:
            ref_idx = [i for i in range(len(recs)) if i != k]
        anom = anomaly_from_refs(F[k], F[ref_idx])
        sc, pk, bx, masked = score_locate(anom, r["hw"], r["y0"], r["y1"], budget, mask_intruder=mask_intruder)
        ok = False
        if r["dets"]:
            ndef += 1
            if pk:
                ok = any(b[0] <= pk[0] <= b[2] and b[1] <= pk[1] <= b[3] for b in r["dets"])
            hit += ok
        scores.append(sc); labels.append(1 if r["dets"] else 0)
        rows.append(dict(file=r["path"].stem, defect=bool(r["dets"]), score=sc, hit=ok, box=bx,
                         masked_patches=int(masked.sum())))
        if save_overlay:
            img = D.read_bgr(r["path"]); H, W = r["hw"]
            m = np.zeros((H, W), np.float32)
            m[r["y0"]:r["y1"]] = cv2.resize(anom, (W, r["y1"] - r["y0"]), interpolation=cv2.INTER_CUBIC)
            m = cv2.GaussianBlur(m, (0, 0), 9)
            hn = np.clip((m - m.min()) / (m.max() - m.min() + 1e-6), 0, 1)
            vis = cv2.addWeighted(img, 0.62, cv2.applyColorMap((hn * 255).astype(np.uint8), cv2.COLORMAP_JET), 0.38, 0)
            if masked.any():   # 被屏蔽的入侵区域画成半透明灰
                mk = cv2.resize(masked.astype(np.uint8), (W, r["y1"] - r["y0"]), interpolation=cv2.INTER_NEAREST)
                full = np.zeros((H, W), np.uint8); full[r["y0"]:r["y1"]] = mk
                grey = vis.copy(); grey[full > 0] = (90, 90, 90)
                vis = cv2.addWeighted(vis, 0.45, grey, 0.55, 0)
            for b in r["dets"]:
                cv2.rectangle(vis, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (255, 255, 255), 4)
            if bx: cv2.rectangle(vis, (bx[0], bx[1]), (bx[2], bx[3]), (0, 0, 255), 4)
            cv2.rectangle(vis, (0, r["y0"]), (W - 1, r["y1"]), (0, 200, 255), 2)
            cv2.putText(vis, f"{tag} score={sc:.0f} {'DEFECT' if r['dets'] else 'normal'} {'HIT' if ok else ''}",
                        (12, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)
            od = OUT / f"overlay_{tag}"; od.mkdir(parents=True, exist_ok=True)
            cv2.imencode(".jpg", cv2.resize(vis, (1200, int(1200 * H / W))), [cv2.IMWRITE_JPEG_QUALITY, 86])[1].tofile(
                str(od / f"{r['path'].stem}.jpg"))
    return np.array(scores), np.array(labels), hit, ndef, rows


def summarize(s, l):
    from sklearn.metrics import roc_auc_score
    neg, pos = s[l == 0], s[l == 1]
    auroc = roc_auc_score(l, s) * 100 if len(set(l)) > 1 else float("nan")
    thr = np.percentile(neg, 95)
    return auroc, thr, (pos >= thr).mean() * 100, (neg >= thr).mean() * 100, (neg >= pos.min()).mean() * 100


def load_f1():
    recs = []
    for sp in ("test", "finetune"):
        for ip in sorted((RYD / sp / "images").glob("F1__*.jpg")):
            lp = RYD / sp / "labels" / f"{ip.stem}.txt"
            dets = []
            if lp.exists():
                for line in lp.read_text(encoding="utf-8").splitlines():
                    t = line.split()
                    if t and t[0] == "0":
                        v = np.array([float(x) for x in t[1:]]).reshape(-1, 2)
                        dets.append(v)
            if not dets and ip.stem not in CONFIRMED_NEG:
                continue      # 未确认的帧不进评测
            recs.append(dict(path=ip, dets=dets))
    recs.sort(key=lambda r: r["path"].stem)
    return recs


def load_black():
    recs = []
    for jp in sorted(glob.glob(str(D.BF / "*.json"))):
        d = json.load(open(jp, encoding="utf-8"))
        dets = [D.pbox(s["points"]) for s in d["shapes"] if s["label"].lower() == "detect"]
        ip = D.BF / (Path(jp).stem + ".jpg")
        if not ip.exists(): continue
        mm = re.search(r"(Cam\d)", ip.name)
        recs.append(dict(path=ip, dets=dets, cam=mm.group(1) if mm else "NA"))
    return D.dedup_recs(recs)


def featurize(model, dev, recs):
    feats = []
    for r in recs:
        img = D.read_bgr(r["path"]); r["hw"] = img.shape[:2]
        f, y0, y1 = D.band_feat(model, dev, img); r["y0"], r["y1"] = y0, y1
        feats.append(f)
    return np.stack(feats)


def main():
    import torch, timm
    OUT.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = timm.create_model(D.MODEL, pretrained=True, num_classes=0, dynamic_img_size=True).eval().to(dev)
    results = []

    # ================= F1 =================
    D.BAND_Y0, D.BAND_Y1 = F1_BAND
    recs = load_f1()
    # GT 框换成像素后转为 GT 框，供 evaluate 用
    for r in recs:
        H, W = D.read_bgr(r["path"]).shape[:2]
        r["dets"] = [[v[:, 0].min() * W, v[:, 1].min() * H, v[:, 0].max() * W, v[:, 1].max() * H] for v in r["dets"]]
    F = featurize(model, dev, recs)
    gh, gw = F.shape[1:3]
    print(f"F1: {len(recs)} 帧（缺陷 {sum(1 for r in recs if r['dets'])} / 正常 {sum(1 for r in recs if not r['dets'])}），"
          f"patch 网格 {gh}x{gw}={gh*gw}，P99 预算={round(gh*gw*0.01)}，绝对预算 K={K_ABS}", flush=True)
    arms = [
        ("R0_基线_q99",          ("q", 99),    False, False),
        ("R1_绝对预算K73",       ("k", K_ABS), False, False),
        ("R2_K73+入侵屏蔽",      ("k", K_ABS), True,  False),
        ("R3_K73+干净参考池",    ("k", K_ABS), False, True),
        ("R4_K73+屏蔽+干净池",   ("k", K_ABS), True,  True),
    ]
    for tag, budget, mask, clean in arms:
        s, l, hit, ndef, rows = evaluate(recs, F, budget, mask, clean, tag, save_overlay=(tag.startswith("R4") or tag.startswith("R0")))
        auroc, thr, rec, fpr, fpr100 = summarize(s, l)
        print(f"  {tag:22} 定位 {hit}/{ndef}  AUROC {auroc:5.1f}  P95阈={thr:4.1f} 召回 {rec:3.0f}% 误报 {fpr:3.0f}%  "
              f"100%召回误报 {fpr100:3.0f}%  屏蔽patch均值 {np.mean([r['masked_patches'] for r in rows]):.0f}", flush=True)
        results.append(dict(domain="F1", arm=tag, hit=hit, ndef=ndef, auroc=round(auroc, 1), thr=round(float(thr), 1),
                            recall=round(float(rec), 1), fpr=round(float(fpr), 1), fpr_at_100=round(float(fpr100), 1),
                            rows=rows))
    del F

    # ================= 黑纱回归 =================
    D.BAND_Y0, D.BAND_Y1 = 0.34, 0.60
    brecs = load_black()
    cams = sorted(set(r["cam"] for r in brecs))
    for tag, mask in (("B0_K73基线", False), ("B1_K73+入侵屏蔽", True)):
        S, L, hit, ndef, allrows = [], [], 0, 0, []
        for c in cams:
            sub = [r for r in brecs if r["cam"] == c]
            if tag == "B0_K73基线":
                sub_F = featurize(model, dev, sub);
                for r, f in zip(sub, sub_F): r["_f"] = f
            F = np.stack([r["_f"] for r in sub])
            s, l, h, n, rows = evaluate(sub, F, ("k", K_ABS), mask, False, tag, save_overlay=False)
            S.append(s); L.append(l); hit += h; ndef += n; allrows += rows
        s, l = np.concatenate(S), np.concatenate(L)
        auroc, thr, rec, fpr, fpr100 = summarize(s, l)
        gh, gw = F.shape[1:3]
        print(f"黑纱 {tag:16} 定位 {hit}/{ndef}  AUROC {auroc:5.1f}  P95阈={thr:4.1f} 召回 {rec:3.0f}% 误报 {fpr:3.0f}%  "
              f"100%召回误报 {fpr100:3.0f}%  (网格 {gh}x{gw}, P99预算={round(gh*gw*0.01)})", flush=True)
        miss = [r["file"][-22:] for r in allrows if r["defect"] and not r["hit"]]
        print(f"     漏检: {miss}")
        results.append(dict(domain="黑纱", arm=tag, hit=hit, ndef=ndef, auroc=round(auroc, 1), thr=round(float(thr), 1),
                            recall=round(float(rec), 1), fpr=round(float(fpr), 1), fpr_at_100=round(float(fpr100), 1),
                            rows=allrows))

    json.dump(results, open(OUT / "repair_results.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
    with open(OUT / "repair_results.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f); w.writerow(["域", "臂", "定位命中", "缺陷数", "AUROC", "P95阈", "召回%", "误报%", "100%召回误报%"])
        for r in results:
            w.writerow([r["domain"], r["arm"], r["hit"], r["ndef"], r["auroc"], r["thr"], r["recall"], r["fpr"], r["fpr_at_100"]])
    print(f"\n已写出 {OUT/'repair_results.csv'}")


if __name__ == "__main__":
    main()
