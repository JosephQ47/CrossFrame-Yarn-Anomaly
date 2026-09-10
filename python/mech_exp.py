# -*- coding: utf-8 -*-
"""把"同位置参考为什么强"从经验变成可量化的机理（黑纱基准，去重后 69 帧 / 16 缺陷）。

三个实验，全部复用 black_yarn_detector 的特征与评分：
  (a) 分布收窄：正常 patch 相对参考池的异常度（1 − top1 cos）在两种参考系下的分布——
      同位置参考 vs 任意位置参考（PatchCore-DINOv2 式）；再叠上缺陷 patch 的异常度。
      "可分性裕度" = 缺陷 patch 异常度中位数 / 正常 patch 异常度 P99。
  (b) 参考帧数曲线：参考池随机取 k ∈ {1,2,3,4,6,8,全部} 帧（重复 5 次取均值），看定位/AUROC/P95 召回。
  (c) 相机抖动：只对查询帧整体平移 dx ∈ {0,3,7,14,28} px（参考帧不动），看性能衰减；
      以及"邻域松弛"——每个 patch 与参考帧同坐标 ±1 patch 的 3×3 邻域取最大相似度。

可选：--white <目录> 对带时间戳的白纱正常帧（按文件名末位机位分组）做回归：
      同位置 vs 任意位置的正常异常度分布 + 误报率（纯正常帧无法算召回）。白纱筘齿带位置由 --white-band 给出。

输出 D:/研究生课程/课题/色纱研究/机理实验/：mech_results.json + 三张图（由 make_mech_figs.py 绘制）
"""
import os, sys, glob, json, re, argparse, time
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE"); os.environ.setdefault("PYTHONUTF8", "1")
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, cv2, torch, timm
from sklearn.metrics import roc_auc_score
import black_yarn_detector as D

OUT = Path(r"D:\研究生课程\课题\色纱研究\机理实验"); OUT.mkdir(parents=True, exist_ok=True)
dev = "cuda" if torch.cuda.is_available() else "cpu"
rng = np.random.default_rng(0)


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


def gt_patch_mask(r, gh, gw, y0, y1):
    """GT 框 → patch 网格布尔掩膜（框内 patch 为 True）。"""
    H, W = r["hw"]; m = np.zeros((gh, gw), bool)
    for b in r["dets"]:
        j0, j1 = int(b[0] / W * gw), int(np.ceil(b[2] / W * gw))
        i0, i1 = int((b[1] - y0) / (y1 - y0) * gh), int(np.ceil((b[3] - y0) / (y1 - y0) * gh))
        m[max(0, i0):min(gh, i1), max(0, j0):min(gw, j1)] = True
    return m


def anom_same(F, k, ref_idx):
    sims = np.einsum('tijc,ijc->tij', F[ref_idx], F[k]); return 1.0 - sims.max(0)


@torch.no_grad()
def anom_any(F, k, ref_idx, sub=0.3, qchunk=2048):
    """任意位置参考：记忆库 = 参考帧全部位置 patch（子采样），余弦 top-1。"""
    bank = F[ref_idx].reshape(-1, F.shape[-1]); bank = bank[rng.choice(len(bank), int(len(bank) * sub), replace=False)]
    B = torch.from_numpy(np.ascontiguousarray(bank)).to(dev); Q = torch.from_numpy(F[k].reshape(-1, F.shape[-1])).to(dev)
    out = torch.empty(len(Q), device=dev)
    for s0 in range(0, len(Q), qchunk):
        out[s0:s0 + qchunk] = (Q[s0:s0 + qchunk] @ B.T).max(1).values
    return (1.0 - out).reshape(F.shape[1:3]).cpu().numpy()


def anom_neigh(F, k, ref_idx, rad=1):
    """邻域松弛：与参考帧同坐标 ±rad patch 的邻域取最大相似度。"""
    cur = F[k]; best = np.full(cur.shape[:2], -1.0, np.float32)
    for di in range(-rad, rad + 1):
        for dj in range(-rad, rad + 1):
            R = np.roll(F[ref_idx], shift=(di, dj), axis=(1, 2))
            s = np.einsum('tijc,ijc->tij', R, cur).max(0)
            best = np.maximum(best, s)
    return 1.0 - best


def evaluate(recs, anoms):
    """anoms: 每帧 patch 异常图列表 → 定位命中、AUROC、P95 召回/误报。"""
    s, l, hit, ndef = [], [], 0, 0
    for r, a in zip(recs, anoms):
        sc, pk, _ = D.score_and_locate(a, r["hw"], r["y0"], r["y1"])
        s.append(sc); l.append(int(bool(r["dets"])))
        if r["dets"]:
            ndef += 1
            hit += bool(pk) and any(b[0] <= pk[0] <= b[2] and b[1] <= pk[1] <= b[3] for b in r["dets"])
    s, l = np.array(s), np.array(l)
    neg, pos = s[l == 0], s[l == 1]
    thr = np.percentile(neg, 95) if neg.size else np.nan
    return dict(hit=hit, ndef=ndef, auroc=round(roc_auc_score(l, s) * 100, 1) if len(set(l)) > 1 else None,
                rec95=round(float((pos >= thr).mean() * 100), 1) if pos.size else None,
                fpr95=round(float((neg >= thr).mean() * 100), 1), n=len(s))


def featurize(model, recs, shift=(0, 0)):
    feats = []
    for r in recs:
        img = D.read_bgr(r["path"])
        if shift != (0, 0):
            M = np.float32([[1, 0, shift[0]], [0, 1, shift[1]]])
            img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), borderMode=cv2.BORDER_REPLICATE)
        r["hw"] = img.shape[:2]
        f, y0, y1 = D.band_feat(model, dev, img); r["y0"], r["y1"] = y0, y1
        feats.append(f)
    return np.stack(feats)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--white", type=Path, default=None)
    ap.add_argument("--white-band", type=float, nargs=2, default=None)
    ap.add_argument("--skip", nargs="*", default=[], help="跳过的实验: a b c")
    a = ap.parse_args()
    t0 = time.time()
    model = timm.create_model(D.MODEL, pretrained=True, num_classes=0, dynamic_img_size=True).eval().to(dev)
    R = {}

    recs = load_black(); cams = sorted(set(r["cam"] for r in recs))
    # ---------- 逐机位提特征（黑纱），三个实验共用 ----------
    F_by = {}
    for c in cams:
        sub = [r for r in recs if r["cam"] == c]
        F_by[c] = (sub, featurize(model, sub))
        print(f"[{c}] 特征 {F_by[c][1].shape}  {time.time()-t0:.0f}s", flush=True)

    # ================= (a) 分布收窄 =================
    if "a" not in a.skip:
        vals = dict(same_normal=[], any_normal=[], same_defect=[], any_defect=[])
        for c in cams:
            sub, F = F_by[c]; gh, gw = F.shape[1:3]
            normal_idx = [i for i, r in enumerate(sub) if not r["dets"]]
            for k, r in enumerate(sub):
                ref = [i for i in normal_idx if i != k]
                if len(ref) < 3: continue
                a_same = anom_same(F, k, ref); a_any = anom_any(F, k, ref)
                if r["dets"]:
                    gm = gt_patch_mask(r, gh, gw, r["y0"], r["y1"])
                    vals["same_defect"] += a_same[gm].tolist(); vals["any_defect"] += a_any[gm].tolist()
                    vals["same_normal"] += a_same[~gm][::7].tolist(); vals["any_normal"] += a_any[~gm][::7].tolist()
                else:
                    vals["same_normal"] += a_same.ravel()[::7].tolist(); vals["any_normal"] += a_any.ravel()[::7].tolist()
        stats = {}
        for mode in ("same", "any"):
            n = np.array(vals[f"{mode}_normal"]); d = np.array(vals[f"{mode}_defect"])
            stats[mode] = dict(normal_median=float(np.median(n)), normal_p99=float(np.percentile(n, 99)), normal_std=float(n.std()),
                               defect_median=float(np.median(d)), defect_p25=float(np.percentile(d, 25)),
                               margin=float(np.median(d) / max(np.percentile(n, 99), 1e-9)),
                               frac_defect_above_normal_p99=float((d > np.percentile(n, 99)).mean()),
                               n_normal=int(n.size), n_defect=int(d.size))
            print(f"(a) {mode:4} 正常 median={stats[mode]['normal_median']:.4f} P99={stats[mode]['normal_p99']:.4f} | "
                  f"缺陷 median={stats[mode]['defect_median']:.4f} | 裕度={stats[mode]['margin']:.2f} | 缺陷patch超过正常P99的比例={stats[mode]['frac_defect_above_normal_p99']:.0%}", flush=True)
        R["a_dispersion"] = dict(stats=stats, hist=dict(
            same_normal=np.histogram(vals["same_normal"], bins=60, range=(0, 0.6))[0].tolist(),
            any_normal=np.histogram(vals["any_normal"], bins=60, range=(0, 0.6))[0].tolist(),
            same_defect=np.histogram(vals["same_defect"], bins=60, range=(0, 0.6))[0].tolist(),
            any_defect=np.histogram(vals["any_defect"], bins=60, range=(0, 0.6))[0].tolist(), edges=np.linspace(0, 0.6, 61).tolist()))

    # ================= (b) 参考帧数曲线 =================
    if "b" not in a.skip:
        curve = []
        for kref in (1, 2, 3, 4, 6, 8, 999):
            reps = []
            for rep in range(5 if kref != 999 else 1):
                all_recs, all_an = [], []
                for c in cams:
                    sub, F = F_by[c]
                    for k, r in enumerate(sub):
                        others = [i for i in range(len(sub)) if i != k]          # 与基准一致：参考池=其他全部帧（含缺陷帧）
                        ref = others if kref == 999 else list(rng.choice(others, min(kref, len(others)), replace=False))
                        all_recs.append(r); all_an.append(anom_same(F, k, ref))
                reps.append(evaluate(all_recs, all_an))
            agg = {m: float(np.mean([x[m] for x in reps if x[m] is not None])) for m in ("hit", "auroc", "rec95", "fpr95")}
            agg["kref"] = "全部" if kref == 999 else kref; agg["ndef"] = reps[0]["ndef"]; curve.append(agg)
            print(f"(b) 参考 {agg['kref']:>3} 帧: 定位 {agg['hit']:.1f}/{agg['ndef']}  AUROC {agg['auroc']:.1f}  P95召回 {agg['rec95']:.0f}%", flush=True)
        R["b_refcount"] = curve

    # ================= (c) 抖动鲁棒性 + 邻域松弛 =================
    if "c" not in a.skip:
        rows = []
        for dx in (0, 3, 7, 14, 28):
            for mode in ("same", "neigh"):
                all_recs, all_an = [], []
                for c in cams:
                    sub, F = F_by[c]
                    subq = [dict(r) for r in sub]
                    Fq = F if dx == 0 else featurize(model, subq, shift=(dx, 0))   # 只平移查询帧
                    for k, r in enumerate(subq):
                        ref = [i for i in range(len(sub)) if i != k]
                        Fmix = F.copy(); Fmix[k] = Fq[k]
                        all_recs.append(r); all_an.append((anom_same if mode == "same" else anom_neigh)(Fmix, k, ref))
                    del Fq
                e = evaluate(all_recs, all_an); e.update(dx=dx, mode=mode); rows.append(e)
                print(f"(c) dx={dx:2d}px {mode:5}: 定位 {e['hit']}/{e['ndef']}  AUROC {e['auroc']}  P95召回 {e['rec95']}%  误报 {e['fpr95']}%  {time.time()-t0:.0f}s", flush=True)
        R["c_jitter"] = rows
    del F_by

    # ================= 白纱回归（可选） =================
    if a.white:
        D.BAND_Y0, D.BAND_Y1 = a.white_band
        files = sorted(a.white.glob("2025-*.jpg"))
        wrecs = [dict(path=p, dets=[], cam="W" + p.stem.split("-")[-1]) for p in files]
        wc = sorted(set(r["cam"] for r in wrecs)); vals = dict(same=[], any=[]); all_recs, all_an = [], []
        for c in wc:
            sub = [r for r in wrecs if r["cam"] == c]
            if len(sub) < 3: continue
            F = featurize(model, sub)
            for k in range(len(sub)):
                ref = [i for i in range(len(sub)) if i != k]
                a_s = anom_same(F, k, ref); vals["same"] += a_s.ravel()[::7].tolist(); vals["any"] += anom_any(F, k, ref).ravel()[::7].tolist()
                all_recs.append(sub[k]); all_an.append(a_s)
        s = [D.score_and_locate(an, r["hw"], r["y0"], r["y1"])[0] for r, an in zip(all_recs, all_an)]
        R["white"] = dict(n=len(s), scores=s, band=a.white_band,
                          same=dict(normal_median=float(np.median(vals["same"])), normal_p99=float(np.percentile(vals["same"], 99))),
                          any=dict(normal_median=float(np.median(vals["any"])), normal_p99=float(np.percentile(vals["any"], 99))))
        print(f"白纱 {len(s)} 帧正常：图级分数 中位 {np.median(s):.0f} 最大 {max(s):.0f}；同位置正常异常度 median={R['white']['same']['normal_median']:.4f} P99={R['white']['same']['normal_p99']:.4f}；"
              f"任意位置 median={R['white']['any']['normal_median']:.4f} P99={R['white']['any']['normal_p99']:.4f}", flush=True)

    json.dump(R, open(OUT / "mech_results.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n已写出 {OUT/'mech_results.json'}  用时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
