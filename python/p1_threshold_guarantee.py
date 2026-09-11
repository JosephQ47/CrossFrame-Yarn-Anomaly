# -*- coding: utf-8 -*-
"""P1 · 把"中位数 + 3.5×MAD"换成有统计保证的定阈：保形 p 值与极值理论。

数据（全部为已有产物，不重跑检测器）
  - 黑纱基准 69 帧 / 16 缺陷 / 53 正常（黑纱评估基准/scores_per_frame.csv，剔除 2 张字节级重复）
  - 白纱回归 27 帧正常（机理实验/mech_results.json）
  - 4 个标准基线 + 本方法的逐帧分数（黑纱评估基准/sota_baselines_scores.json）

四种定阈规则（阈值 thr，score ≥ thr 报警，与部署脚本一致）
  MAD   thr = med + 3.5 × max(1.4826·MAD, 1)          —— 现行部署规则（deploy_black_detector.robust_thr）
  P95   thr = 校准集 95 分位                             —— 基准报告里的"纯正常 P95"
  CONF  保形 p 值：p(s) = (1 + #{c ∈ 校准集 : c ≥ s}) / (n+1)，p ≤ α 报警
        —— 只要校准帧与测试帧可交换，P(误报) ≤ α 严格成立（Bates 等，Ann. Stat. 2023）
        —— n 帧校准最小可声明的 α = 1/(n+1)
  EVT   峭值超阈法 POT：对校准集超过 u=P70 的部分拟合广义帕累托分布，z_q = u + σ/ξ·((q·n/N_u)^(−ξ) − 1)
        —— Siffer 等 KDD 2017 (SPOT)；分数是整数（连通域面积），拟合前加 U(0,1) 抖动做连续化

两种校准集
  percam  该机位的其他正常帧（≈8~12 帧）—— 部署脚本的真实设定
  pooled  全部机位的其他正常帧（52 帧）  —— 分数是面积计数，跨机位量纲一致，可合并

所有误报率都用留一法（对每个正常帧，校准集不含它自己）；召回用全部正常帧做校准。
"""
import os, sys, csv, json, math
os.environ.setdefault("PYTHONUTF8", "1")
from pathlib import Path
import numpy as np
from scipy.stats import genpareto

HERE = Path(__file__).resolve().parent.parent / "data"      # 仓库内 data/
BASE = None
ALPHAS = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30]
RNG = np.random.default_rng(0)


# ------------------------------------------------------------------ 数据
def load_black():
    rows = list(csv.DictReader(open(HERE / "scores_per_frame.csv", encoding="utf-8-sig")))
    rows = [r for r in rows if "副本" not in r["file"]]
    return [dict(file=r["file"], cam=r["cam"], y=int(r["defect"]), s=float(r["score"])) for r in rows]


def load_white():
    return [float(x) for x in json.load(open(HERE / "mech_results_white.json", encoding="utf-8"))["white"]["scores"]]


def load_baselines():
    d = json.load(open(HERE / "sota_baselines_scores.json", encoding="utf-8"))
    return d["labels"], d["scores"]


# ------------------------------------------------------------------ 规则
def thr_mad(c, k=3.5):
    c = np.asarray(c, float); med = np.median(c); mad = 1.4826 * np.median(np.abs(c - med))
    return float(med + k * max(mad, 1.0))


def thr_p95(c):
    return float(np.percentile(np.asarray(c, float), 95))


def conformal_p(s, c):
    c = np.asarray(c, float); return (1.0 + np.sum(c >= s)) / (len(c) + 1.0)


def fit_evt(c, u_q=0.70, n_jit=40, seed=0):
    """POT/GPD 拟合（Siffer 2017）。整数分数加 U(0,1) 抖动做连续化，返回每次抖动的 (u, ξ, σ, n, N_u)。
    拟合与目标风险 q 无关，所以每个校准集只拟合一次，再对所有 α 取阈值。"""
    c = np.asarray(c, float); rng = np.random.default_rng(seed); fits = []
    for _ in range(n_jit):
        cj = c + rng.uniform(0, 1, len(c)); u = np.quantile(cj, u_q); y = cj[cj > u] - u
        if len(y) < 5: continue
        try:
            xi, _, sig = genpareto.fit(y, floc=0)
        except Exception:
            continue
        fits.append((u, xi, sig, len(cj), len(y)))
    return fits


def thr_from_fits(fits, q):
    """z_q = u + σ/ξ·((q·n/N_u)^(−ξ) − 1)，取各次抖动的中位数；q 大于超阈比例时 POT 不适用，返回 nan。"""
    zs = []
    for u, xi, sig, n, nu in fits:
        r = q * n / nu
        if r >= 1: continue
        z = u + (sig / xi) * (r ** (-xi) - 1) if abs(xi) > 1e-6 else u - sig * math.log(r)
        if np.isfinite(z): zs.append(z)
    return float(np.median(zs)) if len(zs) >= max(3, len(fits) // 2) else float("nan")


# ------------------------------------------------------------------ 评估
def boot_ci(flags, n=2000, seed=0):
    flags = np.asarray(flags, float)
    if len(flags) == 0: return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed); m = [rng.choice(flags, len(flags)).mean() for _ in range(n)]
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def evaluate(recs, scheme):
    """scheme ∈ {percam, pooled}。返回各规则在各 α 下的 LOO 误报率与召回（含自助区间）。"""
    normals = [r for r in recs if r["y"] == 0]; defects = [r for r in recs if r["y"] == 1]

    def cal_for(r, exclude_self):
        pool = [x for x in normals if (scheme == "pooled" or x["cam"] == r["cam"])]
        if exclude_self: pool = [x for x in pool if x is not r]
        return np.array([x["s"] for x in pool])

    out = {}
    # 固定规则
    for name, fn in (("MAD", thr_mad), ("P95", thr_p95)):
        fp = [r["s"] >= fn(cal_for(r, True)) for r in normals]
        tp = [r["s"] >= fn(cal_for(r, False)) for r in defects]
        out[name] = dict(fpr=float(np.mean(fp)), fpr_ci=boot_ci(fp), recall=float(np.mean(tp)), recall_ci=boot_ci(tp),
                         thr_median=float(np.median([fn(cal_for(r, False)) for r in defects])))
    # 保形
    out["CONF"] = {}
    for a in ALPHAS:
        fp = [conformal_p(r["s"], cal_for(r, True)) <= a for r in normals]
        tp = [conformal_p(r["s"], cal_for(r, False)) <= a for r in defects]
        n_cal = int(np.median([len(cal_for(r, True)) for r in normals]))
        out["CONF"][str(a)] = dict(fpr=float(np.mean(fp)), fpr_ci=boot_ci(fp), recall=float(np.mean(tp)),
                                   recall_ci=boot_ci(tp), n_cal_median=n_cal, alpha_min=1.0 / (n_cal + 1))
    # 极值理论（只在 pooled 下可拟合）
    out["EVT"] = {}
    if scheme == "pooled":
        fits_full = fit_evt(np.array([x["s"] for x in normals]))
        fits_loo = [fit_evt(cal_for(r, True), n_jit=20, seed=i) for i, r in enumerate(normals)]
        for a in ALPHAS:
            thr_full = thr_from_fits(fits_full, a)
            if not np.isfinite(thr_full):
                out["EVT"][str(a)] = dict(fpr=float("nan"), recall=float("nan"), thr=float("nan")); continue
            fp = []
            for r, fl in zip(normals, fits_loo):
                z = thr_from_fits(fl, a); fp.append(bool(np.isfinite(z) and r["s"] >= z))
            tp = [r["s"] >= thr_full for r in defects]
            out["EVT"][str(a)] = dict(fpr=float(np.mean(fp)), fpr_ci=boot_ci(fp), recall=float(np.mean(tp)),
                                      recall_ci=boot_ci(tp), thr=thr_full)
    return out


def sample_size_table(delta=0.05):
    """无分布假设的上容忍界：以 n 个正常样本的最大值为阈值，P(误报 ≤ α) ≥ 1−δ 需 (1−α)^n ≤ δ。"""
    rows = []
    for a in (0.005, 0.01, 0.02, 0.03, 0.05, 0.10, 0.15, 0.20):
        n_min = math.ceil(math.log(delta) / math.log(1 - a))
        rows.append(dict(alpha=a, n_min_95=n_min))
    return rows


def main():
    black = load_black(); white = load_white(); labels, base_scores = load_baselines()
    n_norm = sum(1 for r in black if r["y"] == 0); n_def = len(black) - n_norm
    cams = sorted({r["cam"] for r in black})
    per_cam_n = {c: sum(1 for r in black if r["cam"] == c and r["y"] == 0) for c in cams}
    print(f"黑纱 {len(black)} 帧 = {n_def} 缺陷 + {n_norm} 正常；每机位正常帧 {per_cam_n}")

    res = dict(n_frames=len(black), n_defect=n_def, n_normal=n_norm, per_cam_normals=per_cam_n,
               black_percam=evaluate(black, "percam"), black_pooled=evaluate(black, "pooled"))

    # ---- 白纱回归：黑纱正常帧的阈值直接套到白纱 27 正常帧；以及白纱自校准的保形 α_min
    bn = np.array([r["s"] for r in black if r["y"] == 0]); w = np.array(white)
    res["white"] = dict(
        n=len(w), scores=white,
        fpr_black_MAD=float(np.mean(w >= thr_mad(bn))), thr_black_MAD=thr_mad(bn),
        fpr_black_P95=float(np.mean(w >= thr_p95(bn))), thr_black_P95=thr_p95(bn),
        fpr_black_CONF={str(a): float(np.mean([conformal_p(s, bn) <= a for s in w])) for a in ALPHAS},
        self_conformal_alpha_min=1.0 / (len(w) - 1 + 1),
        self_conformal_loo_fpr={str(a): float(np.mean([conformal_p(s, np.delete(w, i)) <= a for i, s in enumerate(w)])) for a in ALPHAS},
    )

    # ---- 样本量表
    res["sample_size"] = sample_size_table()
    res["alpha_min_conformal"] = {str(n): 1.0 / (n + 1) for n in (8, 12, 27, 52, 53, 150, 299)}

    # ---- 基线：同一保形规则（pooled LOO）下的召回/误报，α=5% 与 10%
    base = {}
    for m, sc in base_scores.items():
        recs = [dict(file=f, cam=f.split("_")[1], y=int(labels[f]), s=float(sc[f])) for f in labels if f in sc]
        ev = evaluate(recs, "pooled")
        base[m] = {a: dict(recall=ev["CONF"][a]["recall"], recall_ci=ev["CONF"][a]["recall_ci"],
                           fpr=ev["CONF"][a]["fpr"]) for a in ("0.05", "0.1")}
        base[m]["P95"] = dict(recall=ev["P95"]["recall"], fpr=ev["P95"]["fpr"])
        base[m]["MAD"] = dict(recall=ev["MAD"]["recall"], fpr=ev["MAD"]["fpr"])
    res["baselines_pooled"] = base

    json.dump(res, open(HERE / "p1_results.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    # ---- 控制台摘要
    for scheme in ("black_percam", "black_pooled"):
        e = res[scheme]; print(f"\n== {scheme} ==")
        for k in ("MAD", "P95"):
            print(f"  {k:5} 误报 {e[k]['fpr']:.3f} {e[k]['fpr_ci']}  召回 {e[k]['recall']:.3f} {e[k]['recall_ci']}  中位阈值 {e[k]['thr_median']:.1f}")
        for a in ALPHAS:
            c = e["CONF"][str(a)]
            line = f"  CONF α={a:<4} 误报 {c['fpr']:.3f}  召回 {c['recall']:.3f}  (校准 n={c['n_cal_median']}, α_min={c['alpha_min']:.3f})"
            if e["EVT"].get(str(a)):
                v = e["EVT"][str(a)]; line += f"   | EVT 误报 {v['fpr']:.3f} 召回 {v['recall']:.3f} 阈 {v['thr']:.1f}"
            print(line)
    print("\n白纱 27 正常帧：黑纱 MAD 阈 %.1f → 误报 %.3f；黑纱 P95 阈 %.1f → 误报 %.3f" % (
        res["white"]["thr_black_MAD"], res["white"]["fpr_black_MAD"], res["white"]["thr_black_P95"], res["white"]["fpr_black_P95"]))
    print("样本量(95%%置信): " + "  ".join(f"α={r['alpha']:.3f}→n≥{r['n_min_95']}" for r in res["sample_size"]))
    print("\n基线（pooled 保形 α=5%）召回：")
    for m, v in base.items():
        print(f"  {m:28} R={v['0.05']['recall']:.3f} {v['0.05']['recall_ci']} 误报 {v['0.05']['fpr']:.3f}   | α=10% R={v['0.1']['recall']:.3f}")
    print("\n写出", HERE / "p1_results.json")


if __name__ == "__main__":
    main()
