# -*- coding: utf-8 -*-
"""机理实验成图：(a) 分布收窄 (b) 参考帧数曲线 (c) 抖动鲁棒性与邻域松弛；白纱回归单独一张。"""
import json, os
from pathlib import Path
os.environ.setdefault("PYTHONUTF8", "1")
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = Path(__file__).resolve().parent
INK, MUT, OK, BAD, GREY, BLUE = "#1a1e22", "#5b6670", "#2f7d54", "#c0392b", "#b6bcc2", "#3d7dd6"
for n in ("Microsoft YaHei", "SimHei"):
    try:
        font_manager.findfont(n, fallback_to_default=False); plt.rcParams["font.sans-serif"] = [n]; plt.rcParams["axes.unicode_minus"] = False; break
    except Exception: pass
B = json.load(open(HERE / "mech_results_black.json", encoding="utf-8"))
W = json.load(open(HERE / "mech_results.json", encoding="utf-8")).get("white") if (HERE / "mech_results.json").exists() else None


def clean(ax):
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=9)


import json as _j
SJ = _j.load(open(HERE.parent / "黑纱评估基准" / "sota_baselines_scores.json", encoding="utf-8"))
fig, axes = plt.subplots(1, 4, figsize=(20.4, 4.6), gridspec_kw=dict(width_ratios=[1.2, 1.0, 0.95, 1.0]))
# (a) 分布收窄
ax = axes[0]; h = B["a_dispersion"]["hist"]; st = B["a_dispersion"]["stats"]
e = np.array(h["edges"]); c = (e[:-1] + e[1:]) / 2; w = e[1] - e[0]
def dens(v): v = np.array(v, float); return v / max(v.sum(), 1) / w
ax.fill_between(c, dens(h["any_normal"]), step="mid", alpha=.35, color=GREY, label="正常 patch · 任意位置参考")
ax.fill_between(c, dens(h["same_normal"]), step="mid", alpha=.55, color=BLUE, label="正常 patch · 同位置参考")
ax.plot(c, dens(h["any_defect"]), color=GREY, lw=1.6, ls="--", label="缺陷 patch · 任意位置")
ax.plot(c, dens(h["same_defect"]), color=BAD, lw=2.2, label="缺陷 patch · 同位置")
ax.axvline(st["same"]["normal_p99"], color=BLUE, ls=":", lw=1.4); ax.axvline(st["any"]["normal_p99"], color=GREY, ls=":", lw=1.4)
ax.set_xlabel("patch 异常度  1 − max cos", fontsize=10); ax.set_ylabel("密度", fontsize=10)
ax.set_title(f"(a) patch 级：两种参考系裕度相近（{st['same']['margin']:.2f} vs {st['any']['margin']:.2f}）", fontsize=10.6, pad=8)
ax.legend(fontsize=8, frameon=False)
# (b) 参考帧数
ax = axes[1]; cv = B["b_refcount"]; x = np.arange(len(cv)); lab = [str(r["kref"]) for r in cv]
ax.plot(x, [r["rec95"] for r in cv], "-o", color=OK, lw=2, ms=7, label="P95 定阈召回 (%)")
ax.plot(x, [r["auroc"] for r in cv], "-s", color=BLUE, lw=1.6, ms=6, label="图级 AUROC")
ax.plot(x, [r["hit"] / r["ndef"] * 100 for r in cv], "-^", color=GREY, lw=1.6, ms=6, label="定位命中率 (%)")
ax.set_xticks(x); ax.set_xticklabels(lab, fontsize=9.5); ax.set_xlabel("参考池帧数", fontsize=10); ax.set_ylim(0, 105)
ax.axvline(lab.index("8") if "8" in lab else 5, color=GREY, ls=":", lw=1); ax.text(lab.index("8") - .1, 100, "部署 warm-up=8", fontsize=8.5, color=MUT, ha="right")
ax.set_title("(b) 参考帧越多越稳，≥4 帧后收敛", fontsize=10.6, pad=8); ax.legend(fontsize=8.2, frameon=False, loc="lower right"); ax.grid(axis="y", alpha=.22)
# (c) 抖动
ax = axes[2]; rows = B["c_jitter"]; dxs = sorted(set(r["dx"] for r in rows))
for mode, col, name in (("same", BAD, "严格同位置"), ("neigh", OK, "±1 patch 邻域松弛")):
    ys = [next(r for r in rows if r["dx"] == d and r["mode"] == mode)["rec95"] for d in dxs]
    ax.plot(range(len(dxs)), ys, "-o", color=col, lw=2, ms=7, label=f"{name} · P95 召回")
    ys2 = [next(r for r in rows if r["dx"] == d and r["mode"] == mode)["auroc"] for d in dxs]
    ax.plot(range(len(dxs)), ys2, ":s", color=col, lw=1.3, ms=5, alpha=.8, label=f"{name} · AUROC")
ax.set_xticks(range(len(dxs))); ax.set_xticklabels([f"{d} px\n({d/14:.1f} patch)" for d in dxs], fontsize=9)
ax.set_xlabel("查询帧相对参考帧的水平平移", fontsize=10); ax.set_ylim(0, 105)
ax.set_title("(c) 相机抖动：半个 patch（7 px）即崩，邻域松弛反而更差", fontsize=10.6, pad=8); ax.legend(fontsize=7.8, frameon=False, loc="lower left"); ax.grid(axis="y", alpha=.22)
# (d) 图级：连通域面积分布（来自 sota_baselines_scores.json）
ax = axes[3]; lab_ = SJ["labels"]
def split(m):
    s_ = SJ["scores"][m]; return np.array([s_[f] for f in s_ if lab_[f] == 0]), np.array([s_[f] for f in s_ if lab_[f] == 1])
n1, d1 = split("Ours"); n2, d2 = split("PatchCore-DINOv2 +CC评分")
pos_ = [0, 1, 2.6, 3.6]; data = [n1, d1, n2, d2]; cols = [GREY, BAD, GREY, BAD]
bp = ax.boxplot(data, positions=pos_, widths=.7, patch_artist=True, showfliers=True, medianprops=dict(color=INK, lw=1.6))
for b, c_ in zip(bp["boxes"], cols): b.set(facecolor=c_, alpha=.55, edgecolor=INK)
ax.set_xticks([0.5, 3.1]); ax.set_xticklabels(["同位置参考\n（本方法）", "任意位置参考\n（PatchCore-DINOv2，同评分）"], fontsize=9.2)
ax.set_ylabel("图级分数 = 最大连通域面积 (patch)", fontsize=9.6)
ax.text(0, n1.max() + 1.5, "正常", ha="center", fontsize=8.6, color=MUT); ax.text(1, d1.max() + 1.5, "缺陷", ha="center", fontsize=8.6, color=BAD)
ax.text(2.6, n2.max() + 1.5, "正常", ha="center", fontsize=8.6, color=MUT); ax.text(3.6, d2.max() + 1.5, "缺陷", ha="center", fontsize=8.6, color=BAD)
ax.set_title(f"(d) 图级：同位置让缺陷块更完整（中位 {np.median(d1):.0f} vs {np.median(d2):.0f}）、正常假块更小（P95 {np.percentile(n1,95):.0f} vs {np.percentile(n2,95):.0f}）", fontsize=9.4, pad=8)
ax.grid(axis="y", alpha=.22)
for a in axes: clean(a)
fig.suptitle("\"同位置参考为什么强\"的量化（黑纱基准 69 帧 / 16 缺陷 / 6 机位）", fontsize=12.4, y=1.02)
fig.text(0.5, -0.06, "(a) 正常 patch 每 7 个取 1 个、缺陷 patch 取 GT 框内全部，参考池=同机位其他正常帧；patch 级裕度相近说明优势不在单点可分性。(d) 同一评分规则下，任意位置参考让部分缺陷 patch 在别处找到相似邻居、缺陷块被打碎（最小 4 vs 12），而正常帧的假块反而更大——优势来自缺陷块的空间完整性。(b) k<全部 时随机抽 5 次取均值。(c) 只平移查询帧；DINOv2 patch=14 px。",
         ha="center", fontsize=8.6, color=MUT)
fig.tight_layout(); p = HERE / "机理实验_结果.png"; fig.savefig(p, dpi=185, facecolor="white", bbox_inches="tight"); print("已写出", p)

if W:
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    labs = ["黑纱 · 同位置", "黑纱 · 任意位置", "白纱 · 同位置", "白纱 · 任意位置"]
    med = [st["same"]["normal_median"], st["any"]["normal_median"], W["same"]["normal_median"], W["any"]["normal_median"]]
    p99 = [st["same"]["normal_p99"], st["any"]["normal_p99"], W["same"]["normal_p99"], W["any"]["normal_p99"]]
    x = np.arange(4); ax.bar(x - .19, med, .38, color=[BLUE, GREY, BLUE, GREY], label="正常 patch 异常度中位数")
    ax.bar(x + .19, p99, .38, color=[BLUE, GREY, BLUE, GREY], alpha=.45, label="正常 patch 异常度 P99")
    for xi, (m, q) in enumerate(zip(med, p99)): ax.text(xi - .19, m + .004, f"{m:.3f}", ha="center", fontsize=8.6); ax.text(xi + .19, q + .004, f"{q:.3f}", ha="center", fontsize=8.6)
    ax.set_xticks(x); ax.set_xticklabels(labs, fontsize=9); ax.set_ylabel("1 − max cos", fontsize=10)
    ax.set_title(f"白纱回归：{W['n']} 帧正常，图级分数中位 {np.median(W['scores']):.0f} / 最大 {max(W['scores']):.0f}（黑纱阈值 12.4）", fontsize=10.4, pad=8)
    ax.legend(fontsize=8.4, frameon=False); clean(ax)
    fig.tight_layout(); p = HERE / "白纱回归.png"; fig.savefig(p, dpi=185, facecolor="white", bbox_inches="tight"); print("已写出", p)
