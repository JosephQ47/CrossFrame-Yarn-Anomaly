# -*- coding: utf-8 -*-
"""P1 出图：校准曲线、分数分布与各规则阈值、样本量、基线在保形规则下的召回。"""
import sys, json, csv, math
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"; OUT = HERE.parent / "assets"
sys.path.insert(0, str(HERE))
from figstyle import plt, OI, ygrid, panel   # noqa: E402

R = json.load(open(DATA / "p1_results.json", encoding="utf-8"))
ALPHAS = [float(a) for a in R["black_pooled"]["CONF"].keys()]
rows = [r for r in csv.DictReader(open(DATA / "scores_per_frame.csv", encoding="utf-8-sig")) if "副本" not in r["file"]]
neg = np.array([float(r["score"]) for r in rows if r["defect"] == "0"]); pos = np.array([float(r["score"]) for r in rows if r["defect"] == "1"])
white = np.array(R["white"]["scores"])


def conf_thr(cal, a):
    """保形规则等价阈值：最小的 s 使 (1+#{cal≥s})/(n+1) ≤ α。"""
    c = np.sort(cal)[::-1]; k = math.floor(a * (len(c) + 1) - 1)   # 允许的 #{cal ≥ s}
    if k < 0: return float("inf")
    return float(c[k]) + 1e-9 if k < len(c) else 0.0                 # s 需严格大于第 k+1 大的校准分


# ---------------- 图1 校准曲线 ----------------
fig, axes = plt.subplots(1, 2, figsize=(9, 3.7))
ax = axes[0]
ax.plot([0, 0.32], [0, 0.32], color="#999999", lw=0.8, ls="--", zorder=1, label="名义 = 实测")
for key, col, mk, lab in (("black_pooled", OI["blue"], "o", "保形 · 合并六机位校准 (n=52)"),
                          ("black_percam", OI["sky"], "s", "保形 · 按机位校准 (n≈8)")):
    ys = [R[key]["CONF"][str(a)]["fpr"] for a in ALPHAS]
    ax.plot(ALPHAS, ys, "-" + mk, color=col, lw=1.4, ms=4.5, mec="white", mew=0.6, zorder=3, label=lab)
ev = [(a, R["black_pooled"]["EVT"][str(a)]["fpr"]) for a in ALPHAS if np.isfinite(R["black_pooled"]["EVT"][str(a)].get("fpr", float("nan")))]
ax.plot([a for a, _ in ev], [f for _, f in ev], "-^", color=OI["green"], lw=1.4, ms=4.5, mec="white", mew=0.6, zorder=3, label="极值理论 POT · 合并校准")
for key, mk, lab in (("MAD", "D", "中位数+3.5MAD"), ("P95", "X", "P95 分位")):
    for sch, col in (("black_pooled", OI["verm"]), ("black_percam", OI["orange"])):
        ax.scatter([np.nan], [np.nan])  # 占位保持配色顺序
        ax.scatter([R[sch][key]["fpr"]], [R[sch][key]["fpr"]], marker=mk, s=42, color=col, zorder=4,
                   edgecolor="white", linewidth=0.6)
ax.scatter([], [], marker="D", color=OI["verm"], label="MAD（合并 / 按机位）")
ax.scatter([], [], marker="X", color=OI["verm"], label="P95（合并 / 按机位）")
ax.set_xlabel("名义误报率 α（保形 / POT 的风险参数）"); ax.set_ylabel("留一法实测误报率")
ax.set_xlim(0, 0.32); ax.set_ylim(0, 0.32); ygrid(ax); panel(ax, "(a)")
ax.text(0.30, 0.02, "MAD/P95 无 α 参数，画在对角线上：\n横坐标即其实测误报率", ha="right", va="bottom", fontsize=7, color="#555")
ax = axes[1]
for key, col, mk, lab in (("black_pooled", OI["blue"], "o", "保形 · 合并校准"), ("black_percam", OI["sky"], "s", "保形 · 按机位校准")):
    ys = [R[key]["CONF"][str(a)]["recall"] for a in ALPHAS]
    ax.plot(ALPHAS, ys, "-" + mk, color=col, lw=1.4, ms=4.5, mec="white", mew=0.6, zorder=3, label=lab)
ev = [(a, R["black_pooled"]["EVT"][str(a)]["recall"]) for a in ALPHAS if np.isfinite(R["black_pooled"]["EVT"][str(a)].get("recall", float("nan")))]
ax.plot([a for a, _ in ev], [f for _, f in ev], "-^", color=OI["green"], lw=1.4, ms=4.5, mec="white", mew=0.6, zorder=3, label="极值理论 POT · 合并校准")
for sch, col in (("black_pooled", OI["verm"]), ("black_percam", OI["orange"])):
    ax.scatter([R[sch]["MAD"]["fpr"]], [R[sch]["MAD"]["recall"]], marker="D", s=42, color=col, zorder=4, edgecolor="white", linewidth=0.6)
    ax.scatter([R[sch]["P95"]["fpr"]], [R[sch]["P95"]["recall"]], marker="X", s=42, color=col, zorder=4, edgecolor="white", linewidth=0.6)
amin = R["black_percam"]["CONF"]["0.05"]["alpha_min"]
ax.axvline(amin, color=OI["sky"], lw=0.8, ls=":"); ax.text(amin + 0.004, 0.05, f"按机位 α_min=1/(n+1)={amin:.2f}", fontsize=7, color=OI["sky"])
ax.set_xlabel("名义误报率 α（MAD/P95 取其实测误报率）"); ax.set_ylabel("缺陷召回（16 个）")
ax.set_xlim(0, 0.32); ax.set_ylim(-0.03, 1.05); ygrid(ax); panel(ax, "(b)")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.04))
fig.tight_layout(rect=(0, 0, 1, 0.92)); fig.savefig(OUT / "fig11_定阈校准曲线.png"); plt.close(fig)

# ---------------- 图2 分数分布与阈值 ----------------
fig, ax = plt.subplots(figsize=(7.2, 3.8))
rng = np.random.default_rng(0)
for i, (arr, col, lab) in enumerate(((neg, OI["gray"], f"黑纱正常帧 (n={len(neg)})"), (pos, OI["verm"], f"黑纱缺陷帧 (n={len(pos)})"),
                                     (white, OI["sky"], f"白纱正常帧 (n={len(white)})"))):
    ax.scatter(i + rng.uniform(-0.18, 0.18, len(arr)), arr, s=18, color=col, alpha=0.85, edgecolor="white", linewidth=0.4, zorder=3, label=lab)
thrs = [("MAD 合并", R["black_pooled"]["MAD"]["thr_median"], OI["verm"], "-"),
        ("P95 合并", R["black_pooled"]["P95"]["thr_median"], OI["orange"], "-"),
        ("保形 α=5%", conf_thr(neg, 0.05), OI["blue"], "--"),
        ("保形 α=10%", conf_thr(neg, 0.10), OI["blue"], ":"),
        ("POT α=5%", R["black_pooled"]["EVT"]["0.05"]["thr"], OI["green"], "--"),
        ("POT α=1%", R["black_pooled"]["EVT"]["0.01"]["thr"], OI["green"], ":")]
thrs = [t for t in thrs if np.isfinite(t[1])]
# 标签防重叠：按阈值排序，相邻标签 y 至少隔 1.8
ys = [t[1] for t in sorted(thrs, key=lambda z: z[1])]; lab_y = []
for y in ys:
    lab_y.append(y if not lab_y or y - lab_y[-1] >= 1.8 else lab_y[-1] + 1.8)
pos_y = dict(zip(ys, lab_y))
for name, t, col, ls in thrs:
    ax.axhline(t, color=col, lw=1.0, ls=ls, zorder=2)
    ax.annotate(f"{name} {t:.1f}", xy=(2.52, t), xytext=(2.6, pos_y[t]), fontsize=7, color=col, va="center", ha="left",
                arrowprops=dict(arrowstyle="-", color=col, lw=0.5) if abs(pos_y[t] - t) > 0.3 else None)
ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["黑纱正常", "黑纱缺陷", "白纱正常"]); ax.set_xlim(-0.5, 3.6)
ax.set_ylabel("图级异常分数（最大连通域面积，patch 数）"); ygrid(ax)
ax.legend(loc="upper left")
fig.tight_layout(); fig.savefig(OUT / "fig13_分数分布与阈值.png"); plt.close(fig)

# ---------------- 图3 样本量 ----------------
fig, ax = plt.subplots(figsize=(6.2, 3.6))
al = np.linspace(0.005, 0.30, 400)
n_tol = np.ceil(np.log(0.05) / np.log(1 - al))
ax.plot(n_tol, al, color=OI["blue"], lw=1.6, label="无分布假设上容忍界：95% 置信声明 FPR≤α 所需正常帧数")
ns = np.arange(3, 700)
ax.plot(ns, 1 / (ns + 1), color=OI["green"], lw=1.6, ls="--", label="保形 p 值最小可声明 α = 1/(n+1)")
for n, lab, col, y, va in ((8, "按机位 n=8", OI["orange"], 0.29, "top"), (27, "白纱 n=27", OI["sky"], 0.29, "top"),
                           (52, "合并 n=52", OI["verm"], 0.29, "top"),
                           (59, "α=5% 需 n≥59", "#777", 0.012, "bottom"), (299, "α=1% 需 n≥299", "#777", 0.012, "bottom")):
    ax.axvline(n, color=col, lw=0.8, ls=":"); ax.text(n * 1.04, y, lab, rotation=90, fontsize=7, color=col, va=va)
ax.set_xscale("log"); ax.set_xlim(3, 700); ax.set_ylim(0, 0.3)
ax.set_xlabel("校准用正常帧数 n（对数）"); ax.set_ylabel("可保证的误报率 α"); ygrid(ax)
ax.legend(loc="upper right", fontsize=7.5)
fig.tight_layout(); fig.savefig(OUT / "fig14_定阈样本量.png"); plt.close(fig)

# ---------------- 图4 基线在保形规则下 ----------------
B = R["baselines_pooled"]
order = ["PaDiM", "PatchCore", "PatchCore-DINOv2", "SimpleNet", "PatchCore-DINOv2 +CC评分", "Ours"]
order = [o for o in order if o in B]
lbl = {"PaDiM": "PaDiM", "PatchCore": "PatchCore", "PatchCore-DINOv2": "PatchCore\n-DINOv2", "SimpleNet": "SimpleNet*",
       "PatchCore-DINOv2 +CC评分": "PatchCore-DINOv2\n+连通域评分", "Ours": "本方法"}
fig, ax = plt.subplots(figsize=(7.4, 3.6)); x = np.arange(len(order)); w = 0.27
for k, (key, col, lab) in enumerate((("0.05", OI["blue"], "保形 α=5%（有保证）"), ("0.1", OI["sky"], "保形 α=10%（有保证）"), ("P95", OI["gray"], "P95 分位（无保证）"))):
    vals = [B[o][key]["recall"] for o in order]
    bars = ax.bar(x + (k - 1) * w, vals, w, color=col, edgecolor="white", linewidth=0.6, zorder=3, label=lab)
    if key != "P95":
        lo = [B[o][key]["recall"] - B[o][key]["recall_ci"][0] for o in order]; hi = [B[o][key]["recall_ci"][1] - B[o][key]["recall"] for o in order]
        ax.errorbar(x + (k - 1) * w, vals, yerr=[lo, hi], fmt="none", ecolor="#444", elinewidth=0.7, capsize=2, zorder=4)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=6.5, color="#2b2b2b")
ax.set_xticks(x); ax.set_xticklabels([lbl[o] for o in order], fontsize=7.5); ax.set_ylabel("缺陷召回（合并六机位留一校准）"); ax.set_ylim(0, 1.15)
ygrid(ax); ax.legend(loc="upper left", ncol=3)
ax.text(0.99, 0.98, "* SimpleNet 的正常帧同时在其训练集与测试集，对其有利", transform=ax.transAxes, ha="right", va="top", fontsize=6.5, color="#555")
fig.tight_layout(); fig.savefig(OUT / "fig15_基线保形召回.png"); plt.close(fig)

# ---------------- 表 ----------------
with open(DATA / "p1_summary.csv", "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f); w.writerow(["校准方式", "规则", "α", "留一误报率", "误报95%CI", "召回", "召回95%CI", "阈值"])
    for sch, sname in (("black_percam", "按机位(n≈8)"), ("black_pooled", "合并(n=52)")):
        e = R[sch]
        for k in ("MAD", "P95"):
            w.writerow([sname, k, "-", f"{e[k]['fpr']:.3f}", f"[{e[k]['fpr_ci'][0]:.3f},{e[k]['fpr_ci'][1]:.3f}]", f"{e[k]['recall']:.3f}", f"[{e[k]['recall_ci'][0]:.3f},{e[k]['recall_ci'][1]:.3f}]", f"{e[k]['thr_median']:.1f}"])
        for a in ALPHAS:
            c = e["CONF"][str(a)]
            w.writerow([sname, "保形", a, f"{c['fpr']:.3f}", f"[{c['fpr_ci'][0]:.3f},{c['fpr_ci'][1]:.3f}]", f"{c['recall']:.3f}", f"[{c['recall_ci'][0]:.3f},{c['recall_ci'][1]:.3f}]", f"{conf_thr(neg, a):.1f}" if sch == "black_pooled" else "-"])
            if e["EVT"].get(str(a)) and np.isfinite(e["EVT"][str(a)].get("fpr", float("nan"))):
                v = e["EVT"][str(a)]
                w.writerow([sname, "极值理论POT", a, f"{v['fpr']:.3f}", f"[{v['fpr_ci'][0]:.3f},{v['fpr_ci'][1]:.3f}]", f"{v['recall']:.3f}", f"[{v['recall_ci'][0]:.3f},{v['recall_ci'][1]:.3f}]", f"{v['thr']:.1f}"])
print("图与表已写出：", sorted(p.name for p in OUT.glob("fig1[1345]*.png")), "p1_summary.csv")
print(f"保形阈值(合并): α=5%→{conf_thr(neg, .05):.1f}  α=10%→{conf_thr(neg, .10):.1f}  α=2%→{conf_thr(neg, .02):.1f}")
print("正常帧最高分：", sorted(neg)[-5:], " 缺陷帧最低分：", sorted(pos)[:5])
