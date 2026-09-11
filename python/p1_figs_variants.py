# -*- coding: utf-8 -*-
"""P1 图5：四口径敏感性分析 + 跨机位时刻热力图。说明"保形 5% 召回只掉在一帧上"。"""
import sys, json, csv, re
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"; OUT = HERE.parent / "assets"
sys.path.insert(0, str(HERE))
from figstyle import plt, OI, ygrid, panel   # noqa: E402

V = json.load(open(DATA / "p1_variants.json", encoding="utf-8"))
cross = V.pop("_cross_camera")
names = list(V.keys())
SHORT = {"A 原口径": "A\n原口径\n69帧/16缺陷", "B 开外观屏蔽": "B\n开外观屏蔽\n69帧/16缺陷",
         "C 假设Cam3漏标": "C\n假设Cam3漏标\n69帧/17缺陷", "D C+剔除人手帧": "D\nC+剔除人手帧\n68帧/17缺陷"}
RULES = [("MAD", "中位数+3.5MAD\n（无保证）", OI["gray"]), ("P95", "P95 分位\n（无保证）", OI["orange"]),
         ("conf5", "保形 α=5%\n（有保证）", OI["blue"]), ("conf10", "保形 α=10%\n（有保证）", OI["sky"]),
         ("pot5", "极值理论 α=5%\n（有保证）", OI["green"])]

fig = plt.figure(figsize=(12, 4.2))
gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1])

# ---- (a) 四口径 × 五规则的召回 ----
ax = fig.add_subplot(gs[0, 0])
x = np.arange(len(names)); w = 0.16
for i, (key, lab, col) in enumerate(RULES):
    vals = [V[n][key].get("recall", np.nan) for n in names]
    bars = ax.bar(x + (i - 2) * w, vals, w, color=col, edgecolor="white", linewidth=0.6, zorder=3, label=lab)
    for n, b, v in zip(names, bars, vals):
        ci = V[n][key].get("recall_ci")
        if ci and np.isfinite(ci[0]):
            ax.errorbar(b.get_x() + b.get_width() / 2, v, yerr=[[v - ci[0]], [ci[1] - v]], fmt="none",
                        ecolor="#555", elinewidth=0.6, capsize=1.5, zorder=4)
        ax.text(b.get_x() + b.get_width() / 2, v + 0.025, f"{v:.2f}", ha="center", va="bottom", fontsize=6, color="#2b2b2b")
ax.set_xticks(x); ax.set_xticklabels([SHORT[n] for n in names], fontsize=7.5)
ax.set_ylabel("缺陷召回（合并六机位留一校准）"); ax.set_ylim(0, 1.42); ygrid(ax)
ax.set_yticks([0, .2, .4, .6, .8, 1.0])
ax.legend(loc="upper center", ncol=5, fontsize=6.5, bbox_to_anchor=(0.5, 1.005))
panel(ax, "(a) 保形 α=5% 的召回只掉在一帧上")
ax.annotate("", xy=(2 - 2 * w, 1.10), xytext=(0 - 2 * w, 1.10),
            arrowprops=dict(arrowstyle="->", color=OI["blue"], lw=1.3))
ax.text(1 - 2 * w, 1.13, "Cam3 那帧改判为缺陷 → 0.56 升到 0.94", ha="center", fontsize=7, color=OI["blue"])

# ---- (b) 跨机位时刻图 ----
ax = fig.add_subplot(gs[0, 1])
times = [c["time"] for c in cross]
cams = ["Cam1", "Cam2", "Cam3", "Cam4", "Cam5", "Cam6"]
M = np.full((len(cams), len(times)), np.nan); Dlab = np.zeros_like(M)
for j, c in enumerate(cross):
    for tok in c["scores"].split():
        cam, val = tok.split(":"); star = val.endswith("*"); val = float(val.rstrip("*"))
        M[int(cam) - 1, j] = val; Dlab[int(cam) - 1, j] = star
im = ax.imshow(M, cmap="YlOrRd", vmin=0, vmax=50, aspect="auto")
for i in range(len(cams)):
    for j in range(len(times)):
        if np.isnan(M[i, j]): continue
        ax.text(j, i, f"{M[i,j]:.0f}", ha="center", va="center", fontsize=6.5,
                color="white" if M[i, j] > 28 else "#222", fontweight="bold" if Dlab[i, j] else "normal")
        if Dlab[i, j]:
            ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False, ec="#1a1a1a", lw=1.6))
for cx, cy, col in ((3, 2, OI["blue"]), (5, 4, OI["verm"])):
    ax.add_patch(plt.Rectangle((cx - .5, cy - .5), 1, 1, fill=False, ec=col, lw=2.4))
for cx, cy, txt, col, dy in ((3, 2, "Cam3 未标却 23 分\n同刻 Cam4/5/6 均为缺陷\n→ 疑似漏标", OI["blue"], -1.5),
                             (5, 4, "Cam5 19 分 = 工人的手\n同刻 Cam3 缺陷帧因手臂漏检", OI["verm"], 1.3)):
    ax.annotate(txt, xy=(cx, cy + (0.45 if dy > 0 else -0.45)), xytext=(len(times) + 0.3, cy + dy),
                fontsize=6.8, color=col, va="center",
                arrowprops=dict(arrowstyle="->", color=col, lw=1.0, shrinkA=0, shrinkB=3))
ax.set_xticks(range(len(times))); ax.set_xticklabels(times, rotation=90, fontsize=6.5)
ax.set_yticks(range(len(cams))); ax.set_yticklabels(cams, fontsize=7.5)
ax.set_xlim(-0.5, len(times) + 5.0)
cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01); cb.set_label("图级异常分数", fontsize=7.5); cb.ax.tick_params(labelsize=7)
panel(ax, "(b) 黑框 = 已标缺陷；高分几乎只出现在缺陷时刻")
fig.tight_layout(); fig.savefig(OUT / "fig12_定阈敏感性与跨机位.png"); plt.close(fig)
print("已写出 fig12_定阈敏感性与跨机位.png")
for n in names:
    print(f"{n:<16} 保形5% R={V[n]['conf5']['recall']:.2f} FP={V[n]['conf5']['fpr']:.3f} 阈={V[n]['thr_conf5']:.0f}")
