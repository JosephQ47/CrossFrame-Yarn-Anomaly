# -*- coding: utf-8 -*-
"""国际期刊风格 matplotlib 配置（与 论文实验补充/scripts/journal_figs.py 同一套，但不带副作用）。
Okabe–Ito 色盲友好配色；中文用微软雅黑放首位（Arial 放首位会让中文变方框）。"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

for fp in [r"C:/Windows/Fonts/arial.ttf", r"C:/Windows/Fonts/arialbd.ttf",
           r"C:/Windows/Fonts/times.ttf", r"C:/Windows/Fonts/msyh.ttc", r"C:/Windows/Fonts/simsun.ttc"]:
    if os.path.exists(fp):
        try:
            fm.fontManager.addfont(fp)
        except Exception:
            pass

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Microsoft YaHei", "Arial", "DejaVu Sans"],
    "font.size": 9, "axes.linewidth": 0.8, "axes.edgecolor": "#2b2b2b", "axes.labelcolor": "#1a1a1a",
    "axes.titlesize": 9.5, "axes.titleweight": "regular", "axes.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.major.width": 0.8, "ytick.major.width": 0.8, "xtick.major.size": 3, "ytick.major.size": 3,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "xtick.color": "#2b2b2b", "ytick.color": "#2b2b2b",
    "legend.fontsize": 8, "legend.frameon": False, "legend.handlelength": 1.4,
    "legend.columnspacing": 1.2, "legend.handletextpad": 0.5,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight", "axes.grid": False,
    "axes.unicode_minus": False,
})
OI = dict(gray="#BBBBBB", blue="#0072B2", green="#009E73", orange="#E69F00",
          verm="#D55E00", sky="#56B4E9", purple="#CC79A7", black="#333333", yellow="#F0E442")


def ygrid(ax):
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color="#E6E6E6", linewidth=0.6, zorder=0)


def panel(ax, s):
    ax.set_title(s, loc="left", fontsize=10, fontweight="bold")
