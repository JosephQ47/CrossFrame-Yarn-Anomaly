# -*- coding: utf-8 -*-
"""生成不使用热力图的检测结果示例：原图 vs 最终连通域定位。"""
from ast import literal_eval
import csv
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager, patches
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
IMAGE = Path(r"D:\dataset\black_fabric\DI2_Cam3_2026-07-18_10-14-15-409.jpg")
TARGET = IMAGE.stem
SCORES = ROOT / "data" / "scores_per_frame.csv"
OUT = ROOT / "assets" / "fig6_黑纱Cam3_无热力图识别结果.jpg"
# 该帧原检测器保存的连通域重心；scores_per_frame.csv 目前只保存外接框，未保存重心。
PRED_CENTER = (493, 846)


def read_bgr(path):
    return cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)


def main():
    for name in ("Microsoft YaHei", "SimHei"):
        try:
            font_manager.findfont(name, fallback_to_default=False)
            plt.rcParams["font.sans-serif"] = [name]
            plt.rcParams["axes.unicode_minus"] = False
            break
        except Exception:
            pass

    with SCORES.open(encoding="utf-8-sig", newline="") as f:
        row = next(r for r in csv.DictReader(f) if r["file"] == TARGET)
    gt_boxes = literal_eval(row["gt"])
    score = float(row["score"])
    hit = bool(int(row["hit"]))

    bgr = read_bgr(IMAGE)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    crop_y0, crop_y1 = int(h * 0.30), int(h * 0.63)

    fig, axes = plt.subplots(1, 2, figsize=(15.2, 4.5), constrained_layout=True)
    for ax in axes:
        ax.imshow(rgb[crop_y0:crop_y1])
        ax.set_xlim(0, w)
        ax.set_ylim(crop_y1 - crop_y0, 0)
        ax.axis("off")

    axes[0].set_title("(a) 原始图像", fontsize=13, pad=9)
    axes[1].set_title("(b) 最终连通域定位（不显示热力图）", fontsize=13, pad=9)

    # 人工标注：白框；算法定位输出：所选连通域的中心点。
    for box in gt_boxes:
        x0, y0, x1, y1 = box
        axes[1].add_patch(patches.Rectangle(
            (x0, y0 - crop_y0), x1 - x0, y1 - y0,
            fill=False, edgecolor="white", linewidth=3.0, linestyle="--",
        ))
    px, py = PRED_CENTER
    axes[1].add_patch(patches.Circle(
        (px, py - crop_y0), radius=22,
        fill=False, edgecolor="#d62728", linewidth=3.2,
    ))
    axes[1].plot(px, py - crop_y0, marker="+", color="#d62728", markersize=12, markeredgewidth=2.5)

    status = "HIT" if hit else "MISS"
    axes[1].text(
        0.015, 0.965, f"score={score:.0f} patches   {status}",
        transform=axes[1].transAxes, va="top", ha="left", fontsize=12,
        color="white", bbox=dict(facecolor="#1a1a1a", alpha=0.78, edgecolor="none", pad=5),
    )
    axes[1].plot([], [], marker="o", markerfacecolor="none", markeredgecolor="#d62728",
                 markeredgewidth=2.5, linestyle="none", label="预测定位点")
    axes[1].plot([], [], color="white", linewidth=3, linestyle="--", label="人工标注")
    leg = axes[1].legend(loc="lower right", frameon=True, fontsize=10)
    leg.get_frame().set_facecolor("#1a1a1a")
    leg.get_frame().set_alpha(0.80)
    for text in leg.get_texts():
        text.set_color("white")

    fig.suptitle("跨帧异常检测结果：仅展示最终判定，不使用彩色热力图", fontsize=15)
    fig.savefig(OUT, dpi=180, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(OUT)


if __name__ == "__main__":
    main()
