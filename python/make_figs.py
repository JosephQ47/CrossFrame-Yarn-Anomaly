# -*- coding: utf-8 -*-
"""生成仓库配图。

  assets/fig1_基准指标.png   方法对比 / 分数分布 / ROC / PRO 曲线
  assets/fig2_检出示例.jpg   黑纱基准叠加图拼图（含唯一漏检案例）
  assets/fig3_红纱失败.jpg   红纱负结果叠加图
  assets/fig4_消融.png       top-k 与取层数消融

依赖 data/metrics_黑纱基准.json、data/scores_per_frame.csv、data/pro_curve.csv
（由 metrics_full.py 产生）与黑纱基准叠加图目录。缺文件时自动跳过对应图。
"""
import csv, json, os
from pathlib import Path
os.environ.setdefault("PYTHONUTF8", "1")
import numpy as np, cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from sklearn.metrics import roc_curve

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
ASSETS = ROOT / "assets"; ASSETS.mkdir(exist_ok=True)
OVERLAY = Path(r"D:\研究生课程\课题\色纱研究\黑纱评估基准\检测器叠加图")
F1_MONTAGE = Path(r"D:\研究生课程\课题\色纱研究\F1红纱重标\跨帧参考结果\跨帧参考_前12分帧叠加.jpg")

INK, MUT, OK, BAD, GREY = "#1a1e22", "#5b6670", "#2f7d54", "#c0392b", "#b6bcc2"


def cjk():
    for n in ("Microsoft YaHei", "SimHei", "SimSun"):
        try:
            font_manager.findfont(n, fallback_to_default=False)
            plt.rcParams["font.sans-serif"] = [n]
            plt.rcParams["axes.unicode_minus"] = False
            return
        except Exception:
            pass


def clean(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=9, colors=INK)


def fig1():
    M = json.load(open(DATA / "metrics_黑纱基准.json", encoding="utf-8"))
    rows = list(csv.DictReader(open(DATA / "scores_per_frame.csv", encoding="utf-8-sig")))
    s = np.array([float(r["score"]) for r in rows])
    l = np.array([int(r["defect"]) for r in rows])
    neg, pos = s[l == 0], s[l == 1]
    g, loc = M["图级"], M["定位"]
    thr = g["产线定阈"]["阈值(纯正常P95)"]
    hit, ndef = loc["点命中率"].split("/")
    hit, ndef = int(hit), int(ndef)

    fig, axes = plt.subplots(1, 4, figsize=(19.2, 4.4))

    # (a) 与此前所有方法对比
    ax = axes[0]
    names = ["纯白纱\n基线", "FDA\n域适应", "合成\n改色", "哑光\n改色", "深色\n改色",
             "真实\n微调", "跨帧参考\n(本方法)"]
    vals = [0] * 6 + [hit]
    ax.bar(range(7), vals, color=[GREY] * 6 + [OK], width=.68, edgecolor="white", lw=1.1)
    for i, v in enumerate(vals):
        ax.text(i, v + .35, f"{v}/{ndef}", ha="center", fontsize=10.5,
                fontweight="bold" if v else "normal", color=INK if v else MUT)
    ax.set_xticks(range(7)); ax.set_xticklabels(names, fontsize=8.4)
    ax.set_ylabel("缺陷定位命中数", fontsize=10.2); ax.set_ylim(0, ndef * 1.2)
    ax.set_title("(a) 此前 6 类方法全部 0 命中", fontsize=11, pad=8)

    # (b) 图级分数分布
    ax = axes[1]
    rng = np.random.default_rng(0)
    ax.scatter(rng.normal(0, .055, neg.size), neg, s=25, color=GREY, alpha=.85,
               edgecolor="white", lw=.5, label=f"正常帧 n={neg.size}", zorder=3)
    ax.scatter(rng.normal(1, .055, pos.size), pos, s=42, color=BAD, alpha=.9,
               edgecolor="white", lw=.6, label=f"缺陷帧 n={pos.size}", zorder=3)
    ax.axhline(thr, color=OK, ls="--", lw=1.5, zorder=2)
    ax.text(1.36, thr, f" 定阈 {thr:.1f}\n(纯正常帧 P95)", va="center", fontsize=8.4, color=OK)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["正常", "缺陷"], fontsize=10)
    ax.set_xlim(-.45, 1.95)
    ax.set_ylabel("图级异常分数（连通域面积 / patch）", fontsize=9.3)
    ax.set_title(f"(b) 定阈下 召回 {g['产线定阈']['召回']:.0f}% / 精确率 "
                 f"{g['产线定阈']['精确率']:.0f}%", fontsize=11, pad=8)
    ax.legend(fontsize=8.2, frameon=False, loc="upper left")

    # (c) ROC + 两个业务工作点
    ax = axes[2]
    fpr, tpr, _ = roc_curve(l, s)
    ax.plot(fpr, tpr, color=OK, lw=2.2, zorder=3)
    ax.plot([0, 1], [0, 1], ls=":", color=GREY, lw=1.2)
    ax.fill_between(fpr, tpr, alpha=.10, color=OK)
    ax.scatter([0.05], [g["TPR@FPR5%"] / 100], s=64, color=BAD, zorder=5,
               edgecolor="white", lw=1)
    ax.annotate(f"误报 5% 时\n召回 {g['TPR@FPR5%']:.0f}%", (0.05, g["TPR@FPR5%"] / 100),
                textcoords="offset points", xytext=(20, -52), fontsize=8.6, color=INK)
    ax.scatter([g["FPR@TPR100%"] / 100], [1.0], s=64, color=BAD, zorder=5,
               edgecolor="white", lw=1)
    ax.annotate(f"一个不漏时\n误报 {g['FPR@TPR100%']:.1f}%", (g["FPR@TPR100%"] / 100, 1.0),
                textcoords="offset points", xytext=(26, -14), fontsize=8.6, color=INK)
    ax.set_xlabel("误报率 FPR", fontsize=9.6); ax.set_ylabel("召回率 TPR", fontsize=9.6)
    ax.set_xlim(-.02, 1.02); ax.set_ylim(0, 1.05)
    ax.set_title(f"(c) 图级 ROC：AUROC {g['AUROC']:.1f} / AP {g['AP(AUPR)']:.1f}",
                 fontsize=11, pad=8)
    ax.grid(alpha=.22, lw=.6); ax.set_axisbelow(True)

    # (d) PRO 曲线
    ax = axes[3]
    pc = np.loadtxt(DATA / "pro_curve.csv", delimiter=",", skiprows=1)
    pro_key = [k for k in loc if k.startswith("PRO")][0]
    ax.plot(pc[:, 0], pc[:, 1], color=OK, lw=2.2)
    ax.fill_between(pc[:, 0], pc[:, 1], alpha=.12, color=OK)
    ax.set_xlim(0, 0.3); ax.set_ylim(0, 1.02)
    ax.set_xlabel("正常像素误报率 FPR", fontsize=9.6)
    ax.set_ylabel("缺陷区域平均覆盖率", fontsize=9.6)
    ax.set_title(f"(d) PRO {loc[pro_key]:.1f}　像素AUROC {loc['像素AUROC(框掩膜近似)']:.1f}",
                 fontsize=11, pad=8)
    ax.grid(alpha=.22, lw=.6); ax.set_axisbelow(True)

    for a in axes:
        clean(a)
    fig.suptitle("DINOv2 多层特征 + 固定机位跨帧同位置参考 · 黑纱基准"
                 f"（{M['数据']['总帧']} 帧 / {M['数据']['缺陷帧']} 缺陷 / "
                 f"{M['数据']['相机数']} 机位，零训练零缺陷样本）", fontsize=12.6, y=1.02)
    fig.text(0.5, -0.05,
             "(d) 的像素级 GT 由矩形框近似（框内正常像素被当作缺陷像素），偏保守，"
             "不可与 MVTec AD 等精细掩膜数据集横比；像素统计仅在筘齿带内进行。"
             f"缺陷实例仅 {M['数据']['缺陷帧']} 个，{hit}/{ndef} 的 95% Wilson 区间约 [0.72, 0.99]。",
             ha="center", fontsize=8.6, color=MUT)
    fig.tight_layout()
    p = ASSETS / "fig1_基准指标.png"
    fig.savefig(p, dpi=185, facecolor="white", bbox_inches="tight")
    print(f"已写出 {p}")


def fig4():
    """消融：数据取自 data/sweep_results.csv（早于面积上限引入，故 AUROC 上限 96.4）。"""
    rs = list(csv.DictReader(open(DATA / "sweep_results.csv", encoding="utf-8-sig")))
    base = [r for r in rs if r["th"] == "448" and r["band"] == "(0.34, 0.6)"]
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.1))

    # (a) top-k
    sub = sorted([r for r in base if r["layers"] == "8"], key=lambda r: int(r["topk"]))
    ax = axes[0]
    x = [int(r["topk"]) for r in sub]
    ax.plot(range(len(x)), [float(r["rec95"]) * 100 for r in sub], "-o", color=OK, lw=2, ms=7,
            label="产线定阈召回")
    ax.plot(range(len(x)), [float(r["auroc"]) for r in sub], "-s", color=GREY, lw=1.6, ms=6,
            label="图级 AUROC")
    for i, r in enumerate(sub):
        ax.annotate(f"{float(r['rec95'])*100:.0f}%", (i, float(r["rec95"]) * 100),
                    textcoords="offset points", xytext=(0, 9), ha="center", fontsize=9, color=INK)
    ax.set_xticks(range(len(x))); ax.set_xticklabels([f"top-{k}" for k in x], fontsize=9.6)
    ax.set_ylim(50, 105); ax.set_ylabel("%", fontsize=10)
    ax.set_title("(a) 参考帧数：只取最相似 1 帧最抗污染", fontsize=10.6, pad=8)
    ax.legend(fontsize=8.4, frameon=False, loc="lower left")

    # (b) 取层数
    ax = axes[1]
    lay = ["4", "8", "12"]
    au = [float([r for r in base if r["layers"] == L and r["topk"] == "1"][0]["auroc"]) for L in lay]
    ax.bar(range(3), au, color=[GREY, OK, GREY], width=.55, edgecolor="white", lw=1.1)
    for i, v in enumerate(au):
        ax.text(i, v + .25, f"{v:.1f}", ha="center", fontsize=10, color=INK)
    ax.set_xticks(range(3)); ax.set_xticklabels([f"最后 {L} 层" for L in lay], fontsize=9.6)
    ax.set_ylim(88, 99); ax.set_ylabel("图级 AUROC", fontsize=10)
    ax.set_title("(b) 取层数：8 层最优（多层是主要增益）", fontsize=10.6, pad=8)

    # (c) 面积上限（来自专项扫参，记录在 docs/方法说明.md）
    ax = axes[2]
    caps = ["≤42", "45", "50", "55", "无上限"]
    au2 = [92.8, 98.0, 98.2, 98.2, 96.4]
    ax.plot(range(5), au2, "-o", color=OK, lw=2, ms=7)
    ax.scatter([3], [98.2], s=150, facecolor="none", edgecolor=BAD, lw=2, zorder=5)
    ax.annotate("采用 55\n（离断崖最远，\n且低于人手块 57）", (3, 98.2),
                textcoords="offset points", xytext=(-30, -52), fontsize=8.4, color=BAD)
    ax.set_xticks(range(5)); ax.set_xticklabels(caps, fontsize=9.6)
    ax.set_ylim(91, 99.5); ax.set_ylabel("图级 AUROC", fontsize=10)
    ax.set_xlabel("异常块面积上限（patch 数）", fontsize=9.6)
    ax.set_title("(c) 面积上限：滤人手入侵，≤42 处断崖", fontsize=10.6, pad=8)
    ax.grid(axis="y", alpha=.22, lw=.6); ax.set_axisbelow(True)

    for a in axes:
        clean(a)
    fig.suptitle("关键设计消融（黑纱基准 71 帧 / 16 缺陷）", fontsize=12.2, y=1.02)
    fig.text(0.5, -0.06,
             "(a)(b) 取自 data/sweep_results.csv，该轮扫参早于面积上限引入，故 AUROC 上限为 96.4；"
             "(c) 为面积上限专项扫参，最终配置 98.2。",
             ha="center", fontsize=8.6, color=MUT)
    fig.tight_layout()
    p = ASSETS / "fig4_消融.png"
    fig.savefig(p, dpi=185, facecolor="white", bbox_inches="tight")
    print(f"已写出 {p}")


def montage(paths, cols, out, tile_w=560, tag=None):
    ims = []
    for i, p in enumerate(paths):
        im = cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)
        if im is None:
            continue
        h = int(im.shape[0] * tile_w / im.shape[1])
        im = cv2.resize(im, (tile_w, h))
        t = tag[i] if tag else p.stem.split("_")[0]
        c = (60, 175, 90) if t.startswith("HIT") else (50, 50, 210)
        cv2.rectangle(im, (0, 0), (tile_w - 1, h - 1), c, 6)
        cv2.rectangle(im, (0, 0), (168, 34), c, -1)
        cv2.putText(im, t, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, .72, (255, 255, 255), 2, cv2.LINE_AA)
        ims.append(im)
    if not ims:
        return
    hh = max(i.shape[0] for i in ims)
    ims = [cv2.copyMakeBorder(i, 0, hh - i.shape[0], 0, 0, cv2.BORDER_CONSTANT,
                              value=(255, 255, 255)) for i in ims]
    rws = [np.hstack(ims[i:i + cols]) for i in range(0, len(ims), cols)]
    w = max(r.shape[1] for r in rws)
    rws = [cv2.copyMakeBorder(r, 0, 0, 0, w - r.shape[1], cv2.BORDER_CONSTANT,
                              value=(255, 255, 255)) for r in rws]
    grid = np.vstack(rws)
    cv2.imencode(".jpg", grid, [int(cv2.IMWRITE_JPEG_QUALITY), 86])[1].tofile(str(out))
    print(f"已写出 {out}  {grid.shape[1]}x{grid.shape[0]}")


def fig2():
    if not OVERLAY.exists():
        print("跳过 fig2：无叠加图目录"); return
    pick = ["HIT_DI2_Cam1_2026-07-18_10-11-40-176", "HIT_DI2_Cam2_2026-07-18_10-11-45-634",
            "HIT_DI2_Cam4_2026-07-18_10-11-55-179", "HIT_DI2_Cam5_2026-07-18_10-11-56-689",
            "HIT_DI2_Cam6_2026-07-18_10-11-55-178", "MISS_DI2_Cam3_2026-07-18_10-14-15-409"]
    ps = [OVERLAY / f"{n}.jpg" for n in pick if (OVERLAY / f"{n}.jpg").exists()]
    montage(ps, 3, ASSETS / "fig2_检出示例.jpg",
            tag=[("HIT " if p.stem.startswith("HIT") else "MISS ") + p.stem.split("_")[2]
                 for p in ps])


def fig3():
    if not F1_MONTAGE.exists():
        print("跳过 fig3：无 F1 叠加拼图"); return
    im = cv2.imdecode(np.fromfile(str(F1_MONTAGE), np.uint8), cv2.IMREAD_COLOR)
    im = cv2.resize(im, (1600, int(im.shape[0] * 1600 / im.shape[1])))
    cv2.imencode(".jpg", im, [int(cv2.IMWRITE_JPEG_QUALITY), 84])[1].tofile(
        str(ASSETS / "fig3_红纱失败.jpg"))
    print(f"已写出 {ASSETS/'fig3_红纱失败.jpg'}")


if __name__ == "__main__":
    cjk(); fig2(); fig3(); fig4()
    if (DATA / "metrics_黑纱基准.json").exists():
        fig1()
    else:
        print("缺 data/metrics_黑纱基准.json，先跑 metrics_full.py")
