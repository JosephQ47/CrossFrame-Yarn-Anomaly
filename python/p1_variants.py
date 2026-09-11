# -*- coding: utf-8 -*-
"""P1 敏感性分析：保形 α=5% 的召回为何掉到 56%，四种口径下会怎样。

背景：合并六机位校准下，保形 α=5% 的阈值被顶到 19，召回从 93.8%（P95 口径）掉到 56.2%。
顶阈值的是两张标为"正常"的高分帧，人工查看 + 时刻聚合后定性如下：

  Cam3 10-11-55-177（23 分）——同一时刻 10:11:55 有 Cam4=44、Cam5=30、Cam6=17 三帧已标缺陷，
      Cam3 这帧未标；放大图可见 W 形筘齿中部有糊化暗斑。**疑似漏标**，待用户判读。
  Cam5 10-14-15-402（19 分）——放大图可见**工人的手**伸入画面。同一时刻 Cam3=49 是已标缺陷帧
      （其漏检原因即手臂入侵）。属入侵物，不是纱线缺陷。

四种口径：
  A 原口径            69 帧 / 16 缺陷（与基准报告一致）
  B 开外观屏蔽        同上，分数改用 scores_maskon.csv
  C 假设 Cam3 漏标    把 Cam3 10-11-55-177 记为缺陷 → 69 帧 / 17 缺陷
  D C + 剔除人手帧    再移除 Cam5 10-14-15-402 → 68 帧 / 17 缺陷（课题范围只管纱线缺陷）

C 与 D 是**假设性口径**，用于量化"两帧定性若改变，结论会变多少"，不作为最终成绩。
用法： D:/anaconda3/envs/Yolov8/python.exe p1_variants.py
"""
import os, sys, csv, json
os.environ.setdefault("PYTHONUTF8", "1")
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
sys.path.insert(0, str(HERE))
from p1_threshold_guarantee import (evaluate, conformal_p, thr_mad, thr_p95,   # noqa: E402
                                    fit_evt, thr_from_fits, boot_ci, ALPHAS)

CAM3 = "DI2_Cam3_2026-07-18_10-11-55-177"
CAM5 = "DI2_Cam5_2026-07-18_10-14-15-402"


def load(score_col):
    rows = list(csv.DictReader(open(DATA / "scores_maskon.csv", encoding="utf-8-sig")))
    return [dict(file=r["file"], cam=r["cam"], y=int(r["defect"]), s=float(r[score_col])) for r in rows]


def conf_thr(cal, a):
    import math
    c = np.sort(np.asarray(cal, float))[::-1]; k = math.floor(a * (len(c) + 1) - 1)
    if k < 0: return float("inf")
    return float(c[k]) + 1e-9 if k < len(c) else 0.0


def main():
    variants = {}
    A = load("score_maskoff"); variants["A 原口径"] = A
    variants["B 开外观屏蔽"] = load("score_maskon")
    C = [dict(r) for r in A]
    for r in C:
        if r["file"] == CAM3: r["y"] = 1
    variants["C 假设Cam3漏标"] = C
    variants["D C+剔除人手帧"] = [r for r in C if r["file"] != CAM5]

    out = {}
    print(f"{'口径':<16}{'帧数':>5}{'缺陷':>5}  {'MAD':>18}  {'P95':>18}  {'保形5%':>18}  {'保形10%':>18}  {'POT5%':>18}")
    for name, recs in variants.items():
        e = evaluate(recs, "pooled")
        neg = np.array([r["s"] for r in recs if r["y"] == 0])
        row = dict(n=len(recs), n_def=sum(r["y"] for r in recs), **{
            k: dict(fpr=e[k]["fpr"], recall=e[k]["recall"], recall_ci=e[k]["recall_ci"]) for k in ("MAD", "P95")},
            conf5=e["CONF"]["0.05"], conf10=e["CONF"]["0.1"], pot5=e["EVT"].get("0.05", {}),
            thr_conf5=conf_thr(neg, 0.05), thr_conf10=conf_thr(neg, 0.10))
        out[name] = row
        f = lambda d: f"R{d.get('recall', float('nan')):.2f}/FP{d.get('fpr', float('nan')):.3f}"
        print(f"{name:<16}{row['n']:>5}{row['n_def']:>5}  {f(row['MAD']):>18}  {f(row['P95']):>18}  "
              f"{f(row['conf5']):>18}  {f(row['conf10']):>18}  {f(row['pot5']):>18}")

    # 时刻聚合：同一时刻多机位同时高分 = 全局事件（入侵/停机），可作跨机位一致性线索
    import re
    from collections import defaultdict
    ev = defaultdict(list)
    for r in A:
        m = re.search(r"_(\d\d)-(\d\d)-(\d\d)-\d+", r["file"])
        ev[f"{m.group(1)}:{m.group(2)}:{m.group(3)}"].append(r)
    cross = []
    for t, rs in sorted(ev.items()):
        hi = [r for r in rs if r["s"] >= 12.4]          # 12.4 = P95 合并阈值
        cross.append(dict(time=t, n=len(rs), n_defect=sum(r["y"] for r in rs), n_high=len(hi),
                          cams_high=",".join(sorted(r["cam"] for r in hi)),
                          scores=" ".join(f"{r['cam'][-1]}:{r['s']:.0f}{'*' if r['y'] else ''}" for r in sorted(rs, key=lambda r: r["cam"]))))
    out["_cross_camera"] = cross

    json.dump(out, open(DATA / "p1_variants.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    with open(DATA / "p1_variants.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f); w.writerow(["口径", "帧数", "缺陷数", "规则", "留一误报", "召回", "召回95%CI", "阈值"])
        for name, row in out.items():
            if name.startswith("_"): continue
            for key, lab, thr in (("MAD", "MAD", ""), ("P95", "P95", ""), ("conf5", "保形5%", row["thr_conf5"]),
                                  ("conf10", "保形10%", row["thr_conf10"]), ("pot5", "POT5%", row["pot5"].get("thr", ""))):
                d = row[key]
                if not d: continue
                ci = d.get("recall_ci", (float("nan"), float("nan")))
                w.writerow([name, row["n"], row["n_def"], lab, f"{d.get('fpr', float('nan')):.3f}",
                            f"{d.get('recall', float('nan')):.3f}", f"[{ci[0]:.2f},{ci[1]:.2f}]",
                            f"{thr:.1f}" if isinstance(thr, float) and np.isfinite(thr) else ""])

    print("\n== 按时刻聚合（分数 ≥12.4 记为高分；* = 已标缺陷）==")
    for c in cross:
        flag = "  ← 多机位同时高分" if c["n_high"] >= 2 else ""
        print(f"  {c['time']}  {c['scores']}{flag}")
    print("\n写出", HERE / "p1_variants.json", "与 p1_variants.csv")


if __name__ == "__main__":
    main()
