# -*- coding: utf-8 -*-
"""导出黑纱基准 69 帧在"开外观屏蔽"下的逐帧分数，供 P1 定阈分析对比。

动机：P1 发现保形 α=5% 的阈值被两张高分正常帧顶到 19，召回从 93.8% 掉到 56.2%。
人工查看后确认 Cam5 10-14-15-402（19 分）画面里是**工人的手**，属入侵物而非纱线缺陷
（课题范围只管纱线缺陷）。检测器已有可选的外观屏蔽（LAB 中值背景差分抓人手/白卡），
本脚本把开关打开重算分数，看阈值能否降下来。

输出 scores_maskon.csv：file, cam, defect, score_maskoff, score_maskon, hit_maskon
用法： D:/anaconda3/envs/Yolov8/python.exe dump_scores_maskon.py
"""
import os, sys, glob, json, re, csv
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("PYTHONUTF8", "1")
from pathlib import Path
import numpy as np, torch, timm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import black_yarn_detector as D                      # noqa: E402

HERE = Path(__file__).resolve().parent.parent / "data"


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = timm.create_model(D.MODEL, pretrained=True, num_classes=0, dynamic_img_size=True).eval().to(dev)

    recs = []
    for jp in sorted(glob.glob(str(D.BF / "*.json"))):
        d = json.load(open(jp, encoding="utf-8"))
        dets = [D.pbox(s["points"]) for s in d["shapes"] if s["label"].lower() == "detect"]
        ip = D.BF / (Path(jp).stem + ".jpg")
        if not ip.exists(): continue
        mm = re.search(r"(Cam\d)", ip.name)
        recs.append(dict(path=ip, dets=dets, cam=mm.group(1) if mm else "NA"))
    recs = D.dedup_recs(recs)
    print(f"{len(recs)} 帧 = {sum(1 for r in recs if r['dets'])} 缺陷 + {sum(1 for r in recs if not r['dets'])} 正常")

    rows = []
    for c in sorted(set(r["cam"] for r in recs)):
        sub = [r for r in recs if r["cam"] == c]
        feats, labs = [], []
        for r in sub:
            img = D.read_bgr(r["path"]); f, y0, y1 = D.band_feat(model, dev, img)
            r["_y0"], r["_y1"], r["_hw"] = y0, y1, img.shape[:2]
            feats.append(f); labs.append(D.band_lab(img, y0, y1, f.shape[0], f.shape[1]))
        F = np.stack(feats)
        for k, r in enumerate(sub):
            m, anom = D.anomaly_map(F, k, r["_hw"], r["_y0"], r["_y1"])
            s_off, pk_off, _ = D.score_and_locate(anom, r["_hw"], r["_y0"], r["_y1"])
            valid = ~D.intruder_mask(labs[k], labs[:k] + labs[k + 1:], *anom.shape)
            s_on, pk_on, _ = D.score_and_locate(anom, r["_hw"], r["_y0"], r["_y1"], valid=valid)
            hit = ""
            if r["dets"]:
                px, py = pk_on if pk_on else (-1, -1)
                hit = int(any(b[0] <= px <= b[2] and b[1] <= py <= b[3] for b in r["dets"]))
            rows.append(dict(file=r["path"].stem, cam=c, defect=int(bool(r["dets"])),
                             score_maskoff=float(s_off), score_maskon=float(s_on),
                             masked_frac=float(1 - valid.mean()), hit_maskon=hit))
        print(f"  {c} 完成")

    with open(HERE / "scores_maskon.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    off = np.array([r["score_maskoff"] for r in rows]); on = np.array([r["score_maskon"] for r in rows])
    y = np.array([r["defect"] for r in rows])
    from sklearn.metrics import roc_auc_score
    print(f"\nAUROC  屏蔽关 {roc_auc_score(y, off)*100:.1f}  屏蔽开 {roc_auc_score(y, on)*100:.1f}")
    print(f"定位命中（屏蔽开） {sum(r['hit_maskon'] for r in rows if r['defect'])}/{int(y.sum())}")
    print("正常帧最高分  屏蔽关", sorted(off[y == 0])[-4:], " 屏蔽开", sorted(on[y == 0])[-4:])
    print("缺陷帧最低分  屏蔽关", sorted(off[y == 1])[:4], " 屏蔽开", sorted(on[y == 1])[:4])
    for r in sorted(rows, key=lambda r: -r["score_maskoff"])[:6]:
        print(f"  {r['file'][-12:]} 缺陷={r['defect']} {r['score_maskoff']:.0f} → {r['score_maskon']:.0f}  屏蔽面积 {r['masked_frac']*100:.0f}%")
    print("\n写出", HERE / "scores_maskon.csv")


if __name__ == "__main__":
    main()
