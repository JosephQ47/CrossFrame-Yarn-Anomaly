"""C# 流水线一致性验证（Python 侧基准）。

目的：C# BlackYarnDetector 是把 Python 检测器整套重写的（切tile/归一化/特征拼接/
连通域/自校准）。ONNX 导出只验证了模型本身数值一致，**整条流水线从未验证**。
任何一处写错都会静默失效（不报错、只是不报警）——工厂部署前必须先验。

本脚本用 **ONNX**（而非 timm）跑完整 Python 流水线，导出每帧的中间量与最终分数，
供 C# 侧用同一批图跑出结果后逐项对比。

输出: outputs/pipeline_verify/python_reference.json
"""
import os, json, glob, re
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"; os.environ["PYTHONUTF8"] = "1"
from pathlib import Path
import numpy as np, cv2
import onnxruntime as ort

BF = Path(r"D:\dataset\black_fabric")
ONNX = Path(r"C:\Users\quanzhonghui\Documents\色纱课题\outputs\onnx_export\dinov2_vitb14_l8_448x588.onnx")
OUT = Path(r"C:\Users\quanzhonghui\Documents\色纱课题\outputs\pipeline_verify")
OUT.mkdir(parents=True, exist_ok=True)

# 与 BlackYarnDetector.cs 完全一致的常量
TILE_W, TILE_H, OVERLAP, PS = 588, 448, 140, 14
GH = TILE_H // PS
FEAT = 768 * 8
BAND_Y0, BAND_Y1 = 0.34, 0.60
AREA_CAP = 55
TOPK = 1
K_MAD = 3.5
BUF = 8
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def read_bgr(p): return cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)


def band_feat(sess, img):
    """严格复刻 C# ExtractBandFeature 的逻辑。"""
    H, W = img.shape[:2]
    cols = W // PS
    y0, y1 = int(H * BAND_Y0), int(H * BAND_Y1)
    band = cv2.resize(img[y0:y1], (W, TILE_H))
    xs = list(range(0, max(0, W - TILE_W) + 1, TILE_W - OVERLAP))
    if xs[-1] + TILE_W < W: xs.append(W - TILE_W)
    gw = TILE_W // PS
    acc = np.zeros((GH, cols, FEAT), np.float32)
    cnt = np.zeros((GH, cols), np.int32)
    for x in xs:
        tile = band[:, x:x+TILE_W]
        rgb = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        t = ((rgb - MEAN) / STD).transpose(2, 0, 1)[None]
        f = sess.run(None, {"input": t})[0][0]        # (GH*gw, FEAT)
        fm = f.reshape(GH, gw, FEAT)
        j0 = x // PS
        for gc in range(gw):
            dst = min(j0 + gc, cols - 1)
            acc[:, dst] += fm[:, gc]
            cnt[:, dst] += 1
    acc /= np.maximum(cnt, 1)[..., None]
    acc /= np.maximum(np.linalg.norm(acc, axis=-1, keepdims=True), 1e-8)
    return acc, y0, y1


def score_and_locate(anom, hw, y0, y1, cap=AREA_CAP):
    """严格复刻 C# score_and_locate。"""
    thr = np.percentile(anom, 99)
    n, lab, stats, cent = cv2.connectedComponentsWithStats(
        (anom >= thr).astype(np.uint8), connectivity=8)
    cands = [(stats[i, cv2.CC_STAT_AREA], i) for i in range(1, n)
             if cap is None or stats[i, cv2.CC_STAT_AREA] <= cap]
    if not cands: return 0.0, None, None
    area, bi = max(cands)
    gh, gw = anom.shape; H, W = hw
    sx, sy = W / gw, (y1 - y0) / gh
    px, py = int(cent[bi][0] * sx), int(y0 + cent[bi][1] * sy)
    ys, xs_ = np.where(lab == bi)
    box = [int(xs_.min()*sx), int(y0+ys.min()*sy), int((xs_.max()+1)*sx), int(y0+(ys.max()+1)*sy)]
    return float(area), (px, py), box


def anom_map(cur, refs):
    R = np.stack(refs)
    sims = np.einsum('tijc,ijc->tij', R, cur)
    kk = max(1, min(TOPK, sims.shape[0]))
    return 1.0 - np.sort(sims, axis=0)[-kk:].mean(axis=0)


def robust_thr(scores, k=K_MAD):
    s = np.asarray(scores, np.float64)
    med = np.median(s)
    mad = 1.4826 * np.median(np.abs(s - med))
    return float(med + k * max(mad, 1.0))


def main():
    print(f"加载 ONNX: {ONNX.name}")
    sess = ort.InferenceSession(str(ONNX), providers=["CPUExecutionProvider"])

    recs = []
    for jp in sorted(glob.glob(str(BF / "*.json"))):
        ip = BF / (Path(jp).stem + ".jpg")
        if not ip.exists(): continue
        d = json.load(open(jp, encoding="utf-8"))
        has_def = any(s["label"].lower() == "detect" for s in d["shapes"])
        m = re.search(r"(Cam\d)", ip.name)
        recs.append(dict(path=ip, defect=has_def, cam=m.group(1) if m else "NA"))

    cams = sorted(set(r["cam"] for r in recs))
    out = {"config": dict(tile=[TILE_W, TILE_H], overlap=OVERLAP, band=[BAND_Y0, BAND_Y1],
                          area_cap=AREA_CAP, topk=TOPK, k_mad=K_MAD, buf=BUF),
           "frames": []}

    for c in cams:
        sub = [r for r in recs if r["cam"] == c]
        print(f"{c}: {len(sub)} 帧")
        feats, metas = [], []
        for r in sub:
            img = read_bgr(r["path"])
            f, y0, y1 = band_feat(sess, img)
            feats.append(f); metas.append((img.shape[:2], y0, y1))

        # 模拟 C# 因果滚动缓冲：前 BUF 帧只入缓冲，之后开始判定
        buf = []
        thr = None
        for k, r in enumerate(sub):
            rec = dict(file=r["path"].name, cam=c, defect=r["defect"])
            if len(buf) >= BUF:
                anom = anom_map(feats[k], buf)
                sc, pk, box = score_and_locate(anom, *metas[k])
                if thr is None:
                    loo = []
                    for i in range(len(buf)):
                        rest = buf[:i] + buf[i+1:]
                        if len(rest) >= BUF - 1:
                            a2 = anom_map(buf[i], rest)
                            s2, _, _ = score_and_locate(a2, *metas[k])
                            loo.append(s2)
                    if len(loo) >= BUF - 1: thr = robust_thr(loo)
                rec.update(ready=thr is not None, score=round(sc, 4),
                           threshold=round(thr, 4) if thr else None,
                           alarm=bool(thr is not None and sc >= thr),
                           peak=list(pk) if pk else None, box=box,
                           anom_mean=round(float(anom.mean()), 6),
                           anom_max=round(float(anom.max()), 6))
            else:
                rec.update(ready=False, note="warmup")
            out["frames"].append(rec)
            buf.append(feats[k])
            if len(buf) > BUF: buf.pop(0)

    p = OUT / "python_reference.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    judged = [f for f in out["frames"] if f.get("ready")]
    alarms = [f for f in judged if f.get("alarm")]
    print(f"\n判定帧 {len(judged)} | 报警 {len(alarms)}")
    print(f"  其中真缺陷报警 {sum(1 for f in alarms if f['defect'])}"
          f" / 真缺陷总数(判定阶段) {sum(1 for f in judged if f['defect'])}")
    print(f"  误报 {sum(1 for f in alarms if not f['defect'])}"
          f" / 正常帧(判定阶段) {sum(1 for f in judged if not f['defect'])}")
    print(f"\n基准已存: {p}")
    print("→ 下一步用 C# 侧跑同一批图，逐帧比对 score/threshold/alarm")


if __name__ == "__main__":
    main()
