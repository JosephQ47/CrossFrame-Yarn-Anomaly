# -*- coding: utf-8 -*-
"""P1 附带：把"分数异常高的正常帧"生成人工复核材料。

由来：合并六机位校准下，保形规则 α=5% 的阈值被顶到 19，直接原因是两张标注为正常的帧分数很高：
  Cam3 10-11-55-177 → 23 分（它是 Cam3 缺陷帧 10-11-56-691 的前 1.5 秒）
  Cam5 10-14-15-402 → 19 分
若它们其实已有缺陷，那不只是定阈问题，基准标注也要改。本脚本只出图，不改任何标注。

对每帧输出三张：
  A_原图_带框.jpg    原图 + 检测器定位点（不叠热图），供判断"那里到底有没有絮状物"
  B_热图.jpg         异常热图叠加（看检测器在关注哪）
  C_放大.jpg         定位点周围 400×400 原始像素放大 3 倍（判读主依据）
另存同机位时间相邻帧的同位置裁切做对照（D_邻帧对照.jpg）。

用法： D:/anaconda3/envs/Yolov8/python.exe review_suspect_frames.py
"""
import os, sys, glob, json, re
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("PYTHONUTF8", "1")
from pathlib import Path
import numpy as np, cv2, torch, timm
from PIL import Image, ImageDraw, ImageFont

_FONT = None
for _p in (r"C:/Windows/Fonts/msyh.ttc", r"C:/Windows/Fonts/simhei.ttf", r"C:/Windows/Fonts/simsun.ttc"):
    if os.path.exists(_p):
        _FONT = _p; break


def put_cn(img, text, xy, color_bgr, size=20):
    """OpenCV 的 putText 不支持中文（会画成问号），用 PIL 画。color 传 BGR，内部转 RGB。"""
    if _FONT is None:
        cv2.putText(img, text, xy, cv2.FONT_HERSHEY_SIMPLEX, 0.55, color_bgr, 2); return img
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ImageDraw.Draw(pil).text(xy, text, font=ImageFont.truetype(_FONT, size),
                             fill=(color_bgr[2], color_bgr[1], color_bgr[0]),
                             stroke_width=2, stroke_fill=(0, 0, 0))
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import black_yarn_detector as D                      # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "assets" / "待复核帧"
SUSPECTS = {"Cam3": "DI2_Cam3_2026-07-18_10-11-55-177", "Cam5": "DI2_Cam5_2026-07-18_10-14-15-402"}
CROP = 400          # 裁切边长（原始像素）
ZOOM = 3


def main():
    OUT.mkdir(exist_ok=True)
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

    notes = []
    for cam, stem in SUSPECTS.items():
        sub = [r for r in recs if r["cam"] == cam]
        sub.sort(key=lambda r: r["path"].stem)
        feats = []
        for r in sub:
            img = D.read_bgr(r["path"]); f, y0, y1 = D.band_feat(model, dev, img)
            r["_y0"], r["_y1"], r["_hw"] = y0, y1, img.shape[:2]; feats.append(f)
        F = np.stack(feats)
        k = next(i for i, r in enumerate(sub) if r["path"].stem == stem)
        r = sub[k]
        m, anom = D.anomaly_map(F, k, r["_hw"], r["_y0"], r["_y1"])
        sc, pk, bx = D.score_and_locate(anom, r["_hw"], r["_y0"], r["_y1"])
        px, py = pk if pk else (r["_hw"][1] // 2, (r["_y0"] + r["_y1"]) // 2)
        img = D.read_bgr(r["path"]); H, W = r["_hw"]
        tag = f"{cam}_{stem[-12:]}_分数{int(sc)}"

        a = img.copy()
        cv2.circle(a, (int(px), int(py)), 26, (0, 0, 255), 3)
        cv2.rectangle(a, (0, r["_y0"]), (W - 1, r["_y1"]), (0, 255, 255), 2)
        cv2.imencode(".jpg", a, [int(cv2.IMWRITE_JPEG_QUALITY), 92])[1].tofile(str(OUT / f"{tag}_A_原图_带框.jpg"))

        hn = np.clip((m - m.min()) / (m.max() - m.min() + 1e-6), 0, 1)
        vis = cv2.addWeighted(img, 0.55, cv2.applyColorMap((hn * 255).astype(np.uint8), cv2.COLORMAP_JET), 0.45, 0)
        cv2.circle(vis, (int(px), int(py)), 26, (0, 255, 0), 3)
        cv2.imencode(".jpg", vis, [int(cv2.IMWRITE_JPEG_QUALITY), 92])[1].tofile(str(OUT / f"{tag}_B_热图.jpg"))

        def crop_of(im, cx, cy):
            x0 = int(np.clip(cx - CROP // 2, 0, im.shape[1] - CROP)); y0c = int(np.clip(cy - CROP // 2, 0, im.shape[0] - CROP))
            c = im[y0c:y0c + CROP, x0:x0 + CROP]
            return cv2.resize(c, (CROP * ZOOM, CROP * ZOOM), interpolation=cv2.INTER_CUBIC)

        cv2.imencode(".jpg", crop_of(img, px, py), [int(cv2.IMWRITE_JPEG_QUALITY), 95])[1].tofile(str(OUT / f"{tag}_C_放大.jpg"))

        # 同机位其余帧在同一位置的裁切，横向拼接做对照
        tiles = []
        for j, rr in enumerate(sub):
            if j == k: continue
            c = crop_of(D.read_bgr(rr["path"]), px, py)
            c = cv2.resize(c, (360, 360))
            lab = rr["path"].stem[-12:] + ("  [已标缺陷]" if rr["dets"] else "")
            tiles.append(put_cn(c, lab, (6, 6), (0, 255, 255) if rr["dets"] else (255, 255, 255)))
        rows = [np.hstack(tiles[i:i + 5]) for i in range(0, len(tiles), 5)]
        wmax = max(x.shape[1] for x in rows)
        rows = [np.pad(x, ((0, 0), (0, wmax - x.shape[1]), (0, 0))) for x in rows]
        me = cv2.resize(crop_of(img, px, py), (360, 360))
        me = put_cn(me, f"本帧 {stem[-12:]}  分数 {int(sc)}", (6, 6), (0, 0, 255))
        cv2.rectangle(me, (0, 0), (359, 359), (0, 0, 255), 4)
        me = np.pad(me, ((0, 0), (0, wmax - 360), (0, 0)))
        cv2.imencode(".jpg", np.vstack([me] + rows), [int(cv2.IMWRITE_JPEG_QUALITY), 88])[1].tofile(str(OUT / f"{tag}_D_邻帧同位置对照.jpg"))

        notes.append(f"- **{cam} {stem}**：分数 {int(sc)}，定位点 (x={int(px)}, y={int(py)})，筘齿带 y∈[{r['_y0']},{r['_y1']}]，"
                     f"该机位共 {len(sub)} 帧（{sum(1 for x in sub if x['dets'])} 帧标为缺陷）")
        print(notes[-1])

    (OUT / "判读说明.md").write_text(
        "# 两张高分\"正常帧\"的人工复核\n\n"
        "这两帧在基准里标为**正常**，但跨帧同位置检测器给了很高的分数，直接把保形规则 α=5% 的阈值顶到 19，"
        "使召回从 93.8% 掉到 56.2%。请判断定位点处**是否真的有絮状物糊住筘齿**。\n\n"
        + "\n".join(notes) +
        "\n\n## 看图顺序\n\n"
        "1. `C_放大.jpg` — 定位点周围 400×400 原始像素放大 3 倍，**这是判读主依据**。\n"
        "2. `D_邻帧同位置对照.jpg` — 同机位其他帧在同一位置的裁切，红框是本帧；对照看该位置平时长什么样。\n"
        "3. `A_原图_带框.jpg` — 全图与定位点，确认位置在筘齿带内。\n"
        "4. `B_热图.jpg` — 检测器关注区域，**判读时最好最后看**，避免被它引导。\n\n"
        "## 判读结果请写在这里\n\n"
        "| 帧 | 有无缺陷 | 理由 |\n|---|---|---|\n"
        "| Cam3 10-11-55-177 |  |  |\n| Cam5 10-14-15-402 |  |  |\n\n"
        "若判为**有缺陷**：基准要从 16 缺陷改为 17/18，全部指标需复算，且保形 5% 的召回会显著回升。\n"
        "若判为**无缺陷**：说明同位置参考在这两帧上产生了真误报，需要看是不是人手或临时物件（可开外观屏蔽复测）。\n",
        encoding="utf-8")
    print("\n材料已写出：", OUT)


if __name__ == "__main__":
    main()
