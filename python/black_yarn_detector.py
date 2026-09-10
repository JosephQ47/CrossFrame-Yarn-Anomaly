"""黑纱缺陷检测器（生产版）——DINOv2 多层特征 + 固定机位跨帧同位置参考。
零训练、零缺陷样本、不使用任何标签。
扫参最优配置(见 黑纱评估基准/sweep_results.csv)：ViT-B/14，最后 8 层拼接，top-1，带 0.34-0.60，带高 448。
基准成绩(16 真缺陷/55 正常)：定位 14-15/16，图级 AUROC 96.4，纯正常P95定阈下召回 94%/误报 5%。

用法:
  python black_yarn_detector.py                      # 在黑纱基准上评估+出图
  DET_LAYERS=8 DET_TOPK=1 python black_yarn_detector.py
"""
import os, json, glob, re, base64
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"; os.environ["PYTHONUTF8"] = "1"
from pathlib import Path
import numpy as np, cv2, torch, timm
from sklearn.metrics import roc_auc_score

BF = Path(r"D:\dataset\black_fabric")
WORK = Path(r"D:\研究生课程\课题\色纱研究\黑纱评估基准\检测器叠加图"); WORK.mkdir(parents=True, exist_ok=True)
SCRATCH = Path(r"C:\Users\QUANZH~1\AppData\Local\Temp\claude\D---------------\16e543fc-e508-4327-8253-40ffb430b66b\scratchpad")

MODEL  = os.environ.get("DET_MODEL", "vit_base_patch14_dinov2.lvd142m")
LAYERS = int(os.environ.get("DET_LAYERS", 8))     # 取最后 N 层拼接(扫参最优 8: L8>L12>L4)
TOPK   = int(os.environ.get("DET_TOPK", 1))       # 参考帧取最相似 top-k
BAND_Y0, BAND_Y1 = 0.34, 0.60                     # 筘齿带(固定机位)
TILE_W, TILE_H, OVERLAP = 588, 448, 140
MEAN = np.array([0.485,0.456,0.406], np.float32); STD = np.array([0.229,0.224,0.225], np.float32)


def read_bgr(p): return cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)
def pbox(pts):
    a = np.array(pts, float); return [a[:,0].min(), a[:,1].min(), a[:,0].max(), a[:,1].max()]


@torch.no_grad()
def band_feat(model, dev, img):
    """筘齿带切重叠 tile → DINOv2 多层 patch 特征 → 拼回带级特征图(已L2归一化)。"""
    H, W = img.shape[:2]
    y0, y1 = int(H*BAND_Y0), int(H*BAND_Y1)
    band = cv2.resize(img[y0:y1], (W, TILE_H))
    ps = 14; gh = TILE_H//ps; gw = TILE_W//ps; cols = W//ps
    xs = list(range(0, max(1, W-TILE_W+1), TILE_W-OVERLAP))
    if xs[-1] + TILE_W < W: xs.append(W-TILE_W)
    acc = cnt = None
    for x in xs:
        rgb = cv2.cvtColor(band[:, x:x+TILE_W], cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
        t = torch.from_numpy(((rgb-MEAN)/STD).transpose(2,0,1)[None]).to(dev)
        outs = model.get_intermediate_layers(t, n=LAYERS, reshape=False, norm=True)
        fs = [(o[:, o.shape[1]-gh*gw:, :] if o.shape[1] != gh*gw else o)[0] for o in outs]
        f = torch.cat(fs, dim=-1)
        fm = f.reshape(gh, gw, f.shape[-1]).float().cpu().numpy()
        if acc is None:
            acc = np.zeros((gh, cols, fm.shape[-1]), np.float32); cnt = np.zeros((gh, cols, 1), np.float32)
        j0 = x//ps; j1 = min(cols, j0+gw)
        acc[:, j0:j1] += fm[:, :j1-j0]; cnt[:, j0:j1] += 1
    feat = acc/np.maximum(cnt, 1)
    return feat/(np.linalg.norm(feat, axis=-1, keepdims=True)+1e-8), y0, y1


def anomaly_map(F, k, shape, y0, y1):
    """F:(T,gh,gw,C) 同相机所有帧特征; k:当前帧索引 → (全图异常图, patch级异常图)。"""
    others = np.delete(F, k, axis=0); cur = F[k]
    sims = np.einsum('tijc,ijc->tij', others, cur)
    kk = max(1, min(TOPK, sims.shape[0]))
    anom = 1.0 - np.sort(sims, axis=0)[-kk:].mean(axis=0)
    H, W = shape
    full = cv2.resize(anom, (W, y1-y0), interpolation=cv2.INTER_CUBIC)
    m = np.zeros((H, W), np.float32); m[y0:y1] = full
    m = cv2.GaussianBlur(m, (0,0), 9); m[:y0] = 0; m[y1:] = 0
    return m, anom


AREA_CAP = int(os.environ.get("DET_AREA_CAP", 55))   # 异常块面积上限(patch数),筛人手等大面积入侵
                                                     # 45~56 为等效平台;取55离断崖(42)最远,且仍低于人手块(57)


def score_and_locate(anom, hw, y0, y1, q=99, cap=None):
    """图级分数 = 面积≤cap 的最大连通域面积；定位 = 该连通域**重心**。

    · 空间聚集性(连通域面积)远优于 max：AUROC 96.4 vs 81.1、100%召回误报 9% vs 62%，
      且使"纯正常样本定阈"可行(P95 阈值下 召回94%/误报5%)。
    · 尺寸先验 cap=55：已目视核实一个 area=57 的误报正是**人手伸入画面**。
      加 cap 后 AUROC 96.4→98.2、100%召回误报 9%→7%。
      45~56 为等效平台(均 15/16, AUROC 98.0-98.2)，≤42 断崖(92.8, 误报98%)；
      取 55 离断崖最远且仍低于人手块(57)。注意 P99 只选 1% patch(≈73)，故 cap>58 无效果。
      ⚠️ 仅在 16 个缺陷上验证，新数据须重新确认。
    · 定位用连通域重心优于平滑图峰值(实测 15/16 vs 14/16)。
    """
    cap = AREA_CAP if cap is None else cap
    thr = np.percentile(anom, q)
    n, lab, stats, cent = cv2.connectedComponentsWithStats(
        (anom >= thr).astype(np.uint8), connectivity=8)
    cands = [(stats[i, cv2.CC_STAT_AREA], i) for i in range(1, n)
             if cap is None or stats[i, cv2.CC_STAT_AREA] <= cap]
    if not cands:
        return 0.0, None, None
    area, bi = max(cands)
    gh, gw = anom.shape; H, W = hw
    sx, sy = W/gw, (y1-y0)/gh
    px = int(cent[bi][0]*sx); py = int(y0 + cent[bi][1]*sy)
    ys, xs = np.where(lab == bi)
    box = [int(xs.min()*sx), int(y0+ys.min()*sy), int((xs.max()+1)*sx), int(y0+(ys.max()+1)*sy)]
    return float(area), (px, py), box


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = timm.create_model(MODEL, pretrained=True, num_classes=0, dynamic_img_size=True).eval().to(dev)
    print(f"模型 {MODEL} | 最后{LAYERS}层 | top-{TOPK} | device={dev}")

    recs = []
    for jp in sorted(glob.glob(str(BF/"*.json"))):
        d = json.load(open(jp, encoding="utf-8"))
        dets = [pbox(s["points"]) for s in d["shapes"] if s["label"].lower()=="detect"]
        ip = BF/(Path(jp).stem+".jpg")
        if not ip.exists(): continue
        mm = re.search(r"(Cam\d)", ip.name)
        recs.append(dict(path=ip, dets=dets, cam=mm.group(1) if mm else "NA"))
    cams = sorted(set(r["cam"] for r in recs))

    hit=ndef=0; scores=[]; labels=[]; uris=[]
    for c in cams:
        sub = [r for r in recs if r["cam"]==c]
        feats=[]
        for r in sub:
            img = read_bgr(r["path"])
            f, y0, y1 = band_feat(model, dev, img)
            r["_y0"], r["_y1"], r["_hw"] = y0, y1, img.shape[:2]
            feats.append(f)
        F = np.stack(feats)
        for k, r in enumerate(sub):
            m, anom = anomaly_map(F, k, r["_hw"], r["_y0"], r["_y1"])
            sc_i, pk, bx = score_and_locate(anom, r["_hw"], r["_y0"], r["_y1"])
            scores.append(sc_i); labels.append(1 if r["dets"] else 0)
            if not r["dets"]: continue
            ndef += 1
            px, py = pk if pk else (-1, -1)
            ok = any(b[0]<=px<=b[2] and b[1]<=py<=b[3] for b in r["dets"]); hit += ok
            img = read_bgr(r["path"]); H, W = r["_hw"]
            hn = np.clip((m-m.min())/(m.max()-m.min()+1e-6),0,1)
            vis = cv2.addWeighted(img, 0.55, cv2.applyColorMap((hn*255).astype(np.uint8), cv2.COLORMAP_JET), 0.45, 0)
            for b in r["dets"]: cv2.rectangle(vis,(int(b[0]),int(b[1])),(int(b[2]),int(b[3])),(255,255,255),3)
            cv2.circle(vis,(int(px),int(py)),16,(0,255,0) if ok else (0,0,255),3)
            cv2.imencode(".jpg", vis, [int(cv2.IMWRITE_JPEG_QUALITY),85])[1].tofile(
                str(WORK/f"{'HIT' if ok else 'MISS'}_{r['path'].stem}.jpg"))
            small = cv2.resize(vis,(900,int(H*900/W)))
            _, enc = cv2.imencode(".jpg",small,[int(cv2.IMWRITE_JPEG_QUALITY),80])
            uris.append((r['path'].stem[:26], "HIT" if ok else "MISS",
                         "data:image/jpeg;base64,"+base64.b64encode(enc.tobytes()).decode()))
        print(f"  {c} 完成")

    auroc = roc_auc_score(labels, scores)*100
    sc = np.array(scores); lb = np.array(labels)
    pos, neg = sc[lb==1], sc[lb==0]
    thr95 = np.percentile(neg, 95)          # 纯正常样本定阈(产线可行)
    rec95, fp95 = (pos>=thr95).mean(), (neg>=thr95).mean()
    fpr100 = (neg >= pos.min()).mean()
    print(f"\n===== 黑纱缺陷检测器 =====")
    print(f"定位命中: {hit}/{ndef} = {hit/max(1,ndef):.2f}")
    print(f"图级 AUROC: {auroc:.1f}")
    print(f"100%召回时误报: {fpr100:.0%}")
    print(f"产线定阈(纯正常P95={thr95:.0f}): 召回 {rec95:.0%} | 误报 {fp95:.0%}")

    uris.sort(key=lambda t: (t[1] != "HIT", t[0]))
    cards = "\n".join(f'<figure class=c><figcaption>{nm} <b style="color:{"#2ecc71" if h=="HIT" else "#e74c3c"}">{h}</b></figcaption><div class=t><img src="{u}" data-full="{u}" tabindex=0></div></figure>' for nm,h,u in uris)
    html = f"""<title>黑纱缺陷检测器</title><style>
:root{{--bg:#f4f5f6;--pan:#fff;--ink:#1a1e22;--mut:#5b6670;--ln:#dfe3e7}}
@media(prefers-color-scheme:dark){{:root{{--bg:#14171a;--pan:#1c2024;--ink:#e8ecef;--mut:#96a0a8;--ln:#2b3137}}}}
:root[data-theme=dark]{{--bg:#14171a;--pan:#1c2024;--ink:#e8ecef;--mut:#96a0a8;--ln:#2b3137}}
:root[data-theme=light]{{--bg:#f4f5f6;--pan:#fff;--ink:#1a1e22;--mut:#5b6670;--ln:#dfe3e7}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font-family:ui-sans-serif,system-ui,"Microsoft YaHei",sans-serif}}
.wrap{{max-width:1150px;margin:0 auto;padding:30px 20px 60px}}h1{{font-size:24px;margin:0 0 6px}}p.s{{color:var(--mut);margin:0 0 20px;max-width:78ch}}
.kpi{{display:flex;gap:14px;margin:0 0 22px;flex-wrap:wrap}}
.k{{background:var(--pan);border:1px solid var(--ln);border-radius:9px;padding:12px 18px}}
.k b{{display:block;font-size:26px;font-variant-numeric:tabular-nums}}.k span{{font-size:12px;color:var(--mut)}}
.g{{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:12px}}
.c{{margin:0;background:var(--pan);border:1px solid var(--ln);border-radius:9px;overflow:hidden}}
figcaption{{padding:6px 10px;font:12px ui-monospace,monospace}}
.t{{background:#000}}.t img{{width:100%;display:block;cursor:zoom-in}}
#lb{{position:fixed;inset:0;background:rgba(6,8,10,.96);display:none;z-index:9;overflow:auto;cursor:zoom-out}}#lb.on{{display:block}}#lb img{{max-width:none;display:block;margin:0 auto}}#lb .b{{position:sticky;top:0;padding:10px;background:rgba(6,8,10,.75);color:#eee;font:12px monospace}}
</style><div class=wrap><h1>黑纱缺陷检测器 · 基准结果</h1>
<div class=kpi><div class=k><b>{hit}/{ndef}</b><span>定位命中 ({hit/max(1,ndef)*100:.0f}%)</span></div>
<div class=k><b>{auroc:.1f}</b><span>图级 AUROC</span></div>
<div class=k><b>{rec95:.0%}</b><span>产线定阈召回 (误报 {fp95:.0%})</span></div>
<div class=k><b>0</b><span>训练样本 / 缺陷样本</span></div></div>
<p class=s>DINOv2 {MODEL.split('_')[1]}/14 最后{LAYERS}层特征 + 固定机位跨帧同位置参考(top-{TOPK})。白框=真缺陷GT，绿圈=命中/红圈=未命中。此前所有方法均 0/16。点图放大。</p>
<div class=g>{cards}</div></div>
<div id=lb><div class=b>点空白/Esc关闭</div><img id=i></div><script>
var lb=document.getElementById('lb'),i=document.getElementById('i');document.querySelectorAll('.t img').forEach(function(e){{e.onclick=function(){{i.src=e.dataset.full;lb.classList.add('on')}}}});lb.onclick=function(){{lb.classList.remove('on')}};document.onkeydown=function(e){{if(e.key=='Escape')lb.classList.remove('on')}};
</script>"""
    (SCRATCH/"black_detector.html").write_text(html, encoding="utf-8")
    print(f"叠加图: {WORK}\n画廊: {SCRATCH/'black_detector.html'}")


if __name__ == "__main__":
    main()
