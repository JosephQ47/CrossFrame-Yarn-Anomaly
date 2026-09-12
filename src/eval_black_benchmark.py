"""黑纱评估基准:用 black_fabric 的 labelme GT 评估任意 YOLO 权重的 detect+KC 真实 P/R。
GT: D:/dataset/black_fabric/*.json (71 张; 16 张含 detect 真缺陷; 全 71 张 KC)。禁训,仅评估。
用法:
  python eval_black_benchmark.py                 # 跑内置全部现有模型,出基准报告
  python eval_black_benchmark.py <权重.pt> [名称] # 评估单个新模型
输出: D:/.../黑纱评估基准/benchmark_report.md + benchmark_scores.csv + gt_清单.csv
"""
import os, sys, json, glob, csv
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"; os.environ["PYTHONUTF8"] = "1"
from pathlib import Path
import numpy as np, cv2

PROJ = Path(r"C:\Users\quanzhonghui\Documents\色纱课题")
GTD = Path(r"D:\dataset\black_fabric")
OUT = Path(r"D:\研究生课程\课题\色纱研究\黑纱评估基准"); OUT.mkdir(parents=True, exist_ok=True)
IMG = 1280; DET_IOU = 0.3; KC_IOU = 0.5; CONF = 0.25

R = PROJ/"outputs"/"fda_lch_fair_comparison_20260630_003118"/"runs"
FT = PROJ/"outputs"/"real_yarn_dataset"/"runs"
BUILTIN = [
    ("A_纯白纱基线", R/"experiment_A"/"experiment_A"/"weights"/"best.pt"),
    ("B_FDA", R/"experiment_B"/"experiment_B"/"weights"/"best.pt"),
    ("C_伪造色纱", R/"experiment_C"/"experiment_C"/"weights"/"best.pt"),
    ("C_M5哑光", R/"experiment_C_M5"/"experiment_C_M5"/"weights"/"best.pt"),
    ("C_dark加深色", PROJ/"outputs"/"c_dark_quickval"/"runs"/"cdark_quickval"/"weights"/"best.pt"),
    ("real_finetune_full", FT/"real_finetune_full"/"weights"/"best.pt"),
]


def read_bgr(p): return cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)
def pbox(pts): a = np.array(pts, float); return [a[:,0].min(), a[:,1].min(), a[:,0].max(), a[:,1].max()]
def iou(a, b):
    x1,y1,x2,y2 = max(a[0],b[0]),max(a[1],b[1]),min(a[2],b[2]),min(a[3],b[3])
    inter = max(0,x2-x1)*max(0,y2-y1)
    ua = (a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
    return inter/ua if ua > 0 else 0.0


def match(preds, gts, thr):
    used = [False]*len(gts); tp = 0
    for pb, _ in sorted(preds, key=lambda x: -x[1]):
        bi, bs = -1, thr
        for i, g in enumerate(gts):
            if not used[i] and iou(pb, g) >= bs: bs, bi = iou(pb, g), i
        if bi >= 0: used[bi] = True; tp += 1
    return tp, len(preds)-tp, len(gts)-tp


def load_items():
    items = []
    for jp in sorted(glob.glob(str(GTD/"*.json"))):
        ip = GTD/(Path(jp).stem+".jpg")
        if not ip.exists(): continue
        d = json.load(open(jp, encoding="utf-8"))
        det = [pbox(s["points"]) for s in d["shapes"] if s["label"].lower() == "detect"]
        kc = [pbox(s["points"]) for s in d["shapes"] if s["label"].lower() == "kc"]
        items.append((ip, det, kc))
    return items


def eval_one(model, items, conf):
    dTP=dFP=dFN=0; d_imghit=d_imggt=0
    kTP=kFP=kFN=0; k_imghit=k_imggt=0
    for ip, dgt, kgt in items:
        r = model.predict(read_bgr(ip), imgsz=IMG, conf=conf, verbose=False)[0]
        dp, kp = [], []
        if r.boxes is not None and len(r.boxes):
            cls = r.boxes.cls.cpu().numpy().astype(int)
            xy = r.boxes.xyxy.cpu().numpy(); cf = r.boxes.conf.cpu().numpy()
            for i in range(len(cls)):
                (dp if cls[i]==0 else kp).append((xy[i].tolist(), float(cf[i])))
        tp,fp,fn = match(dp, dgt, DET_IOU); dTP+=tp; dFP+=fp; dFN+=fn
        if dgt: d_imggt+=1; d_imghit += tp>0
        tp,fp,fn = match(kp, kgt, KC_IOU); kTP+=tp; kFP+=fp; kFN+=fn
        if kgt: k_imggt+=1; k_imghit += tp>0
    def prf(tp,fp,fn):
        P=tp/(tp+fp) if tp+fp else 0; Rc=tp/(tp+fn) if tp+fn else 0
        return P, Rc, (2*P*Rc/(P+Rc) if P+Rc else 0)
    dP,dR,dF = prf(dTP,dFP,dFN); kP,kR,kF = prf(kTP,kFP,kFN)
    return dict(dP=dP,dR=dR,dF=dF,dTP=dTP,dFP=dFP,dFN=dFN,d_img=f"{d_imghit}/{d_imggt}",
                kP=kP,kR=kR,kF=kF,k_img=f"{k_imghit}/{k_imggt}")


def main():
    from ultralytics import YOLO
    items = load_items()
    ndet = sum(1 for _,d,_ in items if d); ndetbox = sum(len(d) for _,d,_ in items)
    print(f"黑纱评估基准: {len(items)} 张 GT | {ndet} 张含 detect ({ndetbox} 框) | 全含 KC")
    print(f"口径: imgsz{IMG}, conf≥{CONF}, detect IoU≥{DET_IOU}, KC IoU≥{KC_IOU}\n")

    models = BUILTIN
    if len(sys.argv) >= 2:
        models = [(sys.argv[2] if len(sys.argv) >= 3 else Path(sys.argv[1]).stem, Path(sys.argv[1]))]

    rows = []
    print(f"{'模型':22s} | detect P/R/F1        img检出 | KC P/R/F1        img检出")
    print("-"*88)
    for tag, ck in models:
        if not Path(ck).exists(): print(f"{tag:22s} | [缺失]"); continue
        r = eval_one(YOLO(str(ck)), items, CONF)
        rows.append((tag, r))
        print(f"{tag:22s} | {r['dP']:.2f}/{r['dR']:.2f}/{r['dF']:.2f}  {r['d_img']:>6} | "
              f"{r['kP']:.2f}/{r['kR']:.2f}/{r['kF']:.2f}  {r['k_img']:>6}")

    # CSV
    with (OUT/"benchmark_scores.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["model","det_P","det_R","det_F1","det_TP","det_FP","det_FN","det_img_recall",
                    "kc_P","kc_R","kc_F1","kc_img_recall"])
        for tag,r in rows:
            w.writerow([tag,f"{r['dP']:.3f}",f"{r['dR']:.3f}",f"{r['dF']:.3f}",r['dTP'],r['dFP'],r['dFN'],r['d_img'],
                        f"{r['kP']:.3f}",f"{r['kR']:.3f}",f"{r['kF']:.3f}",r['k_img']])
    # gt 清单
    with (OUT/"gt_清单.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f); w.writerow(["file","has_detect","n_detect","n_kc"])
        for ip,d,k in items: w.writerow([ip.name, int(bool(d)), len(d), len(k)])
    # markdown 报告
    md = [f"# 黑纱评估基准 · 成绩报告\n",
          f"- GT: `D:/dataset/black_fabric` {len(items)} 张 | {ndet} 张含 detect 真缺陷 ({ndetbox} 框) | 全 71 张含 KC",
          f"- 口径: imgsz{IMG}, conf≥{CONF}, detect IoU≥{DET_IOU}, KC IoU≥{KC_IOU}",
          f"- **仅评估,永久禁训**\n",
          "| 模型 | detect P | detect R | detect F1 | detect图检出 | KC P | KC R | KC图检出 |",
          "|---|---|---|---|---|---|---|---|"]
    for tag,r in rows:
        md.append(f"| {tag} | {r['dP']:.2f} | **{r['dR']:.2f}** | {r['dF']:.2f} | {r['d_img']} | {r['kP']:.2f} | {r['kR']:.2f} | {r['k_img']} |")
    md += ["\n> detect Recall 是业务核心(缺陷别漏)。当前所有模型 detect R=0 —— 真实黑纱缺陷检出未解决,",
           "> 待真实黑纱缺陷训练样本(本批禁训)或合成缺陷方案突破。"]
    (OUT/"benchmark_report.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\n报告: {OUT/'benchmark_report.md'}\n成绩CSV: {OUT/'benchmark_scores.csv'}\nGT清单: {OUT/'gt_清单.csv'}")


if __name__ == "__main__":
    main()
