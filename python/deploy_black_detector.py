"""黑纱缺陷检测 · 现场部署版（因果滚动缓冲 + 按相机自校准）。

设计要点（均来自基准实测）：
  · 每相机独立维护特征滚动缓冲（deque, maxlen=BUF），参考只用**历史帧**（因果，产线真实）
  · warm-up：累积 ≥MIN_REF(8) 帧才开始判定（实测 <8 帧性能明显下降）
  · 阈值**按相机自校准**：用该机位 warm-up 期的分数取 P95（跨相机批次绝对阈值不可迁移）
  · 图级分数=异常图 P99 二值化后最大连通域面积（空间聚集性，远优于 max）
  · 报警同时输出定位框（异常峰值邻域）

用法:
  python deploy_black_detector.py <帧目录> [--out 输出目录] [--cam-regex "(Cam\\d)"]
  DET_INTRUDER_MASK=1 python deploy_black_detector.py ...   # 开启外观级入侵屏蔽（人手/白卡），默认关
  DET_GATED=1 python deploy_black_detector.py ...           # 门控参考缓冲：报警帧不入池，防持续缺陷被吸收，默认关
目录内文件名需可解析相机标识与时间顺序（默认按文件名排序）。
输出: 报警叠加图 + alarms.jsonl（含相机/时间/分数/阈值/框）。
"""
import os, sys, json, re, argparse
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"; os.environ["PYTHONUTF8"] = "1"
from pathlib import Path
from collections import deque, defaultdict
import numpy as np, cv2, torch, timm
sys.path.insert(0, str(Path(__file__).resolve().parent))
from black_yarn_detector import INTRUDER_MASK, band_lab, intruder_mask   # 可选入侵屏蔽，默认关

MODEL = os.environ.get("DET_MODEL", "vit_base_patch14_dinov2.lvd142m")
LAYERS = int(os.environ.get("DET_LAYERS", 8))
TOPK   = int(os.environ.get("DET_TOPK", 1))   # 扫参最优:topk=1 抗参考池污染,P95定阈召回 75%→94%
BUF     = int(os.environ.get("DET_BUF", 16))     # 滚动缓冲帧数
MIN_REF = int(os.environ.get("DET_MIN_REF", 8))  # warm-up 最少参考帧
CAL_Q   = float(os.environ.get("DET_CAL_Q", 95)) # 自校准分位
GATED   = os.environ.get("DET_GATED", "0") == "1"  # 门控参考缓冲：报警帧不进参考池，防止持续性缺陷被吸收为"正常"
                                                  # 代价：低于阈值的漏检帧仍会被吸收；warm-up/标定期不门控
GATE_MAX = int(os.environ.get("DET_GATE_MAX", 30))  # 连续 N 帧都被门控挡在池外时强制放行 1 帧：
                                                    # 否则场景漂移(换光/机修)会让每帧都报警、每帧都不入池，缓冲永远追不上新常态
BAND_Y0, BAND_Y1 = 0.34, 0.60
TILE_W, TILE_H, OVERLAP = 588, 448, 140
MEAN = np.array([0.485,0.456,0.406], np.float32); STD = np.array([0.229,0.224,0.225], np.float32)


def read_bgr(p): return cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)
def imwrite(p, im): cv2.imencode(".jpg", im, [int(cv2.IMWRITE_JPEG_QUALITY), 88])[1].tofile(str(p))


@torch.no_grad()
def band_feat(model, dev, img):
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
    feat = acc/np.maximum(cnt,1)
    return feat/(np.linalg.norm(feat, axis=-1, keepdims=True)+1e-8), y0, y1


def robust_thr(scores, k=None):
    """鲁棒定阈: 中位数 + k×(归一化MAD)。
    为什么不用 P95: 校准帧若混入少数缺陷帧, P95 会直接取到缺陷分数上,
    把阈值顶到正常分数的 5-10 倍(实测 18~42 vs 正常 3~8), 检测器直接失效。
    中位数与 MAD 对少数污染样本鲁棒。"""
    k = float(os.environ.get("DET_K_MAD", 3.5)) if k is None else k
    s = np.asarray(scores, np.float64)
    med = np.median(s)
    mad = 1.4826 * np.median(np.abs(s - med))
    return float(med + k * max(mad, 1.0))


def analyze(cur, refs, hw, y0, y1, valid=None):
    """→ (图级分数, 定位框, 异常热图)。valid: 可选 patch 级有效掩膜（False=入侵物，剔除）"""
    R = np.stack(refs)
    sims = np.einsum('tijc,ijc->tij', R, cur)
    kk = max(1, min(TOPK, sims.shape[0]))
    anom = 1.0 - np.sort(sims, axis=0)[-kk:].mean(axis=0)
    # 图级：P99 二值化后最大连通域面积
    if valid is None:
        thr = np.percentile(anom, 99)
        bw = (anom >= thr).astype(np.uint8)
    else:   # 预算=全网格 1%（绝对数），在有效 patch 上取前 K 个，与 black_yarn_detector 一致
        K = max(1, min(int(round(anom.size * 0.01)), int(valid.sum())))
        thr = np.sort(anom[valid])[-K]
        bw = ((anom >= thr) & valid).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    areas = [(stats[i, cv2.CC_STAT_AREA], i) for i in range(1, n)]
    score = float(max([a for a,_ in areas] or [0]))
    # 定位：最大连通域在原图的外接框
    H, W = hw; gh, gw = anom.shape
    box = None
    if areas:
        _, bi = max(areas)
        ys, xs = np.where(lab == bi)
        sx, sy = W/gw, (y1-y0)/gh
        box = [int(xs.min()*sx), int(y0+ys.min()*sy), int((xs.max()+1)*sx), int(y0+(ys.max()+1)*sy)]
    hm = cv2.resize(anom, (W, y1-y0), interpolation=cv2.INTER_CUBIC)
    full = np.zeros((H, W), np.float32); full[y0:y1] = hm
    return score, box, cv2.GaussianBlur(full, (0,0), 9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("indir", type=Path)
    ap.add_argument("--out", type=Path, default=Path("./黑纱检测输出"))
    ap.add_argument("--cam-regex", default=r"(Cam\d)")
    ap.add_argument("--thr", type=float, default=None, help="固定阈值(不给则按相机自校准)")
    ap.add_argument("--calib-dir", type=Path, default=None,
                    help="确认正常的帧目录:用于预填参考缓冲+定阈(真实产线流程,上线前先录一段正常运行)")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = timm.create_model(MODEL, pretrained=True, num_classes=0, dynamic_img_size=True).eval().to(dev)
    print(f"模型 {MODEL} | 最后{LAYERS}层 | top-{TOPK} | 缓冲{BUF} | warm-up≥{MIN_REF} | "
          f"入侵屏蔽={'开' if INTRUDER_MASK else '关'} | 门控缓冲={'开' if GATED else '关'} | device={dev}")

    bufs = defaultdict(lambda: deque(maxlen=BUF))     # cam → 历史特征
    labs = defaultdict(lambda: deque(maxlen=BUF))     # cam → 历史帧筘齿带 LAB 缩略图（入侵屏蔽用，约 1 MB/帧）
    gate_run = defaultdict(int)                       # cam → 连续被门控挡在池外的帧数
    def mask_for(lab_cur, lab_refs, gh, gw):
        return ~intruder_mask(lab_cur, list(lab_refs), gh, gw) if INTRUDER_MASK else None
    warm = defaultdict(list)                          # cam → warm-up 期分数(用于自校准)
    thrs = {}                                         # cam → 阈值
    # ── 标定阶段:用确认正常的帧预填缓冲并定阈(产线流程) ──
    if a.calib_dir and a.calib_dir.exists():
        cal = defaultdict(list)
        for p in sorted(a.calib_dir.iterdir()):
            if p.suffix.lower() not in (".jpg", ".jpeg", ".png"): continue
            m = re.search(a.cam_regex, p.name)
            img = read_bgr(p)
            if img is None: continue
            f, y0, y1 = band_feat(model, dev, img)
            lb = band_lab(img, y0, y1, f.shape[0], f.shape[1]) if INTRUDER_MASK else None
            cal[m.group(1) if m else "default"].append((f, img.shape[:2], y0, y1, lb))
        for cam, items in cal.items():
            for f, _, _, _, lb in items[-BUF:]:
                bufs[cam].append(f)
                if INTRUDER_MASK: labs[cam].append(lb)
            loo = []
            for i, (f, hw, y0, y1, lb) in enumerate(items):
                rest = [g for j,(g,_,_,_,_) in enumerate(items) if j != i]
                if len(rest) >= MIN_REF:
                    v = mask_for(lb, [it[4] for j, it in enumerate(items) if j != i], *f.shape[:2])
                    s, _, _ = analyze(f, rest, hw, y0, y1, valid=v); loo.append(s)
            if len(loo) >= MIN_REF:
                thrs[cam] = robust_thr(loo)
                print(f"  [标定 {cam}] {len(items)}帧正常 → 阈值={thrs[cam]:.0f} "
                      f"(分数中位{np.median(loo):.0f}, 最大{max(loo):.0f})")
            else:
                print(f"  [标定 {cam}] 正常帧不足({len(items)}), 将在流中自校准")

    files = sorted(p for p in a.indir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    log = (a.out/"alarms.jsonl").open("w", encoding="utf-8")
    n_alarm = 0
    for p in files:
        m = re.search(a.cam_regex, p.name)
        cam = m.group(1) if m else "default"
        img = read_bgr(p)
        if img is None: continue
        feat, y0, y1 = band_feat(model, dev, img)
        lab_cur = band_lab(img, y0, y1, feat.shape[0], feat.shape[1]) if INTRUDER_MASK else None
        refs = list(bufs[cam])
        status = "warmup"
        if len(refs) >= MIN_REF:
            valid = mask_for(lab_cur, labs[cam], *feat.shape[:2])
            score, box, hm = analyze(feat, refs, img.shape[:2], y0, y1, valid=valid)
            thr = a.thr if a.thr is not None else thrs.get(cam)
            if thr is None:
                # 留一法(LOO)自校准:用缓冲区自身立刻定阈,无需额外等待帧
                # (缓冲帧假定以正常为主 —— 无监督校准的标准假设)
                loo = []
                ref_labs = list(labs[cam]) if INTRUDER_MASK else []
                for i in range(len(refs)):
                    rest = refs[:i] + refs[i+1:]
                    if len(rest) >= MIN_REF - 1:
                        v = mask_for(ref_labs[i], ref_labs[:i] + ref_labs[i+1:], *feat.shape[:2]) if INTRUDER_MASK else None
                        s, _, _ = analyze(refs[i], rest, img.shape[:2], y0, y1, valid=v)
                        loo.append(s)
                if len(loo) >= MIN_REF - 1:
                    thrs[cam] = robust_thr(loo)
                    print(f"  [{cam}] LOO自校准完成 阈值={thrs[cam]:.0f} "
                          f"(缓冲{len(loo)}帧, 中位{np.median(loo):.0f})")
                    thr = thrs[cam]
            if thr is None:
                status = "calibrating"
            else:
                alarm = score >= thr
                status = "ALARM" if alarm else "ok"
                if alarm:
                    n_alarm += 1
                    hn = np.clip((hm-hm.min())/(hm.max()-hm.min()+1e-6),0,1)
                    vis = cv2.addWeighted(img, 0.6, cv2.applyColorMap((hn*255).astype(np.uint8), cv2.COLORMAP_JET), 0.4, 0)
                    if box: cv2.rectangle(vis, (box[0],box[1]), (box[2],box[3]), (0,0,255), 4)
                    cv2.putText(vis, f"ALARM score={score:.0f} thr={thr:.0f}", (30, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0,0,255), 3)
                    imwrite(a.out/f"ALARM_{p.stem}.jpg", vis)
                log.write(json.dumps(dict(file=p.name, cam=cam, score=round(score,1),
                                          threshold=round(thr,1), alarm=bool(alarm), box=box),
                                     ensure_ascii=False)+"\n")
        if GATED and status == "ALARM" and gate_run[cam] < GATE_MAX:
            status = "ALARM(未入池)"          # 门控：报警帧不进参考池
            gate_run[cam] += 1
        else:
            if GATED and status == "ALARM":
                status = "ALARM(强制入池)"    # 连续被挡 GATE_MAX 帧，放行以跟上新常态
            gate_run[cam] = 0
            bufs[cam].append(feat)
            if INTRUDER_MASK: labs[cam].append(lab_cur)
        print(f"  {p.name[:44]:<44} {cam:<6} {status}")
    log.close()
    print(f"\n处理 {len(files)} 帧 | 报警 {n_alarm} 次 | 输出 {a.out}")
    print(f"日志: {a.out/'alarms.jsonl'}")


if __name__ == "__main__":
    main()
