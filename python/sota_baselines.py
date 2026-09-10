# -*- coding: utf-8 -*-
"""在黑纱基准上跑文献里的标准工业异常检测（IAD）基线，与本方法同口径对比。

为什么要跑：此前的对照大多是自己的失败尝试（改色/微调/手工判据），审稿人会问
"为什么不和 PatchCore / PaDiM / SimpleNet 比"。这里按 one-class 标准协议跑一遍。

基线（均为文献中的标准方法，本地按论文要点轻量复现，不依赖 anomalib）：
  PaDiM      (Defard 2020)  WRN50 layer2+3 特征，逐位置多元高斯，马氏距离
  PatchCore  (Roth 2022)    WRN50 layer2+3 特征，记忆库 + 最近邻距离
  SimpleNet  (Liu 2023)     WRN50 特征 → 特征适配器 + 高斯噪声伪异常 + 判别器
  PatchCore-DINOv2          与本方法**同一骨干、同一特征**，但记忆库取其他帧的**任意位置** patch
                            ——它与本方法只差一件事：有没有"同一位置"的约束

协议（尽量偏向基线，让对比保守）：
  · 输入与本方法完全相同：筘齿带 y∈[0.34,0.60] 缩到高 448，切 588×448 重叠 tile
  · 按机位独立：训练集 = 该机位的**全部正常帧**（基线拿到的是干净数据；本方法的参考池里混着缺陷帧）
  · PaDiM / PatchCore / PatchCore-DINOv2 对正常帧做留一（frame k 的模型不含 k 自己），与本方法一致
  · SimpleNet 需训练，按机位训练一次；其正常帧既在训练集又在测试集 → 对它**有利**，如实标注
  · 图级分数 = 异常图最大值（这些方法的惯例）；定位 = 异常图峰值是否落入 GT 框
  · 评估帧集与本方法一致：去重后 69 帧 / 16 缺陷 / 53 正常

输出：D:/研究生课程/课题/色纱研究/黑纱评估基准/sota_baselines.csv（+ 每帧分数 json）
"""
import os, sys, glob, json, re, csv, time
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE"); os.environ.setdefault("PYTHONUTF8", "1")
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, cv2, torch, timm
import torch.nn as nn, torch.nn.functional as Fn
from sklearn.metrics import roc_auc_score
import black_yarn_detector as D

OUT = Path(r"D:\研究生课程\课题\色纱研究\黑纱评估基准")
SEED = 0
TILE_W, TILE_H, OVERLAP = 592, 448, 144      # 592/8=74 整除；步长 448 → /8=56 整除
STRIDE = 8                                    # WRN50 layer2 步长
MEAN = np.array([0.485, 0.456, 0.406], np.float32); STD = np.array([0.229, 0.224, 0.225], np.float32)
torch.manual_seed(SEED); np.random.seed(SEED)
dev = "cuda" if torch.cuda.is_available() else "cpu"


# ------------------------------------------------------------------ 数据
def load_black():
    recs = []
    for jp in sorted(glob.glob(str(D.BF / "*.json"))):
        d = json.load(open(jp, encoding="utf-8"))
        dets = [D.pbox(s["points"]) for s in d["shapes"] if s["label"].lower() == "detect"]
        ip = D.BF / (Path(jp).stem + ".jpg")
        if not ip.exists(): continue
        mm = re.search(r"(Cam\d)", ip.name)
        recs.append(dict(path=ip, dets=dets, cam=mm.group(1) if mm else "NA"))
    return D.dedup_recs(recs)


def band_and_tiles(img):
    """与 black_yarn_detector.band_feat 相同的裁带/切 tile 几何，但 tile 宽 592 以整除 8。"""
    H, W = img.shape[:2]
    y0, y1 = int(H * D.BAND_Y0), int(H * D.BAND_Y1)
    band = cv2.resize(img[y0:y1], (W, TILE_H))
    xs = list(range(0, max(1, W - TILE_W + 1), TILE_W - OVERLAP))
    if xs[-1] + TILE_W < W: xs.append(W - TILE_W)
    return band, xs, y0, y1


def to_tensor(bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return torch.from_numpy(((rgb - MEAN) / STD).transpose(2, 0, 1)[None]).to(dev)


# ------------------------------------------------------------------ WRN50 带级特征
class WRN:
    def __init__(self):
        self.m = timm.create_model("wide_resnet50_2", pretrained=True, features_only=True,
                                   out_indices=(2, 3)).eval().to(dev)   # timm 索引: 2=layer2(512,s8) 3=layer3(1024,s16)
        self.C = 512 + 1024

    @torch.no_grad()
    def band_feat(self, img):
        """→ (gh, gw, C) 带级特征图，重叠区取平均。gh=448/8=56, gw=W/8。"""
        band, xs, y0, y1 = band_and_tiles(img)
        W = band.shape[1]; gh, gw = TILE_H // STRIDE, W // STRIDE
        acc = np.zeros((gh, gw, self.C), np.float32); cnt = np.zeros((gh, gw, 1), np.float32)
        for x in xs:
            f2, f3 = self.m(to_tensor(band[:, x:x + TILE_W]))
            f3 = Fn.interpolate(f3, size=f2.shape[-2:], mode="bilinear", align_corners=False)
            f = torch.cat([f2, f3], 1)[0].permute(1, 2, 0).cpu().numpy()   # (56, 74, 1536)
            j0 = x // STRIDE; j1 = min(gw, j0 + f.shape[1])
            acc[:, j0:j1] += f[:, :j1 - j0]; cnt[:, j0:j1] += 1
        return acc / np.maximum(cnt, 1), y0, y1


# ------------------------------------------------------------------ 评分与定位（各基线共用）
def score_locate(amap, hw, y0, y1, dets):
    """amap: (gh, gw) 异常图 → 图级分数 = 平滑后最大值；定位 = 峰值是否落入 GT 框。"""
    m = cv2.GaussianBlur(amap.astype(np.float32), (0, 0), 1.0)
    gh, gw = m.shape; H, W = hw
    iy, ix = np.unravel_index(int(np.argmax(m)), m.shape)
    px, py = int((ix + 0.5) * W / gw), int(y0 + (iy + 0.5) * (y1 - y0) / gh)
    hit = any(b[0] <= px <= b[2] and b[1] <= py <= b[3] for b in dets) if dets else False
    return float(m.max()), (px, py), hit


def cc_score_locate(amap, hw, y0, y1, dets, q=99, cap_ref=55, ref_patches=32 * 228):
    """把本方法的评分规则套在任意异常图上：P99 二值化 → 连通域 → 面积 ≤ cap 的最大连通域面积为分数，重心定位。
    cap 按网格大小等比缩放（WRN 网格 56×400 比 DINOv2 32×228 细，55 个 patch 对应约 169 个）。"""
    cap = max(1, round(cap_ref * amap.size / ref_patches))
    thr = np.percentile(amap, q)
    n, lab, stats, cent = cv2.connectedComponentsWithStats((amap >= thr).astype(np.uint8), connectivity=8)
    cands = [(stats[i, cv2.CC_STAT_AREA], i) for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] <= cap]
    if not cands:
        return 0.0, None, False
    area, bi = max(cands)
    gh, gw = amap.shape; H, W = hw
    px, py = int(cent[bi][0] * W / gw), int(y0 + cent[bi][1] * (y1 - y0) / gh)
    hit = any(b[0] <= px <= b[2] and b[1] <= py <= b[3] for b in dets) if dets else False
    return float(area) / (amap.size / ref_patches), (px, py), hit      # 面积按网格比例归一，便于跨网格比较


def summarize(name, scores, labels, hits, ndef, note=""):
    s, l = np.array(scores), np.array(labels)
    auroc = roc_auc_score(l, s) * 100
    neg, pos = s[l == 0], s[l == 1]
    thr = np.percentile(neg, 95)
    r = dict(method=name, hit=f"{hits}/{ndef}", auroc=round(auroc, 1),
             rec95=round((pos >= thr).mean() * 100, 1), fpr95=round((neg >= thr).mean() * 100, 1),
             fpr_at_100=round((neg >= pos.min()).mean() * 100, 1), note=note)
    print(f"  {name:20} 定位 {r['hit']:6}  AUROC {r['auroc']:5.1f}  P95召回 {r['rec95']:5.1f}%  "
          f"误报 {r['fpr95']:4.1f}%  100%召回误报 {r['fpr_at_100']:5.1f}%", flush=True)
    return r, s.tolist()


# ------------------------------------------------------------------ 基线 1：PaDiM
def padim_scores(F, normal_idx, k, proj, chunk=4096):
    """F:(T,gh,gw,C)；对帧 k，用 normal_idx 中 ≠k 的帧在每个位置拟合高斯，算马氏距离（GPU 分块）。"""
    tr = [i for i in normal_idx if i != k]
    X = torch.from_numpy(F[tr][..., proj]).to(dev)          # (n, gh, gw, d)
    n, gh, gw, d = X.shape
    mu = X.mean(0)
    Xc = (X - mu).reshape(n, gh * gw, d)
    q = (torch.from_numpy(F[k][..., proj]).to(dev) - mu).reshape(gh * gw, d)
    eye = 0.01 * torch.eye(d, device=dev)
    out = torch.empty(gh * gw, device=dev)
    for s0 in range(0, gh * gw, chunk):
        xc = Xc[:, s0:s0 + chunk]                                        # (n, p, d)
        cov = torch.einsum('npd,npe->pde', xc, xc) / max(n - 1, 1) + eye
        sol = torch.linalg.solve(cov, q[s0:s0 + chunk].unsqueeze(-1)).squeeze(-1)   # cov^-1 q
        out[s0:s0 + chunk] = (q[s0:s0 + chunk] * sol).sum(-1).clamp_min(0).sqrt()
    return out.reshape(gh, gw).cpu().numpy()


# ------------------------------------------------------------------ 基线 2：PatchCore
@torch.no_grad()
def patchcore_scores(F, normal_idx, k, sub=0.25, seed=0, qchunk=2048):
    """记忆库 = 其他正常帧的全部位置 patch（随机子采样 sub），最近邻 L2 距离（GPU 分块 cdist）。"""
    tr = [i for i in normal_idx if i != k]
    bank = F[tr].reshape(-1, F.shape[-1])
    rng = np.random.default_rng(seed)
    if sub < 1: bank = bank[rng.choice(len(bank), int(len(bank) * sub), replace=False)]
    B = torch.from_numpy(np.ascontiguousarray(bank, np.float32)).to(dev)
    Bn = (B * B).sum(1)
    Q = torch.from_numpy(np.ascontiguousarray(F[k].reshape(-1, F.shape[-1]), np.float32)).to(dev)
    out = torch.empty(len(Q), device=dev)
    for s0 in range(0, len(Q), qchunk):
        q = Q[s0:s0 + qchunk]
        d2 = (q * q).sum(1, keepdim=True) + Bn[None] - 2 * q @ B.T
        out[s0:s0 + qchunk] = d2.min(1).values.clamp_min(0).sqrt()
    del B, Bn, Q
    return out.reshape(F.shape[1:3]).cpu().numpy()


# ------------------------------------------------------------------ 基线 3：SimpleNet
class SimpleNetHead(nn.Module):
    def __init__(self, c, hid=1024):
        super().__init__()
        self.adapt = nn.Linear(c, c, bias=False)
        self.disc = nn.Sequential(nn.Linear(c, hid), nn.BatchNorm1d(hid), nn.LeakyReLU(0.2), nn.Linear(hid, 1))

    def forward(self, x):  # x:(N, c) → 正常为正、异常为负的判别分
        return self.disc(self.adapt(x)).squeeze(-1)


def simplenet_train(F, normal_idx, epochs=30, noise=0.015, th_pos=0.5, th_neg=-0.5, lr=1e-4, bs=4096):
    X = torch.from_numpy(F[normal_idx].reshape(-1, F.shape[-1])).float()
    X = X / (X.norm(dim=1, keepdim=True) + 1e-6) * np.sqrt(X.shape[1])   # 尺度归一，噪声 σ 才有意义
    net = SimpleNetHead(X.shape[1]).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-5)
    n = len(X)
    for ep in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            xb = X[perm[i:i + bs]].to(dev)
            xn = xb + torch.randn_like(xb) * noise * np.sqrt(xb.shape[1])
            tp, tn = net(xb), net(xn)
            loss = Fn.relu(th_pos - tp).mean() + Fn.relu(tn - th_neg).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    return net, float(X.norm(dim=1).mean())


@torch.no_grad()
def simplenet_scores(net, Fk):
    X = torch.from_numpy(Fk.reshape(-1, Fk.shape[-1])).float()
    X = X / (X.norm(dim=1, keepdim=True) + 1e-6) * np.sqrt(X.shape[1])
    out = []
    for i in range(0, len(X), 8192):
        out.append((-net(X[i:i + 8192].to(dev))).cpu())      # 取负：越大越异常
    return torch.cat(out).numpy().reshape(Fk.shape[:2])


# ------------------------------------------------------------------ 主流程
def main():
    t0 = time.time()
    recs = load_black()
    cams = sorted(set(r["cam"] for r in recs))
    wrn = WRN()
    dino = timm.create_model(D.MODEL, pretrained=True, num_classes=0, dynamic_img_size=True).eval().to(dev)
    rng = np.random.default_rng(SEED)
    proj = np.sort(rng.choice(wrn.C, 100, replace=False))        # PaDiM 随机降维到 100（原文 ResNet18 用 100；n≈8 帧，维度不能高）

    BASE = ("PaDiM", "PatchCore", "SimpleNet", "PatchCore-DINOv2")
    res = {m: dict(scores=[], labels=[], hits=0, ndef=0, files=[]) for m in
           BASE + tuple(b + " +CC评分" for b in BASE) + ("Ours",)}

    for c in cams:
        sub = [r for r in recs if r["cam"] == c]
        imgs = [D.read_bgr(r["path"]) for r in sub]
        normal_idx = [i for i, r in enumerate(sub) if not r["dets"]]
        print(f"[{c}] {len(sub)} 帧，正常 {len(normal_idx)}，缺陷 {len(sub)-len(normal_idx)}", flush=True)

        # WRN 特征
        Fw = []; geo = []
        for im in imgs:
            f, y0, y1 = wrn.band_feat(im); Fw.append(f); geo.append((im.shape[:2], y0, y1))
        Fw = np.stack(Fw)
        # DINOv2 特征（与本方法完全相同）
        Fd = np.stack([D.band_feat(dino, dev, im)[0] for im in imgs])

        # SimpleNet 按机位训练一次
        net, _ = simplenet_train(Fw, normal_idx)

        for k, r in enumerate(sub):
            hw, y0, y1 = geo[k]
            maps = {
                "PaDiM": padim_scores(Fw, normal_idx, k, proj),
                "PatchCore": patchcore_scores(Fw, normal_idx, k),
                "SimpleNet": simplenet_scores(net, Fw[k]),
                "PatchCore-DINOv2": patchcore_scores(Fd, normal_idx, k, sub=0.3),
            }
            for m, amap in maps.items():
                sc, pk, hit = score_locate(amap, hw, y0, y1, r["dets"])
                R = res[m]; R["scores"].append(sc); R["labels"].append(int(bool(r["dets"]))); R["files"].append(r["path"].stem)
                if r["dets"]: R["ndef"] += 1; R["hits"] += hit
                sc2, pk2, hit2 = cc_score_locate(amap, hw, y0, y1, r["dets"])     # 同一张异常图，换成本方法的评分规则
                R = res[m + " +CC评分"]; R["scores"].append(sc2); R["labels"].append(int(bool(r["dets"]))); R["files"].append(r["path"].stem)
                if r["dets"]: R["ndef"] += 1; R["hits"] += hit2
            # 本方法（同口径复算，参考池含其他全部帧）
            _, anom = D.anomaly_map(Fd, k, hw, y0, y1)
            sc, pk, bx = D.score_and_locate(anom, hw, y0, y1)
            hit = bool(pk) and any(b[0] <= pk[0] <= b[2] and b[1] <= pk[1] <= b[3] for b in r["dets"])
            R = res["Ours"]; R["scores"].append(sc); R["labels"].append(int(bool(r["dets"]))); R["files"].append(r["path"].stem)
            if r["dets"]: R["ndef"] += 1; R["hits"] += hit
        del Fw, Fd, net; torch.cuda.empty_cache()
        print(f"  {c} 完成  {time.time()-t0:.0f}s", flush=True)

    notes = {"PaDiM": "WRN50 L2+L3，随机降维 100，逐位置高斯；正常帧留一",
             "PatchCore": "WRN50 L2+L3，记忆库=其他正常帧全部位置 patch（25% 子采样）；正常帧留一",
             "SimpleNet": "WRN50 L2+L3，适配器+噪声判别器，按机位训练；其正常帧同时在训练集与测试集（对其有利）",
             "PatchCore-DINOv2": "与本方法同骨干同特征，记忆库=其他正常帧**任意位置** patch（30% 子采样）；正常帧留一",
             "Ours": "跨帧**同位置** top-1 + 连通域面积；参考池=其他全部帧（含缺陷帧，更难）"}
    for b in BASE:
        notes[b + " +CC评分"] = notes[b] + "；图级分数改为本方法的 P99 连通域面积规则（cap 按网格等比缩放）"
    print("\n===== 黑纱基准 69 帧 / 16 缺陷 =====")
    rows, per = [], {}
    for m, R in res.items():
        r, s = summarize(m, R["scores"], R["labels"], R["hits"], R["ndef"], notes[m])
        rows.append(r); per[m] = dict(zip(R["files"], s))
    with open(OUT / "sota_baselines.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    json.dump(dict(labels=dict(zip(res["Ours"]["files"], res["Ours"]["labels"])), scores=per),
              open(OUT / "sota_baselines_scores.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n已写出 {OUT/'sota_baselines.csv'}  用时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
