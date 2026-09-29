"""跨帧检测光照泛化压力测试。

三种口径共享同一查询帧，且参考库只含正常帧：
  original_ref          原始光照参考 + 原始阈值（瞬时光照变化）
  adjusted_reference    将参考图片调到查询光照 + 原始阈值（只调样本库）
  matched_recalibration 将参考图片调到查询光照 + 同条件重标定（可达上限）

原始数据只读。合成结果用于测算法敏感边界，不替代真实早/中/晚现场验证。

@spec docs/spec.md#4.7
"""
from __future__ import annotations

import argparse
import csv
import gc
import glob
import hashlib
import json
import math
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import cv2
import numpy as np
import torch
import timm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import black_yarn_detector as D


EV_MIN = -1.5
EV_MAX = 1.5
EV_STEP = 0.25
SAFE_FPR = 0.05
SAFE_RECALL_DROP = 1
SAFE_LOCATION_HITS = 14
MIN_CLEAN_REFS = 8

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "dev_log" / "illumination_robustness_evidence"


@dataclass
class Rec:
    path: Path
    cam: str
    dets: list[list[float]]

    @property
    def defect(self) -> bool:
        return bool(self.dets)


@dataclass(frozen=True)
class Condition:
    family: str
    name: str
    value: float
    transform: Callable[[np.ndarray], np.ndarray]


def srgb_to_linear(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, np.float32) / 255.0
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def linear_to_bgr8(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    y = np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1.0 / 2.4) - 0.055)
    return np.clip(np.rint(y * 255.0), 0, 255).astype(np.uint8)


def exposure_ev(img: np.ndarray, ev: float) -> np.ndarray:
    if abs(ev) < 1e-12:
        return img.copy()
    return linear_to_bgr8(srgb_to_linear(img) * (2.0 ** ev))


def gamma_srgb(img: np.ndarray, gamma: float) -> np.ndarray:
    if abs(gamma - 1.0) < 1e-12:
        return img.copy()
    x = np.asarray(img, np.float32) / 255.0
    return np.clip(np.rint(np.power(x, gamma) * 255.0), 0, 255).astype(np.uint8)


def white_balance(img: np.ndarray, shift: float) -> np.ndarray:
    """BGR 线性域暖/冷偏移。shift>0 为偏暖（R 增、B 减）。"""
    if abs(shift) < 1e-12:
        return img.copy()
    lin = srgb_to_linear(img)
    gain = np.array([1.0 - shift, 1.0, 1.0 + shift], np.float32)
    return linear_to_bgr8(lin * gain[None, None, :])


def _soft_mask(h: int, w: int, orientation: str = "diag") -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    xx /= max(w - 1, 1); yy /= max(h - 1, 1)
    if orientation == "horizontal":
        z = yy
    elif orientation == "vertical":
        z = xx
    else:
        z = (xx + 0.65 * yy) / 1.65
    # 软边覆盖画面约一半，固定几何确保所有策略可复现。
    return np.clip((z - 0.25) / 0.50, 0.0, 1.0)


def local_shadow(img: np.ndarray, strength: float) -> np.ndarray:
    lin = srgb_to_linear(img)
    m = _soft_mask(*img.shape[:2])[..., None]
    return linear_to_bgr8(lin * (1.0 - strength * m))


def stripe_flicker(img: np.ndarray, amplitude: float) -> np.ndarray:
    lin = srgb_to_linear(img)
    h = img.shape[0]
    y = np.arange(h, dtype=np.float32)
    # 12 条明暗带，模拟灯频与滚动快门耦合；平均亮度基本不变。
    gain = 1.0 + amplitude * np.sin(2.0 * np.pi * 12.0 * y / max(h, 1))
    return linear_to_bgr8(lin * gain[:, None, None])


def local_highlight(img: np.ndarray, strength: float) -> np.ndarray:
    lin = srgb_to_linear(img)
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = 0.62 * w, 0.48 * h
    sx, sy = 0.18 * w, 0.16 * h
    m = np.exp(-0.5 * (((xx - cx) / sx) ** 2 + ((yy - cy) / sy) ** 2))[..., None]
    return linear_to_bgr8(lin * (1.0 + strength * m))


def conditions(suite: str) -> list[Condition]:
    evs = np.arange(EV_MIN, EV_MAX + EV_STEP / 2, EV_STEP)
    ev_out = [Condition("exposure_ev", f"ev_{v:+.2f}", float(v),
                        lambda im, x=float(v): exposure_ev(im, x)) for v in evs]
    if suite == "ev":
        return ev_out
    out = [] if suite == "other" else ev_out
    out += [Condition("gamma", f"gamma_{v:.2f}", v,
                      lambda im, x=v: gamma_srgb(im, x))
            for v in (0.65, 0.80, 1.25, 1.50)]
    out += [Condition("white_balance", f"wb_{v:+.2f}", v,
                      lambda im, x=v: white_balance(im, x))
            for v in (-0.20, -0.10, 0.10, 0.20)]
    out += [Condition("local_shadow", f"shadow_{v:.2f}", v,
                      lambda im, x=v: local_shadow(im, x))
            for v in (0.10, 0.20, 0.35, 0.50)]
    out += [Condition("stripe_flicker", f"flicker_{v:.2f}", v,
                      lambda im, x=v: stripe_flicker(im, x))
            for v in (0.05, 0.10, 0.20, 0.30)]
    out += [Condition("local_highlight", f"highlight_{v:.2f}", v,
                      lambda im, x=v: local_highlight(im, x))
            for v in (0.10, 0.20, 0.35, 0.50)]
    return out


def load_records(root: Path) -> list[Rec]:
    seen: set[str] = set()
    recs: list[Rec] = []
    for jp in sorted(glob.glob(str(root / "*.json"))):
        data = json.load(open(jp, encoding="utf-8"))
        ip = root / (Path(jp).stem + ".jpg")
        if not ip.exists():
            continue
        digest = hashlib.md5(ip.read_bytes()).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        dets = [D.pbox(s["points"]) for s in data["shapes"]
                if s["label"].lower() == "detect"]
        mm = re.search(r"(Cam\d)", ip.name)
        recs.append(Rec(ip, mm.group(1) if mm else "NA", dets))
    return recs


def robust_thr(scores: Iterable[float]) -> float:
    a = np.asarray(list(scores), np.float64)
    med = np.median(a)
    mad = 1.4826 * np.median(np.abs(a - med))
    return float(med + 3.5 * max(mad, 1.0))


def auc_rank(labels: list[int], scores: list[float]) -> float:
    y = np.asarray(labels, np.int8); s = np.asarray(scores, np.float64)
    pos, neg = s[y == 1], s[y == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    # Mann–Whitney，含并列 0.5；样本很小，直接广播最清楚。
    return float(((pos[:, None] > neg[None, :]).sum() +
                  0.5 * (pos[:, None] == neg[None, :]).sum()) /
                 (len(pos) * len(neg)))


def wilson(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n <= 0:
        return (float("nan"), float("nan"))
    p = k / n; d = 1.0 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def clipping(img: np.ndarray) -> tuple[float, float]:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float((gray <= 2).mean()), float((gray >= 253).mean())


@torch.no_grad()
def extract_set(model, dev: str, recs: list[Rec], cond: Condition | None):
    feats, meta = [], []
    for rec in recs:
        img = D.read_bgr(rec.path)
        if cond is not None:
            img = cond.transform(img)
        f, y0, y1 = D.band_feat(model, dev, img)
        dark, sat = clipping(img[y0:y1])
        feats.append(f)
        meta.append((img.shape[:2], y0, y1, dark, sat))
    return feats, meta


@torch.no_grad()
def evaluate_with_refs(recs: list[Rec], query_feats: list[np.ndarray], meta,
                       ref_feats: list[np.ndarray], ref_indices: list[int],
                       device: str) -> list[dict]:
    assert all(not recs[i].dets for i in ref_indices)
    refs_gpu = torch.from_numpy(np.stack([ref_feats[i] for i in ref_indices])).to(device)
    rows = []
    for qi, (rec, cur, md) in enumerate(zip(recs, query_feats, meta)):
        cur_gpu = torch.from_numpy(cur).to(device)
        sims = torch.einsum("tijc,ijc->tij", refs_gpu, cur_gpu)
        if qi in ref_indices:
            sims[ref_indices.index(qi)] = -2.0  # 正常查询不得匹配自己的变体
        anom = (1.0 - sims.max(dim=0).values).float().cpu().numpy()
        hw, y0, y1, dark, sat = md
        score, point, box = D.score_and_locate(anom, hw, y0, y1)
        hit = False
        if rec.dets and point:
            hit = any(b[0] <= point[0] <= b[2] and b[1] <= point[1] <= b[3]
                      for b in rec.dets)
        rows.append(dict(file=rec.path.stem, cam=rec.cam, defect=int(rec.defect),
                         score=float(score), hit=int(hit), box=box,
                         dark_frac=dark, saturated_frac=sat))
        del cur_gpu, sims
    del refs_gpu
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return rows


def summarize(rows: list[dict], strategy: str, cond: Condition,
              thresholds: dict[str, float], baseline_recall: int) -> dict:
    for r in rows:
        r["threshold"] = thresholds[r["cam"]]
        r["alarm"] = int(r["score"] >= r["threshold"])
        r["strategy"] = strategy
        r["family"] = cond.family
        r["condition"] = cond.name
        r["value"] = cond.value
    normal = [r for r in rows if not r["defect"]]
    defect = [r for r in rows if r["defect"]]
    fp = sum(r["alarm"] for r in normal)
    tp = sum(r["alarm"] for r in defect)
    loc = sum(r["hit"] for r in defect)
    tp_loc = sum(r["alarm"] and r["hit"] for r in defect)
    fp_ci = wilson(fp, len(normal)); rec_ci = wilson(tp, len(defect))
    location_min = math.ceil((SAFE_LOCATION_HITS / 16.0) * len(defect))
    safe = (fp / max(len(normal), 1) <= SAFE_FPR and
            tp >= baseline_recall - SAFE_RECALL_DROP and
            loc >= location_min)
    return dict(
        strategy=strategy, family=cond.family, condition=cond.name, value=cond.value,
        n_normal=len(normal), false_alarms=fp, fpr=fp / max(len(normal), 1),
        fpr_ci_low=fp_ci[0], fpr_ci_high=fp_ci[1],
        n_defect=len(defect), true_positives=tp, recall=tp / max(len(defect), 1),
        recall_ci_low=rec_ci[0], recall_ci_high=rec_ci[1],
        location_hits=loc, detected_and_located=tp_loc,
        location_hits_min=location_min,
        auroc=auc_rank([r["defect"] for r in rows], [r["score"] for r in rows]),
        threshold_median=float(np.median(list(thresholds.values()))),
        dark_frac_mean=float(np.mean([r["dark_frac"] for r in rows])),
        saturated_frac_mean=float(np.mean([r["saturated_frac"] for r in rows])),
        safe=int(safe),
    )


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", choices=("ev", "other", "all"), default="all")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--data", type=Path, default=D.BF)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--cams", nargs="*", help="调试时只跑指定相机")
    ap.add_argument("--include-underfilled", action="store_true",
                    help="包含正常参考不足8张的机位（仅诊断，不用于部署边界）")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    recs_all = load_records(args.data)
    assert len(recs_all) == 69 and sum(r.defect for r in recs_all) == 16, \
        "只读基准口径必须是去重后 69 帧/16 缺陷"
    cams = sorted(set(r.cam for r in recs_all))
    if args.cams:
        cams = [c for c in cams if c in set(args.cams)]
    ref_counts = {c: sum((r.cam == c and not r.defect) for r in recs_all) for c in cams}
    underfilled = [c for c in cams if ref_counts[c] < MIN_CLEAN_REFS]
    if underfilled and not args.include_underfilled:
        cams = [c for c in cams if c not in underfilled]
        print(f"exclude underfilled cameras: {underfilled} (clean refs < {MIN_CLEAN_REFS})", flush=True)
    conds = conditions(args.suite)
    model = timm.create_model(D.MODEL, pretrained=True, num_classes=0,
                              dynamic_img_size=True).eval().to(args.device)
    print(f"device={args.device} cams={cams} conditions={len(conds)}", flush=True)

    per_frame: list[dict] = []
    # cam -> baseline threshold / baseline recall contribution
    original_thresholds: dict[str, float] = {}
    baseline_rows_by_cam: dict[str, list[dict]] = {}
    condition_rows: dict[tuple[str, str], list[dict]] = {}
    condition_thresholds: dict[tuple[str, str], dict[str, float]] = {}

    for cam in cams:
        recs = [r for r in recs_all if r.cam == cam]
        ref_idx = [i for i, r in enumerate(recs) if not r.defect]
        refs = [recs[i] for i in ref_idx]
        assert all(not r.dets for r in refs)
        print(f"[{cam}] {len(recs)} frames / {len(ref_idx)} clean refs", flush=True)
        orig_feats, orig_meta = extract_set(model, args.device, recs, None)
        base_rows = evaluate_with_refs(recs, orig_feats, orig_meta, orig_feats, ref_idx, args.device)
        base_thr = robust_thr(r["score"] for r in base_rows if not r["defect"])
        original_thresholds[cam] = base_thr
        baseline_rows_by_cam[cam] = base_rows

        for ci, cond in enumerate(conds, 1):
            t0 = time.time()
            if cond.family == "exposure_ev" and abs(cond.value) < 1e-12:
                q_feats, q_meta = orig_feats, orig_meta
            else:
                q_feats, q_meta = extract_set(model, args.device, recs, cond)

            # A: 旧参考/旧阈值；B/C: 样本库图片随条件调整，分数相同但阈值口径不同。
            old_rows = evaluate_with_refs(recs, q_feats, q_meta, orig_feats, ref_idx, args.device)
            adj_rows = evaluate_with_refs(recs, q_feats, q_meta, q_feats, ref_idx, args.device)
            cond_thr = robust_thr(r["score"] for r in adj_rows if not r["defect"])

            condition_rows.setdefault(("original_ref", cond.name), []).extend(old_rows)
            condition_rows.setdefault(("adjusted_reference", cond.name), []).extend(adj_rows)
            condition_rows.setdefault(("matched_recalibration", cond.name), []).extend(
                [dict(r) for r in adj_rows])
            condition_thresholds.setdefault(("original_ref", cond.name), {})[cam] = base_thr
            condition_thresholds.setdefault(("adjusted_reference", cond.name), {})[cam] = base_thr
            condition_thresholds.setdefault(("matched_recalibration", cond.name), {})[cam] = cond_thr
            print(f"  {ci:02d}/{len(conds)} {cond.name:<18} "
                  f"thr {base_thr:.1f}->{cond_thr:.1f} {time.time()-t0:.1f}s", flush=True)
            if q_feats is not orig_feats:
                del q_feats, q_meta
            gc.collect()

        del orig_feats, orig_meta
        gc.collect()

    baseline_all = [r for cam in cams for r in baseline_rows_by_cam[cam]]
    baseline_recall = sum(r["score"] >= original_thresholds[r["cam"]]
                          for r in baseline_all if r["defect"])
    summaries = []
    for cond in conds:
        for strategy in ("original_ref", "adjusted_reference", "matched_recalibration"):
            rows = condition_rows[(strategy, cond.name)]
            sm = summarize(rows, strategy, cond,
                           condition_thresholds[(strategy, cond.name)], baseline_recall)
            summaries.append(sm)
            per_frame.extend(rows)

    write_csv(args.out / "illumination_summary.csv", summaries)
    write_csv(args.out / "illumination_per_frame.csv", per_frame)

    ev_summary = [r for r in summaries if r["family"] == "exposure_ev"]
    boundaries = {}
    for strategy in ("original_ref", "adjusted_reference", "matched_recalibration"):
        safe_vals = sorted(r["value"] for r in ev_summary
                           if r["strategy"] == strategy and r["safe"])
        # 性能边界必须是包含 0 EV 的连续安全区间，不能跨过失败档位取 min/max。
        contiguous = []
        if any(abs(v) < 1e-9 for v in safe_vals):
            by_value = {round(r["value"], 8): bool(r["safe"]) for r in ev_summary
                        if r["strategy"] == strategy}
            contiguous = [0.0]
            v = -EV_STEP
            while v >= EV_MIN - 1e-9 and by_value.get(round(v, 8), False):
                contiguous.append(round(v, 8)); v -= EV_STEP
            v = EV_STEP
            while v <= EV_MAX + 1e-9 and by_value.get(round(v, 8), False):
                contiguous.append(round(v, 8)); v += EV_STEP
            contiguous.sort()
        boundaries[strategy] = dict(
            safe_values=safe_vals,
            contiguous_safe_values=contiguous,
            min_ev=min(contiguous) if contiguous else None,
            max_ev=max(contiguous) if contiguous else None,
        )
    report = dict(
        generated_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        device=args.device, suite=args.suite, cams=cams,
        dataset=dict(frames=len(baseline_all), normal=sum(not r["defect"] for r in baseline_all),
                     defect=sum(r["defect"] for r in baseline_all)),
        baseline_recall_count=int(baseline_recall),
        safe_rule=dict(fpr_max=SAFE_FPR, recall_drop_frames=SAFE_RECALL_DROP,
                       location_ratio_min=SAFE_LOCATION_HITS / 16.0),
        excluded_underfilled=underfilled if not args.include_underfilled else [],
        exposure_boundaries=boundaries,
        caveat="合成光度边界不等于真实照度/色温边界，须由冻结参数的现场早中晚数据复验。",
    )
    (args.out / "illumination_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
