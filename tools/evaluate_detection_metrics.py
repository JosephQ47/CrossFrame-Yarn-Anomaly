"""统一计算检测、定位和单位时间误报警指标。

输入为逐帧 CSV，至少含 ``file,defect,score,alarm,hit,box``。GT 框始终从
LabelMe JSON 回填；每小时误报警只在显式给出连续时长或固定帧周期时计算。

@spec docs/spec.md#4.8
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable


def ratio(k: int, n: int) -> float | None:
    return k / n if n else None


def wilson(k: int, n: int, z: float = 1.959963984540054) -> list[float] | None:
    if not n:
        return None
    p = k / n; den = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [max(0.0, center - half), min(1.0, center + half)]


def auc_rank(labels: list[int], scores: list[float]) -> float | None:
    pos = [s for y, s in zip(labels, scores) if y]
    neg = [s for y, s in zip(labels, scores) if not y]
    if not pos or not neg:
        return None
    wins = sum(1.0 if p > n else 0.5 if p == n else 0.0 for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def parse_value(value: Any) -> Any:
    if value is None or value == "" or str(value).lower() in {"none", "null", "nan"}:
        return None
    if isinstance(value, (list, tuple)):
        return value
    try:
        return json.loads(str(value))
    except json.JSONDecodeError:
        return ast.literal_eval(str(value))


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def pbox(points: list[list[float]]) -> list[float]:
    xs = [float(p[0]) for p in points]; ys = [float(p[1]) for p in points]
    return [min(xs), min(ys), max(xs), max(ys)]


def load_labelme_gt(dataset: Path) -> dict[str, list[list[float]]]:
    gt: dict[str, list[list[float]]] = {}
    for jp in dataset.glob("*.json"):
        data = json.loads(jp.read_text(encoding="utf-8"))
        gt[jp.stem] = [pbox(s["points"]) for s in data.get("shapes", [])
                       if str(s.get("label", "")).lower() == "detect"]
    return gt


def box_iou(a: list[float] | None, b: list[float] | None) -> float:
    if not a or not b:
        return 0.0
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    inter = (x1 - x0) * (y1 - y0)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / max(area_a + area_b - inter, 1e-12)


def best_iou(pred: list[float] | None, gt_boxes: list[list[float]]) -> float:
    return max((box_iou(pred, gt) for gt in gt_boxes), default=0.0)


def median(values: list[float]) -> float | None:
    if not values:
        return None
    s = sorted(values); n = len(s); m = n // 2
    return s[m] if n % 2 else (s[m - 1] + s[m]) / 2.0


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def evaluate_rows(rows: list[dict[str, Any]], gt_by_file: dict[str, list[list[float]]],
                  observation_hours: float | None = None,
                  frame_period_seconds: float | None = None) -> tuple[dict, list[dict]]:
    if observation_hours is not None and frame_period_seconds is not None:
        raise ValueError("observation_hours 与 frame_period_seconds 只能提供一个")
    if observation_hours is not None and observation_hours <= 0:
        raise ValueError("observation_hours 必须 > 0")
    if frame_period_seconds is not None and frame_period_seconds <= 0:
        raise ValueError("frame_period_seconds 必须 > 0")

    enriched = []
    for raw in rows:
        r = dict(raw)
        label = int(r["defect"])
        alarm = as_bool(r["alarm"])
        score = float(r["score"])
        point_hit = None if r.get("hit") in (None, "") else as_bool(r["hit"])
        pred_box = parse_value(r.get("box"))
        key = Path(str(r["file"])).stem
        gt_boxes = gt_by_file.get(key, [])
        if label and key not in gt_by_file:
            raise KeyError(f"缺陷帧缺少GT: {key}")
        iou = best_iou(pred_box, gt_boxes) if label else None
        r.update(defect=label, alarm=int(alarm), score=score,
                 point_hit=None if point_hit is None else int(point_hit),
                 gt_boxes=gt_boxes, best_box_iou=iou)
        enriched.append(r)

    tp = sum(r["defect"] and r["alarm"] for r in enriched)
    fp = sum(not r["defect"] and r["alarm"] for r in enriched)
    tn = sum(not r["defect"] and not r["alarm"] for r in enriched)
    fn = sum(r["defect"] and not r["alarm"] for r in enriched)
    n_pos, n_neg = tp + fn, fp + tn
    point_known = [r for r in enriched if r["defect"] and r["point_hit"] is not None]
    point_hits = sum(r["point_hit"] for r in point_known)
    joint = sum(r["defect"] and r["alarm"] and r["point_hit"] == 1 for r in enriched)
    alarms = tp + fp
    ious = [float(r["best_box_iou"]) for r in enriched if r["defect"]]

    if observation_hours is not None:
        hours = float(observation_hours)
        hour_source = "explicit_observation_hours"
        unavailable = None
    elif frame_period_seconds is not None:
        hours = len(enriched) * float(frame_period_seconds) / 3600.0
        hour_source = "frame_count_x_fixed_period"
        unavailable = None
    else:
        hours = None
        hour_source = None
        unavailable = ("未提供连续观测小时数或固定帧周期；离散抽样图片不能按文件首末时间"
                       "换算每小时误报警")

    false_positive_rate = ratio(fp, n_neg)
    recall = ratio(tp, n_pos)
    precision = ratio(tp, alarms)
    point_hit_rate = ratio(point_hits, len(point_known))
    alarm_and_point_hit_rate = ratio(joint, n_pos)
    alarm_and_point_hit_precision = ratio(joint, alarms)
    result = {
        "frames": len(enriched),
        "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "detection": {
            "false_positive_rate": false_positive_rate,
            "false_positive_rate_ci95": wilson(fp, n_neg),
            "recall": recall,
            "recall_ci95": wilson(tp, n_pos),
            "precision": precision,
            "precision_ci95": wilson(tp, alarms),
            "auroc": auc_rank([r["defect"] for r in enriched],
                                [r["score"] for r in enriched]),
            "false_alarms_per_hour": None if hours is None else fp / hours,
            "observation_hours": hours,
            "hourly_duration_source": hour_source,
            "hourly_metric_unavailable_reason": unavailable,
        },
        "localization": {
            "point_hit_count": point_hits,
            "point_hit_denominator": len(point_known),
            "point_hit_rate": point_hit_rate,
            "box_iou_mean": mean(ious),
            "box_iou_median": median(ious),
            "box_recall_iou_0_1": ratio(sum(v >= 0.1 for v in ious), len(ious)),
            "box_recall_iou_0_3": ratio(sum(v >= 0.3 for v in ious), len(ious)),
            "box_recall_iou_0_5": ratio(sum(v >= 0.5 for v in ious), len(ious)),
            "alarm_and_point_hit_count": joint,
            "alarm_and_point_hit_rate": alarm_and_point_hit_rate,
            "alarm_and_point_hit_precision": alarm_and_point_hit_precision,
        },
    }
    return result, enriched


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            out = dict(row)
            out["gt_boxes"] = json.dumps(out.get("gt_boxes"), ensure_ascii=False)
            out["box"] = json.dumps(parse_value(out.get("box")), ensure_ascii=False)
            w.writerow(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("input", type=Path, help="逐帧CSV")
    ap.add_argument("--dataset", type=Path, required=True, help="LabelMe JSON目录")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--strategy")
    ap.add_argument("--condition")
    ap.add_argument("--exclude-cam", action="append", default=[])
    duration = ap.add_mutually_exclusive_group()
    duration.add_argument("--observation-hours", type=float)
    duration.add_argument("--frame-period-seconds", type=float)
    args = ap.parse_args()

    rows = read_csv(args.input)
    if args.strategy:
        rows = [r for r in rows if r.get("strategy") == args.strategy]
    if args.condition:
        rows = [r for r in rows if r.get("condition") == args.condition]
    if args.exclude_cam:
        rows = [r for r in rows if r.get("cam") not in set(args.exclude_cam)]
    if not rows:
        raise SystemExit("过滤后没有可评估帧")
    result, enriched = evaluate_rows(rows, load_labelme_gt(args.dataset),
                                     args.observation_hours, args.frame_period_seconds)
    result["selection"] = {
        "input": str(args.input), "strategy": args.strategy,
        "condition": args.condition, "excluded_cameras": args.exclude_cam,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(args.out / "metrics_per_frame.csv", enriched)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
