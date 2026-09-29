"""统一指标计算的确定性回归。

@spec docs/spec.md#4.8
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import evaluate_detection_metrics as M


def close(a, b, eps=1e-12):
    assert a is not None and abs(a - b) < eps, (a, b)


def fixture():
    rows = [
        {"file": "n_fp", "defect": 0, "score": .9, "alarm": 1, "hit": 0,
         "box": None},
        {"file": "n_tn", "defect": 0, "score": .1, "alarm": 0, "hit": 0,
         "box": None},
        {"file": "d_tp", "defect": 1, "score": .8, "alarm": 1, "hit": 1,
         "box": [0, 0, 10, 10]},
        {"file": "d_fn", "defect": 1, "score": .2, "alarm": 0, "hit": 0,
         "box": [0, 0, 5, 5]},
    ]
    gt = {"n_fp": [], "n_tn": [], "d_tp": [[0, 0, 10, 10]],
          "d_fn": [[0, 0, 10, 10]]}
    return rows, gt


def main():
    rows, gt = fixture()
    r, enriched = M.evaluate_rows(rows, gt, observation_hours=2.0)
    assert r["confusion_matrix"] == {"tp": 1, "fp": 1, "tn": 1, "fn": 1}
    for key in ("false_positive_rate", "recall", "precision", "auroc"):
        close(r["detection"][key], .5)
    close(r["detection"]["false_alarms_per_hour"], .5)
    close(r["localization"]["point_hit_rate"], .5)
    close(r["localization"]["box_iou_mean"], .625)
    close(r["localization"]["box_iou_median"], .625)
    close(r["localization"]["box_recall_iou_0_1"], 1.0)
    close(r["localization"]["box_recall_iou_0_3"], .5)
    close(r["localization"]["box_recall_iou_0_5"], .5)
    close(r["localization"]["alarm_and_point_hit_rate"], .5)
    close(r["localization"]["alarm_and_point_hit_precision"], .5)
    close(enriched[2]["best_box_iou"], 1.0)
    close(enriched[3]["best_box_iou"], .25)

    unavailable, _ = M.evaluate_rows(rows, gt)
    assert unavailable["detection"]["false_alarms_per_hour"] is None
    assert unavailable["detection"]["hourly_metric_unavailable_reason"]

    by_period, _ = M.evaluate_rows(rows, gt, frame_period_seconds=1800)
    close(by_period["detection"]["observation_hours"], 2.0)
    close(by_period["detection"]["false_alarms_per_hour"], .5)
    print("detection metrics tests: ALL PASS (24 assertions)")


if __name__ == "__main__":
    main()
