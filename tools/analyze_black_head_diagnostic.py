"""Reproducible G2 comparison and label/normal-threshold sensitivity audit."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
JOINT = ROOT / "data/black_dual_head_2026-10-31/results.json"
FROZEN = ROOT / "data/black_frozen_alarm_2026-10-31/results.json"
SUSPECT = "DI2_Cam3_2026-07-18_10-11-55-177"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit():
    joint = json.loads(JOINT.read_text(encoding="utf-8"))
    frozen = json.loads(FROZEN.read_text(encoding="utf-8"))
    a = {r["id"]: r for r in joint["rows"]}
    b = {r["id"]: r for r in frozen["rows"]}
    if set(a) != set(b) or len(a) != 69:
        raise ValueError("run ID mismatch")
    for key in a:
        if any(a[key][field] != b[key][field]
               for field in ("cam", "defect", "m4_score", "m4_threshold",
                             "m4_alarm", "m4_point_hit")):
            raise ValueError(f"M4 baseline mismatch: {key}")
    folds = []
    for fold in frozen["folds"]:
        normals = fold["val_normal_scores"]
        values = np.asarray([r["score"] for r in normals])
        maximum = int(np.argmax(values))
        without_top = float(np.quantile(np.delete(values, maximum), .95, method="linear"))
        folds.append({"test_cam": fold["test_cam"], "val_cam": fold["val_cam"],
                      "normal_count": len(normals), "p95": fold["threshold"],
                      "highest_normal_id": normals[maximum]["id"],
                      "highest_normal_score": float(values[maximum]),
                      "p95_without_highest_diagnostic_only": without_top})
    cam2 = next(f for f in frozen["folds"] if f["test_cam"] == "Cam2")
    original = {r["id"]: r["score"] for r in cam2["val_normal_scores"]}
    if SUSPECT not in original:
        raise ValueError("suspect not in Cam2 validation normals")
    threshold_without = float(np.quantile(
        [score for key, score in original.items() if key != SUSPECT], .95, method="linear"))
    cam2_test = [r for r in frozen["rows"] if r["cam"] == "Cam2"]
    return {"scope": "single July fault; posthoc sensitivity only; do not select threshold",
            "independent_fault_test": False, "causal_references": False,
            "joint_result_sha256": sha(JOINT), "frozen_result_sha256": sha(FROZEN),
            "same_69_ids_and_m4_baseline": True,
            "joint_vs_m4": {"alarm_and_point_hits": [joint["summary"]["dual"]["alarm_and_point_hits"],
                                                     joint["summary"]["m4"]["alarm_and_point_hits"]],
                            "false_alarms": [joint["summary"]["dual"]["false_alarms"],
                                             joint["summary"]["m4"]["false_alarms"]]},
            "frozen_alarm_vs_m4": {
                "alarm_and_point_hits": [frozen["summary"]["frozen_alarm"]["alarm_and_point_hits"],
                                         frozen["summary"]["m4"]["alarm_and_point_hits"]],
                "false_alarms": [frozen["summary"]["frozen_alarm"]["false_alarms"],
                                 frozen["summary"]["m4"]["false_alarms"]]},
            "normal_p95_influence": folds,
            "cam2_suspect_sensitivity": {
                "suspect_id": SUSPECT, "label_status": "unresolved; independent raw-image review needed",
                "original_p95": cam2["threshold"], "p95_without_suspect_diagnostic_only": threshold_without,
                "test_positive_alarms_at_original": sum(r["defect"] and r["frozen_alarm_alarm"]
                                                        for r in cam2_test),
                "test_positive_alarms_without_suspect_diagnostic_only": sum(
                    r["defect"] and r["frozen_alarm_score"] >= threshold_without for r in cam2_test),
                "test_normal_false_alarms_without_suspect_diagnostic_only": sum(
                    not r["defect"] and r["frozen_alarm_score"] >= threshold_without
                    for r in cam2_test)}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path,
                        default=ROOT / "data/black_head_diagnostic_2026-10-31.json")
    args = parser.parse_args()
    result = audit()
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
