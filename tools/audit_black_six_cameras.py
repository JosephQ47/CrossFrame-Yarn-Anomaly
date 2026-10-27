"""Classify archived July-batch M0/M4 failures across all six cameras."""

import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
M0 = ROOT / "data/black_fusion_2026-10-25/roi_fc_results.json"
M4 = ROOT / "data/black_fewshot_semantic_2026-10-25/M4_results.json"
GATE = ROOT / "data/black_hand_gate_tradeoff_2026-10-26.json"
OUT = ROOT / "data/black_six_camera_failure_audit_2026-10-27.json"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def classify(row):
    if not row["defect"]:
        return "normal_false_alarm" if row["alarm"] else "normal_correct"
    if row["alarm"] and row["point_hit"]:
        return "positive_alarm_and_point_hit"
    if row["alarm"]:
        return "positive_alarm_wrong_point"
    if row["point_hit"]:
        return "positive_below_threshold_point_hit"
    return "positive_below_threshold_wrong_point"


def main():
    m0 = {r["id"]: r for r in read(M0)["m0_rows"]}
    m4 = {r["id"]: r for r in read(M4)["rows"]}
    gate = {r["id"]: r for r in read(GATE)["frames"]}
    assert len(m0) == len(m4) == len(gate) == 69
    assert set(m0) == set(m4) == set(gate)
    rows = []
    for frame_id in sorted(m4):
        a, b, g = m0[frame_id], m4[frame_id], gate[frame_id]
        assert a["cam"] == b["cam"] == g["cam"]
        assert a["defect"] == b["defect"] == g["defect"]
        rows.append({
            "id": frame_id, "cam": b["cam"], "defect": bool(b["defect"]),
            "gate_state": g["gate_state"],
            "m0": {key: a[key] for key in ("score", "threshold", "alarm", "point_hit")}
                  | {"failure_class": classify(a)},
            "m4": {key: b[key] for key in ("score", "threshold", "alarm", "point_hit", "pred_box")}
                  | {"failure_class": classify(b)},
        })
    summary = {}
    for cam in sorted({r["cam"] for r in rows}):
        sub = [r for r in rows if r["cam"] == cam]
        summary[cam] = {"frames": len(sub), "positive": sum(r["defect"] for r in sub),
                        "normal": sum(not r["defect"] for r in sub),
                        "gate_rejected": sum(r["gate_state"] != "ELIGIBLE" for r in sub)}
        for model in ("m0", "m4"):
            classes = defaultdict(int)
            for r in sub:
                classes[r[model]["failure_class"]] += 1
            summary[cam][model] = dict(sorted(classes.items()))
        normals = [r["m4"]["score"] for r in sub if not r["defect"]]
        positives = [r["m4"]["score"] for r in sub if r["defect"]]
        summary[cam]["m4_test_label_diagnostic_margin"] = min(positives) - max(normals)
    OUT.write_text(json.dumps({
        "scope": "single July fault event; six-fold retrospective predictions, no retraining",
        "note": "test-label margins diagnose separability only; never select thresholds from them",
        "summary": summary, "rows": rows,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
