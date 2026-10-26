"""Retrospective gate accounting on archived July black-yarn predictions.

This censors already-computed scores; it does not replay references or infer
the performance of an online detector with a changing reference pool.
"""

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hand_gate import HandIntrusionGate  # noqa: E402

M0 = ROOT / "data/black_fusion_2026-10-25/roi_fc_results.json"
M4 = ROOT / "data/black_fewshot_semantic_2026-10-25/M4_results.json"
MANUAL = ROOT / "reports/hand_gate/manual_labels_roi_intrusion.csv"
DLL = ROOT / "reports/hand_gate/production_dll_roi_results.csv"
OUTPUT = ROOT / "data/black_hand_gate_tradeoff_2026-10-26.json"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def boolean(value):
    value = value.strip().lower()
    if value in {"true", "1", "yes"}:
        return True
    if value in {"false", "0", "no"}:
        return False
    raise ValueError(f"invalid boolean: {value!r}")


def keyed(rows, field):
    result = {Path(row["file"]).stem: boolean(row[field]) for row in rows}
    assert len(result) == len(rows), "duplicate basename in hand evidence"
    return result


def count_outcomes(rows, model, eligible_only=False):
    included = [row for row in rows if not eligible_only or row["gate_state"] == "ELIGIBLE"]
    positives = [row for row in included if row["defect"]]
    normals = [row for row in included if not row["defect"]]
    return {
        "frames": len(included),
        "positive_frames": len(positives),
        "normal_frames": len(normals),
        "positive_alarms": sum(row[model]["alarm"] and row["gate_state"] == "ELIGIBLE" for row in positives),
        "normal_false_alarms": sum(row[model]["alarm"] and row["gate_state"] == "ELIGIBLE" for row in normals),
        "positive_alarm_and_point_hit": sum(
            row[model]["alarm"] and row[model]["point_hit"] and row["gate_state"] == "ELIGIBLE"
            for row in positives),
    }


def main():
    m0 = {row["id"]: row for row in read_json(M0)["m0_rows"]}
    m4 = {row["id"]: row for row in read_json(M4)["rows"]}
    manual = keyed(read_csv(MANUAL), "hand_intrusion")
    dll = keyed(read_csv(DLL), "predicted")
    assert len(m0) == len(m4) == 69 and set(m0) == set(m4)
    assert set(m4) <= set(manual) and set(m4) <= set(dll)
    assert all(manual[frame_id] == dll[frame_id] for frame_id in m4), "DLL/hand-label mismatch"

    by_cam = defaultdict(list)
    for row in m4.values():
        by_cam[row["cam"]].append(row)
    states = {}
    for cam, rows in by_cam.items():
        gate = HandIntrusionGate(clear_frames_required=2)
        for row in sorted(rows, key=lambda item: item["id"]):
            result = gate.update(manual[row["id"]])
            states[row["id"]] = result.value if result is not None else "ELIGIBLE"

    joined = []
    for frame_id, row in sorted(m4.items()):
        base = m0[frame_id]
        assert row["cam"] == base["cam"] and row["defect"] == base["defect"]
        joined.append({
            "id": frame_id,
            "cam": row["cam"],
            "defect": bool(row["defect"]),
            "manual_hand": manual[frame_id],
            "production_dll_hand": dll[frame_id],
            "gate_state": states[frame_id],
            "m0": {"alarm": bool(base["alarm"]), "point_hit": bool(base["point_hit"])},
            "m4": {"alarm": bool(row["alarm"]), "point_hit": bool(row["point_hit"])},
        })

    summary = {
        "frames": len(joined),
        "manual_hand_frames": sum(row["manual_hand"] for row in joined),
        "dll_disagreements": sum(row["manual_hand"] != row["production_dll_hand"] for row in joined),
        "gate_states": {state: sum(row["gate_state"] == state for row in joined)
                        for state in ("HAND_INTRUSION", "RECOVERING", "ELIGIBLE")},
        "m0_all_frames_without_gate": count_outcomes(
            [{**row, "gate_state": "ELIGIBLE"} for row in joined], "m0"),
        "m4_all_frames_without_gate": count_outcomes(
            [{**row, "gate_state": "ELIGIBLE"} for row in joined], "m4"),
        "m0_after_gate_all_frames": count_outcomes(joined, "m0"),
        "m4_after_gate_all_frames": count_outcomes(joined, "m4"),
        "m0_after_gate_eligible_only": count_outcomes(joined, "m0", eligible_only=True),
        "m4_after_gate_eligible_only": count_outcomes(joined, "m4", eligible_only=True),
    }
    OUTPUT.write_text(json.dumps({
        "scope": "July retrospective gate accounting only; same-fault batch, archived scores",
        "gate": "per camera; hand frame and first hand-free recovery frame skipped",
        "summary": summary,
        "frames": joined,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
