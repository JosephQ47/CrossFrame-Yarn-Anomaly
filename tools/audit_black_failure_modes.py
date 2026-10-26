"""Join archived six-fold black-yarn results for failure-mode auditing.

This script reads existing predictions only. It does not train, select a
threshold, or assign new ground-truth labels.
"""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
M0 = ROOT / "data/black_fusion_2026-10-25/roi_fc_results.json"
M4 = ROOT / "data/black_fewshot_semantic_2026-10-25/M4_results.json"
RANK = ROOT / "data/black_fusion_2026-10-25/oof/yolo_oof_rank_best_results.json"
CANDIDATES = ROOT / "data/black_fusion_2026-10-25/diagnostic.json"
OUTPUT = ROOT / "data/black_failure_modes_2026-10-26.json"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def by_id(rows):
    result = {row["id"]: row for row in rows}
    assert len(result) == len(rows), "duplicate frame ID"
    return result


def main():
    m0 = by_id(read(M0)["m0_rows"])
    m4 = by_id(read(M4)["rows"])
    rank = by_id(read(RANK)["rows"])
    candidates = by_id(read(CANDIDATES)["candidate_rows"])
    ids = set(m4)
    assert len(ids) == 69
    assert ids == set(m0) == set(rank) == set(candidates)
    assert sum(bool(row["defect"]) for row in m4.values()) == 16

    cameras = {}
    for cam in sorted({row["cam"] for row in m4.values()}):
        rows = [row for row in m4.values() if row["cam"] == cam]
        normals = [row for row in rows if not row["defect"]]
        positives = [row for row in rows if row["defect"]]
        thresholds = {row["threshold"] for row in rows}
        val_cams = {row["val_cam"] for row in rows}
        assert len(thresholds) == len(val_cams) == 1
        cameras[cam] = {
            "val_cam": next(iter(val_cams)),
            "m4_threshold": next(iter(thresholds)),
            "normal_count": len(normals),
            "positive_count": len(positives),
            "m4_normal_scores": sorted(row["score"] for row in normals),
            "m4_positive_scores": sorted(row["score"] for row in positives),
            "m4_normal_false_alarms": sum(bool(row["alarm"]) for row in normals),
            "m4_positive_alarms": sum(bool(row["alarm"]) for row in positives),
            "m0_normal_false_alarms": sum(bool(m0[row["id"]]["alarm"]) for row in normals),
            "m0_positive_alarms": sum(bool(m0[row["id"]]["alarm"]) for row in positives),
        }

    selected_ids = sorted(
        row["id"]
        for row in m4.values()
        if (row["cam"] == "Cam3" and row["defect"])
        or (row["cam"] == "Cam5" and (row["alarm"] and not row["defect"] or row["defect"]))
    )
    selected = []
    for frame_id in selected_ids:
        a, b, c, d = m0[frame_id], m4[frame_id], rank[frame_id], candidates[frame_id]
        assert a["cam"] == b["cam"] == c["cam"] == d["cam"]
        assert a["defect"] == b["defect"] == c["defect"] == d["defect"]
        selected.append({
            "id": frame_id,
            "cam": b["cam"],
            "defect": bool(b["defect"]),
            "m0": {key: a[key] for key in ("score", "threshold", "alarm", "point_hit")},
            "m4": {key: b[key] for key in ("score", "threshold", "alarm", "point_hit", "pred_box")},
            "yolo_dino_rank_best_diagnostic": {key: c[key] for key in (
                "score", "threshold", "alarm", "point_hit", "candidate_hit",
                "best_true_candidate_rank", "candidate_count")},
            "m0_top5_candidate_hits": d["candidate_hits"],
        })

    OUTPUT.write_text(json.dumps({
        "scope": "archived six-fold July batch; retrospective, same fault event",
        "source_files": [str(path.relative_to(ROOT)).replace("\\", "/") for path in (M0, M4, RANK, CANDIDATES)],
        "cameras": cameras,
        "selected_frames": selected,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT} ({len(selected)} selected frames)")


if __name__ == "__main__":
    main()
