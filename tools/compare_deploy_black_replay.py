"""Compare old and cap-55 deployment logs on the same retrospective replay."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "black_cap55_replay_2026-10-25"


def read_log(path):
    with path.open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    assert len(rows) == 16
    return {r["file"]: r for r in rows}


def box_contains_gt_center(box, gt):
    if box is None:
        return False
    x = (gt[0] + gt[2]) / 2
    y = (gt[1] + gt[3]) / 2
    return box[0] <= x <= box[2] and box[1] <= y <= box[3]


def main():
    old = read_log(OUT / "old_alarms.jsonl")
    new = read_log(OUT / "cap55_alarms.jsonl")
    idx = json.loads((ROOT / "data" / "black_fewshot_semantic_2026-10-25" /
                      "index.json").read_text(encoding="utf-8"))["data"]
    gt = {r["id"] + ".jpg": r["gt"][0] for r in idx if r["gt"]}
    assert set(old) == set(new) == set(gt)
    rows = []
    for name in sorted(gt):
        a, b = old[name], new[name]
        if a["state"] == "NOT_READY" or b["state"] == "NOT_READY":
            assert a["state"] == b["state"] == "NOT_READY"
        rows.append({"file": name, "cam": a["cam"], "gt": gt[name],
                     "old": {"state": a["state"], "score": a["score"],
                             "threshold": a["threshold"], "alarm": a["alarm"],
                             "box": a["box"], "box_contains_gt_center":
                             box_contains_gt_center(a["box"], gt[name])},
                     "cap55": {"state": b["state"], "score": b["score"],
                               "threshold": b["threshold"], "alarm": b["alarm"],
                               "box": b["box"], "box_contains_gt_center":
                               box_contains_gt_center(b["box"], gt[name])}})
    ready = [r for r in rows if r["old"]["state"] != "NOT_READY"]

    def metrics(key):
        return {"ready": len(ready), "alarms": sum(r[key]["alarm"] for r in ready),
                "box_contains_gt_center": sum(r[key]["box_contains_gt_center"] for r in ready),
                "alarm_and_box_contains_gt_center": sum(r[key]["alarm"] and
                    r[key]["box_contains_gt_center"] for r in ready)}

    report = {"summary": {"old": metrics("old"), "cap55": metrics("cap55"),
                          "changed_alarm_ids": [r["file"] for r in ready if
                                                r["old"]["alarm"] != r["cap55"]["alarm"]],
                          "changed_box_ids": [r["file"] for r in ready if
                                              r["old"]["box"] != r["cap55"]["box"]],
                          "protocol": "53 confirmed normal images prefilled, 16 positive images "
                                      "replayed; retrospective non-causal reference pool"},
              "rows": rows}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "comparison.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                         encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
