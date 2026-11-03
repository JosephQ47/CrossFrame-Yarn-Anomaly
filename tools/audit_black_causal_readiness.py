"""Count A-reviewed faults reachable after chronological reference warmup.

Counts are an upper bound: they assume every earlier frame can enter the
reference pool. Actual gate and normal-only policies can only reduce them.
"""

import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import black_hand_detail_server as R  # noqa: E402

OUT = ROOT / "outputs/black_yarn_candidate_audit/causal_readiness.json"


def main():
    lock = json.loads(R.LOCK_MANIFEST.read_text(encoding="utf-8"))
    digest = hashlib.sha256(R.LOCKED.read_bytes()).hexdigest()
    if digest != lock["csv_sha256"]:
        raise RuntimeError("A locked review hash mismatch")
    with R.LOCKED.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 69:
        raise RuntimeError("expected 69 A-reviewed frames")
    by_cam = {}
    for cam in sorted({r["cam"] for r in rows}):
        frames = sorted((r for r in rows if r["cam"] == cam), key=lambda r: r["frame_id"])
        pos = [i for i, r in enumerate(frames) if r["defect_status"] == "black_yarn"]
        by_cam[cam] = {"frames": len(frames), "positive": len(pos),
                       "positive_with_3_earlier_any": sum(i >= 3 for i in pos),
                       "positive_with_8_earlier_any": sum(i >= 8 for i in pos),
                       "positive_with_3_earlier_A_normal":
                       sum(sum(r["defect_status"] == "normal" for r in frames[:i]) >= 3
                           for i in pos)}
    totals = {key: sum(row[key] for row in by_cam.values())
              for key in next(iter(by_cam.values()))}
    report = {"scope": "chronological warmup count on one July event, not detector accuracy",
              "A_label_sha256": digest,
              "reference_policy_note": "earlier-any is only an upper bound; actual gate can reject frames",
              "by_camera": by_cam, "total": totals}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["total"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
