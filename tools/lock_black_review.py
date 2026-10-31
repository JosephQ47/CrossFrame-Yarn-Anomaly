"""Lock a completed blind review as a local, immutable CSV and hash manifest.

The output directory is gitignored; a second reviewer must finish independently
before labels are compared or used for model selection.
"""

import argparse
import csv
import hashlib
import io
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import black_review_server as review  # noqa: E402

OUT = ROOT / "outputs/black_july_blind_review"
VALID = {"normal", "black_yarn", "other_defect", "uncertain"}


def lock(reviewer, out):
    rows = review.read_sheet(reviewer)
    review.verified_images()
    if any(not r["reviewed_at"] or r["defect_status"] not in VALID for r in rows):
        raise ValueError(f"{reviewer}: unreviewed or invalid status remains")
    for row in rows:
        boxes = json.loads(row["box_or_centerline_json"] or "[]")
        if row["defect_status"] in ("black_yarn", "other_defect") and not boxes and not row["notes"].strip():
            raise ValueError(f"{reviewer}: defect without location or explanation: {row['frame_id']}")
        if row["defect_status"] == "normal" and boxes:
            raise ValueError(f"{reviewer}: normal frame has a defect box: {row['frame_id']}")
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / f"july_reviewer_{reviewer}_locked.csv"
    manifest_path = out / f"july_reviewer_{reviewer}_lock.json"
    if csv_path.exists() or manifest_path.exists():
        raise FileExistsError("locked review already exists; never overwrite it")
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=review.FIELDS)
    writer.writeheader()
    writer.writerows(rows)
    data = ("\ufeff" + buffer.getvalue()).encode("utf-8")
    digest = hashlib.sha256(data).hexdigest()
    csv_path.write_bytes(data)
    manifest = {
        "reviewer": reviewer,
        "locked_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "rows": len(rows),
        "status_counts": {s: sum(r["defect_status"] == s for r in rows) for s in sorted(VALID)},
        "csv_sha256": digest,
        "source_image_sha256": {r["frame_id"]: r["sha256"] for r in rows},
        "independent_second_review_complete": False,
        "labels_available_for_training": False,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                             encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("reviewer", choices=("A", "B"))
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    print(json.dumps(lock(args.reviewer, args.out), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
