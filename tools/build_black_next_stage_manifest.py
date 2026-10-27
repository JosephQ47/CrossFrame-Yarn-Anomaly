"""Build a read-only July development / F2 locked manifest with provenance."""

import csv
import hashlib
import json
import re
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JULY_INDEX = ROOT / "data/black_fewshot_semantic_2026-10-25/index.json"
HASHES = ROOT / "docs/research/黑纱本地源图哈希清单_2026-10-24.csv"
HAND = ROOT / "reports/hand_gate/manual_labels_roi_intrusion.csv"
READINESS = ROOT / "data/causal_reference_readiness_2026-10-25.json"
OUT = ROOT / "data/black_next_stage_manifest_2026-10-27.csv"


def csv_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def july_time(name):
    m = re.fullmatch(r"DI2_Cam[1-6]_(\d{4}-\d\d-\d\d)_(\d\d)-(\d\d)-(\d\d)-(\d{3})(?: - 副本)?", name)
    if not m:
        raise ValueError(name)
    return datetime.strptime("_".join(m.groups()), "%Y-%m-%d_%H_%M_%S_%f").isoformat(timespec="milliseconds")


def f2_time(name):
    m = re.fullmatch(r"(\d{4})-(\d\d)-(\d\d)-(\d\d)-(\d\d)-(\d\d)-(\d\d)-[0-5]\.jpg", name)
    if not m:
        raise ValueError(name)
    y, mo, d, h, mi, s, cs = map(int, m.groups())
    return (datetime(y, mo, d, h, mi, s) + timedelta(milliseconds=10 * cs)).isoformat(timespec="milliseconds")


def main():
    july = json.loads(JULY_INDEX.read_text(encoding="utf-8"))["data"]
    readiness = json.loads(READINESS.read_text(encoding="utf-8"))
    hashes = csv_rows(HASHES)
    f2 = {r["filename"]: r for r in hashes if r["source"] == "June_F2"}
    hand = {Path(r["file"]).stem: r["hand_intrusion"].strip().lower() == "true"
            for r in csv_rows(HAND)}
    prior_july = {r["id"]: r["prior_confirmed_normal"]
                   for frames in readiness["july_by_cam"].values() for r in frames}
    f2_event = {}
    for cam, events in readiness["f2_events"].items():
        for event in events:
            for position, name in enumerate(event["files"]):
                f2_event[name] = (f"F2_{cam}_cluster{event['event']}", position)
    assert len(july) == len(f2) == len(f2_event) == 69
    assert sum(bool(r["gt"]) for r in july) == 16

    rows = []
    july_hashes = set()
    for r in july:
        source = Path(r["file"])
        actual = digest(source)
        assert actual == r["sha256"], f"July source changed: {source}"
        july_hashes.add(actual)
        rows.append({
            "split": "development_single_fault", "id": r["id"], "source_path": str(source),
            "sha256": actual, "cam": r["cam"], "captured_at": july_time(r["id"]),
            "event_id": "JULY_2026-07-18_single_fault",
            "run_state": "", "speed": "", "defect_label": int(bool(r["gt"])),
            "defect_boxes_json": json.dumps(r["gt"], separators=(",", ":")),
            "defect_label_source": "local LabelMe detect shapes",
            "defect_label_review": ("suspected_missing_label_unconfirmed"
                                    if r["id"] == "DI2_Cam3_2026-07-18_10-11-55-177" else ""),
            "hand_label": int(hand[r["id"]]), "hand_label_source": "manual_labels_roi_intrusion.csv",
            "prior_confirmed_normal": prior_july[r["id"]],
            "causal_8_ref_ready": int(prior_july[r["id"]] >= 8),
        })
    for name, r in sorted(f2.items()):
        source = Path(r["path"])
        actual = digest(source)
        assert actual.lower() == r["sha256"].lower(), f"F2 source changed: {source}"
        event_id, position = f2_event[name]
        rows.append({
            "split": "locked_blind_holdout", "id": Path(name).stem, "source_path": str(source),
            "sha256": actual, "cam": f"cam{name.rsplit('-', 1)[-1][0]}",
            "captured_at": f2_time(name), "event_id": event_id,
            "run_state": "", "speed": "", "defect_label": "",
            "defect_boxes_json": "", "defect_label_source": "", "defect_label_review": "",
            "hand_label": "",
            "hand_label_source": "", "prior_confirmed_normal": "",
            "causal_8_ref_ready": 0,
        })
        assert position <= 2
    assert not july_hashes.intersection(r["sha256"] for r in rows if r["split"] == "locked_blind_holdout")
    assert len({r["sha256"] for r in rows}) == 138
    with OUT.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} verified unique frames: 69 development, 69 locked blind; "
          f"July positives ready={sum(r['causal_8_ref_ready'] for r in rows if r['split']=='development_single_fault' and r['defect_label']==1)}")


if __name__ == "__main__":
    main()
