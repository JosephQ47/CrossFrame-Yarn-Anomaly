"""Read-only audit: can local black/F2 frames evaluate an 8-frame causal normal pool?"""

import csv
import json
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import fewshot_m1 as M1  # noqa: E402

F2 = Path(r"D:\研究生课程\课题\色纱研究\F2标注\images")
F2_LABEL_TABLE = F2.parent / "待人工判读" / "判读表.csv"
OUT = ROOT / "data" / "causal_reference_readiness_2026-10-25.json"
REVIEW_CSV = ROOT / "data" / "f2_blind_review_69_template.csv"
MIN_REF = 8  # src/deploy_black_detector.py default DET_MIN_REF
EVENT_GAP_SECONDS = 10  # splits the documented 15:30:06 and 15:30:31 states
F2_NAME = re.compile(r"^(\d{4})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d)\.jpg$")


def f2_time_and_cam(name):
    m = F2_NAME.fullmatch(name)
    if not m:
        raise ValueError(f"unrecognized F2 frame: {name}")
    year, month, day, hour, minute, second, centisec, cam = map(int, m.groups())
    when = datetime(year, month, day, hour, minute, second) + timedelta(milliseconds=centisec*10)
    return when, f"cam{cam}"


def main():
    deploy_source = (ROOT / "src" / "deploy_black_detector.py").read_text(encoding="utf-8")
    configured_default = re.search(r'MIN_REF\s*=\s*int\(os\.environ\.get\("DET_MIN_REF",\s*(\d+)\)\)',
                                   deploy_source)
    if configured_default is None or int(configured_default.group(1)) != MIN_REF:
        raise RuntimeError("deployment MIN_REF default changed; update the readiness audit")
    july = M1.records()
    july_by_cam = {}
    for cam in sorted({r["cam"] for r in july}):
        ordered = sorted((r for r in july if r["cam"] == cam), key=lambda r: r["id"])
        prior_normal = prior_all = 0
        frames = []
        for r in ordered:
            frames.append({"id": r["id"], "defect": bool(r["gt"]),
                           "prior_confirmed_normal": prior_normal,
                           "prior_all": prior_all,
                           "causal_normal_pool_ready": prior_normal >= MIN_REF})
            prior_all += 1
            if not r["gt"]:
                prior_normal += 1
        july_by_cam[cam] = frames

    images = sorted(F2.glob("*.jpg"))
    annotation_files = list(F2.glob("*.json"))
    with F2_LABEL_TABLE.open(encoding="utf-8-sig", newline="") as f:
        label_rows = list(csv.DictReader(f))
    filled_labels = sum(bool(r["有无缺陷(1/0)"].strip()) for r in label_rows)
    image_names = {p.name for p in images}
    table_only = [r["文件"] for r in label_rows if r["文件"] not in image_names]
    f2_by_cam = {}
    for p in images:
        when, cam = f2_time_and_cam(p.name)
        f2_by_cam.setdefault(cam, []).append((when, p.name))
    f2_events = {}
    for cam, items in sorted(f2_by_cam.items()):
        events = []
        for when, name in sorted(items):
            if not events or (when - events[-1]["last_time"]).total_seconds() > EVENT_GAP_SECONDS:
                events.append({"first_time": when, "last_time": when, "files": [name]})
            else:
                events[-1]["last_time"] = when
                events[-1]["files"].append(name)
        f2_events[cam] = [{"event": i+1, "first": e["first_time"].isoformat(),
                           "last": e["last_time"].isoformat(),
                           "frames": len(e["files"]), "files": e["files"],
                           "max_prior_same_state": len(e["files"])-1,
                           "queries_ready": max(0, len(e["files"])-MIN_REF)}
                          for i, e in enumerate(events)]

    positives = [r for frames in july_by_cam.values() for r in frames if r["defect"]]
    normal = [r for frames in july_by_cam.values() for r in frames if not r["defect"]]
    summary = {"min_required_past_normal_refs": MIN_REF,
               "july_frames": len(july), "july_positive": len(positives),
               "july_normal": len(normal),
               "july_positive_ready": sum(r["causal_normal_pool_ready"] for r in positives),
               "july_max_prior_confirmed_normal_before_positive": max(
                   r["prior_confirmed_normal"] for r in positives),
               "july_positive_ready_with_any_past_frames": sum(
                   r["prior_all"] >= MIN_REF for r in positives),
               "f2_frames": len(images),
               "f2_cameras": len(f2_events),
               "f2_events_per_camera": {c: len(e) for c, e in f2_events.items()},
               "f2_max_prior_same_state": max(e["max_prior_same_state"]
                                              for events in f2_events.values() for e in events),
               "f2_queries_ready": sum(e["queries_ready"]
                                       for events in f2_events.values() for e in events),
               "f2_json_annotations": len(annotation_files),
               "f2_label_table_rows": len(label_rows),
               "f2_table_rows_without_deduplicated_image": table_only,
               "f2_original_table_filled_binary_labels": filled_labels,
               "scope": "feasibility audit only; no DINO inference, no F2 labels inferred"}
    assert len(july) == 69 and len(positives) == 16 and len(normal) == 53
    assert len(images) == 69 and len(f2_events) == 6
    assert len(label_rows) == 72 and len(table_only) == 3
    assert all(len(events) == 4 for events in f2_events.values())
    if not REVIEW_CSV.exists():
        with REVIEW_CSV.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(label_rows[0]))
            writer.writeheader()
            writer.writerows(r for r in label_rows if r["文件"] in image_names)
    OUT.write_text(json.dumps({"summary": summary, "july_by_cam": july_by_cam,
                               "f2_events": f2_events}, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
