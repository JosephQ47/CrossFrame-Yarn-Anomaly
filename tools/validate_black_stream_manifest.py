"""Validate a prospective black-yarn stream before any model sees it.

Blind defect labels live in a separate file and are never read here.
"""

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hand_gate import HandIntrusionGate  # noqa: E402

REQUIRED = ("frame_id", "source_path", "sha256", "camera", "captured_at",
            "state_id", "speed_m_per_min", "session_id", "partition",
            "hand_model_result", "operator_confirmed_normal")
PARTITIONS = {"warmup": 0, "normal_calibration": 1, "blind_stream": 2}
EXISTING = ROOT / "data/black_next_stage_manifest_2026-10-27.csv"


def csv_rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError("manifest has missing or duplicate column names")
        if set(reader.fieldnames) != set(REQUIRED):
            raise ValueError(f"manifest columns must be exactly {REQUIRED}")
        rows = list(reader)
    if not rows:
        raise ValueError("manifest is empty")
    return rows


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def old_hashes():
    with EXISTING.open(encoding="utf-8-sig", newline="") as handle:
        return {row["sha256"].lower() for row in csv.DictReader(handle)}


def validate(rows, *, check_files=True, min_calibration=100, exclude_old=True):
    if min_calibration < 1:
        raise ValueError("min_calibration must be positive")
    forbidden = old_hashes() if exclude_old else set()
    ids, hashes = set(), set()
    parsed = []
    for line, raw in enumerate(rows, 2):
        if set(raw) != set(REQUIRED) or any(value is None for value in raw.values()):
            raise ValueError(f"line {line}: missing or extra fields")
        row = {key: raw[key].strip() for key in REQUIRED}
        if not row["frame_id"] or row["frame_id"] in ids:
            raise ValueError(f"line {line}: empty or duplicate frame_id")
        ids.add(row["frame_id"])
        digest = row["sha256"].lower()
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError(f"line {line}: invalid sha256")
        if digest in hashes or digest in forbidden:
            raise ValueError(f"line {line}: repeated or development/F2 source hash")
        hashes.add(digest)
        path = Path(row["source_path"])
        if not path.is_absolute() or path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            raise ValueError(f"line {line}: source must be an absolute image path")
        if check_files and (not path.is_file() or sha256(path) != digest):
            raise ValueError(f"line {line}: image missing or hash mismatch")
        if row["camera"] not in {f"Cam{i}" for i in range(1, 7)}:
            raise ValueError(f"line {line}: invalid camera")
        try:
            when = datetime.fromisoformat(row["captured_at"])
        except ValueError as exc:
            raise ValueError(f"line {line}: invalid timestamp") from exc
        if when.tzinfo is None or when.utcoffset() is None:
            raise ValueError(f"line {line}: timestamp requires timezone")
        if not row["state_id"] or not row["session_id"]:
            raise ValueError(f"line {line}: state_id/session_id required")
        if row["partition"] not in PARTITIONS:
            raise ValueError(f"line {line}: invalid partition")
        if row["hand_model_result"] not in {"0", "1"}:
            raise ValueError(f"line {line}: hand_model_result must be 0 or 1")
        if row["operator_confirmed_normal"] not in {"", "1"}:
            raise ValueError(f"line {line}: operator normal may be blank or 1 only")
        if row["partition"] == "blind_stream" and row["operator_confirmed_normal"]:
            raise ValueError(f"line {line}: blind truth leaked into manifest")
        if row["speed_m_per_min"]:
            try:
                speed = float(row["speed_m_per_min"])
                if not math.isfinite(speed) or speed < 0:
                    raise ValueError()
            except ValueError as exc:
                raise ValueError(f"line {line}: invalid speed") from exc
        parsed.append({**row, "time": when, "path": path,
                       "hand": row["hand_model_result"] == "1"})

    # The file itself is chronological. Reset the camera gate when session or
    # state changes; never resurrect an old state's reference pool.
    last_global = None
    last_camera = {}
    current_group = {}
    closed_groups = set()
    groups = defaultdict(list)
    for row in parsed:
        if last_global is not None and row["time"] < last_global:
            raise ValueError("manifest rows must be chronological")
        last_global = row["time"]
        cam = row["camera"]
        group = (cam, row["session_id"], row["state_id"])
        if cam in current_group and current_group[cam] != group:
            closed_groups.add(current_group[cam])
        if group in closed_groups:
            raise ValueError(f"group re-entered after state/session change: {group}")
        current_group[cam] = group
        if cam in last_camera and row["time"] <= last_camera[cam]:
            raise ValueError(f"camera time not strictly increasing: {cam}")
        last_camera[cam] = row["time"]
        groups[group].append(row)

    summaries = []
    for group, frames in sorted(groups.items()):
        gate = HandIntrusionGate(2)
        previous_partition = -1
        eligible = defaultdict(int)
        for row in frames:
            rank = PARTITIONS[row["partition"]]
            if rank < previous_partition:
                raise ValueError(f"partition moves backward: {group}")
            previous_partition = rank
            gate_state = gate.update(row["hand"])
            row["gate_state"] = gate_state.value if gate_state is not None else "ELIGIBLE"
            if rank < 2:
                if row["gate_state"] == "ELIGIBLE":
                    if row["operator_confirmed_normal"] != "1":
                        raise ValueError(f"calibration normal not confirmed: {row['frame_id']}")
                    eligible[row["partition"]] += 1
                elif row["operator_confirmed_normal"]:
                    raise ValueError(f"gated frame marked normal: {row['frame_id']}")
        if len(frames) >= 1 and eligible["warmup"] < 8:
            raise ValueError(f"fewer than 8 eligible warmup frames: {group}")
        if any(row["partition"] == "blind_stream" for row in frames):
            if eligible["normal_calibration"] < min_calibration:
                raise ValueError(f"insufficient normal calibration: {group}")
        summaries.append({"camera": group[0], "session_id": group[1], "state_id": group[2],
                          "warmup_normal": eligible["warmup"],
                          "calibration_normal": eligible["normal_calibration"],
                          "blind_frames": sum(row["partition"] == "blind_stream" for row in frames),
                          "hand_or_recovery": sum(row["gate_state"] != "ELIGIBLE" for row in frames)})
    return parsed, {"frames": len(parsed), "groups": summaries,
                    "manifest_has_no_blind_defect_labels": True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    rows = csv_rows(args.manifest)
    _, summary = validate(rows)
    summary["manifest_sha256"] = sha256(args.manifest)
    if args.out:
        args.out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
