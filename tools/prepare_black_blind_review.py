"""Produce independent, prediction-free July image review sheets.

The file list and hashes come from the verified source index. Existing JSON
labels, model scores, heatmaps and boxes are deliberately absent from output.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from train_black_dual_head import source_rows  # noqa: E402

DEFAULT_OUT = ROOT / "data/black_july_blind_review_2026-10-31"
FIELDS = ["frame_id", "source_path", "sha256", "cam", "defect_status",
          "defect_type", "box_or_centerline_json", "run_state", "hand_or_occlusion",
          "event_id", "reviewer_id", "reviewed_at", "notes"]


def run(out):
    rows = source_rows()
    out.mkdir(parents=True, exist_ok=True)
    for reviewer in ("A", "B"):
        path = out / f"july_reviewer_{reviewer}.csv"
        if path.exists():
            raise FileExistsError(f"review sheet already exists; preserving human work: {path}")
        with path.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            for row in rows:
                writer.writerow({"frame_id": row["id"], "source_path": row["file"],
                                 "sha256": row["sha256"], "cam": row["cam"],
                                 "reviewer_id": reviewer})
    (out / "README.md").write_text(
        "# 七月原图双人盲审\n\n"
        "A、B 两人分别填写自己的 CSV，不共享判断。仅看 `source_path` 原图，不看旧 JSON、"
        "模型热图、分数或对方表格。`defect_status` 选 `normal/black_yarn/other_defect/uncertain`；"
        "无法判断必须写 uncertain。缺陷写最小框 `[x0,y0,x1,y1]` 或中心线 `[[x,y],...]`，"
        "标记运行状态、手/遮挡及事件 ID。完成后保存原 CSV，核对 SHA，再由第三人处理分歧。"
        "不得用模型输出给争议帧定真值。F2 使用既有独立盲判模板，不混入本批。\n",
        encoding="utf-8")
    return len(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    print(json.dumps({"frames": run(args.out), "out": str(args.out)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
