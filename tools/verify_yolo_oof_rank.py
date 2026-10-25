"""Audit nested camera splits and archived OOF ranking evidence."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import fewshot_semantic_heads as S  # noqa: E402
from fewshot_yolo_black import DATASET  # noqa: E402

EVIDENCE = ROOT / "data" / "black_fusion_2026-10-25" / "oof"


def read(name):
    return json.loads((EVIDENCE / name).read_text(encoding="utf-8"))


def main():
    rows = json.loads(S.INDEX.read_text(encoding="utf-8"))["data"]
    by_id = {r["id"]: r for r in rows}
    cams = sorted({r["cam"] for r in rows})
    assert len(rows) == len(by_id) == 69
    inner_last = read("yolo_inner_oof_last.json")
    inner_best = read("yolo_inner_oof_best.json")
    assert inner_last == inner_best, "inner best and last should coincide with validation off"
    assert len(inner_last) == 24
    for fi, test_cam in enumerate(cams):
        fold = fi + 1
        val_cam = cams[(fi+1) % len(cams)]
        outer_train = set(cams) - {test_cam, val_cam}
        for heldout in outer_train:
            key = f"{fold}:{heldout}"
            assert key in inner_last
            assert {by_id[image_id]["cam"] for image_id in inner_last[key]} == {heldout}
            train_dir = DATASET / f"inner_outer{fold}_hold{heldout}" / "images" / "train"
            train_cams = {by_id[p.stem]["cam"] for p in train_dir.glob("*.jpg")}
            assert train_cams == outer_train - {heldout}, (fold, heldout, train_cams)
        for choice in ("last", "best"):
            outer = read(f"yolo_outer_{choice}.json")
            assert len(outer) == 6
            assert {by_id[image_id]["cam"] for image_id in outer[str(fold)]} == {
                test_cam, val_cam}
            result = read(f"yolo_oof_rank_{choice}_results.json")
            assert len(result["rows"]) == 69
            assert len({r["id"] for r in result["rows"]}) == 69
            pos = [r for r in result["rows"] if r["defect"]]
            neg = [r for r in result["rows"] if not r["defect"]]
            assert len(pos) == 16 and len(neg) == 53
            s = result["summary"]["oof_yolo_dino_ranker"]
            assert s["detected"] == sum(r["alarm"] for r in pos)
            assert s["false_alarms"] == sum(r["alarm"] for r in neg)
            assert s["point_hits"] == sum(r["point_hit"] for r in pos)
            assert result["summary"]["candidate_point_coverage"] == sum(
                r["candidate_hit"] for r in pos)
    print("Nested 24-model camera isolation and six-fold result evidence verified")


if __name__ == "__main__":
    main()
