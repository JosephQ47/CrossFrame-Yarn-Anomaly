"""Check archived few-shot results and sixfold split integrity."""

import csv
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
M1 = DATA / "black_fewshot_m1_2026-10-25"
SEM = DATA / "black_fewshot_semantic_2026-10-25"
BOX = DATA / "black_fewshot_box_2026-10-25"


def paths(method):
    if method == "M1":
        return M1, M1 / "results.json", M1 / "scores.csv", "m1_fold*.pt"
    if method == "M6":
        return BOX, BOX / "M6_results.json", BOX / "M6_scores.csv", "M6_fold*.pt"
    return SEM, SEM / f"{method}_results.json", SEM / f"{method}_scores.csv", f"{method}_fold*.pt"


def main():
    all_ids = None
    all_hashes = None
    for method in ["M1", "M2", "M3", "M4", "M5", "M6"]:
        folder, result_file, csv_file, pattern = paths(method)
        payload = json.loads(result_file.read_text(encoding="utf-8"))
        rows, summary = payload["rows"], payload["summary"]
        with csv_file.open(encoding="utf-8-sig", newline="") as f:
            csv_rows = list(csv.DictReader(f))
        assert len(rows) == len(csv_rows) == 69
        ids = {r["id"] for r in rows}
        assert len(ids) == 69 and ids == {r["id"] for r in csv_rows}
        if all_ids is None:
            all_ids = ids
        assert ids == all_ids
        labels = [int(r["defect"]) for r in rows]
        scores = [r["score"] for r in rows]
        assert sum(labels) == summary["positive_frames"] == 16
        assert sum(not y for y in labels) == 53
        assert abs(roc_auc_score(labels, scores) - summary["image_auroc"]) < 1e-10
        assert abs(average_precision_score(labels, scores) - summary["image_ap"]) < 1e-10
        pos = [r for r in rows if r["defect"]]
        neg = [r for r in rows if not r["defect"]]
        assert abs(np.mean([r["alarm"] for r in pos]) - summary["fixed_val_threshold_recall"]) < 1e-10
        assert abs(np.mean([r["alarm"] for r in neg]) - summary["fixed_val_threshold_fpr"]) < 1e-10
        assert sum(r["point_hit"] for r in pos) == summary["point_hits"]
        by_cam = {cam: {r["id"] for r in rows if r["cam"] == cam}
                  for cam in sorted({r["cam"] for r in rows})}
        assert len(by_cam) == 6
        ckpts = sorted(folder.glob(pattern))
        assert len(ckpts) == 6
        for checkpoint in ckpts:
            state = torch.load(checkpoint, map_location="cpu", weights_only=False)
            tr, va, te = [set(state[key]) for key in ("train_ids", "val_ids", "test_ids")]
            assert len(tr | va | te) == 69
            assert not (tr & va or tr & te or va & te)
            assert te == by_cam[state["test_cam"]]
            assert va == by_cam[state["val_cam"]]
            assert state["seed"] == 20261024 and state["epochs"] == 20
        print(f"{method}: 69 unique frames, 16 positives, 6 disjoint folds, metrics verified")
    m1_index = json.loads((M1 / "index.json").read_text(encoding="utf-8"))["data"]
    sem_index = json.loads((SEM / "index.json").read_text(encoding="utf-8"))["data"]
    a = {r["id"]: r["sha256"] for r in m1_index}
    b = {r["id"]: r["sha256"] for r in sem_index}
    assert a == b and len(a) == 69
    print("source SHA-256 manifests match across M1-M6")


if __name__ == "__main__":
    main()
