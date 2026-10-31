"""Read-only diagnostics for the locked A-review head experiment."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import fewshot_semantic_heads as S  # noqa: E402
import train_black_reviewer_a as A  # noqa: E402

RUN = ROOT / "outputs/black_reviewer_a_supervised/results.json"
OUT = ROOT / "outputs/black_reviewer_a_supervised/diagnostic.json"
FRACTIONS = (0.001, 0.0025, 0.005, 0.01, 0.02)


def readout(logits, fraction):
    flat = torch.sigmoid(logits).flatten().cpu().numpy()
    k = max(1, round(len(flat) * fraction))
    return float(np.partition(flat, -k)[-k:].mean())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    rows, excluded, digest = A.reviewed_rows()
    by_id = {r["id"]: r for r in rows}
    base = json.loads(RUN.read_text(encoding="utf-8"))
    if base["protocol"]["label_sha256"] != digest:
        raise RuntimeError("results/locked review mismatch")
    device = base["protocol"]["device"]
    diagnostics = {"label_sha256": digest, "excluded": excluded,
                   "fractions": FRACTIONS, "methods": {}}
    for method, bundle in base["methods"].items():
        folds, all_test = [], []
        for fold in bundle["folds"]:
            checkpoint = torch.load(fold["checkpoint"], map_location=device,
                                    weights_only=False)
            model = S.Head("M4").to(device)
            model.load_state_dict(checkpoint["state_dict"])
            model.eval()
            records = []
            with torch.no_grad():
                for split in ("val", "test"):
                    for frame_id in fold[split + "_ids"]:
                        row = by_id[frame_id]
                        x, _ = A.load(row, checkpoint["mean"], checkpoint["std"])
                        logits = model(x[None].to(device))[0, 0]
                        scores = {str(f): readout(logits, f) for f in FRACTIONS}
                        prob = torch.sigmoid(logits).cpu().numpy()
                        area, _, _ = S.D.score_and_locate(prob, row["hw"], row["y0"], row["y1"])
                        records.append({"id": frame_id, "cam": row["cam"], "split": split,
                                        "defect": bool(row["gt"]), "scores": scores,
                                        "component_area": area})
            fold_results = {"test_cam": fold["test_cam"], "val_cam": fold["val_cam"],
                            "val_normals": sum(r["split"] == "val" and not r["defect"]
                                               for r in records), "thresholds": {}}
            for fraction in FRACTIONS:
                key = str(fraction)
                normal = [r["scores"][key] for r in records
                          if r["split"] == "val" and not r["defect"]]
                threshold = float(np.quantile(normal, .95))
                fold_results["thresholds"][key] = threshold
                for r in records:
                    if r["split"] == "test":
                        r.setdefault("alarms", {})[key] = r["scores"][key] >= threshold
            normal_areas = [r["component_area"] for r in records
                            if r["split"] == "val" and not r["defect"]]
            area_threshold = float(np.quantile(normal_areas, .95))
            fold_results["component_area_threshold"] = area_threshold
            for r in records:
                if r["split"] == "test":
                    r.setdefault("alarms", {})["component_area"] = r["component_area"] > area_threshold
            folds.append(fold_results)
            all_test.extend(r for r in records if r["split"] == "test")
        by_pred = {r["id"]: r for r in bundle["rows"]}
        for r in all_test:
            r["point_hit"] = by_pred[r["id"]]["point_hit"]
            if abs(r["scores"][str(.01)] - by_pred[r["id"]]["score"]) > 1e-5:
                raise RuntimeError(f"score replay mismatch: {method} {r['id']}")
        diagnostics["methods"][method] = {"folds": folds, "metrics": {
            str(f): {"alarm_hits": sum(r["alarms"][str(f)] for r in all_test if r["defect"]),
                     "alarm_and_point_hits": sum(r["alarms"][str(f)] and r["point_hit"]
                                                 for r in all_test if r["defect"]),
                     "false_alarms": sum(r["alarms"][str(f)] for r in all_test if not r["defect"])}
            for f in (*FRACTIONS, "component_area")}, "test_rows": all_test}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
    print(json.dumps({m: v["metrics"] for m, v in diagnostics["methods"].items()},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
