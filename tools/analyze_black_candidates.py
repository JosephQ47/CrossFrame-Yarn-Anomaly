"""Check whether A boxes occur among held-out head heatmap components."""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import fewshot_semantic_heads as S  # noqa: E402
import train_black_reviewer_a as A  # noqa: E402

RUNS = {
    "mil": ROOT / "outputs/black_reviewer_a_supervised/results.json",
    "mil_hard": ROOT / "outputs/black_reviewer_a_hard/results.json",
    "mil_normalhard": ROOT / "outputs/black_reviewer_a_normalhard/results.json",
    "mil_rank": ROOT / "outputs/black_reviewer_a_rank/results.json",
}
OUT = ROOT / "outputs/black_reviewer_a_candidates.json"


def component_candidates(prob, row):
    threshold = np.percentile(prob, 99)
    mask = (prob >= threshold).astype(np.uint8)
    count, _, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    gh, gw = prob.shape
    h, w = row["hw"]
    sx, sy = w / gw, (row["y1"] - row["y0"]) / gh
    candidates = []
    for index in range(1, count):
        area = int(stats[index, cv2.CC_STAT_AREA])
        if area > S.D.AREA_CAP:
            continue
        px = int(centroids[index][0] * sx)
        py = int(row["y0"] + centroids[index][1] * sy)
        hit = any(x0 <= px <= x1 and y0 <= py <= y1 for x0, y0, x1, y1 in row["gt"])
        candidates.append({"component_index": index, "area": area,
                           "point": [px, py], "hit": hit})
    # score_and_locate uses max((area, component_index)): later index wins ties.
    candidates.sort(key=lambda x: (x["area"], x["component_index"]), reverse=True)
    return candidates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    rows, excluded, digest = A.reviewed_rows()
    by_id = {r["id"]: r for r in rows}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    result = {"label_sha256": digest, "excluded": excluded, "methods": {}}
    for method, path in RUNS.items():
        run = json.loads(path.read_text(encoding="utf-8"))
        bundle = run["methods"][method]
        output = []
        for fold in bundle["folds"]:
            checkpoint = torch.load(fold["checkpoint"], map_location=device,
                                    weights_only=False)
            model = S.Head("M4").to(device)
            model.load_state_dict(checkpoint["state_dict"])
            model.eval()
            with torch.no_grad():
                for frame_id in fold["test_ids"]:
                    row = by_id[frame_id]
                    if not row["gt"]:
                        continue
                    x, _ = A.load(row, checkpoint["mean"], checkpoint["std"])
                    prob = torch.sigmoid(model(x[None].to(device)))[0, 0].cpu().numpy()
                    candidates = component_candidates(prob, row)
                    hits = [i + 1 for i, c in enumerate(candidates) if c["hit"]]
                    output.append({"id": frame_id, "cam": row["cam"],
                                   "first_hit_rank": min(hits) if hits else None,
                                   "candidate_count": len(candidates),
                                   "top5": candidates[:5],
                                   "box_max_probability": float(prob[
                                       S.M1.box_grid(row["gt"], row["hw"], row["y0"],
                                                     row["y1"], *prob.shape).astype(bool)
                                   ].max())})
        result["methods"][method] = {"top1": sum(r["first_hit_rank"] == 1 for r in output),
                                      "top3": sum(r["first_hit_rank"] is not None and
                                                  r["first_hit_rank"] <= 3 for r in output),
                                      "top5": sum(r["first_hit_rank"] is not None and
                                                  r["first_hit_rank"] <= 5 for r in output),
                                      "any": sum(r["first_hit_rank"] is not None for r in output),
                                      "total": len(output), "rows": output}
        if result["methods"][method]["top1"] != bundle["metrics"]["point_hits"]:
            raise RuntimeError(f"component replay mismatch: {method}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
    print(json.dumps({m: {k: v[k] for k in ("top1", "top3", "top5", "any", "total")}
                      for m, v in result["methods"].items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
