"""Replay frozen M4 maps with the M0 rule and audit M0 ROI proposal recall.

No model is trained here. All thresholds use only the validation camera.
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import fewshot_semantic_heads as S  # noqa: E402
import fewshot_m1 as M1  # noqa: E402
import black_yarn_detector as D  # noqa: E402

OUT = ROOT / "outputs" / "black_fusion_diagnostic"


def hit(point, gt):
    return point is not None and any(
        x0 <= point[0] <= x1 and y0 <= point[1] <= y1
        for x0, y0, x1, y1 in gt)


def proposals(anom, row, limit=5):
    """Match M0's P99, 8-connectivity and area cap; rank by area."""
    bw = (anom >= np.percentile(anom, 99)).astype(np.uint8)
    n, lab, stats, cent = cv2.connectedComponentsWithStats(bw, connectivity=8)
    gh, gw = anom.shape
    h, w = row["hw"]
    sx, sy = w / gw, (row["y1"] - row["y0"]) / gh
    cands = [(int(stats[i, cv2.CC_STAT_AREA]), i) for i in range(1, n)
             if stats[i, cv2.CC_STAT_AREA] <= D.AREA_CAP]
    out = []
    for area, i in sorted(cands, reverse=True)[:limit]:
        x, y, ww, hh, _ = stats[i]
        out.append({"area": area, "point": [int(cent[i][0] * sx),
                                              int(row["y0"] + cent[i][1] * sy)],
                    "box": [int(x * sx), int(row["y0"] + y * sy),
                            int((x + ww) * sx), int(row["y0"] + (y + hh) * sy)],
                    "grid_box": [int(x), int(y), int(x + ww), int(y + hh)]})
    return out


def summarize(rows, score_key="score"):
    pos = [r for r in rows if r["defect"]]
    neg = [r for r in rows if not r["defect"]]
    return {"auroc": float(roc_auc_score([r["defect"] for r in rows],
                                            [r[score_key] for r in rows])),
            "ap": float(average_precision_score([r["defect"] for r in rows],
                                                [r[score_key] for r in rows])),
            "detected": sum(r["alarm"] for r in pos),
            "false_alarms": sum(r["alarm"] for r in neg),
            "point_hits": sum(r["point_hit"] for r in pos),
            "alarm_and_point_hits": sum(r["alarm"] and r["point_hit"] for r in pos)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = json.loads(S.INDEX.read_text(encoding="utf-8"))["data"]
    current = {r["id"]: r for r in M1.records()}
    if len(rows) != 69 or any(r["sha256"] != current[r["id"]]["sha256"] for r in rows):
        raise RuntimeError("black-fabric source changed since cached feature extraction")
    source = json.loads((ROOT / "data" / "black_fewshot_semantic_2026-10-25" /
                         "M4_results.json").read_text(encoding="utf-8"))
    old = {r["id"]: r for r in source["rows"]}
    cams = sorted({r["cam"] for r in rows})
    result = []
    replay_delta = []
    for fold, test_cam in enumerate(cams):
        val_cam = cams[(fold + 1) % len(cams)]
        checkpoint = torch.load(S.OUT / f"M4_fold{fold+1}_{test_cam}.pt",
                                map_location="cpu", weights_only=False)
        assert checkpoint["test_cam"] == test_cam and checkpoint["val_cam"] == val_cam
        model = S.Head("M4").eval()
        model.load_state_dict(checkpoint["state_dict"])

        def score_set(items):
            scored = []
            with torch.no_grad():
                for row in items:
                    x, _ = S.load_row(row, "M4", checkpoint["mean"], checkpoint["std"])
                    prob = torch.sigmoid(model(x[None]))[0, 0].numpy()
                    k = max(1, int(round(prob.size * .01)))
                    top_mean = float(np.partition(prob.ravel(), -k)[-k:].mean())
                    area, point, box = D.score_and_locate(
                        prob, row["hw"], row["y0"], row["y1"])
                    scored.append({"id": row["id"], "cam": row["cam"],
                                   "defect": bool(row["gt"]), "score": area,
                                   "top_mean": top_mean, "point": point,
                                   "pred_box": box, "point_hit": hit(point, row["gt"])})
            return scored

        val = score_set([r for r in rows if r["cam"] == val_cam])
        test = score_set([r for r in rows if r["cam"] == test_cam])
        normal_val = [r["score"] for r in val if not r["defect"]]
        threshold = float(np.quantile(normal_val, .95))
        for r in test:
            replay_delta.append(abs(r["top_mean"] - old[r["id"]]["score"]))
            r.update({"fold": fold + 1, "val_cam": val_cam, "threshold": threshold,
                      "alarm": r["score"] >= threshold})
        result.extend(test)
        print(f"M4 replay fold {fold+1}: {test_cam}, val P95 area {threshold:.2f}", flush=True)
    if max(replay_delta) > 2e-4:
        raise RuntimeError(f"M4 checkpoint replay differs: {max(replay_delta)}")

    coverage = []
    for row in rows:
        with np.load(M1.CACHE / (row["id"] + ".npz")) as z:
            anom = z["anom"].astype(np.float32)
        cand = proposals(anom, row)
        score, point, _ = D.score_and_locate(anom, row["hw"], row["y0"], row["y1"])
        assert score == (cand[0]["area"] if cand else 0)
        coverage.append({"id": row["id"], "cam": row["cam"],
                         "defect": bool(row["gt"]), "m0_score": score,
                         "m0_point_hit": hit(point, row["gt"]),
                         "candidate_hits": [hit(c["point"], row["gt"]) for c in cand],
                         "candidates": cand})
    positives = [r for r in coverage if r["defect"]]
    cover = {f"top{k}": sum(any(r["candidate_hits"][:k]) for r in positives)
             for k in (1, 3, 5)}
    summary = {"m4_p99_component": summarize(result),
               "m4_original": source["summary"],
               "max_m4_top_mean_replay_delta": max(replay_delta),
               "m0_candidate_point_coverage": cover,
               "positive_frames": len(positives), "all_frames": len(rows),
               "data_scope": "69 frames, one known defect event; exploratory only"}
    (OUT / "diagnostic.json").write_text(json.dumps(
        {"summary": summary, "m4_rows": result, "candidate_rows": coverage},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
