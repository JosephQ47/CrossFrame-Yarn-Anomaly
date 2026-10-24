"""One-layer FC verifier for frozen M0 proposals and cached DINOv2 features.

Six camera-held-out folds; validation-camera normals alone set alarm thresholds.
No new M0/DINO inference and no F2 training/evaluation.
"""

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import fewshot_semantic_heads as S  # noqa: E402
from diagnose_m4_candidates import OUT, hit  # noqa: E402


def features(row, cand):
    with np.load(S.CACHE / (row["id"] + ".npz")) as z:
        diff = z["diff"].astype(np.float32)
        cos = z["cos"].astype(np.float32)
    x0, y0, x1, y1 = cand["grid_box"]
    # Candidate plus a fixed two-patch halo for contrast with its background.
    h, w = cos.shape
    a = diff[y0:y1, x0:x1]
    b = diff[max(0, y0-2):min(h, y1+2), max(0, x0-2):min(w, x1+2)]
    c = cos[y0:y1, x0:x1]
    return np.concatenate([a.mean(axis=(0, 1)), b.mean(axis=(0, 1)),
                           np.array([float(c.mean()), float(c.max()),
                                     np.log1p(cand["area"]),
                                     (x0+x1)/(2*w), (y0+y1)/(2*h),
                                     (x1-x0)/w, (y1-y0)/h], np.float32)])


def main():
    data = json.loads((OUT / "diagnostic.json").read_text(encoding="utf-8"))
    rows = {r["id"]: r for r in json.loads(S.INDEX.read_text(encoding="utf-8"))["data"]}
    crows = data["candidate_rows"]
    cams = sorted({r["cam"] for r in crows})
    for r in crows:
        r["x"] = [features(rows[r["id"]], c) for c in r["candidates"][:3]]
        r["y"] = [int(hit(c["point"], rows[r["id"]]["gt"]))
                  for c in r["candidates"][:3]]
    results = []
    m0_results = []
    for fold, test_cam in enumerate(cams):
        val_cam = cams[(fold+1) % len(cams)]
        train = [r for r in crows if r["cam"] not in (test_cam, val_cam)]
        val = [r for r in crows if r["cam"] == val_cam]
        test = [r for r in crows if r["cam"] == test_cam]
        x = np.stack([v for r in train for v in r["x"]])
        y = np.array([v for r in train for v in r["y"]])
        scaler = StandardScaler().fit(x)
        clf = LogisticRegression(C=0.1, class_weight="balanced", max_iter=2000,
                                 random_state=20261024).fit(scaler.transform(x), y)

        def predict(r):
            if not r["x"]:
                return 0.0, None
            p = clf.predict_proba(scaler.transform(np.stack(r["x"])))[:, 1]
            i = int(np.argmax(p))
            return float(p[i]), i

        threshold = float(np.quantile([predict(r)[0] for r in val if not r["defect"]], .95))
        m0_threshold = float(np.quantile([r["m0_score"] for r in val if not r["defect"]], .95))
        for r in test:
            score, i = predict(r)
            selected_hit = bool(i is not None and r["y"][i])
            results.append({"id": r["id"], "cam": test_cam, "defect": r["defect"],
                            "score": score, "threshold": threshold,
                            "alarm": score >= threshold, "point_hit": selected_hit,
                            "selected_candidate": i})
            m0_results.append({"id": r["id"], "cam": test_cam, "defect": r["defect"],
                               "score": r["m0_score"], "threshold": m0_threshold,
                               "alarm": r["m0_score"] >= m0_threshold,
                               "point_hit": r["m0_point_hit"]})
        print(f"ROI-FC fold {fold+1}: {test_cam}, candidates {len(y)}, positive {y.sum()}",
              flush=True)

    def summary(result):
        pos = [r for r in result if r["defect"]]
        neg = [r for r in result if not r["defect"]]
        return {"auroc": float(roc_auc_score([r["defect"] for r in result],
                                           [r["score"] for r in result])),
                "ap": float(average_precision_score([r["defect"] for r in result],
                                               [r["score"] for r in result])),
                "detected": sum(r["alarm"] for r in pos),
                "false_alarms": sum(r["alarm"] for r in neg),
                "point_hits": sum(r["point_hit"] for r in pos),
                "alarm_and_point_hits": sum(r["alarm"] and r["point_hit"] for r in pos)}

    report = {"summary": {"roi_fc": summary(results),
                          "m0_same_split_threshold": summary(m0_results),
                          "candidate_ceiling_point_hits": 15,
                          "protocol": "top3 M0 P99 cap55 candidates; DINO diff ROI means; "
                                      "fixed L2 logistic one-layer FC; 6 camera held-out folds; "
                                      "validation-camera normal P95 threshold; one fault event"},
              "roi_fc_rows": results, "m0_rows": m0_results}
    (OUT / "roi_fc_results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                              encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
