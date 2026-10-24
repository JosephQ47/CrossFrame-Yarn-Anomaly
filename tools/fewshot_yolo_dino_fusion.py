"""Score YOLO + M0 candidates with an FC trained only on M0 ROI examples."""

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
from fewshot_roi_fc import features  # noqa: E402


def yolo_as_candidate(d, row):
    x0, y0, x1, y1 = d["box"]
    gh, gw = 32, 228
    h, w = row["hw"]
    sy = gh / (row["y1"] - row["y0"])
    sx = gw / w
    gx0 = max(0, min(gw-1, int(np.floor(x0*sx))))
    gx1 = max(gx0+1, min(gw, int(np.ceil(x1*sx))))
    gy0 = max(0, min(gh-1, int(np.floor((y0-row["y0"])*sy))))
    gy1 = max(gy0+1, min(gh, int(np.ceil((y1-row["y0"])*sy))))
    return {"grid_box": [gx0, gy0, gx1, gy1],
            "area": (gx1-gx0)*(gy1-gy0),
            "point": [(x0+x1)/2, (y0+y1)/2]}


def summarize(result):
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


def main():
    rows = {r["id"]: r for r in json.loads(S.INDEX.read_text(encoding="utf-8"))["data"]}
    m0 = {r["id"]: r for r in json.loads((OUT / "diagnostic.json").read_text(
        encoding="utf-8"))["candidate_rows"]}
    yd = json.loads((OUT / "yolo_black_results.json").read_text(encoding="utf-8"))["rows"]
    by_fold = {(r["fold"], r["id"]): r for r in yd}
    cams = sorted({r["cam"] for r in rows.values()})
    fusion_results, yolo_results = [], []
    for fi, test_cam in enumerate(cams):
        fold = fi + 1
        val_cam = cams[(fi+1) % len(cams)]
        train = [r for r in rows.values() if r["cam"] not in (test_cam, val_cam)]
        x, y = [], []
        for r in train:
            for c in m0[r["id"]]["candidates"][:3]:
                x.append(features(r, c))
                y.append(int(hit(c["point"], r["gt"])))
        scaler = StandardScaler().fit(np.stack(x))
        clf = LogisticRegression(C=.1, class_weight="balanced", max_iter=2000,
                                 random_state=20261024).fit(scaler.transform(np.stack(x)), y)

        def score(r):
            m = m0[r["id"]]["candidates"][:3]
            ydets = by_fold[(fold, r["id"])]["detections"][:20]
            yc = [yolo_as_candidate(d, r) for d in ydets]
            cs = m + yc
            if not cs:
                return 0., False
            xx = np.stack([features(r, c) for c in cs])
            probs = clf.predict_proba(scaler.transform(xx))[:, 1]
            j = int(probs.argmax())
            return float(probs[j]), hit(cs[j]["point"], r["gt"])

        val = [r for r in rows.values() if r["cam"] == val_cam]
        te = [r for r in rows.values() if r["cam"] == test_cam]
        threshold = float(np.quantile([score(r)[0] for r in val if not r["gt"]], .95))
        yolo_threshold = float(np.quantile([by_fold[(fold, r["id"])]["detections"][0]
                         ["confidence"] if by_fold[(fold, r["id"])]["detections"] else 0.
                         for r in val if not r["gt"]], .95))
        for r in te:
            p, h = score(r)
            fusion_results.append({"id": r["id"], "cam": test_cam,
                                   "defect": bool(r["gt"]), "score": p,
                                   "threshold": threshold, "alarm": p >= threshold,
                                   "point_hit": h})
            dets = by_fold[(fold, r["id"])]["detections"]
            ys = dets[0]["confidence"] if dets else 0.
            yh = dets[0]["point_hit"] if dets else False
            yolo_results.append({"id": r["id"], "cam": test_cam,
                                 "defect": bool(r["gt"]), "score": ys,
                                 "threshold": yolo_threshold,
                                 "alarm": ys >= yolo_threshold, "point_hit": yh})
        print(f"YOLO+DINO fold {fold}/6 complete, test {test_cam}", flush=True)
    report = {"summary": {"yolo_dino_fc": summarize(fusion_results),
                          "yolo_detector": summarize(yolo_results),
                          "protocol": "YOLO top20 + M0 top3 proposals; FC trained on M0 train-camera "
                                      "candidates only; DINO ROI feature means; val-camera normal "
                                      "P95 threshold; test camera held out; one fault event"},
              "fusion_rows": fusion_results, "yolo_rows": yolo_results}
    (OUT / "yolo_dino_fusion_results.json").write_text(json.dumps(
        report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
