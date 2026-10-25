"""Nested camera-held-out YOLO proposals for a leakage-controlled DINO ROI ranker.

Outer fold: one test camera, one calibration camera, four training cameras.
For each training camera, an inner YOLO trained on the other three generates its
candidate boxes. The ranker never sees YOLO predictions made on YOLO train images.
The default uses the fixed final epoch. The exploratory ``--checkpoint-choice best``
uses validation-camera selection for the outer model. With inner validation disabled,
inner best.pt and last.pt are identical; no inner holdout labels select weights.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import fewshot_semantic_heads as S  # noqa: E402
import fewshot_m1 as M1  # noqa: E402
from diagnose_m4_candidates import OUT, hit  # noqa: E402
from fewshot_roi_fc import features  # noqa: E402
from fewshot_yolo_dino_fusion import yolo_as_candidate  # noqa: E402
from fewshot_yolo_black import WEIGHTS, DATASET, RUNS, prepare  # noqa: E402

OOF_RUNS = OUT / "yolo_inner_runs"
def make_inner(rows, fold, heldout, train_cams):
    d = DATASET / f"inner_outer{fold}_hold{heldout}"
    images = d / "images" / "train"
    labels = d / "labels" / "train"
    images.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)
    for r in rows:
        if r["cam"] not in train_cams:
            continue
        dst = images / (r["id"] + ".jpg")
        if not dst.exists():
            os.link(DATASET / "crops" / dst.name, dst)
        w = r["hw"][1]
        h = r["y1"] - r["y0"]
        lines = []
        for x0, y0, x1, y1 in r["gt"]:
            x0, x1 = np.clip([x0, x1], 0, w)
            y0, y1 = np.clip([y0-r["y0"], y1-r["y0"]], 0, h)
            if x1 > x0 and y1 > y0:
                lines.append(f"0 {(x0+x1)/(2*w):.8f} {(y0+y1)/(2*h):.8f} "
                             f"{(x1-x0)/w:.8f} {(y1-y0)/h:.8f}")
        (labels / (r["id"] + ".txt")).write_text("\n".join(lines), encoding="ascii")
    yaml = d / "data.yaml"
    yaml.write_text(f"path: {d.as_posix()}\ntrain: images/train\nval: images/train\n"
                    "names:\n  0: black_yarn_defect\n", encoding="utf-8")
    return yaml


def predict(model, r):
    pred = model.predict(source=str(DATASET / "crops" / (r["id"] + ".jpg")),
                         imgsz=1280, conf=.001, iou=.7, max_det=20, device=0,
                         verbose=False)[0]
    dets = []
    for box, conf in zip(pred.boxes.xyxy.cpu().numpy(), pred.boxes.conf.cpu().numpy()):
        x0, y0, x1, y1 = [float(v) for v in box]
        dets.append({"confidence": float(conf),
                     "box": [x0, y0+r["y0"], x1, y1+r["y0"]]})
    return dets


def generate_predictions(rows, cams, choice):
    oof_file = OUT / f"yolo_inner_oof_{choice}.json"
    outer_file = OUT / f"yolo_outer_{choice}.json"
    prepare(rows)
    current = {r["id"]: r for r in M1.records()}
    if len(rows) != 69 or any(r["sha256"] != current[r["id"]]["sha256"] for r in rows):
        raise RuntimeError("black-fabric source changed")
    inner = json.loads(oof_file.read_text(encoding="utf-8")) if oof_file.exists() else {}
    outer = json.loads(outer_file.read_text(encoding="utf-8")) if outer_file.exists() else {}
    for fi, test_cam in enumerate(cams):
        fold = fi + 1
        val_cam = cams[(fi+1) % len(cams)]
        outer_train = [c for c in cams if c not in (test_cam, val_cam)]
        for heldout in outer_train:
            key = f"{fold}:{heldout}"
            if key in inner:
                continue
            yaml = make_inner(rows, fold, heldout, [c for c in outer_train if c != heldout])
            run_name = f"outer{fold}_hold{heldout}"
            checkpoint = OOF_RUNS / run_name / "weights" / f"{choice}.pt"
            if not checkpoint.exists():
                model = YOLO(str(WEIGHTS))
                model.train(data=str(yaml), imgsz=1280, epochs=12, batch=4,
                            rect=True, device=0, workers=0, seed=20261024,
                            deterministic=True, val=False, patience=12,
                            project=str(OOF_RUNS), name=run_name, exist_ok=True,
                            mosaic=0.0, mixup=0.0, fliplr=0.0,
                            verbose=False, plots=False)
            model = YOLO(str(checkpoint))
            inner[key] = {r["id"]: predict(model, r) for r in rows if r["cam"] == heldout}
            oof_file.write_text(json.dumps(inner, ensure_ascii=False), encoding="utf-8")
            print(f"inner OOF complete: outer {fold}/6, hold {heldout} ({len(inner)}/24)",
                  flush=True)
        if str(fold) not in outer:
            checkpoint = RUNS / f"fold{fold}_{test_cam}" / "weights" / f"{choice}.pt"
            if not checkpoint.exists():
                raise RuntimeError(f"missing outer YOLO checkpoint: {checkpoint}")
            model = YOLO(str(checkpoint))
            outer[str(fold)] = {r["id"]: predict(model, r) for r in rows
                                if r["cam"] in (test_cam, val_cam)}
            outer_file.write_text(json.dumps(outer, ensure_ascii=False), encoding="utf-8")
            print(f"outer {choice} proposals complete: {fold}/6", flush=True)
    return inner, outer


def candidate_features(r, m0_row, detections):
    cs = []
    for c in m0_row["candidates"][:3]:
        cs.append((c, 0., 0.))
    for d in detections[:20]:
        cs.append((yolo_as_candidate(d, r), 1., float(d["confidence"])))
    if not cs:
        return [], []
    xx = [np.concatenate((features(r, c),
                          np.array([source, np.log10(conf+1e-4)], np.float32)))
          for c, source, conf in cs]
    yy = [int(hit(c["point"], r["gt"])) for c, _, _ in cs]
    return xx, yy


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-choice", choices=["last", "best"], default="last")
    choice = parser.parse_args().checkpoint_choice
    rows_list = json.loads(S.INDEX.read_text(encoding="utf-8"))["data"]
    rows = {r["id"]: r for r in rows_list}
    cams = sorted({r["cam"] for r in rows_list})
    m0 = {r["id"]: r for r in json.loads((OUT / "diagnostic.json").read_text(
        encoding="utf-8"))["candidate_rows"]}
    inner, outer = generate_predictions(rows_list, cams, choice)
    results = []
    for fi, test_cam in enumerate(cams):
        fold = fi + 1
        val_cam = cams[(fi+1) % len(cams)]
        outer_train = [c for c in cams if c not in (test_cam, val_cam)]
        xx, yy = [], []
        for cam in outer_train:
            for r in rows_list:
                if r["cam"] != cam:
                    continue
                x, y = candidate_features(r, m0[r["id"]], inner[f"{fold}:{cam}"][r["id"]])
                xx.extend(x)
                yy.extend(y)
        scaler = StandardScaler().fit(np.stack(xx))
        clf = LogisticRegression(C=.1, class_weight="balanced", max_iter=2000,
                                 random_state=20261024).fit(scaler.transform(np.stack(xx)), yy)

        def score(r):
            x, y = candidate_features(r, m0[r["id"]], outer[str(fold)][r["id"]])
            if not x:
                return 0., False, None, None, False
            p = clf.predict_proba(scaler.transform(np.stack(x)))[:, 1]
            j = int(np.argmax(p))
            rank_order = np.argsort(-p)
            true_ranks = [rank for rank, k in enumerate(rank_order, 1) if y[k]]
            return float(p[j]), bool(y[j]), j, min(true_ranks) if true_ranks else None, bool(true_ranks)

        val_normal = [score(r)[0] for r in rows_list
                      if r["cam"] == val_cam and not r["gt"]]
        threshold = float(np.quantile(val_normal, .95))
        for r in rows_list:
            if r["cam"] != test_cam:
                continue
            p, h, j, true_rank, candidate_hit = score(r)
            results.append({"id": r["id"], "cam": test_cam, "defect": bool(r["gt"]),
                            "score": p, "threshold": threshold, "alarm": p >= threshold,
                            "point_hit": h, "selected_candidate": j,
                            "best_true_candidate_rank": true_rank,
                            "candidate_hit": candidate_hit,
                            "candidate_count": min(3, len(m0[r["id"]]["candidates"])) +
                                               len(outer[str(fold)][r["id"]])})
        print(f"OOF ranker outer fold {fold}/6 complete: {test_cam}, "
              f"{len(yy)} train candidates, {sum(yy)} positives", flush=True)
    report = {"summary": {"oof_yolo_dino_ranker": summary(results),
                          "candidate_point_coverage": sum(r["candidate_hit"] for r in results
                                                          if r["defect"]),
                          "protocol": "outer 6-fold test camera; inner 24 YOLO models held "
                                      "each ranker-train camera out; 12 epochs, " + choice + ".pt; "
                                      "YOLO top20 + M0 top3; L2 FC C=.1; calibration camera "
                                      "normal P95; one positive fault event"},
              "rows": results}
    (OUT / f"yolo_oof_rank_{choice}_results.json").write_text(json.dumps(
        report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
