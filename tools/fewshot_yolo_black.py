"""Six camera-held-out YOLOv8n fine-tuning runs on black-fabric bands.

Exploratory only: 16 positive frames all belong to one event. F2 is excluded.
"""

import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
import black_yarn_detector as D  # noqa: E402
import fewshot_semantic_heads as S  # noqa: E402
import fewshot_m1 as M1  # noqa: E402
from diagnose_m4_candidates import OUT, hit  # noqa: E402

WEIGHTS = ROOT.parent / "outputs/factory_red_focus_v2/runs/F5_yolov8s_detect_1280/weights/best.pt"
DATASET = OUT / "yolo_dataset"
RUNS = OUT / "yolo_runs"


def prepare(rows):
    crops = DATASET / "crops"
    crops.mkdir(parents=True, exist_ok=True)
    for r in rows:
        out = crops / (r["id"] + ".jpg")
        if out.exists():
            continue
        img = D.read_bgr(Path(r["file"]))
        if img is None or list(img.shape[:2]) != r["hw"]:
            raise RuntimeError(f"image changed: {r['id']}")
        if not cv2.imwrite(str(out), img[r["y0"]:r["y1"]]):
            raise RuntimeError(f"cannot write crop: {out}")


def make_fold(rows, fold, test_cam, val_cam):
    d = DATASET / f"fold{fold}"
    for split in ("train", "val"):
        (d / "images" / split).mkdir(parents=True, exist_ok=True)
        (d / "labels" / split).mkdir(parents=True, exist_ok=True)
    for r in rows:
        if r["cam"] == test_cam:
            continue
        split = "val" if r["cam"] == val_cam else "train"
        dst = d / "images" / split / (r["id"] + ".jpg")
        if not dst.exists():
            os.link(DATASET / "crops" / dst.name, dst)
        w = r["hw"][1]
        h = r["y1"] - r["y0"]
        labels = []
        for x0, y0, x1, y1 in r["gt"]:
            x0, x1 = np.clip([x0, x1], 0, w)
            y0, y1 = np.clip([y0-r["y0"], y1-r["y0"]], 0, h)
            if x1 > x0 and y1 > y0:
                labels.append(f"0 {(x0+x1)/(2*w):.8f} {(y0+y1)/(2*h):.8f} "
                              f"{(x1-x0)/w:.8f} {(y1-y0)/h:.8f}")
        (d / "labels" / split / (r["id"] + ".txt")).write_text(
            "\n".join(labels), encoding="ascii")
    yaml = d / "data.yaml"
    yaml.write_text(f"path: {d.as_posix()}\ntrain: images/train\nval: images/val\n"
                    "names:\n  0: black_yarn_defect\n", encoding="utf-8")
    return yaml


def main():
    rows = json.loads(S.INDEX.read_text(encoding="utf-8"))["data"]
    current = {r["id"]: r for r in M1.records()}
    if len(rows) != 69 or any(r["sha256"] != current[r["id"]]["sha256"] for r in rows):
        raise RuntimeError("black-fabric source changed since cached feature extraction")
    prepare(rows)
    cams = sorted({r["cam"] for r in rows})
    all_rows = []
    m0 = {r["id"]: r for r in json.loads(
        (OUT / "diagnostic.json").read_text(encoding="utf-8"))["candidate_rows"]}
    for fi, test_cam in enumerate(cams):
        fold = fi + 1
        val_cam = cams[(fi+1) % len(cams)]
        yaml = make_fold(rows, fold, test_cam, val_cam)
        model = YOLO(str(WEIGHTS))
        model.train(data=str(yaml), imgsz=1280, epochs=12, batch=4, rect=True, device=0,
                    workers=0, seed=20261024, deterministic=True, patience=12,
                    project=str(RUNS), name=f"fold{fold}_{test_cam}", exist_ok=True,
                    mosaic=0.0, mixup=0.0, fliplr=0.0, verbose=False, plots=False)
        best = RUNS / f"fold{fold}_{test_cam}" / "weights" / "best.pt"
        model = YOLO(str(best))
        for r in [r for r in rows if r["cam"] in (val_cam, test_cam)]:
            pred = model.predict(source=str(DATASET / "crops" / (r["id"] + ".jpg")),
                                 imgsz=1280, conf=.001, iou=.7, max_det=100,
                                 device=0, verbose=False)[0]
            dets = []
            for box, confidence in zip(pred.boxes.xyxy.cpu().numpy(),
                                       pred.boxes.conf.cpu().numpy()):
                x0, y0, x1, y1 = [float(v) for v in box]
                point = [(x0+x1)/2, (y0+y1)/2 + r["y0"]]
                dets.append({"confidence": float(confidence), "point_hit": hit(point, r["gt"]),
                             "box": [x0, y0+r["y0"], x1, y1+r["y0"]]})
            all_rows.append({"id": r["id"], "cam": r["cam"], "fold": fold,
                             "split": "test" if r["cam"] == test_cam else "val",
                             "defect": bool(r["gt"]), "m0_point_hit": m0[r["id"]]["m0_point_hit"],
                             "detections": dets})
        print(f"YOLO fine-tune fold {fold}/6 complete, test {test_cam}", flush=True)
    test = [r for r in all_rows if r["split"] == "test"]
    pos = [r for r in test if r["defect"]]
    summary = {"test_frames": len(test), "positive_frames": len(pos),
               "yolo_any_candidate_point_hits": sum(any(d["point_hit"] for d in r["detections"])
                                                   for r in pos),
               "yolo_top1_point_hits": sum(bool(r["detections"] and
                                                r["detections"][0]["point_hit"]) for r in pos),
               "m0_or_yolo_any_point_hits": sum(r["m0_point_hit"] or any(
                   d["point_hit"] for d in r["detections"]) for r in pos),
               "test_detections_total": sum(len(r["detections"]) for r in test),
               "protocol": "YOLOv8s initialized from red-congestion detection model; 6 held-out camera folds; "
                           "12 epochs; 1280px band crop; validation-camera model selection; "
                           "conf .001 proposal audit; one fault event"}
    (OUT / "yolo_black_results.json").write_text(json.dumps(
        {"summary": summary, "rows": all_rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
