"""Read-only black-fabric candidate audit of an existing white-yarn YOLO model."""

import json
import sys
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from diagnose_m4_candidates import OUT, hit  # noqa: E402
import fewshot_semantic_heads as S  # noqa: E402

WEIGHTS = ROOT.parent / "outputs/color_aug_factory_v1/runs/A_white/weights/best.pt"


def main():
    rows = json.loads(S.INDEX.read_text(encoding="utf-8"))["data"]
    m0 = {r["id"]: r for r in json.loads(
        (OUT / "diagnostic.json").read_text(encoding="utf-8"))["candidate_rows"]}
    model = YOLO(str(WEIGHTS))
    assert model.names[0] == "detect"
    report = []
    for i, row in enumerate(rows):
        pred = model.predict(source=row["file"], imgsz=1280, conf=.001,
                             iou=.7, max_det=100, classes=[0], device=0,
                             verbose=False)[0]
        boxes = []
        for box, confidence in zip(pred.boxes.xyxy.cpu().numpy(),
                                   pred.boxes.conf.cpu().numpy()):
            x0, y0, x1, y1 = [float(v) for v in box]
            point = [(x0+x1)/2, (y0+y1)/2]
            boxes.append({"box": [x0, y0, x1, y1], "confidence": float(confidence),
                          "point_hit": hit(point, row["gt"])})
        report.append({"id": row["id"], "cam": row["cam"],
                       "defect": bool(row["gt"]), "detections": boxes,
                       "m0_hit": m0[row["id"]]["m0_point_hit"]})
        if (i+1) % 10 == 0:
            print(f"YOLO candidate audit {i+1}/{len(rows)}", flush=True)
    pos = [r for r in report if r["defect"]]
    summ = {"model": str(WEIGHTS), "pretrained_for": model.names,
            "conf": .001, "imgsz": 1280,
            "yolo_any_point_hits": sum(any(d["point_hit"] for d in r["detections"]) for r in pos),
            "yolo_top1_point_hits": sum(bool(r["detections"] and r["detections"][0]["point_hit"])
                                        for r in pos),
            "m0_or_yolo_any_point_hits": sum(r["m0_hit"] or any(
                d["point_hit"] for d in r["detections"]) for r in pos),
            "detections_total": sum(len(r["detections"]) for r in report),
            "frames_with_detections": sum(bool(r["detections"]) for r in report),
            "protocol": "read-only transfer candidate audit; no black-fabric YOLO training"}
    (OUT / "yolo_proposal_audit.json").write_text(json.dumps(
        {"summary": summ, "rows": report}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summ, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
