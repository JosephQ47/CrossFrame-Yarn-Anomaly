"""Cache top YOLO proposals for every camera in each archived YOLO fold.

Weights were trained on older July labels, not reviewer A. This cache only
provides candidate points for a retrospective A-label fusion diagnostic.
"""

import hashlib
import json
import sys
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
import black_yarn_detector as D  # noqa: E402
import train_black_reviewer_a as A  # noqa: E402

WEIGHTS = ROOT / "outputs/black_fusion_diagnostic/yolo_runs"
OUT = ROOT / "outputs/black_yarn_candidate_audit/yolo_fold_top1.json"


def main():
    rows, excluded, digest = A.reviewed_rows()
    cams = sorted({r["cam"] for r in rows})
    images = {}
    for row in rows:
        path = Path(row["file"])
        if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            raise RuntimeError(f"source image changed: {row['id']}")
        image = D.read_bgr(path)
        if image is None or list(image.shape[:2]) != row["hw"]:
            raise RuntimeError(f"invalid image: {row['id']}")
        images[row["id"]] = image[row["y0"]:row["y1"]]
    proposals, weight_hashes = {}, {}
    for index, test_cam in enumerate(cams):
        fold = index+1
        weights = WEIGHTS / f"fold{fold}_{test_cam}/weights/best.pt"
        if not weights.exists():
            raise RuntimeError(f"missing YOLO weights: {weights}")
        weight_hashes[str(fold)] = hashlib.sha256(weights.read_bytes()).hexdigest()
        model = YOLO(str(weights))
        for row in rows:
            result = model.predict(source=images[row["id"]], imgsz=1280,
                                   conf=.001, iou=.7, max_det=1,
                                   device=0, verbose=False)[0]
            proposal = None
            if len(result.boxes):
                box = result.boxes.xyxy[0].cpu().numpy().tolist()
                proposal = {"box": [float(box[0]), float(box[1]+row["y0"]),
                                    float(box[2]), float(box[3]+row["y0"])],
                            "confidence": float(result.boxes.conf[0].cpu())}
            proposals[f"{fold}:{row['id']}"] = proposal
        print(f"YOLO proposal cache fold {fold}/6 {test_cam}: {len(rows)} frames", flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"scope": "archived YOLO weights; all A eligible July frames per fold",
                               "A_label_sha256": digest, "excluded_uncertain": excluded,
                               "weight_sha256": weight_hashes, "proposals": proposals},
                              ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"saved {len(proposals)} proposals", flush=True)


if __name__ == "__main__":
    main()
