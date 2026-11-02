"""GT-centered frozen-DINO resolution diagnostic for two A-review misses.

This deliberately uses ground-truth box centers to ask whether local features
contain a signal. It is not a deployable candidate generator or accuracy test.
"""

import csv
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import timm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import black_yarn_detector as D  # noqa: E402

REVIEW = ROOT / "outputs/black_july_blind_review/july_reviewer_A_locked.csv"
RANK = ROOT / "outputs/black_reviewer_a_rank/results.json"
OUT = ROOT / "outputs/black_roi_resolution_diag/results.json"
TARGETS = (
    "DI2_Cam3_2026-07-18_10-14-15-409",
    "DI2_Cam4_2026-07-18_10-14-15-405",
)
CAM_NORMALS = {
    "Cam3": ("DI2_Cam3_2026-07-18_10-11-45-637",
             "DI2_Cam3_2026-07-18_10-11-40-175"),
    "Cam4": ("DI2_Cam4_2026-07-18_10-11-45-648",
             "DI2_Cam4_2026-07-18_10-11-40-179"),
}
SOURCE_CROP = 224
INPUT_SIZES = (224, 448)


def image(row):
    path = Path(row["source_path"])
    if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
        raise RuntimeError(f"image changed: {row['frame_id']}")
    result = D.read_bgr(path)
    if result is None:
        raise RuntimeError(f"cannot read image: {path}")
    return result


@torch.no_grad()
def features(model, img, center, size, device):
    x, y = center
    crop = cv2.getRectSubPix(img, (SOURCE_CROP, SOURCE_CROP), (float(x), float(y)))
    if size != SOURCE_CROP:
        crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_CUBIC)
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor = torch.from_numpy(((rgb - D.MEAN) / D.STD).transpose(2, 0, 1)[None]).to(device)
    layers = model.get_intermediate_layers(tensor, n=D.LAYERS, reshape=False, norm=True)
    grid = size // 14
    stacked = torch.cat([layer[:, -grid * grid:, :] for layer in layers], dim=-1)
    return F.normalize(stacked[0].reshape(grid, grid, -1), dim=-1).cpu().numpy()


def residual(current, reference):
    anomaly = 1 - np.sum(current * reference, axis=-1)
    grid = anomaly.shape[0]
    # Same 84×84 original-pixel central region for both input resolutions.
    width = round(84 / SOURCE_CROP * grid)
    lo = (grid - width) // 2
    central = anomaly[lo:lo + width, lo:lo + width]
    return {"center_mean": float(central.mean()),
            "center_p95": float(np.quantile(central, .95)),
            "grid": grid, "central_patches": int(central.size)}


def main():
    with REVIEW.open(encoding="utf-8-sig", newline="") as handle:
        rows = {r["frame_id"]: r for r in csv.DictReader(handle)}
    pred = {r["id"]: r for r in json.loads(RANK.read_text(encoding="utf-8"))
            ["methods"]["mil_rank"]["rows"]}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = timm.create_model(D.MODEL, pretrained=True, num_classes=0,
                              dynamic_img_size=True).eval().to(device)
    results = {"scope": "ground-truth-centered representational diagnostic only",
               "source_crop_pixels": SOURCE_CROP, "input_sizes": INPUT_SIZES,
               "model": D.MODEL, "layers": D.LAYERS, "device": device,
               "rows": []}
    for target_id in TARGETS:
        target = rows[target_id]
        if target["defect_status"] != "black_yarn":
            raise RuntimeError("target A label changed")
        cam = target["cam"]
        reference_id, control_id = CAM_NORMALS[cam]
        for frame_id in (reference_id, control_id):
            if rows[frame_id]["defect_status"] != "normal":
                raise RuntimeError("reference/control A label changed")
        box = json.loads(target["box_or_centerline_json"])[0]
        centers = {"target_box": [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2],
                   "wrong_prediction": pred[target_id]["point"]}
        imgs = {key: image(rows[key]) for key in (target_id, reference_id, control_id)}
        item = {"target_id": target_id, "reference_id": reference_id,
                "control_id": control_id, "centers": centers, "scales": {}}
        for size in INPUT_SIZES:
            measures = {}
            for locus, center in centers.items():
                ref = features(model, imgs[reference_id], center, size, device)
                target_feature = features(model, imgs[target_id], center, size, device)
                control_feature = features(model, imgs[control_id], center, size, device)
                measures[locus] = {"target_vs_normal": residual(target_feature, ref),
                                   "normal_vs_normal": residual(control_feature, ref)}
            item["scales"][str(size)] = measures
            print(f"{cam} {size}: target={measures['target_box']['target_vs_normal']['center_mean']:.4f} "
                  f"wrong={measures['wrong_prediction']['target_vs_normal']['center_mean']:.4f}",
                  flush=True)
        results["rows"].append(item)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(OUT, flush=True)


if __name__ == "__main__":
    main()
