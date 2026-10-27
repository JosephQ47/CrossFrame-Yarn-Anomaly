"""Read-only P0/P1/P2 black-yarn shadow replay of a validated time stream.

No blind defect labels are read. No PLC or production alarm action is called.
"""

import argparse
import json
import time
from collections import Counter, defaultdict, deque
from pathlib import Path

import numpy as np

from validate_black_stream_manifest import csv_rows, sha256, validate

ROOT = Path(__file__).resolve().parents[1]
BUFFER_SIZE = 8  # C# BlackYarnDetector.BufSize


def robust_threshold(scores):
    values = np.asarray(scores, dtype=np.float64)
    if len(values) != BUFFER_SIZE:
        raise ValueError("P0 LOO calibration requires exactly eight scores")
    median = np.median(values)
    mad = 1.4826 * np.median(np.abs(values - median))
    return float(median + 3.5 * max(float(mad), 1.0))


def p95(scores):
    return float(np.quantile(np.asarray(scores, dtype=np.float64), .95, method="linear"))


def result(state="NOT_READY", score=None, threshold=None, alarm=False, box=None):
    return {"state": state, "score": score, "threshold": threshold,
            "alarm": bool(alarm), "box": box}


def csharp_score_and_locate(anomaly, hw, y0, y1, cap=55):
    """Match C# component tie order and integer box conversion."""
    import cv2
    threshold = np.float32(np.percentile(anomaly, 99))
    count, _, stats, _ = cv2.connectedComponentsWithStats(
        (anomaly >= threshold).astype(np.uint8), connectivity=8)
    chosen, best_area = -1, 0
    for index in range(1, count):
        area = int(stats[index, cv2.CC_STAT_AREA])
        if area <= cap and area > best_area:
            chosen, best_area = index, area
    if chosen < 0:
        return 0.0, None
    x, y, width, height = (int(value) for value in stats[chosen, :4])
    grid_h, grid_w = anomaly.shape
    sx, sy = hw[1] / grid_w, (y1 - y0) / grid_h
    left = int(x * sx)
    top = int(y0 + y * sy)
    return float(best_area), [left, top, left + int(width * sx), top + int(height * sy)]


def replay(rows, feature_fn, score_fn, m4_fn, *, min_calibration=100):
    """Replay prevalidated rows. Callbacks allow deterministic feature tests."""
    states = {}
    current_group = {}
    output = []
    for row in rows:
        cam = row["camera"]
        group = (cam, row["session_id"], row["state_id"])
        if current_group.get(cam) != group:
            states[cam] = {"refs": deque(maxlen=BUFFER_SIZE), "p0_threshold": None,
                           "p1_cal": [], "p2_cal": [], "p1_threshold": None,
                           "p2_threshold": None, "blind_started": False}
            current_group[cam] = group
        st = states[cam]
        gate_state = row["gate_state"]
        item = {"frame_id": row["frame_id"], "sha256": row["sha256"],
                "camera": cam, "session_id": row["session_id"],
                "state_id": row["state_id"], "captured_at": row["captured_at"],
                "partition": row["partition"], "gate_state": gate_state,
                "reference_ids": [ref["id"] for ref in st["refs"]],
                "reference_count": len(st["refs"]), "admitted_to_pool": False,
                "p0": result(), "p1": result(), "p2": result(),
                "feature_ms": 0.0, "scoring_ms": 0.0,
                "threshold_source": {"p0": "8_reference_loo_robust_mad",
                                     "p1": "earlier_confirmed_normal_p95",
                                     "p2": "earlier_confirmed_normal_p95"}}
        if gate_state != "ELIGIBLE":
            for arm in ("p0", "p1", "p2"):
                item[arm] = result(state=gate_state)
            output.append(item)
            continue

        if row["partition"] == "blind_stream" and not st["blind_started"]:
            if len(st["p1_cal"]) < min_calibration:
                raise ValueError(f"P1 calibration incomplete: {group}")
            if m4_fn is not None and len(st["p2_cal"]) < min_calibration:
                raise ValueError(f"P2 calibration incomplete: {group}")
            st["p1_threshold"] = p95(st["p1_cal"])
            st["p2_threshold"] = p95(st["p2_cal"]) if m4_fn is not None else None
            st["blind_started"] = True

        started = time.perf_counter()
        feature, hw, y0, y1 = feature_fn(row)
        item["feature_ms"] = round((time.perf_counter() - started) * 1000, 3)
        feature = np.asarray(feature, dtype=np.float32)
        if feature.ndim != 3 or not np.isfinite(feature).all():
            raise ValueError(f"invalid feature: {row['frame_id']}")
        refs = list(st["refs"])
        if len(refs) == BUFFER_SIZE:
            scoring_started = time.perf_counter()
            score, box = score_fn(feature, [ref["feature"] for ref in refs], hw, y0, y1)
            if st["p0_threshold"] is None:
                loo = [score_fn(ref["feature"],
                                [other["feature"] for j, other in enumerate(refs) if j != i],
                                hw, y0, y1)[0] for i, ref in enumerate(refs)]
                st["p0_threshold"] = robust_threshold(loo)
                item["p0_loo_scores"] = [float(value) for value in loo]
            p0_alarm = score >= st["p0_threshold"]
            item["p0"] = result("YARN_DEFECT" if p0_alarm else "NORMAL",
                                float(score), st["p0_threshold"], p0_alarm, box)
            if row["partition"] == "normal_calibration":
                st["p1_cal"].append(float(score))
            if st["blind_started"]:
                alarm = score >= st["p1_threshold"]
                item["p1"] = result("YARN_DEFECT" if alarm else "NORMAL", float(score),
                                    st["p1_threshold"], alarm, box)
            elif row["partition"] == "normal_calibration":
                item["p1"] = result("NOT_READY", float(score))

            if m4_fn is not None:
                query = {"id": row["frame_id"], "cam": cam, "run_state": row["state_id"],
                         "time": row["time"], "feature": feature}
                m4_score, m4_box, chosen = m4_fn(query, refs, hw, y0, y1)
                item["m4_reference_patch_counts"] = chosen
                if row["partition"] == "normal_calibration":
                    st["p2_cal"].append(float(m4_score))
                if st["blind_started"]:
                    alarm = m4_score >= st["p2_threshold"]
                    item["p2"] = result("YARN_DEFECT" if alarm else "NORMAL",
                                        float(m4_score), st["p2_threshold"], alarm, m4_box)
                elif row["partition"] == "normal_calibration":
                    item["p2"] = result("NOT_READY", float(m4_score))
            item["scoring_ms"] = round((time.perf_counter() - scoring_started) * 1000, 3)

        if row["partition"] == "warmup":
            reason = "warmup_confirmed"
        elif row["partition"] == "normal_calibration":
            reason = "normal_calibration_confirmed"
        else:
            reason = "ordinary_nonhand"  # includes alarms, as in current C#
        st["refs"].append({"id": row["frame_id"], "cam": cam,
                           "run_state": row["state_id"], "time": row["time"],
                           "feature": feature, "admitted_by_policy": True,
                           "admission_reason": reason})
        item["admitted_to_pool"] = True
        output.append(item)

    counts = {arm: dict(Counter(item[arm]["state"] for item in output))
              for arm in ("p0", "p1", "p2")}
    latencies = np.asarray([item["feature_ms"] + item["scoring_ms"] for item in output])
    return output, {"frames": len(output), "state_counts": counts,
                    "blind_defect_labels_used": False,
                    "latency_ms_p50": float(np.quantile(latencies, .5)),
                    "latency_ms_p95": float(np.quantile(latencies, .95)),
                    "reference_buffer_size": BUFFER_SIZE}


class RealModels:
    def __init__(self, device):
        import sys
        import timm
        import torch
        sys.path.insert(0, str(ROOT / "src"))
        sys.path.insert(0, str(ROOT / "tools"))
        import black_yarn_detector as D
        import fewshot_semantic_heads as S
        import m4_causal_shadow as M4
        self.device, self.D, self.M4 = device, D, M4
        self.backbone = timm.create_model(D.MODEL, pretrained=True, num_classes=0,
                                          dynamic_img_size=True).eval().to(device)
        self.projection = np.load(S.OUT / "fixed_projection.npy")
        self.assets = {"projection_sha256": sha256(S.OUT / "fixed_projection.npy"),
                       "checkpoint_sha256": {}}
        self.heads = []
        for i in range(1, 7):
            checkpoint = S.OUT / f"M4_fold{i}_Cam{i}.pt"
            self.assets["checkpoint_sha256"][f"Cam{i}"] = sha256(checkpoint)
            saved = torch.load(checkpoint,
                               map_location="cpu", weights_only=False)
            model = S.Head("M4").eval().to(device)
            model.load_state_dict(saved["state_dict"])
            self.heads.append((model, saved["mean"], saved["std"]))

    def feature(self, row):
        image = self.D.read_bgr(row["path"])
        if image is None:
            raise ValueError(f"cannot read image: {row['frame_id']}")
        feat, y0, y1 = self.D.band_feat(self.backbone, self.device, image)
        return feat, image.shape[:2], y0, y1

    def m0(self, feature, references, hw, y0, y1):
        best = np.full(feature.shape[:2], -np.inf, np.float32)
        for ref in references:
            best = np.maximum(best, np.einsum("ijc,ijc->ij", ref, feature, optimize=True))
        area, box = csharp_score_and_locate(1.0 - best, hw, y0, y1)
        return float(area), box

    def m4(self, query, references, hw, y0, y1):
        x, chosen = self.M4.causal_m4_array(query, references, self.projection, self.device)
        probabilities = [self.M4.frozen_m4_probability(model, x, mean, std, self.device)
                         for model, mean, std in self.heads]
        ensemble = np.mean(np.stack(probabilities), axis=0)
        score = self.M4.image_score(ensemble)
        _, _, box = self.D.score_and_locate(ensemble, hw, y0, y1)
        return score, box, chosen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    args = parser.parse_args()
    rows, checked = validate(csv_rows(args.manifest))
    import torch
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; choose --device cpu explicitly")
    models = RealModels(args.device)
    output, summary = replay(rows, models.feature, models.m0, models.m4)
    summary["model_assets"] = models.assets
    summary["backbone_name"] = models.D.MODEL
    freeze_path = ROOT / "data/black_shadow_asset_freeze_2026-10-27.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    summary["frozen_backbone_sha256"] = freeze["backbone"]["sha256"]
    summary["asset_freeze_manifest_sha256"] = sha256(freeze_path)
    summary["shadow_source_sha256"] = {
        name: sha256(ROOT / "tools" / name)
        for name in ("replay_black_shadow.py", "validate_black_stream_manifest.py",
                     "m4_causal_shadow.py")}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "shadow_rows.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in output), encoding="utf-8")
    summary.update({"manifest_sha256": sha256(args.manifest), "device": args.device,
                    "manifest_groups": checked["groups"],
                    "scope": "read-only shadow predictions; no blind defect labels loaded"})
    (args.out / "shadow_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
