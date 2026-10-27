"""Frozen M4 shadow input and archived-checkpoint parity checks.

The causal builder accepts only same-camera, same-state, earlier frames
admitted by the frozen online policy. It never reads post-hoc defect labels.
"""

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import fewshot_semantic_heads as S  # noqa: E402

ARCHIVE = ROOT / "data/black_fewshot_semantic_2026-10-25/M4_results.json"
PARITY_OUT = ROOT / "data/black_m4_archive_parity_2026-10-27.json"


def causal_m4_array(query, references, projection, device="cpu"):
    """Build the exact 193-channel training-format tensor from past references.

    Each item is a dict with id, cam, time (datetime), run_state, feature
    (unit-normalized HWC array); refs require admitted_by_policy plus an
    admission_reason assigned online, without later defect labels.
    The caller must supply at least eight eligible references for deployment.
    """
    if len(references) < 8:
        raise ValueError("M4 causal shadow needs at least 8 earlier normal references")
    if not query.get("run_state") or not isinstance(query.get("time"), datetime):
        raise ValueError("query needs a verified run_state and datetime")
    cur = np.asarray(query["feature"], dtype=np.float32)
    if cur.ndim != 3 or cur.shape[-1] != projection.shape[0]:
        raise ValueError("query feature/projection shape mismatch")
    for ref in references:
        if (ref["cam"] != query["cam"] or ref["run_state"] != query["run_state"]
                or not ref.get("admitted_by_policy")
                or ref.get("admission_reason") not in {
                    "warmup_confirmed", "normal_calibration_confirmed", "ordinary_nonhand"}
                or ref["time"] >= query["time"]):
            raise ValueError(f"inadmissible causal M4 reference: {ref['id']}")
        if np.shape(ref["feature"]) != cur.shape:
            raise ValueError(f"reference feature shape mismatch: {ref['id']}")

    # Streaming max avoids another multi-gigabyte stack of 6144-D references.
    best_sim = np.full(cur.shape[:2], -np.inf, np.float32)
    best = np.zeros(cur.shape[:2], np.int16)
    chosen = np.empty_like(cur)
    for i, ref in enumerate(references):
        feature = np.asarray(ref["feature"], dtype=np.float32)
        sim = np.einsum("ijc,ijc->ij", feature, cur, optimize=True)
        better = sim > best_sim
        best_sim[better] = sim[better]
        best[better] = i
        chosen[better] = feature[better]
    anomaly = 1.0 - best_sim
    proj = torch.as_tensor(projection, dtype=torch.float32, device=device)
    with torch.no_grad():
        current = torch.from_numpy(cur.reshape(-1, cur.shape[-1])).to(device)
        reference = torch.from_numpy(chosen.reshape(-1, cur.shape[-1])).to(device)
        h, w = cur.shape[:2]
        cur_p = (current @ proj).cpu().numpy().reshape(h, w, -1)
        ref_p = (reference @ proj).cpu().numpy().reshape(h, w, -1)
        diff_p = ((current - reference).abs() @ proj).cpu().numpy().reshape(h, w, -1)
    # The original M4 training cache stored every channel as float16.
    x = np.concatenate([cur_p.astype(np.float16).astype(np.float32),
                        ref_p.astype(np.float16).astype(np.float32),
                        diff_p.astype(np.float16).astype(np.float32),
                        anomaly.astype(np.float16).astype(np.float32)[..., None]], axis=-1)
    return x, {references[i]["id"]: int(np.count_nonzero(best == i))
               for i in range(len(references)) if np.any(best == i)}


def frozen_m4_probability(model, x, mean, std, device="cpu"):
    """Apply archived M4 normalization and head to a 193-channel HWC array."""
    if x.ndim != 3 or x.shape[-1] != 193 or mean.shape != (193,) or std.shape != (193,):
        raise ValueError("invalid frozen M4 input or normalization")
    scaled = np.clip((x - mean) / std, -5, 5).transpose(2, 0, 1).copy()
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(torch.from_numpy(scaled[None]).to(device)))[0, 0].cpu().numpy()


def image_score(prob):
    k = max(1, int(round(prob.size * 0.01)))
    return float(np.partition(prob.ravel(), -k)[-k:].mean())


def verify_archive():
    """Reproduce all archived M4 scores and boxes from frozen cache/weights."""
    torch.set_num_threads(2)
    rows = json.loads(ARCHIVE.read_text(encoding="utf-8"))["rows"]
    index = {r["id"]: r for r in json.loads(S.INDEX.read_text(encoding="utf-8"))["data"]}
    assert len(rows) == len(index) == 69
    checks = []
    for cam in sorted({r["cam"] for r in rows}):
        fold = int(cam[-1])
        checkpoint = S.OUT / f"M4_fold{fold}_{cam}.pt"
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        assert saved["method"] == "M4" and saved["test_cam"] == cam
        model = S.Head("M4").eval()
        model.load_state_dict(saved["state_dict"])
        current_ids = {r["id"] for r in rows if r["cam"] == cam}
        assert current_ids == set(saved["test_ids"])
        for row in (r for r in rows if r["cam"] == cam):
            record = index[row["id"]]
            raw, _ = S.array_for(record, "M4")
            prob = frozen_m4_probability(model, raw, saved["mean"], saved["std"])
            score = image_score(prob)
            _, point, box = S.D.score_and_locate(prob, record["hw"], record["y0"], record["y1"])
            point_hit = bool(point and any(
                b[0] <= point[0] <= b[2] and b[1] <= point[1] <= b[3]
                for b in record["gt"]))
            # Archived inference used CUDA; CPU convolutions can shift a mean
            # probability by ~1e-5 while leaving the selected region intact.
            if abs(score - row["score"]) > 2e-3 or box != row["pred_box"] or point_hit != row["point_hit"]:
                raise AssertionError(f"M4 archived parity failed: {row['id']} "
                                     f"score={score}/{row['score']} box={box}/{row['pred_box']} "
                                     f"hit={point_hit}/{row['point_hit']}")
            checks.append({"id": row["id"], "score_abs_error": abs(score - row["score"])})
    result = {"frames": len(checks), "max_score_abs_error": max(r["score_abs_error"] for r in checks),
              "boxes_and_point_hits_match": True,
              "scope": "archived retrospective cache/weights only; no causal performance claim"}
    PARITY_OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def smoke_causal():
    rng = np.random.default_rng(20261027)
    now = datetime(2026, 10, 27, 12)
    query = {"id": "query", "cam": "Cam1", "run_state": "steady",
             "time": now, "feature": rng.normal(size=(2, 3, 8)).astype(np.float32)}
    query["feature"] /= np.linalg.norm(query["feature"], axis=-1, keepdims=True)
    refs = []
    for i in range(8):
        f = rng.normal(size=(2, 3, 8)).astype(np.float32)
        f /= np.linalg.norm(f, axis=-1, keepdims=True)
        refs.append({"id": f"ref{i}", "cam": "Cam1", "run_state": "steady",
                     "time": now - timedelta(seconds=8-i), "admitted_by_policy": True,
                     "admission_reason": "warmup_confirmed", "feature": f})
    projection = rng.normal(size=(8, 2)).astype(np.float32)
    x, chosen = causal_m4_array(query, refs, projection)
    assert x.shape == (2, 3, 7) and sum(chosen.values()) == 6
    ordinary = [dict(ref) for ref in refs]
    ordinary[0]["admission_reason"] = "ordinary_nonhand"
    causal_m4_array(query, ordinary, projection)
    stacked = np.stack([ref["feature"] for ref in refs])
    sims = np.einsum("tijc,ijc->tij", stacked, query["feature"], optimize=True)
    best = np.argmax(sims, axis=0)
    ii, jj = np.indices(best.shape)
    selected = stacked[best, ii, jj]
    expected = np.concatenate([
        query["feature"] @ projection, selected @ projection,
        np.abs(query["feature"] - selected) @ projection,
        (1.0 - np.max(sims, axis=0))[..., None]], axis=-1)
    assert np.allclose(x, expected.astype(np.float16).astype(np.float32), atol=0)
    for field, value in (("time", now), ("cam", "Cam2"), ("run_state", "stopped"),
                         ("admitted_by_policy", False), ("admission_reason", "posthoc_label")):
        bad = [dict(ref) for ref in refs]
        bad[0][field] = value
        try:
            causal_m4_array(query, bad, np.ones((8, 2), np.float32))
        except ValueError:
            pass
        else:
            raise AssertionError(f"causal guard failed for {field}")
    try:
        causal_m4_array({**query, "run_state": ""}, refs, projection)
    except ValueError:
        pass
    else:
        raise AssertionError("unknown run_state must be rejected")
    return {"shape": list(x.shape), "selected_patch_count": sum(chosen.values()),
            "future_wrong_camera_wrong_state_and_posthoc_admission_rejected": True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["verify-archive", "smoke-causal"])
    args = parser.parse_args()
    print(json.dumps(verify_archive() if args.mode == "verify-archive" else smoke_causal(),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
