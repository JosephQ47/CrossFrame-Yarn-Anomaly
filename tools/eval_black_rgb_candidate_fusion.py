"""Frozen ImageNet CNN crop embedding + DINO candidate FC reranking.

All fitting (PCA, scalers, logistic FC) uses train cameras only. Validation
camera normals set the P95 alarm threshold. July is one retrospective fault
event and M4 cache references are noncausal; this is a development audit.
"""

import hashlib
import json
import sys
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from torchvision.models import wide_resnet50_2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
import black_yarn_detector as D  # noqa: E402
import eval_black_rank_candidate_fc as E  # noqa: E402
import fewshot_semantic_heads as S  # noqa: E402
import train_black_reviewer_a as A  # noqa: E402

WEIGHTS = Path.home() / ".cache/torch/hub/checkpoints/wide_resnet50_2-95faca4d.pth"
RANK = ROOT / "outputs/black_reviewer_a_rank/results.json"
YOLO_ALL = ROOT / "outputs/black_yarn_candidate_audit/yolo_fold_top1.json"
OUT = ROOT / "outputs/black_yarn_candidate_audit/rgb_fusion.json"
SEED, CROP, PCA_DIM = 20261103, 192, 32
MEAN = np.array([.485, .456, .406], np.float32)
STD = np.array([.229, .224, .225], np.float32)


@lru_cache(maxsize=16)
def original(image_id, path_string, expected_hash):
    path = Path(path_string)
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash:
        raise RuntimeError(f"source image changed: {image_id}")
    image = D.read_bgr(path)
    if image is None:
        raise RuntimeError(f"source image unreadable: {image_id}")
    return image


def crop_tensor(image, point):
    x, y = [int(round(v)) for v in point]
    half = CROP // 2
    pad = cv2.copyMakeBorder(image, half, half, half, half, cv2.BORDER_REFLECT_101)
    patch = pad[y:y+CROP, x:x+CROP]
    if patch.shape[:2] != (CROP, CROP):
        raise RuntimeError(f"invalid crop: {point}")
    patch = cv2.resize(patch, (224, 224), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
    return ((rgb-MEAN)/STD).transpose(2, 0, 1).copy()


def make_encoder(device):
    if not WEIGHTS.exists():
        raise RuntimeError(f"local pretrained CNN weights missing: {WEIGHTS}")
    model = wide_resnet50_2(weights=None)
    model.load_state_dict(torch.load(WEIGHTS, map_location="cpu", weights_only=True))
    model.fc = torch.nn.Identity()
    return model.eval().to(device)


def attach_rgb(items, rows_by_id, encoder, device, cache):
    pending = []

    def flush():
        if not pending:
            return
        keys, tensors = zip(*pending)
        with torch.no_grad():
            values = encoder(torch.from_numpy(np.stack(tensors)).to(device)).cpu().numpy()
        for key, value in zip(keys, values):
            cache[key] = value.astype(np.float32)
        pending.clear()

    for item in items:
        row = rows_by_id[item["id"]]
        image = original(item["id"], row["file"], row["sha256"])
        for peak in item["peaks"]:
            key = (item["id"], tuple(peak["patch"]))
            if key not in cache and not any(k == key for k, _ in pending):
                pending.append((key, crop_tensor(image, peak["point"])))
                if len(pending) >= 32:
                    flush()
    flush()
    for item in items:
        item["rgb"] = np.stack([cache[(item["id"], tuple(p["patch"]))]
                                for p in item["peaks"]])


def summarize(rows, method):
    pos = [r for r in rows if r["defect"]]
    neg = [r for r in rows if not r["defect"]]
    return {"alarm_and_point": sum(r[method+"_alarm"] and r[method+"_point_hit"]
                                   for r in pos),
            "point_hits": sum(r[method+"_point_hit"] for r in pos),
            "false_alarms": sum(r[method+"_alarm"] for r in neg),
            "missed_ids": [r["id"] for r in pos if not
                           (r[method+"_alarm"] and r[method+"_point_hit"])]}


def main():
    rows, excluded, digest = A.reviewed_rows()
    by_id = {r["id"]: r for r in rows}
    yolo_bundle = json.loads(YOLO_ALL.read_text(encoding="utf-8"))
    if yolo_bundle["A_label_sha256"] != digest:
        raise RuntimeError("YOLO proposal cache A label hash mismatch")
    yolo_all = yolo_bundle["proposals"]
    bundle = json.loads(RANK.read_text(encoding="utf-8"))["methods"]["mil_rank"]
    folds_by_cam = {f["test_cam"]: f for f in bundle["folds"]}
    cams = sorted(folds_by_cam)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    encoder = make_encoder(device)
    weight_hash = hashlib.sha256(WEIGHTS.read_bytes()).hexdigest()
    embedding_cache = {}
    predictions, fold_report = [], []
    for index, test_cam in enumerate(cams):
        fold_number = index+1
        val_cam = cams[(index+1) % len(cams)]
        fold = folds_by_cam[test_cam]
        path = Path(fold["checkpoint"])
        if hashlib.sha256(path.read_bytes()).hexdigest() != fold["checkpoint_sha256"]:
            raise RuntimeError(f"DINO checkpoint hash mismatch: {test_cam}")
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        train_rows = [r for r in rows if r["cam"] not in (test_cam, val_cam)]
        val_rows = [r for r in rows if r["cam"] == val_cam]
        test_rows = [r for r in rows if r["cam"] == test_cam]
        if (set(checkpoint["train_ids"]) != {r["id"] for r in train_rows} or
            set(checkpoint["val_ids"]) != {r["id"] for r in val_rows} or
            set(checkpoint["test_ids"]) != {r["id"] for r in test_rows}):
            raise RuntimeError(f"DINO fold split mismatch: {test_cam}")
        rank_model = S.Head("M4").to(device)
        rank_model.load_state_dict(checkpoint["state_dict"])
        rank_model.eval()

        def prepare_group(group):
            items = []
            for row in group:
                key = f"{fold_number}:{row['id']}"
                if key not in yolo_all:
                    raise RuntimeError(f"missing YOLO proposal: {key}")
                items.append(E.prepare(row, rank_model, checkpoint, device,
                                       yolo_all[key], source_score=True))
            attach_rgb(items, by_id, encoder, device, embedding_cache)
            return items

        train = prepare_group(train_rows)
        val = prepare_group(val_rows)
        test = prepare_group(test_rows)
        mx = np.concatenate([r["x"][r["usable"]] for r in train])
        rx = np.concatenate([r["rgb"][r["usable"]] for r in train])
        y = np.concatenate([r["hit"][r["usable"]] for r in train]).astype(int)
        if len(np.unique(y)) != 2:
            raise RuntimeError(f"candidate labels lack one class: {test_cam}")
        m_scaler = StandardScaler().fit(mx)
        pca = PCA(n_components=PCA_DIM, svd_solver="randomized", random_state=SEED).fit(rx)
        r_scaler = StandardScaler().fit(pca.transform(rx))

        def vectors(item, method):
            rgb = r_scaler.transform(pca.transform(item["rgb"]))
            if method == "rgb":
                return rgb
            return np.concatenate([m_scaler.transform(item["x"]), rgb], axis=1)

        for method in ("rgb", "fusion"):
            x_train = r_scaler.transform(pca.transform(rx)) if method == "rgb" else \
                np.concatenate([m_scaler.transform(mx), r_scaler.transform(pca.transform(rx))], axis=1)
            clf = LogisticRegression(C=.1, class_weight="balanced", max_iter=2000,
                                     random_state=SEED).fit(x_train, y)

            def score(item):
                values = clf.predict_proba(vectors(item, method))[:, 1]
                chosen = int(np.argmax(values))
                return float(values[chosen]), chosen

            threshold = float(np.quantile([score(r)[0] for r in val if not r["defect"]], .95))
            fold_report.append({"test_cam": test_cam, "val_cam": val_cam,
                                "method": method, "train_candidates": len(y),
                                "train_positive": int(y.sum()), "threshold": threshold})
            for item in test:
                value, chosen = score(item)
                record = next((r for r in predictions if r["id"] == item["id"]), None)
                if record is None:
                    record = {"id": item["id"], "cam": test_cam, "defect": item["defect"]}
                    predictions.append(record)
                record.update({method+"_score": value, method+"_threshold": threshold,
                               method+"_alarm": value >= threshold,
                               method+"_point_hit": bool(item["hit"][chosen]),
                               method+"_selected_rank": chosen+1})
        print(f"RGB fold {fold_number}/6 {test_cam}: train {len(y)}, positive {y.sum()}, "
              f"cached crops {len(embedding_cache)}", flush=True)
    report = {"scope": "retrospective July one-event; noncausal DINO cache",
              "A_label_sha256": digest, "excluded_uncertain": excluded,
              "rgb_encoder": "frozen ImageNet wide_resnet50_2, native 192px crop resized to 224",
              "rgb_weight_sha256": weight_hash, "pca_components": PCA_DIM,
              "candidate_rule": "DINO top10 NMS2 + YOLO top1, six camera folds",
              "selection": "P95 validation-camera normal score, train-only PCA/scalers/FC",
              "summary": {name: summarize(predictions, name) for name in ("rgb", "fusion")},
              "folds": fold_report, "rows": predictions}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
