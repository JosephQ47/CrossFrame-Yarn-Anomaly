"""Linear candidate head with within-image positive-versus-negative ranking.

Six held-out-camera folds on the retrospective July single fault event. No
hand labels. DINO cache is noncausal; this cannot establish field accuracy.
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import eval_black_rank_candidate_fc as E  # noqa: E402
import fewshot_semantic_heads as S  # noqa: E402
import train_black_reviewer_a as A  # noqa: E402

RANK = ROOT / "outputs/black_reviewer_a_rank/results.json"
YOLO_ALL = ROOT / "outputs/black_yarn_candidate_audit/yolo_fold_top1.json"
OUT = ROOT / "outputs/black_yarn_candidate_audit/pairwise_rank.json"
SEED, EPOCHS = 20261103, 60


def ranking_loss(model, items):
    losses = []
    for item in items:
        scores = model(item["tensor"]).flatten()
        positives = item["positive"]
        negatives = item["negative"]
        if positives.any():
            positive = torch.logsumexp(scores[positives], dim=0) - np.log(int(positives.sum()))
            negative = torch.logsumexp(scores[negatives], dim=0) if negatives.any() else None
            loss = .5 * F.softplus(-positive)
            if negative is not None:
                loss = loss + F.softplus(negative-positive)
        else:
            loss = F.softplus(scores.max())
        losses.append(loss)
    return torch.stack(losses).mean()


def main():
    rows, excluded, digest = A.reviewed_rows()
    archive = json.loads(RANK.read_text(encoding="utf-8"))["methods"]["mil_rank"]
    by_cam = {f["test_cam"]: f for f in archive["folds"]}
    yolo_bundle = json.loads(YOLO_ALL.read_text(encoding="utf-8"))
    if yolo_bundle["A_label_sha256"] != digest:
        raise RuntimeError("YOLO candidate cache label hash mismatch")
    yolo = yolo_bundle["proposals"]
    cams = sorted(by_cam)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    predictions, folds = [], []
    for index, test_cam in enumerate(cams):
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        val_cam = cams[(index+1) % len(cams)]
        fold = by_cam[test_cam]
        path = Path(fold["checkpoint"])
        if hashlib.sha256(path.read_bytes()).hexdigest() != fold["checkpoint_sha256"]:
            raise RuntimeError(f"rank checkpoint hash mismatch: {test_cam}")
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        train_rows = [r for r in rows if r["cam"] not in (test_cam, val_cam)]
        val_rows = [r for r in rows if r["cam"] == val_cam]
        test_rows = [r for r in rows if r["cam"] == test_cam]
        for key, group in (("train_ids", train_rows), ("val_ids", val_rows), ("test_ids", test_rows)):
            if set(checkpoint[key]) != {r["id"] for r in group}:
                raise RuntimeError(f"fold split mismatch: {test_cam}/{key}")
        rank_model = S.Head("M4").to(device)
        rank_model.load_state_dict(checkpoint["state_dict"])
        rank_model.eval()

        def prepare_group(group):
            result = []
            for row in group:
                key = f"{index+1}:{row['id']}"
                if key not in yolo:
                    raise RuntimeError(f"missing YOLO proposal: {key}")
                result.append(E.prepare(row, rank_model, checkpoint, device,
                                        yolo[key], source_score=True))
            return result

        train = prepare_group(train_rows)
        val = prepare_group(val_rows)
        test = prepare_group(test_rows)
        scaler = StandardScaler().fit(np.concatenate([r["x"][r["usable"]] for r in train]))

        def attach(items):
            for item in items:
                item["tensor"] = torch.from_numpy(scaler.transform(item["x"]).astype(np.float32)).to(device)
                item["positive"] = torch.from_numpy(item["hit"]).to(device)
                item["negative"] = torch.from_numpy(item["usable"] & ~item["hit"]).to(device)

        attach(train); attach(val); attach(test)
        model = torch.nn.Linear(train[0]["tensor"].shape[1], 1).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-3, weight_decay=1e-3)
        best_loss, best_epoch, best_state = float("inf"), 0, None
        for epoch in range(EPOCHS):
            model.train()
            optimizer.zero_grad()
            loss = ranking_loss(model, train)
            loss.backward()
            optimizer.step()
            model.eval()
            with torch.no_grad():
                value = float(ranking_loss(model, val))
            if value < best_loss:
                best_loss, best_epoch = value, epoch+1
                best_state = {key: value.detach().cpu().clone()
                              for key, value in model.state_dict().items()}
        model.load_state_dict(best_state)
        model.eval()

        def score(item):
            with torch.no_grad():
                logits = model(item["tensor"]).flatten()
            chosen = int(logits.argmax().item())
            return float(torch.sigmoid(logits[chosen]).item()), chosen

        threshold = float(np.quantile([score(item)[0] for item in val
                                       if not item["defect"]], .95))
        for item in test:
            value, chosen = score(item)
            predictions.append({"id": item["id"], "cam": test_cam,
                                "defect": item["defect"], "score": value,
                                "threshold": threshold, "alarm": value >= threshold,
                                "point_hit": bool(item["hit"][chosen]),
                                "selected_rank": chosen+1,
                                "candidate_reachable": bool(item["hit"].any())})
        folds.append({"test_cam": test_cam, "val_cam": val_cam,
                      "train_frames": len(train), "train_positive": sum(r["defect"] for r in train),
                      "best_epoch": best_epoch, "best_val_loss": best_loss,
                      "threshold": threshold})
        print(f"pairwise fold {index+1}/6 {test_cam}: best epoch {best_epoch}, "
              f"train {len(train)} frames", flush=True)
    pos = [r for r in predictions if r["defect"]]
    neg = [r for r in predictions if not r["defect"]]
    summary = {"positive": len(pos), "normal": len(neg),
               "candidate_reachable": sum(r["candidate_reachable"] for r in pos),
               "alarm_and_point": sum(r["alarm"] and r["point_hit"] for r in pos),
               "point_hits": sum(r["point_hit"] for r in pos),
               "false_alarms": sum(r["alarm"] for r in neg),
               "missed_ids": [r["id"] for r in pos if not (r["alarm"] and r["point_hit"])]}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"scope": "July single-event retrospective; noncausal M4 cache",
                               "A_label_sha256": digest, "excluded_uncertain": excluded,
                               "head": "linear candidate scorer; positive-versus-negative image ranking + normal max suppression",
                               "epochs": EPOCHS, "summary": summary,
                               "folds": folds, "rows": predictions},
                              ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
