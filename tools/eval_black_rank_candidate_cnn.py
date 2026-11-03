"""Small CNN verifier of out-of-fold DINO rank candidates, six camera folds.

Retrospective July one-event development experiment. The frozen M4 cache has
noncausal same-camera leave-one-out references; results are not field recall.
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import audit_black_yarn_candidate_coverage as C  # noqa: E402
import fewshot_semantic_heads as S  # noqa: E402
import train_black_reviewer_a as A  # noqa: E402

RANK = ROOT / "outputs/black_reviewer_a_rank/results.json"
OUT = ROOT / "outputs/black_yarn_candidate_audit/cnn_rerank.json"
SEED = 20261103
K, RADIUS, EPOCHS = 10, 3, 12


class Verifier(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(channels, 8, 1), torch.nn.ReLU(),
            torch.nn.Conv2d(8, 8, 3, padding=1), torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(),
            torch.nn.Linear(8, 1))

    def forward(self, x):
        return self.net(x).flatten()


def near_box(point, boxes, dx, dy):
    x, y = point
    return any(x0-dx <= x <= x1+dx and y0-dy <= y <= y1+dy
               for x0, y0, x1, y1 in boxes)


def prepare(row, rank_model, checkpoint, device):
    m4, _ = S.array_for(row, "M4")
    x = np.clip((m4 - checkpoint["mean"]) / checkpoint["std"], -5, 5)
    with torch.no_grad():
        prob = torch.sigmoid(rank_model(torch.from_numpy(x.transpose(2, 0, 1).copy())[None]
                                        .to(device)))[0, 0].cpu().numpy()
    peaks = C.candidates(prob, row, K)
    gh, gw = prob.shape
    dx, dy = row["hw"][1] / gw, (row["y1"] - row["y0"]) / gh
    stack = np.concatenate([x, prob[..., None]], axis=-1)
    padded = np.pad(stack, ((RADIUS, RADIUS), (RADIUS, RADIUS), (0, 0)), mode="edge")
    patches = np.stack([padded[p["patch"][0]:p["patch"][0]+2*RADIUS+1,
                              p["patch"][1]:p["patch"][1]+2*RADIUS+1]
                        for p in peaks]).transpose(0, 3, 1, 2).copy()
    return {"id": row["id"], "cam": row["cam"], "defect": bool(row["gt"]),
            "x": patches,
            "hit": np.array([C.hit(p["point"], row["gt"]) for p in peaks], bool),
            "usable": np.array([C.hit(p["point"], row["gt"]) or
                                not near_box(p["point"], row["gt"], dx, dy)
                                for p in peaks], bool)}


def main():
    rows, excluded, digest = A.reviewed_rows()
    bundle = json.loads(RANK.read_text(encoding="utf-8"))["methods"]["mil_rank"]
    by_cam = {f["test_cam"]: f for f in bundle["folds"]}
    rank_pred = {r["id"]: r for r in bundle["rows"]}
    cams = sorted(by_cam)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    report, folds = [], []
    for index, test_cam in enumerate(cams):
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        val_cam = cams[(index+1) % len(cams)]
        fold = by_cam[test_cam]
        checkpoint_path = Path(fold["checkpoint"])
        if hashlib.sha256(checkpoint_path.read_bytes()).hexdigest() != fold["checkpoint_sha256"]:
            raise RuntimeError(f"checkpoint hash mismatch: {test_cam}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        tr = [r for r in rows if r["cam"] not in (test_cam, val_cam)]
        va = [r for r in rows if r["cam"] == val_cam]
        te = [r for r in rows if r["cam"] == test_cam]
        if set(checkpoint["train_ids"]) != {r["id"] for r in tr} or \
           set(checkpoint["val_ids"]) != {r["id"] for r in va} or \
           set(checkpoint["test_ids"]) != {r["id"] for r in te}:
            raise RuntimeError(f"split mismatch: {test_cam}")
        rank_model = S.Head("M4").to(device)
        rank_model.load_state_dict(checkpoint["state_dict"])
        rank_model.eval()
        train = [prepare(r, rank_model, checkpoint, device) for r in tr]
        val = [prepare(r, rank_model, checkpoint, device) for r in va]
        test = [prepare(r, rank_model, checkpoint, device) for r in te]
        xx = np.concatenate([r["x"][r["usable"]] for r in train])
        yy = np.concatenate([r["hit"][r["usable"]] for r in train]).astype(np.float32)
        if yy.sum() == 0 or yy.sum() == len(yy):
            raise RuntimeError(f"training candidates lack a class: {test_cam}")
        xx = torch.from_numpy(xx).to(device)
        yy = torch.from_numpy(yy).to(device)
        vx = torch.from_numpy(np.concatenate([r["x"][r["usable"]] for r in val])).to(device)
        vy = torch.from_numpy(np.concatenate([r["hit"][r["usable"]] for r in val])
                              .astype(np.float32)).to(device)
        model = Verifier(xx.shape[1]).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
        weight = torch.tensor(float((len(yy)-yy.sum().item())/yy.sum().item()), device=device)
        best_loss, best_epoch, best_state = float("inf"), 0, None
        for epoch in range(EPOCHS):
            model.train()
            for batch in torch.randperm(len(yy), device=device).split(64):
                optimizer.zero_grad()
                loss = F.binary_cross_entropy_with_logits(model(xx[batch]), yy[batch],
                                                          pos_weight=weight)
                loss.backward()
                optimizer.step()
            model.eval()
            with torch.no_grad():
                val_loss = float(F.binary_cross_entropy_with_logits(model(vx), vy,
                                                                      pos_weight=weight))
            if val_loss < best_loss:
                best_loss, best_epoch = val_loss, epoch+1
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(best_state)
        model.eval()

        def score(item):
            with torch.no_grad():
                probabilities = torch.sigmoid(model(torch.from_numpy(item["x"]).to(device)))
            selected = int(probabilities.argmax().item())
            return float(probabilities[selected].item()), selected

        threshold = float(np.quantile([score(r)[0] for r in val if not r["defect"]], .95))
        for item in test:
            value, selected = score(item)
            old = rank_pred[item["id"]]
            report.append({"id": item["id"], "cam": test_cam,
                           "defect": item["defect"], "cnn_score": value,
                           "cnn_threshold": threshold, "cnn_alarm": value >= threshold,
                           "cnn_selected_rank": selected+1,
                           "cnn_point_hit": bool(item["hit"][selected]),
                           "rank_alarm": bool(old["a_alarm"]),
                           "rank_point_hit": bool(old["a_point_hit"]),
                           "candidate_reachable": bool(item["hit"].any())})
        folds.append({"test_cam": test_cam, "val_cam": val_cam,
                      "train_candidates": int(len(yy)), "train_positive": int(yy.sum().item()),
                      "selected_epoch": best_epoch, "validation_loss": best_loss,
                      "cnn_threshold": threshold})
        print(f"CNN fold {index+1}/6 {test_cam}: epoch {best_epoch}, "
              f"train {len(yy)} candidates, positive {yy.sum().item():.0f}", flush=True)
    pos = [r for r in report if r["defect"]]
    neg = [r for r in report if not r["defect"]]
    summary = {"positive": len(pos), "normal": len(neg),
               "rank_alarm_and_point": sum(r["rank_alarm"] and r["rank_point_hit"] for r in pos),
               "rank_false_alarms": sum(r["rank_alarm"] for r in neg),
               "candidate_reachable": sum(r["candidate_reachable"] for r in pos),
               "cnn_alarm_and_point": sum(r["cnn_alarm"] and r["cnn_point_hit"] for r in pos),
               "cnn_point_hits": sum(r["cnn_point_hit"] for r in pos),
               "cnn_false_alarms": sum(r["cnn_alarm"] for r in neg),
               "new_hits": [r["id"] for r in pos if r["cnn_alarm"] and r["cnn_point_hit"]
                            and not (r["rank_alarm"] and r["rank_point_hit"])],
               "lost_hits": [r["id"] for r in pos if r["rank_alarm"] and r["rank_point_hit"]
                             and not (r["cnn_alarm"] and r["cnn_point_hit"])]}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"scope": "retrospective July one-event; noncausal DINO cache",
                               "A_label_sha256": digest, "excluded_uncertain": excluded,
                               "candidate_top_k": K, "nms_radius_patches": C.NMS_RADIUS,
                               "cnn": "M4+rank 7x7 patch; 1x1 8ch, 3x3 8ch, global avg, linear",
                               "epochs": EPOCHS, "summary": summary, "folds": folds, "rows": report},
                              ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
