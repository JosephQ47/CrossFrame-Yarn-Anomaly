"""G2 diagnostic: freeze archived M4 localization; train only image alarm head.

July single-fault retrospective development split only. F2 is never read.
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import fewshot_semantic_heads as S  # noqa: E402
from train_black_dual_head import OLD, SEED, source_rows  # noqa: E402

DEFAULT_OUT = ROOT / "data/black_frozen_alarm_2026-10-31"


class AlarmHead(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(33, 16), torch.nn.ReLU(), torch.nn.Linear(16, 1))

    def forward(self, x):
        return self.net(x).flatten()


@torch.no_grad()
def frozen_vectors(model, rows, mean, std, device):
    model.eval()
    out = {}
    for row in rows:
        x, _ = S.load_row(row, "M4", mean, std)
        z = model.net[:-1](x[None].to(device))
        mask = model.net[-1](z)
        k = max(1, round(mask.shape[-2] * mask.shape[-1] * .01))
        top = mask.flatten(2).topk(k, dim=-1).values.mean(dim=-1)
        vector = torch.cat([z.mean(dim=(-2, -1)), z.amax(dim=(-2, -1)), top], dim=1)
        out[row["id"]] = vector[0].cpu()
    return out


def metrics(rows, prefix):
    positive = [r for r in rows if r["defect"]]
    negative = [r for r in rows if not r["defect"]]
    return {"alarm_hits": sum(r[f"{prefix}_alarm"] for r in positive),
            "false_alarms": sum(r[f"{prefix}_alarm"] for r in negative),
            "point_hits": sum(r["m4_point_hit"] for r in positive),
            "alarm_and_point_hits": sum(r[f"{prefix}_alarm"] and
                                        r["m4_point_hit"] for r in positive)}


def run(epochs, out):
    if epochs < 1:
        raise ValueError("epochs must be positive")
    rows = source_rows()
    archive = {r["id"]: r for r in json.loads(OLD.read_text(encoding="utf-8"))["rows"]}
    if set(archive) != {r["id"] for r in rows}:
        raise RuntimeError("M4 archive/index mismatch")
    out.mkdir(parents=True, exist_ok=True)
    weight_dir = out / "weights"
    weight_dir.mkdir(exist_ok=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cams = sorted({r["cam"] for r in rows})
    comparison, folds = [], []
    started = time.perf_counter()
    for fold, cam in enumerate(cams):
        val_cam = cams[(fold + 1) % len(cams)]
        checkpoint = S.OUT / f"M4_fold{fold+1}_{cam}.pt"
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        train = [r for r in rows if r["cam"] not in (cam, val_cam)]
        val = [r for r in rows if r["cam"] == val_cam]
        test = [r for r in rows if r["cam"] == cam]
        for key, selected in (("train_ids", train), ("val_ids", val), ("test_ids", test)):
            if set(saved[key]) != {r["id"] for r in selected}:
                raise RuntimeError(f"checkpoint split mismatch: {cam}/{key}")
        m4 = S.Head("M4").to(device).eval()
        m4.load_state_dict(saved["state_dict"])
        vectors = frozen_vectors(m4, train + val + test, saved["mean"], saved["std"], device)
        train_x = torch.stack([vectors[r["id"]] for r in train]).to(device)
        train_y = torch.tensor([float(bool(r["gt"])) for r in train], device=device)
        val_x = torch.stack([vectors[r["id"]] for r in val]).to(device)
        val_y = torch.tensor([float(bool(r["gt"])) for r in val], device=device)
        test_x = torch.stack([vectors[r["id"]] for r in test]).to(device)
        weight = torch.tensor(float((len(train) - train_y.sum()) / max(float(train_y.sum()), 1)),
                              device=device)
        head = AlarmHead().to(device)
        optimizer = torch.optim.Adam(head.parameters(), lr=1e-3)
        best_loss, best_epoch, best_state = float("inf"), -1, None
        for epoch in range(epochs):
            head.train()
            for idx in np.random.permutation(len(train)):
                optimizer.zero_grad()
                logit = head(train_x[idx:idx+1])
                loss = F.binary_cross_entropy_with_logits(logit, train_y[idx:idx+1],
                                                           pos_weight=weight)
                loss.backward()
                optimizer.step()
            head.eval()
            with torch.no_grad():
                val_loss = float(F.binary_cross_entropy_with_logits(
                    head(val_x), val_y, pos_weight=weight))
            if val_loss < best_loss:
                best_loss, best_epoch = val_loss, epoch + 1
                best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
        head.load_state_dict(best_state)
        with torch.no_grad():
            val_scores = torch.sigmoid(head(val_x)).cpu().numpy()
            test_scores = torch.sigmoid(head(test_x)).cpu().numpy()
        normal_val = [float(score) for row, score in zip(val, val_scores) if not row["gt"]]
        threshold = float(np.quantile(normal_val, .95, method="linear"))
        head_path = weight_dir / f"alarm_fold{fold+1}_{cam}.pt"
        torch.save({"state_dict": best_state, "train_ids": saved["train_ids"],
                    "val_ids": saved["val_ids"], "test_ids": saved["test_ids"],
                    "source_m4_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                    "best_epoch": best_epoch, "threshold": threshold}, head_path)
        folds.append({"fold": fold + 1, "test_cam": cam, "val_cam": val_cam,
                      "best_epoch": best_epoch, "best_val_loss": best_loss,
                      "threshold": threshold, "m4_checkpoint_sha256":
                      hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                      "alarm_checkpoint_sha256": hashlib.sha256(head_path.read_bytes()).hexdigest(),
                      "val_normal_scores": [{"id": row["id"], "score": float(score)}
                                            for row, score in zip(val, val_scores) if not row["gt"]]})
        for row, score in zip(test, test_scores):
            old = archive[row["id"]]
            comparison.append({"id": row["id"], "cam": cam, "defect": bool(row["gt"]),
                               "m4_score": old["score"], "m4_threshold": old["threshold"],
                               "m4_alarm": old["alarm"], "m4_point_hit": old["point_hit"],
                               "frozen_alarm_score": float(score),
                               "frozen_alarm_threshold": threshold,
                               "frozen_alarm_alarm": bool(score >= threshold),
                               "m4_box": old["pred_box"]})
        print(f"fold {fold+1}/6 {cam}: epoch={best_epoch} val={best_loss:.4f} "
              f"threshold={threshold:.4f}", flush=True)
    labels = np.array([r["defect"] for r in comparison], dtype=np.uint8)
    scores = np.array([r["frozen_alarm_score"] for r in comparison])
    summary = {"scope": "single July fault; frozen archived M4 map, separate image head",
               "independent_fault_test": False, "causal_references": False,
               "frames": len(comparison), "positive_frames": int(labels.sum()),
               "normal_frames": int((1 - labels).sum()), "seed": SEED, "epochs": epochs,
               "device": device, "elapsed_seconds": round(time.perf_counter() - started, 2),
               "source_index_sha256": hashlib.sha256(S.INDEX.read_bytes()).hexdigest(),
               "archived_m4_sha256": hashlib.sha256(OLD.read_bytes()).hexdigest(),
               "frozen_alarm_auroc": float(roc_auc_score(labels, scores)),
               "frozen_alarm_ap": float(average_precision_score(labels, scores)),
               "m4": metrics(comparison, "m4"),
               "frozen_alarm": metrics(comparison, "frozen_alarm"),
               "by_camera": {cam: {"m4": metrics([r for r in comparison if r["cam"] == cam], "m4"),
                                   "frozen_alarm": metrics([r for r in comparison if r["cam"] == cam],
                                                           "frozen_alarm")}
                             for cam in cams}}
    (out / "results.json").write_text(json.dumps({"summary": summary, "folds": folds,
                                                  "rows": comparison}, ensure_ascii=False,
                                                 indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    run(args.epochs, args.out)


if __name__ == "__main__":
    main()
