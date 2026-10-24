"""M2–M4: fixed DINOv2 semantic projections and supervised heads.

Usage: python tools/fewshot_semantic_heads.py prepare
       python tools/fewshot_semantic_heads.py train

July 2026 black-fabric images are the user-authorized development batch.
F2 is not read. All results are retrospective within one fault event.
"""

import argparse
import csv
import json
import sys
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
import black_yarn_detector as D  # noqa: E402
import fewshot_m1 as M1  # noqa: E402

OUT = ROOT / "outputs" / "black_fewshot_semantic"
CACHE = OUT / "cache"
LAYER_CACHE = OUT / "cache_layerwise"
INDEX = OUT / "index.json"
SEED = 20261024
PROJ_DIM = 64


def make_projection():
    rng = np.random.default_rng(SEED)
    p = rng.standard_normal((D.LAYERS * 768, PROJ_DIM), dtype=np.float32)
    return (p / np.float32(np.sqrt(PROJ_DIM))).astype(np.float32)


def prepare():
    import timm

    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    recs = M1.records()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = timm.create_model(D.MODEL, pretrained=True, num_classes=0,
                              dynamic_img_size=True).eval().to(device)
    proj = torch.from_numpy(make_projection()).to(device)
    np.save(OUT / "fixed_projection.npy", proj.cpu().numpy())
    print(f"M2-M4 prepare: {len(recs)} frames, device={device}, projection={PROJ_DIM}", flush=True)
    for cam in sorted({r["cam"] for r in recs}):
        sub = [r for r in recs if r["cam"] == cam]
        feats = []
        for r in sub:
            img = D.read_bgr(Path(r["file"]))
            if img is None:
                raise RuntimeError(f"cannot read {r['file']}")
            f, y0, y1 = D.band_feat(model, device, img)
            r["hw"], r["y0"], r["y1"] = list(img.shape[:2]), y0, y1
            feats.append(f)
        F = np.stack(feats)
        ii, jj = np.indices(F.shape[1:3])
        for k, r in enumerate(sub):
            sims = np.einsum("tijc,ijc->tij", F, F[k], optimize=True)
            sims[k] = -np.inf
            best = np.argmax(sims, axis=0)
            ref = F[best, ii, jj]
            cos = 1.0 - np.max(sims, axis=0)
            # Verify that all heads use the same original anomaly/reference rule as M1.
            m1_path = M1.CACHE / (r["id"] + ".npz")
            m1_anom = np.load(m1_path)["anom"].astype(np.float32)
            if np.max(np.abs(cos - m1_anom)) > 2e-3:
                raise RuntimeError(f"reference mismatch against M1: {r['id']}")
            with torch.no_grad():
                cur_t = torch.from_numpy(F[k].reshape(-1, F.shape[-1])).to(device)
                ref_t = torch.from_numpy(ref.reshape(-1, F.shape[-1])).to(device)
                cur_p = (cur_t @ proj).cpu().numpy().reshape(*cos.shape, PROJ_DIM)
                ref_p = (ref_t @ proj).cpu().numpy().reshape(*cos.shape, PROJ_DIM)
                diff_p = ((cur_t - ref_t).abs() @ proj).cpu().numpy().reshape(*cos.shape, PROJ_DIM)
            label = M1.box_grid(r["gt"], r["hw"], r["y0"], r["y1"], *cos.shape)
            np.savez_compressed(CACHE / (r["id"] + ".npz"),
                                cur=cur_p.astype(np.float16), ref=ref_p.astype(np.float16),
                                diff=diff_p.astype(np.float16), cos=cos.astype(np.float16),
                                label=label.astype(np.uint8))
        del F, feats
        print(f"  {cam}: {len(sub)} frames", flush=True)
    INDEX.write_text(json.dumps({"data": recs, "projection_seed": SEED,
                                 "projection_dim": PROJ_DIM,
                                 "reference": "same-camera leave-one-out top1 retrospective",
                                 "source": str(D.BF)}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"prepared {INDEX}", flush=True)


def prepare_layerwise():
    """M5: preserve each of the eight DINOv2 layers in a separate 8-D block."""
    import timm

    OUT.mkdir(parents=True, exist_ok=True)
    LAYER_CACHE.mkdir(parents=True, exist_ok=True)
    recs = M1.records()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = timm.create_model(D.MODEL, pretrained=True, num_classes=0,
                              dynamic_img_size=True).eval().to(device)
    rng = np.random.default_rng(SEED)
    proj_np = np.zeros((D.LAYERS * 768, PROJ_DIM), np.float32)
    for layer in range(D.LAYERS):
        block = rng.standard_normal((768, PROJ_DIM // D.LAYERS), dtype=np.float32)
        proj_np[layer*768:(layer+1)*768,
                layer*8:(layer+1)*8] = block / np.float32(np.sqrt(8))
    proj = torch.from_numpy(proj_np).to(device)
    np.save(OUT / "fixed_layerwise_projection.npy", proj_np)
    for cam in sorted({r["cam"] for r in recs}):
        sub = [r for r in recs if r["cam"] == cam]
        feats = []
        for r in sub:
            img = D.read_bgr(Path(r["file"]))
            if img is None:
                raise RuntimeError(f"cannot read {r['file']}")
            f, y0, y1 = D.band_feat(model, device, img)
            r["hw"], r["y0"], r["y1"] = list(img.shape[:2]), y0, y1
            feats.append(f)
        F = np.stack(feats)
        ii, jj = np.indices(F.shape[1:3])
        for k, r in enumerate(sub):
            sims = np.einsum("tijc,ijc->tij", F, F[k], optimize=True)
            sims[k] = -np.inf
            best = np.argmax(sims, axis=0)
            ref = F[best, ii, jj]
            with torch.no_grad():
                diff = torch.from_numpy(np.abs(F[k] - ref).reshape(-1, F.shape[-1])).to(device)
                xp = (diff @ proj).cpu().numpy().reshape(*F.shape[1:3], PROJ_DIM)
            label = M1.box_grid(r["gt"], r["hw"], r["y0"], r["y1"], *F.shape[1:3])
            np.savez_compressed(LAYER_CACHE / (r["id"] + ".npz"),
                                layerdiff=xp.astype(np.float16), label=label.astype(np.uint8))
        del F, feats
        print(f"M5 prepare {cam}: {len(sub)} frames", flush=True)
    print(f"M5 cache complete: {len(list(LAYER_CACHE.glob('*.npz')))} frames", flush=True)


@lru_cache(maxsize=250)
def cached_array(image_id, method):
    path = (LAYER_CACHE if method == "M5" else CACHE) / (image_id + ".npz")
    with np.load(path) as z:
        if method == "M5":
            x = z["layerdiff"].astype(np.float32)
        elif method in ("M2", "M3"):
            x = z["diff"].astype(np.float32)
        elif method == "M4":
            x = np.concatenate([z["cur"].astype(np.float32),
                                z["ref"].astype(np.float32),
                                z["diff"].astype(np.float32),
                                z["cos"].astype(np.float32)[..., None]], axis=-1)
        else:
            raise ValueError(method)
        y = z["label"].astype(np.float32)
    return x, y


def array_for(row, method):
    return cached_array(row["id"], method)


def train_stats(rows, method):
    sums = sums2 = None
    n = 0
    for r in rows:
        x, _ = array_for(r, method)
        a = x.reshape(-1, x.shape[-1]).astype(np.float64)
        sums = a.sum(axis=0) if sums is None else sums + a.sum(axis=0)
        sums2 = (a * a).sum(axis=0) if sums2 is None else sums2 + (a * a).sum(axis=0)
        n += len(a)
    mean = sums / n
    std = np.sqrt(np.maximum(sums2 / n - mean * mean, 1e-8))
    return mean.astype(np.float32), std.astype(np.float32)


def load_row(row, method, mean, std):
    x, y = array_for(row, method)
    x = np.clip((x - mean) / std, -5, 5).transpose(2, 0, 1).copy()
    return torch.from_numpy(x), torch.from_numpy(y[None])


class Head(torch.nn.Module):
    def __init__(self, method):
        super().__init__()
        if method == "M2":
            self.net = torch.nn.Conv2d(PROJ_DIM, 1, 1)
        elif method == "M3":
            self.net = torch.nn.Sequential(
                torch.nn.Conv2d(PROJ_DIM, 32, 1), torch.nn.ReLU(),
                torch.nn.Conv2d(32, 16, 3, padding=1), torch.nn.ReLU(),
                torch.nn.Conv2d(16, 1, 1))
        elif method == "M4":
            self.net = torch.nn.Sequential(
                torch.nn.Conv2d(PROJ_DIM * 3 + 1, 32, 1), torch.nn.ReLU(),
                torch.nn.Conv2d(32, 16, 3, padding=1), torch.nn.ReLU(),
                torch.nn.Conv2d(16, 1, 1))
        elif method == "M5":
            self.branches = torch.nn.ModuleList([
                torch.nn.Sequential(torch.nn.Conv2d(8, 8, 1), torch.nn.ReLU())
                for _ in range(D.LAYERS)])
            self.fuse = torch.nn.Sequential(torch.nn.Conv2d(64, 32, 1), torch.nn.ReLU())
            self.local = torch.nn.Sequential(torch.nn.Conv2d(32, 16, 3, padding=1), torch.nn.ReLU())
            self.context = torch.nn.Sequential(torch.nn.Conv2d(32, 16, 3, stride=2, padding=1),
                                               torch.nn.ReLU())
            self.out = torch.nn.Conv2d(32, 1, 1)
        else:
            raise ValueError(method)

    def forward(self, x):
        if hasattr(self, "branches"):
            z = self.fuse(torch.cat([branch(a) for branch, a in zip(
                self.branches, torch.chunk(x, D.LAYERS, dim=1))], dim=1))
            local = self.local(z)
            context = torch.nn.functional.interpolate(self.context(z), size=z.shape[-2:],
                                                       mode="bilinear", align_corners=False)
            return self.out(torch.cat([local, context], dim=1))
        return self.net(x)


def predict(model, rows, method, mean, std, device):
    model.eval()
    out = []
    with torch.no_grad():
        for r in rows:
            x, _ = load_row(r, method, mean, std)
            prob = torch.sigmoid(model(x[None].to(device)))[0, 0].cpu().numpy()
            k = max(1, int(round(prob.size * .01)))
            score = float(np.partition(prob.ravel(), -k)[-k:].mean())
            _, point, box = D.score_and_locate(prob, r["hw"], r["y0"], r["y1"])
            hit = bool(point and any(b[0] <= point[0] <= b[2]
                                     and b[1] <= point[1] <= b[3] for b in r["gt"]))
            out.append({"id": r["id"], "cam": r["cam"], "defect": bool(r["gt"]),
                        "score": score, "point_hit": hit, "pred_box": box})
    return out


def train_one(method, recs, epochs):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cams = sorted({r["cam"] for r in recs})
    results = []
    started = time.perf_counter()
    for fold, test_cam in enumerate(cams):
        val_cam = cams[(fold + 1) % len(cams)]
        tr = [r for r in recs if r["cam"] not in (test_cam, val_cam)]
        va = [r for r in recs if r["cam"] == val_cam]
        te = [r for r in recs if r["cam"] == test_cam]
        mean, std = train_stats(tr, method)
        model = Head(method).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        best_state, best_loss = None, float("inf")
        for _ in range(epochs):
            model.train()
            for idx in np.random.permutation(len(tr)):
                x, y = load_row(tr[idx], method, mean, std)
                x, y = x[None].to(device), y[None].to(device)
                opt.zero_grad()
                logits = model(x)
                n_pos = float(y.sum())
                weight = min(max((y.numel() - n_pos) / max(n_pos, 1), 1), 100)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits, y, pos_weight=torch.tensor(weight, device=device))
                loss.backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                losses = []
                for r in va:
                    x, y = load_row(r, method, mean, std)
                    logits = model(x[None].to(device))
                    losses.append(float(torch.nn.functional.binary_cross_entropy_with_logits(
                        logits, y[None].to(device)).item()))
                val_loss = float(np.mean(losses))
            if val_loss < best_loss:
                best_loss = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(best_state)
        torch.save({"state_dict": best_state, "method": method, "fold": fold,
                    "test_cam": test_cam, "val_cam": val_cam,
                    "train_ids": [r["id"] for r in tr], "val_ids": [r["id"] for r in va],
                    "test_ids": [r["id"] for r in te], "mean": mean, "std": std,
                    "seed": SEED, "epochs": epochs, "best_val_loss": best_loss},
                   OUT / f"{method}_fold{fold+1}_{test_cam}.pt")
        val_pred = predict(model, va, method, mean, std, device)
        test_pred = predict(model, te, method, mean, std, device)
        val_normal = [x["score"] for x in val_pred if not x["defect"]]
        threshold = float(np.quantile(val_normal, .95))
        for x in test_pred:
            x.update({"fold": fold, "test_cam": test_cam, "val_cam": val_cam,
                      "threshold": threshold, "alarm": x["score"] >= threshold})
        results.extend(test_pred)
        print(f"{method} fold {fold+1}/6 test={test_cam} val_loss={best_loss:.4f}", flush=True)
    labels = np.array([int(x["defect"]) for x in results])
    scores = np.array([x["score"] for x in results])
    pos = [x for x in results if x["defect"]]
    neg = [x for x in results if not x["defect"]]
    summary = {"method": method, "frames": len(results), "positive_frames": len(pos),
               "image_auroc": float(roc_auc_score(labels, scores)),
               "image_ap": float(average_precision_score(labels, scores)),
               "fixed_val_threshold_recall": float(np.mean([x["alarm"] for x in pos])),
               "fixed_val_threshold_fpr": float(np.mean([x["alarm"] for x in neg])),
               "point_hits": sum(x["point_hit"] for x in pos),
               "alarm_and_point_hits": sum(x["alarm"] and x["point_hit"] for x in pos),
               "point_total": len(pos), "epochs": epochs, "seed": SEED,
               "projection_dim": PROJ_DIM,
               "retrospective_reference": True, "independent_fault_test": False,
               "elapsed_seconds": round(time.perf_counter() - started, 2)}
    (OUT / f"{method}_results.json").write_text(json.dumps(
        {"summary": summary, "rows": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    with (OUT / f"{method}_scores.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


def train(methods, epochs):
    index = json.loads(INDEX.read_text(encoding="utf-8"))
    recs = index["data"]
    current = {r["id"]: r for r in M1.records()}
    if len(recs) != 69 or any(r["sha256"] != current[r["id"]]["sha256"] for r in recs):
        raise RuntimeError("source data changed or index incomplete")
    for method in methods:
        train_one(method, recs, epochs)
        cached_array.cache_clear()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["prepare", "prepare_layerwise", "train"])
    ap.add_argument("--methods", nargs="+", choices=["M2", "M3", "M4", "M5"], default=["M2", "M3", "M4"])
    ap.add_argument("--epochs", type=int, default=20)
    args = ap.parse_args()
    if args.step == "prepare":
        prepare()
    elif args.step == "prepare_layerwise":
        prepare_layerwise()
    else:
        train(args.methods, args.epochs)


if __name__ == "__main__":
    main()
