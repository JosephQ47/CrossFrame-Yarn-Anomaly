"""复核 M1 基线，并仅替换监督标签为原始 detect 多边形进行六折对照。"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import fewshot_m1 as M1  # noqa: E402

ARCHIVE = ROOT / "data" / "black_fewshot_m1_2026-10-25"
OUT = ROOT / "data" / "black_polygon_supervision_2026-10-25"
EPOCHS = 20


def polygon_grid(row, shape):
    gh, gw = shape
    ans = np.zeros(shape, np.uint8)
    if not row["gt"]:
        return ans
    label_path = Path(row["file"]).with_suffix(".json")
    shapes = json.loads(label_path.read_text(encoding="utf-8"))["shapes"]
    polygons = [np.asarray(s["points"], np.float32) for s in shapes
                if s["label"].lower() == "detect"]
    assert polygons and all(p.shape[0] >= 3 for p in polygons)
    h, w = row["hw"]
    xs = (np.arange(gw) + .5) * w / gw
    ys = row["y0"] + (np.arange(gh) + .5) * (row["y1"] - row["y0"]) / gh
    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            ans[i, j] = any(cv2.pointPolygonTest(p, (float(x), float(y)), False) >= 0
                             for p in polygons)
    return ans


def check_inputs(rows):
    fresh = {r["id"]: r for r in M1.records()}
    assert len(rows) == len(fresh) == 69
    assert all(r["sha256"] == fresh[r["id"]]["sha256"] for r in rows)
    labels = {}
    audit = []
    for r in rows:
        cache_path = M1.CACHE / (r["id"] + ".npz")
        with np.load(cache_path) as z:
            box = z["label"]
            shape = box.shape
        poly = polygon_grid(r, shape)
        assert not np.any(poly & ~box), r["id"]
        if r["gt"]:
            assert poly.sum() > 0, r["id"]
        else:
            assert poly.sum() == 0, r["id"]
        labels[r["id"]] = poly
        audit.append({"id": r["id"], "camera": r["cam"],
                      "box_positive_patches": int(box.sum()),
                      "polygon_positive_patches": int(poly.sum())})
    return labels, audit


def load_polygon(row, labels):
    x, _ = M1.load_row(row)
    y = torch.from_numpy(labels[row["id"]].astype(np.float32)[None])
    return x, y


def eval_rows(model, rows, device):
    model.eval()
    out = []
    with torch.no_grad():
        for r in rows:
            x, _ = M1.load_row(r)
            prob = torch.sigmoid(model(x[None].to(device)))[0, 0].cpu().numpy()
            k = max(1, int(round(prob.size * .01)))
            score = float(np.partition(prob.ravel(), -k)[-k:].mean())
            _, point, box = M1.D.score_and_locate(prob, r["hw"], r["y0"], r["y1"])
            hit = bool(point and any(b[0] <= point[0] <= b[2]
                                     and b[1] <= point[1] <= b[3] for b in r["gt"]))
            polygon_hit = False
            if point and r["gt"]:
                shapes = json.loads(Path(r["file"]).with_suffix(".json").read_text(
                    encoding="utf-8"))["shapes"]
                polygon_hit = any(cv2.pointPolygonTest(
                    np.asarray(s["points"], np.float32),
                    (float(point[0]), float(point[1])), False) >= 0
                    for s in shapes if s["label"].lower() == "detect")
            out.append({"id": r["id"], "cam": r["cam"], "defect": bool(r["gt"]),
                        "score": score, "point_hit": hit,
                        "polygon_point_hit": polygon_hit, "pred_box": box})
    return out


def summary(rows):
    y = np.asarray([int(r["defect"]) for r in rows])
    scores = np.asarray([r["score"] for r in rows])
    pos = [r for r in rows if r["defect"]]
    neg = [r for r in rows if not r["defect"]]
    return {"image_auroc": float(roc_auc_score(y, scores)),
            "image_ap": float(average_precision_score(y, scores)),
            "alarm_detected": sum(r["alarm"] for r in pos),
            "normal_false_alarms": sum(r["alarm"] for r in neg),
            "point_hits": sum(r["point_hit"] for r in pos),
            "alarm_and_point_hit": sum(r["alarm"] and r["point_hit"] for r in pos),
            "polygon_point_hits": sum(r["polygon_point_hit"] for r in pos),
            "alarm_and_polygon_point_hit": sum(r["alarm"] and r["polygon_point_hit"]
                                               for r in pos)}


def verify_old_fold(fold, cam, va, te, device, archive_rows):
    checkpoint = ARCHIVE / f"m1_fold{fold+1}_{cam}.pt"
    pack = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert pack["test_cam"] == cam
    assert pack["val_ids"] == [r["id"] for r in va]
    assert pack["test_ids"] == [r["id"] for r in te]
    model = M1.Head().to(device)
    model.load_state_dict(pack["state_dict"])
    predicted = eval_rows(model, te, device)
    max_score_error = max(abs(p["score"] - archive_rows[p["id"]]["score"])
                          for p in predicted)
    assert max_score_error < 1e-5, (cam, max_score_error)
    return max_score_error, predicted


def train_polygon_fold(fold, test_cam, tr, va, te, labels, device):
    model = M1.Head().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    best_state, best_loss = None, float("inf")
    for _ in range(EPOCHS):
        model.train()
        for idx in np.random.permutation(len(tr)):
            x, y = load_polygon(tr[idx], labels)
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
                x, y = load_polygon(r, labels)
                logits = model(x[None].to(device))
                losses.append(float(torch.nn.functional.binary_cross_entropy_with_logits(
                    logits, y[None].to(device)).item()))
            val_loss = float(np.mean(losses))
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    torch.save({"state_dict": best_state, "fold": fold, "test_cam": test_cam,
                "val_cam": va[0]["cam"], "train_ids": [r["id"] for r in tr],
                "val_ids": [r["id"] for r in va], "test_ids": [r["id"] for r in te],
                "seed": M1.SEED, "epochs": EPOCHS, "label_mode": "polygon_patch_center",
                "best_val_loss": best_loss}, OUT / f"polygon_fold{fold+1}_{test_cam}.pt")
    val_pred = eval_rows(model, va, device)
    test_pred = eval_rows(model, te, device)
    threshold = float(np.quantile([r["score"] for r in val_pred
                                   if not r["defect"]], .95))
    for r in test_pred:
        r.update({"fold": fold, "test_cam": test_cam, "val_cam": va[0]["cam"],
                  "threshold": threshold, "alarm": r["score"] >= threshold})
    return test_pred


def main():
    index = json.loads(M1.INDEX.read_text(encoding="utf-8"))
    rows = index["data"]
    assert index["seed"] == M1.SEED
    labels, audit = check_inputs(rows)
    old_data = json.loads((ARCHIVE / "results.json").read_text(encoding="utf-8"))
    archive_rows = {r["id"]: r for r in old_data["rows"]}
    assert set(archive_rows) == {r["id"] for r in rows}
    torch.manual_seed(M1.SEED)
    np.random.seed(M1.SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    OUT.mkdir(parents=True, exist_ok=True)
    cams = sorted({r["cam"] for r in rows})
    all_pred, old_pred, errors = [], [], []
    for fold, test_cam in enumerate(cams):
        val_cam = cams[(fold + 1) % len(cams)]
        tr = [r for r in rows if r["cam"] not in (test_cam, val_cam)]
        va = [r for r in rows if r["cam"] == val_cam]
        te = [r for r in rows if r["cam"] == test_cam]
        # 旧权重复核会构造模型，必须恢复 RNG 才能与原 M1 的逐折初始化一致。
        cpu_rng = torch.get_rng_state()
        error, baseline = verify_old_fold(fold, test_cam, va, te, device, archive_rows)
        errors.append(error)
        old_pred.extend(baseline)
        torch.set_rng_state(cpu_rng)
        pred = train_polygon_fold(fold, test_cam, tr, va, te, labels, device)
        all_pred.extend(pred)
        print(f"fold {fold+1}/6 {test_cam}: {len(pred)} 帧", flush=True)
    assert len(all_pred) == 69
    prior = []
    verified_old = {r["id"]: r for r in old_pred}
    for r in all_pred:
        old = dict(verified_old[r["id"]])
        old.update({k: archive_rows[r["id"]][k] for k in
                    ("fold", "test_cam", "val_cam", "threshold", "alarm")})
        prior.append(old)
    report = {"protocol": "仅改变 M1 的训练与验证标签：外接矩形中心 → detect 多边形中心；原异常图缓存、模型、种子、六折划分、20 epoch、阈值与评价定义相同",
              "limitation": "所有 16 张阳性均来自同一故障；同相机批内留一参考含未来帧；F2 未使用",
              "baseline_checkpoint_max_score_error": max(errors),
              "label_audit": audit,
              "summary": {"box": summary(prior), "polygon": summary(all_pred)},
              "rows": [{"id": r["id"], "camera": r["cam"],
                        "box": old, "polygon": r}
                       for old, r in zip(prior, all_pred)]}
    (OUT / "comparison.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                          encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
