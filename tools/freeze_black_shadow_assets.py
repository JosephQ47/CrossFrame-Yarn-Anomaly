"""Record immutable hashes for the frozen M0 backbone and six M4 folds."""

import hashlib
import json
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
M4 = ROOT / "outputs/black_fewshot_semantic"
OUT = ROOT / "data/black_shadow_asset_freeze_2026-10-27.json"
HF = Path.home() / ".cache/huggingface/hub/models--timm--vit_base_patch14_dinov2.lvd142m"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def entry(path):
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def main():
    revision = (HF / "refs/main").read_text(encoding="utf-8").strip()
    backbone = HF / "snapshots" / revision / "model.safetensors"
    checkpoints = {f"Cam{i}": entry(M4 / f"M4_fold{i}_Cam{i}.pt") for i in range(1, 7)}
    payload = {
        "scope": "asset identity and archived-code provenance; no new training or accuracy claim",
        "backbone_name": "vit_base_patch14_dinov2.lvd142m",
        "backbone_hf_revision": revision,
        "backbone": entry(backbone),
        "m4_projection": entry(M4 / "fixed_projection.npy"),
        "m4_checkpoints": checkpoints,
        "archived_m4_results": entry(ROOT / "data/black_fewshot_semantic_2026-10-25/M4_results.json"),
        "source_files": {p: entry(ROOT / p) for p in (
            "src/black_yarn_detector.py", "src/deploy_black_detector.py",
            "tools/fewshot_semantic_heads.py", "tools/m4_causal_shadow.py")},
        "software_versions": {name: version(name) for name in ("torch", "timm", "numpy", "opencv-python")},
        "feature_contract": {"dino_last_layers": 8, "topk_same_position": 1,
                             "band_y_fraction": [0.34, 0.60], "tile_wh": [588, 448],
                             "projection_dim": 64, "m4_channels": 193},
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {OUT}; backbone={payload['backbone']['sha256'][:12]}, "
          f"M4 checkpoints={len(checkpoints)}")


if __name__ == "__main__":
    main()
