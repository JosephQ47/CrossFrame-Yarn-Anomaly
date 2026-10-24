"""监督预测头技术烟测：仅用可训练候选目录中的两张已知正常帧。

不读取冻结黑纱/F2/现场测试集，不产生缺陷性能指标，也不保存权重。
"""

from pathlib import Path
import hashlib
import json
import time

import cv2
import numpy as np
import timm
import torch


ROOT = Path(r"D:\dataset\factory_fabric_2")
NAMES = (
    "2025-12-19-10-21-17-24-3.jpg",
    "2025-12-19-10-21-18-12-3.jpg",
)
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def prepare(path: Path, device: str) -> torch.Tensor:
    img = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"图像读取失败：{path}")
    h, w = img.shape[:2]
    band = img[int(h * 0.34):int(h * 0.60)]
    band = cv2.resize(band, (588, 448), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(band, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
    x = ((rgb - MEAN) / STD).transpose(2, 0, 1)
    return torch.from_numpy(x[None]).to(device)


def main() -> None:
    paths = [ROOT / n for n in NAMES]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    digests = [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
    if digests[0] == digests[1]:
        raise RuntimeError("两帧字节级重复，不能作为差分烟测输入")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.perf_counter()
    backbone = timm.create_model(
        "vit_base_patch14_dinov2.lvd142m",
        pretrained=True,
        num_classes=0,
        img_size=(448, 588),
    ).eval().to(device)
    features = []
    with torch.no_grad():
        for path in paths:
            outs = backbone.get_intermediate_layers(prepare(path, device), n=8, reshape=False, norm=True)
            feat = torch.cat(outs, dim=-1)[:, -1344:, :]
            feat = torch.nn.functional.normalize(feat, dim=-1)
            if tuple(feat.shape) != (1, 1344, 6144):
                raise RuntimeError(f"DINOv2 输出形状异常：{tuple(feat.shape)}")
            features.append(feat.reshape(1, 32, 42, 6144).permute(0, 3, 1, 2).contiguous())

    cur, ref = features
    cos = 1 - (cur * ref).sum(dim=1, keepdim=True)
    diff = (cur - ref).abs()
    head = torch.nn.Sequential(
        torch.nn.Conv2d(6144, 64, 1),
        torch.nn.ReLU(),
        torch.nn.Conv2d(64, 32, 3, padding=1),
        torch.nn.ReLU(),
        torch.nn.Conv2d(32, 1, 1),
    ).to(device)
    head.train()
    logits = head(diff)
    # 两张候选正常帧只用于检查训练计算图；全零目标不是缺陷训练标签。
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.zeros_like(logits))
    loss.backward()
    grad_ok = all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters())
    if not grad_ok or not torch.isfinite(cos).all() or not torch.isfinite(logits).all():
        raise RuntimeError("出现非有限值或预测头梯度缺失")
    print(json.dumps({
        "status": "smoke_pass",
        "data_use": "factory_fabric_2 的两张候选正常帧；无缺陷性能评估",
        "device": device,
        "torch": torch.__version__,
        "timm": timm.__version__,
        "tile_features": [1, 1344, 6144],
        "difference_map": list(diff.shape),
        "cosine_map": list(cos.shape),
        "head_logits": list(logits.shape),
        "head_gradients_finite": bool(grad_ok),
        "elapsed_seconds": round(time.perf_counter() - t0, 2),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
