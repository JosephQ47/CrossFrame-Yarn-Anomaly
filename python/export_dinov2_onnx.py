"""导出黑纱检测器的特征提取骨干(DINOv2 ViT-B/14, 最后8层拼接+L2归一化)为 ONNX。
供浆纱检测系统(C#/onnxruntime)原生集成——跨帧参考/连通域逻辑在 C# 侧实现。
输入: (1,3,448,588) RGB, ImageNet 归一化后
输出: (1,1344,6144)  1344=32×42 patch 网格, 6144=768×8 层拼接, 已 L2 归一化
验证: onnxruntime vs torch 输出一致性。
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"; os.environ["PYTHONUTF8"] = "1"
from pathlib import Path
import numpy as np, torch, torch.nn as nn, timm

OUT = Path(r"C:\Users\quanzhonghui\Documents\色纱课题\outputs\onnx_export")
OUT.mkdir(parents=True, exist_ok=True)
ONNX = OUT / "dinov2_vitb14_l8_448x588.onnx"
H, W, LAYERS = 448, 588, 8
GH, GW = H // 14, W // 14   # 32 x 42 = 1344


class FeatExtractor(nn.Module):
    def __init__(self, dynamic=False):
        super().__init__()
        # dynamic_img_size 的运行时 pos_embed 插值(bicubic AA)无法导 ONNX;
        # 固定 img_size 让 timm 在加载权重时一次性重采样位置编码,图中无该算子。
        if dynamic:
            self.backbone = timm.create_model("vit_base_patch14_dinov2.lvd142m",
                                              pretrained=True, num_classes=0,
                                              dynamic_img_size=True).eval()
        else:
            self.backbone = timm.create_model("vit_base_patch14_dinov2.lvd142m",
                                              pretrained=True, num_classes=0,
                                              img_size=(H, W)).eval()

    def forward(self, x):
        outs = self.backbone.get_intermediate_layers(x, n=LAYERS, reshape=False, norm=True)
        f = torch.cat(outs, dim=-1)              # (1, N, 6144)
        f = f[:, -GH * GW:, :]                   # 去 cls 等前缀 token
        f = f / f.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        return f


def main():
    m = FeatExtractor().eval()
    dummy = torch.randn(1, 3, H, W)
    with torch.no_grad():
        ref = m(dummy)
    print(f"torch 输出: {tuple(ref.shape)}")

    # 与基准所用 dynamic_img_size 版本比对(pos_embed 提前重采样 vs 运行时插值应一致)
    with torch.no_grad():
        ref_dyn = FeatExtractor(dynamic=True).eval()(dummy)
    d = float((ref - ref_dyn).abs().max())
    print(f"固定尺寸 vs dynamic 版最大差: {d:.2e} → {'✓ 等价' if d < 1e-4 else '⚠ 有差异,基准成绩需复核'}")

    torch.onnx.export(m, dummy, str(ONNX),
                      input_names=["input"], output_names=["features"],
                      opset_version=17, do_constant_folding=True)
    print(f"已导出: {ONNX} ({ONNX.stat().st_size/1e6:.0f} MB)")

    import onnxruntime as ort
    sess = ort.InferenceSession(str(ONNX), providers=["CPUExecutionProvider"])
    out = sess.run(None, {"input": dummy.numpy()})[0]
    diff = float(np.abs(out - ref.numpy()).max())
    print(f"onnxruntime 验证: 最大误差 {diff:.2e} → {'✓ 通过' if diff < 1e-3 else '✗ 偏大'}")

    import time
    t0 = time.time(); n = 3
    for _ in range(n):
        sess.run(None, {"input": dummy.numpy()})
    print(f"CPU 单tile推理: {(time.time()-t0)/n*1000:.0f} ms")


if __name__ == "__main__":
    main()
