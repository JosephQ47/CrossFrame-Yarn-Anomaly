# C# 产线实现

`BlackYarnDetector.cs` 是本方法在工厂浆纱检测系统（.NET Framework 4.7.2 + WinForms +
OpenCvSharp + ONNX Runtime）中的落地实现，已集成进现役系统作为一个默认关闭的可选模块。

## 与 Python 版的关系

算法完全一致，常量在文件顶部集中声明，**与基准一一对应，改动必须回基准复测**：

| 常量 | 值 | 对应 Python |
|---|---|---|
| `TileW / TileH / Overlap` | 588 / 448 / 140 | 同名 |
| `FeatDim` | 6144 = 768 × 8 层 | `LAYERS=8` |
| `BandY0 / BandY1` | 0.34 / 0.60 | `BAND_Y0/Y1` |
| `AreaCap` | 55 | `AREA_CAP` |
| `KMad` | 3.5 | `DET_K_MAD` |
| `BufSize` | 8 | `MIN_REF` |

**一致性核验结论**：同后端（均 CPU）下，80 张现场图、32 个判定帧，
**分数与阈值 32/32 完全相同**。

**⚠️ CPU 与 GPU 会有细微差别**：同一批图，C#(CPU) 与 Python(GPU) 有 6/32 帧不同
（如 Cam2 阈值 47.82 ↔ 44.23、某帧分数 8 ↔ 6）。这是 Transformer 前向的浮点差异
在 P99 边界翻动了个别 patch，**两边都不算错**；但意味着基准里那组 94% 召回 / 5% 误报
是 CPU 上测的，**开 GPU 后会略有偏移**。

## 依赖的模型文件（不在本仓库）

需要 `dinov2_vitb14_l8_448x588.onnx`（约 330 MB，超出 GitHub 单文件限制，故未入库）。
用本仓库的脚本自行导出：

```bash
python python/export_dinov2_onnx.py
```

导出后用 `python/verify_onnx_pipeline.py` 核对 ONNX 与 PyTorch 前向一致，再放到
部署目录的 `testdll/` 下。

## 性能实测（RTX 4060 Laptop / 3200×1800 现场原图 / 7 个 tile）

| 后端 | 推理 | 其余（取图、特征拼接、比对、定位） | 合计 |
|---|---|---|---|
| GPU (CUDA EP) | 368 ms | 45 ms | **≈ 0.4 s** |
| CPU | 4.7 s | 45 ms | **≈ 5 s** |

非推理部分已做 SIMD 点积 + `Parallel.For`（只用一半核心，另一半留给原有白纱检测主循环），
占比 < 10%。**因此是否真的挂上 CUDA EP 直接决定 10 倍差距**，
装完务必用 `YarnDectionSys.exe --selftest` 确认后端。

工程侧优化：非推理开销从 5 s 降到 45 ms；6 路复访 60 s → 18 s；6 路内存峰值 5.76 GB
（每路缓存 8 帧参考特征，单帧 171 MB）。

## 已知问题

- **持续性缺陷会被参考缓冲吸收**：缺陷若长时间不变，几帧之后它自己就进了参考池，
  异常度归零。这是"跨帧参考"的结构性缺陷，不是 bug。
- **LOO 定阈对个别帧敏感**：MAD 会跳档，个别相机的阈值可能在两次标定间跳动。
- **一切都未在真实产线验证**，包括卡顿修复。上线请按 `docs/现场部署手册.md` 第五节分阶段。
