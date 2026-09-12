# src/ — 脚本索引

本目录是全部 Python 代码：检测器本体 + 复现全部实验与配图的脚本。
按用途分五组，组内按"先跑谁"排序。

依赖安装：

```bash
pip install -r src/requirements.txt
```

所有脚本都从仓库根目录以 `python src/xxx.py` 的方式运行；
脚本内部用 `Path(__file__).parent.parent` 定位 `data/` 与 `assets/`，
互相之间以同目录平铺方式 import（`import black_yarn_detector as D`），
**不要把本目录再分子目录**，否则这些相对路径全部失效。

---

## 1. 核心（其余脚本都依赖它）

| 脚本 | 作用 |
|---|---|
| `black_yarn_detector.py` | 检测器本体。DINOv2 ViT-B/14 最后 8 层拼接 + 固定机位跨帧同位置参考，零训练、零缺陷样本、不用标签。基准评估走留一法参考，另出叠加图与 HTML 画廊。所有实验脚本都 import 它的特征与评分函数。 |
| `figstyle.py` | 期刊风格 matplotlib 配置（Okabe–Ito 色盲友好配色，中文字体优先微软雅黑）。无副作用，供各出图脚本 import。 |

## 2. 评测（出论文成绩）

| 脚本 | 作用 |
|---|---|
| `metrics_full.py` | 完整指标一次跑全：图级 AUROC / AP / F1-max / 产线定阈 P-R-F1 / TPR@FPR5%，定位侧点命中率 / 像素 AUROC / PRO / 框 IoU。输出 `data/metrics_黑纱基准.json` 与 `data/scores_per_frame.csv`，是 `make_figs.py` 的输入。 |
| `sota_baselines.py` | PaDiM / PatchCore / SimpleNet / PatchCore-DINOv2 同口径基线（含换评分规则的变体）。输出 `data/sota_baselines.csv`、`data/sota_baselines_scores.json`。 |
| `eval_black_benchmark.py` | 用 labelme GT 评估任意 YOLO 权重在同一基准上的 detect+KC 真实 P/R。禁训，仅评估。 |

## 3. 实验（每个脚本对应 README 里的一节结论）

| 脚本 | 对应结论 |
|---|---|
| `mech_exp.py` | 机理量化：分布收窄 / 参考帧数曲线 / 抖动与邻域松弛 / 白纱 27 帧回归。→ `data/mech_results_black.json`、`mech_results_white.json`，README §3.5 |
| `f1_crossframe_eval.py` | 红纱 F1 负结果复现（方法的边界）。→ `data/f1_crossframe_红纱负结果.json`，README §四 |
| `f1_repair_exp.py` | 红纱修复第一轮：干净参考池 / 绝对 patch 预算 / 异常图面积屏蔽 五臂。→ `data/repair_round1_F1修复.csv` |
| `f1_repair_exp2.py` | 第二轮：LAB 中值背景差分的外观屏蔽 + 黑纱回归。3/15→10/15，黑纱 16/16。→ `data/repair_round2_外观屏蔽.csv` |
| `p1_threshold_guarantee.py` | 有统计保证的定阈：保形 p 值 / 极值理论 POT / MAD / P95 四规则 × 两种校准。→ `data/p1_results.json`、`p1_summary.csv` |
| `p1_variants.py` | 定阈敏感性：四口径 + 跨机位时刻聚合，说明保形 5% 的召回只掉在一帧上。→ `data/p1_variants.json / .csv` |
| `dump_scores_maskon.py` | 导出开/关外观屏蔽的逐帧分数对照，供定阈分析用。→ `data/scores_maskon.csv` |
| `review_suspect_frames.py` | 生成高分正常帧的人工复核材料（原图 / 热图 / 放大 / 邻帧对照）。只出图，不改任何标注。 |

## 4. 出图（生成 assets/ 下配图）

| 脚本 | 产物 |
|---|---|
| `make_figs.py` | `assets/fig1_基准指标.png`、`fig2_检出示例.jpg`、`fig3_红纱失败.jpg`、`fig4_消融.png`。依赖 `metrics_full.py` 的输出。 |
| `p1_figs.py` | `fig11_定阈校准曲线.png`、`fig13_分数分布与阈值.png`、`fig14_定阈样本量.png`、`fig15_基线保形召回.png`。依赖 `p1_threshold_guarantee.py` 的输出。 |
| `p1_figs_variants.py` | `fig12_定阈敏感性与跨机位.png`。依赖 `p1_variants.py` 的输出。 |
| `make_mech_figs.py` | 机理量化图与白纱回归图（写出时叫 `机理实验_结果.png` / `白纱回归.png`，入库后重命名为 `fig9_机理量化.png` / `fig10_白纱回归.png`）。依赖 `mech_exp.py` 的输出。 |

⚠ `fig5_修复实验.png`、`fig6_黑纱Cam3_修复前后.jpg`、`fig7_F1_修复前后.jpg`、`fig8_SOTA对比.png`
目前**没有以该文件名写出它们的脚本**：它们由 `f1_repair_exp*.py` / `sota_baselines.py` 跑进
gitignore 掉的 `outputs/` 后手工挑选重命名入库。`fig16`/`fig17` 同理，来自
`review_suspect_frames.py` 写出的 `assets/待复核帧/`。要完全一键复现这几张，需要补出图脚本。

## 5. 部署与导出（C# 产线侧）

| 脚本 | 作用 |
|---|---|
| `deploy_black_detector.py` | 现场部署版：参考只用**历史帧**（因果滚动缓冲），warm-up ≥8 帧才判定，阈值按相机自校准取 P95。与基准评估的留一法不同，这是产线真实流程。 |
| `export_dinov2_onnx.py` | 导出特征骨干为 ONNX 给 C#/onnxruntime。输入 (1,3,448,588)，输出 (1,1344,6144) 已 L2 归一化。 |
| `verify_onnx_pipeline.py` | 整条流水线一致性核验（切 tile / 归一化 / 特征拼接 / 连通域 / 自校准）。**工厂部署前必跑**——任何一处写错都是静默失效，不报错、只是不报警。 |

---

运行顺序参考：核心 → 评测 → 实验 → 出图。
部署侧（第 5 组）独立于前四组，见 [../docs/现场部署手册.md](../docs/现场部署手册.md) 与 [../csharp/README.md](../csharp/README.md)。
