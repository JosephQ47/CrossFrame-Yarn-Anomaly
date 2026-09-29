# 检测与定位指标统一评估报告

**日期**：2026-09-29  
**规格**：[`spec.md` §4.8](spec.md#48-检测与定位指标契约)  
**评估器**：`tools/evaluate_detection_metrics.py`

## 1. 评估口径

- 输入：光照实验逐帧结果中的 `original_ref / ev_+0.00`。
- 数据：满足至少8张正常参考的 Cam1/2/3/5/6；排除只有5张正常参考的 Cam4。
- 总计：62帧，48正常、14缺陷。
- GT：从 `D:/dataset/black_fabric/*.json` 的 LabelMe `detect` 人工框回填。
- 报警阈值：保持光照实验逐机位原阈值，不为本次指标重新调参。

## 2. 检测性能

混淆矩阵：`TP=13, FP=2, TN=46, FN=1`。

| 指标 | 结果 | Wilson 95% CI |
|---|---:|---:|
| 正常误报率 FPR | 2/48 = **4.17%** | 1.15%～13.98% |
| 缺陷召回率 Recall | 13/14 = **92.86%** | 68.53%～98.73% |
| 精确率 Precision | 13/15 = **86.67%** | 62.12%～96.26% |
| 图级 AUROC | **97.92%** | — |
| 每小时误报警 | **不可计算** | 无连续观测时长/固定帧周期 |

现有图片是离散筛选样本，不能用文件首末时间跨度当作连续监控时长。评估器在未提供 `--observation-hours` 或 `--frame-period-seconds` 时强制输出 `null`，避免制造虚假的“次/小时”。

## 3. 定位性能

| 指标 | 结果 |
|---|---:|
| 定位点进入人工框 | 13/14 = **92.86%** |
| 预测框最佳 IoU 均值 | **48.55%** |
| 预测框最佳 IoU 中位数 | **51.48%** |
| 框级召回 IoU≥0.1 | 13/14 = **92.86%** |
| 框级召回 IoU≥0.3 | 13/14 = **92.86%** |
| 框级召回 IoU≥0.5 | 9/14 = **64.29%** |
| 报警且定位点正确（缺陷分母） | 12/14 = **85.71%** |
| 正确定位报警占全部报警 | 12/15 = **80.00%** |

“点定位13/14”和“报警13/14”并非完全相同的13帧，所以联合成功只有12/14。这正是必须单独报告联合指标的原因。

## 4. 每小时误报的现场计算方式

正式现场需要以下任一输入：

1. 明确的连续观测时长，例如确认正常的影子模式运行8小时：`--observation-hours 8`；
2. 固定处理周期，例如每秒判定1帧：`--frame-period-seconds 1`。

计算式固定为：

```text
每小时误报警次数 = 正常帧报警数 FP / 连续观测小时数
```

若现场期间存在真实缺陷，必须先由人工或产线事件记录区分 TP 与 FP，不能把所有报警都算成误报。

## 5. 自动化验证与证据

- 指标公式测试：`tools/test_detection_metrics.py`，24项断言通过。
- 规格检查：23/23契约通过，0 FAIL。
- JSON结果：`dev_log/detection_metrics_evidence/current_baseline/metrics.json`。
- 逐帧GT与IoU：`dev_log/detection_metrics_evidence/current_baseline/metrics_per_frame.csv`。

复现命令：

```powershell
D:\anaconda3\envs\Yolov8\python.exe tools\evaluate_detection_metrics.py `
  dev_log\illumination_robustness_evidence\illumination_per_frame.csv `
  --dataset D:\dataset\black_fabric `
  --strategy original_ref --condition ev_+0.00 --exclude-cam Cam4 `
  --out dev_log\detection_metrics_evidence\current_baseline
```

