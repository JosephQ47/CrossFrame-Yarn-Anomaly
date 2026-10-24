# Transformer 差分特征接监督检测头：文献检索

检索日期：2026-10-23。范围：公开网络中的论文、出版社页面与作者代码。本文是针对架构可行性的检索记录，不是穷尽性系统综述；未运行复现实验。

## 结论

已有直接先例。尤其是 ChangeViT 和 BIT，支持“语义特征差分后接可训练卷积预测网络”。DPT 支持“Transformer token 转为空间特征图后接卷积解码器”；ViTDet 支持“ViT 骨干接检测结构”；织物领域已有 Swin Transformer 接 YOLOv5 颈部和预测头的工作。

这些证据不能直接证明它们优于本项目现有方法，也不能把“Transformer + CNN/YOLO”本身视为新颖贡献。

## 核心文献与证据定位

| 文献 | 年份/来源 | 与设想的对应关系 | 核验入口 |
|---|---|---|---|
| ChangeViT: Unleashing Plain Vision Transformers for Change Detection | 2024 预印本；另有 Pattern Recognition 扩展版本 | ViT-S 使用 DINOv2 预训练初始化；双时相特征与绝对差拼接，通过卷积及上采样生成变化图。另有细节分支和特征注入，不能描述成仅追加 CNN | [2024 原文 III-C、式 5–7](https://arxiv.org/html/2406.12847v1)、[作者代码](https://github.com/zhuduowang/ChangeViT) |
| Remote Sensing Image Change Detection with Transformers（BIT） | 2021 在线发表；TGRS 2022 卷期 | CNN 初始特征经 Transformer 增强后，做绝对差并送入小型 CNN。与“差分后学习判断”直接对应，但不是纯 ViT 骨干 | [论文](https://arxiv.org/abs/2103.00208)、[作者 networks.py](https://github.com/justchenhao/BIT_CD/blob/master/models/networks.py) |
| A Transformer-Based Siamese Network for Change Detection（ChangeFormer） | IGARSS 2022 | 孪生分层 Transformer、多尺度差异建模、轻量 MLP 解码器；相邻路线，不宜笼统称作纯 CNN 解码器 | [论文](https://arxiv.org/abs/2201.01293)、[PDF 表 1](https://arxiv.org/pdf/2201.01293)、[作者代码](https://github.com/wgcban/ChangeFormer) |
| Vision Transformers for Dense Prediction（DPT） | ICCV 2021 | 不同层 token 重组为空间特征图，再用卷积解码器输出密集预测 | [论文](https://arxiv.org/abs/2103.13413)、[作者全文](https://vladlen.info/papers/DPT.pdf) |
| Exploring Plain Vision Transformer Backbones for Object Detection（ViTDet） | ECCV 2022 | 普通 ViT 特征经简单特征金字塔接区域检测结构；并非 YOLO，也没有跨帧差分 | [论文](https://arxiv.org/abs/2203.16527)、[官方实现](https://github.com/facebookresearch/detectron2/tree/main/projects/ViTDet) |
| 基于 Swin Transformer 和 YOLOv5 的无纺布瑕疵检测，刘佳玮、曹江涛、姬晓飞 | 辽宁石油化工大学学报，2024，44(3):80–88 | Swin 主干接 YOLOv5 颈部和预测头，另含 CBAM、数据增强和 anchor 调整；直接对应纺织场景的 Transformer + YOLO | [中文期刊页](https://journal.lnpu.edu.cn/CN/10.12422/j.issn.1672-6952.2024.03.011)、[英文入口与数据表](https://journal.lnpu.edu.cn/EN/10.12422/j.issn.1672-6952.2024.03.011) |

补充文献：Lin 等，An Efficient and Intelligent Detection Method for Fabric Defects based on Improved YOLOv5，Sensors 23(1):97，2023 卷期、2022-12-22 在线发表，DOI 10.3390/s23010097。[全文](https://pmc.ncbi.nlm.nih.gov/articles/PMC9824629/)。涉及 Swin、BiFPN、小目标检测层及损失函数。摘要与正文部分表格的汇总数字存在差异，本记录不将其单个指标作为核心结论。

## 如何理解论文里的提升

- ChangeFormer 表 1 的 LEVIR-CD F1：BIT 89.31%，ChangeFormer 90.40%。这是两套变化检测架构的比较，并不能隔离“追加 CNN”这一因素。
- 中文无纺布论文报告其数据集上的 mAP@0.5：YOLOv5 83.26%，完整改进方法 95.76%；同时包含多项改动，不能把增益全归因于 Transformer。
- DPT、ViTDet 的任务和指标各异，不能把分割 IoU、检测 AP、图级 AUROC 当作同一种“准确率”横向排列。

上述指标仅转述论文，不是本项目复现值。具体来源见对应文献入口。

## 与本项目的对应

依据根目录 agent.md 与 AGENT_HANDOFF.md，当前主线为冻结 DINOv2 ViT-B/14、最后 8 层 patch 特征拼接、跨帧同位置参考、余弦异常度和连通域评分。这是文档记录，未在本次任务中重新核验实现或实验数字。

用户口述“叉分”暂按“差分”理解。当前做法主要将特征比较压缩成每个 patch 的一个异常分数；建议研究的新增模块则保留更多通道信息，用标注学习缺陷与正常变化的区别。

可验证的候选路线（研究建议，非已实现结果）：

当前帧与正常参考 → 共享冻结 DINOv2 → 同位置特征 F 与 R → 拼接 F、R、|F-R|，可附加原余弦距离图 → 1×1 通道压缩 → 小型卷积网络 → 缺陷掩膜；若只需要矩形框，则改接经过尺度适配的检测颈部和 YOLO 风格检测头。

这里保留的是 patch 网格，而非整图池化后的单一向量。普通 2D CNN 需要有空间意义的特征排列；将一个全局向量任意 reshape 并不能恢复原空间布局。1×1 卷积可理解为每个空间位置共享的通道全连接变换，但它本身不建模邻域。

DINOv2 patch 网格和多尺度 YOLO 输入之间需明确通道、步长、尺寸与原图坐标映射。不同 Transformer 深度的输出不自动等于不同空间尺度。若主要问题是边界细节缺失，仅追加预测头未必足够，可另验证浅层图像细节分支。

## 最小比较方案

1. 直接引用已有的当前无缺陷监督方法结果作为基线，不重跑。来源为 `CrossFrame-Yarn-Anomaly/data/metrics_黑纱基准.json` 与 `data/scores_per_frame.csv`；69 帧、16 缺陷、53 正常，图级 AUROC 98.2%、AP 93.7%、定位点命中 15/16。该批数据为冻结评测集，不能用于新增监督头训练或调参。需注明它是原有留一法参考的回顾性评估结果；若新方法采用不同参考时序或新的独立测试集，只能作历史参照，不能宣称严格同口径直接提升。
2. 相同特征和参考，增加逐 patch 线性预测头，判断标注监督本身的收益。
3. 相同输入，改为小 CNN，判断邻域信息的额外收益。
4. 再比较仅差分、当前特征加参考加差分、仅当前特征，以检验参考信息是否必要。
5. 若目标为框检测且有独立框标注训练集，再测试 YOLO 风格检测头。

像素级掩膜适合训练分割；框标注适合常规检测头；图级标签不能直接等价替代精确定位监督。按采集批次、时间段和缺陷事件划分数据，避免同一事件相邻帧跨训练/测试。既有冻结评测集继续只用于评测，新监督实验需要独立训练与验证数据。

重点观察固定误报水平下召回、定位 IoU 或检测 mAP、跨相机泛化及端到端延迟。不能仅凭训练集效果或图级 AUROC 判断精确定位改善。引入缺陷标注后属于监督扩展，应与原零缺陷标注设定分别报告。

## 尚未证实

- 本次未确认一篇与“冻结 DINOv2 最后 8 层 + 多帧同位置正常参考池 + 差分 + YOLO 头 + 浆纱机缺陷”所有细节完全一致的论文；这不等于不存在。
- 尚未证明新增监督头能改善本项目数据上的准确率、误报率或定位效果。
- 尚未确认有多少独立、可用于训练的标注缺陷事件，以及目标输出是框还是像素掩膜。
