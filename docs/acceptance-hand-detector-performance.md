# 产线人手门控检测性能（小样本验收）

## 结论

现役 `DetectPerson.dll + persons.onnx` 在当前可确认的 186 张不重复现场图上，针对“手/手臂进入筘齿或纱线检测带”的门控定义得到：Recall 100%、FPR 0%、Precision 100%、F1 100%。该结果只有 5 张有效入侵正样本，属于功能验收，不代表现场泛化上限。

## 数据与真值

- 总数：186 张；正样本 5 张，负样本 181 张。
- 来源：产线工程 `testImages`、现场黑纱原图、F2 标注原图、误报照片集。
- 真值：人工查看原图；只有手/手臂进入筘齿或纱线检测带才记为正样本。
- 画面顶边或最外侧仅露出少量手指、没有进入检测带的 5 张图记为负样本。这些图片现役 DLL 均未门控，符合本次定义。
- 测试直接调用发布目录的 `DetectPerson.dll`，不是用 Python ONNX 结果代替产线结果。

## 混淆矩阵与指标

| 指标 | 结果 |
|---|---:|
| TP / FP / TN / FN | 5 / 0 / 181 / 0 |
| Recall | 100% |
| 正常误触发率 FPR | 0% |
| Precision | 100% |
| F1 | 100% |

逐图结果：`reports/hand_gate/production_dll_roi_results.csv`；汇总：`reports/hand_gate/production_dll_roi_summary.json`；人工清单：`reports/hand_gate/manual_labels_roi_intrusion.csv`。

## 本轮优化

1. 产线调用顺序为先做人手检测，再做纱线检测；`HAND_INTRUSION` 和第一张 `RECOVERING` 帧不运行纱线推理、不报警、不更新参考池。
2. 发布工程补齐并固定复制人手 DLL 的原生依赖 `opencv_world480.dll`，Debug/x64 与 Release/x64 均可重建并产出完整运行文件。
3. 门控语义收紧为“手进入检测带”，避免把画面边缘、未遮挡纱线的手误算为本相机入侵。
4. 离线工具默认只统计 ONNX 的 `hand` 类；但现役 DLL 只返回布尔值，内部类别不可审计，因此正式指标以 DLL 实际输出为准。

## 边界与下一步

- 正样本来自两种明显入侵外观，人员、肤色、手套、距离、运动速度和早晚光照覆盖不足。
- 当前 DLL 不返回类别、置信度和框，不能证明它只由 `hand` 类触发；若镜头未来会拍到完整人员或人脸，需要增加这类负样本验证。
- 正式上线门槛仍是至少 1–2 个班次影子运行：记录每帧 `hand_intrusion/state`，人工复核全部门控事件与漏检事件。
- 建议现场验收集至少包含 100 次独立入侵事件，并覆盖六路相机、戴手套、快速挥动、只露部分手、强反光和暗光。样本扩充后再报告带 95% 置信区间的 Recall/FPR。

## 复现

```powershell
./tools/evaluate_production_hand_dll.ps1 `
  -ReleaseDir "D:\研究生课程\课题\添加按键拍照功能\纱浆检测\YarnDection\bin\x64\Release" `
  -LabelsCsv reports/hand_gate/manual_labels_roi_intrusion.csv `
  -OutputCsv reports/hand_gate/production_dll_roi_results.csv `
  -OutputJson reports/hand_gate/production_dll_roi_summary.json
```
