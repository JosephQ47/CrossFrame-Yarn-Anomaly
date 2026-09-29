# 人手入侵与纱线缺陷双状态门控验收报告

- 被测版本：工作树（未提交）
- 被测范围：`src/hand_gate.py`、`src/deploy_black_detector.py`、`csharp/HandIntrusionGate.cs`、`csharp/BlackYarnDetector.cs`
- 结论：状态机、参考池门控、规格契约、外部产线接线和 Debug/Release x64 构建均已通过；现场影子模式仍待执行。

## 自动化验收

| 验收项 | 方法 | 实测 | 结论 |
|---|---|---:|---|
| 人手帧不判纱线、不入池 | Python `test_gated_frames_do_not_advance_reference_count` | 通过 | ✅ |
| 两帧无人手恢复 | Python/C# 共用 `data/hand_gate_sequence.csv` | Python 8 tests；C# 25 assertions | ✅ |
| 外部产线门控类 | 独立编译 `ProductionHandGateTests.csproj` | 7 assertions | ✅ |
| 产线 Debug/x64 | Visual Studio 2022 MSBuild Rebuild | 生成 `YarnDectionSys.exe` | ✅ |
| 产线 Release/x64 | Visual Studio 2022 MSBuild Rebuild | 生成 `YarnDectionSys.exe` | ✅ |
| 人手模型配置 | `System.ParamConfig.xml` 与 Release/testdll 核对 | `DetectPerson=true`，模型为 `persons.onnx` 且文件存在 | ✅ |
| 恢复期间重新入侵清零 | Python/C# 状态序列 | 通过 | ✅ |
| 每路相机独立 | Python 单测 | 通过 | ✅ |
| C# 两参数兼容重载 | `spec_check.py` | 通过 | ✅ |
| 规格常量与实现一致 | `spec_check.py` | 23/23 PASS | ✅ |
| 产线人手 DLL 小样本回放 | 186 张人工清单，实际 `DetectPerson.dll` | TP=5、FP=0、TN=181、FN=0 | ✅（仅小样本） |
| 原生运行依赖 | Debug/Release 输出核对 | `opencv_world480.dll`、`DetectPerson.dll`、`persons.onnx` 均随构建复制 | ✅ |

## 尚未完成

| 项 | 阻塞原因 | 解锁条件 |
|---|---|---|
| 现场影子模式与误报/召回统计 | 需要现场相机流和既有人手事件 | 连续运行 1–2 个班次，确认门控期间纱线误报为零且无人手缺陷召回不下降 |

## 复现

```powershell
python tools/test_hand_gate.py
dotnet run --project csharp/tests/HandIntrusionGateTests.csproj
dotnet run --project csharp/tests/ProductionHandGateTests.csproj
python tools/spec_check.py --root .
# Visual Studio MSBuild
MSBuild.exe 浆纱检测系统.sln /t:Rebuild /p:Configuration=Release /p:Platform=x64
```

证据：`dev_log/hand-intrusion-gate_evidence/`、`reports/hand_gate/production_dll_roi_results.csv`。
