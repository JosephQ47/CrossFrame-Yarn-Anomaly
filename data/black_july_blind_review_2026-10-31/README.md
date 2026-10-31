# 七月原图双人盲审

A、B 两人分别填写自己的 CSV，不共享判断。仅看 `source_path` 原图，不看旧 JSON、模型热图、分数或对方表格。`defect_status` 选 `normal/black_yarn/other_defect/uncertain`；无法判断必须写 uncertain。缺陷写最小框 `[x0,y0,x1,y1]` 或中心线 `[[x,y],...]`，标记运行状态、手/遮挡及事件 ID。完成后保存原 CSV，核对 SHA，再由第三人处理分歧。不得用模型输出给争议帧定真值。F2 使用既有独立盲判模板，不混入本批。
