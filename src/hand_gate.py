"""人手入侵门控状态机。

@spec docs/spec.md#4.2
@spec docs/spec.md#4.3
@spec docs/spec.md#4.4
@spec docs/spec.md#4.5
"""
from dataclasses import dataclass
from enum import Enum
import csv
from pathlib import Path


class DetectionState(str, Enum):
    NOT_READY = "NOT_READY"
    NORMAL = "NORMAL"
    HAND_INTRUSION = "HAND_INTRUSION"
    RECOVERING = "RECOVERING"
    YARN_DEFECT = "YARN_DEFECT"


@dataclass
class HandIntrusionGate:
    """每路相机独立持有一个实例；返回 None 表示本帧可进入纱线检测。"""

    clear_frames_required: int = 2
    active: bool = False
    clear_frames: int = 0

    def __post_init__(self):
        if self.clear_frames_required < 1:
            raise ValueError("clear_frames_required must be >= 1")

    def update(self, hand_intrusion: bool):
        if hand_intrusion:
            self.active = True
            self.clear_frames = 0
            return DetectionState.HAND_INTRUSION

        if not self.active:
            return None

        self.clear_frames += 1
        if self.clear_frames < self.clear_frames_required:
            return DetectionState.RECOVERING

        self.active = False
        self.clear_frames = 0
        return None


def load_hand_events(path):
    """读取 ``file,hand_intrusion`` CSV；键统一为文件名。"""
    if path is None:
        return {}
    truthy, falsy = {"1", "true", "yes", "y"}, {"0", "false", "no", "n", ""}
    events = {}
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"file", "hand_intrusion"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"人手事件 CSV 必须包含列 {sorted(required)}")
        for line_no, row in enumerate(reader, 2):
            key = Path(row["file"].strip()).name
            raw = row["hand_intrusion"].strip().lower()
            if not key:
                raise ValueError(f"人手事件 CSV 第 {line_no} 行 file 为空")
            if raw in truthy:
                value = True
            elif raw in falsy:
                value = False
            else:
                raise ValueError(f"人手事件 CSV 第 {line_no} 行 hand_intrusion={raw!r} 非法")
            if key in events and events[key] != value:
                raise ValueError(f"人手事件 CSV 对 {key!r} 给出了冲突值")
            events[key] = value
    return events
