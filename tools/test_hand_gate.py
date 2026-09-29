#!/usr/bin/env python3
"""人手门控的确定性回归，不加载 DINOv2。

@spec docs/spec.md#4.2
@spec docs/spec.md#4.3
@spec docs/spec.md#4.4
@spec docs/spec.md#4.5
"""
from collections import defaultdict
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from hand_gate import DetectionState, HandIntrusionGate, load_hand_events


class HandGateTests(unittest.TestCase):
    def test_no_hand_preserves_legacy_path(self):
        gate = HandIntrusionGate(2)
        self.assertEqual([gate.update(False) for _ in range(4)], [None] * 4)

    def test_two_clear_frames_resume(self):
        gate = HandIntrusionGate(2)
        self.assertEqual(gate.update(True), DetectionState.HAND_INTRUSION)
        self.assertEqual(gate.update(False), DetectionState.RECOVERING)
        self.assertIsNone(gate.update(False))
        self.assertIsNone(gate.update(False))

    def test_reentry_resets_recovery(self):
        gate = HandIntrusionGate(2)
        states = [gate.update(v) for v in (True, False, True, False, False)]
        self.assertEqual(states, [
            DetectionState.HAND_INTRUSION,
            DetectionState.RECOVERING,
            DetectionState.HAND_INTRUSION,
            DetectionState.RECOVERING,
            None,
        ])

    def test_gated_frames_do_not_advance_reference_count(self):
        gate = HandIntrusionGate(2)
        reference_count = 0
        states = []
        for hand in (False, True, False, False, False):
            state = gate.update(hand)
            states.append(state)
            if state is None:
                reference_count += 1
        self.assertEqual(reference_count, 3)
        self.assertEqual(states[1], DetectionState.HAND_INTRUSION)
        self.assertEqual(states[2], DetectionState.RECOVERING)

    def test_cameras_are_independent(self):
        gates = defaultdict(lambda: HandIntrusionGate(2))
        self.assertEqual(gates["Cam1"].update(True), DetectionState.HAND_INTRUSION)
        self.assertIsNone(gates["Cam2"].update(False))
        self.assertEqual(gates["Cam1"].update(False), DetectionState.RECOVERING)

    def test_event_csv(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "events.csv"
            path.write_text(
                "file,hand_intrusion\nfolder/a.jpg,1\nb.jpg,false\n",
                encoding="utf-8-sig",
            )
            self.assertEqual(load_hand_events(path), {"a.jpg": True, "b.jpg": False})

    def test_invalid_event_csv_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "events.csv"
            path.write_text("file,hand_intrusion\na.jpg,maybe\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_hand_events(path)

    def test_shared_python_csharp_sequence(self):
        gate = HandIntrusionGate(2)
        allowed = 0
        with (ROOT / "data" / "hand_gate_sequence.csv").open(encoding="utf-8") as f:
            import csv
            for row in csv.DictReader(f):
                actual = gate.update(row["hand_intrusion"] == "1")
                actual_name = "NONE" if actual is None else actual.value
                self.assertEqual(actual_name, row["expected_gate_state"], row["step"])
                self.assertEqual(actual is None, row["allow_yarn"] == "1", row["step"])
                allowed += int(actual is None)
        self.assertEqual(allowed, 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
