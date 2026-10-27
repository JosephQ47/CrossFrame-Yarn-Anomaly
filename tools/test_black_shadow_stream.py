"""Boundary checks for causal manifest admission and shadow state transitions."""

import copy
import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from replay_black_shadow import csharp_score_and_locate, replay
from validate_black_stream_manifest import REQUIRED, old_hashes, validate


def sample_rows(*, calibration=3):
    base = datetime(2026, 10, 28, 8, tzinfo=timezone(timedelta(hours=8)))
    parts = ["warmup"] * 8 + ["normal_calibration"] * calibration + ["blind_stream"] * 3
    rows = []
    for i, part in enumerate(parts):
        rows.append(dict(zip(REQUIRED, (
            f"f{i}", f"C:/camera/f{i}.jpg", f"{i + 1:064x}", "Cam1",
            (base + timedelta(seconds=i)).isoformat(), "running", "20", "s1",
            part, "0", "" if part == "blind_stream" else "1"))))
    return rows


def fake_feature(row):
    i = int(row["frame_id"][1:])
    value = 100 if i == 11 else i
    return np.full((2, 3, 4), value, np.float32), (100, 100), 30, 60


def fake_score(feature, references, *_):
    return float(feature[0, 0, 0] - min(ref[0, 0, 0] for ref in references)), [1, 2, 3, 4]


def fake_m4(query, references, *_):
    assert all(ref["time"] < query["time"] and ref["cam"] == query["cam"]
               and ref["admitted_by_policy"] for ref in references)
    return float(query["feature"][0, 0, 0]), [2, 3, 4, 5], {references[0]["id"]: 6}


class ShadowStreamTests(unittest.TestCase):
    def run_stream(self, raw, calibration=3):
        parsed, _ = validate(raw, check_files=False, min_calibration=calibration, exclude_old=False)
        return replay(parsed, fake_feature, fake_score, fake_m4,
                      min_calibration=calibration)[0]

    def test_causal_thresholds_and_alarm_admission(self):
        out = self.run_stream(sample_rows())
        self.assertEqual(out[7]["p0"]["state"], "NOT_READY")
        self.assertEqual(out[8]["reference_ids"], [f"f{i}" for i in range(8)])
        self.assertEqual(len(out[8]["p0_loo_scores"]), 8)
        self.assertEqual(out[10]["p1"]["state"], "NOT_READY")
        self.assertIn(out[11]["p1"]["state"], {"NORMAL", "YARN_DEFECT"})
        self.assertIn(out[11]["p2"]["state"], {"NORMAL", "YARN_DEFECT"})
        self.assertTrue(out[11]["p0"]["alarm"])
        self.assertTrue(out[11]["admitted_to_pool"])
        self.assertIn("f11", out[12]["reference_ids"])
        self.assertEqual(out[11]["reference_count"], 8)

    def test_hand_and_recovery_never_enter_pool(self):
        rows = sample_rows()
        rows[11]["hand_model_result"] = "1"
        rows.insert(12, dict(rows[12], frame_id="hclear", sha256=f"{100:064x}",
                             captured_at=(datetime.fromisoformat(rows[11]["captured_at"])
                                          + timedelta(milliseconds=500)).isoformat()))
        out = self.run_stream(rows)
        self.assertEqual(out[11]["gate_state"], "HAND_INTRUSION")
        self.assertEqual(out[12]["gate_state"], "RECOVERING")
        self.assertFalse(out[11]["admitted_to_pool"])
        self.assertFalse(out[12]["admitted_to_pool"])
        self.assertEqual(out[13]["reference_ids"], [f"f{i}" for i in range(3, 11)])

    def test_bad_manifest_rejected(self):
        base = sample_rows()
        variants = []
        a = copy.deepcopy(base); a[-1]["operator_confirmed_normal"] = "1"; variants.append(a)
        a = copy.deepcopy(base); a[9]["sha256"] = a[8]["sha256"]; variants.append(a)
        a = copy.deepcopy(base); a[10]["partition"] = "warmup"; variants.append(a)
        a = copy.deepcopy(base); a[10]["speed_m_per_min"] = "nan"; variants.append(a)
        a = copy.deepcopy(base); a[10]["captured_at"] = a[9]["captured_at"]; variants.append(a)
        a = copy.deepcopy(base); a[10]["state_id"] = "stopped"; variants.append(a)
        for variant in variants:
            with self.subTest(variant=variant):
                with self.assertRaises(ValueError):
                    validate(variant, check_files=False, min_calibration=3, exclude_old=False)

    def test_existing_development_and_f2_hash_rejected(self):
        rows = sample_rows()
        rows[0]["sha256"] = next(iter(old_hashes()))
        with self.assertRaisesRegex(ValueError, "development/F2"):
            validate(rows, check_files=False, min_calibration=3, exclude_old=True)

    def test_replay_is_deterministic_except_timing(self):
        rows, _ = validate(sample_rows(), check_files=False, min_calibration=3, exclude_old=False)
        left, _ = replay(rows, fake_feature, fake_score, fake_m4, min_calibration=3)
        right, _ = replay(rows, fake_feature, fake_score, fake_m4, min_calibration=3)
        for items in (left, right):
            for item in items:
                item.pop("feature_ms")
                item.pop("scoring_ms")
        self.assertEqual(left, right)

    def test_csharp_component_tie_uses_first_and_scales_width(self):
        anomaly = np.zeros((20, 20), np.float32)
        anomaly[2:4, 3:5] = 1
        anomaly[14:16, 15:17] = 1
        score, box = csharp_score_and_locate(anomaly, (100, 200), 20, 80)
        self.assertEqual(score, 4)
        self.assertEqual(box, [30, 26, 50, 32])


if __name__ == "__main__":
    unittest.main()
