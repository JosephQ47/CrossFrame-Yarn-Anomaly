"""Regression: deployment scoring obeys the M0 area cap on a hand-sized blob."""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import deploy_black_detector as deploy  # noqa: E402


def main():
    anomaly = np.full((32, 228), .1, np.float32)
    anomaly[1:7, 1:11] = .9  # 60 patch intrusion; exceeds the contracted cap 55
    anomaly[12:14, 30:34] = .8  # 8 patch defect candidate
    for i in range(5):
        anomaly[20, 60 + i*10] = .7
    cosine = 1 - anomaly
    current = np.stack((cosine, np.sqrt(1 - cosine*cosine)), axis=-1)
    reference = np.zeros_like(current)
    reference[..., 0] = 1
    score, box, heat = deploy.analyze(current, [reference], (1800, 3200), 612, 1080)
    assert score == 8, (score, box)
    assert box is not None and box[0] > 300, box
    assert heat.shape == (1800, 3200)
    print("deployment area-cap scoring passed: 60-patch intrusion ignored, 8-patch target selected")


if __name__ == "__main__":
    main()
