"""绘制光照泛化主结果图。

@spec docs/spec.md#4.7
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


LABELS = {
    "original_ref": "Original reference",
    "adjusted_reference": "Adjusted reference",
    "matched_recalibration": "Adjusted + recalibrated",
}
COLORS = {"original_ref": "#c0392b", "adjusted_reference": "#2471a3",
          "matched_recalibration": "#239b56"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("summary", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    with args.summary.open(encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ("value", "fpr", "recall", "auroc"):
            r[k] = float(r[k])
        for k in ("location_hits", "detected_and_located", "n_defect"):
            r[k] = int(r[k])

    fig, ax = plt.subplots(1, 2, figsize=(12.4, 4.6), constrained_layout=True)
    for strategy in LABELS:
        rr = sorted((r for r in rows if r["family"] == "exposure_ev" and
                     r["strategy"] == strategy), key=lambda x: x["value"])
        x = [r["value"] for r in rr]
        ax[0].plot(x, [100 * r["fpr"] for r in rr], marker="o", lw=2,
                   color=COLORS[strategy], label=LABELS[strategy])
        ax[1].plot(x, [r["detected_and_located"] for r in rr], marker="o", lw=2,
                   color=COLORS[strategy], label=LABELS[strategy])
    ax[0].axhline(5, color="#555", ls="--", lw=1.2, label="5% FPR limit")
    ax[0].set(title="Normal false alarms under exposure shift",
              xlabel="Exposure shift (EV)", ylabel="False positive rate (%)",
              xticks=np.arange(-1.5, 1.51, 0.5), ylim=(-1, 80))
    ax[0].grid(alpha=.25); ax[0].legend(fontsize=8)
    ax[1].axhline(12, color="#555", ls="--", lw=1.2,
                  label="Baseline detected & localized (12/14)")
    ax[1].set(title="Defects both alarmed and localized",
              xlabel="Exposure shift (EV)", ylabel="Count (out of 14)",
              xticks=np.arange(-1.5, 1.51, 0.5), ylim=(0, 14.6))
    ax[1].grid(alpha=.25); ax[1].legend(fontsize=8)
    fig.suptitle("Cross-frame yarn detector: synthetic illumination boundary\n"
                 "Five cameras with >=8 clean references; 48 normal / 14 defect frames",
                 fontsize=13)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
