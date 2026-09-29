"""从逐帧证据生成满足最小参考数的光照泛化主报告。

@spec docs/spec.md#4.7
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import illumination_stress_eval as I


def load(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            for k in ("value", "score", "threshold", "dark_frac", "saturated_frac"):
                r[k] = float(r[k])
            for k in ("defect", "hit", "alarm"):
                r[k] = int(r[k])
            rows.append(r)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    rows = [r for p in args.inputs for r in load(p)]

    base = [r for r in rows if r["strategy"] == "original_ref" and
            r["condition"] == "ev_+0.00"]
    normal_files = defaultdict(set)
    for r in base:
        if not r["defect"]:
            normal_files[r["cam"]].add(r["file"])
    valid_cams = sorted(c for c, fs in normal_files.items() if len(fs) >= I.MIN_CLEAN_REFS)
    excluded = sorted(set(normal_files) - set(valid_cams))
    primary = [r for r in rows if r["cam"] in valid_cams]
    base_p = [r for r in primary if r["strategy"] == "original_ref" and
              r["condition"] == "ev_+0.00"]
    base_recall = sum(r["alarm"] for r in base_p if r["defect"])

    groups = defaultdict(list)
    for r in primary:
        groups[(r["strategy"], r["family"], r["condition"], r["value"])].append(r)
    out = []
    for (strategy, family, condition, value), grp in groups.items():
        n = [r for r in grp if not r["defect"]]
        d = [r for r in grp if r["defect"]]
        fp = sum(r["alarm"] for r in n); tp = sum(r["alarm"] for r in d)
        loc = sum(r["hit"] for r in d)
        tploc = sum(r["alarm"] and r["hit"] for r in d)
        loc_min = math.ceil((I.SAFE_LOCATION_HITS / 16.0) * len(d))
        fci = I.wilson(fp, len(n)); rci = I.wilson(tp, len(d))
        strict = (fp / len(n) <= I.SAFE_FPR and
                  tp >= base_recall - I.SAFE_RECALL_DROP and loc >= loc_min)
        out.append(dict(
            strategy=strategy, family=family, condition=condition, value=value,
            n_normal=len(n), false_alarms=fp, fpr=fp / len(n),
            fpr_ci_low=fci[0], fpr_ci_high=fci[1],
            n_defect=len(d), true_positives=tp, recall=tp / len(d),
            recall_ci_low=rci[0], recall_ci_high=rci[1],
            location_hits=loc, location_hits_min=loc_min,
            detected_and_located=tploc,
            auroc=I.auc_rank([r["defect"] for r in grp], [r["score"] for r in grp]),
            dark_frac_mean=sum(r["dark_frac"] for r in grp) / len(grp),
            saturated_frac_mean=sum(r["saturated_frac"] for r in grp) / len(grp),
            strict_safe=int(strict),
        ))
    out.sort(key=lambda r: (r["family"], r["strategy"], r["value"]))
    args.out.mkdir(parents=True, exist_ok=True)
    I.write_csv(args.out / "illumination_primary_summary.csv", out)

    boundaries = {}
    for strategy in ("original_ref", "adjusted_reference", "matched_recalibration"):
        ev = {round(r["value"], 8): bool(r["strict_safe"]) for r in out
              if r["family"] == "exposure_ev" and r["strategy"] == strategy}
        contiguous = []
        if ev.get(0.0, False):
            contiguous = [0.0]
            v = -I.EV_STEP
            while v >= I.EV_MIN - 1e-9 and ev.get(round(v, 8), False):
                contiguous.append(round(v, 8)); v -= I.EV_STEP
            v = I.EV_STEP
            while v <= I.EV_MAX + 1e-9 and ev.get(round(v, 8), False):
                contiguous.append(round(v, 8)); v += I.EV_STEP
            contiguous.sort()
        boundaries[strategy] = dict(
            contiguous_safe_values=contiguous,
            min_ev=min(contiguous) if contiguous else None,
            max_ev=max(contiguous) if contiguous else None,
            non_contiguous_safe_values=sorted(v for v, ok in ev.items() if ok),
        )
    report = dict(
        valid_cameras=valid_cams, excluded_underfilled=excluded,
        normal_reference_counts={c: len(normal_files[c]) for c in sorted(normal_files)},
        primary_frames=len({r["file"] for r in base_p}),
        primary_normal=sum(not r["defect"] for r in base_p),
        primary_defect=sum(r["defect"] for r in base_p),
        baseline_true_positives=base_recall,
        exposure_boundaries=boundaries,
        family_failure_counts={
            f"{fam}/{st}": dict(Counter("safe" if r["strict_safe"] else "fail" for r in out
                                        if r["family"] == fam and r["strategy"] == st))
            for fam in sorted(set(r["family"] for r in out))
            for st in sorted(set(r["strategy"] for r in out))
        },
        caveat="合成边界只反映光度变换敏感性；真实照度、色温和空间光场必须现场复验。",
    )
    (args.out / "illumination_primary_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
