"""光照压力测试的无模型确定性回归。

@spec docs/spec.md#4.7
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import illumination_stress_eval as I


def check(name: str, ok: bool) -> None:
    if not ok:
        raise AssertionError(name)
    print("PASS", name)


def main() -> None:
    ramp = np.tile(np.arange(256, dtype=np.uint8), (32, 1))
    img = np.dstack([ramp, ramp, ramp])
    check("EV0 恒等", np.array_equal(I.exposure_ev(img, 0.0), img))
    check("Gamma1 恒等", np.array_equal(I.gamma_srgb(img, 1.0), img))
    check("WB0 恒等", np.array_equal(I.white_balance(img, 0.0), img))
    check("曝光单调", I.exposure_ev(img, 1.0).mean() > img.mean() >
          I.exposure_ev(img, -1.0).mean())
    check("Gamma单调", I.gamma_srgb(img, 0.8).mean() > img.mean() >
          I.gamma_srgb(img, 1.25).mean())
    warm = I.white_balance(img, 0.2).astype(float)
    check("暖偏移R增B减", warm[..., 2].mean() > warm[..., 0].mean())
    check("阴影变暗", I.local_shadow(img, 0.5).mean() < img.mean())
    check("闪烁改变且尺寸不变", I.stripe_flicker(img, 0.2).shape == img.shape and
          not np.array_equal(I.stripe_flicker(img, 0.2), img))
    check("高光变亮", I.local_highlight(img, 0.5).mean() > img.mean())
    a = I.conditions("ev")
    check("EV范围", a[0].value == I.EV_MIN and a[-1].value == I.EV_MAX)
    check("EV步长", all(abs((b.value - a_.value) - I.EV_STEP) < 1e-9
                          for a_, b in zip(a, a[1:])))
    check("固定条件可复现", np.array_equal(I.local_shadow(img, 0.35),
                                           I.local_shadow(img, 0.35)))
    lo, hi = I.wilson(5, 10)
    check("Wilson区间", 0.0 < lo < 0.5 < hi < 1.0)
    print("photometric transform tests: ALL PASS")


if __name__ == "__main__":
    main()
