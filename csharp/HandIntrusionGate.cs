using System;

namespace YarnDection
{
    /// <summary>
    /// 每路相机独立的人手入侵恢复门控。
    /// 返回 null 表示当前帧可以继续进入纱线检测。
    /// @spec docs/spec.md#4.2
    /// @spec docs/spec.md#4.3
    /// @spec docs/spec.md#4.4
    /// @spec docs/spec.md#4.5
    /// </summary>
    public sealed class HandIntrusionGate
    {
        public const int DefaultClearFrames = 2;

        private readonly int clearFramesRequired;
        private bool active;
        private int clearFrames;

        public HandIntrusionGate(int clearFramesRequired = DefaultClearFrames)
        {
            if (clearFramesRequired < 1)
                throw new ArgumentOutOfRangeException(nameof(clearFramesRequired));
            this.clearFramesRequired = clearFramesRequired;
        }

        public BlackYarnDetector.DetectionState? Update(bool handIntrusion)
        {
            if (handIntrusion)
            {
                active = true;
                clearFrames = 0;
                return BlackYarnDetector.DetectionState.HandIntrusion;
            }

            if (!active) return null;

            clearFrames++;
            if (clearFrames < clearFramesRequired)
                return BlackYarnDetector.DetectionState.Recovering;

            active = false;
            clearFrames = 0;
            return null;
        }

        public void Reset()
        {
            active = false;
            clearFrames = 0;
        }
    }
}
