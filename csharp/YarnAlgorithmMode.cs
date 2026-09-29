// @spec docs/spec.md#4.9
using System;

namespace YarnDection
{
    public enum YarnAlgorithmMode
    {
        WhiteSupervised,
        BlackCrossFrame
    }

    public static class YarnAlgorithmModeParser
    {
        public static YarnAlgorithmMode ParseWithLegacy(string configuredMode, bool legacyBlackYarnDetect)
        {
            if (!string.IsNullOrWhiteSpace(configuredMode))
            {
                YarnAlgorithmMode parsed;
                if (Enum.TryParse(configuredMode.Trim(), false, out parsed) &&
                    Enum.IsDefined(typeof(YarnAlgorithmMode), parsed))
                    return parsed;
                return YarnAlgorithmMode.WhiteSupervised;
            }

            return legacyBlackYarnDetect
                ? YarnAlgorithmMode.BlackCrossFrame
                : YarnAlgorithmMode.WhiteSupervised;
        }
    }

    public sealed class YarnAlgorithmRoute
    {
        public YarnAlgorithmRoute(YarnAlgorithmMode mode) { Mode = mode; }
        public YarnAlgorithmMode Mode { get; private set; }
        public bool RunWhite { get { return Mode == YarnAlgorithmMode.WhiteSupervised; } }
        public bool RunBlack { get { return Mode == YarnAlgorithmMode.BlackCrossFrame; } }
    }
}
