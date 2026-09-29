using System;
using System.IO;

namespace YarnDection
{
    // 最小替身让纯状态机测试不依赖 ONNX Runtime / OpenCvSharp。
    public class BlackYarnDetector
    {
        public enum DetectionState
        {
            NotReady,
            Normal,
            HandIntrusion,
            Recovering,
            YarnDefect
        }
    }

    internal static class Program
    {
        private static int assertions;

        private static void Equal(object expected, object actual, string name)
        {
            assertions++;
            if (!Equals(expected, actual))
                throw new Exception(name + ": expected=" + expected + ", actual=" + actual);
        }

        private static void Main(string[] args)
        {
            var gate = new HandIntrusionGate();
            Equal(null, gate.Update(false), "legacy path");
            Equal(BlackYarnDetector.DetectionState.HandIntrusion, gate.Update(true), "hand");
            Equal(BlackYarnDetector.DetectionState.Recovering, gate.Update(false), "first clear");
            Equal(null, gate.Update(false), "second clear resumes");

            Equal(BlackYarnDetector.DetectionState.HandIntrusion, gate.Update(true), "second hand");
            Equal(BlackYarnDetector.DetectionState.Recovering, gate.Update(false), "second recovery");
            Equal(BlackYarnDetector.DetectionState.HandIntrusion, gate.Update(true), "reentry resets");
            Equal(BlackYarnDetector.DetectionState.Recovering, gate.Update(false), "reentry first clear");
            Equal(null, gate.Update(false), "reentry second clear");

            gate.Update(true);
            gate.Reset();
            Equal(null, gate.Update(false), "reset");

            string sequencePath = args.Length > 0 ? args[0] : "data/hand_gate_sequence.csv";
            gate.Reset();
            int allowed = 0;
            string[] lines = File.ReadAllLines(sequencePath);
            for (int i = 1; i < lines.Length; i++)
            {
                if (String.IsNullOrWhiteSpace(lines[i])) continue;
                string[] fields = lines[i].Split(',');
                BlackYarnDetector.DetectionState? state = gate.Update(fields[1] == "1");
                string actualName = state.HasValue ? ToContractName(state.Value) : "NONE";
                Equal(fields[2], actualName, "shared sequence step " + fields[0]);
                bool allowYarn = !state.HasValue;
                Equal(fields[3] == "1", allowYarn, "shared allow step " + fields[0]);
                if (allowYarn) allowed++;
            }
            Equal(3, allowed, "shared allowed count");

            Console.WriteLine("PASS: " + assertions + " C# hand-gate assertions");
        }

        private static string ToContractName(BlackYarnDetector.DetectionState state)
        {
            switch (state)
            {
                case BlackYarnDetector.DetectionState.HandIntrusion: return "HAND_INTRUSION";
                case BlackYarnDetector.DetectionState.Recovering: return "RECOVERING";
                default: return state.ToString().ToUpperInvariant();
            }
        }
    }
}
