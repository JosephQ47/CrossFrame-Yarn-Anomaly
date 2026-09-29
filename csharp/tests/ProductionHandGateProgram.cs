using System;
using YarnDection;

internal static class ProductionHandGateProgram
{
    private static int assertions;

    private static void Equal(object expected, object actual, string name)
    {
        assertions++;
        if (!Equals(expected, actual))
            throw new Exception(name + ": expected=" + expected + ", actual=" + actual);
    }

    private static void Main()
    {
        var gate = new HandIntrusionGate();
        Equal(null, gate.Update(false), "legacy");
        Equal(YarnDetectionState.HandIntrusion, gate.Update(true), "hand");
        Equal(YarnDetectionState.Recovering, gate.Update(false), "first clear");
        Equal(null, gate.Update(false), "second clear");
        Equal(YarnDetectionState.HandIntrusion, gate.Update(true), "reentry");
        Equal(YarnDetectionState.Recovering, gate.Update(false), "reentry clear");
        Equal(YarnDetectionState.HandIntrusion, gate.Update(true), "reentry reset");
        Console.WriteLine("PASS: " + assertions + " production gate assertions");
    }
}
