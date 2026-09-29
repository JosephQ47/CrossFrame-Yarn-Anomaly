param(
    [Parameter(Mandatory = $true)][string]$ReleaseDir,
    [Parameter(Mandatory = $true)][string]$LabelsCsv,
    [Parameter(Mandatory = $true)][string]$OutputCsv,
    [Parameter(Mandatory = $true)][string]$OutputJson,
    [switch]$UseFallbackCrops,
    [switch]$UseDarkFallbacks
)

$ErrorActionPreference = 'Stop'
$ReleaseDir = (Resolve-Path -LiteralPath $ReleaseDir).Path
$labels = Import-Csv -LiteralPath $LabelsCsv
$testDllDir = Join-Path $ReleaseDir 'testdll'
$nativeDir = Join-Path $ReleaseDir 'dll\x64'
$modelPath = Join-Path $testDllDir 'persons.onnx'
$personDll = Join-Path $testDllDir 'DetectPerson.dll'
$opencvWorld = Join-Path $ReleaseDir 'opencv_world480.dll'

foreach ($required in @($modelPath, $personDll, $opencvWorld, (Join-Path $ReleaseDir 'OpenCvSharp.dll'))) {
    if (-not (Test-Path -LiteralPath $required)) { throw "Missing runtime dependency: $required" }
}

$env:PATH = "$testDllDir;$nativeDir;$ReleaseDir;$env:PATH"
Add-Type -Path (Join-Path $ReleaseDir 'OpenCvSharp.dll')
if (-not ('ProductionHandNative' -as [type])) {
    Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
public static class ProductionHandNative {
    [DllImport("DetectPerson.dll", CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
    public static extern IntPtr CreateMyClass();
    [DllImport("DetectPerson.dll", CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
    public static extern void DestroyMyClass(IntPtr instance);
    [DllImport("DetectPerson.dll", CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
    public static extern bool LoadModel(IntPtr instance, string path, bool cuda);
    [DllImport("DetectPerson.dll", CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
    public static extern int ModelInference(IntPtr instance, IntPtr image, bool drawImage);
}
'@
}

$detector = [ProductionHandNative]::CreateMyClass()
if ($detector -eq [IntPtr]::Zero) { throw 'CreateMyClass returned null' }
$results = @()
try {
    if (-not [ProductionHandNative]::LoadModel($detector, $modelPath, $false)) {
        throw 'LoadModel returned false'
    }
    foreach ($row in $labels) {
        $path = (Resolve-Path -LiteralPath $row.file).Path
        $image = [OpenCvSharp.Cv2]::ImRead($path)
        try {
            if ($image.Empty()) { throw "Cannot read image: $path" }
            $raw = [ProductionHandNative]::ModelInference($detector, $image.CvPtr, $false)
            $fallbackView = ''
            $inferenceCount = 1
            if ($raw -lt 1 -and $UseDarkFallbacks) {
                foreach ($alpha in @(0.80, 0.65, 0.50)) {
                    $dark = [OpenCvSharp.Mat]::new()
                    try {
                        $image.ConvertTo($dark, [OpenCvSharp.MatType]::CV_8UC3, $alpha, 0)
                        $darkResult = [ProductionHandNative]::ModelInference($detector, $dark.CvPtr, $false)
                        $inferenceCount++
                        if ($darkResult -ge 1) {
                            $raw = $darkResult
                            $fallbackView = "dark_$alpha"
                            break
                        }
                    }
                    finally { $dark.Dispose() }
                }
            }
            if ($raw -lt 1 -and $UseFallbackCrops) {
                $views = @(
                    @{ Name = 'top_left'; X1 = 0.00; X2 = 0.55 },
                    @{ Name = 'top_center'; X1 = 0.225; X2 = 0.775 },
                    @{ Name = 'top_right'; X1 = 0.45; X2 = 1.00 }
                )
                foreach ($view in $views) {
                    $rect = [OpenCvSharp.Rect]::new(
                        [int]($view.X1 * $image.Width), 0,
                        [int](($view.X2 - $view.X1) * $image.Width),
                        [int](0.70 * $image.Height))
                    $crop = [OpenCvSharp.Mat]::new($image, $rect)
                    try {
                        $cropResult = [ProductionHandNative]::ModelInference($detector, $crop.CvPtr, $false)
                        $inferenceCount++
                        if ($cropResult -ge 1) {
                            $raw = $cropResult
                            $fallbackView = $view.Name
                            break
                        }
                    }
                    finally { $crop.Dispose() }
                }
            }
            $truth = $row.hand_intrusion.Trim().ToLowerInvariant() -in @('1', 'true', 'yes', 'y')
            $predicted = $raw -ge 1
            $results += [pscustomobject]@{
                file = $path
                ground_truth = $truth
                predicted = $predicted
                raw_result = $raw
                fallback_view = $fallbackView
                inference_count = $inferenceCount
            }
        }
        finally { $image.Dispose() }
    }
}
finally { [ProductionHandNative]::DestroyMyClass($detector) }

$outputParent = Split-Path -Parent $OutputCsv
if ($outputParent) { New-Item -ItemType Directory -Force -Path $outputParent | Out-Null }
$results | Export-Csv -LiteralPath $OutputCsv -NoTypeInformation -Encoding UTF8
$tp = @($results | Where-Object { $_.ground_truth -and $_.predicted }).Count
$fp = @($results | Where-Object { -not $_.ground_truth -and $_.predicted }).Count
$tn = @($results | Where-Object { -not $_.ground_truth -and -not $_.predicted }).Count
$fn = @($results | Where-Object { $_.ground_truth -and -not $_.predicted }).Count
function Safe-Divide([double]$Numerator, [double]$Denominator) {
    if ($Denominator -eq 0) { return $null }
    return $Numerator / $Denominator
}
$precision = Safe-Divide $tp ($tp + $fp)
$recall = Safe-Divide $tp ($tp + $fn)
$summary = [ordered]@{
    implementation = 'production DetectPerson.dll'
    fallback_crops = [bool]$UseFallbackCrops
    dark_fallbacks = [bool]$UseDarkFallbacks
    model = $modelPath
    images = $results.Count
    tp = $tp; fp = $fp; tn = $tn; fn = $fn
    recall = $recall
    false_positive_rate = Safe-Divide $fp ($fp + $tn)
    precision = $precision
    f1 = if (($precision + $recall) -gt 0) { 2 * $precision * $recall / ($precision + $recall) } else { $null }
}
$summary | ConvertTo-Json | Set-Content -LiteralPath $OutputJson -Encoding UTF8
$summary | ConvertTo-Json
