using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Linq;

namespace YarnDection
{
    /// <summary>
    /// 黑纱/深色纱缺陷检测器 —— DINOv2 语义特征(ONNX) + 固定机位跨帧同位置参考。
    ///
    /// 原理：筘齿区缺陷(絮状物糊住锯齿)在低层像素统计上与正常区不可分(实测)，
    /// 但在 DINOv2 语义特征上可分。固定机位下，每个 patch 位置的"正常参考"取自
    /// 同相机历史帧的同一位置特征；当前帧与参考的最大余弦相似度低 → 该位置异常。
    ///
    /// 黑纱评估基准成绩(16 真缺陷/55 正常，见课题仓库)：
    ///   定位命中 15/16(94%)、图级 AUROC 98.2、100%召回时误报 7%、
    ///   纯正常样本 P95 定阈：召回 94% / 误报 5%。
    /// 零训练、零缺陷样本、不依赖标签；阈值由每路相机的正常运行帧自校准。
    ///
    /// 工程形态：低频后台轮询。单帧实测(3200x1800 现场原图,7 个 tile)：
    ///   GPU  推理 368ms + 其余 45ms ≈ 0.4s      (RTX4060 Laptop)
    ///   CPU  推理 4.7s  + 其余 45ms ≈ 5s
    /// 非推理部分已做 SIMD + 并行优化(见 Dot/Parallel.For)，占比 <10%；
    /// 因此**是否挂上 CUDA EP 直接决定 10 倍差距**，务必用 --selftest 确认。
    /// </summary>
    public class BlackYarnDetector : IDisposable
    {
        // ==== 与基准完全一致的算法常量（勿随意改动，改动须回基准复测）====
        private const int TileW = 588, TileH = 448, Overlap = 140, PatchSize = 14;
        private const int Gh = TileH / PatchSize;          // 32 行 patch
        private const int FeatDim = 768 * 8;               // ViT-B 最后8层拼接 = 6144
        private const double BandY0 = 0.34, BandY1 = 0.60; // 筘齿带(占图高比例)
        private const int AreaCap = 55;                    // 异常块面积上限(patch数)，筛人手等大目标
        private const double KMad = 3.5;                   // 鲁棒定阈: 中位数 + K×MAD

        /// <summary>滚动参考缓冲帧数。实测 ≥8 帧达满性能，&lt;8 明显下降。</summary>
        public int BufSize { get; set; } = 8;

        /// <summary>逐 patch 的循环并行度。只用一半核心——另一半要留给原有的白纱检测主循环。</summary>
        private static readonly System.Threading.Tasks.ParallelOptions PllOpt =
            new System.Threading.Tasks.ParallelOptions
            { MaxDegreeOfParallelism = Math.Max(1, Environment.ProcessorCount / 2) };

        private static readonly float[] Mean = { 0.485f, 0.456f, 0.406f };
        private static readonly float[] Std = { 0.229f, 0.224f, 0.225f };

        private InferenceSession session;
        public event Action<string> DebugInform;

        /// <summary>分段耗时(ms)，仅用于性能诊断：预处理 / ONNX推理 / 特征归一化 / 余弦比对。</summary>
        public readonly double[] Profile = new double[4];

        /// <summary>CUDA EP 是否真正挂上（挂载失败会静默回退 CPU，速度差 10 倍以上）。</summary>
        public bool CudaEnabled { get; private set; }

        /// <summary>每路相机的独立状态：特征滚动缓冲 + 自校准阈值。</summary>
        private class CamState
        {
            public readonly List<float[]> Buffer = new List<float[]>(); // 每帧带特征(gh*cols*FeatDim 扁平)
            public int Cols;                                            // 特征列数 = 图宽/14
            public double Threshold = -1;                               // 自校准阈值(<0=未校准)
        }
        private readonly Dictionary<int, CamState> states = new Dictionary<int, CamState>();
        private readonly object stateLock = new object();

        public class Result
        {
            public bool Ready;        // 缓冲是否已满足判定条件
            public bool Alarm;
            public double Score;      // 图级分数 = 面积≤cap 的最大异常连通域面积
            public double Threshold;
            public Rect Box;          // 异常区域(原图坐标)
        }

        /// <param name="log">日志回调。必须在构造时传入——CUDA 是否启用是在构造函数里判定的，
        /// 事后再订阅 DebugInform 会漏掉这几条最关键的信息。</param>
        public BlackYarnDetector(string onnxPath, bool useCuda, Action<string> log = null)
        {
            if (log != null) DebugInform += log;
            var opt = new SessionOptions();
            opt.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
            if (useCuda)
            {
                try { opt.AppendExecutionProvider_CUDA(0); CudaEnabled = true; Log("BlackYarn: CUDA EP 已启用"); }
                catch (Exception ex) { Log("BlackYarn: CUDA 不可用，回退 CPU (" + ex.Message + ")"); }
            }
            session = new InferenceSession(onnxPath, opt);
            Log("BlackYarn: 模型加载完成 " + onnxPath);
        }

        private void Log(string s) { DebugInform?.Invoke(s); }

        /// <summary>
        /// 处理一帧：提特征 → 与该路历史参考比对 → 判定。帧内部不保留引用，调用方负责释放 frame。
        /// 未满 warm-up 时仅入缓冲（Ready=false）。
        /// </summary>
        public Result Process(int camIdx, Mat frame)
        {
            var r = new Result();
            if (frame == null || frame.Empty()) return r;

            int y0 = (int)(frame.Height * BandY0), y1 = (int)(frame.Height * BandY1);
            int cols; float[] feat = ExtractBandFeature(frame, out cols);
            if (feat == null) return r;

            CamState st;
            lock (stateLock)
            {
                if (!states.TryGetValue(camIdx, out st)) { st = new CamState { Cols = cols }; states[camIdx] = st; }
                if (st.Cols != cols) { st.Buffer.Clear(); st.Threshold = -1; st.Cols = cols; } // 分辨率变了,重建
            }

            lock (st)
            {
                if (st.Buffer.Count >= BufSize)
                {
                    // 异常图: 每 patch 与缓冲同位置的最大余弦相似度(特征已L2归一化,点积即余弦), top-1 抗参考污染
                    int nPatch = Gh * st.Cols;
                    float[] anom = new float[nPatch];
                    var swC = System.Diagnostics.Stopwatch.StartNew();
                    var refs = st.Buffer.ToArray();
                    System.Threading.Tasks.Parallel.For(0, nPatch, PllOpt, p =>
                    {
                        float best = -1f;
                        int off = p * FeatDim;
                        for (int k = 0; k < refs.Length; k++)
                        {
                            float dot = Dot(feat, refs[k], off, FeatDim);
                            if (dot > best) best = dot;
                        }
                        anom[p] = 1f - best;
                    });

                    Profile[3] += swC.Elapsed.TotalMilliseconds;

                    // 图级分数 = P99 二值化后、面积≤AreaCap 的最大连通域面积(空间聚集性+尺寸先验)
                    double score; Rect gridBox;
                    ScoreAndLocate(anom, st.Cols, out score, out gridBox);
                    r.Score = score;

                    // 自校准: 用缓冲自身做留一法打分,鲁棒阈值 = 中位数 + K×MAD(抗少量缺陷帧污染)
                    if (st.Threshold < 0)
                        st.Threshold = CalibrateLoo(st);
                    r.Threshold = st.Threshold;
                    r.Ready = st.Threshold >= 0;
                    r.Alarm = r.Ready && score >= st.Threshold;
                    if (gridBox.Width > 0)
                    {
                        double sx = (double)frame.Width / st.Cols, sy = (double)(y1 - y0) / Gh;
                        r.Box = new Rect((int)(gridBox.X * sx), (int)(y0 + gridBox.Y * sy),
                                         (int)(gridBox.Width * sx), (int)(gridBox.Height * sy));
                    }
                }

                st.Buffer.Add(feat);                       // 因果滚动缓冲(只含历史帧)
                if (st.Buffer.Count > BufSize) st.Buffer.RemoveAt(0);
            }
            return r;
        }

        /// <summary>停机/换布时调用：场景将整体变化，历史参考失效，清空重建。</summary>
        public void ResetAll()
        {
            lock (stateLock)
            {
                foreach (var st in states.Values) lock (st) { st.Buffer.Clear(); st.Threshold = -1; }
            }
            Log("BlackYarn: 参考缓冲已重置");
        }

        /// <summary>筘齿带 → 重叠tile → ONNX 推理 → 拼接带特征(重叠区平均, 再L2归一化)。</summary>
        private float[] ExtractBandFeature(Mat frame, out int cols)
        {
            cols = frame.Width / PatchSize;
            int y0 = (int)(frame.Height * BandY0), y1 = (int)(frame.Height * BandY1);
            using (Mat band = new Mat(frame, new Rect(0, y0, frame.Width, y1 - y0)))
            using (Mat resized = band.Resize(new Size(frame.Width, TileH)))
            {
                int W = resized.Width;
                var xs = new List<int>();
                for (int x = 0; x <= Math.Max(0, W - TileW); x += TileW - Overlap) xs.Add(x);
                if (xs.Count == 0) return null;
                if (xs[xs.Count - 1] + TileW < W) xs.Add(W - TileW);

                float[] acc = new float[Gh * cols * FeatDim];
                int[] cnt = new int[Gh * cols];
                float[] input = new float[3 * TileH * TileW];
                int gw = TileW / PatchSize;

                // 整条带一次性转成 CHW float(ImageNet 归一化)。
                // 原实现是每个 tile 用 GetGenericIndexer<Vec3b> 逐像素取，7×448×588≈184 万次
                // 托管↔原生调用，是单帧耗时的大头；改为 GetArray 一次拷出 + 纯数组算术。
                var swP = System.Diagnostics.Stopwatch.StartNew();
                Vec3b[] px;
                resized.GetArray(out px);
                float[] bandCHW = new float[3 * TileH * W];
                int plane = TileH * W;
                for (int r0 = 0; r0 < TileH; r0++)
                {
                    int rowOff = r0 * W;
                    for (int c0 = 0; c0 < W; c0++)
                    {
                        Vec3b q = px[rowOff + c0];                       // OpenCV 是 BGR
                        bandCHW[0 * plane + rowOff + c0] = (q.Item2 / 255f - Mean[0]) / Std[0];   // R
                        bandCHW[1 * plane + rowOff + c0] = (q.Item1 / 255f - Mean[1]) / Std[1];   // G
                        bandCHW[2 * plane + rowOff + c0] = (q.Item0 / 255f - Mean[2]) / Std[2];   // B
                    }
                }
                px = null;
                Profile[0] += swP.Elapsed.TotalMilliseconds;

                foreach (int x in xs)
                {
                    // 从整条带里按行切出该 tile，纯 Array.Copy
                    for (int ch = 0; ch < 3; ch++)
                        for (int r0 = 0; r0 < TileH; r0++)
                            Array.Copy(bandCHW, ch * plane + r0 * W + x,
                                       input, ch * TileH * TileW + r0 * TileW, TileW);

                    var tensor = new DenseTensor<float>(input, new[] { 1, 3, TileH, TileW });
                    float[] feat;
                    var swI = System.Diagnostics.Stopwatch.StartNew();
                    using (var results = session.Run(new[] { NamedOnnxValue.CreateFromTensor("input", tensor) }))
                        feat = results.First().AsEnumerable<float>().ToArray();   // (1344, 6144) 扁平
                    Profile[1] += swI.Elapsed.TotalMilliseconds;

                    int j0 = x / PatchSize;
                    for (int gr = 0; gr < Gh; gr++)
                        for (int gc = 0; gc < gw; gc++)
                        {
                            int dst = gr * cols + Math.Min(j0 + gc, cols - 1);
                            int src = (gr * gw + gc) * FeatDim;
                            int dstOff = dst * FeatDim;
                            for (int c = 0; c < FeatDim; c++) acc[dstOff + c] += feat[src + c];
                            cnt[dst]++;
                        }
                }

                // 平均 + L2 归一化(重叠平均会破坏单位模长)
                var swN = System.Diagnostics.Stopwatch.StartNew();
                int vw = System.Numerics.Vector<float>.Count;
                System.Threading.Tasks.Parallel.For(0, Gh * cols, PllOpt, p =>
                {
                    int n = Math.Max(1, cnt[p]); int off = p * FeatDim;
                    float invN = 1f / n;
                    var vInvN = new System.Numerics.Vector<float>(invN);
                    double norm = 0;
                    int c = 0;
                    for (; c <= FeatDim - vw; c += vw)
                    {
                        var v = new System.Numerics.Vector<float>(acc, off + c) * vInvN;
                        v.CopyTo(acc, off + c);
                        norm += System.Numerics.Vector.Dot(v, v);
                    }
                    for (; c < FeatDim; c++) { acc[off + c] *= invN; norm += (double)acc[off + c] * acc[off + c]; }

                    var vInv = new System.Numerics.Vector<float>((float)(1.0 / Math.Max(Math.Sqrt(norm), 1e-8)));
                    c = 0;
                    for (; c <= FeatDim - vw; c += vw)
                        (new System.Numerics.Vector<float>(acc, off + c) * vInv).CopyTo(acc, off + c);
                    for (; c < FeatDim; c++) acc[off + c] *= vInv[0];
                });
                Profile[2] += swN.Elapsed.TotalMilliseconds;
                return acc;
            }
        }

        /// <summary>
        /// 两段等长子向量的点积，SIMD 加速。单帧要算 patch数×参考帧数 次(7296×8≈5.8 万次
        /// 6144 维点积 = 3.6 亿次乘加)，标量循环是仅次于推理的第二大开销。
        /// 注意：向量化改变了求和顺序，浮点结果与标量版会有 ~1e-6 量级差异；
        /// 图级分数是整数连通域面积，实测与 Python 基准逐帧一致。
        /// </summary>
        private static float Dot(float[] a, float[] b, int off, int n)
        {
            int w = System.Numerics.Vector<float>.Count;
            var acc = System.Numerics.Vector<float>.Zero;
            int i = 0;
            for (; i <= n - w; i += w)
                acc += new System.Numerics.Vector<float>(a, off + i) *
                       new System.Numerics.Vector<float>(b, off + i);
            float s = 0;
            for (int k = 0; k < w; k++) s += acc[k];
            for (; i < n; i++) s += a[off + i] * b[off + i];
            return s;
        }

        /// <summary>P99 二值化 → 连通域 → 面积≤cap 的最大块(分数=面积, 定位=外接框, 网格坐标)。</summary>
        private static void ScoreAndLocate(float[] anom, int cols, out double score, out Rect box)
        {
            score = 0; box = new Rect();
            float[] sorted = (float[])anom.Clone(); Array.Sort(sorted);
            // 必须与 np.percentile 的默认线性插值一致。原来取 sorted[(int)(n*0.99)]
            // （n=7296 时是第 7223 号），numpy 取的是 0.99*(n-1)=7222.05 处的插值——
            // 差一个元素。真实帧的分数大多不受影响，但会改变 LOO 自校准里少数帧的分数，
            // 进而改变阈值：实测 Cam5 上 C# 得 9.69、Python 得 15.38（差 37%）。
            double pos = 0.99 * (sorted.Length - 1);
            int lo = (int)Math.Floor(pos), hi = Math.Min(lo + 1, sorted.Length - 1);
            float thr = (float)(sorted[lo] + (pos - lo) * (sorted[hi] - sorted[lo]));

            using (Mat m = new Mat(Gh, cols, MatType.CV_8UC1))
            {
                byte[] bin = new byte[Gh * cols];
                for (int k = 0; k < bin.Length; k++) bin[k] = anom[k] >= thr ? (byte)255 : (byte)0;
                m.SetArray(bin);

                using (Mat labels = new Mat())
                using (Mat stats = new Mat())
                using (Mat cent = new Mat())
                {
                    int n = Cv2.ConnectedComponentsWithStats(m, labels, stats, cent, PixelConnectivity.Connectivity8);
                    int bestArea = 0, bi = -1;
                    for (int i = 1; i < n; i++)
                    {
                        int area = stats.At<int>(i, (int)ConnectedComponentsTypes.Area);
                        if (area <= AreaCap && area > bestArea) { bestArea = area; bi = i; }
                    }
                    if (bi < 0) return;
                    score = bestArea;
                    box = new Rect(stats.At<int>(bi, (int)ConnectedComponentsTypes.Left),
                                   stats.At<int>(bi, (int)ConnectedComponentsTypes.Top),
                                   stats.At<int>(bi, (int)ConnectedComponentsTypes.Width),
                                   stats.At<int>(bi, (int)ConnectedComponentsTypes.Height));
                }
            }
        }

        /// <summary>
        /// 中位数。**偶数长度必须取中间两个的平均**，与 numpy.median 语义一致。
        /// 曾因取"偏上那个"导致阈值虚高（19.47 → 27.76，灵敏度下降、边缘缺陷会漏检），
        /// 且该偏差不会报错、只在真缺陷落于两阈值之间时静默漏检。见 SelfTest 一致性验证。
        /// </summary>
        private static double Median(IList<double> v)
        {
            var a = v.OrderBy(x => x).ToArray();
            int n = a.Length;
            if (n == 0) return 0;
            return (n % 2 == 1) ? a[n / 2] : (a[n / 2 - 1] + a[n / 2]) / 2.0;
        }

        /// <summary>留一法自校准：缓冲内每帧以其余帧为参考打分，阈值=中位数+K×MAD。</summary>
        private double CalibrateLoo(CamState st)
        {
            int nPatch = Gh * st.Cols;
            var loo = new List<double>();
            for (int i = 0; i < st.Buffer.Count; i++)
            {
                float[] cur = st.Buffer[i];
                float[] anom = new float[nPatch];
                int ii = i;
                System.Threading.Tasks.Parallel.For(0, nPatch, PllOpt, p =>
                {
                    float best = -1f; int off = p * FeatDim;
                    for (int j = 0; j < st.Buffer.Count; j++)
                    {
                        if (j == ii) continue;
                        float dot = Dot(cur, st.Buffer[j], off, FeatDim);
                        if (dot > best) best = dot;
                    }
                    anom[p] = 1f - best;
                });
                double s; Rect _;
                ScoreAndLocate(anom, st.Cols, out s, out _);
                loo.Add(s);
            }
            if (loo.Count < 3) return -1;
            double med = Median(loo);
            double mad = 1.4826 * Median(loo.Select(v => Math.Abs(v - med)).ToList());
            double thr = med + KMad * Math.Max(mad, 1.0);
            // 把 LOO 明细也打出来：阈值 = 中位数 + 3.5×MAD，而 LOO 分数是小整数(连通域面积)，
            // MAD 在 1 和 2 之间跳一档就会让阈值差 5，现场判读误报率时需要看到这组原始值。
            Log($"BlackYarn: 自校准完成 阈值={thr:F2} (中位{med:F1} MAD{mad / 1.4826:F1}) LOO=[{string.Join(",", loo.Select(v => v.ToString("F0")))}]");
            return thr;
        }

        public void Dispose()
        {
            session?.Dispose(); session = null;
            lock (stateLock) { states.Clear(); }
        }
    }
}
