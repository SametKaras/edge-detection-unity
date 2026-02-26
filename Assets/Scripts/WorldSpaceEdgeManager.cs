using UnityEngine;
using UnityEngine.Rendering;
using Unity.Mathematics;
using Unity.Collections;
using System.Runtime.InteropServices;
using System.Diagnostics;
#if UNITY_EDITOR
using UnityEditor;
#endif

namespace SceneCapture.Edge3D
{
    public enum KernelSizeOption
    {
        _3x3 = 3,
        _5x5 = 5,
        _7x7 = 7,
        _9x9 = 9
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct MicroLine
    {
        public float sx, sy, sz;
        public float ex, ey, ez;
    }

    [RequireComponent(typeof(EdgeDetectionEffect))]
    public class WorldSpaceEdgeManager : MonoBehaviour
    {
        [Header("References")]
        public Shader worldPosShader;
        public ComputeShader microLineCS;

        [Header("Settings")]
        [Range(0.01f, 10f)] public float updateInterval = 0.03f;

        [Header("Detection Sensitivity")]
        [Range(0.01f, 0.9f)] public float minEdgeLuminance = 0.1f;
        [Range(0.0f, 0.05f)] public float nmsRelaxation = 0.008f;
        
        [Header("Edge Geometry Filter")]
        [Tooltip("Smooth/false edge'leri geometri farkina gore bastirir.")]
        public bool useGeomDiscontinuityFilter = true;
        [Range(0.0005f, 0.05f)] public float minGeomDiscontinuity = 0.006f;

        [Header("RANSAC (Kernel)")]
        public KernelSizeOption kernelSize = KernelSizeOption._5x5;
        [Range(2, 25)] public int minPointsForLine = 2;
        [Range(0.01f, 0.5f)] public float inlierThreshold = 0.08f;
        [Range(0.01f, 10f)] public float maxSegmentLength = 0.15f;
        
        [Header("Resolution Normalization")]
        [Tooltip("Cozunurluk arttikca kernel'i otomatik buyutup ayni geometri icin benzer line yogunlugu korur.")]
        public bool autoScaleKernelWithResolution = true;
        [Range(720, 2160)] public int referenceHeight = 1080;
        [Range(0.5f, 1.5f)] public float kernelScaleExponent = 1.0f;
        [Tooltip("Min point sayisini etkin kernel alanina gore otomatik artirir.")]
        public bool autoScaleMinPointsForLine = true;
        [Range(0.01f, 0.2f)] public float minPointAreaRatio = 0.08f;
        [Range(4, 24)] public int maxAutoMinPoints = 14;

        [Header("Line Quality Filters")]
        [Range(0.3f, 0.8f)] public float minInlierRatio = 0.52f;
        [Range(1.0f, 3.0f)] public float minLinearityFactor = 1.5f;

        [Header("Visualization")]
        public bool showLines = true;
        [Range(1f, 15f)] public float visualLineThickness = 4.0f;

        [Header("Line Regularization")]
        [Tooltip("Kapaliysa post-process dogrultu duzeltmeleri uygulanmaz, sadece compute/RANSAC sonucu cizilir.")]
        public bool usePostRegularization = false;
        [Tooltip("Bulunan mikro-line'lari baskin yonde duzeltir (capsule tepesindeki caprazlari azaltir).")]
        public bool regularizeLineDirections = true;
        [Range(1f, 45f)] public float regularizationAngleDeg = 18f;
        [Range(0.0001f, 0.1f)] public float regularizationMinLength = 0.01f;
        [Tooltip("Baskin yonden cok sapan line'lari tamamen siler.")]
        public bool removeDirectionOutliers = true;
        [Range(1f, 45f)] public float outlierAngleDeg = 28f;
        [Tooltip("Line merkezlerini tek bir referans hatta yaklastirir (zigzag'i duzeltir).")]
        public bool stabilizeLineCenter = true;
        [Range(0f, 1f)] public float centerSnapStrength = 0.85f;
        [Range(0.001f, 0.2f)] public float maxLateralDistance = 0.04f;
        [Tooltip("Tum mikro-line'lari tek baskin hatta indirger (sphere/capsule icin ideal).")]
        public bool collapseToSingleDominantLine = false;
        [Tooltip("Baskin yonu aci-inlier tabanli robust fit ile hesaplar.")]
        public bool robustDirectionFit = true;
        [Range(1f, 45f)] public float robustDirectionInlierAngleDeg = 18f;

        [Header("Point Cloud Visualization")]
        public bool showPointCloud = false;
        public Color pointCloudColor = Color.green;
        [Range(0.001f, 0.1f)] public float pointSize = 0.01f;

        [Header("Debug: Raw Edge Points")]
        [Tooltip("Edge piksellerinin ham 3D pozisyonlarını gösterir (RANSAC öncesi)")]
        public bool showRawEdgePoints = false;
        public Color rawPointColor = Color.cyan;

        [Header("Performance (Read-Only)")]
        public bool showPerformance = true;
        [SerializeField] private float _algorithmMs;
        [SerializeField] private float _readbackMs;
        [SerializeField] private float _totalPipelineMs;
        [SerializeField] private int   _lineCountDisplay;

        // ===== Internal =====
        private Camera _mainCam;
        private Camera _posCam;
        private RenderTexture _worldPosRT;
        private RenderTexture _edgePosRT;
        private EdgeDetectionEffect _edgeEffect;

        private ComputeBuffer _lineBuffer;
        private ComputeBuffer _countBuffer;
        private ComputeBuffer _debugBuffer;
        private ComputeBuffer _debugCountBuffer;

        private MicroLine[] _displayLines;
        private int _displayLineCount;
        private MicroLine[] _debugPoints;
        private int _debugPointCount;
        private bool _hasNewData = false;
        private float _lastUpdateTime = -999f;

        private const int MAX_LINES = 100000;
        private const int STRIDE = 24;

        // Async readback state machine
        private enum ReadbackState { Idle, WaitingCount, WaitingLines }
        private ReadbackState _rbState = ReadbackState.Idle;
        private AsyncGPUReadbackRequest _countReq;
        private AsyncGPUReadbackRequest _lineReq;
        private int _pendingLineCount;

        // Timing
        private Stopwatch _sw = new Stopwatch();
        private long _dispatchStartTick;
        private long _dispatchEndTick;
        private long _readbackDoneTick;

        void Start()
        {
            _mainCam = GetComponent<Camera>();
            _edgeEffect = GetComponent<EdgeDetectionEffect>();

            GameObject camObj = new GameObject("WorldPosCamera");
            camObj.transform.SetParent(transform, false);
            _posCam = camObj.AddComponent<Camera>();
            _posCam.CopyFrom(_mainCam);
            _posCam.enabled = false;
            // WorldPos geçerlilik maskesi için arka plan alpha'sı 0 olmalı.
            _posCam.backgroundColor = new Color(0f, 0f, 0f, 0f);
            _posCam.clearFlags = CameraClearFlags.SolidColor;
            _posCam.renderingPath = RenderingPath.Forward;

            _lineBuffer = new ComputeBuffer(MAX_LINES, STRIDE, ComputeBufferType.Append);
            _countBuffer = new ComputeBuffer(1, sizeof(uint) * 3, ComputeBufferType.IndirectArguments);
            _debugBuffer = new ComputeBuffer(MAX_LINES, STRIDE, ComputeBufferType.Append);
            _debugCountBuffer = new ComputeBuffer(1, sizeof(uint) * 3, ComputeBufferType.IndirectArguments);
            _displayLines = new MicroLine[MAX_LINES];
            _debugPoints = new MicroLine[MAX_LINES];

            _sw.Start();
        }

        void Update()
        {
            // 1. Async readback sonuçlarını kontrol et
            PollReadback();

            // 2. Yeni frame dispatch et (sadece idle ise)
            if (_rbState != ReadbackState.Idle) return;
            if (Time.time - _lastUpdateTime < updateInterval) return;
            if (worldPosShader == null || microLineCS == null) return;
            if (_edgeEffect == null || _edgeEffect.EdgeResultTexture == null) return;

            _lastUpdateTime = Time.time;
            DispatchFrame();
        }

        int ComputeEffectiveKernelSize()
        {
            int baseK = (int)kernelSize;
            if (!autoScaleKernelWithResolution) return baseK;

            float safeRef = Mathf.Max(1f, referenceHeight);
            float ratio = Mathf.Max(0.1f, (float)Screen.height / safeRef);
            float scale = Mathf.Pow(ratio, kernelScaleExponent);
            int k = Mathf.RoundToInt(baseK * scale);
            k = Mathf.Clamp(k, 3, 31);
            if ((k & 1) == 0) k += 1; // kernel tek olmali
            return k;
        }

        int ComputeEffectiveMinPoints(int effectiveKernel)
        {
            if (!autoScaleMinPointsForLine) return minPointsForLine;
            int areaBased = Mathf.RoundToInt(effectiveKernel * effectiveKernel * minPointAreaRatio);
            areaBased = Mathf.Clamp(areaBased, 2, maxAutoMinPoints);
            return Mathf.Max(minPointsForLine, areaBased);
        }

        void DispatchFrame()
        {
            // Kamera senkronizasyonu
            _posCam.fieldOfView   = _mainCam.fieldOfView;
            _posCam.nearClipPlane = _mainCam.nearClipPlane;
            _posCam.farClipPlane  = _mainCam.farClipPlane;
            _posCam.aspect        = _mainCam.aspect;

            // WorldPos render
            if (_worldPosRT == null || _worldPosRT.width != Screen.width || _worldPosRT.height != Screen.height)
            {
                if (_worldPosRT != null) _worldPosRT.Release();
                _worldPosRT = new RenderTexture(Screen.width, Screen.height, 0, RenderTextureFormat.ARGBFloat);
                _worldPosRT.filterMode = FilterMode.Point;
                _worldPosRT.Create();
            }

            if (_edgePosRT == null || _edgePosRT.width != Screen.width || _edgePosRT.height != Screen.height)
            {
                if (_edgePosRT != null) _edgePosRT.Release();
                _edgePosRT = new RenderTexture(Screen.width, Screen.height, 0, RenderTextureFormat.ARGBFloat);
                _edgePosRT.enableRandomWrite = true;
                _edgePosRT.filterMode = FilterMode.Point;
                _edgePosRT.Create();
            }
            _posCam.targetTexture = _worldPosRT;
            _posCam.RenderWithShader(worldPosShader, "RenderType");

            // Compute shader dispatch
            if (_lineBuffer == null || _edgeEffect.EdgeResultTexture == null) return;
            _lineBuffer.SetCounterValue(0);
            int buildKernel = microLineCS.FindKernel("BuildEdgePositionBuffer");
            int fitKernel = microLineCS.FindKernel("FitMicroLines");

            microLineCS.SetTexture(buildKernel, "_EdgeTex", _edgeEffect.EdgeResultTexture);
            microLineCS.SetTexture(buildKernel, "_WorldPosTex", _worldPosRT);
            microLineCS.SetTexture(buildKernel, "_RWEdgePosTex", _edgePosRT);

            microLineCS.SetTexture(fitKernel, "_EdgePosTex", _edgePosRT);
            microLineCS.SetFloat("_MinEdgeThreshold", minEdgeLuminance);
            microLineCS.SetFloat("_NmsRelaxation", nmsRelaxation);
            microLineCS.SetInt("_UseGeomDiscontinuityFilter", useGeomDiscontinuityFilter ? 1 : 0);
            float geomScale = Mathf.Pow(Mathf.Max(0.1f, (float)referenceHeight / Mathf.Max(1f, Screen.height)), 0.35f);
            float effectiveGeomDisc = minGeomDiscontinuity * geomScale;
            microLineCS.SetFloat("_MinGeomDiscontinuity", effectiveGeomDisc);
            microLineCS.SetFloat("_MinInlierRatio", minInlierRatio);
            microLineCS.SetFloat("_MinLinearityFactor", minLinearityFactor);
            int k = ComputeEffectiveKernelSize();
            int effectiveMinPoints = ComputeEffectiveMinPoints(k);
            microLineCS.SetInt("_KernelSize", k);
            microLineCS.SetInt("_MinPointsForLine", effectiveMinPoints);
            microLineCS.SetFloat("_InlierThreshold", inlierThreshold);
            microLineCS.SetFloat("_MaxSegLength", maxSegmentLength);
            microLineCS.SetInt("_TexWidth", Screen.width);
            microLineCS.SetInt("_TexHeight", Screen.height);
            microLineCS.SetBuffer(fitKernel, "_OutputLines", _lineBuffer);
            microLineCS.SetInt("_FrameSeed", Time.frameCount);

            // Debug mode
            if (_debugBuffer != null)
            {
                _debugBuffer.SetCounterValue(0);
                microLineCS.SetBuffer(fitKernel, "_DebugPoints", _debugBuffer);
            }
            microLineCS.SetInt("_DebugMode", showRawEdgePoints ? 1 : 0);

            int tilesX = Mathf.CeilToInt((float)Screen.width / k);
            int tilesY = Mathf.CeilToInt((float)Screen.height / k);

            // === ALGORITHM TIMING START ===
            UnityEngine.Profiling.Profiler.BeginSample("MicroLine_RANSAC_5x5");
            _dispatchStartTick = _sw.ElapsedTicks;

            microLineCS.Dispatch(buildKernel, Mathf.CeilToInt(Screen.width / 8f), Mathf.CeilToInt(Screen.height / 8f), 1);
            microLineCS.Dispatch(fitKernel, Mathf.CeilToInt(tilesX / 8f), Mathf.CeilToInt(tilesY / 8f), 1);

            _dispatchEndTick = _sw.ElapsedTicks;
            UnityEngine.Profiling.Profiler.EndSample();
            // === ALGORITHM TIMING END ===

            // Async readback başlat (CPU'yu bloklamaz!)
            ComputeBuffer.CopyCount(_lineBuffer, _countBuffer, 0);
            _countReq = AsyncGPUReadback.Request(_countBuffer);
            _rbState = ReadbackState.WaitingCount;

            // Debug noktaları için sünkron readback (debug modda)
            if (showRawEdgePoints && _debugBuffer != null)
            {
                ComputeBuffer.CopyCount(_debugBuffer, _debugCountBuffer, 0);
                int[] debugCount = new int[3];
                _debugCountBuffer.GetData(debugCount);
                _debugPointCount = Mathf.Min(debugCount[0], MAX_LINES);
                if (_debugPointCount > 0)
                {
                    _debugBuffer.GetData(_debugPoints, 0, 0, _debugPointCount);
                }
            }
            else
            {
                _debugPointCount = 0;
            }
        }

        void PollReadback()
        {
            if (_rbState == ReadbackState.WaitingCount && _countReq.done)
            {
                if (_countReq.hasError)
                {
                    _rbState = ReadbackState.Idle;
                    return;
                }

                var countData = _countReq.GetData<int>();
                _pendingLineCount = Mathf.Min(countData[0], MAX_LINES);

                if (_pendingLineCount > 0)
                {
                    _lineReq = AsyncGPUReadback.Request(_lineBuffer, _pendingLineCount * STRIDE, 0);
                    _rbState = ReadbackState.WaitingLines;
                }
                else
                {
                    _displayLineCount = 0;
                    _hasNewData = true;
                    _rbState = ReadbackState.Idle;
                    UpdateTimings(0);
                }
            }

            if (_rbState == ReadbackState.WaitingLines && _lineReq.done)
            {
                if (!_lineReq.hasError)
                {
                    var lineData = _lineReq.GetData<MicroLine>();
                    NativeArray<MicroLine>.Copy(lineData, 0, _displayLines, 0, _pendingLineCount);
                    _displayLineCount = _pendingLineCount;
                    _hasNewData = true;
                }

                _rbState = ReadbackState.Idle;
                UpdateTimings(_displayLineCount);

#if UNITY_EDITOR
                SceneView.RepaintAll();
#endif
            }
        }

        void UpdateTimings(int lineCount)
        {
            double ticksToMs = 1000.0 / Stopwatch.Frequency;
            _algorithmMs = (float)((_dispatchEndTick - _dispatchStartTick) * ticksToMs);
            _readbackDoneTick = _sw.ElapsedTicks;
            _readbackMs = (float)((_readbackDoneTick - _dispatchEndTick) * ticksToMs);
            _totalPipelineMs = (float)((_readbackDoneTick - _dispatchStartTick) * ticksToMs);
            _lineCountDisplay = lineCount;
        }

        int RegularizeLinesInPlace(int lineCount)
        {
            if (_displayLines == null || lineCount <= 1) return lineCount;
            if (!regularizeLineDirections && !removeDirectionOutliers) return lineCount;

            // 1) Frame'in baskin yonunu bul (ilk eleman yerine en uzun line'dan seed al)
            Vector3 v = Vector3.up;
            bool hasSeed = false;
            float maxSeedLen = 0f;

            for (int i = 0; i < lineCount; i++)
            {
                var l = _displayLines[i];
                Vector3 d = new Vector3(l.ex - l.sx, l.ey - l.sy, l.ez - l.sz);
                float len = d.magnitude;
                if (len < regularizationMinLength) continue;
                if (len > maxSeedLen)
                {
                    maxSeedLen = len;
                    v = d / len;
                    hasSeed = true;
                }
            }
            if (!hasSeed) return lineCount;

            for (int it = 0; it < 8; it++)
            {
                Vector3 sum = Vector3.zero;
                for (int i = 0; i < lineCount; i++)
                {
                    var l = _displayLines[i];
                    Vector3 d = new Vector3(l.ex - l.sx, l.ey - l.sy, l.ez - l.sz);
                    float len = d.magnitude;
                    if (len < regularizationMinLength) continue;

                    Vector3 n = d / len;
                    float w = len;
                    float proj = Vector3.Dot(n, v);
                    sum += n * (proj * proj * w);
                }
                if (sum.sqrMagnitude < 1e-10f) break;
                v = sum.normalized;
            }

            if (robustDirectionFit)
                v = ComputeRobustDirection(lineCount, v);

            float cosThreshold = Mathf.Cos(regularizationAngleDeg * Mathf.Deg2Rad);
            float cosOutlier = Mathf.Cos(outlierAngleDeg * Mathf.Deg2Rad);

            // Outlier temizliği bu frame'de çok agresif olacaksa otomatik gevşet.
            int predictedKeep = 0;
            if (removeDirectionOutliers)
            {
                for (int i = 0; i < lineCount; i++)
                {
                    var l = _displayLines[i];
                    Vector3 d = new Vector3(l.ex - l.sx, l.ey - l.sy, l.ez - l.sz);
                    float len = d.magnitude;
                    if (len < regularizationMinLength) { predictedKeep++; continue; }
                    Vector3 n = d / len;
                    float ad = Mathf.Abs(Vector3.Dot(n, v));
                    if (ad >= cosOutlier) predictedKeep++;
                }
            }
            bool applyOutlierThisFrame = removeDirectionOutliers && predictedKeep >= Mathf.Max(8, Mathf.RoundToInt(lineCount * 0.25f));

            // 2) Baskin yone yakin line'lari bu yone snap et
            for (int i = 0; i < lineCount; i++)
            {
                var l = _displayLines[i];
                Vector3 s = new Vector3(l.sx, l.sy, l.sz);
                Vector3 e = new Vector3(l.ex, l.ey, l.ez);
                Vector3 d = e - s;
                float len = d.magnitude;
                if (len < regularizationMinLength) continue;

                Vector3 n = d / len;
                float ad = Mathf.Abs(Vector3.Dot(n, v));
                if (applyOutlierThisFrame && ad < cosOutlier)
                {
                    // Outlier'i silmek icin uzunlugu 0'a cek; asagida compaction yapilacak.
                    l.ex = l.sx; l.ey = l.sy; l.ez = l.sz;
                    _displayLines[i] = l;
                    continue;
                }
                if (ad < cosThreshold) continue;

                // Yon isaretini koru, merkez ve uzunlugu degistirme.
                Vector3 dir = Vector3.Dot(n, v) >= 0 ? v : -v;
                Vector3 c = (s + e) * 0.5f;
                Vector3 hs = dir * (len * 0.5f);

                l.sx = c.x - hs.x; l.sy = c.y - hs.y; l.sz = c.z - hs.z;
                l.ex = c.x + hs.x; l.ey = c.y + hs.y; l.ez = c.z + hs.z;
                _displayLines[i] = l;
            }

            // 2.5) Merkezleri referans hatta yaklastir (acisal degil konumsal zigzag'i azalt)
            if (stabilizeLineCenter)
            {
                Vector3 meanCenter = Vector3.zero;
                float meanW = 0f;

                // Referans merkez: baskin yone yakin segmentlerin agirlikli ortalamasi
                for (int i = 0; i < lineCount; i++)
                {
                    var l = _displayLines[i];
                    Vector3 s = new Vector3(l.sx, l.sy, l.sz);
                    Vector3 e = new Vector3(l.ex, l.ey, l.ez);
                    Vector3 d = e - s;
                    float len = d.magnitude;
                    if (len < regularizationMinLength) continue;

                    Vector3 n = d / len;
                    float ad = Mathf.Abs(Vector3.Dot(n, v));
                    if (ad < cosThreshold) continue;

                    Vector3 c = (s + e) * 0.5f;
                    float w = len * ad;
                    meanCenter += c * w;
                    meanW += w;
                }

                if (meanW > 1e-6f)
                {
                    meanCenter /= meanW;

                    for (int i = 0; i < lineCount; i++)
                    {
                        var l = _displayLines[i];
                        Vector3 s = new Vector3(l.sx, l.sy, l.sz);
                        Vector3 e = new Vector3(l.ex, l.ey, l.ez);
                        Vector3 d = e - s;
                        float len = d.magnitude;
                        if (len < regularizationMinLength) continue;

                        Vector3 n = d / len;
                        float ad = Mathf.Abs(Vector3.Dot(n, v));
                        if (ad < cosThreshold) continue;

                        Vector3 c = (s + e) * 0.5f;
                        Vector3 toC = c - meanCenter;
                        Vector3 perp = toC - v * Vector3.Dot(toC, v);
                        float lateral = perp.magnitude;

                        if (removeDirectionOutliers && lateral > maxLateralDistance)
                        {
                            l.ex = l.sx; l.ey = l.sy; l.ez = l.sz;
                            _displayLines[i] = l;
                            continue;
                        }

                        Vector3 cSnapped = c - perp * centerSnapStrength;
                        Vector3 hs = n * (len * 0.5f);
                        l.sx = cSnapped.x - hs.x; l.sy = cSnapped.y - hs.y; l.sz = cSnapped.z - hs.z;
                        l.ex = cSnapped.x + hs.x; l.ey = cSnapped.y + hs.y; l.ez = cSnapped.z + hs.z;
                        _displayLines[i] = l;
                    }
                }
            }

            // 3) Uzunlugu sifirlanan outlier'lari diziden cikar
            if (applyOutlierThisFrame)
            {
                int write = 0;
                for (int i = 0; i < lineCount; i++)
                {
                    var l = _displayLines[i];
                    float dx = l.ex - l.sx;
                    float dy = l.ey - l.sy;
                    float dz = l.ez - l.sz;
                    float lenSq = dx * dx + dy * dy + dz * dz;
                    // Sadece bilincli olarak sifirladigimiz outlier'lari cikar.
                    if (lenSq <= 1e-12f) continue;
                    _displayLines[write++] = l;
                }
                lineCount = write;
            }

            if (collapseToSingleDominantLine)
                return CollapseToSingleDominantLine(lineCount, v);

            return lineCount;
        }

        Vector3 ComputeRobustDirection(int lineCount, Vector3 seedDir)
        {
            if (_displayLines == null || lineCount <= 0) return seedDir;
            Vector3 bestDir = seedDir.sqrMagnitude > 1e-8f ? seedDir.normalized : Vector3.up;
            float bestScore = -1f;
            float cosInlier = Mathf.Cos(robustDirectionInlierAngleDeg * Mathf.Deg2Rad);

            // Her line yonunu aday kabul et; en cok aci-inlier toplayani sec.
            for (int i = 0; i < lineCount; i++)
            {
                var li = _displayLines[i];
                Vector3 di = new Vector3(li.ex - li.sx, li.ey - li.sy, li.ez - li.sz);
                float leni = di.magnitude;
                if (leni < regularizationMinLength) continue;
                Vector3 ci = di / leni;

                float score = 0f;
                for (int j = 0; j < lineCount; j++)
                {
                    var lj = _displayLines[j];
                    Vector3 dj = new Vector3(lj.ex - lj.sx, lj.ey - lj.sy, lj.ez - lj.sz);
                    float lenj = dj.magnitude;
                    if (lenj < regularizationMinLength) continue;
                    Vector3 nj = dj / lenj;
                    float ad = Mathf.Abs(Vector3.Dot(nj, ci));
                    if (ad < cosInlier) continue;
                    score += lenj * ad * ad;
                }

                if (score > bestScore)
                {
                    bestScore = score;
                    bestDir = ci;
                }
            }

            // Secilen yon etrafindaki inlier'lardan imzali ortalama ile son yonu rafine et.
            Vector3 refined = Vector3.zero;
            for (int i = 0; i < lineCount; i++)
            {
                var l = _displayLines[i];
                Vector3 d = new Vector3(l.ex - l.sx, l.ey - l.sy, l.ez - l.sz);
                float len = d.magnitude;
                if (len < regularizationMinLength) continue;
                Vector3 n = d / len;
                float dot = Vector3.Dot(n, bestDir);
                float ad = Mathf.Abs(dot);
                if (ad < cosInlier) continue;
                refined += (dot >= 0f ? n : -n) * (len * ad);
            }

            if (refined.sqrMagnitude > 1e-10f)
                bestDir = refined.normalized;

            return bestDir;
        }

        int CollapseToSingleDominantLine(int lineCount, Vector3 seedDir)
        {
            if (_displayLines == null || lineCount <= 0) return 0;

            Vector3 mean = Vector3.zero;
            float wsum = 0f;
            for (int i = 0; i < lineCount; i++)
            {
                var l = _displayLines[i];
                Vector3 s = new Vector3(l.sx, l.sy, l.sz);
                Vector3 e = new Vector3(l.ex, l.ey, l.ez);
                float len = (e - s).magnitude;
                if (len < regularizationMinLength) continue;
                Vector3 c = (s + e) * 0.5f;
                mean += c * len;
                wsum += len;
            }
            if (wsum <= 1e-6f) return 0;
            mean /= wsum;

            // Basit power-iteration ile baskin yon
            Vector3 v = seedDir.sqrMagnitude > 1e-8f ? seedDir.normalized : Vector3.up;
            for (int it = 0; it < 8; it++)
            {
                Vector3 sum = Vector3.zero;
                for (int i = 0; i < lineCount; i++)
                {
                    var l = _displayLines[i];
                    Vector3 s = new Vector3(l.sx, l.sy, l.sz);
                    Vector3 e = new Vector3(l.ex, l.ey, l.ez);
                    float len = (e - s).magnitude;
                    if (len < regularizationMinLength) continue;
                    Vector3 c = (s + e) * 0.5f;
                    Vector3 q = c - mean;
                    float p = Vector3.Dot(q, v);
                    sum += q * (p * len);
                }
                if (sum.sqrMagnitude < 1e-10f) break;
                v = sum.normalized;
            }

            float minProj = float.MaxValue;
            float maxProj = float.MinValue;
            bool hasAny = false;

            // Uclari axis'e projekte edip toplam kapsami bul
            for (int i = 0; i < lineCount; i++)
            {
                var l = _displayLines[i];
                Vector3 s = new Vector3(l.sx, l.sy, l.sz);
                Vector3 e = new Vector3(l.ex, l.ey, l.ez);
                float len = (e - s).magnitude;
                if (len < regularizationMinLength) continue;

                Vector3 ds = s - mean;
                Vector3 de = e - mean;
                float ps = Vector3.Dot(ds, v);
                float pe = Vector3.Dot(de, v);

                minProj = Mathf.Min(minProj, Mathf.Min(ps, pe));
                maxProj = Mathf.Max(maxProj, Mathf.Max(ps, pe));
                hasAny = true;
            }

            if (!hasAny || maxProj <= minProj) return 0;

            Vector3 a = mean + v * minProj;
            Vector3 b = mean + v * maxProj;

            _displayLines[0] = new MicroLine
            {
                sx = a.x, sy = a.y, sz = a.z,
                ex = b.x, ey = b.y, ez = b.z
            };
            return 1;
        }

        void OnDrawGizmos()
        {
            if (!_hasNewData) return;

            if (showLines && _displayLines != null)
            {
#if UNITY_EDITOR
                Handles.zTest = CompareFunction.Always;
                Handles.color = Color.red;
                for (int i = 0; i < _displayLineCount; i++)
                {
                    var l = _displayLines[i];
                    Handles.DrawAAPolyLine(visualLineThickness,
                        new Vector3(l.sx, l.sy, l.sz),
                        new Vector3(l.ex, l.ey, l.ez));
                }
#endif
            }

            if (showPointCloud)
            {
                Gizmos.color = pointCloudColor;
                for (int i = 0; i < _displayLineCount; i++)
                {
                    var l = _displayLines[i];
                    Gizmos.DrawSphere(new Vector3(
                        (l.sx + l.ex) * 0.5f,
                        (l.sy + l.ey) * 0.5f,
                        (l.sz + l.ez) * 0.5f), pointSize);
                }
            }

            // Raw edge pixel pozisyonları (debug)
            if (showRawEdgePoints && _debugPoints != null)
            {
                Gizmos.color = rawPointColor;
                for (int i = 0; i < _debugPointCount; i++)
                {
                    var p = _debugPoints[i];
                    Gizmos.DrawSphere(new Vector3(p.sx, p.sy, p.sz), pointSize);
                }
            }
        }

        void OnGUI()
        {
            if (!showPerformance) return;

            GUIStyle style = new GUIStyle(GUI.skin.box);
            style.fontSize = 14;
            style.alignment = TextAnchor.UpperLeft;
            style.normal.textColor = Color.white;
            style.richText = true;

            string algoColor = _algorithmMs < 1f ? "#00FF88" : "#FF4444";

            string info = $"<b>Edge Detection Performance</b>\n" +
                          $"Algorithm:  <color={algoColor}>{_algorithmMs:F4} ms</color>\n" +
                          $"Readback:   {_readbackMs:F2} ms  <color=#888888>(async)</color>\n" +
                          $"Lines:      {_lineCountDisplay}\n" +
                          $"Kernel:     {kernelSize}×{kernelSize}";

            GUI.Box(new Rect(10, 10, 300, 115), info, style);
        }

        void OnDisable() { DisposeBuffers(); }
        void OnDestroy() { DisposeBuffers(); }

        void DisposeBuffers()
        {
            if (_lineBuffer != null) { _lineBuffer.Release(); _lineBuffer = null; }
            if (_countBuffer != null) { _countBuffer.Release(); _countBuffer = null; }
            if (_debugBuffer != null) { _debugBuffer.Release(); _debugBuffer = null; }
            if (_debugCountBuffer != null) { _debugCountBuffer.Release(); _debugCountBuffer = null; }
            if (_worldPosRT != null) { _worldPosRT.Release(); _worldPosRT = null; }
            if (_edgePosRT != null) { _edgePosRT.Release(); _edgePosRT = null; }
        }
    }
}