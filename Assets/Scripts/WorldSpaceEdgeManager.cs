using UnityEngine;
using UnityEngine.Rendering;
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
        [Tooltip("Raw point'i sadece world-pos valid/invalid sinir gecisinde uretir (en robust mod).")]
        public bool requireBoundaryTransition = false;

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

        [Header("Debug: Raw Edge Points")]
        [Tooltip("Edge piksellerinin ham 3D pozisyonlarını gösterir (RANSAC öncesi)")]
        public bool showRawEdgePoints = false;
        public Color rawPointColor = Color.cyan;
        [Range(0.001f, 0.1f)] public float rawPointSize = 0.01f;

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
            microLineCS.SetInt("_RequireBoundaryTransition", requireBoundaryTransition ? 1 : 0);
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

            // Raw edge pixel pozisyonları (debug)
            if (showRawEdgePoints && _debugPoints != null)
            {
                Gizmos.color = rawPointColor;
                for (int i = 0; i < _debugPointCount; i++)
                {
                    var p = _debugPoints[i];
                    Gizmos.DrawSphere(new Vector3(p.sx, p.sy, p.sz), rawPointSize);
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