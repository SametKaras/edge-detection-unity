using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Profiling;
using Unity.Collections;
using System.Runtime.InteropServices;
using System.Diagnostics;
#if UNITY_EDITOR
using UnityEditor;
#endif

namespace SceneCapture.Edge3D
{
    public enum KernelSizeOption { _3x3=3, _5x5=5, _7x7=7, _9x9=9, _11x11=11, _13x13=13, _15x15=15, _17x17=17, _19x19=19, _21x21=21 }

    [StructLayout(LayoutKind.Sequential)]
    public struct MicroLine { public float sx, sy, sz; public float ex, ey, ez; }

    [RequireComponent(typeof(EdgeDetectionEffect))]
    public class WorldSpaceEdgeManager : MonoBehaviour
    {
        [Header("References")] public ComputeShader microLineCS;
        [Header("Settings")] [Range(0.01f, 10f)] public float updateInterval = 0.03f;
        [Header("Detection Sensitivity")] [Range(0.01f, 0.9f)] public float minEdgeLuminance = 0.15f;
        [Range(0.0f, 0.05f)] public float nmsRelaxation = 0.008f;
        [Header("Edge Geometry Filter")] public bool useGeomDiscontinuityFilter = true;
        [Range(0.0005f, 0.05f)] public float minGeomDiscontinuity = 0.006f;
        public bool requireBoundaryTransition = false;
        [Header("RANSAC (Kernel)")] public KernelSizeOption kernelSize = KernelSizeOption._7x7;
        [Range(2, 25)] public int minPointsForLine = 5;
        [Range(0.05f, 0.5f)] public float minEdgePointRatio = 0.20f;
        [Range(0.01f, 0.5f)] public float inlierThreshold = 0.08f;
        [Range(0.01f, 10f)] public float maxSegmentLength = 0.15f;
        [Header("Normal Consistency")] [Range(0.0f, 0.98f)] public float normalDotThreshold = 0.7f;
        [Header("Resolution Normalization")] public bool autoScaleKernelWithResolution = true;
        [Range(720, 2160)] public int referenceHeight = 1080;
        [Range(0.5f, 1.5f)] public float kernelScaleExponent = 1.0f;
        public bool autoScaleMinPointsForLine = true;
        [Range(0.01f, 0.2f)] public float minPointAreaRatio = 0.08f;
        [Range(4, 24)] public int maxAutoMinPoints = 14;
        [Header("Line Quality Filters")] [Range(0.3f, 0.8f)] public float minInlierRatio = 0.60f;
        [Range(1.0f, 3.0f)] public float minLinearityFactor = 2.0f;
        [Header("Stability")] public bool useStableSeed = true;
        public int stableSeedValue = 1337;
        [Header("Visualization")] public bool showLines = true;
        public Color lineColor = Color.red;
        [Range(1f, 15f)] public float visualLineThickness = 4.0f;
        [Range(100, 5000)] public int maxOutputLines = 5000;
        [Header("Debug: Raw Edge Points")] public bool showRawEdgePoints = false;
        public Color rawPointColor = Color.cyan;
        [Range(0.001f, 0.1f)] public float rawPointSize = 0.01f;
        [Header("Performance (Read-Only)")] public bool showPerformance = true;

        [SerializeField] private float _algorithmMs, _readbackMs, _totalPipelineMs;
        [SerializeField] private int _lineCountDisplay;

        private Camera _mainCam;
        private RenderTexture _edgePosRT;
        private EdgeDetectionEffect _edgeEffect;

        private ComputeBuffer _lineBuffer, _countBuffer, _debugBuffer, _debugCountBuffer;
        private MicroLine[] _displayLines, _debugPoints;
        private int _displayLineCount, _debugPointCount, _pendingLineCount;
        private bool _hasNewData = false;
        private float _lastUpdateTime = -999f;
        private const int MAX_LINES = 100000, STRIDE = 24;

        private enum ReadbackState { Idle, WaitingCount, WaitingLines }
        private ReadbackState _rbState = ReadbackState.Idle;
        private AsyncGPUReadbackRequest _countReq, _lineReq;
        private Stopwatch _sw = new Stopwatch();
        private long _dispatchStartTick, _dispatchEndTick, _readbackDoneTick;

        void Start()
        {
            _mainCam = GetComponent<Camera>();
            _edgeEffect = GetComponent<EdgeDetectionEffect>();
            _mainCam.depthTextureMode |= DepthTextureMode.Depth | DepthTextureMode.DepthNormals;

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
            PollReadback();
#if UNITY_EDITOR
            if (_hasNewData) UnityEditor.SceneView.RepaintAll();
#endif
            if (_rbState != ReadbackState.Idle || Time.time - _lastUpdateTime < updateInterval) return;
            if (microLineCS == null || _edgeEffect == null || _edgeEffect.EdgeResultTexture == null || _edgeEffect.WorldPosTexture == null) return;

            _lastUpdateTime = Time.time;
            DispatchFrame();
        }

        int ComputeEffectiveKernelSize(int currentHeight)
        {
            int baseK = (int)kernelSize;
            if (!autoScaleKernelWithResolution) return baseK;
            float ratio = Mathf.Max(0.1f, (float)currentHeight / Mathf.Max(1f, referenceHeight));
            float scale = Mathf.Pow(ratio, kernelScaleExponent);
            int k = Mathf.Clamp(Mathf.RoundToInt(baseK * scale), 3, 21);
            return (k & 1) == 0 ? k + 1 : k;
        }

        int ComputeEffectiveMinPoints(int effectiveKernel)
        {
            if (!autoScaleMinPointsForLine) return minPointsForLine;
            int areaBased = Mathf.Clamp(Mathf.RoundToInt(effectiveKernel * effectiveKernel * minPointAreaRatio), 2, maxAutoMinPoints);
            return Mathf.Max(minPointsForLine, areaBased);
        }

        void DispatchFrame()
        {
            // DOĞRUDAN TEXTURE ÇÖZÜNÜRLÜKLERİNİ ALIYORUZ!
            int camW = _edgeEffect.EdgeResultTexture.width;
            int camH = _edgeEffect.EdgeResultTexture.height;

            EnsureRT(ref _edgePosRT, camW, camH, RenderTextureFormat.ARGBFloat, true, FilterMode.Point);

            if (_lineBuffer == null) return;
            _lineBuffer.SetCounterValue(0);

            int buildKernel = microLineCS.FindKernel("BuildEdgePositionBuffer");
            int fitKernel = microLineCS.FindKernel("FitMicroLines");

            microLineCS.SetTexture(buildKernel, "_EdgeTex", _edgeEffect.EdgeResultTexture);
            microLineCS.SetTexture(buildKernel, "_WorldPosTex", _edgeEffect.WorldPosTexture);
            microLineCS.SetTexture(buildKernel, "_RWEdgePosTex", _edgePosRT);

            Texture normalTex = Shader.GetGlobalTexture("_CameraDepthNormalsTexture") ?? Texture2D.whiteTexture;
            microLineCS.SetTexture(buildKernel, "_NormalTex", normalTex);
            microLineCS.SetTexture(fitKernel, "_EdgePosTex", _edgePosRT);
            microLineCS.SetTexture(fitKernel, "_NormalTex", normalTex);

            int k = ComputeEffectiveKernelSize(camH);
            microLineCS.SetFloat("_MinEdgeThreshold", minEdgeLuminance);
            microLineCS.SetFloat("_NmsRelaxation", nmsRelaxation);
            microLineCS.SetInt("_UseGeomDiscontinuityFilter", useGeomDiscontinuityFilter ? 1 : 0);
            microLineCS.SetInt("_RequireBoundaryTransition", requireBoundaryTransition ? 1 : 0);
            microLineCS.SetFloat("_MinGeomDiscontinuity", minGeomDiscontinuity * Mathf.Pow(Mathf.Max(0.1f, (float)referenceHeight / Mathf.Max(1f, camH)), 0.35f));
            microLineCS.SetInt("_KernelSize", k);
            microLineCS.SetInt("_MinPointsForLine", ComputeEffectiveMinPoints(k));
            microLineCS.SetFloat("_MinEdgePointRatio", minEdgePointRatio);
            microLineCS.SetFloat("_InlierThreshold", inlierThreshold);
            microLineCS.SetFloat("_MaxSegLength", maxSegmentLength);
            microLineCS.SetFloat("_MinInlierRatio", minInlierRatio);
            microLineCS.SetFloat("_MinLinearityFactor", minLinearityFactor);
            microLineCS.SetFloat("_NormalDotThreshold", normalDotThreshold);
            
            // ÇÖZÜNÜRLÜĞÜ SABİTLEDİK!
            microLineCS.SetInt("_TexWidth", camW);
            microLineCS.SetInt("_TexHeight", camH);
            
            microLineCS.SetInt("_FrameSeed", useStableSeed ? stableSeedValue : Time.frameCount);
            microLineCS.SetVector("_CameraWorldPos", _mainCam.transform.position);

            microLineCS.SetBuffer(fitKernel, "_OutputLines", _lineBuffer);
            if (_debugBuffer != null)
            {
                _debugBuffer.SetCounterValue(0);
                microLineCS.SetBuffer(fitKernel, "_DebugPoints", _debugBuffer);
            }
            microLineCS.SetInt("_DebugMode", showRawEdgePoints ? 1 : 0);

            _dispatchStartTick = _sw.ElapsedTicks;
            
            // HESAPLAMALARI camW ve camH İLE YAPIYORUZ
            microLineCS.Dispatch(buildKernel, Mathf.CeilToInt((float)camW / 8f), Mathf.CeilToInt((float)camH / 8f), 1);
            microLineCS.Dispatch(fitKernel, Mathf.CeilToInt((float)camW / k), Mathf.CeilToInt((float)camH / k), 1);
            
            _dispatchEndTick = _sw.ElapsedTicks;

            ComputeBuffer.CopyCount(_lineBuffer, _countBuffer, 0);
            _countReq = AsyncGPUReadback.Request(_countBuffer);
            _rbState = ReadbackState.WaitingCount;

            if (showRawEdgePoints && _debugBuffer != null)
            {
                ComputeBuffer.CopyCount(_debugBuffer, _debugCountBuffer, 0);
                int[] debugCount = new int[3];
                _debugCountBuffer.GetData(debugCount);
                _debugPointCount = Mathf.Min(debugCount[0], MAX_LINES);
                if (_debugPointCount > 0) _debugBuffer.GetData(_debugPoints, 0, 0, _debugPointCount);
            }
            else _debugPointCount = 0;
        }

        void PollReadback()
        {
            if (_rbState == ReadbackState.WaitingCount && _countReq.done)
            {
                if (_countReq.hasError) { _rbState = ReadbackState.Idle; return; }
                _pendingLineCount = Mathf.Min(_countReq.GetData<int>()[0], Mathf.Clamp(maxOutputLines, 1, MAX_LINES));
                if (_pendingLineCount > 0)
                {
                    _lineReq = AsyncGPUReadback.Request(_lineBuffer, _pendingLineCount * STRIDE, 0);
                    _rbState = ReadbackState.WaitingLines;
                }
                else
                {
                    _displayLineCount = 0; _hasNewData = true; _rbState = ReadbackState.Idle; UpdateTimings(0);
                }
            }
            if (_rbState == ReadbackState.WaitingLines && _lineReq.done)
            {
                if (!_lineReq.hasError)
                {
                    NativeArray<MicroLine>.Copy(_lineReq.GetData<MicroLine>(), 0, _displayLines, 0, _pendingLineCount);
                    _displayLineCount = _pendingLineCount;
                    _hasNewData = true;
                }
                _rbState = ReadbackState.Idle; UpdateTimings(_displayLineCount);
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
                Handles.color = lineColor;
                for (int i = 0; i < _displayLineCount; i++) Handles.DrawAAPolyLine(visualLineThickness, new Vector3(_displayLines[i].sx, _displayLines[i].sy, _displayLines[i].sz), new Vector3(_displayLines[i].ex, _displayLines[i].ey, _displayLines[i].ez));
#endif
            }
            if (showRawEdgePoints && _debugPoints != null)
            {
                Gizmos.color = rawPointColor;
                for (int i = 0; i < _debugPointCount; i++) Gizmos.DrawSphere(new Vector3(_debugPoints[i].sx, _debugPoints[i].sy, _debugPoints[i].sz), rawPointSize);
            }
        }

        void OnGUI()
        {
            if (!showPerformance) return;
            GUIStyle style = new GUIStyle(GUI.skin.box) { fontSize = 14, alignment = TextAnchor.UpperLeft, richText = true };
            style.normal.textColor = Color.white;
            int h = _edgeEffect != null && _edgeEffect.EdgeResultTexture != null ? _edgeEffect.EdgeResultTexture.height : Screen.height;
            GUI.Box(new Rect(10, 10, 320, 135), $"<b>Edge Detection Performance</b>\nAlgorithm:  <color={(_algorithmMs < 1f ? "#00FF88" : "#FF4444")}>{_algorithmMs:F4} ms</color>\nReadback:   {_readbackMs:F2} ms  <color=#888888>(async)</color>\nLines:      {_lineCountDisplay}\nKernel:     {ComputeEffectiveKernelSize(h)}×{ComputeEffectiveKernelSize(h)}\nNormal:     {(normalDotThreshold > 0.01f ? $"dot≥{normalDotThreshold:F2}" : "OFF")}", style);
        }

        void OnDisable() { DisposeBuffers(); }
        void OnDestroy() { DisposeBuffers(); }
        void DisposeBuffers() { ReleaseBuffer(ref _lineBuffer); ReleaseBuffer(ref _countBuffer); ReleaseBuffer(ref _debugBuffer); ReleaseBuffer(ref _debugCountBuffer); ReleaseRT(ref _edgePosRT); }
        void ReleaseBuffer(ref ComputeBuffer buf) { if (buf != null) { buf.Release(); buf = null; } }
        void ReleaseRT(ref RenderTexture rt) { if (rt != null) { rt.Release(); rt = null; } }
        void EnsureRT(ref RenderTexture rt, int w, int h, RenderTextureFormat fmt, bool randomWrite, FilterMode filter) { if (rt != null && rt.width == w && rt.height == h) return; ReleaseRT(ref rt); rt = new RenderTexture(w, h, 0, fmt); rt.enableRandomWrite = randomWrite; rt.filterMode = filter; rt.Create(); }
    }
}