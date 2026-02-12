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

        [Header("RANSAC (Kernel)")]
        [Range(3, 7)] public int kernelSize = 5;
        [Range(2, 25)] public int minPointsForLine = 2;
        [Range(0.01f, 0.5f)] public float inlierThreshold = 0.08f;
        [Range(0.01f, 10f)] public float maxSegmentLength = 0.15f;

        [Header("Visualization")]
        public bool showLines = true;
        [Range(1f, 15f)] public float visualLineThickness = 4.0f;

        [Header("Point Cloud Visualization")]
        public bool showPointCloud = false;
        public Color pointCloudColor = Color.green;
        [Range(0.001f, 0.1f)] public float pointSize = 0.01f;

        [Header("Performance (Read-Only)")]
        public bool showPerformance = true;
        [SerializeField] private float _algorithmMs;   // Compute dispatch + GPU fence
        [SerializeField] private float _readbackMs;     // Async readback tamamlanma
        [SerializeField] private float _totalPipelineMs;
        [SerializeField] private int   _lineCountDisplay;

        // ===== Internal =====
        private Camera _mainCam;
        private Camera _posCam;
        private RenderTexture _worldPosRT;
        private EdgeDetectionEffect _edgeEffect;

        private ComputeBuffer _lineBuffer;
        private ComputeBuffer _countBuffer;

        private MicroLine[] _displayLines;
        private int _displayLineCount;
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
            _posCam.backgroundColor = Color.black;
            _posCam.clearFlags = CameraClearFlags.SolidColor;
            _posCam.renderingPath = RenderingPath.Forward;

            _lineBuffer = new ComputeBuffer(MAX_LINES, STRIDE, ComputeBufferType.Append);
            _countBuffer = new ComputeBuffer(1, sizeof(uint) * 3, ComputeBufferType.IndirectArguments);
            _displayLines = new MicroLine[MAX_LINES];

            _sw.Start();
        }

        void Update()
        {
            // 1. Async readback sonuÃ§larÄ±nÄ± kontrol et
            PollReadback();

            // 2. Yeni frame dispatch et (sadece idle ise)
            if (_rbState != ReadbackState.Idle) return;
            if (Time.time - _lastUpdateTime < updateInterval) return;
            if (worldPosShader == null || microLineCS == null) return;
            if (_edgeEffect == null || _edgeEffect.EdgeResultTexture == null) return;

            _lastUpdateTime = Time.time;
            DispatchFrame();
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
                _worldPosRT.Create();
            }

            _posCam.targetTexture = _worldPosRT;
            _posCam.RenderWithShader(worldPosShader, "RenderType");

            // Compute shader dispatch â€” Profiler marker ile
            _lineBuffer.SetCounterValue(0);
            int kernel = microLineCS.FindKernel("FitMicroLines");

            microLineCS.SetTexture(kernel, "_EdgeTex", _edgeEffect.EdgeResultTexture);
            microLineCS.SetTexture(kernel, "_WorldPosTex", _worldPosRT);
            microLineCS.SetFloat("_MinEdgeThreshold", minEdgeLuminance);
            microLineCS.SetInt("_KernelSize", kernelSize);
            microLineCS.SetInt("_MinPointsForLine", minPointsForLine);
            microLineCS.SetFloat("_InlierThreshold", inlierThreshold);
            microLineCS.SetFloat("_MaxSegLength", maxSegmentLength);
            microLineCS.SetInt("_TexWidth", Screen.width);
            microLineCS.SetInt("_TexHeight", Screen.height);
            microLineCS.SetBuffer(kernel, "_OutputLines", _lineBuffer);
            microLineCS.SetInt("_FrameSeed", Time.frameCount);

            int tilesX = Mathf.CeilToInt((float)Screen.width / kernelSize);
            int tilesY = Mathf.CeilToInt((float)Screen.height / kernelSize);

            // === ALGORITHM TIMING START ===
            UnityEngine.Profiling.Profiler.BeginSample("MicroLine_RANSAC_5x5");
            _dispatchStartTick = _sw.ElapsedTicks;

            microLineCS.Dispatch(kernel, Mathf.CeilToInt(tilesX / 8f), Mathf.CeilToInt(tilesY / 8f), 1);

            _dispatchEndTick = _sw.ElapsedTicks;
            UnityEngine.Profiling.Profiler.EndSample();
            // === ALGORITHM TIMING END ===

            // Async readback baÅŸlat (CPU'yu bloklamaz!)
            ComputeBuffer.CopyCount(_lineBuffer, _countBuffer, 0);
            _countReq = AsyncGPUReadback.Request(_countBuffer);
            _rbState = ReadbackState.WaitingCount;
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
                          $"Kernel:     {kernelSize}Ã—{kernelSize}";

            GUI.Box(new Rect(10, 10, 300, 115), info, style);
        }

        void OnDisable() { DisposeBuffers(); }
        void OnDestroy() { DisposeBuffers(); }

        void DisposeBuffers()
        {
            if (_lineBuffer != null) { _lineBuffer.Release(); _lineBuffer = null; }
            if (_countBuffer != null) { _countBuffer.Release(); _countBuffer = null; }
            if (_worldPosRT != null) { _worldPosRT.Release(); _worldPosRT = null; }
        }
    }
}