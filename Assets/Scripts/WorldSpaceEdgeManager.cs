using System.IO;
using System.Text;
using UnityEngine;
using UnityEngine.Rendering;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;
using Unity.Burst;
using Random = Unity.Mathematics.Random;
#if UNITY_EDITOR
using UnityEditor;
#endif

namespace SceneCapture.Edge3D
{
    public struct Line3D
    {
        public float3 Start;
        public float3 End;
        public bool IsValid;
    }

    [RequireComponent(typeof(EdgeDetectionEffect))]
    public class WorldSpaceEdgeManager : MonoBehaviour
    {
        [Header("References")]
        public Shader worldPosShader;
        public ComputeShader pointExtractorCS;

        [Header("⏱️ Timing Control")]
        [Tooltip("İşlem bittikten sonra en az bu kadar saniye bekle.")]
        [Range(0.1f, 10f)] public float updateInterval = 2.0f; 
        
        [Header("🚀 Speed vs Quality")]
        [Tooltip("1 = En Yüksek Kalite (Yavaş), 4 = En Hızlı (Düşük Kalite). 20sn süren işlem 4 yaparsan 1sn sürer.")]
        [Range(1, 8)] public int pixelStep = 1; 

        [Header("Detection Sensitivity")]
        [Range(0.01f, 0.9f)] public float minEdgeLuminance = 0.1f;

        [Header("RANSAC Settings")]
        [Range(1, 5000)] public int maxLinesToDetect = 3000;
        [Range(100, 50000)] public int ransacIterations = 15000; 
        [Range(0.01f, 0.5f)] public float lineThickness = 0.08f;
        [Range(2, 500)] public int minPointsForLine = 2;
        [Range(0.01f, 10f)] public float maxSegmentLength = 0.15f;

        [Header("Visualization")]
        public bool showLines = true;
        [Range(1f, 15f)] public float visualLineThickness = 4.0f; 
        
        // Private Refs
        private Camera _mainCam;
        private Camera _posCam;
        private RenderTexture _worldPosRT;
        private EdgeDetectionEffect _edgeEffect;
        
        // Compute Buffers
        private ComputeBuffer _pointBuffer;
        private ComputeBuffer _argsBuffer;
        
        // Job System Variables
        private NativeArray<float3> _inputPoints;
        private NativeArray<Line3D> _outputLines;
        private JobHandle _ransacJobHandle;
        
        private bool _isJobRunning = false;
        private bool _hasNewData = false;
        private float _lastJobFinishTime = -999f;
        
        private Line3D[] _displayLines;

        void Start()
        {
            _mainCam = GetComponent<Camera>();
            _edgeEffect = GetComponent<EdgeDetectionEffect>();

            GameObject camObj = new GameObject("WorldPosCamera");
            camObj.transform.SetParent(transform);
            _posCam = camObj.AddComponent<Camera>();
            _posCam.CopyFrom(_mainCam);
            _posCam.enabled = false;
            _posCam.backgroundColor = Color.black;
            _posCam.clearFlags = CameraClearFlags.SolidColor;
            _posCam.renderingPath = RenderingPath.Forward;

            CreateBuffers();
            _displayLines = new Line3D[maxLinesToDetect];
        }

        void CreateBuffers()
        {
            if (_pointBuffer == null || !_pointBuffer.IsValid())
                _pointBuffer = new ComputeBuffer(150000, sizeof(float) * 3, ComputeBufferType.Append);
            
            if (_argsBuffer == null || !_argsBuffer.IsValid())
                _argsBuffer = new ComputeBuffer(1, sizeof(uint) * 3, ComputeBufferType.IndirectArguments);
        }

        void Update()
        {
            if (_isJobRunning)
            {
                if (_ransacJobHandle.IsCompleted)
                {
                    CompleteJob();
                }
                return;
            }

            if (Time.time - _lastJobFinishTime < updateInterval)
            {
                return;
            }

            if (worldPosShader == null || pointExtractorCS == null || _edgeEffect.EdgeResultTexture == null) return;
            RenderAndExtractPoints();
        }

        void RenderAndExtractPoints()
        {
            if (_worldPosRT == null || _worldPosRT.width != Screen.width || _worldPosRT.height != Screen.height)
            {
                if (_worldPosRT != null) _worldPosRT.Release();
                _worldPosRT = new RenderTexture(Screen.width, Screen.height, 0, RenderTextureFormat.ARGBFloat);
                _worldPosRT.Create();
            }

            _posCam.targetTexture = _worldPosRT;
            _posCam.RenderWithShader(worldPosShader, "RenderType");

            _pointBuffer.SetCounterValue(0);
            int kernel = pointExtractorCS.FindKernel("ExtractPoints");
            
            pointExtractorCS.SetTexture(kernel, "_EdgeTex", _edgeEffect.EdgeResultTexture);
            pointExtractorCS.SetTexture(kernel, "_WorldPosTex", _worldPosRT);
            pointExtractorCS.SetFloat("_MinEdgeThreshold", minEdgeLuminance);
            
            pointExtractorCS.SetInt("_Step", pixelStep); 
            
            // İYİLEŞTİRME: Boyutları uniform olarak geçir (GetDimensions yerine)
            pointExtractorCS.SetInt("_TexWidth", Screen.width);
            pointExtractorCS.SetInt("_TexHeight", Screen.height);
            
            pointExtractorCS.SetBuffer(kernel, "_PointBuffer", _pointBuffer);

            // İYİLEŞTİRME: Dispatch sayısını _Step'e böl
            // ESKİ: tüm pixeller dispatch → %94'ü boşa (Step=4)
            // YENİ: sadece örneklenecek pixeller dispatch → sıfır israf
            // numthreads(16,16,1) olduğu için 16'ya bölüyoruz
            int groupsX = Mathf.CeilToInt((float)Screen.width / (16 * pixelStep));
            int groupsY = Mathf.CeilToInt((float)Screen.height / (16 * pixelStep));
            pointExtractorCS.Dispatch(kernel, groupsX, groupsY, 1);

            ComputeBuffer.CopyCount(_pointBuffer, _argsBuffer, 0);
            AsyncGPUReadback.Request(_argsBuffer, OnArgBufferReadback);
        }

        void OnArgBufferReadback(AsyncGPUReadbackRequest request)
        {
            if (request.hasError || _pointBuffer == null || !_pointBuffer.IsValid()) return;

            var data = request.GetData<uint>();
            int pointCount = (int)data[0];

            if (pointCount > 0)
            {
                AsyncGPUReadback.Request(_pointBuffer, pointCount * 12, 0, (req) => OnPointsReadback(req, pointCount));
            }
        }

        void OnPointsReadback(AsyncGPUReadbackRequest request, int count)
        {
            if (request.hasError) return;
            if (_isJobRunning) return; 

            _inputPoints = new NativeArray<float3>(request.GetData<float3>(), Allocator.Persistent);
            _outputLines = new NativeArray<Line3D>(maxLinesToDetect, Allocator.Persistent);

            var ransacJob = new LocalRansacJob
            {
                InputPoints = _inputPoints,
                ResultLines = _outputLines,
                MaxLinesToFind = maxLinesToDetect,
                MaxIterations = ransacIterations,
                Threshold = lineThickness,
                MinInliers = minPointsForLine,
                MaxSegLength = maxSegmentLength,
                RandomSeed = (uint)Time.frameCount
            };

            _ransacJobHandle = ransacJob.Schedule();
            _isJobRunning = true; 
        }

        void CompleteJob()
        {
            _ransacJobHandle.Complete();

            _outputLines.CopyTo(_displayLines); 
            _hasNewData = true;

            if (_inputPoints.IsCreated) _inputPoints.Dispose();
            if (_outputLines.IsCreated) _outputLines.Dispose();

            _isJobRunning = false;
            
            // Zamanlayıcıyı başlat
            _lastJobFinishTime = Time.time; 
        }

        void OnDrawGizmos()
        {
            if (showLines && _hasNewData && _displayLines != null)
            {
#if UNITY_EDITOR
                Handles.zTest = CompareFunction.Always;
                Handles.color = Color.red; 
                for (int i = 0; i < _displayLines.Length; i++)
                {
                    var line = _displayLines[i];
                    if (line.IsValid)
                    {
                        Handles.DrawAAPolyLine(visualLineThickness, line.Start, line.End);
                    }
                }
#endif
            }
        }

        void OnDisable() { DisposeBuffers(); }
        void OnDestroy() { DisposeBuffers(); }

        void DisposeBuffers()
        {
            if (_isJobRunning) _ransacJobHandle.Complete();
            if (_inputPoints.IsCreated) _inputPoints.Dispose();
            if (_outputLines.IsCreated) _outputLines.Dispose();

            if (_pointBuffer != null) _pointBuffer.Release();
            if (_argsBuffer != null) _argsBuffer.Release();
            if (_worldPosRT != null) _worldPosRT.Release();
        }
    }

    [BurstCompile(CompileSynchronously = true)]
    public struct LocalRansacJob : IJob
    {
        [ReadOnly] public NativeArray<float3> InputPoints;
        public NativeArray<Line3D> ResultLines;
        
        public int MaxLinesToFind;
        public int MaxIterations;
        public float Threshold;
        public int MinInliers;
        public float MaxSegLength;
        public uint RandomSeed;

        public void Execute()
        {
            int pointCount = InputPoints.Length;
            if (pointCount < 2) return;

            NativeArray<bool> usedPoints = new NativeArray<bool>(pointCount, Allocator.Temp);
            Random rng = new Random(RandomSeed > 0 ? RandomSeed : 1);

            int linesFound = 0;
            int totalPointsUsed = 0;
            int consecutiveFailures = 0;
            
            float thresholdSq = Threshold * Threshold;
            float segLengthSq = MaxSegLength * MaxSegLength;

            int neighborSearchRange = 4000; 
            int preCheckCount = 64; 

            while (linesFound < MaxLinesToFind && totalPointsUsed < pointCount - MinInliers)
            {
                float3 bestP1 = float3.zero;
                float3 bestP2 = float3.zero;
                int bestInlierCount = -1;
                
                for (int iter = 0; iter < MaxIterations; iter++)
                {
                    int idx1 = rng.NextInt(pointCount);
                    if (usedPoints[idx1]) continue;
                    float3 p1 = InputPoints[idx1];

                    bool useGlobalSearch = (iter % 3 == 0); 

                    int idx2;
                    if (useGlobalSearch)
                    {
                        idx2 = rng.NextInt(pointCount);
                    }
                    else
                    {
                        int minIdx = math.max(0, idx1 - neighborSearchRange);
                        int maxIdx = math.min(pointCount, idx1 + neighborSearchRange);
                        idx2 = rng.NextInt(minIdx, maxIdx);
                    }

                    if (idx2 == idx1 || usedPoints[idx2]) continue;
                    float3 p2 = InputPoints[idx2];

                    if (math.distancesq(p1, p2) > segLengthSq) continue;

                    float3 lineVec = math.normalize(p2 - p1);
                    float3 lineStart = p1;

                    int preInliers = 0;
                    for(int k=0; k < preCheckCount; k++)
                    {
                        int testIdx;
                        if(useGlobalSearch) 
                            testIdx = rng.NextInt(pointCount);
                        else
                        {
                            int tMin = math.max(0, idx1 - neighborSearchRange); 
                            int tMax = math.min(pointCount, idx1 + neighborSearchRange);
                            testIdx = rng.NextInt(tMin, tMax);
                        }

                        if (!usedPoints[testIdx])
                        {
                             float3 tp = InputPoints[testIdx];
                             if (math.distancesq(tp, lineStart) <= segLengthSq)
                             {
                                 if (math.lengthsq(math.cross(tp - lineStart, lineVec)) < thresholdSq)
                                 {
                                     preInliers++;
                                     if(preInliers >= 2) break; 
                                 }
                             }
                        }
                    }

                    if (preInliers == 0 && pointCount > 2000) continue; 

                    int currentInliers = 0;
                    for (int i = 0; i < pointCount; i++)
                    {
                        if (usedPoints[i]) continue;
                        float3 p = InputPoints[i];
                        
                        if (math.distancesq(p, lineStart) > segLengthSq) continue;
                        if (math.lengthsq(math.cross(p - lineStart, lineVec)) < thresholdSq) currentInliers++;
                    }

                    if (currentInliers > bestInlierCount)
                    {
                        bestInlierCount = currentInliers;
                        bestP1 = p1; bestP2 = p2;
                        
                        if (bestInlierCount > 100) break; 
                    }
                }

                if (bestInlierCount >= MinInliers)
                {
                    consecutiveFailures = 0;
                    float3 lineDir = math.normalize(bestP2 - bestP1);
                    float minProj = float.MaxValue;
                    float maxProj = float.MinValue;

                    for (int i = 0; i < pointCount; i++)
                    {
                        if (usedPoints[i]) continue;
                        float3 p = InputPoints[i];
                        if (math.distancesq(p, bestP1) > segLengthSq) continue;

                        float3 vec = p - bestP1;
                        if (math.lengthsq(math.cross(vec, lineDir)) < thresholdSq)
                        {
                            usedPoints[i] = true;
                            totalPointsUsed++;
                            float proj = math.dot(vec, lineDir);
                            if (proj < minProj) minProj = proj;
                            if (proj > maxProj) maxProj = proj;
                        }
                    }
                    
                    if (maxProj > minProj)
                    {
                        ResultLines[linesFound] = new Line3D { Start = bestP1 + lineDir * minProj, End = bestP1 + lineDir * maxProj, IsValid = true };
                        linesFound++;
                    }
                }
                else 
                {
                    consecutiveFailures++;
                    if (consecutiveFailures > 50) break; 
                }
            }
            usedPoints.Dispose();
        }
    }
}