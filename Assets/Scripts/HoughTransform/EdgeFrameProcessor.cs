using UnityEngine;
using UnityEngine.Rendering;
using Unity.Collections;
using System;

namespace SceneCapture.Hough
{
    /// <summary>
    /// GPU'daki edge texture'ını CPU'ya asenkron aktarır.
    /// 
    /// İYİLEŞTİRME:
    ///   Gradient material'ına EdgeDetection.shader ile aynı kaynak parametrelerini geçirir.
    ///   Böylece EdgeDirection.shader Combined/Depth/Normal modda doğru gradient hesaplar.
    /// </summary>
    public class EdgeFrameProcessor : IDisposable
    {
        private NativeArray<byte> _edgeBuffer;
        private NativeArray<float> _gradientXBuffer;
        private NativeArray<float> _gradientYBuffer;
        private NativeArray<float> _magnitudeBuffer;
        private NativeArray<Vector2Int> _edgePixels;
        
        private int _width, _height;
        private bool _readbackPending;
        private bool _dataReady;
        private bool _disposed;
        private int _frameCounter;
        private int _pendingReadbackCount;
        
        private RenderTexture _downsampledRT;
        private RenderTexture _gradientRT;
        private Material _gradientMaterial;
        
        private const int MAX_EDGE_PIXELS = 100000;
        private int _edgePixelCount;
        
        public bool IsDataReady => _dataReady && !_disposed;
        public int Width => _width;
        public int Height => _height;
        public int EdgePixelCount => _edgePixelCount;
        
        public NativeArray<float> GradientXBuffer => _gradientXBuffer;
        public NativeArray<float> GradientYBuffer => _gradientYBuffer;
        public NativeArray<float> MagnitudeBuffer => _magnitudeBuffer;
        public NativeArray<byte> EdgeBuffer => _edgeBuffer;
        
        // İYİLEŞTİRME: Source keyword'leri
        private static readonly string[] SourceKeywords = { "_SOURCE_LUMINANCE", "_SOURCE_DEPTH", "_SOURCE_NORMAL", "_SOURCE_COMBINED" };
        
        public EdgeFrameProcessor()
        {
            var shader = Shader.Find("Custom/EdgeDirection");
            if (shader != null)
                _gradientMaterial = new Material(shader);
        }
        
        /// <summary>
        /// Her frame çağrılır, GPU'daki texture'ı işler ve async readback başlatır
        /// 
        /// İYİLEŞTİRME: edgeSource parametresi eklendi — gradient shader'a kaynak türünü iletir
        /// </summary>
        public void ProcessFrame(RenderTexture sourceRT, float edgeThreshold, int downsample, int updateInterval,
            int edgeSourceIndex = 0, float depthSensitivity = 10f, float maxDepth = 50f, float normalSensitivity = 1f,
            float depthWeight = 0.5f, float normalWeight = 0.5f, float colorWeight = 0.3f)
        {
            if (sourceRT == null || _disposed || _readbackPending) return;
            
            _frameCounter++;
            
            if (_frameCounter % updateInterval != 0) return;
            
            int targetWidth = sourceRT.width / downsample;
            int targetHeight = sourceRT.height / downsample;
            
            EnsureResources(targetWidth, targetHeight);
            
            Graphics.Blit(sourceRT, _downsampledRT);
            
            if (_gradientMaterial != null)
            {
                _gradientMaterial.SetFloat("_EdgeThreshold", edgeThreshold * 0.5f);
                
                // İYİLEŞTİRME: Kaynak parametrelerini gradient shader'a geçir
                // EdgeDetection.shader ile tutarlılık sağlanır
                foreach (var kw in SourceKeywords) _gradientMaterial.DisableKeyword(kw);
                int sourceIdx = Mathf.Clamp(edgeSourceIndex, 0, SourceKeywords.Length - 1);
                _gradientMaterial.EnableKeyword(SourceKeywords[sourceIdx]);
                
                _gradientMaterial.SetFloat("_DepthSensitivity", depthSensitivity);
                _gradientMaterial.SetFloat("_MaxDepth", maxDepth);
                _gradientMaterial.SetFloat("_NormalSensitivity", normalSensitivity);
                _gradientMaterial.SetFloat("_DepthWeight", depthWeight);
                _gradientMaterial.SetFloat("_NormalWeight", normalWeight);
                _gradientMaterial.SetFloat("_ColorWeight", colorWeight);
                
                Graphics.Blit(sourceRT, _gradientRT, _gradientMaterial);
            }
            
            StartReadback(targetWidth, targetHeight, edgeThreshold);
        }
        
        // ÖNCEKİ İMZAYI KORUYORUZ (geriye uyumluluk)
        // Bu overload eskisi gibi çalışır — varsayılan Luminance modu
        public void ProcessFrame(RenderTexture sourceRT, float edgeThreshold, int downsample, int updateInterval)
        {
            ProcessFrame(sourceRT, edgeThreshold, downsample, updateInterval, 0);
        }
        
        private void EnsureResources(int width, int height)
        {
            if (_downsampledRT == null || _downsampledRT.width != width)
            {
                if (_downsampledRT != null) _downsampledRT.Release();
                _downsampledRT = new RenderTexture(width, height, 0, RenderTextureFormat.ARGB32);
                _downsampledRT.filterMode = FilterMode.Point;
                _downsampledRT.Create();
            }
            
            if (_gradientRT == null || _gradientRT.width != width)
            {
                if (_gradientRT != null) _gradientRT.Release();
                _gradientRT = new RenderTexture(width, height, 0, RenderTextureFormat.ARGBFloat);
                _gradientRT.filterMode = FilterMode.Point;
                _gradientRT.Create();
            }
            
            int pixelCount = width * height;
            EnsureBuffer(ref _edgeBuffer, pixelCount);
            EnsureBuffer(ref _gradientXBuffer, pixelCount);
            EnsureBuffer(ref _gradientYBuffer, pixelCount);
            EnsureBuffer(ref _magnitudeBuffer, pixelCount);
            
            if (!_edgePixels.IsCreated)
                _edgePixels = new NativeArray<Vector2Int>(MAX_EDGE_PIXELS, Allocator.Persistent);
        }
        
        private void EnsureBuffer<T>(ref NativeArray<T> buffer, int size) where T : struct
        {
            if (!buffer.IsCreated || buffer.Length != size)
            {
                if (buffer.IsCreated) buffer.Dispose();
                buffer = new NativeArray<T>(size, Allocator.Persistent);
            }
        }
        
        private void StartReadback(int width, int height, float threshold)
        {
            _width = width;
            _height = height;
            _readbackPending = true;
            _dataReady = false;
            _pendingReadbackCount = 2;
            
            AsyncGPUReadback.Request(_downsampledRT, 0, TextureFormat.RGBA32, 
                req => OnEdgeReadback(req, width, height, threshold));
            
            AsyncGPUReadback.Request(_gradientRT, 0, TextureFormat.RGBAFloat, 
                req => OnGradientReadback(req, width, height));
        }
        
        private void OnEdgeReadback(AsyncGPUReadbackRequest request, int width, int height, float threshold)
        {
            try
            {
                if (_disposed || !_edgeBuffer.IsCreated || request.hasError) 
                { 
                    CompleteReadback(); 
                    return; 
                }
                
                var data = request.GetData<Color32>();
                int count = Mathf.Min(data.Length, _edgeBuffer.Length);
                
                for (int i = 0; i < count; i++)
                    _edgeBuffer[i] = data[i].r;
                
                CollectEdgePixels(width, height, threshold);
                CompleteReadback();
            }
            catch { }
        }
        
        private void OnGradientReadback(AsyncGPUReadbackRequest request, int width, int height)
        {
            try
            {
                if (_disposed || !_gradientXBuffer.IsCreated || request.hasError) 
                { 
                    CompleteReadback(); 
                    return; 
                }
                
                var data = request.GetData<Color>();
                int count = Mathf.Min(data.Length, _gradientXBuffer.Length);
                
                for (int i = 0; i < count; i++)
                {
                    float gx = data[i].r * 2f - 1f;
                    float gy = data[i].g * 2f - 1f;
                    _gradientXBuffer[i] = gx;
                    _gradientYBuffer[i] = gy;
                    _magnitudeBuffer[i] = Mathf.Sqrt(gx * gx + gy * gy);
                }
                CompleteReadback();
            }
            catch { }
        }
        
        private void CompleteReadback()
        {
            _pendingReadbackCount--;
            if (_pendingReadbackCount <= 0)
            {
                _readbackPending = false;
                _dataReady = true;
            }
        }
        
        private void CollectEdgePixels(int width, int height, float threshold)
        {
            _edgePixelCount = 0;
            byte thresholdByte = (byte)(threshold * 255f);
            
            for (int y = 0; y < height && _edgePixelCount < MAX_EDGE_PIXELS; y++)
            {
                for (int x = 0; x < width && _edgePixelCount < MAX_EDGE_PIXELS; x++)
                {
                    if (_edgeBuffer[y * width + x] > thresholdByte)
                        _edgePixels[_edgePixelCount++] = new Vector2Int(x, y);
                }
            }
        }
        
        public NativeArray<Vector2Int>.ReadOnly GetEdgePixels() => _edgePixels.AsReadOnly();
        
        public Vector2 GetGradientAt(int x, int y)
        {
            if (!_dataReady || x < 0 || x >= _width || y < 0 || y >= _height) return Vector2.zero;
            int idx = y * _width + x;
            return new Vector2(_gradientXBuffer[idx], _gradientYBuffer[idx]);
        }
        
        public float GetMagnitudeAt(int x, int y)
        {
            if (!_dataReady || x < 0 || x >= _width || y < 0 || y >= _height) return 0f;
            return _magnitudeBuffer[y * _width + x];
        }
        
        public bool IsEdgeAt(int x, int y, float threshold)
        {
            if (!_dataReady || x < 0 || x >= _width || y < 0 || y >= _height) return false;
            return _edgeBuffer[y * _width + x] > (byte)(threshold * 255f);
        }
        
        public Vector2Int ScreenToTextureCoords(Vector2 screenPos, int screenWidth, int screenHeight)
        {
            float nx = screenPos.x / screenWidth;
            float ny = 1f - (screenPos.y / screenHeight);
            return new Vector2Int(
                Mathf.Clamp((int)(nx * _width), 0, _width - 1),
                Mathf.Clamp((int)(ny * _height), 0, _height - 1)
            );
        }
        
        public Vector2 TextureToScreenCoords(Vector2 texCoords, int screenWidth, int screenHeight)
        {
            return new Vector2(
                (texCoords.x / _width) * screenWidth,
                (1f - texCoords.y / _height) * screenHeight
            );
        }
        
        public void Dispose()
        {
            _disposed = true;
            if (_edgeBuffer.IsCreated) _edgeBuffer.Dispose();
            if (_gradientXBuffer.IsCreated) _gradientXBuffer.Dispose();
            if (_gradientYBuffer.IsCreated) _gradientYBuffer.Dispose();
            if (_magnitudeBuffer.IsCreated) _magnitudeBuffer.Dispose();
            if (_edgePixels.IsCreated) _edgePixels.Dispose();
            if (_downsampledRT != null) { _downsampledRT.Release(); _downsampledRT = null; }
            if (_gradientRT != null) { _gradientRT.Release(); _gradientRT = null; }
            if (_gradientMaterial != null)
            {
                if (Application.isPlaying) UnityEngine.Object.Destroy(_gradientMaterial);
                else UnityEngine.Object.DestroyImmediate(_gradientMaterial);
            }
        }
    }
}