using UnityEngine;
using UnityEngine.Rendering;
using Unity.Collections;
using System;

namespace SceneCapture.Hough
{
    /// <summary>
    /// GPU'daki edge texture'ını CPU'ya asenkron aktarır.
    /// GPU'da hesaplanan edge ve gradient verilerini ana thread'i bloklamadan CPU'ya taşır.
    /// </summary>
    public class EdgeFrameProcessor : IDisposable
    {
        // CPU tarafında tutulan bufferlar - GPU'dan gelen veriler buraya kopyalanır
        private NativeArray<byte> _edgeBuffer;           // Edge map: 0=edge yok, 255=edge var
        private NativeArray<float> _gradientXBuffer;     // X yönünde gradient değerleri (-1 to +1)
        private NativeArray<float> _gradientYBuffer;     // Y yönünde gradient değerleri (-1 to +1)
        private NativeArray<float> _magnitudeBuffer;     // Gradient magnitude: sqrt(gx² + gy²)
        private NativeArray<Vector2Int> _edgePixels;     // Edge olan pixellerin koordinat listesi
        
        // Texture boyutları ve state bilgileri
        private int _width, _height;                     // Downsampled texture boyutları
        private bool _readbackPending;                   // Async readback bekliyor mu?
        private bool _dataReady;                         // Data kullanıma hazır mı?
        private bool _disposed;                          // Dispose edildi mi?
        private int _frameCounter;                       // Frame sayacı (updateInterval için)
        private int _pendingReadbackCount;               // Kaç readback bekleniyor (edge + gradient = 2)
        
        // GPU resources
        private RenderTexture _downsampledRT;            // Küçültülmüş edge texture
        private RenderTexture _gradientRT;               // Gradient bilgisi içeren texture (float)
        private Material _gradientMaterial;              // EdgeDirection shader için material
        
        // Edge pixel limitleri
        private const int MAX_EDGE_PIXELS = 100000;      // Maksimum edge pixel sayısı
        private int _edgePixelCount;                     // Gerçekte bulunan edge pixel sayısı
        
        // Public propertyler - dışarıdan okunabilir
        public bool IsDataReady => _dataReady && !_disposed;
        public int Width => _width;
        public int Height => _height;
        public int EdgePixelCount => _edgePixelCount;
        
        // Burst job'lar için NativeArray erişimi
        public NativeArray<float> GradientXBuffer => _gradientXBuffer;
        public NativeArray<float> GradientYBuffer => _gradientYBuffer;
        public NativeArray<float> MagnitudeBuffer => _magnitudeBuffer;
        public NativeArray<byte> EdgeBuffer => _edgeBuffer;
        
        public EdgeFrameProcessor()
        {
            // EdgeDirection shader'ını bulup material oluştur
            var shader = Shader.Find("Custom/EdgeDirection");
            if (shader != null)
                _gradientMaterial = new Material(shader);
        }
        
        /// <summary>
        /// Her frame çağrılır, GPU'daki texture'ı işler ve async readback başlatır
        /// </summary>
        public void ProcessFrame(RenderTexture sourceRT, float edgeThreshold, int downsample, int updateInterval)
        {
            // Null check, disposed check ve pending check
            if (sourceRT == null || _disposed || _readbackPending) return;
            
            // Frame sayacını artır
            _frameCounter++;
            
            // updateInterval kontrolü: Her N frame'de bir işle
            if (_frameCounter % updateInterval != 0) return;
            
            // Hedef resolution hesapla (downsampling ile küçültme)
            int targetWidth = sourceRT.width / downsample;
            int targetHeight = sourceRT.height / downsample;
            
            // RenderTexture'ları ve NativeArray'leri hazırla
            EnsureResources(targetWidth, targetHeight);
            
            // Source RT'yi downsampled RT'ye kopyala (GPU'da hızlı)
            Graphics.Blit(sourceRT, _downsampledRT);
            
            // Gradient hesaplama shader'ını çalıştır
            if (_gradientMaterial != null)
            {
                // Threshold parametresini shader'a gönder
                _gradientMaterial.SetFloat("_EdgeThreshold", edgeThreshold * 0.5f);
                
                // Shader ile gradient texture oluştur (R=gx, G=gy, B=magnitude)
                Graphics.Blit(sourceRT, _gradientRT, _gradientMaterial);
            }
            
            // Async GPU→CPU transfer başlat
            StartReadback(targetWidth, targetHeight, edgeThreshold);
        }
        
        /// <summary>
        /// RenderTexture'ları ve NativeArray bufferları hazırla/yeniden oluştur
        /// </summary>
        private void EnsureResources(int width, int height)
        {
            // Downsampled RT kontrolü - yoksa veya boyut değişmişse yeniden oluştur
            if (_downsampledRT == null || _downsampledRT.width != width)
            {
                if (_downsampledRT != null) _downsampledRT.Release();
                _downsampledRT = new RenderTexture(width, height, 0, RenderTextureFormat.ARGB32);
                _downsampledRT.filterMode = FilterMode.Point;  // Nearest neighbor filtering
                _downsampledRT.Create();
            }
            
            // Gradient RT kontrolü - float precision gerekli
            if (_gradientRT == null || _gradientRT.width != width)
            {
                if (_gradientRT != null) _gradientRT.Release();
                _gradientRT = new RenderTexture(width, height, 0, RenderTextureFormat.ARGBFloat);
                _gradientRT.filterMode = FilterMode.Point;
                _gradientRT.Create();
            }
            
            // NativeArray bufferları kontrol et ve gerekirse yeniden oluştur
            int pixelCount = width * height;
            EnsureBuffer(ref _edgeBuffer, pixelCount);
            EnsureBuffer(ref _gradientXBuffer, pixelCount);
            EnsureBuffer(ref _gradientYBuffer, pixelCount);
            EnsureBuffer(ref _magnitudeBuffer, pixelCount);
            
            // Edge pixel listesi - sadece bir kere oluştur
            if (!_edgePixels.IsCreated)
                _edgePixels = new NativeArray<Vector2Int>(MAX_EDGE_PIXELS, Allocator.Persistent);
        }
        
        /// <summary>
        /// NativeArray buffer'ı kontrol et, yoksa/yanlış boyuttaysa oluştur
        /// </summary>
        private void EnsureBuffer<T>(ref NativeArray<T> buffer, int size) where T : struct
        {
            if (!buffer.IsCreated || buffer.Length != size)
            {
                if (buffer.IsCreated) buffer.Dispose();  // Eski buffer'ı temizle
                buffer = new NativeArray<T>(size, Allocator.Persistent);
            }
        }
        
        /// <summary>
        /// İki ayrı async readback başlat: edge texture ve gradient texture
        /// </summary>
        private void StartReadback(int width, int height, float threshold)
        {
            _width = width;
            _height = height;
            _readbackPending = true;    // Readback bekliyor flag'i
            _dataReady = false;         // Data henüz hazır değil
            _pendingReadbackCount = 2;  // 2 readback bekliyoruz (edge + gradient)
            
            // Readback 1: Edge texture (RGBA32 formatında)
            AsyncGPUReadback.Request(_downsampledRT, 0, TextureFormat.RGBA32, 
                req => OnEdgeReadback(req, width, height, threshold));
            
            // Readback 2: Gradient texture (RGBAFloat formatında)
            AsyncGPUReadback.Request(_gradientRT, 0, TextureFormat.RGBAFloat, 
                req => OnGradientReadback(req, width, height));
        }
        
        /// <summary>
        /// Edge texture readback tamamlandığında çağrılır
        /// </summary>
        private void OnEdgeReadback(AsyncGPUReadbackRequest request, int width, int height, float threshold)
        {
            try
            {
                // Hata kontrolü
                if (_disposed || !_edgeBuffer.IsCreated || request.hasError) 
                { 
                    CompleteReadback(); 
                    return; 
                }
                
                // GPU'dan gelen data'yı al
                var data = request.GetData<Color32>();
                int count = Mathf.Min(data.Length, _edgeBuffer.Length);
                
                // Sadece R kanalını kopyala (0 veya 255)
                for (int i = 0; i < count; i++)
                    _edgeBuffer[i] = data[i].r;
                
                // Edge pixel listesi oluştur
                CollectEdgePixels(width, height, threshold);
                CompleteReadback();
            }
            catch { }
        }
        
        /// <summary>
        /// Gradient texture readback tamamlandığında çağrılır
        /// </summary>
        private void OnGradientReadback(AsyncGPUReadbackRequest request, int width, int height)
        {
            try
            {
                if (_disposed || !_gradientXBuffer.IsCreated || request.hasError) 
                { 
                    CompleteReadback(); 
                    return; 
                }
                
                // GPU'dan gradient data'yı al
                var data = request.GetData<Color>();
                int count = Mathf.Min(data.Length, _gradientXBuffer.Length);
                
                // Her pixel için gradient decode et (0-1 → -1 to +1)
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
        
        /// <summary>
        /// Readback counter azalt, her ikisi de bitince data hazır
        /// </summary>
        private void CompleteReadback()
        {
            _pendingReadbackCount--;
            if (_pendingReadbackCount <= 0)
            {
                _readbackPending = false;
                _dataReady = true;
            }
        }
        
        /// <summary>
        /// Threshold'dan büyük pixelleri listele (Hough için)
        /// </summary>
        private void CollectEdgePixels(int width, int height, float threshold)
        {
            _edgePixelCount = 0;
            byte thresholdByte = (byte)(threshold * 255f);
            
            // Tüm pixelleri tara
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
        
        /// <summary>
        /// Screen koordinatı → Texture koordinatı (Y flip)
        /// </summary>
        public Vector2Int ScreenToTextureCoords(Vector2 screenPos, int screenWidth, int screenHeight)
        {
            float nx = screenPos.x / screenWidth;
            float ny = 1f - (screenPos.y / screenHeight);  // Y flip
            return new Vector2Int(
                Mathf.Clamp((int)(nx * _width), 0, _width - 1),
                Mathf.Clamp((int)(ny * _height), 0, _height - 1)
            );
        }
        
        /// <summary>
        /// Texture koordinatı → Screen koordinatı (Y flip)
        /// </summary>
        public Vector2 TextureToScreenCoords(Vector2 texCoords, int screenWidth, int screenHeight)
        {
            return new Vector2(
                (texCoords.x / _width) * screenWidth,
                (1f - texCoords.y / _height) * screenHeight
            );
        }
        
        /// <summary>
        /// Tüm resources temizle (memory leak önleme)
        /// </summary>
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