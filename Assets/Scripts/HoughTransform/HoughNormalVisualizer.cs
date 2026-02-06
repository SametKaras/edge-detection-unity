using UnityEngine;
using Unity.Collections;
using Unity.Jobs;
using Unity.Burst;
using Unity.Mathematics;

namespace SceneCapture.Hough
{
    /// <summary>
    /// Ana visualizer sınıfı - Hough Transform ve kernel karşılaştırması
    /// 
    /// İYİLEŞTİRME:
    ///   ProcessFrame çağrısında EdgeDetectionEffect'in kaynak parametreleri
    ///   (source, depthSensitivity, normalSensitivity, weights) EdgeFrameProcessor'a iletiliyor.
    ///   Bu sayede gradient yönü ile edge algılama kaynağı arasında tutarlılık sağlanıyor.
    /// </summary>
    [DefaultExecutionOrder(-100)]
    [RequireComponent(typeof(Camera))]
    public class HoughNormalVisualizer : MonoBehaviour
    {
        // ==================== INSPECTOR PARAMETRELERİ ====================
        
        [Header("Hough Parametreleri")]
        public HoughParameters houghParams = new HoughParameters();
        
        [Header("Edge Algılama")]
        [Range(0.01f, 1f)]
        public float edgeThreshold = 0.1f;
        
        // İYİLEŞTİRME: EdgeDetectionEffect referansı — kaynak parametrelerini almak için
        [Header("Edge Source Senkronizasyonu")]
        [Tooltip("EdgeDetectionEffect bileşeni. Atanmazsa aynı GameObject'te aranır.")]
        public EdgeDetectionEffect edgeDetectionEffect;
        
        [Header("Normal Hesaplama")]
        public bool useLocalGradient = true;
        [Range(2, 30)]
        public int localGradientRadius = 8;
        public bool fallbackToLocalGradient = true;
        
        [Header("Görselleştirme")]
        public float arrowLength = 60f;
        public Color segmentColor = new Color(0f, 1f, 1f, 0.5f);
        public float hoverRadius = 25f;
        public bool showAllSegments = false;
        
        // ==================== KERNEL KARŞILAŞTIRMA ====================
        
        [Header("Kernel Karşılaştırma")]
        [Tooltip("5x5 ve 7x7 kernel karşılaştırmasını aktif et")]
        public bool enableKernelComparison = true;
        
        [Tooltip("Hough normalini göster (beyaz)")]
        public bool showHoughNormal = true;
        
        [Tooltip("5x5 kernel normalini göster (magenta)")]
        public bool showKernel5x5 = true;
        
        [Tooltip("7x7 kernel normalini göster (sarı)")]
        public bool showKernel7x7 = true;
        
        [Tooltip("İstatistik panelini göster")]
        public bool showStatsPanel = true;
        
        // ==================== SEGMENT DEBUG ====================
        
        [Header("Segment Debug")]
        [Tooltip("Segment debug panelini göster")]
        public bool showSegmentDebug = true;
        
        [Tooltip("Edge detection'da kullanılan kernel")]
        public EdgeKernel edgeDetectionKernel = EdgeKernel.Sobel;
        
        [Tooltip("Düşük kaplama uyarı eşiği")]
        [Range(0.1f, 0.5f)]
        public float lowCoverageThreshold = 0.3f;
        
        [Tooltip("Düşük tutarlılık uyarı eşiği")]
        [Range(0.3f, 0.7f)]
        public float lowConsistencyThreshold = 0.5f;
        
        public enum EdgeKernel { Sobel, Roberts, Prewitt }
        
        // ==================== RENKLER ====================
        private static readonly Color COLOR_HOUGH = Color.white;
        private static readonly Color COLOR_5X5 = new Color(1f, 0.4f, 1f);
        private static readonly Color COLOR_7X7 = new Color(1f, 1f, 0.3f);
        private static readonly Color COLOR_WARNING = new Color(1f, 0.3f, 0.3f);
        private static readonly Color COLOR_OK = new Color(0.3f, 1f, 0.3f);
        
        // ==================== BİLEŞENLER ====================
        private Camera _camera;
        private EdgeFrameProcessor _processor;
        private HoughLineDetector _detector;
        
        private LineSegment2D? _hoveredSegment;
        private Vector2 _mousePos;
        private Vector2 _currentNormal;
        private bool _isInitialized;
        
        private RenderTexture _edgeSourceRT;
        private Texture2D _whiteTex;
        private Texture2D _boxTex;
        
        // ==================== KERNEL BUFFER'LARI ====================
        private NativeArray<float2> _kernel5x5Normals;
        private NativeArray<float2> _kernel7x7Normals;
        private bool _kernelBuffersReady;
        private int _kernelBufferWidth, _kernelBufferHeight;
        
        // ==================== İSTATİSTİKLER ====================
        private float _avgError5x5, _avgError7x7;
        private float _computeTime5x5, _computeTime7x7;
        private System.Diagnostics.Stopwatch _stopwatch = new System.Diagnostics.Stopwatch();
        
        // ==================== GUI STYLES ====================
        private GUIStyle _boxStyle, _labelStyle, _titleStyle, _statsStyle;
        private bool _stylesInitialized;
        
        // ==================== UNITY YAŞAM DÖNGÜSÜ ====================
        
        void OnEnable()
        {
            _camera = GetComponent<Camera>();
            _camera.depthTextureMode |= DepthTextureMode.Depth | DepthTextureMode.DepthNormals;
            
            _processor = new EdgeFrameProcessor();
            _detector = new HoughLineDetector(houghParams);
            
            _whiteTex = new Texture2D(1, 1);
            _whiteTex.SetPixel(0, 0, Color.white);
            _whiteTex.Apply();
            
            _boxTex = new Texture2D(1, 1);
            _boxTex.SetPixel(0, 0, new Color(0, 0, 0, 0.85f));
            _boxTex.Apply();
            
            // İYİLEŞTİRME: EdgeDetectionEffect'i otomatik bul
            if (edgeDetectionEffect == null)
                edgeDetectionEffect = GetComponent<EdgeDetectionEffect>();
            
            _isInitialized = true;
        }
        
        void OnDisable()
        {
            _processor?.Dispose();
            _detector?.Dispose();
            DisposeKernelBuffers();
            
            if (_edgeSourceRT != null) { _edgeSourceRT.Release(); _edgeSourceRT = null; }
            if (_whiteTex != null) { if (Application.isPlaying) Destroy(_whiteTex); else DestroyImmediate(_whiteTex); }
            if (_boxTex != null) { if (Application.isPlaying) Destroy(_boxTex); else DestroyImmediate(_boxTex); }
            
            _isInitialized = false;
        }
        
        void Update()
        {
            _detector?.UpdateParameters(houghParams);
        }
        
        void OnRenderImage(RenderTexture src, RenderTexture dst)
        {
            if (!_isInitialized) { Graphics.Blit(src, dst); return; }
            
            if (_edgeSourceRT == null || _edgeSourceRT.width != src.width || _edgeSourceRT.height != src.height)
            {
                if (_edgeSourceRT != null) _edgeSourceRT.Release();
                _edgeSourceRT = new RenderTexture(src.width, src.height, 0, RenderTextureFormat.ARGBFloat);
                _edgeSourceRT.filterMode = FilterMode.Point;
                _edgeSourceRT.Create();
            }
            
            Graphics.Blit(src, _edgeSourceRT);
            
            // İYİLEŞTİRME: EdgeDetectionEffect'ten kaynak parametrelerini al ve processor'a geçir
            if (edgeDetectionEffect != null)
            {
                _processor.ProcessFrame(
                    _edgeSourceRT, edgeThreshold,
                    houghParams.downsampleFactor, houghParams.updateInterval,
                    (int)edgeDetectionEffect.source,     // Edge source türü
                    edgeDetectionEffect.depthSensitivity,
                    edgeDetectionEffect.maxDepth,
                    edgeDetectionEffect.normalSensitivity,
                    edgeDetectionEffect.depthWeight,
                    edgeDetectionEffect.normalWeight,
                    edgeDetectionEffect.colorWeight
                );
            }
            else
            {
                // Fallback: Eski davranış (sadece Luminance)
                _processor.ProcessFrame(_edgeSourceRT, edgeThreshold, houghParams.downsampleFactor, houghParams.updateInterval);
            }
            
            if (_processor.IsDataReady && _processor.EdgePixelCount > 0)
            {
                _detector.DetectLines(
                    _processor.GetEdgePixels(),
                    _processor.EdgePixelCount,
                    _processor.Width,
                    _processor.Height,
                    _processor,
                    (int)edgeDetectionKernel
                );
                
                if (enableKernelComparison)
                    ProcessKernels();
            }
            
            Graphics.Blit(src, dst);
        }
        
        // ==================== KERNEL HESAPLAMA ====================
        
        private void ProcessKernels()
        {
            int w = _processor.Width;
            int h = _processor.Height;
            int pixelCount = w * h;
            
            if (!_kernelBuffersReady || _kernelBufferWidth != w || _kernelBufferHeight != h)
            {
                DisposeKernelBuffers();
                _kernel5x5Normals = new NativeArray<float2>(pixelCount, Allocator.Persistent);
                _kernel7x7Normals = new NativeArray<float2>(pixelCount, Allocator.Persistent);
                _kernelBufferWidth = w;
                _kernelBufferHeight = h;
                _kernelBuffersReady = true;
            }
            
            _stopwatch.Restart();
            new KernelJob {
                gradX = _processor.GradientXBuffer,
                gradY = _processor.GradientYBuffer,
                normals = _kernel5x5Normals,
                width = w, height = h,
                radius = 2
            }.Schedule(pixelCount, 64).Complete();
            _computeTime5x5 = (float)_stopwatch.Elapsed.TotalMilliseconds;
            
            _stopwatch.Restart();
            new KernelJob {
                gradX = _processor.GradientXBuffer,
                gradY = _processor.GradientYBuffer,
                normals = _kernel7x7Normals,
                width = w, height = h,
                radius = 3
            }.Schedule(pixelCount, 64).Complete();
            _computeTime7x7 = (float)_stopwatch.Elapsed.TotalMilliseconds;
            
            CalculateStatistics();
        }
        
        private void CalculateStatistics()
        {
            if (_detector.LineCount == 0) return;
            
            float errorSum5x5 = 0, errorSum7x7 = 0;
            int validCount = 0;
            
            foreach (var segment in _detector.DetectedLines)
            {
                float houghAngle = Mathf.Atan2(segment.normal.y, segment.normal.x) * Mathf.Rad2Deg;
                
                int samples = Mathf.Max(5, (int)segment.Length / 2);
                for (int s = 0; s < samples; s++)
                {
                    float t = s / (float)(samples - 1);
                    Vector2 point = Vector2.Lerp(segment.startPoint, segment.endPoint, t);
                    
                    int x = Mathf.Clamp(Mathf.RoundToInt(point.x), 0, _kernelBufferWidth - 1);
                    int y = Mathf.Clamp(Mathf.RoundToInt(point.y), 0, _kernelBufferHeight - 1);
                    int idx = y * _kernelBufferWidth + x;
                    
                    var n5 = _kernel5x5Normals[idx];
                    if (math.lengthsq(n5) > 0.01f)
                    {
                        float angle5 = math.degrees(math.atan2(n5.y, n5.x));
                        float diff5 = CalculateAngleDiff(angle5, houghAngle);
                        errorSum5x5 += diff5;
                    }
                    
                    var n7 = _kernel7x7Normals[idx];
                    if (math.lengthsq(n7) > 0.01f)
                    {
                        float angle7 = math.degrees(math.atan2(n7.y, n7.x));
                        float diff7 = CalculateAngleDiff(angle7, houghAngle);
                        errorSum7x7 += diff7;
                    }
                    
                    validCount++;
                }
            }
            
            _avgError5x5 = validCount > 0 ? errorSum5x5 / validCount : 0;
            _avgError7x7 = validCount > 0 ? errorSum7x7 / validCount : 0;
        }
        
        private float CalculateAngleDiff(float angle1, float angle2)
        {
            float diff = Mathf.Abs(angle1 - angle2);
            if (diff > 180) diff = 360 - diff;
            if (diff > 90) diff = 180 - diff;
            return diff;
        }
        
        private void DisposeKernelBuffers()
        {
            if (_kernel5x5Normals.IsCreated) _kernel5x5Normals.Dispose();
            if (_kernel7x7Normals.IsCreated) _kernel7x7Normals.Dispose();
            _kernelBuffersReady = false;
        }
        
        // ==================== GUI ====================
        
        void OnGUI()
        {
            if (!_isInitialized || !_processor.IsDataReady) return;
            
            if (!_stylesInitialized)
            {
                _boxStyle = new GUIStyle { normal = { background = _boxTex } };
                _labelStyle = new GUIStyle(GUI.skin.label) { fontSize = 12, fontStyle = FontStyle.Bold };
                _labelStyle.normal.textColor = Color.white;
                _titleStyle = new GUIStyle(GUI.skin.label) { fontSize = 12, fontStyle = FontStyle.Bold, alignment = TextAnchor.MiddleCenter };
                _titleStyle.normal.textColor = Color.white;
                _statsStyle = new GUIStyle(GUI.skin.label) { fontSize = 11 };
                _stylesInitialized = true;
            }
            
            var e = Event.current;
            if (e == null) return;
            
            _mousePos = e.mousePosition;
            
            DrawStatusBar();
            
            if (enableKernelComparison && showStatsPanel && _kernelBuffersReady)
                DrawStatsPanel();
            
            if (showAllSegments) DrawAllSegments();
            
            FindHoveredSegment();
            
            var texCoord = _processor.ScreenToTextureCoords(_mousePos, Screen.width, Screen.height);
            bool isOnEdge = _processor.IsEdgeAt(texCoord.x, texCoord.y, edgeThreshold * 0.5f);
            
            if (_hoveredSegment.HasValue)
            {
                DrawHoveredSegment();
                DrawNormalArrows(texCoord);
                
                if (showSegmentDebug)
                    DrawSegmentDebugPanel(texCoord);
            }
            else if (fallbackToLocalGradient && isOnEdge)
            {
                DrawFallbackNormals(texCoord);
            }
        }
        
        private void DrawStatusBar()
        {
            string status = $"Segment: {_detector.LineCount} | Edge: {_processor.EdgePixelCount}";
            GUI.Box(new Rect(10, 10, 280, 25), "", _boxStyle);
            GUI.Label(new Rect(15, 12, 270, 25), status, _labelStyle);
        }
        
        private void DrawStatsPanel()
        {
            float panelWidth = 180;
            float panelHeight = 95;
            float x = Screen.width - panelWidth - 10;
            float y = 10;
            
            GUI.Box(new Rect(x, y, panelWidth, panelHeight), "", _boxStyle);
            GUI.Label(new Rect(x, y + 5, panelWidth, 20), "KERNEL KARŞILAŞTIRMA", _titleStyle);
            
            float lineY = y + 30;
            
            GUI.color = COLOR_5X5;
            GUI.DrawTexture(new Rect(x + 10, lineY + 4, 12, 12), _whiteTex);
            GUI.color = Color.white;
            _statsStyle.normal.textColor = COLOR_5X5;
            GUI.Label(new Rect(x + 28, lineY, panelWidth - 35, 20), 
                $"5x5: Δ{_avgError5x5:F1}° | {_computeTime5x5:F2}ms", _statsStyle);
            
            lineY += 22;
            
            GUI.color = COLOR_7X7;
            GUI.DrawTexture(new Rect(x + 10, lineY + 4, 12, 12), _whiteTex);
            GUI.color = Color.white;
            _statsStyle.normal.textColor = COLOR_7X7;
            GUI.Label(new Rect(x + 28, lineY, panelWidth - 35, 20), 
                $"7x7: Δ{_avgError7x7:F1}° | {_computeTime7x7:F2}ms", _statsStyle);
            
            lineY += 22;
            
            _statsStyle.normal.textColor = Color.gray;
            string better = _avgError5x5 < _avgError7x7 ? "5x5 daha iyi" : "7x7 daha iyi";
            GUI.Label(new Rect(x + 10, lineY, panelWidth - 20, 20), better, _statsStyle);
        }
        
        private void DrawSegmentDebugPanel(Vector2Int texCoord)
        {
            var seg = _hoveredSegment.Value;
            
            float houghAngle = Mathf.Atan2(seg.normal.y, seg.normal.x) * Mathf.Rad2Deg;
            
            bool lowCoverage = seg.edgeCoverage < lowCoverageThreshold;
            bool lowConsistency = seg.directionConsistency < lowConsistencyThreshold;
            bool shortSegment = seg.Length < houghParams.segmentMinLength * 1.5f;
            bool hasIssue = lowCoverage || lowConsistency || shortSegment;
            
            string kernelName = seg.GetKernelName();
            Color kernelColor = seg.GetKernelColor();
            string info = "═══ SEGMENT DEBUG ═══\n\n";
            info += $"Kaynak Kernel: {kernelName}\n";
            info += $"─────────────────────\n";
            info += $"Uzunluk: {seg.Length:F1} px\n";
            info += $"Piksel Sayısı: {seg.supportingPixelCount}\n";
            info += $"Edge Kaplama: {seg.edgeCoverage:F2}\n";
            info += $"Yön Tutarlılığı: {seg.directionConsistency:F2}\n";
            info += $"Hough Açısı: {houghAngle:F1}°\n";
            info += $"Score: {seg.score}\n";
            
            if (enableKernelComparison && _kernelBuffersReady)
            {
                int idx = texCoord.y * _kernelBufferWidth + texCoord.x;
                if (idx >= 0 && idx < _kernel5x5Normals.Length)
                {
                    info += "\n─── Kernel Açıları ───\n";
                    
                    var n5 = _kernel5x5Normals[idx];
                    var n7 = _kernel7x7Normals[idx];
                    
                    if (math.lengthsq(n5) > 0.01f)
                    {
                        float angle5 = math.degrees(math.atan2(n5.y, n5.x));
                        float diff5 = CalculateAngleDiff(angle5, houghAngle);
                        info += $"5x5: {angle5:F1}° (Δ{diff5:F1}°)\n";
                    }
                    
                    if (math.lengthsq(n7) > 0.01f)
                    {
                        float angle7 = math.degrees(math.atan2(n7.y, n7.x));
                        float diff7 = CalculateAngleDiff(angle7, houghAngle);
                        info += $"7x7: {angle7:F1}° (Δ{diff7:F1}°)\n";
                    }
                }
            }
            
            if (hasIssue)
            {
                info += "\n─── ⚠ UYARILAR ───\n";
                if (lowCoverage) info += "• Düşük edge kaplama!\n";
                if (lowConsistency) info += "• Tutarsız gradient yönleri!\n";
                if (shortSegment) info += "• Çok kısa segment!\n";
            }
            else
            {
                info += "\n✓ Segment sağlıklı görünüyor";
            }
            
            var size = _statsStyle.CalcSize(new GUIContent(info));
            float panelWidth = Mathf.Max(size.x + 20, 200);
            float panelHeight = size.y + 16;
            
            float x = _mousePos.x + 30;
            float y = _mousePos.y + 30;
            
            if (x + panelWidth > Screen.width) x = _mousePos.x - panelWidth - 30;
            if (y + panelHeight > Screen.height) y = _mousePos.y - panelHeight - 30;
            
            GUI.Box(new Rect(x, y, panelWidth, panelHeight), "", _boxStyle);
            
            GUI.color = hasIssue ? COLOR_WARNING : COLOR_OK;
            GUI.DrawTexture(new Rect(x, y, panelWidth, 3), _whiteTex);
            GUI.color = Color.white;
            
            _statsStyle.normal.textColor = Color.white;
            GUI.Label(new Rect(x + 10, y + 8, size.x, size.y), info, _statsStyle);
        }
        
        private void DrawNormalArrows(Vector2Int texCoord)
        {
            var seg = _hoveredSegment.Value;
            Vector2 projectedPoint = seg.ProjectPoint(new Vector2(texCoord.x, texCoord.y));
            Vector2 screenBase = TextureToScreen(projectedPoint);
            
            if (showHoughNormal)
            {
                Vector2 screenDir = TextureToScreen(projectedPoint + seg.normal) - screenBase;
                DrawArrow(screenBase, screenDir, arrowLength, COLOR_HOUGH, 2.5f);
            }
            
            if (enableKernelComparison && _kernelBuffersReady)
            {
                int idx = texCoord.y * _kernelBufferWidth + texCoord.x;
                
                if (showKernel5x5 && idx >= 0 && idx < _kernel5x5Normals.Length)
                {
                    var n5 = _kernel5x5Normals[idx];
                    if (math.lengthsq(n5) > 0.01f)
                    {
                        Vector2 normal5 = new Vector2(n5.x, n5.y);
                        Vector2 screenDir = TextureToScreen(projectedPoint + normal5) - screenBase;
                        DrawArrow(screenBase, screenDir, arrowLength * 0.9f, COLOR_5X5, 2f);
                    }
                }
                
                if (showKernel7x7 && idx >= 0 && idx < _kernel7x7Normals.Length)
                {
                    var n7 = _kernel7x7Normals[idx];
                    if (math.lengthsq(n7) > 0.01f)
                    {
                        Vector2 normal7 = new Vector2(n7.x, n7.y);
                        Vector2 screenDir = TextureToScreen(projectedPoint + normal7) - screenBase;
                        DrawArrow(screenBase, screenDir, arrowLength * 0.8f, COLOR_7X7, 2f);
                    }
                }
            }
            
            DrawCircle(screenBase, 4f, COLOR_HOUGH);
        }
        
        private void FindHoveredSegment()
        {
            if (!_processor.IsDataReady) { _hoveredSegment = null; return; }
            
            var texCoord = _processor.ScreenToTextureCoords(_mousePos, Screen.width, Screen.height);
            _hoveredSegment = _detector.FindNearestSegment(
                new Vector2(texCoord.x, texCoord.y),
                hoverRadius / houghParams.downsampleFactor
            );
        }
        
        private void DrawAllSegments()
        {
            Color faded = segmentColor;
            faded.a = 0.3f;
            
            foreach (var seg in _detector.DetectedLines)
            {
                var start = TextureToScreen(seg.startPoint);
                var end = TextureToScreen(seg.endPoint);
                DrawLine(start, end, faded, 1f);
            }
        }
        
        private void DrawHoveredSegment()
        {
            var seg = _hoveredSegment.Value;
            var start = TextureToScreen(seg.startPoint);
            var end = TextureToScreen(seg.endPoint);
            
            DrawLine(start, end, segmentColor, 3f);
            DrawCircle(start, 5f, segmentColor);
            DrawCircle(end, 5f, segmentColor);
        }
        
        private void DrawFallbackNormals(Vector2Int texCoord)
        {
            Vector2 normal = CalculateLocalNormal(texCoord.x, texCoord.y, localGradientRadius);
            
            if (normal.sqrMagnitude > 0.01f)
            {
                Vector2 screenBase = TextureToScreen(new Vector2(texCoord.x, texCoord.y));
                Vector2 screenEnd = TextureToScreen(new Vector2(texCoord.x, texCoord.y) + normal);
                Vector2 screenDir = screenEnd - screenBase;
                
                DrawArrow(screenBase, screenDir, arrowLength * 0.7f, new Color(1f, 0.5f, 0f, 0.7f), 2f);
            }
        }
        
        private Vector2 CalculateLocalNormal(int cx, int cy, int radius)
        {
            Vector2 gradSum = Vector2.zero;
            float totalWeight = 0f;
            
            var centerGrad = _processor.GetGradientAt(cx, cy);
            centerGrad.y = -centerGrad.y;
            
            if (centerGrad.sqrMagnitude > 0.01f)
            {
                gradSum += centerGrad.normalized * 3f;
                totalWeight += 3f;
            }
            
            int radiusSq = radius * radius;
            
            for (int dy = -radius; dy <= radius; dy++)
            {
                for (int dx = -radius; dx <= radius; dx++)
                {
                    if (dx == 0 && dy == 0) continue;
                    
                    int distSq = dx * dx + dy * dy;
                    if (distSq > radiusSq) continue;
                    
                    var grad = _processor.GetGradientAt(cx + dx, cy + dy);
                    grad.y = -grad.y;
                    
                    float mag = grad.magnitude;
                    if (mag < 0.05f) continue;
                    
                    float distWeight = 1f - Mathf.Sqrt(distSq) / (radius + 1f);
                    float magWeight = Mathf.Min(mag * 2f, 1f);
                    float edgeBonus = _processor.IsEdgeAt(cx + dx, cy + dy, edgeThreshold * 0.5f) ? 1.5f : 1f;
                    
                    float weight = distWeight * magWeight * edgeBonus;
                    gradSum += grad.normalized * weight;
                    totalWeight += weight;
                }
            }
            
            if (totalWeight < 0.1f) return Vector2.zero;
            return -(gradSum / totalWeight).normalized;
        }
        
        // ==================== DRAWING UTILITIES ====================
        
        private Vector2 TextureToScreen(Vector2 texCoord) =>
            _processor.TextureToScreenCoords(texCoord, Screen.width, Screen.height);
        
        private void DrawLine(Vector2 start, Vector2 end, Color color, float thickness)
        {
            var delta = end - start;
            float length = delta.magnitude;
            if (length < 0.001f) return;
            
            var saved = GUI.color;
            GUI.color = color;
            
            var savedMatrix = GUI.matrix;
            GUIUtility.RotateAroundPivot(Mathf.Atan2(delta.y, delta.x) * Mathf.Rad2Deg, start);
            GUI.DrawTexture(new Rect(start.x, start.y - thickness / 2, length, thickness), _whiteTex);
            GUI.matrix = savedMatrix;
            
            GUI.color = saved;
        }
        
        private void DrawArrow(Vector2 origin, Vector2 direction, float length, Color color, float thickness)
        {
            if (direction.sqrMagnitude < 0.001f) return;
            
            var dir = direction.normalized;
            var end = origin + dir * length;
            var perp = new Vector2(-dir.y, dir.x);
            float headSize = length * 0.25f;
            
            DrawLine(origin, end, color, thickness);
            DrawLine(end, end - dir * headSize + perp * headSize * 0.5f, color, thickness);
            DrawLine(end, end - dir * headSize - perp * headSize * 0.5f, color, thickness);
            DrawCircle(origin, thickness * 1.5f, color);
        }
        
        private void DrawCircle(Vector2 center, float radius, Color color)
        {
            var saved = GUI.color;
            GUI.color = color;
            GUI.DrawTexture(new Rect(center.x - radius, center.y - radius, radius * 2, radius * 2), _whiteTex);
            GUI.color = saved;
        }
        
        // ==================== BURST JOB ====================
        
        [BurstCompile]
        private struct KernelJob : IJobParallelFor
        {
            [ReadOnly] public NativeArray<float> gradX;
            [ReadOnly] public NativeArray<float> gradY;
            [WriteOnly] public NativeArray<float2> normals;
            
            public int width, height, radius;
            
            public void Execute(int index)
            {
                int x = index % width;
                int y = index / width;
                
                float2 gradSum = float2.zero;
                float weightSum = 0;
                int radiusSq = radius * radius;
                
                for (int dy = -radius; dy <= radius; dy++)
                {
                    for (int dx = -radius; dx <= radius; dx++)
                    {
                        int distSq = dx * dx + dy * dy;
                        if (distSq > radiusSq) continue;
                        
                        int px = math.clamp(x + dx, 0, width - 1);
                        int py = math.clamp(y + dy, 0, height - 1);
                        int pidx = py * width + px;
                        
                        float gxVal = gradX[pidx];
                        float gyVal = -gradY[pidx];
                        float mag = math.sqrt(gxVal * gxVal + gyVal * gyVal);
                        
                        if (mag < 0.05f) continue;
                        
                        float distWeight = 1f - math.sqrt((float)distSq) / (radius + 1f);
                        float weight = distWeight * mag;
                        
                        if (dx == 0 && dy == 0) weight *= 3f;
                        
                        gradSum += math.normalize(new float2(gxVal, gyVal)) * weight;
                        weightSum += weight;
                    }
                }
                
                if (weightSum > 0.1f)
                {
                    float2 avg = gradSum / weightSum;
                    float len = math.length(avg);
                    normals[index] = len > 0.01f ? -math.normalize(avg) : float2.zero;
                }
                else
                {
                    normals[index] = float2.zero;
                }
            }
        }
    }
}