using UnityEngine;

namespace SceneCapture.Hough
{
    /// <summary>
    /// 2D çizgi segmenti veri yapısı - Struct olarak tanımlanmış (hafif, stack allocation, Burst-compatible)
    /// </summary>
    [System.Serializable]
    public struct LineSegment2D
    {
        // ==================== HOUGH TRANSFORM PARAMETRELERİ ====================
        public float rho;        // Çizginin origin'e dik uzaklığı (pixel)
        public float theta;      // Çizginin açısı (radyan, 0 to π)
        public int score;        // Accumulator'daki oy sayısı (yüksek = güçlü çizgi)
        
        // ==================== GEOMETRİK BİLGİLER ====================
        public Vector2 startPoint;   // Segment başlangıç noktası
        public Vector2 endPoint;     // Segment bitiş noktası
        public Vector2 tangent;      // Çizgiye paralel birim vektör
        public Vector2 normal;       // Çizgiye dik birim vektör
        
        // ==================== KALİTE METRİKLERİ ====================
        public int supportingPixelCount;     // Segment üzerindeki edge pixel sayısı
        public float edgeCoverage;           // Pixel yoğunluğu (pixels/length)
        public float directionConsistency;   // Gradient yönü tutarlılığı (0-1)
        
        // ==================== DEBUG BİLGİSİ ====================
        public int kernelSourceId;  // Hangi kernel üretti? 0=Sobel, 1=Roberts, 2=Prewitt, -1=Unknown
        
        // ==================== COMPUTED PROPERTIES ====================
        public float Length => Vector2.Distance(startPoint, endPoint);     // Segment uzunluğu
        public Vector2 MidPoint => (startPoint + endPoint) * 0.5f;        // Orta nokta
        
        /// <summary>
        /// Bir noktanın segment'e olan en kısa mesafesi (hover detection için)
        /// </summary>
        public float DistanceToPoint(Vector2 point)
        {
            Vector2 ab = endPoint - startPoint;  // Segment vektörü
            Vector2 ap = point - startPoint;     // Nokta vektörü
            
            float lengthSq = ab.sqrMagnitude;
            if (lengthSq < 0.0001f) return ap.magnitude;  // Degenerate case
            
            // T parametresi: noktanın segment üzerindeki projection konumu (0-1 arası)
            float t = Mathf.Clamp01(Vector2.Dot(ap, ab) / lengthSq);
            
            return Vector2.Distance(point, startPoint + t * ab);
        }
        
        /// <summary>
        /// Noktayı segment üzerine project et (en yakın nokta)
        /// </summary>
        public Vector2 ProjectPoint(Vector2 point)
        {
            Vector2 ab = endPoint - startPoint;
            Vector2 ap = point - startPoint;
            
            float lengthSq = ab.sqrMagnitude;
            if (lengthSq < 0.0001f) return startPoint;
            
            float t = Mathf.Clamp01(Vector2.Dot(ap, ab) / lengthSq);
            return startPoint + t * ab;
        }
        
        /// <summary>
        /// Kernel adını string olarak döndür (debug için)
        /// </summary>
        public string GetKernelName()
        {
            return kernelSourceId switch
            {
                0 => "Sobel",
                1 => "Roberts",
                2 => "Prewitt",
                _ => "Unknown"
            };
        }
        
        /// <summary>
        /// Kernel için debug rengi döndür
        /// </summary>
        public Color GetKernelColor()
        {
            return kernelSourceId switch
            {
                0 => new Color(0f, 1f, 1f),      // Cyan - Sobel
                1 => new Color(1f, 0.5f, 0f),    // Orange - Roberts
                2 => new Color(0.5f, 1f, 0f),    // Yellow-Green - Prewitt
                _ => Color.white
            };
        }
    }

    /// <summary>
    /// Hough Transform parametreleri - Unity Inspector'da düzenlenebilir
    /// </summary>
    [System.Serializable]
    public class HoughParameters
    {
        // ==================== ACCUMULATOR ====================
        [Header("Accumulator")]
        [Range(90, 360)] public int thetaSteps = 180;     // Açı çözünürlüğü (daha fazla = daha hassas)
        [Range(0.5f, 4f)] public float rhoBinSize = 1f;   // Rho bin boyutu (daha küçük = daha hassas)
        
        // ==================== PEAK DETECTION ====================
        [Header("Peak Detection")]
        [Range(5, 200)] public int peakThreshold = 15;         // Minimum oy sayısı (çizgi olması için)
        [Range(3, 15)] public int nmsWindowSize = 5;          // Non-Maximum Suppression pencere boyutu
        [Range(10, 500)] public int maxLines = 200;           // Maksimum segment sayısı
        
        // ==================== SEGMENT ====================
        [Header("Segment")]
        [Range(3f, 100f)] public float segmentMinLength = 8f;        // Minimum segment uzunluğu
        [Range(20f, 500f)] public float segmentMaxLength = 60f;      // Maksimum segment uzunluğu (uzun olanlar bölünür)
        [Range(0.5f, 5f)] public float lineDistanceThreshold = 2.5f; // Pixel'in çizgiye maksimum uzaklığı
        
        // ==================== GELİŞMİŞ FİLTRELEME ====================
        [Header("Gelişmiş Filtreleme")]
        [Range(5f, 45f)] public float gradientAngleWindow = 25f;          // Gradient açı penceresi (derece)
        [Range(0.1f, 0.8f)] public float minEdgeCoverage = 0.25f;         // Minimum pixel yoğunluğu
        [Range(0.2f, 0.95f)] public float minDirectionConsistency = 0.4f; // Minimum yön tutarlılığı
        [Range(3, 50)] public int minSupportingPixels = 6;                // Minimum pixel sayısı
        
        // ==================== PERFORMANS ====================
        [Header("Performans")]
        [Range(1, 8)] public int downsampleFactor = 2;    // Downsampling faktörü (2 = 4x az pixel)
        [Range(1, 30)] public int updateInterval = 3;     // Kaç frame'de bir işle
        
        // ==================== COMPUTED PROPERTIES ====================
        public float ThetaStep => Mathf.PI / thetaSteps;                          // Açı adım büyüklüğü
        public int RhoBins(float diagonal) => Mathf.CeilToInt(diagonal * 2f / rhoBinSize);  // Rho bin sayısı
        public float GradientAngleWindowRad => gradientAngleWindow * Mathf.Deg2Rad;         // Derece → Radyan
    }
}