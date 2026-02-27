using UnityEngine;

namespace SceneCapture
{
    [DefaultExecutionOrder(100)]
    [ExecuteInEditMode]
    [RequireComponent(typeof(Camera))]
    public class EdgeDetectionEffect : MonoBehaviour
    {
        [Header("Edge Detection Method")]
        public EdgeMethod method = EdgeMethod.Sobel;
        
        // ÖNEMLİ: Varsayılanı 'Depth' veya 'Normal' yapıyoruz. Combined riskli.
        [Header("Edge Source")]
        public EdgeSource source = EdgeSource.Combined; 
        
        [Header("Edge Settings")]
        [Range(0.001f, 1f)] public float edgeThreshold = 0.1f;
        public Color edgeColor = Color.white;
        public Color backgroundColor = Color.black;
        
        [Header("Weights (Combined Mode)")]
        [Range(0f, 1f)] public float depthWeight = 1.0f;   // Derinlik kenarları (Kritik)
        [Range(0f, 1f)] public float normalWeight = 1.0f;  // Köşe kenarları (Kritik)
        [Range(0f, 1f)] public float colorWeight = 0.0f;   // KAPATILDI (Gölge sorunu yaratır)
        
        [Header("Sensitivity")]
        [Range(0.1f, 100f)] public float depthSensitivity = 10f;
        public float maxDepth = 50f;
        [Range(0.1f, 100f)] public float normalSensitivity = 1f;

        [Header("Crease Angle Filter (Combined / Normal mode)")]
        [Tooltip("Yalnızca bu açıdan BÜYÜK normal farkları crease edge sayılır.\n" +
                 "Sphere / capsule mesh edge'lerini bastırmak için 20-30° önerilir.\n" +
                 "0° = tüm normal farkları edge, 90° = yalnızca dik köşeler edge.")]
        [Range(0f, 90f)] public float minCreaseAngleDeg = 26f;

        [Header("Algorithm Edge AA")]
        [Tooltip("Algoritma icin edge texture'i once daha yuksek cozumlulukte ureterek aliasing'i azaltir.")]
        public bool useAlgorithmSupersampling = true;
        [Range(1, 3)] public int algorithmSupersampleScale = 2;
        [Tooltip("Algoritma edge texture'ina hafif blur uygular (merdiven etkisini azaltir).")]
        [Range(0, 2)] public int algorithmSmoothPasses = 1;
        [Tooltip("AA/downsample sonrasi zayiflayan raw edge magnitude'i telafi eder.")]
        [Range(0.5f, 4f)] public float algorithmMagnitudeScale = 1.6f;

        public bool invertOutput = false;
        
        public enum EdgeMethod { Sobel, Roberts, Prewitt }
        public enum EdgeSource { Luminance, Depth, Normal, Combined }
        
        public RenderTexture EdgeResultTexture => _edgeResultTexture;

        private Material _material;
        private Camera _camera;
        private RenderTexture _edgeResultTexture;
        private RenderTexture _algorithmEdgeRT;
        
        private static readonly string[] MethodKeywords = { "_METHOD_SOBEL", "_METHOD_ROBERTS", "_METHOD_PREWITT" };
        private static readonly string[] SourceKeywords = { "_SOURCE_LUMINANCE", "_SOURCE_DEPTH", "_SOURCE_NORMAL", "_SOURCE_COMBINED" };

        void OnEnable()
        {
            _camera = GetComponent<Camera>();
            _camera.depthTextureMode |= DepthTextureMode.Depth | DepthTextureMode.DepthNormals;
            
            Shader shader = Shader.Find("Custom/EdgeDetection");
            if (shader != null) _material = new Material(shader);
        }

        void OnDisable()
        {
            if (_material != null) DestroyImmediate(_material);
            if (_edgeResultTexture != null) { _edgeResultTexture.Release(); _edgeResultTexture = null; }
            if (_algorithmEdgeRT != null) { _algorithmEdgeRT.Release(); _algorithmEdgeRT = null; }
        }

        void EnsureRT(ref RenderTexture rt, int w, int h, RenderTextureFormat fmt, bool randomWrite, FilterMode filter)
        {
            if (rt != null && rt.width == w && rt.height == h) return;
            if (rt != null) rt.Release();
            rt = new RenderTexture(w, h, 0, fmt);
            rt.enableRandomWrite = randomWrite;
            rt.filterMode = filter;
            rt.wrapMode = TextureWrapMode.Clamp;
            rt.Create();
        }

        void OnRenderImage(RenderTexture src, RenderTexture dst)
        {
            if (_material == null) { Graphics.Blit(src, dst); return; }
            
            // Parametreler
            _material.SetFloat("_EdgeThreshold", edgeThreshold);
            _material.SetColor("_EdgeColor", edgeColor);
            _material.SetColor("_BackgroundColor", backgroundColor);
            _material.SetFloat("_InvertOutput", invertOutput ? 1f : 0f);
            
            _material.SetFloat("_DepthSensitivity", depthSensitivity);
            _material.SetFloat("_MaxDepth", maxDepth);
            _material.SetFloat("_NormalSensitivity", normalSensitivity);
            
            // BURASI ÖNEMLİ: Inspector ayarlarını shader'a gönderiyoruz
            _material.SetFloat("_DepthWeight", depthWeight);
            _material.SetFloat("_NormalWeight", normalWeight);
            _material.SetFloat("_ColorWeight", colorWeight);
            // Crease filtresi: derece → cos(açı) dönüşümü (shader dot product ile karşılaştırır)
            _material.SetFloat("_MinCreaseDot", Mathf.Cos(minCreaseAngleDeg * Mathf.Deg2Rad));
            
            foreach (var kw in MethodKeywords) _material.DisableKeyword(kw);
            _material.EnableKeyword(MethodKeywords[(int)method]);
            
            foreach (var kw in SourceKeywords) _material.DisableKeyword(kw);
            _material.EnableKeyword(SourceKeywords[(int)source]);
            
            // --- PASS 1: RAW DATA (Algoritma İçin) ---
            EnsureRT(ref _edgeResultTexture, src.width, src.height, RenderTextureFormat.RFloat, true, FilterMode.Bilinear);

            int ss = useAlgorithmSupersampling ? Mathf.Clamp(algorithmSupersampleScale, 1, 3) : 1;
            int algoW = src.width * ss;
            int algoH = src.height * ss;
            EnsureRT(ref _algorithmEdgeRT, algoW, algoH, RenderTextureFormat.RFloat, false, FilterMode.Bilinear);

            _material.SetFloat("_MagnitudeScale", algorithmMagnitudeScale);
            _material.SetFloat("_OutputMagnitude", 1.0f);
            Graphics.Blit(src, _algorithmEdgeRT, _material);

            // Hafif smoothing + downsample zinciri (algoritma icin)
            RenderTexture current = _algorithmEdgeRT;
            for (int i = 0; i < Mathf.Max(0, algorithmSmoothPasses); i++)
            {
                var tmp = RenderTexture.GetTemporary(current.width, current.height, 0, RenderTextureFormat.RFloat);
                tmp.filterMode = FilterMode.Bilinear;
                tmp.wrapMode = TextureWrapMode.Clamp;
                Graphics.Blit(current, tmp);
                if (current != _algorithmEdgeRT) RenderTexture.ReleaseTemporary(current);
                current = tmp;
            }

            if (current.width != _edgeResultTexture.width || current.height != _edgeResultTexture.height)
            {
                var ds = RenderTexture.GetTemporary(_edgeResultTexture.width, _edgeResultTexture.height, 0, RenderTextureFormat.RFloat);
                ds.filterMode = FilterMode.Bilinear;
                ds.wrapMode = TextureWrapMode.Clamp;
                Graphics.Blit(current, ds);
                Graphics.Blit(ds, _edgeResultTexture);
                RenderTexture.ReleaseTemporary(ds);
            }
            else
            {
                Graphics.Blit(current, _edgeResultTexture);
            }

            if (current != _algorithmEdgeRT) RenderTexture.ReleaseTemporary(current);

            // --- PASS 2: VISUAL (Ekran İçin) ---
            _material.SetFloat("_MagnitudeScale", 1.0f);
            _material.SetFloat("_OutputMagnitude", 0.0f);
            Graphics.Blit(src, dst, _material);
        }
    }
}