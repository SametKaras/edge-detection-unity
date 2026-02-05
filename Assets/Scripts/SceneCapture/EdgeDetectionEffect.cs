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

        public bool invertOutput = false;
        
        public enum EdgeMethod { Sobel, Roberts, Prewitt }
        public enum EdgeSource { Luminance, Depth, Normal, Combined }
        
        public RenderTexture EdgeResultTexture { get; private set; }

        private Material _material;
        private Camera _camera;
        
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
            if (EdgeResultTexture != null) { EdgeResultTexture.Release(); EdgeResultTexture = null; }
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
            
            foreach (var kw in MethodKeywords) _material.DisableKeyword(kw);
            _material.EnableKeyword(MethodKeywords[(int)method]);
            
            foreach (var kw in SourceKeywords) _material.DisableKeyword(kw);
            _material.EnableKeyword(SourceKeywords[(int)source]);
            
            // --- PASS 1: RAW DATA (Algoritma İçin) ---
            if (EdgeResultTexture == null || EdgeResultTexture.width != src.width || EdgeResultTexture.height != src.height)
            {
                if (EdgeResultTexture != null) EdgeResultTexture.Release();
                EdgeResultTexture = new RenderTexture(src.width, src.height, 0, RenderTextureFormat.RFloat);
                EdgeResultTexture.enableRandomWrite = true;
                EdgeResultTexture.Create();
            }

            _material.SetFloat("_OutputMagnitude", 1.0f);
            Graphics.Blit(src, EdgeResultTexture, _material);

            // --- PASS 2: VISUAL (Ekran İçin) ---
            _material.SetFloat("_OutputMagnitude", 0.0f);
            Graphics.Blit(src, dst, _material);
        }
    }
}