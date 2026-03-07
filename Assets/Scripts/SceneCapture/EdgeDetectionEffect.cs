using UnityEngine;

namespace SceneCapture
{
    /// <summary>
    /// Kameraya eklenen image effect.
    /// EdgeDetection.shader'ı kullanarak kenar haritası üretir.
    /// İki pass: 1) Raw magnitude → algoritma RT,  2) Binary → ekran.
    /// </summary>
    [ExecuteInEditMode]
    [RequireComponent(typeof(Camera))]
    public class EdgeDetectionEffect : MonoBehaviour
    {
        // ==================== ENUMS ====================
        public enum EdgeMethod { Sobel, Roberts, Prewitt }
        public enum EdgeSource { Luminance, Depth, Normal, Combined }

        // ==================== INSPECTOR ====================
        [Header("Edge Detection Method")]
        public EdgeMethod method = EdgeMethod.Sobel;

        [Header("Edge Source")]
        public EdgeSource source = EdgeSource.Combined;

        [Header("Edge Settings")]
        [Range(0.001f, 1f)] public float edgeThreshold = 0.1f;
        public Color edgeColor = Color.white;
        public Color backgroundColor = Color.black;

        [Header("Weights (Combined Mode)")]
        [Range(0f, 1f)] public float depthWeight = 1.0f;
        [Range(0f, 1f)] public float normalWeight = 1.0f;
        [Range(0f, 1f)] public float colorWeight = 0.0f;

        [Header("Sensitivity")]
        [Range(0.1f, 100f)] public float depthSensitivity = 10f;
        public float maxDepth = 50f;
        [Range(0.1f, 100f)] public float normalSensitivity = 1f;

        [Header("Crease Angle Filter")]
        [Tooltip("Bu açıdan küçük normal farkları edge sayılmaz.\n" +
                 "Sphere/capsule mesh edge bastırmak için 20-30° önerilir.")]
        [Range(0f, 90f)] public float minCreaseAngleDeg = 26f;

        [Header("Algorithm Edge AA")]
        [Tooltip("Algoritma için edge texture'ı supersampling ile üretir.")]
        public bool useAlgorithmSupersampling = true;
        [Range(1, 3)] public int algorithmSupersampleScale = 2;
        [Tooltip("Algoritma edge texture'ına hafif blur uygular.")]
        [Range(0, 2)] public int algorithmSmoothPasses = 1;
        [Tooltip("AA/downsample sonrası zayıflayan magnitude'ü telafi eder.")]
        [Range(0.5f, 4f)] public float algorithmMagnitudeScale = 1.6f;

        public bool invertOutput = false;

        [Header("World Position")]
        [Tooltip("WorldPosBuffer.shader — depth'ten world pos hesaplar.")]
        public Shader worldPosShader;

        // ==================== PUBLIC PROPERTY ====================
        /// <summary>
        /// Compute shader'ın okuyacağı edge magnitude texture (RFloat).
        /// </summary>
        public RenderTexture EdgeResultTexture => _edgeResultTexture;

        /// <summary>
        /// Compute shader'ın okuyacağı world-space pozisyon texture (ARGBFloat).
        /// Edge texture ile AYNI OnRenderImage çağrısında, aynı frame'de üretilir.
        /// w=1: geçerli piksel, w=0: arka plan / skybox.
        /// </summary>
        public RenderTexture WorldPosTexture  => _worldPosRT;

        /// <summary>
        /// EdgeResultTexture ve WorldPosTexture'ın güncellendiği son frame.
        /// WorldSpaceEdgeManager bunu kontrol ederek texture'ların bu frame'e ait olup olmadığını anlar.
        /// </summary>
        public int LastRenderedFrame { get; private set; } = -1;

        // ==================== PRIVATE ====================
        private Material _material;
        private Material _worldPosMat;
        private Camera _camera;
        private RenderTexture _edgeResultTexture;
        private RenderTexture _algorithmEdgeRT;
        private RenderTexture _worldPosRT;

        private static readonly string[] MethodKeywords =
            { "_METHOD_SOBEL", "_METHOD_ROBERTS", "_METHOD_PREWITT" };

        private static readonly string[] SourceKeywords =
            { "_SOURCE_LUMINANCE", "_SOURCE_DEPTH", "_SOURCE_NORMAL", "_SOURCE_COMBINED" };

        // ==================== LIFECYCLE ====================
        void OnEnable()
        {
            _camera = GetComponent<Camera>();
            _camera.depthTextureMode |= DepthTextureMode.Depth
                                      | DepthTextureMode.DepthNormals;

            Shader shader = Shader.Find("Custom/EdgeDetection");
            if (shader != null)
                _material = new Material(shader);

            // WorldPos materyali
            var wpShader = worldPosShader != null
                ? worldPosShader
                : Shader.Find("Custom/WorldPosBuffer");
            if (wpShader != null)
                _worldPosMat = new Material(wpShader);
        }

        void OnDisable()
        {
            if (_material    != null) DestroyImmediate(_material);
            if (_worldPosMat != null) DestroyImmediate(_worldPosMat);
            ReleaseRT(ref _edgeResultTexture);
            ReleaseRT(ref _algorithmEdgeRT);
            ReleaseRT(ref _worldPosRT);
        }

        // ==================== RENDER ====================

        // OnPreRender: OnRenderImage'dan ÖNCE, aynı frame'de çalışır.
        // World pos RT burada doldurulur → EdgeResultTexture ile senkron garantisi.
        void OnPreRender()
        {
            if (_worldPosMat == null) return;

            // Camera.current: hangi kamera render ediyorsa onu kullan.
            // Game view → Game kamerası, Scene view → Scene kamerası.
            Camera cam = Camera.current != null ? Camera.current : _camera;

            EnsureRT(ref _worldPosRT, cam.pixelWidth, cam.pixelHeight,
                     RenderTextureFormat.ARGBFloat, false, FilterMode.Point);

            Matrix4x4 proj  = GL.GetGPUProjectionMatrix(cam.projectionMatrix, true);
            Matrix4x4 view  = cam.worldToCameraMatrix;
            Matrix4x4 invVP = (proj * view).inverse;
            _worldPosMat.SetMatrix("unity_MatrixInvVP", invVP);

            Graphics.Blit(null, _worldPosRT, _worldPosMat);
        }

        void OnRenderImage(RenderTexture src, RenderTexture dst)
        {
            if (_material == null)
            {
                UnityEngine.Debug.LogWarning("[EdgeEffect] OnRenderImage: _material null, blitting through");
                Graphics.Blit(src, dst);
                return;
            }

            SetShaderParams();

            // --- PASS 1: Raw magnitude → algoritma RT ---
            EnsureRT(ref _edgeResultTexture, src.width, src.height,
                     RenderTextureFormat.RFloat, true, FilterMode.Bilinear);

            int ss = useAlgorithmSupersampling
                     ? Mathf.Clamp(algorithmSupersampleScale, 1, 3) : 1;
            int algoW = src.width  * ss;
            int algoH = src.height * ss;
            EnsureRT(ref _algorithmEdgeRT, algoW, algoH,
                     RenderTextureFormat.RFloat, false, FilterMode.Bilinear);

            _material.SetFloat("_MagnitudeScale", algorithmMagnitudeScale);
            _material.SetFloat("_OutputMagnitude", 1.0f);
            Graphics.Blit(src, _algorithmEdgeRT, _material);

            // Smooth passes (algoritma aliasing azaltma)
            RenderTexture current = _algorithmEdgeRT;
            for (int i = 0; i < Mathf.Max(0, algorithmSmoothPasses); i++)
            {
                var tmp = RenderTexture.GetTemporary(
                    current.width, current.height, 0, RenderTextureFormat.RFloat);
                tmp.filterMode = FilterMode.Bilinear;
                tmp.wrapMode   = TextureWrapMode.Clamp;
                Graphics.Blit(current, tmp);
                if (current != _algorithmEdgeRT)
                    RenderTexture.ReleaseTemporary(current);
                current = tmp;
            }

            // Downsample → edgeResultTexture
            if (current.width  != _edgeResultTexture.width ||
                current.height != _edgeResultTexture.height)
            {
                var ds = RenderTexture.GetTemporary(
                    _edgeResultTexture.width, _edgeResultTexture.height,
                    0, RenderTextureFormat.RFloat);
                ds.filterMode = FilterMode.Bilinear;
                ds.wrapMode   = TextureWrapMode.Clamp;
                Graphics.Blit(current, ds);
                Graphics.Blit(ds, _edgeResultTexture);
                RenderTexture.ReleaseTemporary(ds);
            }
            else
            {
                Graphics.Blit(current, _edgeResultTexture);
            }

            if (current != _algorithmEdgeRT)
                RenderTexture.ReleaseTemporary(current);

            // Texture'lar hazır — frame numarasını kaydet
            // Frame numarasını kaydet — buraya ulaşıldıysa her iki texture da hazır
            LastRenderedFrame = Time.frameCount;
            if (Time.frameCount % 120 == 0) UnityEngine.Debug.Log($"[EdgeEffect] Frame {Time.frameCount} rendered, edgeRT={_edgeResultTexture != null}, worldRT={_worldPosRT != null}");

            // --- PASS 2: Binary threshold → ekran ---
            _material.SetFloat("_MagnitudeScale", 1.0f);
            _material.SetFloat("_OutputMagnitude", 0.0f);
            Graphics.Blit(src, dst, _material);
        }

        // ==================== HELPERS ====================
        void SetShaderParams()
        {
            _material.SetFloat("_EdgeThreshold",    edgeThreshold);
            _material.SetColor("_EdgeColor",        edgeColor);
            _material.SetColor("_BackgroundColor",   backgroundColor);
            _material.SetFloat("_InvertOutput",     invertOutput ? 1f : 0f);

            _material.SetFloat("_DepthSensitivity", depthSensitivity);
            _material.SetFloat("_MaxDepth",         maxDepth);
            _material.SetFloat("_NormalSensitivity", normalSensitivity);

            _material.SetFloat("_DepthWeight",  depthWeight);
            _material.SetFloat("_NormalWeight", normalWeight);
            _material.SetFloat("_ColorWeight",  colorWeight);

            // Derece → cos dönüşümü (shader dot product ile karşılaştırır)
            _material.SetFloat("_MinCreaseDot",
                Mathf.Cos(minCreaseAngleDeg * Mathf.Deg2Rad));

            // Keyword'ler
            foreach (var kw in MethodKeywords) _material.DisableKeyword(kw);
            _material.EnableKeyword(MethodKeywords[(int)method]);

            foreach (var kw in SourceKeywords) _material.DisableKeyword(kw);
            _material.EnableKeyword(SourceKeywords[(int)source]);
        }

        void EnsureRT(ref RenderTexture rt, int w, int h,
                      RenderTextureFormat fmt, bool randomWrite, FilterMode filter)
        {
            if (rt != null && rt.width == w && rt.height == h) return;
            ReleaseRT(ref rt);
            rt = new RenderTexture(w, h, 0, fmt);
            rt.enableRandomWrite = randomWrite;
            rt.filterMode = filter;
            rt.wrapMode   = TextureWrapMode.Clamp;
            rt.Create();
        }

        void ReleaseRT(ref RenderTexture rt)
        {
            if (rt != null) { rt.Release(); rt = null; }
        }
    }
}