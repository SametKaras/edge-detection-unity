// ============================================================
// WorldSpaceEdgeManager.cs
// ============================================================
//
// AMAÇ:
//   2D kenar piksellerini 3D dünya uzayında çizgilere dönüştürmek.
//   Sahne içindeki objelerin kenarlarını tespit edip Scene view'da
//   Gizmo olarak görselleştirmek.
//
// GENEL PİPELİNE (adım adım):
//
//   ADIM 1 — DÜNYA POZİSYONU RENDER:
//     Ayrı bir kamera (WorldPosCamera) oluşturulur.
//     Bu kamera, WorldPosBuffer.shader ile sahneyi renderlar.
//     Sonuç: Her pikselin 3D dünya koordinatını içeren 128-bit float texture.
//
//   ADIM 2 — EDGE + POZİSYON BİRLEŞTİRME (GPU):
//     EdgeToPointCloud.compute shader'ı iki texture'ı birleştirir:
//       - _EdgeTex: Hangi pikseller kenar? (beyaz/siyah)
//       - _WorldPosTex: O pikselin 3D koordinatı ne?
//     Beyaz piksellerin 3D pozisyonları AppendStructuredBuffer'a yazılır.
//
//   ADIM 3 — GPU → CPU TRANSFER (Asenkron):
//     AsyncGPUReadback ile buffer verileri ana belleğe taşınır.
//     İki aşamalı okuma:
//       a) Önce kaç nokta bulunduğu okunur (_argsBuffer → CopyCount)
//       b) Sonra o kadar nokta okunur (_pointBuffer → float3[])
//     Asenkron olduğu için ana thread bloklanmaz.
//
//   ADIM 4 — RANSAC ÇİZGİ BULMA (CPU, Burst Job):
//     3D nokta bulutu LocalRansacJob'a verilir.
//     RANSAC algoritması bu noktalardan 3D çizgi segmentleri bulur.
//     Burst Compiler ile derlenir → native hızda çalışır.
//
//   ADIM 5 — GÖRSELLEŞTİRME (Gizmo):
//     Bulunan çizgiler OnDrawGizmos() ile Scene view'da çizilir.
//     Opsiyonel olarak ham nokta bulutu da gösterilebilir.
//
// ZAMANLAMA:
//   updateInterval: İşlem bittikten sonra minimum bekleme süresi.
//   Bu sayede GPU/CPU sürekli meşgul edilmez, cooldown uygulanır.
//
// ============================================================

using System.IO;
using System.Text;
using UnityEngine;
using UnityEngine.Rendering;   // AsyncGPUReadback, CompareFunction
using Unity.Collections;        // NativeArray
using Unity.Jobs;               // JobHandle, IJob
using Unity.Mathematics;        // float3, math
using Unity.Burst;              // BurstCompile
using Random = Unity.Mathematics.Random;  // Burst-uyumlu random
#if UNITY_EDITOR
using UnityEditor;              // Handles (Scene view çizim)
#endif

namespace SceneCapture.Edge3D
{
    // ==================== VERİ YAPISI ====================
    
    /// <summary>
    /// RANSAC'ın bulduğu 3D çizgi segmenti.
    /// Start ve End: Çizginin iki uç noktası (dünya koordinatları).
    /// IsValid: Bu slot'ta geçerli bir çizgi var mı?
    /// (NativeArray sabit boyutlu olduğu için boş slot'lar IsValid=false olur)
    /// </summary>
    public struct Line3D
    {
        public float3 Start;    // Çizginin başlangıç noktası (world space)
        public float3 End;      // Çizginin bitiş noktası (world space)
        public bool IsValid;    // Bu çizgi geçerli mi? (boş slot kontrolü)
    }

    // ==================== ANA SINIF ====================
    
    /// <summary>
    /// 3D kenar algılama ve görselleştirme yöneticisi.
    /// Camera bileşenine ihtiyaç duyar (ana kameranın üzerine eklenir).
    /// EdgeDetectionEffect de aynı kamerada olmalıdır (edge texture'ı sağlar).
    /// </summary>
    [RequireComponent(typeof(EdgeDetectionEffect))]
    public class WorldSpaceEdgeManager : MonoBehaviour
    {
        // ==================== INSPECTOR PARAMETRELERİ ====================
        
        [Header("References")]
        [Tooltip("WorldPosBuffer.shader — Her pikselin 3D pozisyonunu yazan shader")]
        public Shader worldPosShader;
        
        [Tooltip("EdgeToPointCloud.compute — Edge piksellerin 3D koordinatlarını çıkaran compute shader")]
        public ComputeShader pointExtractorCS;

        [Header("⏱️ Timing Control")]
        [Tooltip("Bir RANSAC işlemi bittikten sonra en az bu kadar saniye bekle.\n" +
                 "Düşük değer = daha sık güncelleme ama daha fazla CPU/GPU kullanımı.\n" +
                 "Yüksek değer = daha nadir güncelleme ama daha az kaynak tüketimi.")]
        [Range(0.1f, 10f)] public float updateInterval = 2.0f; 
        
        [Header("🚀 Speed vs Quality")]
        [Tooltip("Piksel örnekleme adımı (downsampling).\n" +
                 "1 = Her piksel işlenir (en yüksek kalite, en yavaş)\n" +
                 "4 = Her 4. piksel işlenir (16x daha hızlı, daha düşük kalite)\n" +
                 "Compute shader'daki thread sayısını doğrudan etkiler.")]
        [Range(1, 8)] public int pixelStep = 1; 

        [Header("Detection Sensitivity")]
        [Tooltip("Minimum kenar parlaklığı eşiği.\n" +
                 "Bu değerin altındaki piksel kenar sayılmaz.\n" +
                 "Düşük = daha fazla nokta (gürültülü), Yüksek = daha az nokta (temiz).")]
        [Range(0.01f, 0.9f)] public float minEdgeLuminance = 0.1f;

        [Header("RANSAC Settings")]
        [Tooltip("Maksimum bulunacak çizgi sayısı.\n" +
                 "NativeArray bu boyutta oluşturulur — çok yüksek değerler bellek kullanır.")]
        [Range(1, 5000)] public int maxLinesToDetect = 3000;
        
        [Tooltip("RANSAC iterasyon sayısı.\n" +
                 "Daha fazla = daha doğru sonuç ama daha yavaş.\n" +
                 "Her iterasyonda rastgele 2 nokta seçilip çizgi test edilir.")]
        [Range(100, 50000)] public int ransacIterations = 15000; 
        
        [Tooltip("Çizgi kalınlığı (tolerans mesafesi).\n" +
                 "Bir noktanın çizgiye olan mesafesi bu değerden küçükse 'inlier' sayılır.\n" +
                 "Büyük = daha kalın çizgiler (daha fazla nokta yakalar), Küçük = daha ince.")]
        [Range(0.01f, 0.5f)] public float lineThickness = 0.08f;
        
        [Tooltip("Bir çizginin geçerli sayılması için minimum destekleyen nokta sayısı.")]
        [Range(2, 500)] public int minPointsForLine = 2;
        
        [Tooltip("Maksimum segment uzunluğu (dünya birimi).\n" +
                 "Bu değerden uzun çizgi adayları reddedilir.\n" +
                 "Farklı objelerin noktalarının yanlışlıkla birleşmesini önler.")]
        [Range(0.01f, 10f)] public float maxSegmentLength = 0.15f;

        [Header("Visualization")]
        [Tooltip("RANSAC çizgilerini Scene view'da göster")]
        public bool showLines = true;
        
        [Tooltip("Çizgi kalınlığı (Scene view görsel)")]
        [Range(1f, 15f)] public float visualLineThickness = 4.0f; 
        
        [Header("☁️ Point Cloud Visualization")]
        [Tooltip("Edge piksellerinden oluşan 3D nokta bulutunu Scene view'da göster.\n" +
                 "RANSAC'a giren ham veriyi görmek için kullanışlıdır.")]
        public bool showPointCloud = false;
        
        [Tooltip("Nokta bulutu rengi")]
        public Color pointCloudColor = Color.green;
        
        [Tooltip("Her bir noktanın küre büyüklüğü (dünya birimi)")]
        [Range(0.001f, 0.1f)] public float pointSize = 0.01f;
        
        // ==================== PRİVATE REFERANSLAR ====================
        
        private Camera _mainCam;        // Ana kamera (bu bileşenin bağlı olduğu)
        private Camera _posCam;         // Dünya pozisyonu renderlayan yardımcı kamera
        private RenderTexture _worldPosRT;  // 128-bit float RT — dünya pozisyonları
        private EdgeDetectionEffect _edgeEffect;  // Kenar algılama efekti — edge texture sağlar
        
        // ==================== COMPUTE BUFFER'LAR ====================
        
        // Compute shader'ın yazdığı 3D nokta listesi
        // AppendStructuredBuffer olarak kullanılır (GPU tarafında)
        // Boyut: 150.000 × 12 byte (float3) = ~1.8 MB
        private ComputeBuffer _pointBuffer;
        
        // Append buffer'daki eleman sayısını öğrenmek için kullanılır
        // ComputeBuffer.CopyCount → bu buffer'a yazar → AsyncGPUReadback ile okunur
        private ComputeBuffer _argsBuffer;
        
        // ==================== JOB SYSTEM DEĞİŞKENLERİ ====================
        
        // RANSAC'a giren 3D noktalar (GPU'dan CPU'ya kopyalanmış)
        private NativeArray<float3> _inputPoints;
        
        // RANSAC'ın çıktısı — bulunan 3D çizgiler
        private NativeArray<Line3D> _outputLines;
        
        // Burst Job'un tamamlanma durumunu takip eder
        private JobHandle _ransacJobHandle;
        
        // ==================== DURUM YÖNETİMİ ====================
        
        private bool _isJobRunning = false;      // RANSAC job'u çalışıyor mu?
        private bool _hasNewData = false;         // Gösterilecek yeni veri var mı?
        private float _lastJobFinishTime = -999f; // Son işlemin bitiş zamanı (cooldown için)
        
        // ==================== GÖRÜNTÜLEME VERİLERİ ====================
        
        // RANSAC sonuçları — managed array (OnDrawGizmos'ta kullanılır)
        // NativeArray OnDrawGizmos'ta güvenli değil, bu yüzden kopyalanır
        private Line3D[] _displayLines;
        
        // Nokta bulutu verileri (opsiyonel görselleştirme)
        private Vector3[] _displayPoints;
        private int _displayPointCount;

        // ================================================================
        // UNITY YAŞAM DÖNGÜSÜ
        // ================================================================
        
        /// <summary>
        /// İlk başlatma — kamera ve buffer'ları oluştur.
        /// </summary>
        void Start()
        {
            // Ana kamerayı ve edge efektini al
            _mainCam = GetComponent<Camera>();
            _edgeEffect = GetComponent<EdgeDetectionEffect>();

            // ========== DÜNYA POZİSYON KAMERASI OLUŞTUR ==========
            // Ana kameranın child'ı olarak yeni bir kamera oluşturuyoruz.
            // Bu kamera SADECE WorldPosBuffer.shader ile renderlar.
            // Ana kameranın parametrelerini (FOV, near/far plane, pozisyon) kopyalar
            // böylece aynı pikseller aynı objelere karşılık gelir.
            GameObject camObj = new GameObject("WorldPosCamera");
            camObj.transform.SetParent(transform);  // Ana kameranın child'ı
            _posCam = camObj.AddComponent<Camera>();
            _posCam.CopyFrom(_mainCam);             // FOV, clip plane vs. kopyala
            _posCam.enabled = false;                 // Otomatik render etme (biz manuel çağıracağız)
            _posCam.backgroundColor = Color.black;   // Boş alanlar siyah (w=0)
            _posCam.clearFlags = CameraClearFlags.SolidColor;
            _posCam.renderingPath = RenderingPath.Forward;  // Forward — replacement shader için gerekli

            // GPU buffer'larını oluştur
            CreateBuffers();
            
            // Sonuç dizisini ayır (maxLinesToDetect boyutunda)
            _displayLines = new Line3D[maxLinesToDetect];
        }

        /// <summary>
        /// GPU buffer'larını oluştur veya yeniden oluştur.
        /// </summary>
        void CreateBuffers()
        {
            // Nokta listesi buffer'ı — Append modda (compute shader .Append() kullanır)
            // 150.000 nokta kapasitesi — çoğu sahne için yeterli
            // Her eleman: float3 = 3 × 4 byte = 12 byte
            if (_pointBuffer == null || !_pointBuffer.IsValid())
                _pointBuffer = new ComputeBuffer(150000, sizeof(float) * 3, ComputeBufferType.Append);
            
            // Sayaç buffer'ı — Append buffer'daki eleman sayısını öğrenmek için
            // CopyCount bu buffer'a yazar, biz de AsyncGPUReadback ile okuruz
            // 3 uint: [count, 0, 0] (IndirectArguments formatı)
            if (_argsBuffer == null || !_argsBuffer.IsValid())
                _argsBuffer = new ComputeBuffer(1, sizeof(uint) * 3, ComputeBufferType.IndirectArguments);
        }

        // ================================================================
        // ANA GÜNCELLEME DÖNGÜSÜ
        // ================================================================

        /// <summary>
        /// Her frame çağrılır — duruma göre iş planlar veya sonuç toplar.
        /// 
        /// Durum makinesi:
        ///   1. Job çalışıyorsa → bitmesini bekle
        ///   2. Cooldown süresi dolmadıysa → bekle
        ///   3. Her şey hazırsa → yeni işlem başlat
        /// </summary>
        void Update()
        {
            // DURUM 1: RANSAC job'u hâlâ çalışıyor
            if (_isJobRunning)
            {
                // Job bitti mi kontrol et (bloklamadan)
                if (_ransacJobHandle.IsCompleted)
                {
                    // Sonuçları topla ve gösterime hazırla
                    CompleteJob();
                }
                return; // Job bitene kadar yeni iş başlatma
            }

            // DURUM 2: Cooldown süresi dolmadı
            // Son işlemden bu yana yeterince zaman geçmedi
            if (Time.time - _lastJobFinishTime < updateInterval)
            {
                return;
            }

            // DURUM 3: Gerekli bileşenler hazır mı?
            if (worldPosShader == null || pointExtractorCS == null || _edgeEffect.EdgeResultTexture == null) return;
            
            // Her şey hazır — yeni işlem başlat
            RenderAndExtractPoints();
        }

        // ================================================================
        // ADIM 1-2: RENDER + NOKTA ÇIKARMA (GPU)
        // ================================================================

        /// <summary>
        /// Dünya pozisyon texture'ını renderla ve compute shader ile
        /// kenar piksellerinin 3D koordinatlarını çıkar.
        /// 
        /// İş akışı:
        ///   1. WorldPosCamera ile sahneyi renderla → _worldPosRT
        ///   2. Compute shader: _EdgeTex + _worldPosRT → _PointBuffer
        ///   3. Eleman sayısını oku (async) → OnArgBufferReadback
        /// </summary>
        void RenderAndExtractPoints()
        {
            // ========== RENDER TEXTURE OLUŞTUR/GÜNCELLE ==========
            // Ekran boyutu değiştiyse yeniden oluştur
            // ARGBFloat: 4 kanal × 32-bit float = 128-bit/piksel
            // Bu sayede dünya koordinatları (ör: x=15.372, y=2.841, z=-8.093) 
            // hassas şekilde saklanır. 8-bit texture'da bu bilgi kaybolurdu.
            if (_worldPosRT == null || _worldPosRT.width != Screen.width || _worldPosRT.height != Screen.height)
            {
                if (_worldPosRT != null) _worldPosRT.Release();
                _worldPosRT = new RenderTexture(Screen.width, Screen.height, 0, RenderTextureFormat.ARGBFloat);
                _worldPosRT.Create();
            }

            // ========== DÜNYA POZİSYONUNU RENDERLA ==========
            // RenderWithShader: Sahnedeki tüm objeleri WorldPosBuffer.shader ile renderlar
            // "RenderType" filtresi: Sadece aynı RenderType tag'ine sahip objeler renderlanır
            // Sonuç: _worldPosRT'nin her pikseli o objenin 3D dünya koordinatını içerir
            _posCam.targetTexture = _worldPosRT;
            _posCam.RenderWithShader(worldPosShader, "RenderType");

            // ========== COMPUTE SHADER'I ÇALIŞTIR ==========
            // Append buffer'ın sayacını sıfırla (yeni frame, yeni veriler)
            _pointBuffer.SetCounterValue(0);
            
            // "ExtractPoints" kernel'ını bul
            int kernel = pointExtractorCS.FindKernel("ExtractPoints");
            
            // Shader'a texture ve parametreleri bağla
            pointExtractorCS.SetTexture(kernel, "_EdgeTex", _edgeEffect.EdgeResultTexture);
            pointExtractorCS.SetTexture(kernel, "_WorldPosTex", _worldPosRT);
            pointExtractorCS.SetFloat("_MinEdgeThreshold", minEdgeLuminance);
            pointExtractorCS.SetInt("_Step", pixelStep); 
            
            // Texture boyutlarını uniform olarak geçir
            // (Compute shader içinde GetDimensions() çağrısını önlemek için)
            pointExtractorCS.SetInt("_TexWidth", Screen.width);
            pointExtractorCS.SetInt("_TexHeight", Screen.height);
            
            pointExtractorCS.SetBuffer(kernel, "_PointBuffer", _pointBuffer);

            // Dispatch sayısını hesapla — _Step'e bölerek sadece gerekli thread'leri başlat
            // numthreads(16,16,1) olduğu için her grup 16×16 piksel işler
            // pixelStep=4: 1920/(16×4)=30, 1080/(16×4)≈17 → 30×17=510 grup (130K thread)
            // pixelStep=1: 1920/16=120, 1080/16≈68 → 120×68=8160 grup (2M thread)
            int groupsX = Mathf.CeilToInt((float)Screen.width / (16 * pixelStep));
            int groupsY = Mathf.CeilToInt((float)Screen.height / (16 * pixelStep));
            pointExtractorCS.Dispatch(kernel, groupsX, groupsY, 1);

            // ========== ASENKRON OKUMA BAŞLAT ==========
            // Önce append buffer'daki eleman sayısını öğren
            // CopyCount: GPU'daki atomic counter'ı _argsBuffer'a kopyalar
            ComputeBuffer.CopyCount(_pointBuffer, _argsBuffer, 0);
            
            // Asenkron okuma: GPU→CPU transfer tamamlanınca callback çağrılır
            // Bu sayede ana thread bloklanmaz
            AsyncGPUReadback.Request(_argsBuffer, OnArgBufferReadback);
        }

        // ================================================================
        // ADIM 3: GPU → CPU TRANSFER (Asenkron Callback'ler)
        // ================================================================

        /// <summary>
        /// İlk callback: Kaç adet 3D nokta bulunduğunu öğren.
        /// Sonra o kadar noktayı okumak için ikinci readback başlat.
        /// </summary>
        void OnArgBufferReadback(AsyncGPUReadbackRequest request)
        {
            // Hata kontrolü — buffer geçersizse veya readback başarısızsa atla
            if (request.hasError || _pointBuffer == null || !_pointBuffer.IsValid()) return;

            // Sayaç değerini oku (uint[0] = eleman sayısı)
            var data = request.GetData<uint>();
            int pointCount = (int)data[0];

            if (pointCount > 0)
            {
                // İkinci readback: Gerçek nokta verilerini oku
                // pointCount × 12 byte (her float3 = 12 byte)
                // offset = 0 (buffer'ın başından itibaren)
                AsyncGPUReadback.Request(_pointBuffer, pointCount * 12, 0, (req) => OnPointsReadback(req, pointCount));
            }
        }

        /// <summary>
        /// İkinci callback: 3D nokta verilerini al ve RANSAC job'unu başlat.
        /// Bu noktada veriler GPU'dan CPU'ya aktarılmış olur.
        /// </summary>
        void OnPointsReadback(AsyncGPUReadbackRequest request, int count)
        {
            if (request.hasError) return;
            if (_isJobRunning) return;  // Zaten bir job çalışıyorsa üst üste başlatma

            // ========== VERİLERİ NativeArray'E KOPYALA ==========
            // GPU'dan gelen float3 verileri → Burst Job için NativeArray'e
            // Allocator.Persistent: Job tamamlanana kadar bellekte kalır
            _inputPoints = new NativeArray<float3>(request.GetData<float3>(), Allocator.Persistent);
            
            // Sonuç dizisi — RANSAC burada çizgileri yazacak
            _outputLines = new NativeArray<Line3D>(maxLinesToDetect, Allocator.Persistent);

            // ========== RANSAC JOB'UNU OLUŞTUR VE PLANLA ==========
            var ransacJob = new LocalRansacJob
            {
                InputPoints = _inputPoints,
                ResultLines = _outputLines,
                MaxLinesToFind = maxLinesToDetect,
                MaxIterations = ransacIterations,
                Threshold = lineThickness,
                MinInliers = minPointsForLine,
                MaxSegLength = maxSegmentLength,
                RandomSeed = (uint)Time.frameCount  // Her frame farklı seed
            };

            // Job'u arka planda çalıştır (ana thread bloklanmaz)
            _ransacJobHandle = ransacJob.Schedule();
            _isJobRunning = true; 
        }

        // ================================================================
        // ADIM 4: RANSAC TAMAMLANDI — SONUÇLARI TOPLA
        // ================================================================

        /// <summary>
        /// RANSAC job'u tamamlandığında çağrılır.
        /// Sonuçları managed dizilere kopyalar (Gizmo çizimi için)
        /// ve NativeArray'leri serbest bırakır.
        /// </summary>
        void CompleteJob()
        {
            // Job'un gerçekten tamamlandığından emin ol
            _ransacJobHandle.Complete();

            // Çizgi sonuçlarını managed diziye kopyala
            // (NativeArray, OnDrawGizmos sırasında güvenli olmayabilir)
            _outputLines.CopyTo(_displayLines); 
            _hasNewData = true;
            
            // ========== NOKTA BULUTU VERİSİNİ KOPYALA ==========
            // showPointCloud aktifse, RANSAC'a giren ham noktaları sakla
            // Dispose'dan ÖNCE kopyalanmalı — sonra NativeArray silinecek
            if (showPointCloud && _inputPoints.IsCreated)
            {
                _displayPointCount = _inputPoints.Length;
                
                // Dizi boyutu yetersizse yeniden oluştur
                if (_displayPoints == null || _displayPoints.Length < _displayPointCount)
                    _displayPoints = new Vector3[_displayPointCount];
                
                // float3 → Vector3 kopyalama
                for (int i = 0; i < _displayPointCount; i++)
                    _displayPoints[i] = _inputPoints[i];
            }

            // NativeArray'leri serbest bırak (bellek sızıntısını önle)
            if (_inputPoints.IsCreated) _inputPoints.Dispose();
            if (_outputLines.IsCreated) _outputLines.Dispose();

            _isJobRunning = false;
            
            // Cooldown zamanlayıcısını başlat
            // Bir sonraki işlem en erken (Time.time + updateInterval)'da başlar
            _lastJobFinishTime = Time.time; 
        }

        // ================================================================
        // ADIM 5: GÖRSELLEŞTİRME (Scene View Gizmo)
        // ================================================================

        /// <summary>
        /// Scene view'da çizgi ve nokta bulutu çizimi.
        /// Unity Editor'da her frame çağrılır (sadece Scene view için).
        /// Game view'da görünmez — sadece debug/geliştirme amaçlıdır.
        /// </summary>
        void OnDrawGizmos()
        {
            if (!_hasNewData) return;
            
            // ========== RANSAC ÇİZGİLERİ ==========
            if (showLines && _displayLines != null)
            {
#if UNITY_EDITOR
                // Derinlik testi kapalı → çizgiler objelerin arkasında da görünür
                Handles.zTest = CompareFunction.Always;
                Handles.color = Color.red; 
                
                for (int i = 0; i < _displayLines.Length; i++)
                {
                    var line = _displayLines[i];
                    if (line.IsValid)
                    {
                        // Anti-aliased kalın çizgi çiz (Start → End)
                        Handles.DrawAAPolyLine(visualLineThickness, line.Start, line.End);
                    }
                }
#endif
            }
            
            // ========== NOKTA BULUTU ==========
            // Her 3D kenar noktasını küçük küre olarak çiz
            // Inspector'dan renk ve boyut ayarlanabilir
            if (showPointCloud && _displayPoints != null && _displayPointCount > 0)
            {
                Gizmos.color = pointCloudColor;
                for (int i = 0; i < _displayPointCount; i++)
                {
                    Gizmos.DrawSphere(_displayPoints[i], pointSize);
                }
            }
        }

        // ================================================================
        // TEMİZLİK
        // ================================================================

        void OnDisable() { DisposeBuffers(); }
        void OnDestroy() { DisposeBuffers(); }

        /// <summary>
        /// Tüm native kaynakları serbest bırak.
        /// NativeArray ve ComputeBuffer dispose edilmezse bellek sızıntısı olur.
        /// </summary>
        void DisposeBuffers()
        {
            // Çalışan job varsa önce tamamla (aksi halde NativeArray hata verir)
            if (_isJobRunning) _ransacJobHandle.Complete();
            
            // NativeArray'leri serbest bırak
            if (_inputPoints.IsCreated) _inputPoints.Dispose();
            if (_outputLines.IsCreated) _outputLines.Dispose();

            // GPU buffer'larını serbest bırak
            if (_pointBuffer != null) _pointBuffer.Release();
            if (_argsBuffer != null) _argsBuffer.Release();
            
            // Render texture'ı serbest bırak
            if (_worldPosRT != null) _worldPosRT.Release();
        }
    }

    // ================================================================
    // RANSAC BURST JOB
    // ================================================================
    
    /// <summary>
    /// RANSAC (RANdom SAmple Consensus) ile 3D nokta bulutundan çizgi bulma.
    /// 
    /// ALGORİTMA ÖZETİ:
    ///   1. Rastgele 2 nokta seç → bir çizgi hipotezi oluştur
    ///   2. Tüm noktaların bu çizgiye mesafesini hesapla
    ///   3. Mesafesi Threshold'dan küçük olanları say (inlier)
    ///   4. En çok inlier'a sahip hipotezi "en iyi çizgi" olarak sakla
    ///   5. Bu çizginin inlier'larını "kullanılmış" işaretle
    ///   6. Kalan noktalarla tekrarla (bir sonraki çizgiyi bul)
    ///
    /// OPTİMİZASYONLAR:
    ///   - Hybrid arama: Her 3 iterasyonun 1'i global, 2'si lokal komşuluk
    ///     Global: Tüm noktalardan rastgele seç (uzak çizgileri de yakala)
    ///     Lokal: Yakın indekslerden seç (aynı bölgedeki noktalar genelde yakın)
    ///
    ///   - Ön kontrol (pre-check): 64 rastgele noktaya bak,
    ///     hiç inlier yoksa tam taramayı atla → boş iterasyonları hızlandır
    ///
    ///   - MaxSegLength: İki nokta arası mesafe sınırı
    ///     Farklı objelerin noktalarının yanlışlıkla birleşmesini önler
    ///
    ///   - Segment kırpma (clamping): Sonsuz çizgi yerine
    ///     gerçek inlier noktalarının başı-sonu kullanılır
    ///
    /// Burst Compiler:
    ///   [BurstCompile] ile işaretlendiğinde Unity, bu C# kodunu
    ///   doğrudan native makine koduna (LLVM IR → x86/ARM) derler.
    ///   Mono/IL2CPP'ye göre 5-50x hızlanma sağlar.
    ///   CompileSynchronously = true: İlk çağrıda anında derle (arka planda bekleme).
    /// </summary>
    [BurstCompile(CompileSynchronously = true)]
    public struct LocalRansacJob : IJob
    {
        // Girdi: 3D nokta bulutu (compute shader'dan gelen edge pozisyonları)
        [ReadOnly] public NativeArray<float3> InputPoints;
        
        // Çıktı: Bulunan 3D çizgi segmentleri
        public NativeArray<Line3D> ResultLines;
        
        public int MaxLinesToFind;    // Maksimum bulunacak çizgi sayısı
        public int MaxIterations;     // Her çizgi için RANSAC iterasyon limiti
        public float Threshold;       // Inlier mesafe eşiği (çizgi kalınlığı)
        public int MinInliers;        // Çizginin geçerli sayılması için minimum inlier
        public float MaxSegLength;    // Maksimum segment uzunluğu (farklı objeler arası bağlantıyı önler)
        public uint RandomSeed;       // Rastgele sayı üreteci tohumu

        public void Execute()
        {
            int pointCount = InputPoints.Length;
            if (pointCount < 2) return;  // En az 2 nokta gerekli (çizgi tanımı için)

            // Her noktanın kullanılıp kullanılmadığını takip et
            // Bir çizgiye atanan nokta tekrar kullanılmaz
            NativeArray<bool> usedPoints = new NativeArray<bool>(pointCount, Allocator.Temp);
            
            // Burst-uyumlu rastgele sayı üreteci (System.Random Burst'te çalışmaz)
            Random rng = new Random(RandomSeed > 0 ? RandomSeed : 1);

            int linesFound = 0;
            int totalPointsUsed = 0;
            int consecutiveFailures = 0;  // Art arda başarısız deneme sayacı
            
            // Kare mesafe hesaplamalarını önceden yap (sqrt pahalı, sq ucuz)
            float thresholdSq = Threshold * Threshold;
            float segLengthSq = MaxSegLength * MaxSegLength;

            // Lokal arama yarıçapı (indeks bazında)
            // Buffer'da yakın indeksler genelde 3D'de de yakındır
            // (compute shader pikselleri sıralı yazar)
            int neighborSearchRange = 4000; 
            
            // Ön kontrol örnekleme sayısı
            int preCheckCount = 64; 

            // ==================== ANA DÖNGÜ ====================
            // Her iterasyonda bir çizgi bulmaya çalış
            // Durak koşulları: Yeterli çizgi bulundu VEYA yeterli nokta kalmadı
            while (linesFound < MaxLinesToFind && totalPointsUsed < pointCount - MinInliers)
            {
                float3 bestP1 = float3.zero;    // En iyi çizginin 1. noktası
                float3 bestP2 = float3.zero;    // En iyi çizginin 2. noktası
                int bestInlierCount = -1;       // En iyi çizginin inlier sayısı
                
                // ========== RANSAC İTERASYONLARI ==========
                for (int iter = 0; iter < MaxIterations; iter++)
                {
                    // Rastgele 1. nokta seç
                    int idx1 = rng.NextInt(pointCount);
                    if (usedPoints[idx1]) continue;  // Zaten kullanılmış → atla
                    float3 p1 = InputPoints[idx1];

                    // Hybrid arama stratejisi:
                    // Her 3 iterasyonun 1'i GLOBAL (tüm noktalar), 2'si LOKAL (yakın komşuluk)
                    // Global: Uzak çizgileri de yakalar
                    // Lokal: Aynı bölgedeki noktalar genelde aynı kenara ait → daha hızlı yakınsama
                    bool useGlobalSearch = (iter % 3 == 0); 

                    // Rastgele 2. nokta seç
                    int idx2;
                    if (useGlobalSearch)
                    {
                        idx2 = rng.NextInt(pointCount);  // Tüm noktalardan
                    }
                    else
                    {
                        // Lokal: idx1 ± neighborSearchRange arasından
                        int minIdx = math.max(0, idx1 - neighborSearchRange);
                        int maxIdx = math.min(pointCount, idx1 + neighborSearchRange);
                        idx2 = rng.NextInt(minIdx, maxIdx);
                    }

                    if (idx2 == idx1 || usedPoints[idx2]) continue;
                    float3 p2 = InputPoints[idx2];

                    // İki nokta arası mesafe kontrolü
                    // Çok uzak noktalar farklı objelere ait olabilir → atla
                    if (math.distancesq(p1, p2) > segLengthSq) continue;

                    // Çizgi hipotezi oluştur (p1'den p2'ye yön vektörü)
                    float3 lineVec = math.normalize(p2 - p1);
                    float3 lineStart = p1;

                    // ========== ÖN KONTROL (PRE-CHECK) ==========
                    // Tam taramadan önce 64 rastgele noktaya bak
                    // Hiç inlier yoksa bu hipotez büyük ihtimalle kötüdür → atla
                    // Bu optimizasyon boş iterasyonları ~10x hızlandırır
                    int preInliers = 0;
                    for(int k = 0; k < preCheckCount; k++)
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
                             
                             // Segment uzunluk kontrolü
                             if (math.distancesq(tp, lineStart) <= segLengthSq)
                             {
                                 // Çizgiye olan mesafe = |cross(nokta-çizgibaşı, yönvektörü)|
                                 // cross sonucu 3D vektör, length'i = uzaklık
                                 if (math.lengthsq(math.cross(tp - lineStart, lineVec)) < thresholdSq)
                                 {
                                     preInliers++;
                                     if(preInliers >= 2) break;  // 2 inlier yeterli → devam et
                                 }
                             }
                        }
                    }

                    // Ön kontrol başarısız ve çok nokta varsa → bu hipotezi atla
                    if (preInliers == 0 && pointCount > 2000) continue; 

                    // ========== TAM TARAMA ==========
                    // Tüm kullanılmamış noktaların çizgiye mesafesini hesapla
                    int currentInliers = 0;
                    for (int i = 0; i < pointCount; i++)
                    {
                        if (usedPoints[i]) continue;
                        float3 p = InputPoints[i];
                        
                        // Segment mesafe kontrolü
                        if (math.distancesq(p, lineStart) > segLengthSq) continue;
                        
                        // Çizgiye mesafe: |cross(P-A, dir)| < threshold
                        // lengthsq kullanarak sqrt'ten kaçınıyoruz
                        if (math.lengthsq(math.cross(p - lineStart, lineVec)) < thresholdSq) 
                            currentInliers++;
                    }

                    // En iyi sonucu güncelle
                    if (currentInliers > bestInlierCount)
                    {
                        bestInlierCount = currentInliers;
                        bestP1 = p1; bestP2 = p2;
                        
                        // 100+ inlier yeterince iyi → erken çıkış (hız optimizasyonu)
                        if (bestInlierCount > 100) break; 
                    }
                }

                // ========== SONUÇ DEĞERLENDİRME ==========
                if (bestInlierCount >= MinInliers)
                {
                    consecutiveFailures = 0;
                    
                    // ========== SEGMENT KIRPMA (CLAMPING) ==========
                    // Sonsuz çizgi yerine gerçek inlier noktalarının
                    // başlangıç-bitiş aralığını bul
                    float3 lineDir = math.normalize(bestP2 - bestP1);
                    float minProj = float.MaxValue;   // En küçük projeksiyon (segment başı)
                    float maxProj = float.MinValue;   // En büyük projeksiyon (segment sonu)

                    for (int i = 0; i < pointCount; i++)
                    {
                        if (usedPoints[i]) continue;
                        float3 p = InputPoints[i];
                        if (math.distancesq(p, bestP1) > segLengthSq) continue;

                        float3 vec = p - bestP1;
                        if (math.lengthsq(math.cross(vec, lineDir)) < thresholdSq)
                        {
                            // Bu nokta inlier → kullanıldı olarak işaretle
                            usedPoints[i] = true;
                            totalPointsUsed++;
                            
                            // Çizgi üzerindeki projeksiyonunu hesapla
                            // dot(vec, dir) = noktanın çizgi üzerindeki skaler pozisyonu
                            float proj = math.dot(vec, lineDir);
                            if (proj < minProj) minProj = proj;
                            if (proj > maxProj) maxProj = proj;
                        }
                    }
                    
                    // Segment'i kaydet (projeksiyonlardan gerçek 3D uç noktaları hesapla)
                    if (maxProj > minProj)
                    {
                        ResultLines[linesFound] = new Line3D 
                        { 
                            Start = bestP1 + lineDir * minProj,   // Segment başlangıcı
                            End = bestP1 + lineDir * maxProj,     // Segment bitişi
                            IsValid = true 
                        };
                        linesFound++;
                    }
                }
                else 
                {
                    // Çizgi bulunamadı → ardışık başarısızlık sayacını artır
                    consecutiveFailures++;
                    
                    // 50 ardışık başarısızlık → muhtemelen anlamlı çizgi kalmadı → dur
                    if (consecutiveFailures > 50) break; 
                }
            }
            
            // Geçici belleği serbest bırak
            usedPoints.Dispose();
        }
    }
}