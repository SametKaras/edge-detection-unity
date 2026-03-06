# EdgeToMicroLines.compute — Detaylı Açıklama

## Genel Yapı
Bu compute shader iki kernel'dan oluşur:
1. **BuildEdgePositionBuffer**: Her edge pikseli için en doğru 3D world position'ı belirler
2. **FitMicroLines**: Tile bazlı 3D RANSAC + PCA ile micro-line'lar üretir

---

## Sabitler ve Parametreler

```hlsl
#define MAX_PTS 441    // Tile içinde toplanabilecek max nokta (21×21)
#define PCA_ITER 6     // 2D PCA power iteration sayısı
#define PCA_ITER_3D 8  // 3D PCA power iteration sayısı
```

| Parametre | Tip | Açıklama |
|-----------|-----|----------|
| `_KernelSize` | int | Tile boyutu (3–21, tek sayı) |
| `_MinEdgeThreshold` | float | Edge magnitude eşiği |
| `_NmsRelaxation` | float | NMS toleransı (sub-pixel düzeltme) |
| `_InlierThreshold` | float | RANSAC inlier mesafe eşiği (world unit) |
| `_MaxSegLength` | float | Max segment uzunluğu (çok uzun çizgileri filtreler) |
| `_MinPointsForLine` | int | Çizgi üretmek için gereken min nokta sayısı |
| `_MinInlierRatio` | float | Kabul edilebilir min inlier oranı |
| `_MinLinearityFactor` | float | Çizgisellik testi çarpanı |
| `_DebugMode` | int | 1 = RANSAC yerine ham noktaları export et |

---

## Yardımcı Fonksiyonlar

### WangHash — Deterministik Rastgele Sayı
```hlsl
uint WangHash(uint seed)
{
    seed = (seed ^ 61u) ^ (seed >> 16u);
    seed *= 9u;
    seed = seed ^ (seed >> 4u);
    seed *= 0x27d4eb2du;
    seed = seed ^ (seed >> 15u);
    return seed;
}
```
RANSAC'ta rastgele nokta çifti seçmek için kullanılır. Her thread farklı seed alır → paralel çalışır.

### EdgeGradSobel — Edge Gradyan Yönü
```hlsl
float2 EdgeGradSobel(int px, int py)
```
Edge magnitude texture üzerinde Sobel gradyanı hesaplar. Bu gradyan, kenarın **dik yönünü** verir — silhouette/crease ayrımı ve NMS için kritik.

### IsEdgeNmsPeak — Non-Maximum Suppression
```hlsl
bool IsEdgeNmsPeak(int px, int py, float edgeCenter, float2 grad)
{
    float2 n = grad / length(grad);           // Gradyan yönü (normalize)
    float e1 = SampleEdgeBilinear(p + n);     // +gradyan yönünde komşu
    float e2 = SampleEdgeBilinear(p - n);     // -gradyan yönünde komşu
    return edgeCenter + _NmsRelaxation >= max(e1, e2);
    // Merkez piksel, gradyan boyunca en yüksekse → gerçek kenar
}
```
**NMS Neden Gerekli**: Edge detection tüm yüzeyde genellikle kalın bantlar üretir. NMS bu bantı tek piksel genişliğine indirir — sadece tepe noktaları kalır.

### SubpixelPeakOffset — Sub-piksel Düzeltme
```hlsl
float SubpixelPeakOffset(float2 p, float2 n)
{
    float eM = SampleEdgeBilinear(p - n);  // Önceki piksel
    float e0 = SampleEdgeBilinear(p);       // Merkez piksel
    float eP = SampleEdgeBilinear(p + n);   // Sonraki piksel
    
    // Parabolik interpolasyon: 3 noktadan tepe konumunu tahmin et
    float t = 0.5 * (eM - eP) / (eM - 2*e0 + eP);
    return clamp(t, -0.5, 0.5);  // Piksel merkezinden max ±0.5 kayma
}
```
**Neden sub-piksel**: Gerçek kenar tam piksel sınırına düşmez. Bu düzeltme kenarın gerçek konumunu ~0.1 piksel hassasiyetle bulur.

### SampleWorldPosBilinearValid — Bilinear World Position
```hlsl
float4 SampleWorldPosBilinearValid(float2 p)
```
Sub-piksel konumunda bilinear interpolasyonla world position örnekler. **Geçersiz pikselleri** (w=0) otomatik olarak dışlar — sadece geçerli komşulardan ağırlıklı ortalama alır.

### PickBestWorldPos — Silhouette/Crease Karar Mekanizması
```hlsl
float4 PickBestWorldPos(int px, int py, float2 n, float2 subPix, out bool sideMismatch)
{
    // Gradyan yönünün iki tarafındaki world position'ları kontrol et
    float4 s1 = LoadWorld(px + sx, py + sy);  // +yön tarafı
    float4 s2 = LoadWorld(px - sx, py - sy);  // -yön tarafı
    
    bool v1 = IsValidPos(s1);  // +taraf geçerli mi?
    bool v2 = IsValidPos(s2);  // -taraf geçerli mi?
    sideMismatch = (v1 != v2);  // Bir taraf geçersiz → muhtemelen silhouette
    
    // Karar tablosu:
    // v1=✓, v2=✗ → Silhouette: +taraftan örnekle
    // v1=✗, v2=✓ → Silhouette: -taraftan örnekle
    // v1=✓, v2=✓ → Crease: En yakın tarafı seç, bilinear ile blend
    // v1=✗, v2=✗ → Geçersiz → atla
}
```

---

## Kernel 1: BuildEdgePositionBuffer

```hlsl
[numthreads(8, 8, 1)]
void BuildEdgePositionBuffer(uint3 id : SV_DispatchThreadID)
```

**Her piksel için bağımsız çalışır** (piksel başına 1 thread).

### İşlem Adımları:

```
1. Edge magnitude >= eşik mi?  ──NO──→ _RWEdgePosTex = (0,0,0,0) → RETURN
                                  │
2. Sobel gradyanı hesapla          YES
   Gradyan çok zayıf mı?    ──YES──→ _RWEdgePosTex = (0,0,0,0) → RETURN
                                  │
3. NMS (Non-Maximum Suppression)   NO
   Bu piksel lokal tepe mi?  ──NO───→ _RWEdgePosTex = (0,0,0,0) → RETURN
                                  │
4. Sub-pixel düzeltme              YES
   Parabolik interpolasyonla
   gerçek kenar konumunu bul
                  │
5. PickBestWorldPos
   Silhouette/Crease ayrımı
   En doğru 3D pozisyonu seç
                  │
6. Opsiyonel filtreler:
   - Boundary transition kontrolü
   - Geometrik süreksizlik filtresi
                  │
7. _RWEdgePosTex = float4(worldX, worldY, worldZ, 1.0)
```

### Geometrik Süreksizlik Filtresi
```hlsl
if (_UseGeomDiscontinuityFilter > 0)
{
    float4 a = LoadWorld(px + sx, py + sy);  // Bir taraftaki world pos
    float4 b = LoadWorld(px - sx, py - sy);  // Diğer taraftaki world pos
    
    float d = length(a.xyz - b.xyz);         // İki taraf arası 3D mesafe
    if (d < _MinGeomDiscontinuity)            // Mesafe çok küçük → smooth yüzey
    {
        // Smooth yüzey + zayıf edge = texture/lighting artifact → ATLA
        if (!nearBoundary && !strongEdge) return;
    }
}
```

---

## Kernel 2: FitMicroLines

```hlsl
[numthreads(8, 8, 1)]
void FitMicroLines(uint3 id : SV_DispatchThreadID)
```

**Her tile için bağımsız çalışır** (tile başına 1 thread). Ekran `_KernelSize × _KernelSize` boyutunda tile'lara bölünür.

### Aşama 1: Nokta Toplama
```hlsl
// Tile içindeki her piksel için _EdgePosTex'ten geçerli world position'ları topla
for (y = startY; y < endY; y++)
    for (x = startX; x < endX; x++)
    {
        float4 ep = _EdgePosTex[uint2(x, y)];
        if (ep.w <= 0.0) continue;       // Edge değilse atla
        pts3d[cnt] = ep.xyz;             // 3D pozisyon
        pts2d[cnt] = float2(x, y);       // 2D piksel koordinatı
        cnt++;
    }

if (cnt < _MinPointsForLine) return;     // Yeterli nokta yoksa çıktı yok
```

### Aşama 2: Debug Mode
```hlsl
if (_DebugMode > 0)
{
    // RANSAC yapmadan ham noktaları doğrudan export et
    // → MeshLab'da PLY olarak incelenebilir
    for (i = 0; i < cnt; i++)
        _DebugPoints.Append(pts3d[i]);
    return;
}
```

### Aşama 3: 2D PCA Ön-Filtreleme (Histogram Multi-Edge Detection)

Büyük kernel'larda (≥7×7) tile içinde **birden fazla kenar** olabilir. Bu aşama dominant kenarı izole eder.

```hlsl
// 1. Tüm noktaların 2D PCA'sını hesapla → kenar doğrultusu
float2 mean2d = ortalama(pts2d);
// Kovaryans matrisi → power iteration → principal direction (pcaDir)

// 2. Her noktayı kenar doğrultusunun DİK yönüne projekte et
float2 pcaNorm = float2(-pcaDir.y, pcaDir.x);  // Dik yön
float di = dot(pts2d[i] - mean2d, pcaNorm);     // Dikine mesafe
```

```
Dikine mesafe dağılımı (2 kenar varsa):
                      
  Sayı │  ██                    
       │  ██                    ██
       │  ██  ██                ██
       │  ██  ██  ██        ██  ██
       └──────────────────────────→ Dikine mesafe
          Kenar 1    Boşluk    Kenar 2
          (dominant)
```

```hlsl
// 3. 12 bin'lik histogram oluştur
const int BINS = 12;
int hist[BINS]; // Her bin'de kaç nokta var?

// 4. En dolu bin'i bul = dominant kenar
// 5. Sadece dominant kenar etrafındaki noktaları tut
float band = max(binSize * lerp(1.12, 0.92, kT), ...);
if (abs(dist[i] - center) <= band)  // Band içinde mi?
    pts3d[keep] = pts3d[i];          // Tut
cnt = keep;                          // Sadece filtered noktalar kaldı
```

### Aşama 4: Adaptive Two-Pass 3D RANSAC

```hlsl
// İki aşamalı threshold:
float tightInlier = effInlier * 0.05;  // Sıkı: cube kenarları için
float looseInlier = effInlier;          // Gevşek: sphere/capsule silhouette için

// Aynı iterasyonlarda her iki threshold'u da test et
for (it = 0; it < maxIter; it++)
{
    // Rastgele 2 nokta seç
    int i1 = seed % cnt;
    int i2 = seed % cnt;
    
    // Bu 2 noktadan geçen doğruyu tanımla
    float3 dir = normalize(pts3d[i2] - pts3d[i1]);
    
    // Her noktanın doğruya mesafesini hesapla
    for (j = 0; j < cnt; j++)
    {
        float3 d = pts3d[j] - pts3d[i1];
        float3 cr = cross(d, dir);        // Çapraz çarpım
        float distSq = dot(cr, cr);       // Mesafe² = |cross|²
        
        if (distSq < thSqTight) inlTight++;  // Her iki eşiği
        if (distSq < thSqLoose) inlLoose++;  // paralel kontrol et
    }
}
```

```
Karar mantığı:

  Sıkı inlier oranı ≥ %40?
         │
     YES │         NO
         │          │
    Düz kenar!   Eğrisel kenar!
    (cube)       (sphere/capsule)
         │          │
    thSq = tight   thSq = loose
    (0.01²)        (0.2²)
```

**Neden two-pass**: Cube kenarında komşu yüz noktaları tight eşikle filtrelenir → çapraz çizgi oluşmaz. Sphere'de ise tight yetmez → loose devreye girer → çizgiler korunur.

### Aşama 5: PCA Refinement

RANSAC'ın bulduğu yön 2 rastgele noktaya bağlıdır → gürültülü olabilir. PCA tüm inlier'ları kullanarak **optimal yönü** bulur.

```hlsl
// 1. İnlier'ların 3D ortalaması (centroid)
for (i = 0; i < cnt; i++)
    if (isInlier(i)) mean3d += pts3d[i];
mean3d /= inlierCount;

// 2. 3×3 Kovaryans matrisi
//    C = Σ (pi - mean) × (pi - mean)ᵀ
for (i = 0; i < cnt; i++)
    if (isInlier(i))
    {
        float3 dv = pts3d[i] - mean3d;
        c11 += dv.x * dv.x;  c12 += dv.x * dv.y;  c13 += dv.x * dv.z;
                              c22 += dv.y * dv.y;   c23 += dv.y * dv.z;
                                                     c33 += dv.z * dv.z;
    }
// Not: Matris simetrik → sadece üst üçgen yeterli (6 eleman)

// 3. Power Iteration (8 iterasyon)
//    v(k+1) = normalize(C × v(k))
//    Converge eder → dominant eigenvector (en büyük varyans yönü = çizgi yönü)
float3 v = bestDir;  // RANSAC yönünü başlangıç tahmini olarak kullan
for (it = 0; it < 8; it++)
{
    float3 Cv = float3(
        c11*v.x + c12*v.y + c13*v.z,   // Matris-vektör çarpımı
        c12*v.x + c22*v.y + c23*v.z,
        c13*v.x + c23*v.y + c33*v.z
    );
    v = normalize(Cv);  // Normalize et → sonraki iterasyon
}
// v = en hassas 3D çizgi yönü
```

### Aşama 6: Endpoint Projection + Outlier Trimming

```hlsl
// Her inlier'ı PCA yönüne projekte et
for (i = 0; i < cnt; i++)
    if (isInlier(i))
    {
        float3 dm = pts3d[i] - mean3d;
        float p = dot(dm, v);           // Projeksiyon: çizgi üzerindeki konum
        minP = min(minP, p);            // En gerideki nokta
        maxP = max(maxP, p);            // En öndeki nokta
        
        float3 q = dm - v * p;          // Çizgiye dik bileşen
        perpAccum += dot(q, q);         // Dik varyans (çizgisellik ölçüsü)
    }

// Outlier tail trimming (sigma bazlı)
// Projeksiyon dağılımının ucundaki aşırı noktaları kes
float sigma = sqrt(varyans);
float lo = mean - sigma * 2.0;
float hi = mean + sigma * 2.0;
// [lo, hi] dışındaki noktaları endpoint hesabından çıkar
```

### Aşama 7: Kalite Filtreleri

#### a) Occupancy Test (k ≥ 19 için)
```hlsl
// Çizgi boyunca noktalar eşit mi dağılmış?
// 14 bin'e böl → her bin'de nokta var mı kontrol et
// Büyük boşluklar varsa → iki farklı kenar segmenti birleşmiş → REDDET
if (runs >= 2 && maxGap >= 2) return;
```

#### b) 3D Linearity Test
```hlsl
// Çizgi boyunca varyans vs. çizgiye dik varyans
// alongVar >> perpVar → noktalar çizgisel ✓
// alongVar ≈ perpVar → noktalar dağınık (vertex bölgesi) ✗
float alongVar = (segLen²) * 0.0833;     // Uniform dağılım varsayımı
float perpVar = perpAccum / perpCount;    // Ölçülen dik varyans
if (alongVar < perpVar * _MinLinearityFactor) return;
```

#### c) 2D Straightness Test
```hlsl
// İnlier'ların 2D piksel koordinatlarında da düz çizgi oluşturup oluşturmadığını kontrol et
// 2D PCA → along/perp varyans oranı
// Oranı düşükse (noktalar 2D'de dağınık) → REDDET
bool pass2D = along2Var >= perp2Var * straight2D;
if (!pass2D && !(yüksek_inlier_oranı)) return;
```

### Aşama 8: Çıktı
```hlsl
// Çizgi uç noktaları: centroid ± projeksiyon
float3 segStart = mean3d + v * minP;
float3 segEnd   = mean3d + v * maxP;

MicroLine ml;
ml.sx = segStart.x; ml.sy = segStart.y; ml.sz = segStart.z;
ml.ex = segEnd.x;   ml.ey = segEnd.y;   ml.ez = segEnd.z;
_OutputLines.Append(ml);  // GPU append buffer'a ekle
```

---

## Adaptive Parametreler

Kernel boyutu ve nokta sayısına göre birçok parametre dinamik olarak ayarlanır:

```hlsl
float kT = saturate(((float)k - 5.0) / 16.0);  // Kernel büyüklük faktörü [0, 1]
// k=5 → kT=0, k=21 → kT=1

float cntT = saturate((float)cnt / 20.0);       // Nokta yoğunluk faktörü [0, 1]
// Az nokta → gevşek parametreler, çok nokta → sıkı parametreler
```

| Parametre | Az Nokta (cntT=0) | Çok Nokta (cntT=1) |
|-----------|-------------------|---------------------|
| `effInlier` | × 1.35 (gevşek) | × 1.0 (orijinal) |
| `effMaxLen` | × 1.55 (uzun) | × 1.0 (orijinal) |
| `effMinRatio` | × 0.52 (gevşek) | × 1.0 (sıkı) |
