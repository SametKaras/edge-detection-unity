# WorldSpaceEdgeManager.cs — Detaylı Açıklama

## Genel Amaç
CPU tarafı orkestratör. GPU pipeline'ını yönetir, async readback yapar, sonuçları Gizmo olarak çizer.

---

## Sınıf Yapısı

```csharp
[RequireComponent(typeof(EdgeDetectionEffect))]
public class WorldSpaceEdgeManager : MonoBehaviour
```
Ana kameraya eklenir. `EdgeDetectionEffect` ile birlikte çalışır.

---

## Inspector Parametreleri

### Detection Sensitivity
```csharp
[Range(0.01f, 0.9f)] public float minEdgeLuminance = 0.1f;
// Edge magnitude bu değerin altındaysa → edge değil
// Düşük değer = daha fazla edge (gürültü dahil)
// Yüksek değer = sadece güçlü edge'ler

[Range(0.0f, 0.05f)] public float nmsRelaxation = 0.008f;
// NMS toleransı: 0 = çok sıkı (sadece tam tepe), 0.05 = gevşek
```

### Edge Geometry Filter
```csharp
public bool useGeomDiscontinuityFilter = true;
// Açıkken: Komşu pikseller arasındaki 3D mesafe kontrol edilir
// Mesafe < minGeomDiscontinuity → smooth yüzey → edge sayma

[Range(0.0005f, 0.05f)] public float minGeomDiscontinuity = 0.006f;
// World unit cinsinden. Daha düşük → daha hassas filtreleme

public bool requireBoundaryTransition = false;
// Açıkken: Sadece valid/invalid geçişinde edge üretilir (en sıkı mod)
```

### RANSAC Parametreleri
```csharp
public KernelSizeOption kernelSize = KernelSizeOption._5x5;
// Tile boyutu: 3x3, 5x5, ..., 21x21
// Büyük → daha az ama daha uzun çizgi, çapraz riski artar
// Küçük → daha çok ama daha kısa çizgi, zigzag riski artar

[Range(2, 25)] public int minPointsForLine = 2;
// Tile'da en az bu kadar edge pikseli olmalı

[Range(0.01f, 0.5f)] public float inlierThreshold = 0.08f;
// RANSAC inlier mesafe eşiği (world unit)
// 0.01 = çok sıkı (cube), 0.2 = gevşek (sphere/capsule)
// Two-pass adaptive: tight = 0.05×, loose = 1.0×

[Range(0.01f, 10f)] public float maxSegmentLength = 0.15f;
// Segment uzunluğu bu değeri aşarsa → reddet
// Uzak nesneler için artırılmalı
```

### Resolution Normalization
```csharp
public bool autoScaleKernelWithResolution = true;
// 1080p'de 5×5 → 4K'da otomatik 10×10'a yakın kernel kullanır
// Böylece farklı çözünürlüklerde benzer sonuç alınır

[Range(720, 2160)] public int referenceHeight = 1080;
// Referans çözünürlük: Bu yükseklikte kernel olduğu gibi kullanılır
```

### Line Quality Filters
```csharp
[Range(0.3f, 0.8f)] public float minInlierRatio = 0.52f;
// RANSAC inlier'ların toplam noktaya oranı bu değerin altındaysa → reddet
// Vertex bölgelerinde noktalar dağınık → düşük oran → çizgi üretilmez

[Range(1.0f, 3.0f)] public float minLinearityFactor = 1.5f;
// 3D linearity testi: alongVar >= perpVar × factor olmalı
// Yüksek değer = daha sıkı doğrusallık kontrolü
```

---

## Internal Yapı

### Buffer'lar
```csharp
// MicroLine çıktı buffer'ı — GPU'da append, CPU'da okunur
private ComputeBuffer _lineBuffer;     // AppendStructuredBuffer<MicroLine>
private ComputeBuffer _countBuffer;     // Append buffer'ın eleman sayısı

// Debug buffer — raw edge noktaları export için
private ComputeBuffer _debugBuffer;     // AppendStructuredBuffer<MicroLine>
private ComputeBuffer _debugCountBuffer;

// Render Target'lar
private RenderTexture _worldPosRT;      // Her piksel → world position (ARGBFloat)
private RenderTexture _edgePosRT;       // BuildEdgePositionBuffer çıktısı (RW)
```

### Async Readback State Machine
```csharp
enum ReadbackState { Idle, WaitingCount, WaitingLines }
```

```
       DispatchFrame()
            │
            ▼
    ┌─── Idle ◄──────────────────────┐
    │                                 │
    │  ComputeBuffer.CopyCount()     │
    │  AsyncGPUReadback.Request()    │
    │                                 │
    ▼                                 │
WaitingCount ──(countReq.done)──→ count = 0? ──YES──→ Idle
    │                                 │
    │                              count > 0
    │                                 │
    │  AsyncGPUReadback.Request(    │
    │    lineBuffer, count*STRIDE)   │
    │                                 │
    ▼                                 │
WaitingLines ──(lineReq.done)──→ Copy to ────→ Idle
                              _displayLines[]
```

**Neden Async**: `GetData()` CPU'yu GPU bitmesini beklerken bloklar (~1-5ms). Async readback, GPU bittiğinde otomatik bildirim alır → CPU kayıp yok.

---

## Ana Metotlar

### ComputeEffectiveKernelSize()
```csharp
int ComputeEffectiveKernelSize()
{
    float ratio = Screen.height / referenceHeight;  // Çözünürlük oranı
    float scale = Pow(ratio, kernelScaleExponent);  // Üstel ölçekleme
    int k = Round(baseK * scale);                   // Etkin kernel
    k = Clamp(k, 3, 21);                            // Sınırla
    if (k % 2 == 0) k++;                            // Tek sayı olmalı
    return k;
}
```
**Örnek**: `kernelSize=5`, `referenceHeight=1080`, ekran `2160p` → `ratio=2.0` → `k=10→11`

### DispatchFrame()
```csharp
void DispatchFrame()
{
    // 1. WorldPos kamerasını senkronize et
    _posCam.CopyFrom(_mainCam);
    
    // 2. WorldPos render (replacement shader ile)
    _posCam.RenderWithShader(worldPosShader, "RenderType");
    
    // 3. Kernel 1: BuildEdgePositionBuffer
    microLineCS.Dispatch(buildKernel,
        CeilToInt(width / 8f),    // Her piksel = 1 thread
        CeilToInt(height / 8f),
        1);
    
    // 4. Kernel 2: FitMicroLines
    microLineCS.Dispatch(fitKernel,
        CeilToInt(tilesX / 8f),   // Her tile = 1 thread
        CeilToInt(tilesY / 8f),
        1);
    
    // 5. Async readback başlat
    ComputeBuffer.CopyCount(_lineBuffer, _countBuffer, 0);
    _countReq = AsyncGPUReadback.Request(_countBuffer);
}
```

### OnDrawGizmos()
```csharp
void OnDrawGizmos()
{
    // Micro-line'ları kırmızı çizgi olarak çiz
    Handles.DrawAAPolyLine(thickness, start, end);
    
    // Debug: Raw edge noktalarını cyan küre olarak çiz
    Gizmos.DrawSphere(position, rawPointSize);
}
```

---

## Zamanlama

```csharp
// Stopwatch ile mikrosaniye hassasiyetinde ölçüm
_dispatchStartTick = _sw.ElapsedTicks;    // GPU dispatch öncesi
// ... Dispatch ...
_dispatchEndTick = _sw.ElapsedTicks;      // GPU dispatch sonrası
// ... Readback ...
_readbackDoneTick = _sw.ElapsedTicks;     // CPU readback sonrası

_algorithmMs = (dispatchEnd - dispatchStart) * ticksToMs;
_readbackMs = (readbackDone - dispatchEnd) * ticksToMs;
```
> **Not**: `_algorithmMs` sadece CPU'nun dispatch komutunu göndermesini ölçer (GPU'nun gerçek çalışma süresi değil). GPU profiling için Unity Profiler kullanılmalı.
