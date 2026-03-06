# EdgeDetection.shader — Detaylı Açıklama

## Genel Amaç
Bu fragment shader, kamera görüntüsündeki her piksel için **edge magnitude** (kenar şiddeti) hesaplar. Çıktısı, compute shader'ın girişi olarak kullanılır.

## Desteklenen Kenar Tespit Yöntemleri

### Sobel Operatörü (3×3)
```
Gx kernel:          Gy kernel:
[-1  0 +1]          [-1 -2 -1]
[-2  0 +2]          [ 0  0  0]
[-1  0 +1]          [+1 +2 +1]
```
```hlsl
// Sobel: 3×3 komşuluktan yatay (gx) ve dikey (gy) gradyan hesaplar
float2 Sobel(float2 uv, float2 t)
{
    // t = _MainTex_TexelSize.xy → bir pikselin UV boyutu
    // 8 komşu pikselin "source value"sini al
    float tl = GetSourceValue(uv + float2(-t.x, t.y));  // Sol üst
    float tm = GetSourceValue(uv + float2(0, t.y));      // Üst orta
    // ... (8 komşu)
    
    // Sobel kernel çarpımları
    float gx = -tl - 2.0*ml - bl + tr + 2.0*mr + br;   // Yatay gradyan
    float gy = -tl - 2.0*tm - tr + bl + 2.0*bm + br;   // Dikey gradyan
    return float2(gx, gy);  // Gradyan vektörü
}
```
**Magnitude**: `sqrt(gx² + gy²)` → Kenar şiddeti. Yüksek değer = güçlü kenar.

### Roberts Cross (2×2)
```hlsl
// Roberts: Çapraz fark — daha basit, daha az dayanıklı
float2 Roberts(float2 uv, float2 t)
{
    float c  = GetSourceValue(uv);                       // Merkez piksel
    float br = GetSourceValue(uv + float2(t.x, -t.y));  // Sağ alt
    float r  = GetSourceValue(uv + float2(t.x, 0));     // Sağ
    float b  = GetSourceValue(uv + float2(0, -t.y));     // Alt
    
    return float2(c - br, r - b);  // 2×2 çapraz fark
}
```

### Prewitt (3×3)
Sobel'e benzer ama ağırlıksız — tüm komşular eşit katkı yapar.

---

## Edge Source (Kenar Kaynağı)

### Depth Edge
```hlsl
// Derinlik haritasından kenar: Nesne sınırlarında büyük derinlik farkı oluşur
float dc = SampleDepth(uv);                    // Merkez piksel derinliği
float dt = SampleDepth(uv + float2(0, t.y));   // Üst komşu derinliği
// ...
float de = abs(dc-dt) + abs(dc-db) + abs(dc-dl) + abs(dc-dr);
depthEdge = de * _DepthSensitivity * _DepthWeight;
```
**Ne zaman çalışır**: Silhouette kenarları (nesne→arka plan geçişi), derinlik süreksizlikleri.

### Normal Edge + Crease Filter
```hlsl
// Her komşu normal ile merkez normal arasındaki açıyı ölç
float3 nc = SampleNormal(uv);                  // Merkez piksel normal'ı
float3 nt = SampleNormal(uv + float2(0, t.y)); // Üst komşu normal'ı

float dotT = saturate(dot(nc, nt));  // 1.0 = aynı yön, 0.0 = dik açı
float sharpT = 1.0 - dotT;          // Açı farkı → kenar şiddeti

// CREASE FİLTRESİ:
// dot > _MinCreaseDot → açı çok küçük → sphere/capsule mesh edge → ATLA
// dot < _MinCreaseDot → açı büyük → gerçek crease (kutu köşesi) → EKLE
if (dotT < _MinCreaseDot) ne += pow(1.0 - dotT, 2);
```
**Crease Filter Mantığı**: Sphere mesh'inin üçgenleri arasında ~10-20° fark var. `_MinCreaseDot = 0.9` (≈cos(26°)) ayarıyla bu küçük açılar filtrelenir, sadece gerçek köşeler (>26°) edge olarak algılanır.

### Combined Mode
```hlsl
// Üç kaynağın ağırlıklı toplamı
return depthEdge + normalEdge + colorEdge;
```

### Hybrid Mode
```hlsl
// Yerel "keskinlik" (normal farkı) bazında dinamik ağırlıklandırma
float creaseMask = smoothstep(0, 1, (localSharpness - start) / (end - start));

// Smooth bölge: sadece depth kenarları
// Keskin bölge: normal + color katkısı artar
float hybridNormal = normalEdge * lerp(0.0, _HybridNormalBoost, creaseMask);
return depthEdge + hybridNormal + hybridColor;
```

---

## Çıktı Modları

```hlsl
if (_OutputMagnitude > 0.5)
{
    // Thinning pass için: sürekli (continuous) magnitude değeri [0, 1]
    // → Compute shader bu değeri edge tespiti için kullanır
    float normalizedMag = saturate(mag * _MagnitudeScale);
    return fixed4(normalizedMag, normalizedMag, normalizedMag, 1);
}

// Normal mod: Binary threshold → siyah/beyaz
float edge = step(_EdgeThreshold, mag);
return lerp(_BackgroundColor, _EdgeColor, edge);
```

İki render pass yapılır:
1. **Pass 1** (`_OutputMagnitude=1`): Edge magnitude → `EdgeResultTexture` (compute shader için)
2. **Pass 2** (`_OutputMagnitude=0`): Binary threshold → ekran (görselleştirme için)
