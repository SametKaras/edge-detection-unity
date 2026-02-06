// ============================================================
// EdgeDirection.shader
// Her pikselin gradyan yönünü RGB olarak çıktı verir
// R = gx (normalize), G = gy (normalize), B = magnitude
//
// İYİLEŞTİRME:
//   ÖNCEKİ SORUN: Sadece Luminance üzerinden Sobel gradient hesaplıyordu.
//   EdgeDetection.shader Combined modda (Depth + Normal + Color) çalışırken
//   EdgeDirection.shader hep Luminance kullanıyordu → Gradient yönü tutarsızlığı.
//   
//   ÇÖZÜM: EdgeDetection.shader ile aynı multi_compile keyword'leri eklendi.
//   Artık hangi kaynak seçilirse (Depth/Normal/Combined) gradient de o kaynağa göre hesaplanır.
// ============================================================

Shader "Custom/EdgeDirection"
{
    Properties
    {
        _MainTex ("Texture", 2D) = "white" {}
        _EdgeThreshold ("Edge Threshold", Range(0.001, 1.0)) = 0.1
        
        // İYİLEŞTİRME: EdgeDetection.shader ile aynı parametreler
        [KeywordEnum(Luminance, Depth, Normal, Combined)]
        _Source ("Edge Source", Float) = 0
        
        _DepthSensitivity ("Depth Sensitivity", Range(0.1, 100)) = 10.0
        _MaxDepth ("Max Depth", Float) = 50.0
        _NormalSensitivity ("Normal Sensitivity", Range(0.1, 10)) = 1.0
        
        _DepthWeight ("Depth Weight", Range(0, 1)) = 0.5
        _NormalWeight ("Normal Weight", Range(0, 1)) = 0.5
        _ColorWeight ("Color Weight", Range(0, 1)) = 0.3
    }
    
    SubShader
    {
        Tags { "RenderType"="Opaque" }
        
        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            
            // İYİLEŞTİRME: Multi-source desteği
            #pragma multi_compile _SOURCE_LUMINANCE _SOURCE_DEPTH _SOURCE_NORMAL _SOURCE_COMBINED
            
            #include "UnityCG.cginc"
            
            struct appdata
            {
                float4 vertex : POSITION;
                float2 uv : TEXCOORD0;
            };
            
            struct v2f
            {
                float2 uv : TEXCOORD0;
                float4 vertex : SV_POSITION;
            };
            
            sampler2D _MainTex;
            float4 _MainTex_TexelSize;
            float _EdgeThreshold;
            
            // İYİLEŞTİRME: Depth/Normal texture'ları
            sampler2D _CameraDepthTexture;
            sampler2D _CameraDepthNormalsTexture;
            
            float _DepthSensitivity;
            float _MaxDepth;
            float _NormalSensitivity;
            float _DepthWeight;
            float _NormalWeight;
            float _ColorWeight;
            
            v2f vert (appdata v)
            {
                v2f o;
                o.vertex = UnityObjectToClipPos(v.vertex);
                o.uv = v.uv;
                return o;
            }
            
            float Luminance(float3 color)
            {
                return dot(color, float3(0.299, 0.587, 0.114));
            }
            
            // İYİLEŞTİRME: Depth ve Normal sampling fonksiyonları
            float SampleDepth(float2 uv)
            {
                float depth = SAMPLE_DEPTH_TEXTURE(_CameraDepthTexture, uv);
                return LinearEyeDepth(depth);
            }
            
            float3 SampleNormal(float2 uv)
            {
                float4 dn = tex2D(_CameraDepthNormalsTexture, uv);
                float3 normal;
                float depth;
                DecodeDepthNormal(dn, depth, normal);
                return normal;
            }
            
            // ============================================================
            // KAYNAK DEĞERİ ALMA — EdgeDetection.shader ile tutarlı
            // ============================================================
            float GetSourceValue(float2 uv)
            {
                #if _SOURCE_LUMINANCE
                    return Luminance(tex2D(_MainTex, uv).rgb);
                    
                #elif _SOURCE_DEPTH
                    return saturate(SampleDepth(uv) / _MaxDepth) * _DepthSensitivity;
                    
                #elif _SOURCE_NORMAL
                    return length(SampleNormal(uv)) * _NormalSensitivity;
                    
                #else // _SOURCE_COMBINED veya fallback
                    return Luminance(tex2D(_MainTex, uv).rgb);
                #endif
            }
            
            // ============================================================
            // SOBEL GRADIENT — Tüm kaynaklar için çalışır
            // ============================================================
            float2 SobelGradient(float2 uv, float2 t)
            {
                float tl = GetSourceValue(uv + float2(-t.x, t.y));
                float tm = GetSourceValue(uv + float2(0, t.y));
                float tr = GetSourceValue(uv + float2(t.x, t.y));
                float ml = GetSourceValue(uv + float2(-t.x, 0));
                float mr = GetSourceValue(uv + float2(t.x, 0));
                float bl = GetSourceValue(uv + float2(-t.x, -t.y));
                float bm = GetSourceValue(uv + float2(0, -t.y));
                float br = GetSourceValue(uv + float2(t.x, -t.y));
                
                float gx = -tl - 2.0*ml - bl + tr + 2.0*mr + br;
                float gy = -tl - 2.0*tm - tr + bl + 2.0*bm + br;
                
                return float2(gx, gy);
            }
            
            // ============================================================
            // İYİLEŞTİRME: COMBINED GRADIENT
            // EdgeDetection.shader'daki GetCombinedEdge ile tutarlı
            // Fark: Skalar edge yerine 2D gradient vektörü döndürür
            // ============================================================
            #if _SOURCE_COMBINED
            float2 GetCombinedGradient(float2 uv, float2 t)
            {
                float2 gradient = float2(0, 0);
                
                // RENK KENARLARI (Sobel on Luminance)
                if (_ColorWeight > 0.01)
                {
                    float tl = Luminance(tex2D(_MainTex, uv + float2(-t.x, t.y)).rgb);
                    float tm = Luminance(tex2D(_MainTex, uv + float2(0, t.y)).rgb);
                    float tr = Luminance(tex2D(_MainTex, uv + float2(t.x, t.y)).rgb);
                    float ml = Luminance(tex2D(_MainTex, uv + float2(-t.x, 0)).rgb);
                    float mr = Luminance(tex2D(_MainTex, uv + float2(t.x, 0)).rgb);
                    float bl = Luminance(tex2D(_MainTex, uv + float2(-t.x, -t.y)).rgb);
                    float bm = Luminance(tex2D(_MainTex, uv + float2(0, -t.y)).rgb);
                    float br = Luminance(tex2D(_MainTex, uv + float2(t.x, -t.y)).rgb);
                    
                    float cgx = -tl - 2.0*ml - bl + tr + 2.0*mr + br;
                    float cgy = -tl - 2.0*tm - tr + bl + 2.0*bm + br;
                    
                    gradient += float2(cgx, cgy) * _ColorWeight;
                }
                
                // DERİNLİK KENARLARI (Sobel on Depth)
                if (_DepthWeight > 0.01)
                {
                    float dc = SampleDepth(uv);
                    float dt = SampleDepth(uv + float2(0, t.y));
                    float db = SampleDepth(uv + float2(0, -t.y));
                    float dl = SampleDepth(uv + float2(-t.x, 0));
                    float dr = SampleDepth(uv + float2(t.x, 0));
                    
                    // 2D gradient from depth differences
                    float dgx = (dr - dl) * _DepthSensitivity;
                    float dgy = (dt - db) * _DepthSensitivity;
                    
                    gradient += float2(dgx, dgy) * _DepthWeight;
                }
                
                // NORMAL KENARLARI (Sobel on Normal dot product)
                if (_NormalWeight > 0.01)
                {
                    float3 nc = SampleNormal(uv);
                    float3 nt = SampleNormal(uv + float2(0, t.y));
                    float3 nb = SampleNormal(uv + float2(0, -t.y));
                    float3 nl = SampleNormal(uv + float2(-t.x, 0));
                    float3 nr = SampleNormal(uv + float2(t.x, 0));
                    
                    // Normal farklarından gradient çıkar
                    // pow(1 - dot, 2) kontrastı ile — EdgeDetection.shader ile tutarlı
                    float edgeR = pow(1.0 - saturate(dot(nc, nr)), 2);
                    float edgeL = pow(1.0 - saturate(dot(nc, nl)), 2);
                    float edgeT = pow(1.0 - saturate(dot(nc, nt)), 2);
                    float edgeB = pow(1.0 - saturate(dot(nc, nb)), 2);
                    
                    float ngx = (edgeR - edgeL) * _NormalSensitivity;
                    float ngy = (edgeT - edgeB) * _NormalSensitivity;
                    
                    gradient += float2(ngx, ngy) * _NormalWeight;
                }
                
                return gradient;
            }
            #endif

            fixed4 frag (v2f i) : SV_Target
            {
                float2 t = _MainTex_TexelSize.xy;
                
                float gx, gy, mag;
                
                #if _SOURCE_COMBINED
                    // İYİLEŞTİRME: Combined modda ağırlıklı gradient hesapla
                    float2 combinedGrad = GetCombinedGradient(i.uv, t);
                    gx = combinedGrad.x;
                    gy = combinedGrad.y;
                    mag = sqrt(gx*gx + gy*gy);
                #else
                    // Sobel gradient (Luminance, Depth veya Normal üzerinde)
                    float2 gradient = SobelGradient(i.uv, t);
                    gx = gradient.x;
                    gy = gradient.y;
                    mag = sqrt(gx*gx + gy*gy);
                #endif
                
                // Edge yoksa nötr değer döndür
                if (mag < _EdgeThreshold)
                    return fixed4(0.5, 0.5, 0, 1);
                
                // Gradyanı normalize et ve 0-1 aralığına çevir
                float nx = (gx / mag) * 0.5 + 0.5;  // -1..+1 → 0..1
                float ny = (gy / mag) * 0.5 + 0.5;  // -1..+1 → 0..1
                
                // Magnitude'u normalize et (0-1)
                float nm = saturate(mag);
                
                return fixed4(nx, ny, nm, 1);
            }
            ENDCG
        }
    }
    
    FallBack "Diffuse"
}
