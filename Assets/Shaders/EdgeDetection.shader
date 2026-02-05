Shader "Custom/EdgeDetection"
{
    Properties
    {
        // Ana texture - kameranın renderladığı görüntü buraya gelir
        _MainTex ("Texture", 2D) = "white" {}
        
        // Kenar eşiği - bu değerin altındaki gradyanlar kenar sayılmaz
        _EdgeThreshold ("Edge Threshold", Range(0.001, 1.0)) = 0.1
        
        // Renk ayarları
        _EdgeColor ("Edge Color", Color) = (1, 1, 1, 1)
        _BackgroundColor ("Background Color", Color) = (0, 0, 0, 1)
        
        [KeywordEnum(Sobel, Roberts, Prewitt)] 
        _Method ("Detection Method", Float) = 0
        
        [KeywordEnum(Luminance, Depth, Normal, Combined)]
        _Source ("Edge Source", Float) = 0
        
        // Derinlik ayarları
        _DepthSensitivity ("Depth Sensitivity", Range(0.1, 100)) = 10.0
        _MaxDepth ("Max Depth", Float) = 50.0
        
        // Normal ayarları
        _NormalSensitivity ("Normal Sensitivity", Range(0.1, 10)) = 1.0
        
        // Combined modda her kaynağın ağırlığı
        _DepthWeight ("Depth Weight", Range(0, 1)) = 0.5
        _NormalWeight ("Normal Weight", Range(0, 1)) = 0.5
        _ColorWeight ("Color Weight", Range(0, 1)) = 0.3
        
        [Toggle] _InvertOutput ("Invert Output", Float) = 0
        
        // YENİ: Thinning için magnitude output modu
        // 1 = magnitude değerini doğrudan çıkar (thinning pass için)
        // 0 = binary threshold uygula (final output için)
        [Toggle] _OutputMagnitude ("Output Raw Magnitude", Float) = 0
    }
    
    SubShader
    {
        Tags { "RenderType"="Opaque" }
        
        Pass
        {
            CGPROGRAM
            
            #pragma vertex vert
            #pragma fragment frag
            
            #pragma multi_compile _METHOD_SOBEL _METHOD_ROBERTS _METHOD_PREWITT
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
            
            sampler2D _CameraDepthTexture;        
            sampler2D _CameraDepthNormalsTexture; 
            
            float _EdgeThreshold;
            float4 _EdgeColor;
            float4 _BackgroundColor;
            float _DepthSensitivity;
            float _MaxDepth;
            float _NormalSensitivity;
            float _DepthWeight;
            float _NormalWeight;
            float _ColorWeight;
            float _InvertOutput;
            float _OutputMagnitude;  // YENİ
            
            v2f vert (appdata v)
            {
                v2f o;
                o.vertex = UnityObjectToClipPos(v.vertex);
                o.uv = v.uv;
                return o;
            }
            
            /// RGB rengi parlaklık değerine çevirir
            float Luminance(float3 color)
            {
                return dot(color, float3(0.299, 0.587, 0.114));
            }
            
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
            
            /// Seçilen kaynağa göre piksel değeri döndürür
            float GetSourceValue(float2 uv)
            {
                #if _SOURCE_LUMINANCE
                    return Luminance(tex2D(_MainTex, uv).rgb);
                    
                #elif _SOURCE_DEPTH
                    return saturate(SampleDepth(uv) / _MaxDepth) * _DepthSensitivity;
                    
                #elif _SOURCE_NORMAL
                    return length(SampleNormal(uv)) * _NormalSensitivity;
                    
                #else
                    return Luminance(tex2D(_MainTex, uv).rgb);
                #endif
            }
            
            // ========================================================
            // EDGE DETECTION ALGORİTMALARI
            // ========================================================
            
            /// SOBEL Operatörü (3x3)
            float2 Sobel(float2 uv, float2 t)
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
            
            /// ROBERTS Cross Operatörü (2x2)
            float2 Roberts(float2 uv, float2 t)
            {
                float c  = GetSourceValue(uv);
                float r  = GetSourceValue(uv + float2(t.x, 0));
                float b  = GetSourceValue(uv + float2(0, -t.y));
                float br = GetSourceValue(uv + float2(t.x, -t.y));
                
                return float2(c - br, r - b);
            }
            
            /// PREWITT Operatörü (3x3)
            float2 Prewitt(float2 uv, float2 t)
            {
                float tl = GetSourceValue(uv + float2(-t.x, t.y));
                float tm = GetSourceValue(uv + float2(0, t.y));
                float tr = GetSourceValue(uv + float2(t.x, t.y));
                float ml = GetSourceValue(uv + float2(-t.x, 0));
                float mr = GetSourceValue(uv + float2(t.x, 0));
                float bl = GetSourceValue(uv + float2(-t.x, -t.y));
                float bm = GetSourceValue(uv + float2(0, -t.y));
                float br = GetSourceValue(uv + float2(t.x, -t.y));
                
                float gx = -tl - ml - bl + tr + mr + br;
                float gy = -tl - tm - tr + bl + bm + br;
                
                return float2(gx, gy);
            }

            float GetCombinedEdge(float2 uv, float2 t)
            {
                float edge = 0;
                
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
                    
                    float gx = -tl - 2.0*ml - bl + tr + 2.0*mr + br;
                    float gy = -tl - 2.0*tm - tr + bl + 2.0*bm + br;
                    
                    edge += sqrt(gx*gx + gy*gy) * _ColorWeight;
                }
                
                // DERİNLİK KENARLARI
                if (_DepthWeight > 0.01)
                {
                    float dc = SampleDepth(uv);
                    float dt = SampleDepth(uv + float2(0, t.y));
                    float db = SampleDepth(uv + float2(0, -t.y));
                    float dl = SampleDepth(uv + float2(-t.x, 0));
                    float dr = SampleDepth(uv + float2(t.x, 0));
                    
                    float de = abs(dc-dt) + abs(dc-db) + abs(dc-dl) + abs(dc-dr);
                    edge += de * _DepthSensitivity * _DepthWeight;
                }
                
                // EdgeDetection.shader - Satır ~46 civarı
                // NORMAL KENARLARI
                if (_NormalWeight > 0.01)
                {
                    float3 nc = SampleNormal(uv);
                    float3 nt = SampleNormal(uv + float2(0, t.y));
                    float3 nb = SampleNormal(uv + float2(0, -t.y));
                    float3 nl = SampleNormal(uv + float2(-t.x, 0));
                    float3 nr = SampleNormal(uv + float2(t.x, 0));
                    
                    float ne = 0;
                    // YENİ MATEMATİK: 'pow' ekleyerek kontrastı artırıyoruz.
                    // Zayıf kenarlar (yüzey eğimi) sıfıra yaklaşır, sert kenarlar 1 kalır.
                    ne += pow(1.0 - saturate(dot(nc, nt)), 2); 
                    ne += pow(1.0 - saturate(dot(nc, nb)), 2);
                    ne += pow(1.0 - saturate(dot(nc, nl)), 2);
                    ne += pow(1.0 - saturate(dot(nc, nr)), 2);
                    
                    edge += ne * _NormalSensitivity * _NormalWeight;
                }
                
                return edge;
            }
            
            fixed4 frag (v2f i) : SV_Target
            {
                float2 t = _MainTex_TexelSize.xy;
                
                float mag = 0;
                
                #if _SOURCE_COMBINED
                    mag = GetCombinedEdge(i.uv, t);
                #else
                    #if _METHOD_SOBEL
                        float2 g = Sobel(i.uv, t);
                        mag = sqrt(g.x*g.x + g.y*g.y);
                        
                    #elif _METHOD_ROBERTS
                        float2 g = Roberts(i.uv, t);
                        mag = sqrt(g.x*g.x + g.y*g.y);
                        
                    #elif _METHOD_PREWITT
                        float2 g = Prewitt(i.uv, t);
                        mag = sqrt(g.x*g.x + g.y*g.y);
                    #endif
                #endif
                
                // ========================================
                // YENİ: Magnitude output modu
                // Thinning için continuous değer lazım
                // ========================================
                if (_OutputMagnitude > 0.5)
                {
                    // Thinning pass için: raw magnitude değerini çıkar
                    // Normalize et ki 0-1 aralığında kalsın
                    float normalizedMag = saturate(mag);
                    return fixed4(normalizedMag, normalizedMag, normalizedMag, 1);
                }
                
                // Normal mod: binary threshold
                float edge = step(_EdgeThreshold, mag);
                
                if (_InvertOutput > 0.5)
                    edge = 1.0 - edge;
                
                return lerp(_BackgroundColor, _EdgeColor, edge);
            }
            ENDCG
        }
    }
    
    FallBack "Diffuse"
}
