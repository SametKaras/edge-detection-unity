// ============================================================
// EdgeDetection.shader
// Image-space edge detection: Sobel / Roberts / Prewitt
// Kaynak: Luminance / Depth / Normal / Combined
// İki output modu:
//   _OutputMagnitude=1 → raw magnitude (compute shader için)
//   _OutputMagnitude=0 → binary threshold (ekran için)
// ============================================================
Shader "Custom/EdgeDetection"
{
    Properties
    {
        _MainTex ("Texture", 2D) = "white" {}
        _EdgeThreshold ("Edge Threshold", Range(0.001, 1.0)) = 0.1

        _EdgeColor ("Edge Color", Color) = (1, 1, 1, 1)
        _BackgroundColor ("Background Color", Color) = (0, 0, 0, 1)

        [KeywordEnum(Sobel, Roberts, Prewitt)]
        _Method ("Detection Method", Float) = 0

        [KeywordEnum(Luminance, Depth, Normal, Combined)]
        _Source ("Edge Source", Float) = 0

        // Derinlik
        _DepthSensitivity ("Depth Sensitivity", Range(0.1, 100)) = 10.0
        _MaxDepth ("Max Depth", Float) = 50.0

        // Normal
        _NormalSensitivity ("Normal Sensitivity", Range(0.1, 10)) = 1.0

        // Combined ağırlıklar
        _DepthWeight  ("Depth Weight",  Range(0, 1)) = 0.5
        _NormalWeight ("Normal Weight", Range(0, 1)) = 0.5
        _ColorWeight  ("Color Weight",  Range(0, 1)) = 0.3

        // Crease filtresi: cos(açı) eşiği
        // dot > _MinCreaseDot → smooth yüzey → edge sayma
        _MinCreaseDot ("Min Crease Dot", Range(0, 1)) = 0.9

        [Toggle] _InvertOutput ("Invert Output", Float) = 0

        // Magnitude output (algoritma için)
        [Toggle] _OutputMagnitude ("Output Raw Magnitude", Float) = 0
        _MagnitudeScale ("Magnitude Scale", Range(0.5, 4.0)) = 1.0
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
                float2 uv     : TEXCOORD0;
            };

            struct v2f
            {
                float2 uv     : TEXCOORD0;
                float4 vertex : SV_POSITION;
            };

            sampler2D _MainTex;
            float4    _MainTex_TexelSize;
            sampler2D _CameraDepthTexture;
            sampler2D _CameraDepthNormalsTexture;

            float  _EdgeThreshold;
            float4 _EdgeColor;
            float4 _BackgroundColor;
            float  _DepthSensitivity;
            float  _MaxDepth;
            float  _NormalSensitivity;
            float  _DepthWeight;
            float  _NormalWeight;
            float  _ColorWeight;
            float  _MinCreaseDot;
            float  _InvertOutput;
            float  _OutputMagnitude;
            float  _MagnitudeScale;

            // ===================== VERTEX =====================
            v2f vert(appdata v)
            {
                v2f o;
                o.vertex = UnityObjectToClipPos(v.vertex);
                o.uv     = v.uv;
                return o;
            }

            // ===================== HELPERS =====================
            float Luma(float3 c)
            {
                return dot(c, float3(0.299, 0.587, 0.114));
            }

            float SampleDepth(float2 uv)
            {
                float d = SAMPLE_DEPTH_TEXTURE(_CameraDepthTexture, uv);
                return LinearEyeDepth(d);
            }

            float3 SampleNormal(float2 uv)
            {
                float4 dn = tex2D(_CameraDepthNormalsTexture, uv);
                float3 n; float d;
                DecodeDepthNormal(dn, d, n);
                return n;
            }

            // Kaynak değeri: seçilen moda göre skaler döndürür
            float GetSourceValue(float2 uv)
            {
                #if _SOURCE_LUMINANCE
                    return Luma(tex2D(_MainTex, uv).rgb);
                #elif _SOURCE_DEPTH
                    return saturate(SampleDepth(uv) / _MaxDepth) * _DepthSensitivity;
                #elif _SOURCE_NORMAL
                    return length(SampleNormal(uv)) * _NormalSensitivity;
                #else
                    return Luma(tex2D(_MainTex, uv).rgb);
                #endif
            }

            // ===================== OPERATÖRLER =====================
            // Sobel 3×3
            float2 Sobel(float2 uv, float2 t)
            {
                float tl = GetSourceValue(uv + float2(-t.x,  t.y));
                float tm = GetSourceValue(uv + float2(   0,  t.y));
                float tr = GetSourceValue(uv + float2( t.x,  t.y));
                float ml = GetSourceValue(uv + float2(-t.x,    0));
                float mr = GetSourceValue(uv + float2( t.x,    0));
                float bl = GetSourceValue(uv + float2(-t.x, -t.y));
                float bm = GetSourceValue(uv + float2(   0, -t.y));
                float br = GetSourceValue(uv + float2( t.x, -t.y));

                float gx = -tl - 2.0*ml - bl + tr + 2.0*mr + br;
                float gy = -tl - 2.0*tm - tr + bl + 2.0*bm + br;
                return float2(gx, gy);
            }

            // Roberts Cross 2×2
            float2 Roberts(float2 uv, float2 t)
            {
                float c  = GetSourceValue(uv);
                float r  = GetSourceValue(uv + float2(t.x,    0));
                float b  = GetSourceValue(uv + float2(   0, -t.y));
                float br = GetSourceValue(uv + float2(t.x, -t.y));
                return float2(c - br, r - b);
            }

            // Prewitt 3×3
            float2 Prewitt(float2 uv, float2 t)
            {
                float tl = GetSourceValue(uv + float2(-t.x,  t.y));
                float tm = GetSourceValue(uv + float2(   0,  t.y));
                float tr = GetSourceValue(uv + float2( t.x,  t.y));
                float ml = GetSourceValue(uv + float2(-t.x,    0));
                float mr = GetSourceValue(uv + float2( t.x,    0));
                float bl = GetSourceValue(uv + float2(-t.x, -t.y));
                float bm = GetSourceValue(uv + float2(   0, -t.y));
                float br = GetSourceValue(uv + float2( t.x, -t.y));

                float gx = -tl - ml - bl + tr + mr + br;
                float gy = -tl - tm - tr + bl + bm + br;
                return float2(gx, gy);
            }

            // ===================== COMBINED EDGE =====================
            // Depth + Normal + Color birleşik kenar hesabı
            float GetCombinedEdge(float2 uv, float2 t)
            {
                float edge = 0;

                // Renk kenarları (Sobel on luminance)
                if (_ColorWeight > 0.01)
                {
                    float tl = Luma(tex2D(_MainTex, uv + float2(-t.x,  t.y)).rgb);
                    float tm = Luma(tex2D(_MainTex, uv + float2(   0,  t.y)).rgb);
                    float tr = Luma(tex2D(_MainTex, uv + float2( t.x,  t.y)).rgb);
                    float ml = Luma(tex2D(_MainTex, uv + float2(-t.x,    0)).rgb);
                    float mr = Luma(tex2D(_MainTex, uv + float2( t.x,    0)).rgb);
                    float bl = Luma(tex2D(_MainTex, uv + float2(-t.x, -t.y)).rgb);
                    float bm = Luma(tex2D(_MainTex, uv + float2(   0, -t.y)).rgb);
                    float br = Luma(tex2D(_MainTex, uv + float2( t.x, -t.y)).rgb);

                    float gx = -tl - 2.0*ml - bl + tr + 2.0*mr + br;
                    float gy = -tl - 2.0*tm - tr + bl + 2.0*bm + br;
                    edge += sqrt(gx*gx + gy*gy) * _ColorWeight;
                }

                // Derinlik kenarları
                if (_DepthWeight > 0.01)
                {
                    float dc = SampleDepth(uv);
                    float dt = SampleDepth(uv + float2(0,  t.y));
                    float db = SampleDepth(uv + float2(0, -t.y));
                    float dl = SampleDepth(uv + float2(-t.x, 0));
                    float dr = SampleDepth(uv + float2( t.x, 0));
                    edge += (abs(dc-dt) + abs(dc-db) + abs(dc-dl) + abs(dc-dr))
                            * _DepthSensitivity * _DepthWeight;
                }

                // Normal kenarları (crease filtreli)
                if (_NormalWeight > 0.01)
                {
                    float3 nc = SampleNormal(uv);
                    float3 nt = SampleNormal(uv + float2(0,  t.y));
                    float3 nb = SampleNormal(uv + float2(0, -t.y));
                    float3 nl = SampleNormal(uv + float2(-t.x, 0));
                    float3 nr = SampleNormal(uv + float2( t.x, 0));

                    float dotT = saturate(dot(nc, nt));
                    float dotB = saturate(dot(nc, nb));
                    float dotL = saturate(dot(nc, nl));
                    float dotR = saturate(dot(nc, nr));

                    float ne = 0;
                    // Crease filtresi: dot < _MinCreaseDot → açı büyük → gerçek kenar
                    if (dotT < _MinCreaseDot) ne += pow(1.0 - dotT, 2);
                    if (dotB < _MinCreaseDot) ne += pow(1.0 - dotB, 2);
                    if (dotL < _MinCreaseDot) ne += pow(1.0 - dotL, 2);
                    if (dotR < _MinCreaseDot) ne += pow(1.0 - dotR, 2);

                    edge += ne * _NormalSensitivity * _NormalWeight;
                }

                return edge;
            }

            // ===================== FRAGMENT =====================
            fixed4 frag(v2f i) : SV_Target
            {
                float2 t = _MainTex_TexelSize.xy;
                float mag = 0;

                #if _SOURCE_COMBINED
                    mag = GetCombinedEdge(i.uv, t);
                #else
                    float2 g = float2(0, 0);

                    #if _METHOD_SOBEL
                        g = Sobel(i.uv, t);
                    #elif _METHOD_ROBERTS
                        g = Roberts(i.uv, t);
                    #elif _METHOD_PREWITT
                        g = Prewitt(i.uv, t);
                    #endif

                    mag = sqrt(g.x*g.x + g.y*g.y);
                #endif

                // Raw magnitude output (algoritma/compute shader için)
                if (_OutputMagnitude > 0.5)
                {
                    float m = saturate(mag * _MagnitudeScale);
                    return fixed4(m, m, m, 1);
                }

                // Binary threshold (ekran için)
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