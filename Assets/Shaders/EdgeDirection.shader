// ============================================================
// EdgeDirection.shader
// Her pikselin gradyan yönünü RGB olarak çıktı verir
// R = gx (normalize), G = gy (normalize), B = magnitude
// ============================================================

Shader "Custom/EdgeDirection"
{
    Properties
    {
        _MainTex ("Texture", 2D) = "white" {}
        _EdgeThreshold ("Edge Threshold", Range(0.001, 1.0)) = 0.1
    }
    
    SubShader
    {
        Tags { "RenderType"="Opaque" }
        
        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            
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
            
            // Sobel gradyan hesapla
            float2 SobelGradient(float2 uv, float2 t)
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
                
                return float2(gx, gy);
            }

            fixed4 frag (v2f i) : SV_Target
            {
                float2 t = _MainTex_TexelSize.xy;
                float2 gradient = SobelGradient(i.uv, t);
                
                float gx = gradient.x;
                float gy = gradient.y;
                float mag = sqrt(gx*gx + gy*gy);
                
                // Edge yoksa siyah döndür
                if (mag < _EdgeThreshold)
                    return fixed4(0.5, 0.5, 0, 1);  // Nötr değer (0 gradyan)
                
                // Gradyanı normalize et ve 0-1 aralığına çevir
                // gx, gy: -1 ile +1 arası → 0 ile 1 arası
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
