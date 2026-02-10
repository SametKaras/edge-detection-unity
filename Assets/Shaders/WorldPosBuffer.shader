Shader "Custom/WorldPosBuffer"
{
    SubShader
    {
        // Sadece opak objeleri işle.
        Tags { "RenderType"="Opaque" }
        
        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #include "UnityCG.cginc"
            
            // Vertex Shader'dan Fragment Shader'a taşınacak veriler
            struct v2f
            {
                float4 vertex : SV_POSITION;    // Ekranda çizilecek konum 2D
                float4 worldPos : TEXCOORD0;    // Hesaplanan 3D dünya konumu
            };
            
            // ==================== VERTEX SHADER ====================
            // Köşe noktalarının (vertex) konumlarını hesaplar.
            v2f vert (appdata_base v)
            {
                v2f o;
                
                // 1. Çizim için: Modeli ekran koordinatlarına çevir.
                o.vertex = UnityObjectToClipPos(v.vertex);
                
                // 2. Veri için: Modeli 3D dünya koordinatlarına çevir.
                o.worldPos = mul(unity_ObjectToWorld, v.vertex);
                
                return o;
            }
            
            // ==================== FRAGMENT SHADER ====================
            // Her piksel için çalışır ve rengi belirler.
            // Rengi "renk" olarak değil, "koordinat verisi" olarak kullanıyoruz.
            float4 frag (v2f i) : SV_Target
            {
                // RGB kanallarına X,Y,Z koordinatlarını yaz.
                // Alpha (W) kanalına 1.0 yaz (pikselin dolu olduğunu belirtmek için).
                return float4(i.worldPos.xyz, 1.0);
            }
            ENDCG
        }
    }
}