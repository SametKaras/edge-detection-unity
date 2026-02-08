// ============================================================
// WorldPosBuffer.shader
// ============================================================
//
// AMAÇ:
//   Sahne (scene) içindeki her objenin her pikselinin 3D dünya
//   pozisyonunu (world position) bir texture'a yazmak.
//
// NASIL ÇALIŞIR:
//   1. Bu shader, Camera.RenderWithShader() ile çağrılır.
//      Bu sayede sahnedeki TÜM objeler bu shader ile renderlanır
//      (orijinal materyalleri geçici olarak devre dışı kalır).
//
//   2. Vertex shader'da: Unity'nin model→world dönüşüm matrisi
//      (unity_ObjectToWorld) kullanılarak her vertex'in 3D dünya
//      koordinatı hesaplanır.
//
//   3. Fragment shader'da: Interpolasyon sonrası her pikselin
//      dünya koordinatı (x, y, z) float4 olarak yazılır.
//      W kanalı = 1.0 → "bu piksel geçerli bir pozisyon içeriyor"
//      anlamına gelir (boş/siyah alanlar w=0 olur).
//
// ÇIKTI FORMAT:
//   RenderTextureFormat.ARGBFloat (128-bit) kullanılmalıdır.
//   Her kanal 32-bit float → x,y,z koordinatları hassas şekilde
//   saklanır. 8-bit RGBA texture'da koordinatlar kaybolur!
//
// KULLANIM:
//   WorldSpaceEdgeManager.cs → _posCam.RenderWithShader(worldPosShader, "RenderType")
//   Sonuç → _worldPosRT (128-bit float texture)
//   Bu texture daha sonra EdgeToPointCloud.compute tarafından okunur.
//
// PİPELİNE'DAKİ YERİ:
//   [Camera] → [WorldPosBuffer.shader] → [128-bit RT] → [Compute Shader] → [RANSAC]
//   ^^^^^^^^                               ^^^^^^^^^^^
//   RenderWithShader ile çalışır           ARGBFloat formatı
// ============================================================

Shader "Custom/WorldPosBuffer"
{
    SubShader
    {
        // RenderType = "Opaque" → Sadece opak (saydam olmayan) objeler renderlanır.
        // RenderWithShader çağrısındaki "RenderType" filtresi bunu kullanır.
        Tags { "RenderType"="Opaque" }
        
        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #include "UnityCG.cginc"
            
            // ==================== VERTEX → FRAGMENT VERİ YAPISI ====================
            struct v2f
            {
                float4 vertex : SV_POSITION;    // Ekran uzayı pozisyonu (GPU rasterizer için)
                float4 worldPos : TEXCOORD0;    // Dünya uzayı pozisyonu (fragment'e aktarılır)
            };
            
            // ==================== VERTEX SHADER ====================
            // Her vertex için çalışır.
            // İki dönüşüm yapar:
            //   1. Object → World (dünya pozisyonu hesaplama)
            //   2. Object → Clip  (ekranda nereye düşeceğini belirleme)
            v2f vert (appdata_base v)
            {
                v2f o;
                
                // Model uzayındaki vertex'i ekran uzayına dönüştür
                // (GPU'nun üçgeni hangi piksellere çizeceğini bilmesi için)
                o.vertex = UnityObjectToClipPos(v.vertex);
                
                // Model uzayındaki vertex'i dünya uzayına dönüştür
                // unity_ObjectToWorld = objenin Transform bileşenindeki konum/dönüş/ölçek matrisi
                // Bu sayede her pikselin gerçek 3D pozisyonunu öğrenebiliriz
                o.worldPos = mul(unity_ObjectToWorld, v.vertex);
                
                return o;
            }
            
            // ==================== FRAGMENT (PİKSEL) SHADER ====================
            // Her piksel için çalışır.
            // Vertex'ler arasındaki worldPos değerleri GPU tarafından otomatik
            // olarak interpole edilir (lineer interpolasyon).
            // Sonuç: Her pikselin tam dünya koordinatı.
            float4 frag (v2f i) : SV_Target
            {
                // xyz = dünya pozisyonu (metre cinsinden)
                // w = 1.0 → geçerli piksel işareti
                // (Boş alan / skybox pikselleri siyah kalır → w=0 → compute shader bunları atlar)
                return float4(i.worldPos.xyz, 1.0);
            }
            ENDCG
        }
    }
}
