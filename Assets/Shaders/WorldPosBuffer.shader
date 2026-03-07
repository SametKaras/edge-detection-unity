Shader "Custom/WorldPosBuffer"
{
    SubShader
    {
        Tags { "RenderType"="Opaque" }
        ZTest Always ZWrite Off Cull Off

        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #include "UnityCG.cginc"

            struct appdata { float4 vertex : POSITION; float2 uv : TEXCOORD0; };
            struct v2f     { float4 vertex : SV_POSITION; float2 uv : TEXCOORD0; };

            sampler2D _CameraDepthTexture;
            float4    _CameraDepthTexture_TexelSize;
            float4x4  unity_MatrixInvVP;

            v2f vert(appdata v) { v2f o; o.vertex = UnityObjectToClipPos(v.vertex); o.uv = v.uv; return o; }

            float4 frag(v2f i) : SV_Target
            {
                float rawDepth = SAMPLE_DEPTH_TEXTURE(_CameraDepthTexture, i.uv);

                float linear01 = Linear01Depth(rawDepth);
                if (linear01 >= 0.9999) return float4(0, 0, 0, 0);

                // renderIntoTexture=true → C#'ta invVP Y-flip içeriyor
                // D3D: UV(0,0)=sol-üst, NDC Y=+1 üst → uv.y=0 → ndcY=+1 → negate şart
                float ndcX =  i.uv.x * 2.0 - 1.0;
                float ndcY = -(i.uv.y * 2.0 - 1.0);

                #if defined(UNITY_REVERSED_Z)
                    float ndcZ = rawDepth;
                #else
                    float ndcZ = rawDepth * 2.0 - 1.0;
                #endif

                float4 clipPos = float4(ndcX, ndcY, ndcZ, 1.0);
                float4 wp = mul(unity_MatrixInvVP, clipPos);
                return float4(wp.xyz / wp.w, 1.0);
            }
            ENDCG
        }
    }
}
