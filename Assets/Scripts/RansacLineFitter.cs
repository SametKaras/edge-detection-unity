using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

public struct Line3D
{
    public float3 Start;
    public float3 End;
    public int InlierCount;
    public bool IsValid;
}

[BurstCompile]
public struct MultiLineRansacJob : IJob
{
    [ReadOnly] public NativeArray<float3> InputPoints;
    [WriteOnly] public NativeArray<Line3D> ResultLines;
    
    public int MaxLinesToFind;
    public int MaxIterations;
    public float Threshold;
    public int MinInliers;
    
    // YENİ: Sonsuzluğu ve Objeler arası atlamayı engelleyen parametre
    public float MaxSegLength; 
    
    public uint RandomSeed;

    public void Execute()
    {
        int pointCount = InputPoints.Length;
        if (pointCount < 2) return;

        NativeArray<bool> usedPoints = new NativeArray<bool>(pointCount, Allocator.Temp);
        Unity.Mathematics.Random rng = new Unity.Mathematics.Random(RandomSeed);

        for (int lineIndex = 0; lineIndex < MaxLinesToFind; lineIndex++)
        {
            Line3D bestLine = new Line3D { IsValid = false };
            int bestInliers = -1;
            
            // --- Model Arama ---
            for (int i = 0; i < MaxIterations; i++)
            {
                int idx1 = rng.NextInt(0, pointCount);
                int idx2 = rng.NextInt(0, pointCount);

                if (usedPoints[idx1] || usedPoints[idx2] || idx1 == idx2) continue;

                float3 p1 = InputPoints[idx1];
                float3 p2 = InputPoints[idx2];

                // KORUMA 1: İki nokta birbirinden çok uzaksa (farklı objeler), bağlama.
                if (math.distance(p1, p2) > MaxSegLength) continue;

                float3 lineDir = math.normalize(p2 - p1);
                int currentInliers = 0;

                for (int k = 0; k < pointCount; k++)
                {
                    if (usedPoints[k]) continue;
                    
                    // KORUMA 2: Uzaktaki noktaları bu çizgiye dahil etme
                    if (math.distance(InputPoints[k], p1) > MaxSegLength) continue;

                    float3 vecToPoint = InputPoints[k] - p1;
                    float crossMag = math.length(math.cross(vecToPoint, lineDir));

                    if (crossMag < Threshold) currentInliers++;
                }

                if (currentInliers > bestInliers)
                {
                    bestInliers = currentInliers;
                    bestLine.Start = p1; 
                    bestLine.End = p2;   
                    bestLine.InlierCount = bestInliers;
                    bestLine.IsValid = true;
                }
            }

            // --- Segment Kırpma (Clamping) ---
            if (bestLine.IsValid && bestLine.InlierCount > MinInliers)
            {
                float3 lineDir = math.normalize(bestLine.End - bestLine.Start);
                float3 origin = bestLine.Start;

                float minT = float.MaxValue;
                float maxT = float.MinValue;
                bool foundAny = false;

                for (int k = 0; k < pointCount; k++)
                {
                    if (usedPoints[k]) continue;

                    if (math.distance(InputPoints[k], origin) > MaxSegLength) continue;

                    float3 point = InputPoints[k];
                    float3 vec = point - origin;
                    float dist = math.length(math.cross(vec, lineDir));

                    if (dist < Threshold)
                    {
                        usedPoints[k] = true;
                        float t = math.dot(point - origin, lineDir);
                        if (t < minT) minT = t;
                        if (t > maxT) maxT = t;
                        foundAny = true;
                    }
                }

                if (foundAny)
                {
                    // Sonsuz doğru yerine sadece noktaların olduğu aralığı al
                    bestLine.Start = origin + lineDir * minT;
                    bestLine.End = origin + lineDir * maxT;
                    ResultLines[lineIndex] = bestLine;
                }
            }
            else
            {
                // Artık iyi çizgi bulunamıyor, döngüyü bitir
                if (lineIndex > 5 && bestInliers < MinInliers / 2) break; 
            }
        }
        usedPoints.Dispose();
    }
}