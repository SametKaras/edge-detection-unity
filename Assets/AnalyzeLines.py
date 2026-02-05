import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import os

# HATA ÇÖZÜMÜ: Sklearn kütüphanesini sildik, çünkü zaten numpy ile hesaplıyoruz.
# from sklearn.neighbors import NearestNeighbors  <-- BU SATIR SİLİNDİ

def fit_line_segment(points):
    """Bir grup noktaya en uygun doğruyu (PCA ile) bulur."""
    # Verinin ortalamasını al (Centroid)
    mean = np.mean(points, axis=0)
    # Ortalamayı çıkar
    uu, dd, vv = np.linalg.svd(points - mean)
    # İlk eigenvector (doğrunun yönü)
    direction = vv[0]
    return mean, direction

def run_local_ransac(data, neighborhood_radius=0.5, min_samples=5, threshold=0.05):
    """
    Mentörün istediği 'küçük küçük line'lar için
    noktaları önce kümelere (cluster) ayırıp, her kümeye ayrı line fit ederiz.
    """
    points = data.copy()
    lines = []
    
    print("RANSAC İşlemi Başlıyor... (Bu biraz sürebilir)")
    
    # Sonsuz döngüye girmemek için max deneme sayısı
    max_iterations = 5000 
    iter_count = 0

    while len(points) > min_samples and iter_count < max_iterations:
        iter_count += 1
        
        # Rastgele bir nokta seç
        idx = np.random.randint(len(points))
        seed_point = points[idx]
        
        # Bu noktaya yakın olan komşuları bul (Local Search - Numpy ile)
        # Mentörün bahsettiği 5px mantığı burası
        dists = np.linalg.norm(points - seed_point, axis=1)
        
        # Yarıçap içindeki noktalar
        local_indices = np.where(dists < neighborhood_radius)[0]
        
        if len(local_indices) < min_samples:
            # Yeterli komşu yoksa bu noktayı silip devam et (Gürültü)
            points = np.delete(points, idx, axis=0)
            continue
            
        local_points = points[local_indices]
        
        # Bu lokal noktalarla bir doğru oluştur (Line Fitting)
        mean, direction = fit_line_segment(local_points)
        
        # Şimdi bu doğruya gerçekten uyanları (Inliers) seçelim
        vecs = local_points - mean
        cross_prods = np.cross(vecs, direction)
        dists_to_line = np.linalg.norm(cross_prods, axis=1)
        
        inliers_idx = np.where(dists_to_line < threshold)[0]
        
        if len(inliers_idx) >= min_samples:
            # Doğruyu oluşturduk!
            # Çizgiyi sonsuza uzatmak yerine, en uçtaki inlier noktaları alıyoruz.
            inlier_points = local_points[inliers_idx]
            
            # Doğru üzerindeki projeksiyonlarına göre sırala (Başlangıç ve Bitiş için)
            projections = np.dot(inlier_points - mean, direction)
            p_min = mean + direction * np.min(projections)
            p_max = mean + direction * np.max(projections)
            
            lines.append((p_min, p_max))
            
            # Kullanılan noktaları ana listeden sil (Böylece aynı yere tekrar çizgi çekmeyiz)
            # Global indeksleri bulup siliyoruz
            points = np.delete(points, local_indices[inliers_idx], axis=0)
        else:
            # Line oluşturamadık, bu seed noktasını atla
            points = np.delete(points, idx, axis=0)
            
    return lines

# --- ANA KOD ---

# Dosya yolunu otomatik bul
current_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(current_dir, 'PointCloudData.csv')

print(f"Dosya aranıyor: {csv_path}")

try:
    df = pd.read_csv(csv_path) 
    # Sadece sayısal verileri al (başlık satırını atla)
    data = df[['x', 'y', 'z']].values
    print(f"Toplam {len(data)} nokta yüklendi.")
    
    # 2. RANSAC Çalıştır
    # neighborhood_radius: Çizginin maksimum uzunluğu (Mentörün 5px dediği şey - burada Unity birimi cinsinden)
    # threshold: Çizgi kalınlığı toleransı
    found_lines = run_local_ransac(data, neighborhood_radius=2.0, min_samples=8, threshold=0.05)
    print(f"Toplam {len(found_lines)} adet kısa çizgi (segment) bulundu.")

    # 3. Görselleştirme
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Ham noktalar (Siyah, küçük)
    # Numpy warning hatasını engellemek için downsample yapabiliriz gerekirse
    step = 1 if len(data) < 10000 else 5 # Çok nokta varsa seyrelt
    ax.scatter(data[::step,0], data[::step,1], data[::step,2], c='k', s=0.5, alpha=0.3, label='Point Cloud')

    # Bulunan çizgiler (Kırmızı, kalın)
    for start, end in found_lines:
        ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], c='r', linewidth=2)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f"Mentör Analizi: {len(found_lines)} Segment Bulundu")
    plt.show()

except FileNotFoundError:
    print(f"HATA: '{csv_path}' dosyası bulunamadı!")
    print("Unity'de oyunu başlatıp 'E' tuşuna bastığından emin misin?")
except Exception as e:
    print(f"Beklenmedik bir hata oluştu: {e}")