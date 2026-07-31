import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh

# YENİ: Kaşları dışarıda bırakan, sadece üst alın bölgesini (üst saç çizgisi ve altı) hedefleyen noktalar
FOREHEAD_INDICES = [10, 151, 109, 108, 69, 67, 338, 337, 299, 297]

LEFT_CHEEK_INDICES = [118, 119, 100, 126, 49, 50, 137, 177, 215, 138, 135]

RIGHT_CHEEK_INDICES = [347, 348, 329, 355, 279, 280, 366, 401, 435, 367, 364]

def get_bounding_box(image, landmarks, indices):
    h, w, _ = image.shape
    x_min, y_min = w, h
    x_max, y_max = 0, 0
    
    for idx in indices:
        pt = landmarks.landmark[idx]
        x, y = int(pt.x * w), int(pt.y * h)
        if x < x_min: x_min = x
        if x > x_max: x_max = x
        if y < y_min: y_min = y
        if y > y_max: y_max = y
        
    x_min, y_min = max(0, x_min), max(0, y_min)
    x_max, y_max = min(w, x_max), min(h, y_max)
    
    if x_max > x_min and y_max > y_min:
        return image[y_min:y_max, x_min:x_max], (x_min, y_min, x_max, y_max)
    return None, None

def analyze_hair(roi_image):
    """Bölgedeki saç yoğunluğunu analiz eder ve kenar haritasını döndürür."""
    gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
    
    # Bulanıklaştırma
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # Canny eşikleri 
    edges = cv2.Canny(blurred, 90, 110)
    
    total_pixels = edges.shape[0] * edges.shape[1]
    hair_pixels = cv2.countNonZero(edges)
    
    hair_ratio = (hair_pixels / total_pixels) * 100 if total_pixels > 0 else 0
    
    return edges, hair_ratio
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break
            
        image = cv2.flip(image, 1)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_image)
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                
                regions = {
                    "Alin": (FOREHEAD_INDICES, (255, 0, 0)),
                    "Sol Yanak": (LEFT_CHEEK_INDICES, (0, 255, 0)),
                    "Sag Yanak": (RIGHT_CHEEK_INDICES, (0, 0, 255))
                }
                
                for region_name, (indices, color) in regions.items():
                    roi_data, bbox = get_bounding_box(image, face_landmarks, indices)
                    
                    if roi_data is not None and roi_data.size > 0:
                        # Saç analizi fonksiyonumuzu çağır
                        edges, hair_ratio = analyze_hair(roi_data)
                        
                        # Kenar haritasını ayrı pencerede göster (Görselleştirme)
                        cv2.imshow(f"{region_name} - Kenar Haritasi", cv2.resize(edges, (150, 150)))
                        
                        x_min, y_min, x_max, y_max = bbox
                        
                        # Eğer beyaz çizgi oranı %10'dan fazlaysa SAÇ VAR diyoruz
                        if hair_ratio > 1.5:
                            status_text = f"SAC VAR ({hair_ratio:.1f}%)"
                            box_color = (0, 0, 255) # Kırmızı (Uyarı)
                        else:
                            status_text = f"TEMIZ ({hair_ratio:.1f}%)"
                            box_color = (0, 255, 0) # Yeşil (Sorun yok)
                            
                        # Ana görüntüye kutuyu ve yazı durumunu çiz
                        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), box_color, 2)
                        cv2.putText(image, status_text, (x_min, y_min - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

        cv2.imshow('Ana Kamera', image)
        
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()