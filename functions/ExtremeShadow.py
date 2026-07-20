import cv2
import numpy as np
from typing import Dict, Union, Tuple
from functions.FaceControl import check_face_in_image


def check_extreme_shadow(image: np.ndarray,
                         bbox: Tuple[int, int, int, int],
                         shadow_thresh: float = 40.0) -> Dict[str, Union[bool, float, str]]:
    """
    Yüzün sağ ve sol yarılarının ortalama parlaklık değerlerini karşılaştırarak
    fotoğrafta aşırı, sert bir gölge olup olmadığını kontrol eder.

    Args:
        image (np.ndarray): Orijinal BGR veya Gri tonlamalı görüntü.
        bbox (tuple): (x, y, w, h) formatında yüzün sınırlayıcı kutusu.
        shadow_thresh (float): Sol ve sağ yarılar arasındaki kabul edilebilir
                               maksimum parlaklık farkı (0-255 arası).

    Returns:
        dict: Kontrol durumu, fark skoru, sol/sağ parlaklıkları ve hata sebebi.
    """
    x, y, w, h = bbox
    img_h, img_w = image.shape[:2]

    # 1. Sınır Güvenliği (Kutunun resim dışına taşmamasını garantiye al)
    x = max(0, x)
    y = max(0, y)
    w = min(w, img_w - x)
    h = min(h, img_h - y)

    if w <= 0 or h <= 0:
        return {"passed": False, "reason": "invalid_bbox", "shadow_diff": 0.0}

    # 2. Yüzü Kırp (Crop)
    face_crop = image[y:y + h, x:x + w]

    # 3. Parlaklık analizi için görüntüyü Gri Tonlamaya (Grayscale) çevir
    if len(face_crop.shape) == 3:
        gray_face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    else:
        gray_face = face_crop

    # 4. Yüzü Dikey Olarak İkiye Böl
    mid_x = w // 2
    left_half = gray_face[:, :mid_x]
    right_half = gray_face[:, mid_x:]

    # 5. Her iki yarının ortalama piksel parlaklığını hesapla (0-255)
    mean_left = np.mean(left_half)
    mean_right = np.mean(right_half)

    # 6. Mutlak parlaklık farkını hesapla
    diff = abs(mean_left - mean_right)

    # Eğer fark belirlenen eşikten büyükse, bir tarafa sert gölge vurmuştur
    is_extreme_shadow = diff > shadow_thresh

    return {
        "passed": not bool(is_extreme_shadow),
        "shadow_diff": float(diff),
        "left_brightness": float(mean_left),
        "right_brightness": float(mean_right),
        "reason": "extreme_shadow" if is_extreme_shadow else None
    }


# --- LOKAL TEST İÇİN ÖRNEK KULLANIM ---
if __name__ == "__main__":
    test_image = cv2.imread("/Users/halitbektas/Desktop/MobileCaptureQuality/test_images/shadow.jpeg")
    bbox = check_face_in_image(test_image)["bbox"]

    result = check_extreme_shadow(test_image, bbox, shadow_thresh=40.0)
    print(f"Aşırı Gölge Testi Sonucu: {result}")