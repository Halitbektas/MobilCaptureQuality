import cv2
import numpy as np
from typing import Dict, Union, Tuple
import mediapipe as mp
def check_face_size(image: np.ndarray,
                    bbox: Tuple[int, int, int, int],
                    min_area_ratio: float = 0.1,
                    max_area_ratio: float = 0.55,
                    edge_margin: int = 15) -> Dict[str, Union[bool, float, str]]:

    image_height, image_width = image.shape[:2]
    x, y, w, h = bbox

    is_cropped_left = x <= edge_margin
    is_cropped_top = y <= edge_margin
    is_cropped_right = (x + w) >= (image_width - edge_margin)
    is_cropped_bottom = (y + h) >= (image_height - edge_margin)

    face_area = w * h
    image_area = image_width * image_height
    face_ratio = face_area / image_area

    if is_cropped_left or is_cropped_top or is_cropped_right or is_cropped_bottom:
        return {
            "passed": False,
            "face_ratio": round(face_ratio, 3),
            "reason": "cropped_out"
        }

    if face_ratio < min_area_ratio:
        return {
            "passed": False,
            "face_ratio": round(face_ratio, 3),
            "reason": "face_too_small"
        }

    if face_ratio > max_area_ratio:
        return {
            "passed": False,
            "face_ratio": round(face_ratio, 3),
            "reason": "face_too_close"
        }

    return {
        "passed": True,
        "face_ratio": round(face_ratio, 3),
        "reason": None
    }


# --- LOKAL TEST İÇİN ÖRNEK KULLANIM ---
if __name__ == "__main__":
    # Test için boş bir siyah görüntü oluşturalım (1920x1080)
    test_image = np.zeros((1080, 1920, 3), dtype=np.uint8)

    # Senaryo 1: İdeal Yüz (Ekranda ortalanmış, normal boyut)
    ideal_bbox = (800, 300, 320, 480)
    print("İdeal Yüz:", check_face_size(test_image, ideal_bbox))

    # Senaryo 2: Çok Küçük Yüz (Uzakta kalmış)
    small_bbox = (900, 500, 100, 150)
    print("Çok Küçük Yüz:", check_face_size(test_image, small_bbox))

    # Senaryo 3: Kenardan Kesilmiş Yüz (Sağ kenara dayanmış)
    cropped_bbox = (1700, 300, 300, 400)
    print("Kesilmiş Yüz:", check_face_size(test_image, cropped_bbox))