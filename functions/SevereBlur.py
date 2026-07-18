import cv2
import numpy as np
from typing import Dict, Union

def check_severe_blur(image: np.ndarray, threshold: float = 100.0, max_size: int = 480) -> dict:
    h,w = image.shape[:2]
    if max(h,w) > max_size:
        scale = max_size / max(h,w)
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image

    variance = cv2.Laplacian(gray_image, cv2.CV_64F).var()
    is_blurry = variance < threshold
    return {
        "passed": not bool(is_blurry),
        "score": float(variance),
        "reason": "severe blur" if is_blurry else None
    }


if __name__ == "__main__":
    test_image = cv2.imread("/Users/halitbektas/Desktop/MobileCaptureQuality/test_images/16_jpg.rf.a4968b32027bd7229872bdd878755bc2.jpg")

    result = check_severe_blur(test_image, threshold=100.0)
    print(f"Bulanıklık Testi Sonucu: {result}")


