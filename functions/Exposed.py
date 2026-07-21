import numpy as np
import cv2
from typing import Dict, Union

def check_over_under_exposed(image: np.ndarray,
                             dark_thresh: int = 30,
                             bright_thresh: int = 225,
                             max_dark_ratio: float = 0.40,
                             max_bright_ratio: float = 0.30,
                             max_size: int = 480) -> Dict[str, Union[bool, float, str]]:
    height, width = image.shape[:2]

    if max(height, width) > max_size:
        scale = max_size / float(max(height, width))
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    total_pixels = gray.size

    dark_pixels = np.sum(gray < dark_thresh)
    bright_pixels = np.sum(gray > bright_thresh)

    dark_ratio = dark_pixels / total_pixels
    bright_ratio = bright_pixels / total_pixels

    passed = True
    reason = None

    if bright_ratio > max_bright_ratio:
        passed = False
        reason = "overexposure"
    elif dark_ratio > max_dark_ratio:
        passed = False
        reason = "underexposure"

    return {
        "passed": passed,
        "dark_ratio": float(dark_ratio),
        "bright_ratio": float(bright_ratio),
        "reason": reason
    }


if __name__ == "__main__":
    # Test için karanlık bir matris oluşturalım (örn: az ışıklı ortam)
    test_image = cv2.imread("/Users/halitbektas/Desktop/MobileCaptureQuality/test_images/images.jpeg")
    result = check_over_under_exposed(test_image)
    print(f"Pozlama Testi Sonucu: {result}")