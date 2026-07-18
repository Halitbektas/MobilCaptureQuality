import numpy as np
import cv2
from typing import Dict, Union

def check_color_cast(image: np.ndarray, cast_t: float = 35.0, max_size: int = 480) -> Dict[str, Union[bool, float, str]]:
    if len(image.shape) != 3:
        return {"passed": True, "score": 0.0, "reason": None}

    height, width = image.shape[:2]
    if max(height, width) > max_size:
        scale = max_size / float(max(height, width))
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    l, a, b = cv2.split(lab_image)

    mean_a = np.mean(a)
    mean_b = np.mean(b)

    distance_from_neutral = np.sqrt((mean_a - 128)**2 + (mean_b - 128)**2)

    is_extreme_cast = distance_from_neutral > cast_t

    return {
        "passed": not bool(is_extreme_cast),
        "score": float(distance_from_neutral),
        "reason": "extreme_color_cast" if is_extreme_cast else None
    }


if __name__ == "__main__":
    test_image = cv2.imread("/Users/halitbektas/Desktop/MobileCaptureQuality/test_images/levle3_42.jpg")
    result = check_color_cast(test_image)
    print(f"Renk Tonu Testi Sonucu: {result}")