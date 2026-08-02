import cv2
import numpy as np
import mediapipe as mp
from typing import Dict, Any


def check_face_occlusion(image: np.ndarray, face_mesh_model, segmenter_model, skin_threshold: float = 80.0) -> Dict[
    str, Any]:
    h, w, _ = image.shape
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


    mesh_results = face_mesh_model.process(rgb_image)
    if not mesh_results.multi_face_landmarks:
        return {
            "passed": False,
            "skin_ratio": 0.0,
            "reason": "Occlusion icin yuz bulunamadi."
        }


    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
    segmentation_result = segmenter_model.segment(mp_image)
    category_mask = segmentation_result.category_mask.numpy_view()


    actual_skin_mask = np.where(category_mask == 3, 255, 0).astype(np.uint8)


    face_landmarks = mesh_results.multi_face_landmarks[0]


    y_coords = [int(pt.y * h) for pt in face_landmarks.landmark]
    min_y = min(y_coords)
    max_y = max(y_coords)
    face_height = max_y - min_y
    top_offset = int(face_height * 0.15)


    all_points = []
    for pt in face_landmarks.landmark:
        x = int(pt.x * w)
        y = int(pt.y * h)
        if y < min_y + (face_height * 0.25):
            y = max(0, y - top_offset)
        all_points.append([x, y])

    all_points = np.array(all_points, dtype=np.int32)
    hull = cv2.convexHull(all_points)


    expected_face_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(expected_face_mask, hull, 255)

    expected_pixels = np.count_nonzero(expected_face_mask)
    if expected_pixels == 0:
        return {"passed": False, "skin_ratio": 0.0, "reason": "Gecersiz yuz alani hesaplandi."}


    face_skin_pixels = cv2.bitwise_and(expected_face_mask, actual_skin_mask)
    actual_skin_count = np.count_nonzero(face_skin_pixels)

    skin_ratio = (actual_skin_count / expected_pixels) * 100

    passed = skin_ratio >= skin_threshold
    reason = None if passed else "Yuzde engel tespit edildi (Gozluk, Sac, Maske vb.)"

    return {
        "passed": passed,
        "skin_ratio": round(skin_ratio, 1),
        "reason": reason
    }