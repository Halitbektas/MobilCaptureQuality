import cv2
import mediapipe as mp
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from typing import Dict, Any


def extract_cheek_rois(image, face_mesh_model, roi_size=64):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh_model.process(image_rgb)

    if not results.multi_face_landmarks:
        return None, None

    landmarks = results.multi_face_landmarks[0].landmark
    img_h, img_w, _ = image.shape

    left_cheek_idx = 117
    right_cheek_idx = 346

    cx_left = int(landmarks[left_cheek_idx].x * img_w)
    cy_left = int(landmarks[left_cheek_idx].y * img_h)

    cx_right = int(landmarks[right_cheek_idx].x * img_w)
    cy_right = int(landmarks[right_cheek_idx].y * img_h)

    half_size = roi_size // 2

    def crop_roi(cx, cy):
        start_y = max(0, cy - half_size)
        end_y = min(img_h, cy + half_size)
        start_x = max(0, cx - half_size)
        end_x = min(img_w, cx + half_size)

        roi = image[start_y:end_y, start_x:end_x]

        if roi.size > 0 and (roi.shape[0] != roi_size or roi.shape[1] != roi_size):
            roi = cv2.resize(roi, (roi_size, roi_size))
        return roi if roi.size > 0 else None

    left_cheek_roi = crop_roi(cx_left, cy_left)
    right_cheek_roi = crop_roi(cx_right, cy_right)

    return left_cheek_roi, right_cheek_roi


def preprocess_roi_for_glcm(roi_bgr):
    gray_roi = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized_roi = clahe.apply(gray_roi)
    return equalized_roi


def calculate_glcm_features(roi_gray):
    distances = [1, 2]
    angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]

    glcm_matrix = graycomatrix(
        roi_gray, distances=distances, angles=angles,
        levels=256, symmetric=True, normed=True
    )

    contrast_matrix = graycoprops(glcm_matrix, 'contrast')
    homogeneity_matrix = graycoprops(glcm_matrix, 'homogeneity')

    return np.mean(contrast_matrix), np.mean(homogeneity_matrix)


def check_beauty_filter(image: np.ndarray, face_mesh_model, t_contrast=315.0, t_homogeneity=0.141) -> Dict[str, Any]:
    left_roi, right_roi = extract_cheek_rois(image, face_mesh_model, roi_size=64)

    if left_roi is None or right_roi is None:
        return {
            "passed": False,
            "mean_contrast": 0.0,
            "mean_homogeneity": 0.0,
            "reason": "Yanak bolgeleri (ROI) cikarilamadi."
        }

    left_roi_gray = preprocess_roi_for_glcm(left_roi)
    right_roi_gray = preprocess_roi_for_glcm(right_roi)

    c_left, h_left = calculate_glcm_features(left_roi_gray)
    c_right, h_right = calculate_glcm_features(right_roi_gray)

    mean_face_contrast = (c_left + c_right) / 2
    mean_face_homogeneity = (h_left + h_right) / 2

    is_filtered = (mean_face_contrast > t_contrast and mean_face_homogeneity < t_homogeneity)
    passed = not is_filtered

    return {
        "passed": passed,
        "mean_contrast": round(float(mean_face_contrast), 2),
        "mean_homogeneity": round(float(mean_face_homogeneity), 4),
        "reason": None if passed else "Yogun guzellik filtresi tespit edildi!"
    }