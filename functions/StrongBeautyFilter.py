import cv2
import mediapipe as mp
import numpy as np
from skimage.feature import graycomatrix, graycoprops


# --- PHASE 1: Pure Skin ROI Extraction ---
def extract_cheek_rois(image, roi_size=64):
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.5
    )

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)

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

        if roi.shape[0] != roi_size or roi.shape[1] != roi_size:
            roi = cv2.resize(roi, (roi_size, roi_size))
        return roi

    left_cheek_roi = crop_roi(cx_left, cy_left)
    right_cheek_roi = crop_roi(cx_right, cy_right)

    face_mesh.close()
    return left_cheek_roi, right_cheek_roi


# --- PHASE 2: Preprocessing ---
def preprocess_roi_for_glcm(roi_bgr):
    gray_roi = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized_roi = clahe.apply(gray_roi)
    return equalized_roi


# --- PHASE 3 & 4: GLCM Matrix Calculation & Feature Extraction ---
def calculate_glcm_features(roi_gray):
    """
    Calculates the GLCM matrix based on neighboring pixels,
    then extracts the average Contrast and Homogeneity values.
    """
    # Distance parameters (1 and 2 pixels) for micro-texture (pores)
    distances = [1, 2]

    # Angle parameters in radians (0, 45, 90, 135 degrees)
    # This ensures the algorithm is not biased towards specific lighting directions
    angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]

    # Phase 3: GLCM Matrix Calculation
    # levels=256 because grayscale images have a pixel range of 0-255
    # symmetric=True and normed=True ensure stable and comparable values
    glcm_matrix = graycomatrix(
        roi_gray,
        distances=distances,
        angles=angles,
        levels=256,
        symmetric=True,
        normed=True
    )

    # Phase 4: Feature Extraction from the matrix
    # The result is a 2D array (number of distances x number of angles). We take the average.
    contrast_matrix = graycoprops(glcm_matrix, 'contrast')
    homogeneity_matrix = graycoprops(glcm_matrix, 'homogeneity')

    # Calculate the mean across all distances and angles
    mean_contrast = np.mean(contrast_matrix)
    mean_homogeneity = np.mean(homogeneity_matrix)

    return mean_contrast, mean_homogeneity

# --- PHASE 5: Decision Logic (Thresholding) ---
def detect_beauty_filter(contrast, homogeneity, t_contrast=315.0, t_homogeneity=0.141):
    """
    Returns True if the image is indicated to be using a Strong Beauty Filter.
    Uses default thresholds based on GLCM texture research.
    """
    # Logic: If contrast is destroyed (low) AND skin is too uniform (high)
    if contrast < t_contrast and homogeneity > t_homogeneity:
        return True
    else:
        return False


import cv2


def analyze_beauty_filter(image_path):
    img = cv2.imread(image_path)

    if img is None:
        return {
            "passed": False,
            "mean_contrast": None,
            "mean_homogeneity": None,
            "reason": "Failed to load image."
        }

    # Resize image
    max_height = 800
    if img.shape[0] > max_height:
        ratio = max_height / img.shape[0]
        new_width = int(img.shape[1] * ratio)
        img = cv2.resize(img, (new_width, max_height))

    # Proceed to ROI extraction
    left_roi, right_roi = extract_cheek_rois(img, roi_size=64)

    if left_roi is None or right_roi is None:
        return {
            "passed": False,
            "mean_contrast": None,
            "mean_homogeneity": None,
            "reason": "FAILED: Face could not be detected by MediaPipe."
        }

    # Preprocess both cheeks
    left_roi_gray = preprocess_roi_for_glcm(left_roi)
    right_roi_gray = preprocess_roi_for_glcm(right_roi)

    # Calculate Matrix for both cheeks
    c_left, h_left = calculate_glcm_features(left_roi_gray)
    c_right, h_right = calculate_glcm_features(right_roi_gray)

    # average calculation
    mean_face_contrast = (c_left + c_right) / 2
    mean_face_homogeneity = (h_left + h_right) / 2

    # Execute Decision based on averaged values
    is_filtered = detect_beauty_filter(mean_face_contrast, mean_face_homogeneity)

    # wether pass or not with reason
    if not is_filtered:
        passed = True
        reason = "PASS: Skin looks natural based on overall face texture."
    else:
        passed = False
        reason = "REJECT: Strong Beauty Filter detected based on overall face texture!"

    # Return
    return {
        "passed": passed,
        "mean_contrast": round(mean_face_contrast, 2),
        "mean_homogeneity": round(mean_face_homogeneity, 4),
        "reason": reason
    }


if __name__ == "__main__":
    image_path = "../1.jpg"

    # call function and save the result
    result = analyze_beauty_filter(image_path)

    # print dict
    print(result)