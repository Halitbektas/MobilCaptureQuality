import cv2
import mediapipe as mp
import numpy as np
import time
from typing import Dict, Union

ANGLE_THRESHOLDS = {
    # Merkez (x: ~4, y: ~0)
    "front": {"y": (-2.5, 2.5), "x": (-1.0, 5.5)},

    # Sola Dönüşler (y ekseni negatif)
    "left_45": {"y": (-7.5, -2.5), "x": (-1.0, 6.5)},
    "left_90": {"y": (-15.0, -7.5), "x": (-1.0, 8.5)},

    # Sağa Dönüşler (y ekseni pozitif)
    "right_45": {"y": (2.5, 7.5), "x": (-1.0, 6.5)},
    "right_90": {"y": (7.5, 15.0), "x": (-1.0, 8.5)},

    # Yukarı Bakışlar (x ekseni pozitif artar)
    "chin_up": {"y": (-3.0, 3.0), "x": (6.0, 12.0)},
    "left_upper_diag": {"y": (-7.5, -2.0), "x": (5.5, 12.0)},
    "right_upper_diag": {"y": (2.0, 7.5), "x": (5.5, 12.0)}
}

def check_head_position(image: np.ndarray,
                        face_mesh,
                        target_position: str = "front") -> Dict[str, Union[bool, str, float]]:

    if target_position not in ANGLE_THRESHOLDS:
        return {"passed": False, "reason": "invalid_target", "pitch": 0.0, "yaw": 0.0, "roll": 0.0}

    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = face_mesh.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if not results.multi_face_landmarks:
        return {"passed": False, "reason": "no_face", "pitch": 0.0, "yaw": 0.0, "roll": 0.0}

    img_h, img_w, _ = image.shape
    face_3d = []
    face_2d = []

    face_landmarks = results.multi_face_landmarks[0]

    for idx, lm in enumerate(face_landmarks.landmark):
        if idx in [33, 263, 1, 61, 291, 199]:
            if idx == 1:
                nose_2d = (lm.x * img_w, lm.y * img_h)
                nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)

            px, py = int(lm.x * img_w), int(lm.y * img_h)
            face_2d.append([px, py])
            face_3d.append([px, py, lm.z])

    face_2d = np.array(face_2d, dtype=np.float64)
    face_3d = np.array(face_3d, dtype=np.float64)

    focal_length = 1 * img_w
    cam_matrix = np.array([
        [focal_length, 0, img_h / 2],
        [0, focal_length, img_w / 2],
        [0, 0, 1]
    ], dtype="double")

    dist_matrix = np.zeros((4, 1), dtype=np.float64)

    _, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
    rmat, _ = cv2.Rodrigues(rot_vec)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

    x = angles[0] * 360  # Pitch
    y = angles[1] * 360  # Yaw
    z = angles[2] * 360  # Roll

    limits = ANGLE_THRESHOLDS[target_position]
    y_min, y_max = limits["y"]
    x_min, x_max = limits["x"]

    is_yaw_ok = y_min <= y <= y_max
    is_pitch_ok = x_min <= x <= x_max

    if target_position in ["left_90", "right_90"]:
        is_passed = is_yaw_ok
    else:
        is_passed = is_yaw_ok and is_pitch_ok

    return {
        "passed": bool(is_passed),
        "target": target_position,
        "pitch": float(x),
        "yaw": float(y),
        "roll": float(z),
        "reason": None if is_passed else "not_in_target_angle"
    }


if __name__ == "__main__":
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    cap = cv2.VideoCapture(0)

    targets = list(ANGLE_THRESHOLDS.keys())
    current_target_index = 0

    print("Test basladi.")
    print("Sonraki hedefe gecmek icin klavyeden 'n' tusuna basin.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        start = time.time()
        current_target = targets[current_target_index]

        # Fonksiyon içeride frame üzerine çizim yapacak
        result = check_head_position(frame, face_mesh, target_position=current_target)

        if result["reason"] != "no_face":
            color = (0, 255, 0) if result["passed"] else (0, 0, 255)
            status_text = "HEDEF BULUNDU!" if result["passed"] else "ACIDAN CIKTI"

            cv2.putText(frame, f"Hedef: {current_target.upper()}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (255, 255, 0), 2)
            cv2.putText(frame, status_text, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            cv2.putText(frame, f"Pitch (x): {np.round(result['pitch'], 1)}", (20, 130), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"Yaw (y): {np.round(result['yaw'], 1)}", (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 255), 2)
            cv2.putText(frame, f"Roll (z): {np.round(result['roll'], 1)}", (20, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 255), 2)
        else:
            cv2.putText(frame, "YUZ BULUNAMADI", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        end = time.time()
        fps = 1 / (end - start)
        cv2.putText(frame, f"FPS: {int(fps)}", (20, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.putText(frame, "Sonraki hedef icin 'N', Cik icin 'ESC'", (20, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow("360 Degree Capture Protocol Test", frame)

        key = cv2.waitKey(5) & 0xFF
        if key == 27 or key == ord('q'):
            break
        elif key == ord('n'):
            current_target_index = (current_target_index + 1) % len(targets)

    cap.release()
    cv2.destroyAllWindows()