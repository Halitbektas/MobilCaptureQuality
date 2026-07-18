import cv2
import mediapipe as mp
import numpy as np
from typing import Dict, Union

# --- AÇI EŞİKLERİ VE KONFİGÜRASYON ---
ANGLE_THRESHOLDS = {
    "front": {"yaw": (-15, 15), "pitch": (-15, 15)},
    "right_45": {"yaw": (30, 60), "pitch": (-15, 15)},
    "right_90": {"yaw": (70, 95), "pitch": (-20, 20)},
    "left_45": {"yaw": (-60, -30), "pitch": (-15, 15)},
    "left_90": {"yaw": (-95, -70), "pitch": (-20, 20)},
    "chin_up": {"yaw": (-15, 15), "pitch": (20, 45)},
    "right_lower_diag": {"yaw": (20, 60), "pitch": (20, 45)},
    "left_lower_diag": {"yaw": (-60, -20), "pitch": (20, 45)}
}


# --- KONTROL FONKSİYONU ---
def check_target_pose(image: np.ndarray,
                      face_mesh,  # Performans için face_mesh dışarıdan alınıyor
                      target_angle_id: str = "front",
                      roll_thresh: float = 15.0) -> Dict[str, Union[bool, str, float]]:
    img_h, img_w, _ = image.shape
    focal_length = img_w
    cam_matrix = np.array([
        [focal_length, 0, img_w / 2],
        [0, focal_length, img_h / 2],
        [0, 0, 1]
    ], dtype="double")

    model_points = np.array([
        (0.0, 0.0, 0.0),
        (0.0, -330.0, -65.0),
        (-225.0, 170.0, -135.0),
        (225.0, 170.0, -135.0),
        (-150.0, -150.0, -125.0),
        (150.0, -150.0, -125.0)
    ], dtype="double")

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_image)

    if not results.multi_face_landmarks:
        return {"passed": False, "reason": "no_face", "yaw": 0.0, "pitch": 0.0, "roll": 0.0}

    landmarks = results.multi_face_landmarks[0].landmark

    image_points = np.array([
        (landmarks[1].x * img_w, landmarks[1].y * img_h),
        (landmarks[152].x * img_w, landmarks[152].y * img_h),
        (landmarks[33].x * img_w, landmarks[33].y * img_h),
        (landmarks[263].x * img_w, landmarks[263].y * img_h),
        (landmarks[61].x * img_w, landmarks[61].y * img_h),
        (landmarks[291].x * img_w, landmarks[291].y * img_h)
    ], dtype="double")

    _, rot_vec, trans_vec = cv2.solvePnP(model_points, image_points, cam_matrix, None)
    rmat, _ = cv2.Rodrigues(rot_vec)

    # Sadece ilk dönen değeri (Euler açılarını) alıyoruz
    euler_angles = cv2.RQDecomp3x3(rmat)[0]

    pitch, yaw, roll = euler_angles[0], euler_angles[1], euler_angles[2]

    if abs(roll) > roll_thresh:
        return {"passed": False, "reason": "head_tilted", "yaw": float(yaw), "pitch": float(pitch), "roll": float(roll)}

    target_limits = ANGLE_THRESHOLDS[target_angle_id]
    yaw_passed = target_limits["yaw"][0] <= yaw <= target_limits["yaw"][1]
    pitch_passed = target_limits["pitch"][0] <= pitch <= target_limits["pitch"][1]

    is_target_reached = yaw_passed and pitch_passed

    return {
        "passed": is_target_reached,
        "target": target_angle_id,
        "yaw": float(yaw),
        "pitch": float(pitch),
        "roll": float(roll),
        "reason": None if is_target_reached else "not_in_target_angle"
    }


# --- CANLI KAMERA TESTİ ---
def main():
    mp_face_mesh = mp.solutions.face_mesh
    cap = cv2.VideoCapture(0)

    targets = list(ANGLE_THRESHOLDS.keys())
    current_target_index = 0

    print("Test basladi.")
    print("Sonraki hedefe gecmek icin klavyeden 'n' tusuna basin.")
    print("Cikmak icin klavyeden 'q' tusuna basin.")

    # Performans için FaceMesh modelini döngü dışında sadece 1 kez başlatıyoruz
    with mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5) as face_mesh:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Kamerayı ayna moduna alıyoruz (Sağ-sol karmaşasını önlemek için)
            frame = cv2.flip(frame, 1)
            current_target = targets[current_target_index]

            # Açı kontrolünü yap
            result = check_target_pose(frame, face_mesh, target_angle_id=current_target)

            # --- GÖRSELLEŞTİRME ---
            color = (0, 255, 0) if result["passed"] else (0, 0, 255)

            # Hedef ve Durum
            status_text = "HEDEF BULUNDU!" if result["passed"] else "ACIDAN CIKTI"
            cv2.putText(frame, f"Hedef: {current_target.upper()}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0),
                        2)
            cv2.putText(frame, status_text, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            # Açı Değerleri
            yaw_text = f"Yaw (Sag/Sol): {int(result['yaw'])}"
            pitch_text = f"Pitch (Asagi/Yukari): {int(result['pitch'])}"
            roll_text = f"Roll (Egilme): {int(result['roll'])}"

            cv2.putText(frame, yaw_text, (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, pitch_text, (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, roll_text, (20, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Talimat
            cv2.putText(frame, "Sonraki hedef icin 'N', Cikmak icin 'Q'", (20, frame.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            cv2.imshow('360 Degree Capture Protocol Test', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('n'):
                # Sonraki hedefe geç (döngüsel)
                current_target_index = (current_target_index + 1) % len(targets)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()