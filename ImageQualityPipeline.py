import cv2
import mediapipe as mp
import numpy as np
import time
from typing import Dict, Any
from collections import deque

from functions.FaceControl import check_face_in_image
from functions.ColorCast import check_color_cast
from functions.Exposed import check_over_under_exposed
from functions.SevereBlur import check_severe_blur
from functions.FaceSize import check_face_size
from functions.FaceAngle import check_head_position, ANGLE_THRESHOLDS
from functions.ExtremeShadow import check_extreme_shadow

class ImageQualityPipeline:
    def __init__(self):
        self.face_detection = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.6)
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def process_image(self, image: np.ndarray, target_angle: str = "front") -> Dict[str, Any]:
        ui_states = {
            "Blur": "SKIPPED", "Exposure": "SKIPPED", "Color": "SKIPPED",
            "Face Count": "SKIPPED", "Size": "SKIPPED", "Shadow": "SKIPPED", "Angle": "SKIPPED"
        }
        bbox = None
        angles = None

        res = check_severe_blur(image)
        ui_states["Blur"] = f"PASS {res['score']}" if res["passed"] else f"FAIL {res['score']}"
        if not res["passed"]: return self._compile_result(False, res["reason"], ui_states)

        res = check_over_under_exposed(image)
        ui_states["Exposure"] = f"PASS {res['dark_ratio']} | {res['bright_ratio']}" if res["passed"] else f"FAIL {res['dark_ratio']} | {res['bright_ratio']}"
        if not res["passed"]: return self._compile_result(False, res["reason"], ui_states)

        res = check_color_cast(image)
        ui_states["Color"] = f"PASS {res['score']}" if res["passed"] else f"FAIL {res['score']}"
        if not res["passed"]: return self._compile_result(False, res["reason"], ui_states)

        res = check_face_in_image(image, self.face_detection)
        ui_states["Face Count"] = f"PASS {res['face_count']}" if res["passed"] else f"FAIL {res['face_count']}"
        if not res["passed"]: return self._compile_result(False, res["reason"], ui_states)
        bbox = res["bbox"]

        res = check_face_size(image, bbox, 0.05)
        ui_states["Size"] = f"PASS {res['face_ratio']}" if res["passed"] else f"FAIL {res['face_ratio']}"
        if not res["passed"]: return self._compile_result(False, res["reason"], ui_states, bbox)

        res = check_extreme_shadow(image, bbox, 60.0)
        ui_states["Shadow"] = f"PASS {res['shadow_diff']}" if res["passed"] else f"FAIL {res['shadow_diff']}"
        if not res["passed"]: return self._compile_result(False, res["reason"], ui_states, bbox)

        res = check_head_position(image, self.face_mesh, target_angle)
        ui_states["Angle"] = "PASS" if res["passed"] else "FAIL"
        if res["passed"]:
            angles = {"pitch": res["pitch"], "yaw": res["yaw"], "roll": res["roll"]}
        if not res["passed"]: return self._compile_result(False, res["reason"], ui_states, bbox)

        return self._compile_result(True, None, ui_states, bbox, angles)

    def _compile_result(self, is_success, reason, ui_states, bbox=None, angles=None):
        return {
            "status": "success" if is_success else "rejected",
            "reason": reason,
            "ui_states": ui_states,
            "bbox": bbox,
            "angles": angles
        }

    def close(self):
        self.face_detection.close()
        self.face_mesh.close()


if __name__ == "__main__":
    pipeline = ImageQualityPipeline()
    cap = cv2.VideoCapture(0)

    # Hedefleri import ettiğimiz sözlükten alıyoruz
    targets = list(ANGLE_THRESHOLDS.keys())
    current_target_index = 0

    print("Pipeline Testi Basladi.")
    print("Sonraki hedefe gecmek icin 'N' tusuna basin.")

    buffer_size = 5
    success_threshold = 4

    success_buffer = deque(maxlen=buffer_size)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        current_target = targets[current_target_index]

        start = time.time()
        result = pipeline.process_image(frame, target_angle=current_target)
        fps = int(1 / (time.time() - start))

        is_current_frame_success = (result["status"] == "success")
        success_buffer.append(is_current_frame_success)

        stable_success_count = sum(success_buffer)

        is_stable_success = stable_success_count >= success_threshold

        # --- UI ÇİZİMLERİ ---
        if result["bbox"]:
            x, y, w, h = result["bbox"]
            color = (0, 255, 0) if is_stable_success else (0, 165, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        y_offset = 30
        cv2.putText(frame, f"HEDEF: {current_target.upper()}", (15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        y_offset += 30

        for key, state in result["ui_states"].items():
            if "PASS" in state: color = (0, 255, 0)
            elif "FAIL" in state: color = (0, 0, 255)
            else: color = (150, 150, 150)

            text = f"{key}: {state}"
            cv2.putText(frame, text, (15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y_offset += 25

        if is_stable_success:
            cv2.putText(frame, "MUKEMMEL CEKIM! (STABIL)", (15, y_offset + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            reason_text = str(result["reason"]).replace("_", " ").upper() if result["reason"] else "SABIT KALIN..."
            cv2.putText(frame, f"HATA: {reason_text}", (15, y_offset + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.putText(frame, f"Buffer: [{stable_success_count}/{buffer_size}]", (15, y_offset + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.putText(frame, f"FPS: {fps}", (frame.shape[1] - 100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, "[N] Sonraki Hedef  |  [ESC] Cikis", (15, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("Quality Pipeline HUD", frame)

        key = cv2.waitKey(5) & 0xFF
        if key == 27 or key == ord('q'):
            break
        elif key == ord('n'):
            current_target_index = (current_target_index + 1) % len(targets)
            success_buffer.clear()

    pipeline.close()
    cap.release()
    cv2.destroyAllWindows()