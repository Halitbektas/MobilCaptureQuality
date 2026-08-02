import cv2
import mediapipe as mp
import numpy as np
import time
import requests
from typing import Dict, Any
from collections import deque

from functions.FaceControl import check_face_in_image
from functions.ColorCast import check_color_cast
from functions.Exposed import check_over_under_exposed
from functions.SevereBlur import check_severe_blur
from functions.FaceSize import check_face_size
from functions.FaceAngle import check_head_position, ANGLE_THRESHOLDS
from functions.ExtremeShadow import check_extreme_shadow
from functions.StrongBeautyFilter import check_beauty_filter
import os
import urllib.request
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from functions.Occlusion import check_face_occlusion
import threading

class ImageQualityPipeline:
    def __init__(self):
        self.face_detection = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.6)
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

        model_dir = "Model"
        model_path = os.path.join(model_dir, "selfie_multiclass_256x256.tflite")
        model_url = "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite"

        if not os.path.exists(model_path):
            os.makedirs(model_dir, exist_ok=True)
            print("Segmentasyon modeli indiriliyor...")
            urllib.request.urlretrieve(model_url, model_path)

        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.ImageSegmenterOptions(base_options=base_options, output_category_mask=True)
        self.segmenter = vision.ImageSegmenter.create_from_options(options)

    def process_image(self, image: np.ndarray, target_angle: str = "front") -> Dict[str, Any]:
        ui_states = {
            "Blur": "SKIPPED", "Exposure": "SKIPPED", "Color": "SKIPPED",
            "Face Count": "SKIPPED", "Size": "SKIPPED", "Occlusion": "SKIPPED", "Shadow": "SKIPPED", "Filter": "SKIPPED", "Angle": "SKIPPED"
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

        if target_angle == "front":
            res = check_face_occlusion(image, self.face_mesh, self.segmenter)
            ui_states["Occlusion"] = f"PASS %{res['skin_ratio']}" if res["passed"] else f"FAIL %{res['skin_ratio']}"
            if not res["passed"]: return self._compile_result(False, res["reason"], ui_states, bbox)

        res = check_extreme_shadow(image, bbox, 90.0)
        ui_states["Shadow"] = f"PASS {res['shadow_diff']}" if res["passed"] else f"FAIL {res['shadow_diff']}"
        if not res["passed"]: return self._compile_result(False, res["reason"], ui_states, bbox)

        if target_angle == "front":
            res = check_beauty_filter(image, self.face_mesh)
            ui_states["Filter"] = f"PASS | C:{res['mean_contrast']} | H: {res['mean_homogeneity']}" if res["passed"] else f"FAIL | C:{res['mean_contrast']} | H: {res['mean_homogeneity']}"
            if not res["passed"]: return self._compile_result(False, res["reason"], ui_states, bbox)

        res = check_head_position(image, self.face_mesh, target_angle)
        ui_states["Angle"] = f"PASS pitch : {np.round(res['pitch'], 1)} | yaw: {np.round(res['yaw'], 1)}" if res["passed"] else f"FAIL pitch : {np.round(res['pitch'], 1)} | yaw: {np.round(res['yaw'], 1)}"
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
        self.segmenter.close()


if __name__ == "__main__":
    pipeline = ImageQualityPipeline()
    cap = cv2.VideoCapture(0)


    targets = list(ANGLE_THRESHOLDS.keys())
    current_target_index = 0

    print("Pipeline Testi Basladi.")
    print("Sonraki hedefe gecmek icin 'N' tusuna basin.")

    buffer_size = 5
    success_threshold = 4

    success_buffer = deque(maxlen=buffer_size)

    latest_frame = None
    latest_result = None
    is_running = True
    ai_fps = 0

    def pipeline_worker():
        global latest_frame, latest_result, ai_fps, is_running

        while is_running:
            if latest_frame is not None:
                start_ai = time.time()
                current_target = targets[current_target_index]
                frame_to_process = latest_frame.copy()

                res=pipeline.process_image(frame_to_process, target_angle=current_target)
                latest_result= res
                ai_fps = int(1 / (time.time() - start_ai))
                time.sleep(0.01)
            else:
                time.sleep(0.05)

    worker_thread = threading.Thread(target=pipeline_worker, daemon=True)
    worker_thread.start()

    while cap.isOpened():
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)

        if not ret: break

        start_cam = time.time()

        latest_frame = frame.copy()
        current_target = targets[current_target_index]

        if latest_result is not None:
            is_current_frame_success = (latest_result["status"] == "success")
            success_buffer.append(is_current_frame_success)
            stable_success_count = sum(success_buffer)
            is_stable_success = stable_success_count >= success_threshold

            if latest_result["bbox"]:
                x, y, w, h = latest_result["bbox"]
                color = (0, 255, 0) if is_stable_success else (0, 165, 255)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            y_offset = 30
            cv2.putText(frame, f"HEDEF: {current_target.upper()}", (15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 0), 2)
            y_offset += 30

            for key, state in latest_result["ui_states"].items():
                if "PASS" in state:
                    color = (0, 255, 0)
                elif "FAIL" in state:
                    color = (0, 0, 255)
                else:
                    color = (150, 150, 150)

                text = f"{key}: {state}"
                cv2.putText(frame, text, (15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                y_offset += 25

            if is_stable_success:
                cv2.putText(frame, "MUKEMMEL CEKIM! (STABIL)", (15, y_offset + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (0, 255, 0), 2)
            else:
                reason_text = str(latest_result["reason"]).replace("_", " ").upper() if latest_result[
                    "reason"] else "SABIT KALIN..."
                cv2.putText(frame, f"HATA: {reason_text}", (15, y_offset + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (0, 0, 255), 2)

            cv2.putText(frame, f"Buffer: [{stable_success_count}/{buffer_size}]", (15, y_offset + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cam_fps = int(1 / (time.time() - start_cam))

        cv2.putText(frame, f"Cam FPS: {cam_fps} | AI FPS: {ai_fps}", (frame.shape[1] - 320, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, "[N] Sonraki Hedef  |  [ESC] Cikis", (15, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 2)

        global api_makeup_result
        if 'api_makeup_result' not in globals():
            api_makeup_result = "Makyaj Testi Bekleniyor... (Test için 'M'ye bas)"
            api_makeup_color = (255, 255, 255)
        cv2.putText(frame, api_makeup_result, (15, frame.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    api_makeup_color, 2)

        cv2.imshow("Quality Pipeline HUD", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            is_running = False
            break
        elif key == ord('n'):
            current_target_index = (current_target_index + 1) % len(targets)
            success_buffer.clear()
        elif key == ord('m'):
            if latest_result is not None and latest_result["bbox"] is not None:
                x, y, w, h = latest_result["bbox"]
                face_crop = frame[y:y+h, x:x+w]
                def send_makeup_req(img_crop):
                    global api_makeup_result, api_makeup_color
                    api_makeup_result = "API'ye gonderiliyor"
                    api_makeup_color = (0,255,255)

                    try:
                        _, img_encoded = cv2.imencode('.jpg', img_crop)
                        files = {'file': ('face.jpg', img_encoded.tobytes(), 'image/jpeg')}

                        response = requests.post("http://127.0.0.1:8000/predict_makeup", files=files)
                        data = response.json()

                        if data['status'] == 'success':
                            if data['has_makeup']:
                                api_makeup_result = f"Makyaj Var (%{data['confidence']})"
                                api_makeup_color = (0, 0, 255)
                            else:
                                api_makeup_result = f"Makyaj Yok (%{data['confidence']})"
                                api_makeup_color = (0, 255, 0)
                    except Exception as e:
                        api_makeup_result = f"API Hatasi: {str(e)}"
                        api_makeup_color = (0, 0, 255)

                threading.Thread(target=send_makeup_req, args=(face_crop,), daemon=True).start()
            else:
                api_makeup_result = "Yuz bulunamadi, istek atilamadi"
                api_makeup_color = (0, 0, 255)

    is_running = False
    pipeline.close()
    cap.release()
    cv2.destroyAllWindows()