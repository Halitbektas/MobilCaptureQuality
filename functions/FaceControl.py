import cv2
import numpy as np
from typing import Dict, Union, Any
import mediapipe as mp

mp_face_detection = mp.solutions.face_detection

def check_face_in_image(image: np.ndarray, min_confidence: float = 0.6) -> Dict[str, Any]:
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    rgb_image.flags.writeable = False

    with mp_face_detection.FaceDetection(min_detection_confidence=min_confidence) as face_detection:
        results = face_detection.process(rgb_image)

    # Görüntü tekrar yazılabilir yapılıyor (sonraki işlemler için)
    rgb_image.flags.writeable = True

    face_count = 0
    bbox = None

    if results.detections:
        face_count = len(results.detections)
        if face_count == 1:
            detection = results.detections[0]
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = image.shape

            x = int(bboxC.xmin * iw)
            y = int(bboxC.ymin * ih)
            w = int(bboxC.width * iw)
            h = int(bboxC.height * ih)

            x, y = max(0, x), max(0, y)

            bbox = (x, y, w, h)

    passed = False
    reason = None

    if face_count ==0:
        reason = "No face detected."
    elif face_count > 1:
        reason = "Multiple faces detected."
    else:
        passed = True

    return {
        "passed": passed,
        "face_count": face_count,
        "bbox": bbox,
        "reason": reason
    }


if __name__ == "__main__":
    # Test için kameradan anlık görüntü alarak deneyebilirsin:
    cap = cv2.VideoCapture(0)

    print("Test için kameraya bakın. Çıkmak için 'q' tuşuna basın.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        result = check_face_in_image(frame)

        # Sonucu ekrana yazdır
        text = f"Durum: {'GECTI' if result['passed'] else 'RED'} - Sebep: {result['reason']}"
        color = (0, 255, 0) if result['passed'] else (0, 0, 255)
        cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Eğer bir yüz varsa kutu çiz
        if result['bbox']:
            x, y, w, h = result['bbox']
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow("Yuz Kontrol Testi", frame)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()