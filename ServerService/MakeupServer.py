from fastapi import FastAPI, UploadFile, File
import uvicorn
import cv2
import numpy as np
import onnxruntime as ort
import os

app = FastAPI(title="Makeup Detection API")
model_dir = "Model"
MODEL_PATH = os.path.join(model_dir, "makeup_model_timm.onnx")
print(f"ONNX modeli yükleniyor {MODEL_PATH}")

try:
    ort_session = ort.InferenceSession(MODEL_PATH)
    input_name = ort_session.get_inputs()[0].name
    print("model yüklendi")
except Exception as e:
    print(f"Model yüklenirken hata oluştu: {e}")
    ort_session = None

def preprocess_image(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (224, 224))
    img_normalized = img_resized.astype(np.float32) / 255.0

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_normalized = (img_normalized - mean) / std

    img_transposed = np.transpose(img_normalized, (2, 0, 1))
    img_expanded = np.expand_dims(img_transposed, axis=0)

    return img_expanded

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=1, keepdims=True)


@app.post("/predict_makeup")
async def predict_makeup(file: UploadFile = File(...)):
    if ort_session is None:
        return {
            "status": "error",
            "message": "Model yüklenemedi, lütfen sunucu loglarını kontrol edin."
        }

    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return {
                "status": "error",
                "message": "Geçersiz resim dosyası."
            }

        input_tensor = preprocess_image(img)
        outputs = ort_session.run(None, {input_name: input_tensor})
        logits = outputs[0]
        probabilities = softmax(logits)[0]

        makeup_prob = float(probabilities[0])

        has_makeup = bool(makeup_prob > 0.5)

        return {
            "status": "success",
            "has_makeup": has_makeup,
            "confidence": round(makeup_prob * 100, 2),
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)

