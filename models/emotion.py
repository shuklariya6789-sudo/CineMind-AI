import cv2
import numpy as np
from pathlib import Path


EMOTIONS = [
    "neutral",
    "happiness",
    "surprise",
    "sadness",
    "anger",
    "disgust",
    "fear",
    "contempt"
]

# Face-detection confidence threshold. Detections below this are ignored.
FACE_CONF_THRESHOLD = 0.5

# Resolve model paths relative to this file, not the caller's cwd, so the
# module works no matter where the script that imports it is run from.
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

FACE_MODEL = MODEL_DIR / "res10_300x300_ssd_iter_140000.caffemodel"
FACE_CONFIG = MODEL_DIR / "deploy.prototxt"
EMOTION_MODEL = MODEL_DIR / "emotion-ferplus-8.onnx"

_missing = [
    str(p) for p in (FACE_MODEL, FACE_CONFIG, EMOTION_MODEL) if not p.exists()
]
if _missing:
    raise FileNotFoundError(
        "Missing required model file(s):\n  "
        + "\n  ".join(_missing)
        + "\nDownload the face-detection and FER+ ONNX models into the "
          "'models/' folder before running this module."
    )

face_net = cv2.dnn.readNetFromCaffe(
    str(FACE_CONFIG),
    str(FACE_MODEL)
)

emotion_net = cv2.dnn.readNetFromONNX(
    str(EMOTION_MODEL)
)


def detect_emotion(image):

    h, w = image.shape[:2]

    # Face detection
    blob = cv2.dnn.blobFromImage(
        image,
        1.0,
        (300, 300),
        (104, 177, 123),
        swapRB=False,
        crop=False
    )

    face_net.setInput(blob)
    detections = face_net.forward()

    results = []

    for i in range(detections.shape[2]):

        confidence = detections[0, 0, i, 2]

        if confidence < FACE_CONF_THRESHOLD:
            continue

        box = detections[0, 0, i, 3:7] * np.array(
            [w, h, w, h]
        )

        x1, y1, x2, y2 = box.astype(int)

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        face = image[y1:y2, x1:x2]

        if face.size == 0:
            continue

        # Convert face to grayscale
        gray_face = cv2.cvtColor(
            face,
            cv2.COLOR_BGR2GRAY
        )

        # Resize for FER+ model
        emotion_blob = cv2.dnn.blobFromImage(
            gray_face,
            1.0,
            (64, 64),
            (0,),
            swapRB=False,
            crop=False
        )

        emotion_net.setInput(emotion_blob)

        predictions = emotion_net.forward()

        # Convert scores to probabilities
        scores = predictions[0]

        exp_scores = np.exp(
            scores - np.max(scores)
        )

        probabilities = (
            exp_scores / np.sum(exp_scores)
        )

        emotion_index = np.argmax(probabilities)

        emotion = EMOTIONS[emotion_index]

        emotion_probabilities = {
            EMOTIONS[i]: float(probabilities[i] * 100)
            for i in range(len(EMOTIONS))
        }

        results.append({
            "box": (x1, y1, x2, y2),
            "emotion": emotion,
            "confidence": float(confidence),
            "probabilities": emotion_probabilities
        })

    return results