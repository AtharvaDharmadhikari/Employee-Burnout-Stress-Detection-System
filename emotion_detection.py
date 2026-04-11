"""
Emotion Detection Module
Supports three modes:
  1. Webcam / Image — Custom model (model_optimal.keras) as primary,
                      DeepFace as fallback if no face detected
  2. Manual         — User selects mood from a dropdown (always available)

Custom model details:
  - Input : 48x48 grayscale, normalized (/ 255.0)
  - Labels: ["angry", "disgust", "fear", "happy", "sad", "surprised", "neutral"]
  - Face detection: OpenCV Haar cascade

Returns a dict: { "mood": str, "stress_level": int, "source": str }
"""

from __future__ import annotations
import os

# Path to your trained model (sits in the project root)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model_optimal.keras")
IMG_SIZE   = (48, 48)

# Labels must match the training order exactly
EMOTION_LABELS = ["angry", "disgust", "fear", "happy", "sad", "surprised", "neutral"]

# Stress level mapping (0-10 scale)
MOOD_STRESS_MAP: dict[str, int] = {
    "happy":     2,
    "calm":      1,
    "neutral":   4,
    "surprised": 5,
    "surprise":  5,
    "sad":       6,
    "fear":      7,
    "angry":     8,
    "disgust":   8,
    "stressed":  9,
    "tired":     6,
}

# Mood options shown in the manual UI
MOOD_OPTIONS = ["Happy", "Calm", "Neutral", "Tired", "Sad",
                "Stressed", "Angry", "Surprised", "Fear", "Disgust"]


def mood_to_stress(mood: str) -> int:
    """Return a 0-10 stress level for a given mood string."""
    return MOOD_STRESS_MAP.get(mood.lower(), 4)


# ── Custom model loader (cached so it loads only once) ────────────────────────

_model = None

def _get_model():
    global _model
    if _model is None:
        from tensorflow.keras.models import load_model
        _model = load_model(MODEL_PATH)
    return _model


# ── Core detection using your custom model ────────────────────────────────────

def detect_from_frame(frame) -> dict:
    """
    Detect emotion from an OpenCV BGR or RGB numpy array.
    Pipeline:
      1. Haar cascade face detection
      2. Crop & resize to 48x48 grayscale
      3. Normalize and predict with model_optimal.keras
      4. Fall back to DeepFace if no face found by Haar cascade
    """
    import cv2
    import numpy as np
    from tensorflow.keras.preprocessing.image import img_to_array

    try:
        model = _get_model()
        detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        # Works with both BGR (webcam) and RGB (Streamlit camera_input)
        if frame.shape[2] == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)

        faces = detector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        if len(faces) == 0:
            # No face found by Haar — fall back to DeepFace
            return _deepface_fallback(frame)

        x, y, w, h = faces[0]
        roi = cv2.resize(gray[y:y+h, x:x+w], IMG_SIZE)
        roi_array = img_to_array(roi.astype("float") / 255.0)
        roi_array = np.expand_dims(roi_array, axis=0)

        preds   = model.predict(roi_array, verbose=0)[0]
        emotion = EMOTION_LABELS[int(np.argmax(preds))]
        confidence = float(np.max(preds))

        # Build per-emotion confidence dict for display
        raw_emotions = {EMOTION_LABELS[i]: float(preds[i]) for i in range(len(preds))}

        return {
            "mood":        emotion.capitalize(),
            "stress_level": mood_to_stress(emotion),
            "source":      "custom_model",
            "confidence":  round(confidence * 100, 1),
            "raw_emotions": raw_emotions,
        }

    except Exception as e:
        return {
            "mood":        "Neutral",
            "stress_level": 4,
            "source":      "detection_error",
            "error":       str(e),
        }


def _deepface_fallback(frame) -> dict:
    """Use DeepFace when Haar cascade finds no face."""
    try:
        from deepface import DeepFace
        result = DeepFace.analyze(
            frame,
            actions=["emotion"],
            enforce_detection=False,
            silent=True,
        )
        if isinstance(result, list):
            result = result[0]
        dominant = result.get("dominant_emotion", "neutral").lower()
        return {
            "mood":        dominant.capitalize(),
            "stress_level": mood_to_stress(dominant),
            "source":      "deepface_fallback",
            "raw_emotions": result.get("emotion", {}),
        }
    except Exception as e:
        return {
            "mood":        "Neutral",
            "stress_level": 4,
            "source":      "deepface_error",
            "error":       str(e),
        }


# ── Webcam snapshot ───────────────────────────────────────────────────────────

def capture_and_detect() -> dict:
    """Open webcam, capture one frame, run detection, release camera."""
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return {"mood": "Neutral", "stress_level": 4,
                    "source": "webcam_error", "error": "Could not open webcam."}
        ret, frame = cap.read()
        cap.release()
        if not ret:
            return {"mood": "Neutral", "stress_level": 4,
                    "source": "webcam_error", "error": "Failed to read frame."}
        return detect_from_frame(frame)
    except ImportError:
        return {"mood": "Neutral", "stress_level": 4,
                "source": "webcam_error", "error": "opencv-python not installed."}


# ── Manual override ───────────────────────────────────────────────────────────

def manual_detection(mood: str) -> dict:
    return {
        "mood":        mood.capitalize(),
        "stress_level": mood_to_stress(mood),
        "source":      "manual",
    }


# ── Streamlit camera_input widget ─────────────────────────────────────────────

def streamlit_webcam_widget():
    """
    Render Streamlit's camera_input, run detection on the captured photo.
    Returns detection result dict or None if no photo taken yet.
    """
    import streamlit as st
    from PIL import Image
    import numpy as np

    img_file = st.camera_input("Take a photo for emotion detection")
    if img_file is not None:
        image = Image.open(img_file).convert("RGB")
        frame = np.array(image)
        # Convert RGB → BGR for OpenCV Haar cascade
        import cv2
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        return detect_from_frame(frame_bgr)
    return None