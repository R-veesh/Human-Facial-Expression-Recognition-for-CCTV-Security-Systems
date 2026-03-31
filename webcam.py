"""
Real-time Facial Expression Recognition via Webcam
---------------------------------------------------
Controls:
  Q  — quit
  S  — save screenshot
  P  — pause / resume
"""

import cv2
import numpy as np
import joblib
import warnings
import os
import time

warnings.filterwarnings("ignore", category=UserWarning)

# ── Load pipeline ─────────────────────────────────────────────────────────────
model   = joblib.load('saved_model/best_model.pkl')
scaler  = joblib.load('saved_model/scaler.pkl')
pca     = joblib.load('saved_model/pca.pkl')
le      = joblib.load('saved_model/label_encoder.pkl')

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# ── Colour map per emotion ────────────────────────────────────────────────────
EMOTION_COLORS = {
    'anger':    (0,   0,   220),   # red
    'contempt': (0,   165, 255),   # orange
    'disgust':  (0,   200, 150),   # teal
    'fear':     (180, 0,   180),   # purple
    'happy':    (0,   220, 0  ),   # green
    'sadness':  (220, 100, 0  ),   # blue
    'surprise': (0,   200, 220),   # yellow
}

# ── Inference helper ──────────────────────────────────────────────────────────
def predict_frame(gray_face):
    """Takes a grayscale, any-size face crop → returns emotion string."""
    face = cv2.resize(gray_face, (48, 48))
    face = cv2.equalizeHist(face)
    arr  = face.flatten().reshape(1, -1).astype(np.float64)
    arr  = scaler.transform(arr)
    arr  = pca.transform(arr)
    return le.inverse_transform(model.predict(arr))[0]

# ── UI helpers ────────────────────────────────────────────────────────────────
def draw_label(frame, text, x, y, color):
    """Draw a filled pill-style label above the bounding box."""
    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.75
    thickness  = 2
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    pad = 6
    cv2.rectangle(frame, (x - pad, y - th - pad * 2),
                  (x + tw + pad, y + baseline), color, -1)
    cv2.putText(frame, text, (x, y - pad),
                font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

def draw_hud(frame, fps, paused, face_count):
    """Draw FPS, status, and key hints in the top-left corner."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (260, 110), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

    font = cv2.FONT_HERSHEY_SIMPLEX
    status = "PAUSED" if paused else f"LIVE  {fps:.1f} FPS"
    color  = (0, 165, 255) if paused else (0, 255, 100)
    cv2.putText(frame, status,       (10, 28),  font, 0.7, color, 2, cv2.LINE_AA)
    cv2.putText(frame, f"Faces: {face_count}", (10, 55),  font, 0.6, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(frame, "Q-quit  S-save  P-pause", (10, 80),  font, 0.50, (180, 180, 180), 1, cv2.LINE_AA)
    cv2.putText(frame, "Facial Expression Monitor", (10, 105), font, 0.45, (120, 120, 120), 1, cv2.LINE_AA)

# ── Main loop ─────────────────────────────────────────────────────────────────
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open webcam. Check it is connected and not in use.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    os.makedirs('screenshots', exist_ok=True)

    paused      = False
    prev_time   = time.time()
    fps         = 0.0
    last_frame  = None     # keep last frame when paused
    screenshot_n = 0

    print("Webcam started. Press Q to quit, S to save screenshot, P to pause.")

    while True:
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('p'):
            paused = not paused
            print("Paused" if paused else "Resumed")
        elif key == ord('s'):
            if last_frame is not None:
                fname = f"screenshots/capture_{screenshot_n:04d}.png"
                cv2.imwrite(fname, last_frame)
                screenshot_n += 1
                print(f"Saved: {fname}")

        if paused and last_frame is not None:
            cv2.imshow('Facial Expression Recognition  [Q=quit  S=save  P=pause]', last_frame)
            continue

        ret, frame = cap.read()
        if not ret:
            print("ERROR: Failed to grab frame.")
            break

        # ── FPS ──────────────────────────────────────────────────────────────
        now      = time.time()
        fps      = 0.9 * fps + 0.1 * (1.0 / max(now - prev_time, 1e-6))
        prev_time = now

        # ── Face detection ───────────────────────────────────────────────────
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
        )

        for (x, y, w, h) in faces:
            face_gray = gray[y:y+h, x:x+w]
            try:
                emotion = predict_frame(face_gray)
            except Exception:
                emotion = "error"

            color = EMOTION_COLORS.get(emotion, (200, 200, 200))

            # Bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            # Label
            draw_label(frame, emotion.upper(), x, y, color)

        # ── HUD ──────────────────────────────────────────────────────────────
        draw_hud(frame, fps, paused, len(faces))

        last_frame = frame.copy()
        cv2.imshow('Facial Expression Recognition  [Q=quit  S=save  P=pause]', frame)

    cap.release()
    cv2.destroyAllWindows()
    print("Webcam closed.")

if __name__ == '__main__':
    main()
