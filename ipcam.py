"""
Real-time Facial Expression Recognition via IP Camera
------------------------------------------------------
Supports:
  • Android  "IP Webcam" app  →  http://192.168.x.x:8080/video
  • Any MJPEG HTTP stream     →  http://IP:PORT/video  (or /mjpeg, /stream)
  • RTSP stream               →  rtsp://user:pass@IP:554/stream
  • USB/built-in webcam       →  0  (or 1, 2 …)

Usage:
  python ipcam.py                          # prompts for URL
  python ipcam.py http://192.168.1.5:8080/video
  python ipcam.py rtsp://admin:admin@192.168.1.10:554/stream1
  python ipcam.py 0                        # local webcam index

Controls (while window is open):
  Q  — quit
  S  — save screenshot to screenshots/
  P  — pause / resume
  R  — reconnect (if stream dropped)
"""

import cv2
import numpy as np
import joblib
import warnings
import os
import sys
import time

warnings.filterwarnings("ignore", category=UserWarning)

# ── Load pipeline ─────────────────────────────────────────────────────────────
model  = joblib.load('saved_model/best_model.pkl')
scaler = joblib.load('saved_model/scaler.pkl')
pca    = joblib.load('saved_model/pca.pkl')
le     = joblib.load('saved_model/label_encoder.pkl')

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# ── Colour map per emotion ────────────────────────────────────────────────────
EMOTION_COLORS = {
    'anger':    (0,   0,   220),
    'contempt': (0,   165, 255),
    'disgust':  (0,   200, 150),
    'fear':     (180, 0,   180),
    'happy':    (0,   220, 0  ),
    'sadness':  (220, 100, 0  ),
    'surprise': (0,   200, 220),
}

RECONNECT_DELAY = 3      # seconds to wait before reconnecting
MAX_FAIL_FRAMES = 60     # consecutive read failures before reconnecting

# ── Inference ─────────────────────────────────────────────────────────────────
def predict_face(gray_face):
    face = cv2.resize(gray_face, (48, 48))
    face = cv2.equalizeHist(face)
    arr  = face.flatten().reshape(1, -1).astype(np.float64)
    arr  = scaler.transform(arr)
    arr  = pca.transform(arr)
    return le.inverse_transform(model.predict(arr))[0]

# ── UI helpers ────────────────────────────────────────────────────────────────
def draw_label(frame, text, x, y, color):
    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.75
    thickness  = 2
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    pad = 6
    cv2.rectangle(frame,
                  (x - pad,      y - th - pad * 2),
                  (x + tw + pad, y + baseline), color, -1)
    cv2.putText(frame, text, (x, y - pad),
                font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

def draw_hud(frame, fps, paused, face_count, source_label):
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (320, 120), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

    font   = cv2.FONT_HERSHEY_SIMPLEX
    status = "PAUSED" if paused else f"LIVE  {fps:.1f} FPS"
    color  = (0, 165, 255) if paused else (0, 255, 100)
    cv2.putText(frame, status,                      (10, 28),  font, 0.70, color,           2, cv2.LINE_AA)
    cv2.putText(frame, f"Faces: {face_count}",      (10, 55),  font, 0.60, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(frame, "Q-quit  S-save  P-pause  R-reconnect",
                                                    (10, 80),  font, 0.45, (180, 180, 180), 1, cv2.LINE_AA)
    # Show truncated source URL
    src = source_label if len(source_label) <= 40 else "…" + source_label[-38:]
    cv2.putText(frame, src,                         (10, 105), font, 0.42, (120, 120, 120), 1, cv2.LINE_AA)

def draw_reconnecting(frame, attempt):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    font = cv2.FONT_HERSHEY_SIMPLEX
    msg  = f"Reconnecting... (attempt {attempt})"
    (tw, th), _ = cv2.getTextSize(msg, font, 0.8, 2)
    cv2.putText(frame, msg,
                ((w - tw) // 2, h // 2),
                font, 0.8, (0, 165, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "Press Q to quit",
                ((w - 180) // 2, h // 2 + 40),
                font, 0.6, (180, 180, 180), 1, cv2.LINE_AA)

# ── Open stream ───────────────────────────────────────────────────────────────
def open_stream(source):
    """Open a cv2.VideoCapture from URL string or integer index."""
    # If source is a digit string, treat as device index
    if isinstance(source, str) and source.isdigit():
        source = int(source)

    cap = cv2.VideoCapture(source)

    if cap.isOpened():
        # For local webcam set a reasonable resolution
        if isinstance(source, int):
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        # Reduce internal buffer so frames are fresh (not stale)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    return cap

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    # ── Determine source ─────────────────────────────────────────────────────
    if len(sys.argv) > 1:
        source = sys.argv[1]
    else:
        print("\nFacial Expression Recognition — IP Camera")
        print("─" * 45)
        print("Examples:")
        print("  Android IP Webcam:  http://192.168.1.5:8080/video")
        print("  RTSP stream:        rtsp://admin:admin@192.168.1.10:554/stream1")
        print("  Local webcam:       0")
        source = input("\nEnter stream URL or device index: ").strip()
        if not source:
            source = "0"

    source_label = str(source)
    print(f"\nConnecting to: {source_label}")

    cap = open_stream(source)
    if not cap.isOpened():
        print(f"ERROR: Cannot connect to '{source_label}'")
        print("Tips:")
        print("  • Make sure your phone/camera is on the same Wi-Fi network")
        print("  • For IP Webcam app: open the app, tap 'Start server', use shown IP")
        print("  • Try the URL in a browser first to confirm it works")
        return

    print("Connected! Press Q to quit, S to save screenshot, P to pause, R to reconnect.")
    os.makedirs('screenshots', exist_ok=True)

    paused       = False
    prev_time    = time.time()
    fps          = 0.0
    last_frame   = None
    screenshot_n = 0
    fail_count   = 0
    reconnect_n  = 0

    TITLE = 'Facial Expression Recognition — IP Cam  [Q=quit  S=save  P=pause  R=reconnect]'

    while True:
        key = cv2.waitKey(1) & 0xFF

        # ── Key handling ──────────────────────────────────────────────────────
        if key == ord('q'):
            break

        elif key == ord('p'):
            paused = not paused
            print("Paused" if paused else "Resumed")

        elif key == ord('s'):
            if last_frame is not None:
                fname = f"screenshots/ipcam_{screenshot_n:04d}.png"
                cv2.imwrite(fname, last_frame)
                screenshot_n += 1
                print(f"Saved: {fname}")

        elif key == ord('r'):
            print("Manual reconnect...")
            cap.release()
            cap = open_stream(source)
            fail_count  = 0
            reconnect_n = 0

        # ── Paused: show last frame ───────────────────────────────────────────
        if paused and last_frame is not None:
            cv2.imshow(TITLE, last_frame)
            continue

        # ── Grab frame ───────────────────────────────────────────────────────
        ret, frame = cap.read()

        if not ret:
            fail_count += 1
            # Show "Reconnecting" overlay on last known frame
            if last_frame is not None:
                disp = last_frame.copy()
                draw_reconnecting(disp, reconnect_n + 1)
                cv2.imshow(TITLE, disp)

            if fail_count >= MAX_FAIL_FRAMES:
                reconnect_n += 1
                print(f"Stream lost — reconnecting (attempt {reconnect_n})...")
                cap.release()
                time.sleep(RECONNECT_DELAY)
                cap = open_stream(source)
                fail_count = 0
            continue

        fail_count = 0  # reset on successful read

        # ── FPS calculation ───────────────────────────────────────────────────
        now       = time.time()
        fps       = 0.9 * fps + 0.1 * (1.0 / max(now - prev_time, 1e-6))
        prev_time = now

        # ── Face detection & expression prediction ────────────────────────────
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
        )

        for (x, y, w, h) in faces:
            face_gray = gray[y:y+h, x:x+w]
            try:
                emotion = predict_face(face_gray)
            except Exception:
                emotion = "error"

            color = EMOTION_COLORS.get(emotion, (200, 200, 200))
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            draw_label(frame, emotion.upper(), x, y, color)

        # ── HUD ───────────────────────────────────────────────────────────────
        draw_hud(frame, fps, paused, len(faces), source_label)

        last_frame = frame.copy()
        cv2.imshow(TITLE, frame)

    cap.release()
    cv2.destroyAllWindows()
    print("Stream closed.")


if __name__ == '__main__':
    main()
