import joblib
import numpy as np
import cv2
import sys
import os
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# Load pipeline (SVM RBF - best F1 score: 0.9847)
model = joblib.load('saved_model/best_model.pkl')
scaler = joblib.load('saved_model/scaler.pkl')
pca = joblib.load('saved_model/pca.pkl')
le = joblib.load('saved_model/label_encoder.pkl')

# Load face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def predict(image_path):
    if not os.path.exists(image_path):
        # Try common extensions if file not found
        base, ext = os.path.splitext(image_path)
        for try_ext in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']:
            if os.path.exists(base + try_ext):
                image_path = base + try_ext
                break
        else:
            raise FileNotFoundError(f"Image not found: {image_path}")

    # Step 1: Read image (any size, any color)
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    h_orig, w_orig = img.shape[:2]
    print(f"Original image: {w_orig}x{h_orig}, channels={img.shape[2] if len(img.shape)==3 else 1}")

    # Step 2: Convert to grayscale (black & white)
    if len(img.shape) == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        print("Converted to grayscale")
    else:
        img_gray = img
        print("Already grayscale")

    # Step 3: Detect face
    faces = face_cascade.detectMultiScale(img_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) > 0:
        # Use the largest face detected
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        face_crop = img_gray[y:y+h, x:x+w]
        print(f"Face detected at ({x},{y}) size {w}x{h} -> cropped")
    else:
        face_crop = img_gray
        print("No face detected, using full image")

    # Step 4: Resize to 48x48 (same as training data)
    face_resized = cv2.resize(face_crop, (48, 48))

    # Step 5: Histogram equalization (normalize lighting/contrast)
    face_resized = cv2.equalizeHist(face_resized)
    print(f"Resized to 48x48 grayscale + histogram equalized -> ready for model")

    # Step 6: Flatten, scale, PCA, predict
    img_array = face_resized.flatten().reshape(1, -1).astype(np.float64)
    img_scaled = scaler.transform(img_array)
    img_pca = pca.transform(img_scaled)
    prediction = model.predict(img_pca)
    return le.inverse_transform(prediction)[0]

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path>")
        print("Example: python predict.py TryImages/6.jpg")
        sys.exit(1)
    path = sys.argv[1]
    result = predict(path)
    print(f"Predicted emotion: {result}")
