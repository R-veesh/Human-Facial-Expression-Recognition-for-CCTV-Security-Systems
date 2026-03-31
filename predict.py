import joblib
import numpy as np
from PIL import Image
import sys

# Load pipeline
model = joblib.load('saved_model/svm_model.pkl')
scaler = joblib.load('saved_model/scaler.pkl')
pca = joblib.load('saved_model/pca.pkl')
le = joblib.load('saved_model/label_encoder.pkl')

def predict(image_path):
    img = Image.open(image_path).convert('L').resize((48, 48))
    img_array = np.array(img).flatten().reshape(1, -1).astype(np.float64)
    img_scaled = scaler.transform(img_array)
    img_pca = pca.transform(img_scaled)
    prediction = model.predict(img_pca)
    return le.inverse_transform(prediction)[0]

if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else 'images/Face expression_images/validation/happy/PrivateTest_88305.jpg'
    result = predict(path)
    print(f"Predicted emotion: {result}")
