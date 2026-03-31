# Human Facial Expression Recognition for CCTV Security Systems

A complete machine learning pipeline that classifies human facial expressions from grayscale face images into **7 emotion categories** for potential integration into CCTV security monitoring systems.

## Problem Statement

Classify 48×48 grayscale face images into one of 7 emotions:

| # | Emotion   |
|---|-----------|
| 0 | Anger     |
| 1 | Contempt  |
| 2 | Disgust   |
| 3 | Fear      |
| 4 | Happy     |
| 5 | Sadness   |
| 6 | Surprise  |

## Project Structure

```
├── assignment.ipynb      # Full ML pipeline notebook
├── dataset/              # Face images organised by emotion
│   ├── anger/
│   ├── contempt/
│   ├── disgust/
│   ├── fear/
│   ├── happy/
│   ├── sadness/
│   └── surprise/
├── saved_model/          # Trained model artefacts
│   ├── svm_model.pkl
│   ├── scaler.pkl
│   ├── pca.pkl
│   └── label_encoder.pkl
├── .gitignore
└── README.md
```

## Pipeline Overview

1. **Data Loading** — images loaded from directory structure, resized to 48×48 grayscale
2. **Exploratory Data Analysis** — class distribution, sample visualisation, pixel statistics
3. **Preprocessing** — flattening, StandardScaler normalisation
4. **Dimensionality Reduction** — PCA (2,304 features → 150 components)
5. **Model Training** — Logistic Regression, SVM, KNN, Random Forest
6. **Hyperparameter Tuning** — GridSearchCV for all models
7. **Evaluation** — Accuracy, Precision, Recall, F1-Score, Confusion Matrices
8. **Deployment Function** — single-image prediction ready for OpenCV integration

## Models Compared

| Model               | Description                                |
|----------------------|--------------------------------------------|
| Logistic Regression  | Linear baseline classifier                 |
| SVM (RBF kernel)     | Non-linear decision boundaries             |
| KNN                  | Distance-based neighbourhood voting        |
| Random Forest        | Ensemble of decision trees                 |

## Requirements

- Python 3.8+
- NumPy
- Pandas
- Matplotlib / Seaborn
- Scikit-learn
- Pillow (PIL)
- Joblib

Install dependencies:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn pillow joblib
```

## Usage

### Run the notebook

Open `assignment.ipynb` in Jupyter Notebook or VS Code and run all cells sequentially.

### Load the saved model

```python
import joblib
import numpy as np
from PIL import Image

model = joblib.load('saved_model/svm_model.pkl')
scaler = joblib.load('saved_model/scaler.pkl')
pca = joblib.load('saved_model/pca.pkl')
le = joblib.load('saved_model/label_encoder.pkl')

# Predict on a new image
img = Image.open('path/to/face.jpg').convert('L').resize((48, 48))
flat = np.array(img).flatten().reshape(1, -1)
scaled = scaler.transform(flat)
reduced = pca.transform(scaled)
prediction = le.inverse_transform(model.predict(reduced))
print(f"Predicted emotion: {prediction[0]}")
```

## References

- Ekman, P. and Friesen, W.V. (1971) 'Constants across cultures in the face and emotion', *Journal of Personality and Social Psychology*, 17(2), pp. 124–129.
- Goodfellow, I.J. et al. (2013) 'Challenges in representation learning: A report on three machine learning contests', *Neural Information Processing*, pp. 117–124.
- Turk, M. and Pentland, A. (1991) 'Eigenfaces for recognition', *Journal of Cognitive Neuroscience*, 3(1), pp. 71–86.
- Pedregosa, F. et al. (2011) 'Scikit-learn: Machine learning in Python', *JMLR*, 12, pp. 2825–2830.
