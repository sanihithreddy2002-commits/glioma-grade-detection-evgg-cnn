"""
Main Training Pipeline
Glioma Grade Detection using EVGG-CNN + Firefly Optimizer + SVM
KaggleHub-integrated, Reproducible, Paper-aligned
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import joblib

from updated_core_pipeline import build_dataset, Config

# ================================
# EVGG-CNN MODEL
# ================================
class EVGG_CNN:
    @staticmethod
    def build(input_shape=(128, 128, 4)):
        model = models.Sequential()

        for filters in [32, 64, 128, 256]:
            model.add(layers.Conv2D(filters, (3, 3), padding='same', activation='relu'))
            model.add(layers.BatchNormalization())
            model.add(layers.MaxPooling2D((2, 2)))

        model.add(layers.GlobalAveragePooling2D())
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(2, activation='softmax'))

        model.compile(
            optimizer=optimizers.Adam(learning_rate=1e-4),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model


# ================================
# FIRELY OPTIMIZER (SIMPLIFIED)
# ================================
class ModifiedFireflyOptimizer:
    def segment(self, image):
        gray = image[:, :, 0]
        thresh = np.mean(gray)
        return (gray > thresh).astype(np.uint8)


# ================================
# FEATURE EXTRACTION
# ================================
class GliomaFeatureExtractor:
    def extract(self, image, mask):
        area = np.sum(mask)
        mean_intensity = np.mean(image[:, :, 0][mask == 1]) if area > 0 else 0
        return np.array([area, mean_intensity])


# ================================
# MAIN TRAINING FUNCTION
# ================================
def main():
    print("Loading dataset...")
    X_train, X_val, X_test, y_train, y_val, y_test = build_dataset()

    print("Training EVGG-CNN...")
    model = EVGG_CNN.build()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=20,
        batch_size=8,
        verbose=1
    )

    os.makedirs('models', exist_ok=True)
    model.save('models/evgg_cnn_model.h5')

    print("Extracting features for SVM...")
    firefly = ModifiedFireflyOptimizer()
    extractor = GliomaFeatureExtractor()

    features, labels = [], []
    for img, label in zip(X_train, y_train):
        mask = firefly.segment(img)
        feat = extractor.extract(img, mask)
        features.append(feat)
        labels.append(label)

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    svm = SVC(kernel='rbf', C=10, probability=True)
    svm.fit(features_scaled, labels)

    joblib.dump(svm, 'models/svm_classifier.pkl')
    joblib.dump(scaler, 'models/feature_scaler.pkl')

    print("Evaluating SVM...")
    test_features, test_labels = [], []
    for img, label in zip(X_test, y_test):
        mask = firefly.segment(img)
        feat = extractor.extract(img, mask)
        test_features.append(feat)
        test_labels.append(label)

    test_features = scaler.transform(test_features)
    preds = svm.predict(test_features)
    acc = accuracy_score(test_labels, preds)

    print(f"Final Test Accuracy: {acc:.4f}")
    return model, svm, scaler, history


if __name__ == '__main__':
    main()
