"""
Inference Module
Glioma Grade Prediction using trained EVGG-CNN + SVM
Paper-aligned | KaggleHub compatible | GitHub-ready
"""

import os
import numpy as np
import nibabel as nib
import tensorflow as tf
import joblib

from updated_core_pipeline import preprocess_patient_folder, Config
from main_implementation import ModifiedFireflyOptimizer, GliomaFeatureExtractor

# ================================
# LABEL MAP
# ================================
GRADE_MAP = {
    0: "Low-Grade Glioma",
    1: "High-Grade Glioma"
}


# ================================
# PREDICTOR CLASS
# ================================
class GliomaGradePredictor:
    def __init__(self,
                 cnn_model_path='models/evgg_cnn_model.h5',
                 svm_model_path='models/svm_classifier.pkl',
                 scaler_path='models/feature_scaler.pkl'):

        self.cnn = tf.keras.models.load_model(cnn_model_path)
        self.svm = joblib.load(svm_model_path)
        self.scaler = joblib.load(scaler_path)

        self.firefly = ModifiedFireflyOptimizer()
        self.extractor = GliomaFeatureExtractor()

    # ----------------------------
    # SINGLE PATIENT PREDICTION
    # ----------------------------
    def predict_from_patient_folder(self, patient_folder):
        image = preprocess_patient_folder(patient_folder)

        # CNN prediction
        cnn_probs = self.cnn.predict(np.expand_dims(image, axis=0), verbose=0)[0]
        cnn_label = int(np.argmax(cnn_probs))

        # Feature extraction for SVM
        mask = self.firefly.segment(image)
        features = self.extractor.extract(image, mask)
        features = self.scaler.transform([features])

        svm_label = int(self.svm.predict(features)[0])
        svm_conf = np.max(self.svm.predict_proba(features))

        return {
            'cnn_prediction': GRADE_MAP[cnn_label],
            'cnn_confidence': float(np.max(cnn_probs)),
            'svm_prediction': GRADE_MAP[svm_label],
            'svm_confidence': float(svm_conf)
        }

    # ----------------------------
    # BATCH PREDICTION
    # ----------------------------
    def batch_predict(self, patient_folders, save_results=False):
        results = []

        for folder in patient_folders:
            pred = self.predict_from_patient_folder(folder)
            pred['patient_folder'] = folder
            results.append(pred)

        if save_results:
            import pandas as pd
            os.makedirs('results', exist_ok=True)
            df = pd.DataFrame(results)
            df.to_csv('results/batch_predictions.csv', index=False)

        return results


# ================================
# CLI SUPPORT
# ================================
if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Usage: python inference.py <patient_folder>")
        sys.exit(1)

    predictor = GliomaGradePredictor()
    output = predictor.predict_from_patient_folder(sys.argv[1])

    print("\nPrediction Results")
    print("------------------")
    for k, v in output.items():
        print(f"{k}: {v}")
