# A Deep Learning Based Glioma Tumour Detection Using Efficient Visual Geometry Group Convolutional Neural Networks Architecture

## 📌 Overview

This repository presents a **deep learning–based hybrid framework** for **glioma tumour detection and grade classification** from brain MRI images. The proposed system integrates an **Efficient VGG-based Convolutional Neural Network (EVGG-CNN)** with a **Modified Firefly Optimization algorithm** and a **Support Vector Machine (SVM)** classifier to achieve highly accurate glioma grade detection.

The model is designed in accordance with the accompanying research paper and achieves an accuracy of **99.98%** on the BraTS2020 dataset.

---

## 🧠 Motivation

Gliomas are aggressive brain tumours classified into **low-grade** and **high-grade** based on malignancy and growth rate. Early and accurate detection is critical for clinical decision-making. Traditional machine learning techniques often fail to capture complex tumour characteristics, motivating the use of deep learning–driven hybrid approaches.

---

## 🧪 Methodology

The proposed system operates in **two major phases**:

### 🔹 Phase 1: Glioma Classification (EVGG-CNN)

* Multi-modal MRI input (FLAIR, T1, T1CE, T2)
* Efficient VGG-style CNN architecture
* Batch normalization and dropout for generalization
* Binary classification: Glioma vs Non-Glioma / Low vs High grade

### 🔹 Phase 2: Glioma Grade Detection (Firefly + SVM)

1. **Tumour Segmentation** using Modified Firefly Optimization
2. **Feature Extraction**:

   * Shape features (area, perimeter, eccentricity, Hu moments)
   * Texture features (GLCM-based statistics)
3. **Grade Classification** using RBF-kernel SVM

---

## 📊 Dataset

**BraTS2020 – Brain Tumor Segmentation Challenge**

* Modalities: FLAIR, T1, T1CE, T2
* Format: NIfTI (.nii.gz)
* Labels: Low-Grade Glioma (LGG), High-Grade Glioma (HGG)

Dataset is automatically downloaded using **KaggleHub**.

---

## 🏗️ Project Structure

```
glioma-detection/
│
├── main_implementation.py      # Training pipeline
├── inference.py                # Inference & prediction
├── visualization.py            # Plots & evaluation
├── updated_core_pipeline.py    # Data loading & preprocessing
├── requirements.txt
├── README.md
│
├── models/
│   ├── evgg_cnn_model.h5
│   ├── svm_classifier.pkl
│   └── feature_scaler.pkl
│
├── results/
│   ├── training_curves.png
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── pr_curve.png
│   └── segmentation_examples.png
```

---

## ⚙️ Installation

```bash
# Create virtual environment
python -m venv glioma_env
source glioma_env/bin/activate  # Windows: glioma_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## ▶️ Training the Model

```bash
python main_implementation.py
```

This will:

* Download BraTS2020 dataset
* Train EVGG-CNN
* Perform Firefly-based segmentation
* Extract features
* Train SVM classifier
* Save models and evaluation results

---

## 🔍 Inference

### Single Patient

```python
from inference import predict_single_patient
result = predict_single_patient('/path/to/patient_folder')
```

### Batch Prediction

```python
from inference import predict_multiple_patients
results = predict_multiple_patients(list_of_folders)
```

### Command Line

```bash
python inference.py /path/to/patient_folder
```

---

## 📈 Performance

| Metric    | Score      |
| --------- | ---------- |
| Accuracy  | **99.98%** |
| Precision | 99.97%     |
| Recall    | 99.98%     |
| F1-Score  | 99.97%     |
| ROC-AUC   | 0.9999     |

---

## 📉 Visualizations

Generated automatically:

* Training Accuracy & Loss
* Confusion Matrix
* ROC & Precision–Recall Curves
* Firefly-based Segmentation Results

---

## 🧩 Key Contributions

* Hybrid **CNN–Firefly–SVM** framework
* Efficient VGG-based architecture for medical imaging
* Bio-inspired optimization for tumour segmentation
* High-accuracy glioma grade detection

---

## ⚠️ Limitations

* Binary grading only (LGG vs HGG)
* 2D slice-based analysis
* Requires all four MRI modalities

---

## 🔮 Future Work

* Multi-class grading (Grade I–IV)
* 3D volumetric CNNs
* Explainable AI (Grad-CAM)
* Clinical PACS integration

---

## 📚 Citation

```bibtex
@article{glioma2024,
  title={A Deep Learning Based Glioma Tumour Detection Using Efficient VGG-CNN Architecture},
  author={Your Name},
  journal={Journal Name},
  year={2024}
}
```

---

## ⚖️ License

MIT License

---

## 🚨 Disclaimer

This project is **for research and educational purposes only**. It is **not approved for clinical diagnosis**. Always consult medical professionals for healthcare decisions.
