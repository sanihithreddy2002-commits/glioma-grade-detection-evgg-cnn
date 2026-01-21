"""
Glioma Grade Detection - Updated Core Pipeline
Dataset handling integrated with KaggleHub
Compatible with BraTS2020 / BraTS2019
"""

import os
import numpy as np
import kagglehub
import nibabel as nib
from scipy.ndimage import zoom
from sklearn.model_selection import train_test_split

# ================================
# CONFIGURATION
# ================================
class Config:
    IMG_SIZE = (128, 128)
    MODALITIES = ['flair', 't1', 't1ce', 't2']
    TEST_SIZE = 0.15
    VAL_SIZE = 0.15
    RANDOM_STATE = 42

    @staticmethod
    def get_dataset_path():
        root = kagglehub.dataset_download("awsaf49/brats2020-training-data")
        return os.path.join(root, "MICCAI_BraTS2020_TrainingData")

# ================================
# DATA LOADING UTILITIES
# ================================
def load_nifti(path):
    return nib.load(path).get_fdata()


def normalize(img):
    if img.max() > 0:
        img = (img - img.min()) / (img.max() - img.min())
    return img


def resize(img, size):
    factors = (size[0] / img.shape[0], size[1] / img.shape[1])
    return zoom(img, factors, order=1)


# ================================
# SAMPLE GENERATOR
# ================================
def load_patient_sample(patient_dir):
    volumes = []
    for mod in Config.MODALITIES:
        file = [f for f in os.listdir(patient_dir) if mod in f.lower()][0]
        vol = load_nifti(os.path.join(patient_dir, file))
        mid = vol.shape[2] // 2
        slice_img = normalize(resize(vol[:, :, mid], Config.IMG_SIZE))
        volumes.append(slice_img)
    return np.stack(volumes, axis=-1)


# ================================
# DATASET BUILDER
# ================================
def build_dataset():
    dataset_path = Config.get_dataset_path()
    X, y = [], []

    for patient in os.listdir(dataset_path):
        patient_dir = os.path.join(dataset_path, patient)
        if not os.path.isdir(patient_dir):
            continue

        label = 0 if 'LGG' in patient else 1
        try:
            img = load_patient_sample(patient_dir)
            X.append(img)
            y.append(label)
        except Exception:
            continue

    X = np.array(X)
    y = np.array(y)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=Config.TEST_SIZE + Config.VAL_SIZE,
        random_state=Config.RANDOM_STATE, stratify=y
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=Config.TEST_SIZE / (Config.TEST_SIZE + Config.VAL_SIZE),
        random_state=Config.RANDOM_STATE, stratify=y_temp
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


if __name__ == "__main__":
    print("Downloading dataset and building dataset splits...")
    X_train, X_val, X_test, y_train, y_val, y_test = build_dataset()
    print("Train:", X_train.shape, "Val:", X_val.shape, "Test:", X_test.shape)
