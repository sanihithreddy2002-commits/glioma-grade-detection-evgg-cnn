"""
Visualization Module
Glioma Detection & Grade Classification
Publication-quality plots | Paper-aligned | GitHub-ready
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve

# ================================
# TRAINING HISTORY PLOTS
# ================================
def plot_training_history(history, save_dir='results'):
    os.makedirs(save_dir, exist_ok=True)

    plt.figure()
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training Accuracy Curve')
    plt.savefig(f'{save_dir}/training_accuracy.png', dpi=300)
    plt.close()

    plt.figure()
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training Loss Curve')
    plt.savefig(f'{save_dir}/training_loss.png', dpi=300)
    plt.close()


# ================================
# CONFUSION MATRIX
# ================================
def plot_confusion_matrix(y_true, y_pred, save_dir='results'):
    os.makedirs(save_dir, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)

    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.savefig(f'{save_dir}/confusion_matrix.png', dpi=300)
    plt.close()


# ================================
# ROC CURVE
# ================================
def plot_roc_curve(y_true, y_proba, save_dir='results'):
    os.makedirs(save_dir, exist_ok=True)
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.4f}')
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.savefig(f'{save_dir}/roc_curve.png', dpi=300)
    plt.close()


# ================================
# PRECISION-RECALL CURVE
# ================================
def plot_precision_recall_curve(y_true, y_proba, save_dir='results'):
    os.makedirs(save_dir, exist_ok=True)
    precision, recall, _ = precision_recall_curve(y_true, y_proba)

    plt.figure()
    plt.plot(recall, precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.savefig(f'{save_dir}/pr_curve.png', dpi=300)
    plt.close()


# ================================
# SEGMENTATION VISUALIZATION
# ================================
def plot_segmentation(image, mask, save_dir='results', idx=0):
    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(image[:, :, 0], cmap='gray')
    plt.title('Original MRI')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(mask, cmap='hot')
    plt.title('Firefly Segmentation')
    plt.axis('off')

    plt.savefig(f'{save_dir}/segmentation_{idx}.png', dpi=300)
    plt.close()


# ================================
# MASTER FUNCTION
# ================================
def visualize_all(history, y_true, y_pred, y_proba, image=None, mask=None):
    plot_training_history(history)
    plot_confusion_matrix(y_true, y_pred)
    plot_roc_curve(y_true, y_proba)
    plot_precision_recall_curve(y_true, y_proba)

    if image is not None and mask is not None:
        plot_segmentation(image, mask)
