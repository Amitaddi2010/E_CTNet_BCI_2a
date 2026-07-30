import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
import torch

# Publication styling settings
plt.rcParams['font.sans-serif'] = 'Helvetica'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.8


def plot_confusion_matrix(y_true, y_pred, class_names, title="Normalized Confusion Matrix", save_path=None, dpi=300):
    """
    Plots a high-resolution 300 DPI normalized confusion matrix heatmap.
    """
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    cm_norm = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-8)

    fig, ax = plt.subplots(figsize=(6, 5), dpi=dpi)
    sns.heatmap(
        cm_norm, annot=True, fmt='.2f', cmap='Blues',
        xticklabels=class_names, yticklabels=class_names,
        cbar=True, square=True, linewidths=0.5,
        annot_kws={"size": 11, "weight": "bold"}, ax=ax
    )

    ax.set_title(title, fontsize=12, fontweight='bold', pad=12)
    ax.set_xlabel('Predicted Label', fontsize=11, labelpad=8)
    ax.set_ylabel('True Label', fontsize=11, labelpad=8)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"[Visualization] Saved confusion matrix figure: {save_path}")
    
    plt.close(fig)


def plot_tsne_features(model, dataloader, class_names, device, title="t-SNE Latent Feature Manifold", save_path=None, dpi=300):
    """
    Extracts bottleneck features from trained deep learning model and raw EEG signals,
    computes 2D t-SNE embedding, and plots publication scatter graphics.
    """
    model.eval()
    all_features = []
    all_labels = []

    with torch.no_grad():
        for x, _, y_idx in dataloader:
            x = x.to(device)
            feats = model.extract_features(x)
            all_features.append(feats.cpu().numpy())
            all_labels.append(y_idx.numpy())

    X_feat = np.concatenate(all_features, axis=0)
    y_feat = np.concatenate(all_labels, axis=0)

    # Compute t-SNE projection
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, init='pca', learning_rate='auto')
    X_embedded = tsne.fit_transform(X_feat)

    fig, ax = plt.subplots(figsize=(7, 6), dpi=dpi)
    palette = sns.color_palette("deep", len(class_names))
    markers = ['o', 's', '^', 'D']

    for cls_idx, cls_name in enumerate(class_names):
        mask = (y_feat == cls_idx)
        ax.scatter(
            X_embedded[mask, 0], X_embedded[mask, 1],
            label=cls_name, color=palette[cls_idx], marker=markers[cls_idx],
            alpha=0.85, edgecolors='k', linewidths=0.5, s=50
        )

    ax.set_title(title, fontsize=12, fontweight='bold', pad=12)
    ax.set_xlabel('t-SNE Dimension 1', fontsize=11)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=11)
    ax.legend(title="Motor Imagery Class", frameon=True, facecolor='white', framealpha=0.9, loc='best')
    ax.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"[Visualization] Saved t-SNE latent feature graphic: {save_path}")

    plt.close(fig)


def plot_training_curves(history, subject_id, save_path=None, dpi=300):
    """
    Plots high-resolution training & testing loss and accuracy curves over fine-tuning epochs.
    """
    epochs = range(1, len(history['train_loss']) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=dpi)

    # Loss Curve
    ax1.plot(epochs, history['train_loss'], label='Train Loss', color='#1f77b4', linewidth=2)
    ax1.plot(epochs, history['test_loss'], label='Test Loss (Session 2)', color='#ff7f0e', linestyle='--', linewidth=2)
    ax1.set_title(f'Subject {subject_id} - Cross-Entropy Loss', fontsize=11, fontweight='bold')
    ax1.set_xlabel('Epochs', fontsize=10)
    ax1.set_ylabel('Loss', fontsize=10)
    ax1.legend(frameon=True)
    ax1.grid(True, linestyle='--', alpha=0.3)

    # Accuracy Curve
    ax2.plot(epochs, [a * 100 for a in history['train_acc']], label='Train Acc', color='#2ca02c', linewidth=2)
    ax2.plot(epochs, [a * 100 for a in history['test_acc']], label='Test Acc (Session 2)', color='#d62728', linestyle='--', linewidth=2)
    ax2.set_title(f'Subject {subject_id} - Accuracy (%)', fontsize=11, fontweight='bold')
    ax2.set_xlabel('Epochs', fontsize=10)
    ax2.set_ylabel('Accuracy (%)', fontsize=10)
    ax2.legend(frameon=True)
    ax2.grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"[Visualization] Saved training curves graphic: {save_path}")

    plt.close(fig)
