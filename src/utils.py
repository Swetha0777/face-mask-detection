"""
Utility Functions for Face Mask Detection
Provides preprocessing, visualization, synthetic data generation, and evaluation helpers.
"""

import os
import cv2
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

# Global Class Labels (Alphabetical order matched with ImageDataGenerator / Keras)
CLASS_NAMES = ["with_mask", "without_mask"]
TARGET_IMAGE_SIZE = (224, 224)


def preprocess_face(face_img, target_size=TARGET_IMAGE_SIZE):
    """
    Preprocess a face region for model prediction.
    
    Args:
        face_img (np.ndarray): Face image in BGR or RGB format, or PIL Image.
        target_size (tuple): Target (height, width) for model input.
        
    Returns:
        np.ndarray: Preprocessed image tensor of shape (1, height, width, 3).
    """
    if isinstance(face_img, Image.Image):
        face_img = np.array(face_img)

    # Convert BGR to RGB if image has 3 channels
    if len(face_img.shape) == 3 and face_img.shape[2] == 3:
        # Check if BGR by assuming OpenCV default
        face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    elif len(face_img.shape) == 2:
        face_rgb = cv2.cvtColor(face_img, cv2.COLOR_GRAY2RGB)
    else:
        face_rgb = face_img

    # Resize image to target dimensions
    resized = cv2.resize(face_rgb, target_size, interpolation=cv2.INTER_AREA)

    # Convert to float32 and scale to [-1, 1] for MobileNetV2 architecture
    normalized = (resized.astype(np.float32) / 127.5) - 1.0

    # Add batch dimension
    tensor = np.expand_dims(normalized, axis=0)
    return tensor


def get_label_badge(label_idx_or_str):
    """
    Returns formatted badge string for UI display.
    """
    if isinstance(label_idx_or_str, (int, np.integer)):
        label_str = CLASS_NAMES[label_idx_or_str]
    else:
        label_str = str(label_idx_or_str).lower()

    if "with_mask" in label_str or label_str == "mask":
        return "🟢 MASK", "with_mask", "#10B981"
    else:
        return "🔴 NO MASK", "without_mask", "#EF4444"


def generate_synthetic_sample_data(data_dir, num_samples_per_class=30):
    """
    Generates synthetic sample images for testing train.py when no raw dataset is present.
    Creates colored geometric face drawings with and without face masks.
    """
    print(f"[INFO] Generating {num_samples_per_class} synthetic samples per class in '{data_dir}'...")

    for class_name in CLASS_NAMES:
        class_dir = os.path.join(data_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)

        for i in range(num_samples_per_class):
            img_path = os.path.join(class_dir, f"sample_{i+1:03d}.png")
            if os.path.exists(img_path):
                continue

            # Create RGB image with variable skin tones / backgrounds
            bg_color = (
                np.random.randint(200, 240),
                np.random.randint(200, 240),
                np.random.randint(200, 240)
            )
            img = Image.new("RGB", (200, 200), color=bg_color)
            draw = ImageDraw.Draw(img)

            # Draw face oval
            skin_color = (
                np.random.randint(210, 255),
                np.random.randint(170, 210),
                np.random.randint(140, 180)
            )
            draw.ellipse([40, 30, 160, 170], fill=skin_color, outline=(50, 50, 50), width=2)

            # Draw eyes
            draw.ellipse([70, 70, 85, 85], fill=(30, 30, 30))
            draw.ellipse([115, 70, 130, 85], fill=(30, 30, 30))

            if class_name == "with_mask":
                # Draw protective face mask (blue/cyan/white rectangle covering nose and mouth)
                mask_color = (
                    np.random.randint(30, 80),
                    np.random.randint(120, 220),
                    np.random.randint(200, 255)
                )
                draw.rectangle([55, 95, 145, 155], fill=mask_color, outline=(255, 255, 255), width=2)
                # Mask straps
                draw.line([40, 105, 55, 115], fill=(220, 220, 220), width=2)
                draw.line([160, 105, 145, 115], fill=(220, 220, 220), width=2)
            else:
                # Draw nose and mouth for unmasked face
                draw.line([100, 90, 100, 115], fill=(120, 80, 60), width=2)
                draw.arc([75, 120, 125, 145], start=0, end=180, fill=(180, 50, 50), width=3)

            img.save(img_path)

    print("[INFO] Synthetic dataset generated successfully!")


def plot_training_history(history_dict, save_path=None):
    """
    Plots training and validation accuracy and loss curves.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    acc = history_dict.get('accuracy', history_dict.get('acc', []))
    val_acc = history_dict.get('val_accuracy', history_dict.get('val_acc', []))
    loss = history_dict.get('loss', [])
    val_loss = history_dict.get('val_loss', [])

    epochs = range(1, len(acc) + 1)

    # Accuracy Plot
    axes[0].plot(epochs, acc, 'b-o', label='Training Accuracy', linewidth=2)
    axes[0].plot(epochs, val_acc, 'g-s', label='Validation Accuracy', linewidth=2)
    axes[0].set_title('Training & Validation Accuracy', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.6)

    # Loss Plot
    axes[1].plot(epochs, loss, 'b-o', label='Training Loss', linewidth=2)
    axes[1].plot(epochs, val_loss, 'r-s', label='Validation Loss', linewidth=2)
    axes[1].set_title('Training & Validation Loss', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig


def plot_confusion_matrix(cm, class_names=CLASS_NAMES, save_path=None):
    """
    Plots confusion matrix heat map.
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names, yticklabels=class_names,
           title='Confusion Matrix',
           ylabel='True Class',
           xlabel='Predicted Class')

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations
    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontweight="bold")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig
