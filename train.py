"""
Training Pipeline for Face Mask Detection Model
Trains MobileNetV2 or custom CNN model using dataset in data/ with_mask and without_mask.
"""

import os
import json
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from src.model import build_mask_detector
from src.utils import generate_synthetic_sample_data, plot_training_history, CLASS_NAMES


def train_model(data_dir="data", model_dir="model", epochs=15, batch_size=32, lr=1e-4):
    """
    Executes model training pipeline.
    
    Args:
        data_dir (str): Root directory containing with_mask and without_mask folders.
        model_dir (str): Directory where trained model and metrics will be saved.
        epochs (int): Maximum number of training epochs.
        batch_size (int): Training batch size.
        lr (float): Initial learning rate.
    """
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    with_mask_dir = os.path.join(data_dir, "with_mask")
    without_mask_dir = os.path.join(data_dir, "without_mask")

    # Check if dataset directories exist and have images
    has_with_mask = os.path.exists(with_mask_dir) and len(os.listdir(with_mask_dir)) > 0
    has_without_mask = os.path.exists(without_mask_dir) and len(os.listdir(without_mask_dir)) > 0

    if not (has_with_mask and has_without_mask):
        print("[NOTICE] Raw image dataset not detected or incomplete in data/ directory.")
        print("[NOTICE] Generating synthetic sample images so training pipeline can execute...")
        generate_synthetic_sample_data(data_dir, num_samples_per_class=40)

    print(f"[INFO] Initializing ImageDataGenerator from '{data_dir}'...")

    # Define Data Augmentation for training set
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 127.5,
        preprocessing_function=lambda x: x - 1.0,  # Scaled to [-1, 1] for MobileNetV2
        rotation_range=20,
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest",
        validation_split=0.20  # 80% train, 20% validation
    )

    print("[INFO] Loading training dataset...")
    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode="categorical",
        classes=CLASS_NAMES,
        subset="training",
        shuffle=True
    )

    print("[INFO] Loading validation dataset...")
    val_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode="categorical",
        classes=CLASS_NAMES,
        subset="validation",
        shuffle=False
    )

    print(f"[INFO] Classes identified: {train_generator.class_indices}")

    # Build model architecture
    model = build_mask_detector(
        input_shape=(224, 224, 3),
        num_classes=len(CLASS_NAMES),
        use_transfer_learning=True,
        learning_rate=lr
    )

    output_model_path = os.path.join(model_dir, "face_mask_detector.keras")

    # Callbacks configuration
    callbacks = [
        ModelCheckpoint(
            filepath=output_model_path,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        )
    ]

    print(f"[INFO] Starting training for {epochs} epochs...")
    history = model.fit(
        train_generator,
        steps_per_epoch=max(1, train_generator.samples // batch_size),
        validation_data=val_generator,
        validation_steps=max(1, val_generator.samples // batch_size),
        epochs=epochs,
        callbacks=callbacks
    )

    print(f"[SUCCESS] Best model successfully saved to '{output_model_path}'!")

    # Save metrics metadata
    history_dict = history.history
    plot_path = os.path.join(model_dir, "training_history.png")
    plot_training_history(history_dict, save_path=plot_path)

    final_train_acc = float(history_dict['accuracy'][-1])
    final_val_acc = float(history_dict['val_accuracy'][-1])
    final_train_loss = float(history_dict['loss'][-1])
    final_val_loss = float(history_dict['val_loss'][-1])

    metrics_summary = {
        "epochs_completed": len(history_dict['accuracy']),
        "train_accuracy": round(final_train_acc, 4),
        "val_accuracy": round(final_val_acc, 4),
        "train_loss": round(final_train_loss, 4),
        "val_loss": round(final_val_loss, 4),
        "classes": CLASS_NAMES,
        "input_shape": [224, 224, 3],
        "model_architecture": "MobileNetV2 Transfer Learning"
    }

    metrics_json_path = os.path.join(model_dir, "metrics.json")
    with open(metrics_json_path, "w") as f:
        json.dump(metrics_summary, f, indent=4)

    print("\n" + "=" * 50)
    print("TRAINING SUMMARY")
    print("=" * 50)
    print(f"Final Train Accuracy : {final_train_acc * 100:.2f}%")
    print(f"Final Val Accuracy   : {final_val_acc * 100:.2f}%")
    print(f"Final Train Loss     : {final_train_loss:.4f}")
    print(f"Final Val Loss       : {final_val_loss:.4f}")
    print(f"Saved Model Path     : {output_model_path}")
    print(f"Saved Metrics Path   : {metrics_json_path}")
    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Face Mask Detection Model")
    parser.add_argument("--data_dir", type=str, default="data", help="Path to data directory")
    parser.add_argument("--model_dir", type=str, default="model", help="Path to model directory")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")

    args = parser.parse_args()
    train_model(
        data_dir=args.data_dir,
        model_dir=args.model_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr
    )
