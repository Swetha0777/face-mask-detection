"""
Evaluation Script for Trained Face Mask Detection Model
Calculates Accuracy, Precision, Recall, F1-Score, and plots Confusion Matrix.
"""

import os
import json
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support

from src.model import load_mask_detector
from src.utils import generate_synthetic_sample_data, plot_confusion_matrix, CLASS_NAMES


def evaluate_model(model_path="model/face_mask_detector.keras", data_dir="data", output_dir="model"):
    """
    Evaluates saved model performance on test/validation dataset.
    """
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(model_path):
        print(f"[ERROR] Model file not found at '{model_path}'. Please run 'python train.py' first.")
        return

    # Check data directory
    with_mask_dir = os.path.join(data_dir, "with_mask")
    without_mask_dir = os.path.join(data_dir, "without_mask")

    if not (os.path.exists(with_mask_dir) and os.path.exists(without_mask_dir)):
        print("[NOTICE] Data directory empty or missing. Generating synthetic sample dataset...")
        generate_synthetic_sample_data(data_dir, num_samples_per_class=30)

    # Load Model
    print(f"[INFO] Loading model from '{model_path}'...")
    model = load_mask_detector(model_path)
    if model is None:
        print("[ERROR] Failed to load model for evaluation.")
        return

    # Prepare Data Generator for Validation/Test evaluation
    test_datagen = ImageDataGenerator(
        rescale=1.0 / 127.5,
        preprocessing_function=lambda x: x - 1.0
    )

    test_generator = test_datagen.flow_from_directory(
        data_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode="categorical",
        classes=CLASS_NAMES,
        shuffle=False
    )

    if test_generator.samples == 0:
        print("[ERROR] No evaluation samples found in data directory.")
        return

    print(f"[INFO] Running predictions on {test_generator.samples} dataset images...")
    predictions = model.predict(test_generator, verbose=1)
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_generator.classes

    # Metrics calculation
    accuracy = float(accuracy_score(y_true, y_pred))
    precision, recall, f1_score, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    cm = confusion_matrix(y_true, y_pred)
    report_dict = classification_report(
        y_true, y_pred, target_names=CLASS_NAMES, output_dict=True, zero_division=0
    )

    # Plot and save confusion matrix
    cm_plot_path = os.path.join(output_dir, "confusion_matrix.png")
    plot_confusion_matrix(cm, class_names=CLASS_NAMES, save_path=cm_plot_path)

    # Save summary report
    eval_results = {
        "accuracy": round(accuracy, 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1_score": round(float(f1_score), 4),
        "confusion_matrix": cm.tolist(),
        "classification_report": report_dict
    }

    eval_json_path = os.path.join(output_dir, "evaluation_report.json")
    with open(eval_json_path, "w") as f:
        json.dump(eval_results, f, indent=4)

    print("\n" + "=" * 50)
    print("MODEL EVALUATION RESULTS")
    print("=" * 50)
    print(f"Overall Accuracy : {accuracy * 100:.2f}%")
    print(f"Precision        : {precision * 100:.2f}%")
    print(f"Recall           : {recall * 100:.2f}%")
    print(f"F1-Score         : {f1_score * 100:.2f}%")
    print("-" * 50)
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, zero_division=0))
    print(f"Confusion Matrix Plot saved to : {cm_plot_path}")
    print(f"Evaluation JSON saved to       : {eval_json_path}")
    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Face Mask Detection Model")
    parser.add_argument("--model_path", type=str, default="model/face_mask_detector.keras")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--output_dir", type=str, default="model")

    args = parser.parse_args()
    evaluate_model(args.model_path, args.data_dir, args.output_dir)
