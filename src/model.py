"""
Model Architecture Definition for Face Mask Detection
Provides lightweight MobileNetV2 Transfer Learning and Custom CNN options.
"""

import os
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input, Conv2D, MaxPooling2D
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.optimizers import Adam


def build_mask_detector(input_shape=(224, 224, 3), num_classes=2, use_transfer_learning=True, learning_rate=1e-4):
    """
    Constructs and compiles the Face Mask Detection deep learning model.
    
    Args:
        input_shape (tuple): Shape of input images (Height, Width, Channels).
        num_classes (int): Number of target classes (default 2: with_mask, without_mask).
        use_transfer_learning (bool): Use pre-trained MobileNetV2 base model if True, else custom CNN.
        learning_rate (float): Learning rate for Adam optimizer.
        
    Returns:
        tf.keras.Model: Compiled Keras model instance.
    """
    if use_transfer_learning:
        print("[INFO] Building MobileNetV2 Transfer Learning Architecture...")
        # Load pre-trained MobileNetV2 without top classification head
        base_model = MobileNetV2(
            weights="imagenet",
            include_top=False,
            input_tensor=Input(shape=input_shape)
        )

        # Freeze base model layers so their learned representations aren't destroyed
        for layer in base_model.layers:
            layer.trainable = False

        # Construct classification top head
        head_model = base_model.output
        head_model = AveragePooling2D(pool_size=(7, 7))(head_model)
        head_model = Flatten(name="flatten")(head_model)
        head_model = Dense(128, activation="relu")(head_model)
        head_model = Dropout(0.5)(head_model)
        head_model = Dense(num_classes, activation="softmax", name="output")(head_model)

        model = Model(inputs=base_model.input, outputs=head_model)
    else:
        print("[INFO] Building Custom CNN Architecture...")
        model = Sequential([
            Input(shape=input_shape),
            Conv2D(32, (3, 3), activation='relu', padding='same'),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            MaxPooling2D((2, 2)),
            Conv2D(128, (3, 3), activation='relu', padding='same'),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(num_classes, activation='softmax')
        ])

    # Compile the model
    opt = Adam(learning_rate=learning_rate)
    model.compile(
        loss="categorical_crossentropy",
        optimizer=opt,
        metrics=["accuracy"]
    )
    
    return model


def load_mask_detector(model_path):
    """
    Loads a saved Keras model from file.
    
    Args:
        model_path (str): Path to the saved model file (.keras or .h5).
        
    Returns:
        tf.keras.Model or None: Loaded model object if successful, else None.
    """
    if not os.path.exists(model_path):
        print(f"[WARNING] Model file not found at path: '{model_path}'")
        return None
    
    try:
        model = load_model(model_path)
        print(f"[SUCCESS] Loaded model successfully from '{model_path}'")
        return model
    except Exception as e:
        print(f"[ERROR] Failed to load model: {str(e)}")
        return None
