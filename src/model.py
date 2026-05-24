"""
Model architecture module for the Food Classification & Nutrition AI model.

This module sets up:
1. MobileNetV2 pretrained on ImageNet as a feature extraction base.
2. Custom classification head tailored to our Food101 subset (10 classes).
3. Fine-tuning capability to unfreeze and retrain top layers of MobileNetV2.
"""

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam

from src.config import (
    IMAGE_SIZE,
    NUM_CLASSES,
    DENSE_UNITS,
    DROPOUT_RATE_1,
    DROPOUT_RATE_2,
    LEARNING_RATE_PHASE1,
    LEARNING_RATE_PHASE2,
    FINE_TUNE_LAYERS,
)

# =========================================================================
# Model Creation Functions
# =========================================================================

def build_base_model():
    """
    Instantiates the MobileNetV2 base model pretrained on ImageNet.

    The final fully-connected classification layer is excluded so it can be 
    replaced by our custom classification head.

    Returns:
        tf.keras.Model: Instantiated MobileNetV2 model.
    """
    base_model = MobileNetV2(
        input_shape=(*IMAGE_SIZE, 3),
        include_top=False,
        weights="imagenet"
    )
    return base_model


def build_model():
    """
    Constructs and compiles the full classification network.

    Combines the pretrained MobileNetV2 base model (initially frozen to keep
    pretrained weights intact) with a custom classification head including
    pooling, batch normalization, dropout, dense, and softmax layers.

    Returns:
        tuple: (model, base_model)
            - model (tf.keras.Model): Compiled complete model ready for Phase 1.
            - base_model (tf.keras.Model): Pretrained MobileNetV2 base layer.
    """
    # Load pretrained base model
    base_model = build_base_model()

    # Freeze base model during initial training phase
    base_model.trainable = False

    # Input layer
    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3))

    # Feature extraction via base
    x = base_model(inputs, training=False)

    # Custom classification head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(DROPOUT_RATE_1)(x)
    x = layers.Dense(DENSE_UNITS, activation="relu")(x)
    x = layers.Dropout(DROPOUT_RATE_2)(x)
    
    # Output class probabilities
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    # Assemble Keras Model
    model = models.Model(inputs, outputs, name="FoodClassifier_MobileNetV2")

    # Compile with Adam optimizer for categorical crossentropy
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE_PHASE1),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model, base_model


# =========================================================================
# Fine-Tuning Setup
# =========================================================================

def fine_tune_model(model, base_model):
    """
    Prepares the model for Phase 2 fine-tuning.

    Unfreezes the base MobileNetV2 model and refreezes all layers except for 
    the top `FINE_TUNE_LAYERS` layers. Recompiles the model using a substantially
    lower learning rate to avoid destructive weights update.

    Args:
        model (tf.keras.Model): Complete assembled model.
        base_model (tf.keras.Model): MobileNetV2 base inside the model.

    Returns:
        tf.keras.Model: Recompiled model ready for fine-tuning.
    """
    # Unfreeze the base model
    base_model.trainable = True

    # Refreeze lower layers and leave only top layers trainable
    for layer in base_model.layers[:-FINE_TUNE_LAYERS]:
        layer.trainable = False

    # Recompile with a significantly lower learning rate
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE_PHASE2),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model