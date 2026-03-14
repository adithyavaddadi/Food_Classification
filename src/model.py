"""
model.py

Defines the deep learning model used for food classification.

Uses Transfer Learning with MobileNetV2 pretrained on ImageNet.
Adds a custom classification head for the Food101 subset.
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
    FINE_TUNE_LAYERS
)


# --------------------------------------------------
# Build Base Model
# --------------------------------------------------

def build_base_model():
    """
    Load MobileNetV2 pretrained on ImageNet.
    The top classification layer is removed.
    """

    base_model = MobileNetV2(
        input_shape=(*IMAGE_SIZE, 3),
        include_top=False,
        weights="imagenet"
    )

    return base_model


# --------------------------------------------------
# Build Full Model
# --------------------------------------------------

def build_model():
    """
    Build the full classification model with a custom head.
    """

    # Load pretrained base model
    base_model = build_base_model()

    # Freeze base model during initial training
    base_model.trainable = False

    # Input layer
    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3))

    # Feature extraction
    x = base_model(inputs, training=False)

    # Global feature pooling
    x = layers.GlobalAveragePooling2D()(x)

    # Stabilize training
    x = layers.BatchNormalization()(x)

    # Regularization
    x = layers.Dropout(DROPOUT_RATE_1)(x)

    # Dense layer
    x = layers.Dense(DENSE_UNITS, activation="relu")(x)

    # Additional dropout
    x = layers.Dropout(DROPOUT_RATE_2)(x)

    # Output classification layer
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    # Build model
    model = models.Model(inputs, outputs, name="FoodClassifier_MobileNetV2")

    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE_PHASE1),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model, base_model


# --------------------------------------------------
# Fine Tuning
# --------------------------------------------------

def fine_tune_model(model, base_model):
    """
    Unfreeze top layers of the base model for fine tuning.
    """

    # Unfreeze base model
    base_model.trainable = True

    # Freeze lower layers
    for layer in base_model.layers[:-FINE_TUNE_LAYERS]:
        layer.trainable = False

    # Recompile model with lower learning rate
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE_PHASE2),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model