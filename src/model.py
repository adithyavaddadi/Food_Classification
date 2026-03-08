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
    Load MobileNetV2 pretrained on ImageNet without the top layer.
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
    Build the full classification model with custom head.
    """

    base_model = build_base_model()

    # Freeze base model for Phase 1 training
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3))

    x = base_model(inputs, training=False)

    x = layers.GlobalAveragePooling2D()(x)

    x = layers.BatchNormalization()(x)

    x = layers.Dropout(DROPOUT_RATE_1)(x)

    x = layers.Dense(DENSE_UNITS, activation="relu")(x)

    x = layers.Dropout(DROPOUT_RATE_2)(x)

    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = models.Model(inputs, outputs)

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

    base_model.trainable = True

    # Freeze all layers except top N layers
    for layer in base_model.layers[:-FINE_TUNE_LAYERS]:
        layer.trainable = False

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE_PHASE2),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model