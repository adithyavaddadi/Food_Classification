"""
train.py

Main training pipeline for the Food Classification model.

Steps:
1. Load dataset
2. Build model
3. Train feature extraction phase
4. Fine tune model
5. Save trained model
6. Plot training results
"""

import os
import matplotlib.pyplot as plt

from src.dataset import load_datasets
from src.model import build_model, fine_tune_model
from src.config import (
    MODEL_PATH,
    PLOTS_DIR,
    EPOCHS_PHASE1,
    EPOCHS_PHASE2,
    EARLY_STOPPING_PATIENCE,
    REDUCE_LR_PATIENCE,
    REDUCE_LR_FACTOR,
)

import tensorflow as tf
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    TensorBoard,
)

from src.config import LOG_DIR


# --------------------------------------------------
# Create Callbacks
# --------------------------------------------------

def get_callbacks():

    callbacks = [

        EarlyStopping(
            monitor="val_loss",
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True
        ),

        ModelCheckpoint(
            MODEL_PATH,
            monitor="val_loss",
            save_best_only=True
        ),

        ReduceLROnPlateau(
            monitor="val_loss",
            factor=REDUCE_LR_FACTOR,
            patience=REDUCE_LR_PATIENCE
        ),

        TensorBoard(
            log_dir=LOG_DIR
        )
    ]

    return callbacks


# --------------------------------------------------
# Plot Training Results
# --------------------------------------------------

def plot_history(history, filename):

    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]

    loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    epochs = range(len(acc))

    plt.figure()

    plt.plot(epochs, acc, label="Training Accuracy")
    plt.plot(epochs, val_acc, label="Validation Accuracy")

    plt.legend()
    plt.title("Training vs Validation Accuracy")

    plt.savefig(os.path.join(PLOTS_DIR, filename))


# --------------------------------------------------
# Training Pipeline
# --------------------------------------------------

def train():

    print("Loading dataset...")

    train_ds, val_ds = load_datasets()

    print("Building model...")

    model, base_model = build_model()

    callbacks = get_callbacks()

    print("Starting Phase 1 Training (Feature Extraction)...")

    history_phase1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_PHASE1,
        callbacks=callbacks
    )

    plot_history(history_phase1, "phase1_training.png")

    print("Starting Phase 2 Training (Fine Tuning)...")

    model = fine_tune_model(model, base_model)

    history_phase2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_PHASE2,
        callbacks=callbacks
    )

    plot_history(history_phase2, "phase2_training.png")

    print("Training complete.")
    print(f"Model saved at: {MODEL_PATH}")


# --------------------------------------------------
# Run Training
# --------------------------------------------------

if __name__ == "__main__":
    train()