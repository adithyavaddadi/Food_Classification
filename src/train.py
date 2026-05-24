"""
Model training pipeline module.

This module orchestrates the model training process:
1. Loads training and validation datasets.
2. Builds the MobileNetV2 base model with the custom classification head.
3. Sets up robust optimization callbacks (EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard).
4. Executes Phase 1: Feature Extraction (base weights remain frozen).
5. Plots and saves accuracy/loss training history curves.
6. Executes Phase 2: Fine-Tuning (top base layers are unfrozen and trained at a very low learning rate).
7. Saves the optimized model weights to disk.
"""

import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    TensorBoard,
)

from src.dataset import load_datasets
from src.model import build_model, fine_tune_model
from src.config import (
    MODEL_PATH,
    PLOTS_DIR,
    LOG_DIR,
    EPOCHS_PHASE1,
    EPOCHS_PHASE2,
    EARLY_STOPPING_PATIENCE,
    REDUCE_LR_PATIENCE,
    REDUCE_LR_FACTOR,
)


# =========================================================================
# Callbacks Management
# =========================================================================

def get_callbacks():
    """
    Initializes standard training callbacks for optimizer step regulation.

    Returns:
        list: Collection of tf.keras.callbacks.Callback instances.
    """
    # Create directories for outputs if they don't already exist
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            MODEL_PATH,
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=REDUCE_LR_FACTOR,
            patience=REDUCE_LR_PATIENCE,
            verbose=1
        ),
        TensorBoard(
            log_dir=LOG_DIR
        )
    ]

    return callbacks


# =========================================================================
# Visualization Utilities
# =========================================================================

def plot_history(history, filename):
    """
    Generates and saves model training history plots (accuracy and loss curves).

    Args:
        history (tf.keras.callbacks.History): Keras training history object.
        filename (str): Target filename for outputting the plot.
    """
    os.makedirs(PLOTS_DIR, exist_ok=True)

    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 5))

    # Plot Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label="Training Accuracy", color="#3b82f6", linewidth=2)
    plt.plot(epochs_range, val_acc, label="Validation Accuracy", color="#10b981", linewidth=2)
    plt.legend(loc="lower right")
    plt.title("Training & Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(True, linestyle="--", alpha=0.5)

    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label="Training Loss", color="#ef4444", linewidth=2)
    plt.plot(epochs_range, val_loss, label="Validation Loss", color="#f59e0b", linewidth=2)
    plt.legend(loc="upper right")
    plt.title("Training & Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, linestyle="--", alpha=0.5)

    save_path = os.path.join(PLOTS_DIR, filename)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Training history visualization saved to: {save_path}")


# =========================================================================
# Execution Pipeline
# =========================================================================

def train():
    """
    Orchestrates the entire model training pipeline.

    Flows from data ingestion to model assembly, Phase 1 feature extraction
    training, Phase 2 fine-tuning training, and final serialization of model weights.
    """
    print("==================================================")
    print("Initializing Food Classification Training Pipeline")
    print("==================================================")

    # 1. Load prepared splits
    train_ds, val_ds = load_datasets()

    # 2. Build model and load base layers
    model, base_model = build_model()

    # 3. Load callback monitors
    callbacks = get_callbacks()

    # 4. Phase 1: Feature Extraction
    print("\n>>> Starting Phase 1: Feature Extraction (Base Frozen)...")
    history_phase1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_PHASE1,
        callbacks=callbacks
    )
    plot_history(history_phase1, "phase1_training.png")

    # 5. Phase 2: Fine-Tuning
    print("\n>>> Unfreezing top base layers for Phase 2: Fine-Tuning...")
    model = fine_tune_model(model, base_model)

    history_phase2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_PHASE2,
        callbacks=callbacks
    )
    plot_history(history_phase2, "phase2_training.png")

    print("\n==================================================")
    print("Training Pipeline Successfully Completed!")
    print(f"Serialized Optimal Weights: {MODEL_PATH}")
    print("==================================================")


if __name__ == "__main__":
    train()