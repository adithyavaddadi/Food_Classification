"""
Model evaluation module.

This module processes quantitative validation metrics for the trained model:
1. Loads the saved serialization checkpoint.
2. Runs inference across validation datasets.
3. Computes and logs overall Categorical Crossentropy Loss and Accuracy.
4. Synthesizes a scikit-learn classification report (Precision, Recall, F1-score per class).
5. Visualizes a confusion matrix and exports it as a PNG plot.
"""

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

from src.dataset import load_datasets
from src.config import MODEL_PATH, FOOD_CLASSES, PLOTS_DIR


# =========================================================================
# Model Loader
# =========================================================================

def load_model():
    """
    Helper function to load the serialized Keras model from the disk.

    Returns:
        tf.keras.Model: Loaded Keras model instance.
    
    Raises:
        FileNotFoundError: If the model checkpoint file is missing.
    """
    print("Loading trained model checkpoint...")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. "
            "Please execute train.py first to create the classifier."
        )

    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully.")
    return model


# =========================================================================
# Visualization Plotters
# =========================================================================

def plot_confusion_matrix(y_true, y_pred):
    """
    Generates and saves a publication-quality confusion matrix plot.

    Args:
        y_true (np.ndarray): True target class indices.
        y_pred (np.ndarray): Predicted class indices.
    """
    os.makedirs(PLOTS_DIR, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)

    # Convert food class names to titled versions for clean labels
    titled_classes = [c.replace("_", " ").title() for c in FOOD_CLASSES]

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=titled_classes,
        yticklabels=titled_classes,
        square=True,
        cbar_kws={"shrink": .8},
        annot_kws={"size": 10}
    )

    plt.xlabel("Predicted Label", fontsize=12, fontweight="bold", labelpad=10)
    plt.ylabel("True Label", fontsize=12, fontweight="bold", labelpad=10)
    plt.title("Confusion Matrix — Food Classification & Nutrition AI", fontsize=14, fontweight="bold", pad=20)
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()

    save_path = os.path.join(PLOTS_DIR, "confusion_matrix.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Confusion matrix visualization saved to: {save_path}")


# =========================================================================
# Evaluation Orchestrator
# =========================================================================

def evaluate():
    """
    Orchestrates the evaluation workflow.

    Loads the saved model, extracts the validation split, calculates standard loss/accuracy
    evaluations, prints high-level classification reports, and compiles the confusion matrix.
    """
    # 1. Load weights
    model = load_model()

    # 2. Extract validation split
    print("Ingesting validation data...")
    _, val_ds = load_datasets()

    # 3. Evaluate basic accuracy/loss metrics
    print("Calculating overall loss and accuracy...")
    loss, accuracy = model.evaluate(val_ds, verbose=1)

    print("\n" + "=" * 40)
    print("Model Evaluation Summary")
    print("=" * 40)
    print(f"Validation Loss:     {loss:.4f}")
    print(f"Validation Accuracy: {accuracy * 100:.2f}%")
    print("=" * 40 + "\n")

    y_true = []
    y_pred = []

    print("Running validation inference loops...")
    for images, labels in val_ds:
        predictions = model.predict(images, verbose=0)
        predicted_labels = np.argmax(predictions, axis=1)

        y_true.extend(labels.numpy())
        y_pred.extend(predicted_labels)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    titled_classes = [c.replace("_", " ").title() for c in FOOD_CLASSES]

    print("Synthesizing Detailed Classification Report:\n")
    print(classification_report(y_true, y_pred, target_names=titled_classes))

    # 4. Generate confusion matrix
    plot_confusion_matrix(y_true, y_pred)


if __name__ == "__main__":
    evaluate()