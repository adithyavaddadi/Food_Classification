"""
evaluate.py

Evaluates the trained food classification model.

Functions:
- Load trained model
- Evaluate performance on validation dataset
- Generate confusion matrix
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from sklearn.metrics import confusion_matrix, classification_report

from src.dataset import load_datasets
from src.config import MODEL_PATH, FOOD_CLASSES, PLOTS_DIR


# --------------------------------------------------
# Load Model
# --------------------------------------------------

def load_model():

    print("Loading trained model...")

    model = tf.keras.models.load_model(MODEL_PATH)

    return model


# --------------------------------------------------
# Evaluate Model
# --------------------------------------------------

def evaluate():

    model = load_model()

    _, val_ds = load_datasets()

    print("Evaluating model...")

    loss, accuracy = model.evaluate(val_ds)

    print(f"\nValidation Loss: {loss:.4f}")
    print(f"Validation Accuracy: {accuracy:.4f}")

    y_true = []
    y_pred = []

    for images, labels in val_ds:

        predictions = model.predict(images)

        predicted_labels = np.argmax(predictions, axis=1)

        y_true.extend(labels.numpy())
        y_pred.extend(predicted_labels)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    print("\nClassification Report:\n")

    print(classification_report(y_true, y_pred, target_names=FOOD_CLASSES))

    plot_confusion_matrix(y_true, y_pred)


# --------------------------------------------------
# Confusion Matrix
# --------------------------------------------------

def plot_confusion_matrix(y_true, y_pred):

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=FOOD_CLASSES,
        yticklabels=FOOD_CLASSES
    )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")

    save_path = os.path.join(PLOTS_DIR, "confusion_matrix.png")

    plt.savefig(save_path)

    print(f"Confusion matrix saved to: {save_path}")


# --------------------------------------------------
# Run Evaluation
# --------------------------------------------------

if __name__ == "__main__":

    evaluate()