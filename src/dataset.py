"""
dataset.py

Handles loading and preprocessing of the Food101 dataset.

Responsibilities:
- Download dataset using TensorFlow Datasets
- Use 10% subset for faster training
- Filter the first 10 food classes
- Resize and normalize images
- Create training and validation datasets
"""

import tensorflow as tf
import tensorflow_datasets as tfds

from src.config import (
    DATASET_NAME,
    IMAGE_SIZE,
    BATCH_SIZE,
    FOOD_CLASSES,
)

# --------------------------------------------------
# Image Preprocessing
# --------------------------------------------------

def preprocess_image(image, label):
    """
    Resize and normalize images.
    """
    image = tf.image.resize(image, IMAGE_SIZE)
    image = image / 255.0
    return image, label


# --------------------------------------------------
# Class Filtering
# --------------------------------------------------

def filter_classes(image, label):
    """
    Keep only the first 10 classes from Food101.
    """
    return label < len(FOOD_CLASSES)


# --------------------------------------------------
# Dataset Loading
# --------------------------------------------------

def load_datasets():
    """
    Load Food101 dataset subset and prepare training and validation datasets.
    """

    print("Loading Food101 dataset (10% subset)...")

    train_ds = tfds.load(
        DATASET_NAME,
        split="train[:10%]",
        as_supervised=True,
        shuffle_files=True
    )

    val_ds = tfds.load(
        DATASET_NAME,
        split="validation[:10%]",
        as_supervised=True,
        shuffle_files=False
    )

    # --------------------------------------------------
    # Filter only the first 10 classes
    # --------------------------------------------------

    train_ds = train_ds.filter(filter_classes)
    val_ds = val_ds.filter(filter_classes)

    # --------------------------------------------------
    # Preprocess images
    # --------------------------------------------------

    train_ds = train_ds.map(
        preprocess_image,
        num_parallel_calls=2   # limit threads for 8GB RAM
    )

    val_ds = val_ds.map(
        preprocess_image,
        num_parallel_calls=2
    )

    # --------------------------------------------------
    # Shuffle, batch, prefetch
    # --------------------------------------------------

    train_ds = train_ds.shuffle(500).batch(BATCH_SIZE).prefetch(1)
    val_ds = val_ds.batch(BATCH_SIZE).prefetch(1)

    print("Dataset ready")

    return train_ds, val_ds