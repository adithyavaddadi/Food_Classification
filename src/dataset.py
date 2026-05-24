"""
Dataset loading and preprocessing module for the Food101 dataset.

This module handles:
1. Downloading the Food101 dataset via TensorFlow Datasets (TFDS).
2. Using a 10% subset of the data for faster local training/resource efficiency.
3. Filtering the dataset to include only the specified 10 classes.
4. Resizing and normalizing the input images.
5. Packaging the data into optimized TF pipelines (shuffle, batch, prefetch).
"""

import tensorflow as tf
import tensorflow_datasets as tfds

from src.config import (
    DATASET_NAME,
    IMAGE_SIZE,
    BATCH_SIZE,
    FOOD_CLASSES,
)

# =========================================================================
# Image Preprocessing & Filtering
# =========================================================================

def preprocess_image(image, label):
    """
    Resizes and normalizes a single image.

    Args:
        image (tf.Tensor): Raw input image tensor.
        label (tf.Tensor): Class index label.

    Returns:
        tuple: Preprocessed image tensor resized to config specification
               and normalized to [0, 1], and the unmodified label tensor.
    """
    image = tf.image.resize(image, IMAGE_SIZE)
    image = image / 255.0
    return image, label


def filter_classes(image, label):
    """
    Filters out labels that are not within the subset of target food classes.

    Args:
        image (tf.Tensor): Preprocessed image tensor.
        label (tf.Tensor): Class index label.

    Returns:
        tf.Tensor: Boolean tensor representing whether the label is in our subset.
    """
    return label < len(FOOD_CLASSES)


# =========================================================================
# Dataset Loading Pipeline
# =========================================================================

def load_datasets():
    """
    Loads and prepares the Food101 dataset subset for training and validation.

    Loads 10% of the Food101 splits, applies the target class filters, maps 
    resizing/normalization operations, and returns optimized datasets with shuffling,
    batching, and prefetching enabled.

    Returns:
        tuple: (train_ds, val_ds) - Ready-to-train tf.data.Dataset objects.
    """
    print("Loading Food101 dataset (10% subset)...")

    # Load 10% subsets to comply with 8GB RAM constraints and faster training
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

    # Filter to only keep our first 10 specified classes
    train_ds = train_ds.filter(filter_classes)
    val_ds = val_ds.filter(filter_classes)

    # Map preprocessing function (parallelize to handle resources gracefully)
    train_ds = train_ds.map(
        preprocess_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )
    val_ds = val_ds.map(
        preprocess_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    # Shuffling, batching, and prefetching to optimize TPU/GPU/CPU utilization
    train_ds = train_ds.shuffle(500).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    print("Dataset loading and processing complete.")
    return train_ds, val_ds