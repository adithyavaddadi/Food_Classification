"""
Central configuration module for the Food Classification & Nutrition AI project.

This module defines directory paths, model hyperparameters, training configurations,
and API links. Changes here will propagate throughout the entire pipeline.
"""

import os

# =========================================================================
# Path Configurations
# =========================================================================

# Base directory (project root)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data & results directories
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Model checkpoints
MODEL_DIR = os.path.join(RESULTS_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "food_classifier.h5")

# Evaluation plots and training logs
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
LOG_DIR = os.path.join(RESULTS_DIR, "logs")


# =========================================================================
# Dataset Configurations
# =========================================================================

# TensorFlow Datasets name
DATASET_NAME = "food101"

# The 10 food classes selected from Food101 for the subset
FOOD_CLASSES = [
    "apple_pie",
    "baby_back_ribs",
    "baklava",
    "beef_carpaccio",
    "beef_tartare",
    "beet_salad",
    "beignets",
    "bibimbap",
    "bread_pudding",
    "bruschetta",
]

NUM_CLASSES = len(FOOD_CLASSES)


# =========================================================================
# Model & Image Configurations
# =========================================================================

# Image shape dimensions and batch size
IMAGE_SIZE = (224, 224)
CHANNELS = 3
BATCH_SIZE = 32

# Model architecture options
BASE_MODEL_NAME = "MobileNetV2"
DENSE_UNITS = 128
DROPOUT_RATE_1 = 0.3
DROPOUT_RATE_2 = 0.15


# =========================================================================
# Training Configurations
# =========================================================================

# Phase 1: Feature extraction (base frozen)
EPOCHS_PHASE1 = 10
LEARNING_RATE_PHASE1 = 1e-3

# Phase 2: Fine-tuning (top layers unfrozen)
EPOCHS_PHASE2 = 5
LEARNING_RATE_PHASE2 = 1e-5
FINE_TUNE_LAYERS = 30  # Number of top layers of base model to unfreeze

# Callbacks configurations
EARLY_STOPPING_PATIENCE = 3
REDUCE_LR_PATIENCE = 2
REDUCE_LR_FACTOR = 0.3


# =========================================================================
# External APIs & Seed
# =========================================================================

# Fallback Nutrition API (Open Food Facts)
OPEN_FOOD_FACTS_API = "https://world.openfoodfacts.org/cgi/search.pl"

# Random state reproducibility seed
RANDOM_SEED = 42