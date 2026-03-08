"""
config.py

Central configuration file for the Food Classification + Nutrition Analysis project.

All hyperparameters, dataset settings, paths, and constants are defined here.
Other modules should import from this file instead of hardcoding values.
"""

import os

# --------------------------------------------------
# Base Project Paths
# --------------------------------------------------

# Root directory of the project
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data directory
DATA_DIR = os.path.join(BASE_DIR, "data")

# Results directory
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Model saving directory
MODEL_DIR = os.path.join(RESULTS_DIR, "model")

# Plot directory
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")

# Saved model path
MODEL_PATH = os.path.join(MODEL_DIR, "food_classifier.h5")

# TensorBoard logs
LOG_DIR = os.path.join(RESULTS_DIR, "logs")

# --------------------------------------------------
# Dataset Configuration
# --------------------------------------------------

DATASET_NAME = "food101"

# First 10 classes from Food101
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
    "bruschetta"
]

NUM_CLASSES = len(FOOD_CLASSES)

# --------------------------------------------------
# Image Preprocessing
# --------------------------------------------------

IMAGE_SIZE = (224, 224)
CHANNELS = 3

# Training batch size
BATCH_SIZE = 32

# --------------------------------------------------
# Training Parameters
# --------------------------------------------------

# Phase 1 training
EPOCHS_PHASE1 = 10
LEARNING_RATE_PHASE1 = 1e-3

# Phase 2 fine tuning
EPOCHS_PHASE2 = 5
LEARNING_RATE_PHASE2 = 1e-5

# Number of layers to unfreeze
FINE_TUNE_LAYERS = 30

# --------------------------------------------------
# Model Architecture
# --------------------------------------------------

BASE_MODEL_NAME = "MobileNetV2"

DENSE_UNITS = 128
DROPOUT_RATE_1 = 0.3
DROPOUT_RATE_2 = 0.15

# --------------------------------------------------
# Callback Configuration
# --------------------------------------------------

EARLY_STOPPING_PATIENCE = 3

REDUCE_LR_PATIENCE = 2
REDUCE_LR_FACTOR = 0.3

# --------------------------------------------------
# Nutrition API
# --------------------------------------------------

OPEN_FOOD_FACTS_API = "https://world.openfoodfacts.org/cgi/search.pl"

# --------------------------------------------------
# Random Seed
# --------------------------------------------------

RANDOM_SEED = 42