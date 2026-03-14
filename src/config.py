"""
Central configuration for the Food Classification project
"""

import os

# Base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

MODEL_DIR = os.path.join(RESULTS_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "food_classifier.h5")

PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
LOG_DIR = os.path.join(RESULTS_DIR, "logs")

# Dataset
DATASET_NAME = "food101"

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

# Image
IMAGE_SIZE = (224, 224)
CHANNELS = 3
BATCH_SIZE = 32

# Model
BASE_MODEL_NAME = "MobileNetV2"

DENSE_UNITS = 128
DROPOUT_RATE_1 = 0.3
DROPOUT_RATE_2 = 0.15

# Training
EPOCHS_PHASE1 = 10
EPOCHS_PHASE2 = 5

LEARNING_RATE_PHASE1 = 1e-3
LEARNING_RATE_PHASE2 = 1e-5

FINE_TUNE_LAYERS = 30

# Callbacks
EARLY_STOPPING_PATIENCE = 3
REDUCE_LR_PATIENCE = 2
REDUCE_LR_FACTOR = 0.3

# API
OPEN_FOOD_FACTS_API = "https://world.openfoodfacts.org/cgi/search.pl"

# Seed
RANDOM_SEED = 42