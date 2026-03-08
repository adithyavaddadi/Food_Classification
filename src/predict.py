"""
predict.py

Handles model inference with improved error handling.
"""

import numpy as np
import tensorflow as tf
from PIL import Image, UnidentifiedImageError

from src.config import MODEL_PATH, IMAGE_SIZE, FOOD_CLASSES
from src.nutrition import get_nutrition_info

# --------------------------------------------------
# Load Model (lazy singleton)
# --------------------------------------------------

model = None

def load_model():
    global model
    if model is None:
        model = tf.keras.models.load_model(MODEL_PATH)


# --------------------------------------------------
# Image Validation
# --------------------------------------------------

def validate_image(image):
    """Validate image before prediction."""
    if image is None:
        raise ValueError("No image provided.")

    if not isinstance(image, Image.Image):
        raise ValueError("Invalid image format.")

    w, h = image.size
    if w < 32 or h < 32:
        raise ValueError("Image is too small. Please upload a clearer photo.")

    if w > 8000 or h > 8000:
        raise ValueError("Image is too large. Please upload a smaller photo.")

    return True


# --------------------------------------------------
# Image Preprocessing
# --------------------------------------------------

def preprocess_image(image):
    """Resize, convert and normalize input image."""
    image = image.convert("RGB")
    image = image.resize(IMAGE_SIZE, Image.LANCZOS)
    image = np.array(image, dtype=np.float32) / 255.0
    image = np.expand_dims(image, axis=0)
    return image


# --------------------------------------------------
# Confidence Check
# --------------------------------------------------

def is_low_confidence(confidence, threshold=0.35):
    """Flag predictions below confidence threshold."""
    return confidence < threshold


# --------------------------------------------------
# Prediction Function
# --------------------------------------------------

def predict_food(image):
    """
    Predict food class from input image.
    Returns structured result dict with prediction, nutrition, health info.
    """

    # Validate image
    validate_image(image)

    # Load model
    load_model()

    # Preprocess
    processed = preprocess_image(image)

    # Predict
    predictions = model.predict(processed, verbose=0)[0]

    # Top 3 predictions
    top_indices = predictions.argsort()[-3:][::-1]
    top_predictions = [
        {
            "food": FOOD_CLASSES[idx],
            "confidence": float(predictions[idx]),
        }
        for idx in top_indices
    ]

    predicted_food = top_predictions[0]["food"]
    top_confidence = top_predictions[0]["confidence"]

    # Low confidence warning
    warning = None
    if is_low_confidence(top_confidence):
        warning = f"⚠️ Low confidence ({round(top_confidence*100, 1)}%) — this food may not be in our database."

    # Get nutrition (now returns 5 values)
    nutrition, health_score, tip, category, color = get_nutrition_info(predicted_food)

    return {
        "prediction":   predicted_food,
        "confidence":   top_confidence,
        "top3":         top_predictions,
        "nutrition":    nutrition,
        "health_score": health_score,
        "health_category": category,
        "health_color": color,
        "tip":          tip,
        "warning":      warning,
        "total_classes": len(FOOD_CLASSES),
    }