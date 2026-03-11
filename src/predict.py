"""
predict.py - Handles model inference with improved error handling.
"""
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import custom_object_scope
from PIL import Image, UnidentifiedImageError
from src.config import MODEL_PATH, IMAGE_SIZE, FOOD_CLASSES
from src.nutrition import get_nutrition_info

class FixedDepthwiseConv2D(tf.keras.layers.DepthwiseConv2D):
    def __init__(self, *args, **kwargs):
        kwargs.pop("groups", None)
        super().__init__(*args, **kwargs)

model = None

def load_model():
    global model
    if model is None:
        with custom_object_scope({"DepthwiseConv2D": FixedDepthwiseConv2D}):
            model = tf.keras.models.load_model(MODEL_PATH)

def validate_image(image):
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

def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize(IMAGE_SIZE, Image.LANCZOS)
    image = np.array(image, dtype=np.float32) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

def is_low_confidence(confidence, threshold=0.35):
    return confidence < threshold

def predict_food(image):
    validate_image(image)
    load_model()
    processed = preprocess_image(image)
    predictions = model.predict(processed, verbose=0)[0]
    top_indices = predictions.argsort()[-3:][::-1]
    top_predictions = [
        {"food": FOOD_CLASSES[idx], "confidence": float(predictions[idx])}
        for idx in top_indices
    ]
    predicted_food = top_predictions[0]["food"]
    top_confidence = top_predictions[0]["confidence"]
    warning = None
    if is_low_confidence(top_confidence):
        warning = f"⚠️ Low confidence ({round(top_confidence*100, 1)}%) — this food may not be in our database."
    nutrition, health_score, tip, category, color = get_nutrition_info(predicted_food)
    return {
        "prediction":      predicted_food,
        "confidence":      top_confidence,
        "top3":            top_predictions,
        "nutrition":       nutrition,
        "health_score":    health_score,
        "health_category": category,
        "health_color":    color,
        "tip":             tip,
        "warning":         warning,
        "total_classes":   len(FOOD_CLASSES),
    }
