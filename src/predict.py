"""
predict.py - Handles model inference with improved error handling
and lazy model loading for Hugging Face Spaces.
"""

import os
import numpy as np
import tensorflow as tf
from PIL import Image

from src.config import MODEL_PATH, IMAGE_SIZE, FOOD_CLASSES
from src.nutrition import get_nutrition_info


# --------------------------------------------------
# Fix: patch DepthwiseConv2D to strip unsupported 'groups' kwarg
# --------------------------------------------------

@tf.keras.utils.register_keras_serializable()
class FixedDepthwiseConv2D(tf.keras.layers.DepthwiseConv2D):

    def __init__(self, *args, **kwargs):
        kwargs.pop("groups", None)
        super().__init__(*args, **kwargs)


# --------------------------------------------------
# Lazy model loading (prevents HF startup timeout)
# --------------------------------------------------

model = None


def get_model():

    global model

    if model is None:

        print("Loading model...")

        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model file not found at {MODEL_PATH}. "
                "Make sure food_classifier.h5 exists in results/model/"
            )

        model = tf.keras.models.load_model(
            MODEL_PATH,
            compile=False,
            custom_objects={"DepthwiseConv2D": FixedDepthwiseConv2D}
        )

        print("Model loaded successfully")
        print(f"Input shape: {model.input_shape}")

    return model


# --------------------------------------------------
# Image validation
# --------------------------------------------------

def validate_image(image):

    if image is None:
        raise ValueError("No image provided.")

    if not isinstance(image, Image.Image):
        raise ValueError("Invalid image format.")

    width, height = image.size

    if width < 32 or height < 32:
        raise ValueError("Image is too small. Please upload a clearer photo.")

    if width > 8000 or height > 8000:
        raise ValueError("Image is too large. Please upload a smaller photo.")

    return True


# --------------------------------------------------
# Image preprocessing
# --------------------------------------------------

def preprocess_image(image):

    image = image.convert("RGB")

    image = image.resize(IMAGE_SIZE, Image.LANCZOS)

    image = np.array(image, dtype=np.float32) / 255.0

    image = np.expand_dims(image, axis=0)

    return image


# --------------------------------------------------
# Confidence check
# --------------------------------------------------

def is_low_confidence(confidence, threshold=0.35):

    return confidence < threshold


# --------------------------------------------------
# Main prediction function
# --------------------------------------------------

def predict_food(image):

    print("Starting prediction...")

    validate_image(image)

    print("Preprocessing image...")
    processed = preprocess_image(image)

    model = get_model()

    print("Running model inference...")
    predictions = model.predict(processed, verbose=0)[0]

    top_indices = predictions.argsort()[-3:][::-1]

    top_predictions = [
        {"food": FOOD_CLASSES[idx], "confidence": float(predictions[idx])}
        for idx in top_indices
    ]

    predicted_food = top_predictions[0]["food"]
    top_confidence = top_predictions[0]["confidence"]

    print(f"Prediction: {predicted_food} ({round(top_confidence * 100, 2)}%)")

    warning = None

    if is_low_confidence(top_confidence):

        warning = (
            f"Warning: Low confidence ({round(top_confidence * 100, 2)}%) "
            "– this food may not be in our database."
        )

    print("Fetching nutrition info...")

    nutrition, health_score, tip, category, color = get_nutrition_info(predicted_food)

    print(f"Nutrition source: {nutrition.get('source', 'unknown')}")

    return {

        "prediction": predicted_food,
        "confidence": top_confidence,
        "top3": top_predictions,

        "nutrition": nutrition,

        "health_score": health_score,
        "health_category": category,
        "health_color": color,

        "tip": tip,

        "warning": warning,

        "total_classes": len(FOOD_CLASSES),
    }