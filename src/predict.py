"""
Prediction and inference module.

This module exposes functions to perform end-to-end deep learning inference on
input images. It handles:
1. Lazy-loading the Keras H5 model to keep container startups lightweight.
2. Validating input image format, resolution, and dimensions.
3. Resizing and normalizing the input image.
4. Executing predictions with custom error checking.
5. Packaging the top 3 prediction probabilities and warnings for low confidence.
6. Retrieving nutritional facts and health advice.
"""

import os
import numpy as np
import tensorflow as tf
from PIL import Image

from src.config import MODEL_PATH, IMAGE_SIZE, FOOD_CLASSES
from src.nutrition import get_nutrition_info


# =========================================================================
# Keras Compatibility Patches
# =========================================================================

@tf.keras.utils.register_keras_serializable()
class FixedDepthwiseConv2D(tf.keras.layers.DepthwiseConv2D):
    """
    Custom DepthwiseConv2D layer that strips unsupported 'groups' arguments.

    This ensures older or newer Keras model checkpoints are compatible across
    differing runtime versions (highly critical for HF Spaces deployments).
    """
    def __init__(self, *args, **kwargs):
        kwargs.pop("groups", None)
        super().__init__(*args, **kwargs)


# =========================================================================
# Lazy Model Caching
# =========================================================================

_model_cache = None


def get_model():
    """
    Loads and returns the cached classification model.

    Utilizes lazy loading to ensure model weights are loaded into memory only
    when the first classification query executes. This prevents Hugging Face
    startup timeouts (which occur when heavy models are loaded inside app.py global scope).

    Returns:
        tf.keras.Model: Loaded Keras model instance.
    """
    global _model_cache

    if _model_cache is None:
        print("Lazy-loading deep learning model weights...")
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model file not found at {MODEL_PATH}. "
                "Ensure food_classifier.h5 is placed in results/model/ directory."
            )

        _model_cache = tf.keras.models.load_model(
            MODEL_PATH,
            compile=False,
            custom_objects={"DepthwiseConv2D": FixedDepthwiseConv2D}
        )
        print("Model loaded successfully.")
        print(f"Accepted Input Shape: {_model_cache.input_shape}")

    return _model_cache


# =========================================================================
# Image Validation & Preprocessing
# =========================================================================

def validate_image(image):
    """
    Ensures input image satisfies dimensional limits and formats.

    Args:
        image (PIL.Image.Image): Uploaded image object.

    Returns:
        bool: True if image is valid, otherwise raises ValueError.
    """
    if image is None:
        raise ValueError("No image provided. Please upload a clear photo of food.")

    if not isinstance(image, Image.Image):
        raise ValueError("Invalid image format. Expected a standard PIL Image.")

    width, height = image.size

    # Prevent extremely small/blurry or excessively massive input sizes
    if width < 32 or height < 32:
        raise ValueError("Image resolution is too low. Please upload a clearer photo.")

    if width > 8000 or height > 8000:
        raise ValueError("Image file dimension is too large. Please upload a smaller photo.")

    return True


def preprocess_image(image):
    """
    Normalizes and reshapes PIL Image to match model tensor inputs.

    Args:
        image (PIL.Image.Image): Input image.

    Returns:
        np.ndarray: Image tensor of shape (1, 224, 224, 3) normalized to [0, 1].
    """
    # Force RGB representation to discard transparency/alpha channel if present
    image = image.convert("RGB")
    image = image.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
    
    # Scale pixel intensities and inject batch dimension
    image_array = np.array(image, dtype=np.float32) / 255.0
    image_tensor = np.expand_dims(image_array, axis=0)
    
    return image_tensor


# =========================================================================
# Inference Function
# =========================================================================

def predict_food(image):
    """
    Executes deep learning classification and nutritional retrieval pipeline.

    Args:
        image (PIL.Image.Image): Uploaded PIL image.

    Returns:
        dict: Complete analysis payload containing predictions, top3 list,
              nutrition facts, health score, category details, warning, and tips.
    """
    print("Initiating classification...")
    validate_image(image)

    print("Formatting image tensor...")
    processed_tensor = preprocess_image(image)

    # Retrieve lazy-cached model
    classifier = get_model()

    print("Running forward propagation...")
    predictions = classifier.predict(processed_tensor, verbose=0)[0]

    # Extract the top 3 highest probabilities
    top_indices = predictions.argsort()[-3:][::-1]
    top_predictions = [
        {"food": FOOD_CLASSES[idx], "confidence": float(predictions[idx])}
        for idx in top_indices
    ]

    predicted_food = top_predictions[0]["food"]
    top_confidence = top_predictions[0]["confidence"]

    print(f"Top Class Match: {predicted_food} ({round(top_confidence * 100, 2)}%)")

    # Add warning banner if model confidence is low (ambiguous image)
    warning = None
    if top_confidence < 0.35:
        warning = (
            f"Warning: Low confidence match ({round(top_confidence * 100, 2)}%). "
            "The food item may not be present in our 10 trained classes."
        )

    # Fetch USDA/OFF nutritional info
    print("Retrieving nutritional facts...")
    nutrition, health_score, tip, category, color = get_nutrition_info(predicted_food)

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