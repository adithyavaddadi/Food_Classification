"""
Nutrition information and health scoring module.

This module retrieves nutritional facts (calories, protein, carbs, fats, fiber, etc.)
for recognized food items using:
1. USDA FoodData Central API (primary, highly accurate for US-centric dishes).
2. Open Food Facts API (secondary, excellent for international foods).
3. Local fallback database (tertiary, robust estimate backup to keep app operational offline).

It also implements a comprehensive WHO guidelines-based health scoring system (1-10)
and generates descriptive actionable dietary tips.
"""

import os
import requests
from dotenv import load_dotenv

from src.config import OPEN_FOOD_FACTS_API

# Load local environment variables (if any)
load_dotenv()

# =========================================================================
# API Credentials & Configurations
# =========================================================================

USDA_API_URL = "https://api.nal.usda.gov/fdc/v1/foods/search"
USDA_API_KEY = os.getenv("USDA_API_KEY", "DEMO_KEY")


# =========================================================================
# Fallback Nutrition Database (Per 100g serving)
# =========================================================================

FALLBACK_DATA = {
    "apple_pie":        {"calories": 237, "protein": 2.4, "carbs": 34, "fat": 11,  "fiber": 1.5, "sugar": 19, "sodium": 266},
    "baby_back_ribs":   {"calories": 292, "protein": 25,  "carbs": 0,  "fat": 21,  "fiber": 0,   "sugar": 0,  "sodium": 600},
    "baklava":          {"calories": 334, "protein": 6,   "carbs": 40, "fat": 18,  "fiber": 2,   "sugar": 30, "sodium": 200},
    "beef_carpaccio":   {"calories": 120, "protein": 20,  "carbs": 1,  "fat": 4,   "fiber": 0,   "sugar": 0,  "sodium": 300},
    "beef_tartare":     {"calories": 180, "protein": 21,  "carbs": 0,  "fat": 10,  "fiber": 0,   "sugar": 0,  "sodium": 350},
    "beet_salad":       {"calories": 70,  "protein": 2,   "carbs": 15, "fat": 1,   "fiber": 3,   "sugar": 10, "sodium": 120},
    "beignets":         {"calories": 315, "protein": 6,   "carbs": 40, "fat": 15,  "fiber": 1,   "sugar": 12, "sodium": 220},
    "bibimbap":         {"calories": 150, "protein": 6,   "carbs": 20, "fat": 5,   "fiber": 3,   "sugar": 3,  "sodium": 400},
    "bread_pudding":    {"calories": 250, "protein": 5,   "carbs": 35, "fat": 10,  "fiber": 1,   "sugar": 20, "sodium": 180},
    "bruschetta":       {"calories": 180, "protein": 5,   "carbs": 22, "fat": 7,   "fiber": 2,   "sugar": 3,  "sodium": 300},
}


# =========================================================================
# API Retrievals
# =========================================================================

def fetch_from_usda(food_name):
    """
    Fetches nutritional values from the USDA FoodData Central API.

    Queries the Survey (FNDDS) database for a specified food name, parses out 
    crucial macro and micronutrients per 100g, and formats them.

    Args:
        food_name (str): The name of the food class (e.g. "apple_pie").

    Returns:
        dict or None: Formatted nutrition dict if successful, otherwise None.
    """
    try:
        params = {
            "query": food_name.replace("_", " "),
            "dataType": ["Survey (FNDDS)"],
            "pageSize": 1,
            "api_key": USDA_API_KEY,
        }

        response = requests.get(USDA_API_URL, params=params, timeout=5)
        if response.status_code != 200:
            return None

        data = response.json()
        if not data.get("foods"):
            return None

        food = data["foods"][0]
        nutrients = {
            n["nutrientName"]: n["value"]
            for n in food.get("foodNutrients", [])
        }

        return {
            "calories": round(nutrients.get("Energy", 0)),
            "protein": round(nutrients.get("Protein", 0), 1),
            "carbs": round(nutrients.get("Carbohydrate, by difference", 0), 1),
            "fat": round(nutrients.get("Total lipid (fat)", 0), 1),
            "fiber": round(nutrients.get("Fiber, total dietary", 0), 1),
            "sugar": round(nutrients.get("Sugars, total including NLEA", 0), 1),
            "sodium": round(nutrients.get("Sodium, Na", 0)),
            "source": "USDA FoodData Central",
        }
    except Exception:
        return None


def fetch_from_openfoodfacts(food_name):
    """
    Fetches nutritional values from the Open Food Facts JSON API.

    Serves as the secondary API fallback for international or packaged food queries.

    Args:
        food_name (str): The name of the food class.

    Returns:
        dict or None: Formatted nutrition dict if successful, otherwise None.
    """
    try:
        params = {
            "search_terms": food_name.replace("_", " "),
            "search_simple": 1,
            "json": 1,
        }

        response = requests.get(OPEN_FOOD_FACTS_API, params=params, timeout=5)
        if response.status_code != 200:
            return None

        data = response.json()
        if not data.get("products"):
            return None

        product = data["products"][0]
        n = product.get("nutriments", {})

        return {
            "calories": round(n.get("energy-kcal_100g", 0)),
            "protein": round(n.get("proteins_100g", 0), 1),
            "carbs": round(n.get("carbohydrates_100g", 0), 1),
            "fat": round(n.get("fat_100g", 0), 1),
            "fiber": round(n.get("fiber_100g", 0), 1),
            "sugar": round(n.get("sugars_100g", 0), 1),
            "sodium": round(n.get("sodium_100g", 0) * 1000),  # Convert g to mg
            "source": "Open Food Facts",
        }
    except Exception:
        return None


# =========================================================================
# Health Scoring & Advisory Logic
# =========================================================================

def calculate_health_score(nutrition):
    """
    Computes a health score (1 to 10) based on WHO nutritional guidelines.

    Deducts points for excessive calories, high fat, high sugar, or elevated sodium.
    Awards bonus points for high fiber content.

    Args:
        nutrition (dict): Dictionary containing macro/micronutrient counts.

    Returns:
        int: Score between 1 and 10.
    """
    score = 10
    cal = nutrition.get("calories", 0)

    # Calories penalty
    if cal > 450:
        score -= 3
    elif cal > 300:
        score -= 2
    elif cal > 200:
        score -= 1

    # Total lipid fat penalty
    if nutrition.get("fat", 0) > 20:
        score -= 2

    # High simple sugars penalty
    if nutrition.get("sugar", 0) > 25:
        score -= 2

    # High sodium / salt penalty
    if nutrition.get("sodium", 0) > 700:
        score -= 2

    # Dietary fiber bonus
    if nutrition.get("fiber", 0) >= 3:
        score += 1

    # Confined output strictly to range [1, 10]
    return max(1, min(10, score))


def get_health_category(score):
    """
    Maps numerical health score to standard category and hex color code.

    Args:
        score (int): Health score.

    Returns:
        tuple: (category_name, hex_color_string)
    """
    if score >= 8:
        return "Excellent", "#22c55e"
    if score >= 6:
        return "Good", "#84cc16"
    if score >= 4:
        return "Moderate", "#f59e0b"
    return "Indulgent", "#ef4444"


def generate_health_tip(food_name, nutrition, score):
    """
    Synthesizes tailored actionable dietary guidance based on nutritional profile.

    Args:
        food_name (str): Cleaned class name of the target food.
        nutrition (dict): Nutrient counts.
        score (int): Health score of the food item.

    Returns:
        str: Descriptive personalized nutritional advisory.
    """
    name = food_name.replace("_", " ").title()

    if score >= 8:
        return f"{name} is a highly nutritious choice. Enjoy it as part of a balanced diet!"
    if nutrition.get("calories", 0) > 300:
        return f"{name} is calorie-dense. Consider pairing it with fresh greens and keeping portions moderate."
    if nutrition.get("sugar", 0) > 20:
        return f"{name} contains elevated levels of sugar. Pair it with healthy proteins to avoid rapid glucose spikes."
    if nutrition.get("sodium", 0) > 500:
        return f"{name} is relatively high in sodium. Be sure to drink plenty of water and balance other meals."
    
    return f"{name} is a comforting dish that is best enjoyed occasionally as part of a varied, healthy diet."


# =========================================================================
# Main Retrieval Orchestrator
# =========================================================================

def get_nutrition_info(food_name):
    """
    Orchestrates the fallback chain to retrieve nutrition data for a food class.

    Retrieves values from USDA API first, falls back to Open Food Facts API second,
    and falls back to pre-calculated local values third. Then computes scores
    and advisory messages.

    Args:
        food_name (str): Exact classification string.

    Returns:
        tuple: (nutrition, health_score, tip, category, color)
    """
    # 1. Attempt USDA Central Query
    nutrition = fetch_from_usda(food_name)

    # 2. Attempt OFF Secondary Query
    if nutrition is None:
        nutrition = fetch_from_openfoodfacts(food_name)

    # 3. Apply offline local database estimates
    if nutrition is None:
        nutrition = FALLBACK_DATA.get(
            food_name,
            {
                "calories": 0,
                "protein": 0,
                "carbs": 0,
                "fat": 0,
                "fiber": 0,
                "sugar": 0,
                "sodium": 0,
            },
        )
        nutrition["source"] = "Estimated"

    # Compute additional values
    health_score = calculate_health_score(nutrition)
    category, color = get_health_category(health_score)
    tip = generate_health_tip(food_name, nutrition, health_score)

    return nutrition, health_score, tip, category, color