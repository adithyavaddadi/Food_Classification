"""
nutrition.py

Handles nutrition information retrieval.

Steps:
1. Try USDA FoodData Central API (free, no key needed for basic search)
2. Fallback to Open Food Facts API
3. If both fail, return fallback data
4. Compute improved health score and tips
"""

import requests
from src.config import OPEN_FOOD_FACTS_API

# --------------------------------------------------
# Fallback Nutrition Data (per 100g) - All 10 classes
# --------------------------------------------------

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


# --------------------------------------------------
# USDA FoodData Central API (free, no key required)
# --------------------------------------------------

USDA_API_URL = "https://api.nal.usda.gov/fdc/v1/foods/search"
USDA_API_KEY = "DEMO_KEY"  # Free demo key, works for low usage


def fetch_from_usda(food_name):
    """Fetch nutrition from USDA FoodData Central."""
    try:
        query = food_name.replace("_", " ")
        params = {
            "query": query,
            "dataType": ["Survey (FNDDS)"],
            "pageSize": 1,
            "api_key": USDA_API_KEY,
        }
        response = requests.get(USDA_API_URL, params=params, timeout=6)
        data = response.json()

        if not data.get("foods"):
            return None

        food = data["foods"][0]
        nutrients = {n["nutrientName"]: n["value"] for n in food.get("foodNutrients", [])}

        return {
            "calories": round(nutrients.get("Energy", 0)),
            "protein":  round(nutrients.get("Protein", 0), 1),
            "carbs":    round(nutrients.get("Carbohydrate, by difference", 0), 1),
            "fat":      round(nutrients.get("Total lipid (fat)", 0), 1),
            "fiber":    round(nutrients.get("Fiber, total dietary", 0), 1),
            "sugar":    round(nutrients.get("Sugars, total including NLEA", 0), 1),
            "sodium":   round(nutrients.get("Sodium, Na", 0)),
            "source":   "USDA FoodData Central",
        }
    except Exception:
        return None


# --------------------------------------------------
# Open Food Facts API (backup)
# --------------------------------------------------

def fetch_from_openfoodfacts(food_name):
    """Fetch nutrition from Open Food Facts."""
    try:
        params = {"search_terms": food_name.replace("_", " "), "search_simple": 1, "json": 1}
        response = requests.get(OPEN_FOOD_FACTS_API, params=params, timeout=5)
        data = response.json()

        if not data.get("products"):
            return None

        product = data["products"][0]
        n = product.get("nutriments", {})

        return {
            "calories": round(n.get("energy-kcal_100g", 0)),
            "protein":  round(n.get("proteins_100g", 0), 1),
            "carbs":    round(n.get("carbohydrates_100g", 0), 1),
            "fat":      round(n.get("fat_100g", 0), 1),
            "fiber":    round(n.get("fiber_100g", 0), 1),
            "sugar":    round(n.get("sugars_100g", 0), 1),
            "sodium":   round(n.get("sodium_100g", 0) * 1000),
            "source":   "Open Food Facts",
        }
    except Exception:
        return None


# --------------------------------------------------
# Improved Health Score (multi-factor, weighted)
# --------------------------------------------------

def calculate_health_score(nutrition):
    """
    Multi-factor weighted health score (1-10).
    Based on WHO/NHS daily intake guidelines per 100g.
    """
    score = 10.0

    # Calories (ref: 200 kcal per 100g is moderate)
    cal = nutrition.get("calories", 0)
    if cal > 450:   score -= 2.5
    elif cal > 300: score -= 1.5
    elif cal > 200: score -= 0.5

    # Fat (ref: >17.5g per 100g is high)
    fat = nutrition.get("fat", 0)
    if fat > 25:    score -= 2.0
    elif fat > 17:  score -= 1.0

    # Sugar (ref: >22.5g per 100g is high)
    sugar = nutrition.get("sugar", 0)
    if sugar > 30:  score -= 2.0
    elif sugar > 22: score -= 1.0

    # Sodium (ref: >600mg per 100g is high)
    sodium = nutrition.get("sodium", 0)
    if sodium > 800:  score -= 2.0
    elif sodium > 600: score -= 1.0

    # Fiber bonus (fiber is healthy)
    fiber = nutrition.get("fiber", 0)
    if fiber >= 6:   score += 1.0
    elif fiber >= 3: score += 0.5

    # Protein bonus (protein is filling & healthy)
    protein = nutrition.get("protein", 0)
    if protein >= 20: score += 0.5

    return max(1, min(10, round(score)))


def get_health_category(score):
    if score >= 8:  return "Excellent", "#22c55e"
    if score >= 6:  return "Good", "#84cc16"
    if score >= 4:  return "Moderate", "#f59e0b"
    return "Indulgent", "#ef4444"


# --------------------------------------------------
# Health Tips (context-aware)
# --------------------------------------------------

def generate_health_tip(food_name, nutrition, score):
    name = food_name.replace("_", " ").title()
    tips = []

    if nutrition.get("calories", 0) > 300:
        tips.append("high in calories — best enjoyed in smaller portions")
    if nutrition.get("sugar", 0) > 20:
        tips.append("contains significant sugar — pair with protein to slow absorption")
    if nutrition.get("sodium", 0) > 500:
        tips.append("high sodium content — balance with plenty of water")
    if nutrition.get("fiber", 0) >= 3:
        tips.append("good source of fiber — supports digestion")
    if nutrition.get("protein", 0) >= 15:
        tips.append("rich in protein — great for muscle recovery")

    if not tips:
        if score >= 7:
            return f"{name} is a nutritious choice. Enjoy it as part of a balanced diet."
        return f"{name} can be enjoyed in moderation as part of a varied diet."

    base = f"{name} is " + tips[0]
    if len(tips) > 1:
        base += f", and {tips[1]}"
    return base.capitalize() + "."


# --------------------------------------------------
# Main Function
# --------------------------------------------------

def get_nutrition_info(food_name):
    """Get nutrition from API with fallback chain."""

    # Try USDA first (most accurate)
    nutrition = fetch_from_usda(food_name)

    # Try Open Food Facts
    if nutrition is None:
        nutrition = fetch_from_openfoodfacts(food_name)

    # Use fallback data
    if nutrition is None:
        nutrition = FALLBACK_DATA.get(food_name, {
            "calories": 0, "protein": 0, "carbs": 0,
            "fat": 0, "fiber": 0, "sugar": 0, "sodium": 0,
        })
        nutrition["source"] = "Estimated"

    health_score = calculate_health_score(nutrition)
    category, color = get_health_category(health_score)
    tip = generate_health_tip(food_name, nutrition, health_score)

    return nutrition, health_score, tip, category, color