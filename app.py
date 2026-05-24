"""
Main web application entry point for the Food Recognition & Nutrition AI project.

This script launches the Gradio user interface that allows users to upload
food images (or capture them via webcam), run deep learning classifications,
view detailed USDA/OFF nutritional analysis, and inspect WHO health scoring dashboards.
"""

import os
import warnings
import urllib.request
import gradio as gr
from dotenv import load_dotenv

# Set environmental flags before importing heavy deep learning libraries
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

warnings.filterwarnings("ignore")

# Load environment variables at startup
load_dotenv()

from src.predict import predict_food
from src.config import FOOD_CLASSES

# =========================================================================
# Pre-download Sample/Example Images
# =========================================================================

SAMPLES_DIR = "data/samples"
os.makedirs(SAMPLES_DIR, exist_ok=True)

SAMPLE_URLS = {
    "pizza.jpg": "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a3/Eq_it-na_pizza-margherita_sep2005_sml.jpg/400px-Eq_it-na_pizza-margherita_sep2005_sml.jpg",
    "biryani.jpg": "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Biryani_at_Hyderabad%2CIndia.jpg/400px-Biryani_at_Hyderabad%2CIndia.jpg",
    "coffee.jpg": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/45/A_small_cup_of_coffee.JPG/400px-A_small_cup_of_coffee.JPG",
    "mango.jpg": "https://upload.wikimedia.org/wikipedia/commons/thumb/9/90/Hapus_Mango.jpg/400px-Hapus_Mango.jpg",
    "salad.jpg": "https://upload.wikimedia.org/wikipedia/commons/thumb/6/6d/Good_Food_Display_-_NCI_Visuals_Online.jpg/400px-Good_Food_Display_-_NCI_Visuals_Online.jpg",
    "strawberry.jpg": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4d/Chocolate_covered_strawberries.jpg/400px-Chocolate_covered_strawberries.jpg",
}

SAMPLE_PATHS = []

print("Validating local example images...")
for filename, url in SAMPLE_URLS.items():
    path = os.path.join(SAMPLES_DIR, filename)
    if not os.path.exists(path):
        print(f"Downloading example sample: {filename}")
        try:
            urllib.request.urlretrieve(url, path)
        except Exception as e:
            print(f"Skipped downloading {filename} due to networking: {e}")
            continue
    SAMPLE_PATHS.append([path])

print("Example samples validation complete.")


# =========================================================================
# HTML Template Blocks
# =========================================================================

MODEL_INFO = f"""
<div class='model-info-bar'>
    <div class='model-stat'><span class='ms-label'>Base Classifier</span><span class='ms-val'>MobileNetV2</span></div>
    <div class='model-divider'></div>
    <div class='model-stat'><span class='ms-label'>Training Corpus</span><span class='ms-val'>Food101 Subset</span></div>
    <div class='model-divider'></div>
    <div class='model-stat'><span class='ms-label'>Active Classes</span><span class='ms-val'>{len(FOOD_CLASSES)} Classes</span></div>
    <div class='model-divider'></div>
    <div class='model-stat'><span class='ms-label'>Training Mode</span><span class='ms-val'>Two-Phase Fine-Tuned</span></div>
    <div class='model-divider'></div>
    <div class='model-stat'><span class='ms-label'>Resolution</span><span class='ms-val'>224 × 224</span></div>
</div>
"""


# =========================================================================
# Gradio Input-Output Event Handler
# =========================================================================

def classify_food(image):
    """
    Interface callback to parse classifications and format Gradio layouts.

    Args:
        image (PIL.Image.Image): Uploaded image object.

    Returns:
        tuple: Formatted markdown grids and dataframe outputs for Gradio elements.
    """
    if image is None:
        return ("### ❌ Upload an image first.", "", [], "", "", "", "")

    try:
        result = predict_food(image)

        # Format output predictions
        prediction = result["prediction"].replace("_", " ").title()
        confidence = round(result["confidence"] * 100, 2)
        warning = result.get("warning", "")

        pred_text = f"### Identified Food: **{prediction}**"
        if warning:
            pred_text += f"\n\n<div class='warning-box'>⚠️ {warning}</div>"

        conf_text = f"### Top Match Confidence: **{confidence}%**"

        # Compile Top 3
        top3 = [
            [item["food"].replace("_", " ").title(), f"{round(item['confidence']*100, 2)}%"]
            for item in result["top3"]
        ]

        # Extract nutrition data source details
        nutrition = result["nutrition"]
        source = nutrition.get("source", "Estimated")
        source_badge = f"<span class='source-badge'>📡 {source}</span>"

        # Build Nutrition Facts HTML
        nutrition_text = f"""
<div class='card-header'>
    <h3>Nutrition Facts &nbsp;{source_badge}</h3>
    <p class='nut-sub'>Calculated per 100g portion</p>
</div>
<div class='nutrition-grid'>
    <div class='nut-item cal'><span class='nut-icon'>🔥</span><div><span class='nut-label'>Energy</span><span class='nut-val'>{nutrition['calories']}<em> kcal</em></span></div></div>
    <div class='nut-item'><span class='nut-icon'>💪</span><div><span class='nut-label'>Protein</span><span class='nut-val'>{nutrition['protein']}<em> g</em></span></div></div>
    <div class='nut-item'><span class='nut-icon'>🍞</span><div><span class='nut-label'>Carbs</span><span class='nut-val'>{nutrition['carbs']}<em> g</em></span></div></div>
    <div class='nut-item'><span class='nut-icon'>🧈</span><div><span class='nut-label'>Total Fat</span><span class='nut-val'>{nutrition['fat']}<em> g</em></span></div></div>
    <div class='nut-item'><span class='nut-icon'>🌿</span><div><span class='nut-label'>Fiber</span><span class='nut-val'>{nutrition['fiber']}<em> g</em></span></div></div>
    <div class='nut-item'><span class='nut-icon'>🍬</span><div><span class='nut-label'>Sugar</span><span class='nut-val'>{nutrition['sugar']}<em> g</em></span></div></div>
    <div class='nut-item'><span class='nut-icon'>🧂</span><div><span class='nut-label'>Sodium</span><span class='nut-val'>{nutrition['sodium']}<em> mg</em></span></div></div>
</div>
"""

        # Build Health Score HTML
        health_score = result["health_score"]
        category = result["health_category"]
        color = result["health_color"]
        tip = result["tip"]
        bar_width = health_score * 10

        health_text = f"""
<div class='card-header'>
    <h3>WHO Health Score</h3>
</div>
<div class='health-container'>
    <div class='health-bar-wrap'>
        <div class='health-bar-bg'>
            <div class='health-bar-fill' style='width:{bar_width}%; background:{color}; box-shadow:0 0 12px {color}88'></div>
        </div>
        <span class='health-num' style='color:{color}'>{health_score}<span class='health-denom'>/10</span></span>
    </div>
    <div class='health-category' style='color:{color}; border-color:{color}'>{category}</div>
</div>
"""
        tip_text = f"<div class='tip-box'>💡 **Health Advisory:** {tip}</div>"

        return (
            pred_text,
            conf_text,
            top3,
            nutrition_text,
            health_text,
            tip_text,
            "",
        )

    except ValueError as e:
        return (f"### ⚠️ Validation Error\n{str(e)}", "", [], "", "", "", "")
    except Exception as e:
        return (f"### ❌ Inference Failure\n{str(e)}", "", [], "", "", "", "")


# =========================================================================
# Premium Design Aesthetics CSS
# =========================================================================

custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;700&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

/* Main Body & Layout */
body {
    background: radial-gradient(circle at top right, #111827, #030712) !important;
    color: #f3f4f6 !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}

#hero {
    text-align: center;
    padding: 30px 10px 10px 10px;
    background: linear-gradient(180deg, rgba(249, 115, 22, 0.05) 0%, rgba(3, 7, 18, 0) 100%);
    border-bottom: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 20px;
    margin-bottom: 20px;
}

#hero h1 {
    font-family: 'Outfit', sans-serif;
    font-size: 2.6rem;
    font-weight: 700;
    letter-spacing: -0.05em;
    background: linear-gradient(to right, #fb923c, #f97316, #ea580c);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 8px;
}

#hero p {
    color: #9ca3af;
    font-size: 1.1rem;
    font-weight: 400;
}

/* Info Bar styling */
.model-info-bar {
    display: flex;
    justify-content: space-around;
    align-items: center;
    background: rgba(17, 24, 39, 0.6);
    border: 1px solid rgba(255, 255, 255, 0.08);
    backdrop-filter: blur(16px);
    border-radius: 14px;
    padding: 12px 16px;
    margin-bottom: 24px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
    flex-wrap: wrap;
    gap: 10px;
}

.model-stat {
    display: flex;
    flex-direction: column;
    align-items: center;
    text-align: center;
}

.ms-label {
    font-size: 0.75rem;
    color: #6b7280;
    text-transform: uppercase;
    font-weight: 600;
    letter-spacing: 0.05em;
}

.ms-val {
    font-size: 0.95rem;
    color: #f3f4f6;
    font-weight: 700;
    margin-top: 2px;
}

.model-divider {
    width: 1px;
    height: 32px;
    background-color: rgba(255, 255, 255, 0.08);
}

@media(max-width: 768px) {
    .model-divider { display: none; }
    .model-info-bar { flex-direction: column; align-items: stretch; }
    .model-stat { flex-direction: row; justify-content: space-between; padding: 4px 0; }
}

/* Buttons */
#analyze-btn {
    background: linear-gradient(135deg, #fb923c, #ea580c) !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 14px 20px !important;
    color: white !important;
    font-family: 'Outfit', sans-serif !important;
    font-size: 1.1rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.02em !important;
    box-shadow: 0 4px 15px rgba(234, 88, 12, 0.3) !important;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
    cursor: pointer !important;
}

#analyze-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(234, 88, 12, 0.45) !important;
    filter: brightness(1.08) !important;
}

/* Warning & Info Banners */
.warning-box {
    background: rgba(245, 158, 11, 0.1) !important;
    border-left: 4px solid #f59e0b !important;
    color: #f59e0b !important;
    padding: 12px 16px !important;
    border-radius: 8px !important;
    margin-top: 10px !important;
    font-size: 0.9rem !important;
}

.tip-box {
    background: rgba(59, 130, 246, 0.08) !important;
    border: 1px solid rgba(59, 130, 246, 0.15) !important;
    color: #93c5fd !important;
    padding: 14px 18px !important;
    border-radius: 12px !important;
    font-size: 0.95rem !important;
    line-height: 1.5 !important;
    margin-top: 12px !important;
    box-shadow: 0 4px 12px rgba(59, 130, 246, 0.05) !important;
}

/* Source badge */
.source-badge {
    background: rgba(255, 255, 255, 0.06);
    border: 1px solid rgba(255, 255, 255, 0.1);
    color: #cbd5e1;
    font-size: 0.75rem;
    font-weight: 600;
    padding: 3px 8px;
    border-radius: 20px;
    vertical-align: middle;
}

/* Card layout containers */
.card-header {
    margin-bottom: 16px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.06);
    padding-bottom: 10px;
}

.card-header h3 {
    font-family: 'Outfit', sans-serif;
    font-size: 1.3rem;
    font-weight: 600;
    color: #f3f4f6;
    margin: 0;
}

.nut-sub {
    font-size: 0.8rem;
    color: #6b7280;
    margin: 2px 0 0 0;
}

/* Nutrition Grid Styling */
.nutrition-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
    gap: 12px;
    margin-bottom: 20px;
}

.nut-item {
    background: rgba(31, 41, 55, 0.35);
    border: 1px solid rgba(255, 255, 255, 0.04);
    border-radius: 12px;
    padding: 12px 14px;
    display: flex;
    align-items: center;
    gap: 10px;
    transition: all 0.2s ease;
}

.nut-item:hover {
    background: rgba(31, 41, 55, 0.55);
    border-color: rgba(255, 255, 255, 0.08);
    transform: translateY(-1px);
}

.nut-item.cal {
    background: rgba(249, 115, 22, 0.05);
    border-color: rgba(249, 115, 22, 0.15);
}

.nut-item.cal:hover {
    background: rgba(249, 115, 22, 0.08);
    border-color: rgba(249, 115, 22, 0.25);
}

.nut-icon {
    font-size: 1.5rem;
    display: flex;
    align-items: center;
    justify-content: center;
    width: 36px;
    height: 36px;
    background: rgba(255, 255, 255, 0.03);
    border-radius: 50%;
}

.cal .nut-icon {
    background: rgba(249, 115, 22, 0.1);
}

.nut-label {
    display: block;
    font-size: 0.75rem;
    color: #9ca3af;
    font-weight: 500;
}

.nut-val {
    display: block;
    font-size: 1.1rem;
    font-weight: 700;
    color: #f3f4f6;
    margin-top: 1px;
}

.nut-val em {
    font-size: 0.8rem;
    font-style: normal;
    font-weight: 500;
    color: #6b7280;
    margin-left: 2px;
}

/* Health Score Dashboard */
.health-container {
    background: rgba(31, 41, 55, 0.2);
    border: 1px solid rgba(255, 255, 255, 0.04);
    border-radius: 16px;
    padding: 16px 20px;
    display: flex;
    flex-direction: column;
    gap: 14px;
}

.health-bar-wrap {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 20px;
    width: 100%;
}

.health-bar-bg {
    flex-grow: 1;
    height: 12px;
    background: rgba(255, 255, 255, 0.05);
    border-radius: 20px;
    overflow: hidden;
    position: relative;
}

.health-bar-fill {
    height: 100%;
    border-radius: 20px;
    transition: width 1s cubic-bezier(0.4, 0, 0.2, 1);
}

.health-num {
    font-family: 'Outfit', sans-serif;
    font-size: 1.8rem;
    font-weight: 700;
    white-space: nowrap;
    display: flex;
    align-items: baseline;
}

.health-denom {
    font-size: 1rem;
    color: #4b5563;
    font-weight: 500;
    margin-left: 1px;
}

.health-category {
    font-family: 'Outfit', sans-serif;
    font-size: 1.15rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    padding: 6px 14px;
    border-radius: 8px;
    border: 1px solid;
    background: rgba(255, 255, 255, 0.01);
    align-self: flex-start;
}
"""


# =========================================================================
# Gradio UI Layout Building
# =========================================================================

with gr.Blocks(
    title="Food Recognition & Nutrition AI Dashboard",
    css=custom_css,
    theme=gr.themes.Default(primary_hue="orange", secondary_hue="neutral")
) as app:

    gr.HTML("""
    <div id="hero">
        <h1>Food Recognition & Nutrition AI</h1>
        <p>Analyze food photography to identify dishes, fetch live nutritional logs from the USDA, and inspect WHO health scores.</p>
    </div>
    """)

    gr.HTML(MODEL_INFO)

    with gr.Row():

        with gr.Column(scale=4):
            image_input = gr.Image(
                sources=["upload", "webcam"],
                type="pil",
                height=320,
                label="🍽️ Choose Image Source"
            )

            predict_button = gr.Button("⚡ Analyze Food Dish", elem_id="analyze-btn")

            if SAMPLE_PATHS:
                gr.Examples(
                    examples=SAMPLE_PATHS,
                    inputs=image_input,
                    label="📸 Example Dishes"
                )

        with gr.Column(scale=6):
            with gr.Group():
                prediction_output = gr.Markdown()
                confidence_output = gr.Markdown()

                top3_output = gr.Dataframe(
                    headers=["Food Match", "Probability"],
                    row_count=3,
                    col_count=2,
                    label="📊 Top 3 Classifier Predictions"
                )

            with gr.Group():
                nutrition_output = gr.Markdown()

            with gr.Group():
                health_score_output = gr.Markdown()
                tip_output = gr.Markdown()

    error_output = gr.Markdown(visible=False)

    predict_button.click(
        fn=classify_food,
        inputs=image_input,
        outputs=[
            prediction_output,
            confidence_output,
            top3_output,
            nutrition_output,
            health_score_output,
            tip_output,
            error_output,
        ],
    )


if __name__ == "__main__":
    app.launch(
        show_error=True,
        server_name=os.getenv("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.getenv("GRADIO_SERVER_PORT", 7860)),
    )