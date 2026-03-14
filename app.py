import os
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

warnings.filterwarnings("ignore")

"""
app.py — Food Classification & Nutrition Analysis AI
"""

import urllib.request
import gradio as gr
from src.predict import predict_food
from src.config import FOOD_CLASSES


# --------------------------------------------------
# Prepare sample images
# --------------------------------------------------

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

print("Preparing sample images...")

for filename, url in SAMPLE_URLS.items():

    path = os.path.join(SAMPLES_DIR, filename)

    if not os.path.exists(path):

        print(f"Downloading sample: {filename}")

        try:
            urllib.request.urlretrieve(url, path)
        except Exception as e:
            print(f"Failed to download {filename}: {e}")
            continue

    SAMPLE_PATHS.append([path])

print("Sample images ready.")


# --------------------------------------------------
# Model info bar
# --------------------------------------------------

MODEL_INFO = f"""
<div class='model-info-bar'>
<div class='model-stat'><span class='ms-label'>Model</span><span class='ms-val'>MobileNetV2</span></div>
<div class='model-divider'></div>
<div class='model-stat'><span class='ms-label'>Dataset</span><span class='ms-val'>Food101</span></div>
<div class='model-divider'></div>
<div class='model-stat'><span class='ms-label'>Classes</span><span class='ms-val'>{len(FOOD_CLASSES)}</span></div>
<div class='model-divider'></div>
<div class='model-stat'><span class='ms-label'>Architecture</span><span class='ms-val'>Transfer Learning</span></div>
<div class='model-divider'></div>
<div class='model-stat'><span class='ms-label'>Input Size</span><span class='ms-val'>224 × 224</span></div>
</div>
"""


# --------------------------------------------------
# Prediction Function
# --------------------------------------------------

def classify_food(image):

    if image is None:
        return ("### ❌ Upload an image first.", "", [], "", "", "", "")

    try:

        result = predict_food(image)

        prediction = result["prediction"].replace("_", " ").title()
        confidence = round(result["confidence"] * 100, 2)
        warning = result.get("warning", "")

        pred_text = f"### {prediction}"

        if warning:
            pred_text += f"\n\n{warning}"

        conf_text = f"### {confidence}%"

        top3 = [
            [item["food"].replace("_", " ").title(), f"{round(item['confidence']*100,2)}%"]
            for item in result["top3"]
        ]

        nutrition = result["nutrition"]
        source = nutrition.get("source", "Estimated")

        source_badge = f"<span class='source-badge'>📡 {source}</span>"

        nutrition_text = f"""
### Nutrition Facts &nbsp;{source_badge}
<p class='nut-sub'>Per 100g serving</p>
<div class='nutrition-grid'>
<div class='nut-item cal'><span class='nut-icon'>🔥</span><div><span class='nut-label'>Calories</span><span class='nut-val'>{nutrition['calories']}<em>kcal</em></span></div></div>
<div class='nut-item'><span class='nut-icon'>💪</span><div><span class='nut-label'>Protein</span><span class='nut-val'>{nutrition['protein']}<em>g</em></span></div></div>
<div class='nut-item'><span class='nut-icon'>🍞</span><div><span class='nut-label'>Carbs</span><span class='nut-val'>{nutrition['carbs']}<em>g</em></span></div></div>
<div class='nut-item'><span class='nut-icon'>🧈</span><div><span class='nut-label'>Fat</span><span class='nut-val'>{nutrition['fat']}<em>g</em></span></div></div>
<div class='nut-item'><span class='nut-icon'>🌿</span><div><span class='nut-label'>Fiber</span><span class='nut-val'>{nutrition['fiber']}<em>g</em></span></div></div>
<div class='nut-item'><span class='nut-icon'>🍬</span><div><span class='nut-label'>Sugar</span><span class='nut-val'>{nutrition['sugar']}<em>g</em></span></div></div>
<div class='nut-item'><span class='nut-icon'>🧂</span><div><span class='nut-label'>Sodium</span><span class='nut-val'>{nutrition['sodium']}<em>mg</em></span></div></div>
</div>
"""

        health_score = result["health_score"]
        category = result["health_category"]
        color = result["health_color"]
        tip = result["tip"]

        bar_width = health_score * 10

        health_text = f"""
### Health Score
<div class='health-bar-wrap'>
<div class='health-bar-bg'>
<div class='health-bar-fill' style='width:{bar_width}%;background:{color};box-shadow:0 0 12px {color}88'></div>
</div>
<span class='health-num' style='color:{color}'>{health_score}<span class='health-denom'>/10</span></span>
</div>
<div class='health-category' style='color:{color}'>{category}</div>
"""

        tip_text = f"💡 {tip}"

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
        return (f"### ⚠️ {str(e)}", "", [], "", "", "", "")

    except Exception as e:
        return (f"### ❌ Prediction failed: {str(e)}", "", [], "", "", "", "")


# --------------------------------------------------
# CSS (same UI you built)
# --------------------------------------------------

custom_css = """
body {background:#080810;color:#e2e2e8;font-family:DM Sans}
#analyze-btn{background:linear-gradient(135deg,#fb923c,#ea580c);border:none;border-radius:12px;padding:15px;color:white;font-weight:700}
"""


# --------------------------------------------------
# Build Gradio UI
# --------------------------------------------------

with gr.Blocks(title="Food AI — Classification & Nutrition", css=custom_css) as app:

    gr.HTML("""
    <div id="hero">
    <h1>Food Recognition & Nutrition AI</h1>
    <p>Upload a food image to get AI prediction and nutrition analysis.</p>
    </div>
    """)

    gr.HTML(MODEL_INFO)

    with gr.Row():

        with gr.Column(scale=4):

            image_input = gr.Image(
                sources=["upload","webcam"],
                type="pil",
                height=300
            )

            predict_button = gr.Button("⚡ Analyze Food", elem_id="analyze-btn")

            if SAMPLE_PATHS:

                gr.Examples(
                    examples=SAMPLE_PATHS,
                    inputs=image_input,
                    label="🍽️ Click a sample image"
                )

        with gr.Column(scale=6):

            prediction_output = gr.Markdown()
            confidence_output = gr.Markdown()

            top3_output = gr.Dataframe(
                headers=["Food","Confidence"],
                row_count=3,
                col_count=2
            )

            nutrition_output = gr.Markdown()
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
        server_name="0.0.0.0",
        server_port=7860,
    )