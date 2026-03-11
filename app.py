from src.predict import predict_food
from src.config import FOOD_CLASSES
import os
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore')

"""
app.py — Food Classification & Nutrition Analysis AI
"""

import gradio as gr
from src.predict import predict_food
from src.config import FOOD_CLASSES


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

# Sample images (URL-based, no upload needed)
SAMPLE_IMAGES = [
    ["https://upload.wikimedia.org/wikipedia/commons/thumb/a/a3/Eq_it-na_pizza-margherita_sep2005_sml.jpg/800px-Eq_it-na_pizza-margherita_sep2005_sml.jpg"],
    ["https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Biryani_at_Hyderabad%2CIndia.jpg/800px-Biryani_at_Hyderabad%2CIndia.jpg"],
    ["https://upload.wikimedia.org/wikipedia/commons/thumb/6/6d/Good_Food_Display_-_NCI_Visuals_Online.jpg/800px-Good_Food_Display_-_NCI_Visuals_Online.jpg"],
    ["https://upload.wikimedia.org/wikipedia/commons/thumb/4/45/A_small_cup_of_coffee.JPG/800px-A_small_cup_of_coffee.JPG"],
    ["https://upload.wikimedia.org/wikipedia/commons/thumb/9/90/Hapus_Mango.jpg/800px-Hapus_Mango.jpg"],
    ["https://upload.wikimedia.org/wikipedia/commons/thumb/4/4d/Chocolate_covered_strawberries.jpg/800px-Chocolate_covered_strawberries.jpg"],
]


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
            [item["food"].replace("_", " ").title(), f"{round(item['confidence'] * 100, 2)}%"]
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
<div class='health-bar-bg'><div class='health-bar-fill' style='width:{bar_width}%;background:{color};box-shadow:0 0 12px {color}88'></div></div>
<span class='health-num' style='color:{color}'>{health_score}<span class='health-denom'>/10</span></span>
</div>
<div class='health-category' style='color:{color}'>{category}</div>
"""

        tip_text = f"💡 {tip}"

        return (pred_text, conf_text, top3, nutrition_text, health_text, tip_text, "")

    except ValueError as e:
        return (f"### ⚠️ {str(e)}", "", [], "", "", "", "")
    except Exception as e:
        return (f"### ❌ Prediction failed: {str(e)}", "", [], "", "", "", "")


custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;1,9..40,300&display=swap');

*, *::before, *::after { box-sizing: border-box; }

body, .gradio-container {
background: #080810 !important;
font-family: 'DM Sans', sans-serif !important;
color: #e2e2e8 !important;
min-height: 100vh;
}

.gradio-container { max-width: 1240px !important; margin: 0 auto !important; padding: 0 20px !important; }

/* ---- HERO ---- */
#hero {
text-align: center;
padding: 56px 20px 32px;
position: relative;
overflow: hidden;
}
#hero::after {
content: '';
position: absolute;
top: -60px; left: 50%; transform: translateX(-50%);
width: 700px; height: 400px;
background: radial-gradient(ellipse, rgba(251,146,60,0.10) 0%, transparent 65%);
pointer-events: none; z-index: 0;
}
.hero-eyebrow {
display: inline-flex; align-items: center; gap: 6px;
background: rgba(251,146,60,0.08);
border: 1px solid rgba(251,146,60,0.25);
color: #fb923c;
font-size: 0.7em; font-weight: 600; letter-spacing: 0.18em;
text-transform: uppercase; padding: 5px 16px;
border-radius: 100px; margin-bottom: 20px;
position: relative; z-index: 1;
}
.hero-title {
font-family: 'Syne', sans-serif !important;
font-size: clamp(2.2em, 5.5vw, 3.6em) !important;
font-weight: 800 !important; line-height: 1.08 !important;
letter-spacing: -0.025em;
background: linear-gradient(150deg, #ffffff 30%, #fdba74 70%, #fb923c 100%);
-webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
margin: 0 0 16px; position: relative; z-index: 1;
}
.hero-sub {
color: #5a5a72; font-size: 1em; font-weight: 300;
max-width: 400px; margin: 0 auto; line-height: 1.65;
position: relative; z-index: 1;
}

/* ---- MODEL INFO BAR ---- */
.model-info-bar {
display: flex; align-items: center; justify-content: center;
gap: 0; flex-wrap: wrap;
background: rgba(255,255,255,0.025);
border: 1px solid rgba(255,255,255,0.07);
border-radius: 12px; padding: 12px 24px;
margin: 24px 0 32px;
}
.model-stat { display: flex; flex-direction: column; align-items: center; padding: 0 20px; }
.ms-label { font-size: 0.6em; font-weight: 600; letter-spacing: 0.14em; text-transform: uppercase; color: #4a4a62; margin-bottom: 3px; }
.ms-val { font-family: 'Syne', sans-serif; font-weight: 700; font-size: 0.9em; color: #c8c8d8; }
.model-divider { width: 1px; height: 28px; background: rgba(255,255,255,0.06); }

/* ---- SECTION LABELS ---- */
.section-label {
font-size: 0.62em; font-weight: 700; letter-spacing: 0.18em;
text-transform: uppercase; color: #3e3e56;
margin-bottom: 10px; padding-bottom: 8px;
border-bottom: 1px solid rgba(255,255,255,0.04);
}

/* ---- CARDS ---- */
.card {
background: rgba(255,255,255,0.022);
border: 1px solid rgba(255,255,255,0.06);
border-radius: 16px; padding: 20px;
transition: border-color 0.25s;
}
.card:hover { border-color: rgba(251,146,60,0.18); }

/* ---- RESULT CARDS ---- */
.pred-card, .conf-card {
background: rgba(255,255,255,0.022) !important;
border: 1px solid rgba(255,255,255,0.06) !important;
border-radius: 14px !important; padding: 18px 20px !important;
}
.pred-label {
font-size: 0.6em; font-weight: 700; letter-spacing: 0.18em;
text-transform: uppercase; color: #fb923c;
margin-bottom: 6px; display: block;
}
.pred-card h3 {
font-family: 'Syne', sans-serif !important;
font-weight: 800 !important; font-size: 1.5em !important;
color: #fff !important; margin: 0 !important; line-height: 1.2 !important;
}
.conf-card h3 {
font-family: 'Syne', sans-serif !important;
font-size: 2.4em !important; font-weight: 800 !important;
background: linear-gradient(135deg, #fb923c, #fbbf24) !important;
-webkit-background-clip: text !important; -webkit-text-fill-color: transparent !important;
background-clip: text !important; margin: 0 !important;
}

/* ---- ANALYZE BUTTON ---- */
#analyze-btn {
background: linear-gradient(135deg, #fb923c 0%, #ea580c 100%) !important;
border: none !important; border-radius: 12px !important;
font-family: 'Syne', sans-serif !important; font-weight: 700 !important;
font-size: 1em !important; letter-spacing: 0.06em !important;
color: #fff !important; padding: 15px !important;
width: 100% !important; margin-top: 14px !important;
box-shadow: 0 4px 24px rgba(251,146,60,0.28) !important;
transition: all 0.2s ease !important; cursor: pointer !important;
}
#analyze-btn:hover {
box-shadow: 0 6px 36px rgba(251,146,60,0.45) !important;
transform: translateY(-2px) !important;
}

/* ---- NUTRITION ---- */
.source-badge {
display: inline-block;
background: rgba(34,197,94,0.1); border: 1px solid rgba(34,197,94,0.25);
color: #4ade80; font-size: 0.65em; font-weight: 600;
letter-spacing: 0.08em; padding: 2px 10px; border-radius: 100px;
vertical-align: middle; margin-left: 6px;
}
.nut-sub { color: #4a4a62 !important; font-size: 0.75em !important; margin: -4px 0 8px !important; }
.nutrition-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 7px; margin-top: 10px; }
.nut-item {
display: flex; align-items: center; gap: 10px;
background: rgba(255,255,255,0.025); border-radius: 10px;
padding: 9px 12px; border: 1px solid rgba(255,255,255,0.05);
transition: background 0.2s;
}
.nut-item:hover { background: rgba(255,255,255,0.04); }
.nut-item.cal { grid-column: 1 / -1; background: rgba(251,146,60,0.06); border-color: rgba(251,146,60,0.15); }
.nut-icon { font-size: 1.1em; }
.nut-item > div { display: flex; flex-direction: column; }
.nut-label { font-size: 0.7em; color: #5a5a72; font-weight: 500; }
.nut-val { font-family: 'Syne', sans-serif; font-weight: 700; font-size: 0.95em; color: #e8e8f0; }
.nut-val em { font-style: normal; font-weight: 400; font-size: 0.78em; color: #4a4a62; margin-left: 2px; }

/* ---- HEALTH SCORE ---- */
.health-bar-wrap { display: flex; align-items: center; gap: 14px; margin-top: 12px; }
.health-bar-bg { flex: 1; height: 10px; background: rgba(255,255,255,0.06); border-radius: 100px; overflow: hidden; }
.health-bar-fill { height: 100%; border-radius: 100px; transition: width 0.8s cubic-bezier(0.34,1.56,0.64,1); }
.health-num { font-family: 'Syne', sans-serif; font-weight: 800; font-size: 1.8em; min-width: 72px; }
.health-denom { font-size: 0.45em; opacity: 0.5; font-weight: 400; }
.health-category { font-family: 'Syne', sans-serif; font-weight: 700; font-size: 0.75em; letter-spacing: 0.12em; text-transform: uppercase; margin-top: 6px; }

/* ---- TIP BOX ---- */
.tip-wrap {
background: rgba(251,146,60,0.05);
border: 1px solid rgba(251,146,60,0.15);
border-left: 3px solid #fb923c;
border-radius: 10px; padding: 14px 16px;
color: #fcd9aa !important; font-size: 0.9em; line-height: 1.6;
}

/* ---- EXAMPLES ---- */
.examples-section {
margin-top: 16px;
}
.gr-examples .label {
color: #fb923c !important;
font-size: 0.65em !important;
font-weight: 700 !important;
letter-spacing: 0.14em !important;
text-transform: uppercase !important;
}
.gr-samples-table img {
border-radius: 10px !important;
border: 2px solid rgba(255,255,255,0.06) !important;
transition: border-color 0.2s, transform 0.2s !important;
cursor: pointer !important;
}
.gr-samples-table img:hover {
border-color: #fb923c !important;
transform: scale(1.04) !important;
}

/* ---- FOOTER ---- */
#footer {
text-align: center; padding: 36px 20px;
color: #2e2e42; font-size: 0.78em;
border-top: 1px solid rgba(255,255,255,0.04); margin-top: 48px;
}
#footer a { color: #fb923c; text-decoration: none; }
#footer a:hover { text-decoration: underline; }

/* ---- GRADIO OVERRIDES ---- */
footer { display: none !important; }
.block { background: transparent !important; border: none !important; padding: 0 !important; box-shadow: none !important; }
label { color: #4a4a62 !important; font-size: 0.72em !important; letter-spacing: 0.08em !important; font-weight: 600 !important; }
.gradio-container .prose h3 { color: #fff !important; font-family: 'Syne', sans-serif !important; }
"""


with gr.Blocks(title="Food AI — Classification & Nutrition", css=custom_css) as app:

    gr.HTML("""
    <div id="hero">
      <div class="hero-eyebrow">✦ Deep Learning · Computer Vision</div>
      <h1 class="hero-title">Food Recognition<br>& Nutrition AI</h1>
      <p class="hero-sub">Upload any food photo for instant AI-powered identification and real nutritional data.</p>
    </div>
    """)

    gr.HTML(MODEL_INFO)

    with gr.Row(equal_height=False):

        # Left — Input
        with gr.Column(scale=4):
            gr.HTML('<div class="section-label">📷 Upload Image</div>')
            image_input = gr.Image(
                sources=["upload", "webcam"],
                type="pil",
                label="",
                height=300,
                show_label=False,
            )
            predict_button = gr.Button("⚡ Analyze Food", variant="primary", elem_id="analyze-btn")

            # Sample images — click to test instantly
            gr.Examples(
                examples=SAMPLE_IMAGES,
                inputs=image_input,
                label="🍽️ No image? Click a sample below to test instantly",
                examples_per_page=6,
            )

        # Right — Results
        with gr.Column(scale=6):
            gr.HTML('<div class="section-label">🎯 Prediction</div>')
            with gr.Row(equal_height=True):
                with gr.Column(scale=3, elem_classes=["pred-card"]):
                    gr.HTML('<span class="pred-label">Detected Food</span>')
                    prediction_output = gr.Markdown()
                with gr.Column(scale=2, elem_classes=["conf-card"]):
                    gr.HTML('<span class="pred-label">Confidence</span>')
                    confidence_output = gr.Markdown()

            gr.HTML('<div style="height:12px"></div>')
            gr.HTML('<div class="section-label">🏆 Top 3 Predictions</div>')
            top3_output = gr.Dataframe(
                headers=["Food", "Confidence"],
                label="",
                row_count=3,
                col_count=2,
                show_label=False,
            )

            gr.HTML('<div style="height:20px"></div>')
            gr.HTML('<div class="section-label">🥗 Nutrition & Health</div>')

            with gr.Row():
                with gr.Column(scale=5, elem_classes=["card"]):
                    nutrition_output = gr.Markdown()
                with gr.Column(scale=5):
                    with gr.Column(elem_classes=["card"]):
                        health_score_output = gr.Markdown()
                    gr.HTML('<div style="height:10px"></div>')
                    with gr.Column(elem_classes=["tip-wrap"]):
                        tip_output = gr.Markdown()

    # Hidden error output
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

    gr.HTML("""
    <div id="footer">
      TensorFlow &nbsp;·&nbsp; MobileNetV2 &nbsp;·&nbsp; Gradio &nbsp;·&nbsp; USDA FoodData Central &nbsp;·&nbsp; Food101
      <br><br>
      <a href="https://github.com" target="_blank">View Source on GitHub →</a>
    </div>
    """)


if __name__ == "__main__":
    app.launch(
        show_error=True,
        server_name="0.0.0.0",
        server_port=7860,
    )