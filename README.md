---
title: Food Classification
emoji: "🍕"
colorFrom: red
colorTo: yellow
sdk: gradio
sdk_version: "5.23.0"
python_version: "3.13"
app_file: app.py
pinned: false
---
# ≡ƒìö Food Recognition & Nutrition AI

> Deep learning model that identifies food from images and provides real-time nutritional analysis powered by USDA FoodData Central.

![Python](https://img.shields.io/badge/Python-3.13-blue?style=flat-square&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20-orange?style=flat-square&logo=tensorflow)
![Gradio](https://img.shields.io/badge/Gradio-5.23-purple?style=flat-square)
![Model](https://img.shields.io/badge/Model-MobileNetV2-green?style=flat-square)
![Dataset](https://img.shields.io/badge/Dataset-Food101-red?style=flat-square)

---

## ≡ƒô╕ Screenshots

![Hero](assets/screenshots/hero.png)

![Prediction](assets/screenshots/prediction.png)

![Nutrition](assets/screenshots/nutrition.png)

---

## Γ£¿ Features

- **AI Food Classification** ΓÇö MobileNetV2 Transfer Learning on Food101 dataset
- **97%+ Accuracy** ΓÇö on trained food classes
- **Real Nutrition Data** ΓÇö pulled live from USDA FoodData Central API
- **Health Scoring** ΓÇö multi-factor WHO-guideline based health score (1ΓÇô10)
- **Top 3 Predictions** ΓÇö with confidence percentages
- **Webcam Support** ΓÇö capture food in real time
- **Smart Error Handling** ΓÇö low confidence warnings, image validation

---

## ≡ƒÅù∩╕Å Architecture

```
Input Image (224├ù224)
        Γåô
MobileNetV2 (pretrained ImageNet, frozen)
        Γåô
GlobalAveragePooling2D
        Γåô
BatchNormalization ΓåÆ Dropout(0.3)
        Γåô
Dense(128, ReLU)
        Γåô
Dropout(0.15)
        Γåô
Dense(10, Softmax) ΓåÆ Food Class
```

**Two-phase training:**
- **Phase 1** ΓÇö Feature extraction (base frozen, 10 epochs, lr=1e-3)
- **Phase 2** ΓÇö Fine-tuning (top 30 layers unfrozen, 5 epochs, lr=1e-5)

---

## ≡ƒôè Model Performance

| Metric | Value |
|---|---|
| Architecture | MobileNetV2 + Custom Head |
| Dataset | Food101 (10 classes, 10% subset) |
| Input Size | 224 ├ù 224 ├ù 3 |
| Parameters | ~2.3M trainable |
| Top-1 Confidence | 97.97% (baklava), 81.18% (apple pie) |
| Training Strategy | Transfer Learning + Fine-tuning |

> ≡ƒÜº **Roadmap:** Retraining on all 101 classes currently in progress on Kaggle GPU.

---

## ≡ƒì╜∩╕Å Supported Food Classes (v1.0)

| # | Class | # | Class |
|---|---|---|---|
| 1 | Apple Pie | 6 | Beet Salad |
| 2 | Baby Back Ribs | 7 | Beignets |
| 3 | Baklava | 8 | Bibimbap |
| 4 | Beef Carpaccio | 9 | Bread Pudding |
| 5 | Beef Tartare | 10 | Bruschetta |

---

## ≡ƒÜÇ Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/adithyavaddadi/food-classification.git
cd food-classification
```

### 2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
tensorflow-cpu==2.21.0
keras==3.12.0
tensorflow-datasets==4.9.4
numpy==2.1.0
pandas==2.2.3
scikit-learn==1.5.2
matplotlib==3.10.0
seaborn==0.13.2
requests==2.31.0
python-dotenv==1.0.1
Pillow==10.4.0
gradio==5.23.0
```

### 4. Add your trained model
```
results/model/food_classifier.h5
```

### 5. Run the app
```bash
python app.py
```
Open **http://localhost:7860**

---

## ≡ƒôü Project Structure

```
food-classification/
Γö£ΓöÇΓöÇ app.py                  # Gradio UI
Γö£ΓöÇΓöÇ requirements.txt
Γö£ΓöÇΓöÇ README.md
Γö£ΓöÇΓöÇ assets/
Γöé   ΓööΓöÇΓöÇ screenshots/        # Demo screenshots
Γö£ΓöÇΓöÇ src/
Γöé   Γö£ΓöÇΓöÇ config.py           # Hyperparameters & paths
Γöé   Γö£ΓöÇΓöÇ dataset.py          # Food101 data loading
Γöé   Γö£ΓöÇΓöÇ model.py            # MobileNetV2 architecture
Γöé   Γö£ΓöÇΓöÇ train.py            # Training pipeline
Γöé   Γö£ΓöÇΓöÇ predict.py          # Inference + error handling
Γöé   Γö£ΓöÇΓöÇ evaluate.py         # Metrics & confusion matrix
Γöé   ΓööΓöÇΓöÇ nutrition.py        # USDA API + health scoring
Γö£ΓöÇΓöÇ data/
Γöé   ΓööΓöÇΓöÇ food101/            # Dataset (auto-downloaded)
ΓööΓöÇΓöÇ results/
    Γö£ΓöÇΓöÇ model/              # Saved model weights
    Γö£ΓöÇΓöÇ plots/              # Training curves
    ΓööΓöÇΓöÇ logs/               # TensorBoard logs
```

---

## ≡ƒºá Tech Stack

| Component | Technology |
|---|---|
| Deep Learning | TensorFlow 2.15 + Keras |
| Base Model | MobileNetV2 (ImageNet pretrained) |
| Dataset | Food101 via TensorFlow Datasets |
| UI | Gradio 6.0 |
| Nutrition API | USDA FoodData Central |
| Language | Python 3.11 |

---

## ≡ƒôê Training Details

```python
# Phase 1 ΓÇö Feature Extraction
optimizer = Adam(lr=1e-3)
epochs    = 10
frozen    = all MobileNetV2 layers

# Phase 2 ΓÇö Fine Tuning
optimizer = Adam(lr=1e-5)
epochs    = 5
unfrozen  = top 30 layers of MobileNetV2
```

Callbacks: `EarlyStopping`, `ModelCheckpoint`, `ReduceLROnPlateau`, `TensorBoard`

---

## ≡ƒö« Roadmap

- [x] 10-class Food101 classifier
- [x] Real nutrition data via USDA API
- [x] Health scoring algorithm
- [x] Gradio web UI with webcam support
- [ ] Retrain on all 101 Food101 classes
- [ ] Deploy on Hugging Face Spaces
- [ ] Add meal logging feature
- [ ] Daily nutrition tracker

---

## ≡ƒæñ Author

**Adithya**
- GitHub: [@adithyavaddadi](https://github.com/adithyavaddadi)
- LinkedIn: [your-linkedin](https://www.linkedin.com/in/adithya-vaddadi-536176330/)

---

## ≡ƒôä License

MIT License ΓÇö feel free to use and modify.

---

<p align="center">Built with Γ¥ñ∩╕Å using TensorFlow ┬╖ MobileNetV2 ┬╖ Gradio ┬╖ USDA FoodData Central</p>

