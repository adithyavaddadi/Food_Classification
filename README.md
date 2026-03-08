---
title: Food Classification
emoji: 🍕
colorFrom: red
colorTo: yellow
sdk: gradio
sdk_version: "5.23.0"
python_version: "3.13"
app_file: app.py
pinned: false
---
# 🍔 Food Recognition & Nutrition AI

> Deep learning model that identifies food from images and provides real-time nutritional analysis powered by USDA FoodData Central.

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange?style=flat-square&logo=tensorflow)
![Gradio](https://img.shields.io/badge/Gradio-UI-purple?style=flat-square)
![Model](https://img.shields.io/badge/Model-MobileNetV2-green?style=flat-square)
![Dataset](https://img.shields.io/badge/Dataset-Food101-red?style=flat-square)

---

## 📸 Screenshots

![Hero](assets/screenshots/hero.png)

![Prediction](assets/screenshots/prediction.png)

![Nutrition](assets/screenshots/nutrition.png)

---

## ✨ Features

- **AI Food Classification** — MobileNetV2 Transfer Learning on Food101 dataset
- **97%+ Accuracy** — on trained food classes
- **Real Nutrition Data** — pulled live from USDA FoodData Central API
- **Health Scoring** — multi-factor WHO-guideline based health score (1–10)
- **Top 3 Predictions** — with confidence percentages
- **Webcam Support** — capture food in real time
- **Smart Error Handling** — low confidence warnings, image validation

---

## 🏗️ Architecture

```
Input Image (224×224)
        ↓
MobileNetV2 (pretrained ImageNet, frozen)
        ↓
GlobalAveragePooling2D
        ↓
BatchNormalization → Dropout(0.3)
        ↓
Dense(128, ReLU)
        ↓
Dropout(0.15)
        ↓
Dense(10, Softmax) → Food Class
```

**Two-phase training:**
- **Phase 1** — Feature extraction (base frozen, 10 epochs, lr=1e-3)
- **Phase 2** — Fine-tuning (top 30 layers unfrozen, 5 epochs, lr=1e-5)

---

## 📊 Model Performance

| Metric | Value |
|---|---|
| Architecture | MobileNetV2 + Custom Head |
| Dataset | Food101 (10 classes, 10% subset) |
| Input Size | 224 × 224 × 3 |
| Parameters | ~2.3M trainable |
| Top-1 Confidence | 97.97% (baklava), 81.18% (apple pie) |
| Training Strategy | Transfer Learning + Fine-tuning |

> 🚧 **Roadmap:** Retraining on all 101 classes currently in progress on Kaggle GPU.

---

## 🍽️ Supported Food Classes (v1.0)

| # | Class | # | Class |
|---|---|---|---|
| 1 | Apple Pie | 6 | Beet Salad |
| 2 | Baby Back Ribs | 7 | Beignets |
| 3 | Baklava | 8 | Bibimbap |
| 4 | Beef Carpaccio | 9 | Bread Pudding |
| 5 | Beef Tartare | 10 | Bruschetta |

---

## 🚀 Quick Start

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

## 📁 Project Structure

```
food-classification/
├── app.py                  # Gradio UI
├── requirements.txt
├── README.md
├── assets/
│   └── screenshots/        # Demo screenshots
├── src/
│   ├── config.py           # Hyperparameters & paths
│   ├── dataset.py          # Food101 data loading
│   ├── model.py            # MobileNetV2 architecture
│   ├── train.py            # Training pipeline
│   ├── predict.py          # Inference + error handling
│   ├── evaluate.py         # Metrics & confusion matrix
│   └── nutrition.py        # USDA API + health scoring
├── data/
│   └── food101/            # Dataset (auto-downloaded)
└── results/
    ├── model/              # Saved model weights
    ├── plots/              # Training curves
    └── logs/               # TensorBoard logs
```

---

## 🧠 Tech Stack

| Component | Technology |
|---|---|
| Deep Learning | TensorFlow 2.15 + Keras |
| Base Model | MobileNetV2 (ImageNet pretrained) |
| Dataset | Food101 via TensorFlow Datasets |
| UI | Gradio 6.0 |
| Nutrition API | USDA FoodData Central |
| Language | Python 3.11 |

---

## 📈 Training Details

```python
# Phase 1 — Feature Extraction
optimizer = Adam(lr=1e-3)
epochs    = 10
frozen    = all MobileNetV2 layers

# Phase 2 — Fine Tuning
optimizer = Adam(lr=1e-5)
epochs    = 5
unfrozen  = top 30 layers of MobileNetV2
```

Callbacks: `EarlyStopping`, `ModelCheckpoint`, `ReduceLROnPlateau`, `TensorBoard`

---

## 🔮 Roadmap

- [x] 10-class Food101 classifier
- [x] Real nutrition data via USDA API
- [x] Health scoring algorithm
- [x] Gradio web UI with webcam support
- [ ] Retrain on all 101 Food101 classes
- [ ] Deploy on Hugging Face Spaces
- [ ] Add meal logging feature
- [ ] Daily nutrition tracker

---

## 👤 Author

**Adithya**
- GitHub: [@adithyavaddadi](https://github.com/adithyavaddadi)
- LinkedIn: [your-linkedin](https://www.linkedin.com/in/adithya-vaddadi-536176330/)

---

## 📄 License

MIT License — feel free to use and modify.

---

<p align="center">Built with ❤️ using TensorFlow · MobileNetV2 · Gradio · USDA FoodData Central</p>