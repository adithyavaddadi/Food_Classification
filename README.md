---
title: Food Classification
emoji: 🍕
colorFrom: red
colorTo: yellow
sdk: gradio
sdk_version: "5.23.0"
python_version: "3.10"
app_file: app.py
pinned: false
---

# 🍕 Food Recognition & Nutrition AI

> A state-of-the-art deep learning classifier that identifies dishes from photos and delivers real-time, comprehensive nutritional profiling and WHO-based dietary health scoring. Built for local development and optimized for seamless deployment on Hugging Face Spaces.

[![Python Version](https://img.shields.io/badge/Python-3.10%20%7C%203.13-blue?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16%2B-orange?style=flat-square&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Gradio](https://img.shields.io/badge/Gradio-5.23%2B-purple?style=flat-square&logo=gradio&logoColor=white)](https://gradio.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=flat-square)](LICENSE)

---

## 📸 Dashboard Screenshots

![Hero](assets/screenshots/hero.png)

![Prediction](assets/screenshots/prediction.png)

![Nutrition](assets/screenshots/nutrition.png)

---

## ✨ Features

- **🧠 Deep Learning Classification** — MobileNetV2 architecture with a custom classification head fine-tuned on the Food101 dataset (10 targeted classes).
- **🚀 Two-Phase Transfer Learning** — Initial feature extraction phase with frozen base layers, followed by a meticulous fine-tuning phase of top layers using a decay-tuned learning rate.
- **📡 USDA API Integration** — Live retrieval of micro and macronutrient content from the official USDA FoodData Central API, with Open Food Facts and local databases as automated fallbacks.
- **📊 WHO-based Health Scoring** — A robust health scoring algorithm (1 to 10 scale) implementing dietary limits set by the World Health Organization for sodium, sugars, and lipids, paired with actionable advice.
- **⚡ Lazy Caching Optimization** — Smart lazy model caching at the prediction layer, preventing Hugging Face Spaces startup timeouts and saving container resources.
- **🎯 Smart Robust Design** — Dynamic validation checks on image format/resolution, warning signals for low confidence matches, and clean webcam support.

---

## 🏗️ Neural Network Architecture

```text
Input Image (224 × 224 × 3)
        ↓
MobileNetV2 (Pretrained on ImageNet, base feature extractor)
        ↓
Global Average Pooling 2D
        ↓
Batch Normalization (stabilizes dense layer inputs)
        ↓
Dropout (Rate: 0.3)
        ↓
Dense Layer (128 Units, ReLU Activation)
        ↓
Dropout (Rate: 0.15)
        ↓
Dense Output Layer (10 Units, Softmax Activation) → Predicted Dish Match
```

### Training Strategy:
1. **Phase 1 (Feature Extraction)**: All base MobileNetV2 layers frozen. Adam optimizer with $LR = 10^{-3}$ trained for 10 epochs.
2. **Phase 2 (Fine-Tuning)**: Top 30 layers of the base MobileNetV2 unfrozen. Adam optimizer with a precise decay rate $LR = 10^{-5}$ trained for 5 epochs.

---

## 🍽️ Supported Food Classes

The classifier is tuned to identify the following 10 popular dishes:

| # | Class Name | # | Class Name |
|:-:|---|:-:|---|
| **1** | Apple Pie | **6** | Beet Salad |
| **2** | Baby Back Ribs | **7** | Beignets |
| **3** | Baklava | **8** | Bibimbap |
| **4** | Beef Carpaccio | **9** | Bread Pudding |
| **5** | Beef Tartare | **10** | Bruschetta |

---

## 📂 Production Directory Tree

```text
food-classification/
├── app.py                # Main Gradio application dashboard
├── requirements.txt      # Production package dependencies
├── .gitignore            # Clean git exclusion rules
├── .env.template         # Environment variables template
├── README.md             # Space/GitHub homepage documentation
├── assets/
│   └── screenshots/      # Demo UI visualization pictures
├── src/
│   ├── __init__.py
│   ├── config.py         # Global variables, hyperparams, & directories
│   ├── dataset.py        # Food101 pipeline loading and parallel scaling
│   ├── model.py          # MobileNetV2 Keras structure & fine-tune toggling
│   ├── train.py          # Two-phase training pipeline execution
│   ├── evaluate.py       # Metrics evaluation and confusion matrix generator
│   ├── predict.py        # Lazy loading classification and validation
│   └── nutrition.py      # USDA API connecting and health scoring logic
├── data/
│   └── samples/          # Cached UI sample photos
└── results/
    ├── model/            # Serialized food_classifier.h5 weights
    ├── plots/            # Train curves and evaluation figures
    └── logs/             # TensorBoard diagnostics logging
```

---

## 🚀 Local Quick Start

Follow these simple steps to spin up the dashboard on your machine:

### 1. Clone the Repository
```bash
git clone https://github.com/adithyavaddadi/Food_Classification.git
cd Food_Classification
```

### 2. Configure Your Environment
Create a virtual environment to isolate package configurations:
```bash
# Create environment
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies
Install all required libraries including Keras and CPU-optimized TensorFlow:
```bash
pip install -r requirements.txt
```

### 4. Setup API Credentials (Optional)
Obtain a free API Key from the [USDA FoodData Central Portal](https://api.nal.usda.gov/).
Create a `.env` file in the root directory:
```bash
copy .env.template .env
```
Open `.env` and fill in your credential:
```env
USDA_API_KEY=YOUR_ACTUAL_USDA_KEY_HERE
```
*Note: If no `.env` is created, the application naturally defaults to `DEMO_KEY`.*

### 5. Add Your Trained Model
Ensure you place your trained `food_classifier.h5` inside the `results/model/` directory:
```text
results/model/food_classifier.h5
```

### 6. Run the App
Launch the Gradio server:
```bash
python app.py
```
Open your browser and navigate to **[http://localhost:7860](http://localhost:7860)**.

---

## 📈 Model Performance & Metrics

| Evaluation Metric | Value |
|---|---|
| **Base Classifier** | MobileNetV2 (ImageNet Pretrained) |
| **Input Shape** | 224 × 224 × 3 |
| **Trainable Weights** | ~2.3M parameters |
| **Accuracy Score** | 87% overall validation accuracy (after Phase 2) |
| **Highest Class Recall** | Baklava (~97.97%) |
| **Validation Strategy** | Sparse Categorical Crossentropy |

*Metrics visualizations are automatically saved to `results/plots/` during the training and validation loops.*

---

## ☁️ Hugging Face Spaces Deployment

To deploy this application to Hugging Face Spaces:
1. Create a new **Gradio SDK Space** in your Hugging Face account.
2. Push this repository's codebase including `app.py`, `src/`, `requirements.txt`, and the model file.
3. Add your `USDA_API_KEY` to the Space's **Variables and Secrets** settings to ensure high API quota availability.

---

## 🔮 Project Roadmap

- [x] Two-Phase Transfer Learning (MobileNetV2 subset)
- [x] Live USDA API Integration & fallback algorithms
- [x] WHO Health Scoring system
- [x] Elegant Gradio dashboard (Webcam + Example inputs)
- [x] Full production cleanup and typing documentation
- [x] Hugging Face Space optimization layout
- [ ] Retrain classifier on all 101 classes of Food101 dataset
- [ ] Add historical meal logging with SQLite caching
- [ ] Implement daily diet tracking charts

---

## 👤 Credits & Author

Developed with ❤️ by **Adithya**
- **GitHub**: [@adithyavaddadi](https://github.com/adithyavaddadi)
- **LinkedIn**: [Adithya Vaddadi](https://www.linkedin.com/in/adithya-vaddadi-536176330/)

---

## 📄 License

This project is licensed under the MIT License — feel free to utilize, modify, and distribute as desired.

<p align="center">Built with TensorFlow • MobileNetV2 • Gradio • USDA FoodData Central</p>
