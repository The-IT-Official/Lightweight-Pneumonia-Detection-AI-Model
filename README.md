# 🫁 Lightweight Pneumonia Detection AI Model

> Deep learning model developed in collaboration with Stanford faculty to enable multi-level pneumonia classification in under-resourced hospitals. Optimized for deployment in environments with limited compute, minimal infrastructure, and low-bandwidth constraints.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org)
[![Dataset](https://img.shields.io/badge/Dataset-RSNA%20Pneumonia-green.svg)](https://www.kaggle.com/datasets/iamtapendu/rsna-pneumonia-processed-dataset)

---

## 📌 Overview

Pneumonia is one of the leading causes of death globally, yet many hospitals — particularly in low-income regions — lack the radiological infrastructure to diagnose it reliably. This project addresses that gap by building a **lightweight, high-accuracy pneumonia detection model** that can be deployed under real-world resource constraints.

The model uses transfer learning on **DenseNet-121** (pretrained on ImageNet) fine-tuned on the **RSNA Pneumonia Detection Dataset**, with a focus on:

- High **recall** (minimizing missed diagnoses)
- Strong **AUC-ROC** performance
- Efficient inference suitable for edge or low-resource environments

---

## 🧠 Model Architecture

| Component | Detail |
|---|---|
| Base Model | DenseNet-121 (ImageNet pretrained) |
| Classifier Head | Linear(1024 → 1) |
| Loss Function | BCEWithLogitsLoss (with pos_weight for class imbalance) |
| Optimizer | Adam (lr=1e-3) |
| Input Size | 224×224 RGB |
| Output | Binary (Pneumonia / Normal) |

The feature extractor is **frozen** during training — only the classifier head is trained. This makes the model significantly faster to train and more deployable on constrained hardware.

---

## 📂 Dataset

**RSNA Pneumonia Detection Challenge** (processed version via Kaggle)

- ~26,000 chest X-ray images
- Binary labels: `0` = Normal, `1` = Pneumonia
- Duplicate patient IDs dropped (some patients have multiple bounding boxes)
- 80/20 train/val split on training data
- Separate held-out test set

```python
import kagglehub
path = kagglehub.dataset_download("iamtapendu/rsna-pneumonia-processed-dataset")
```

---

## ⚙️ Setup & Installation

### Requirements

```bash
pip install torch torchvision kagglehub pandas Pillow scikit-learn
```

Or install from `requirements.txt`:

```bash
pip install -r requirements.txt
```

### Requirements.txt

```
torch
torchvision
kagglehub
pandas
Pillow
scikit-learn
```

---

## 🚀 Usage

### Run in Google Colab (Recommended)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

1. Open `pneumonia_model.ipynb` in Colab
2. Run cells top to bottom
3. Dataset downloads automatically via `kagglehub`

### Run Locally

```bash
git clone https://github.com/The-IT-Official/Lightweight-Pneumonia-Detection-AI-Model.git
cd Lightweight-Pneumonia-Detection-AI-Model
pip install -r requirements.txt
jupyter notebook pneumonia_model.ipynb
```

---

## 📊 Training Pipeline

```
Data Download (kagglehub)
        ↓
Preprocessing (Resize 224x224, Grayscale→RGB, Normalize)
        ↓
Dataset Split (80% train / 20% val + separate test)
        ↓
DenseNet-121 (frozen features, trainable classifier)
        ↓
Training Loop (7 epochs, BCEWithLogitsLoss + pos_weight)
        ↓
Evaluation (Acc, Precision, Recall, F1, AUC, Specificity, Confusion Matrix)
```

---

## 📈 Evaluation Metrics

The model is evaluated on the following metrics after each epoch:

| Metric | Description |
|---|---|
| Accuracy | Overall correct predictions |
| Precision | Of predicted positives, how many are correct |
| Recall | Of actual positives, how many were caught |
| F1 Score | Harmonic mean of precision and recall |
| AUC-ROC | Area under the ROC curve |
| Specificity | True negative rate |
| Confusion Matrix | TP / TN / FP / FN breakdown |

> **Recall is prioritized** — in clinical settings, missing a pneumonia case (false negative) is far more costly than a false alarm.

---

## 🗂️ Project Structure

```
Lightweight-Pneumonia-Detection-AI-Model/
│
├── pneumonia_model.ipynb     # Main training notebook (Colab-ready)
├── requirements.txt          # Python dependencies
├── README.md                 # You're here
└── LICENSE                   # MIT License
```

---

## 🔭 Roadmap

- [x] Binary classification (Pneumonia vs Normal)
- [ ] Multi-class classification (Normal / Bacterial / Viral)
- [ ] Model quantization for edge deployment (INT8)
- [ ] ONNX export for cross-platform inference
- [ ] REST API wrapper for hospital integration
- [ ] Grad-CAM visualizations for explainability
- [ ] arXiv paper submission

---

## 🤝 Collaboration

This project is developed in collaboration with Stanford faculty as part of ongoing research into AI-assisted diagnostics for under-resourced healthcare environments.

If you're a researcher, clinician, or engineer interested in contributing or collaborating, feel free to open an issue or reach out.

---

## ⚠️ Disclaimer

This model is intended for **research purposes only**. It is not FDA-approved and should not be used as a substitute for professional medical diagnosis.

---

## 📄 License

MIT License © 2026 Logits — see [LICENSE](LICENSE) for full details.
=======
# Lightweight-Pneumonia-Detection-AI-Model
Deep learning model developed in collaboration with Stanford faculty to enable multi-level pneumonia classification in under-resourced hospitals. Optimized for deployment constraints including limited computational resources, minimal infrastructure, and low-bandwidth environments. 
