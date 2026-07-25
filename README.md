# 🍄 Mushroom Edibility Classification

> **Will it kill you or feed you?** A complete ML project that classifies mushrooms as `EDIBLE` or `POISONOUS` from 22 physical features — with a Jupyter training pipeline **and** a live Streamlit prediction app.

![Python](https://img.shields.io/badge/Python-3.13-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Project Structure](#-project-structure)
- [Dataset](#-dataset)
- [ML Pipeline (Notebook)](#-ml-pipeline-notebook)
- [Results](#-results)
- [Streamlit App](#-streamlit-app)
- [Requirements](#-requirements)
- [How to Run](#-how-to-run)

---

## 🎯 Project Overview

Two deliverables, one dataset:

| Part | File | Purpose |
|---|---|---|
| **1. Training Pipeline** | `Mushroom_Edibility.ipynb` | Full ML workflow — clean, explore, encode, train 7 models, compare |
| **2. Prediction App** | `app.py` | Interactive Streamlit UI for live predictions + visualizations |

The notebook finds the best model; the app deploys it behind a friendly interface.

---

## 🗂️ Project Structure

```
Mushroom_Edibility/
│
├── Mushroom_Edibility.csv         # Dataset (8,416 rows × 23 cols)
├── Mushroom_Edibility.ipynb       # Full ML training pipeline
├── app.py                         # Streamlit prediction app
├── model_comparison_results.csv   # Output: model accuracy table
├── 01_target_distribution.png     # EDA: class balance (bar + pie)
├── 02_feature_distributions.png   # EDA: 6 key features
├── 03_odor_vs_class.png           # EDA: odor — the strongest signal
├── 04_correlation_heatmap.png     # EDA: feature correlation heatmap
├── 06_confusion_matrix_best.png   # Best model confusion matrix
└── README.md                      # This file
```

---

## 📊 Dataset

**`Mushroom_Edibility.csv`** — 23 columns, all categorical (string).

| Property | Value |
|---|---|
| Raw rows | **8,416** |
| After cleaning (dedup) | **8,124** |
| Features | **22** + 1 target (`class`) |
| Target | `EDIBLE` (4,488) / `POISONOUS` (3,928) |
| Missing values | None |

### The 22 Features

| Group | Features |
|---|---|
| **Cap** | `cap-shape`, `cap-surface`, `cap-color` |
| **Other** | `bruises`, `odor` |
| **Gills** | `gill-attachment`, `gill-spacing`, `gill-size`, `gill-color` |
| **Stalk** | `stalk-shape`, `stalk-root`, `stalk-surface-above-ring`, `stalk-surface-below-ring`, `stalk-color-above-ring`, `stalk-color-below-ring` |
| **Veil** | `veil-type`, `veil-color` |
| **Ring** | `ring-number`, `ring-type` |
| **Ecology** | `spore-print-color`, `population`, `habitat` |

> 💡 **Key EDA finding:** `odor` is the single strongest predictor — certain odors (foul, pungent, spicy) map almost perfectly to poisonous classes.

---

## 🔄 ML Pipeline (Notebook)

`Mushroom_Edibility.ipynb` runs a clean, step-by-step workflow:

| Step | Stage | What Happens |
|---|---|---|
| 1 | **Import** | All libraries loaded at once |
| 2 | **Load** | Read CSV → `(8416, 23)` |
| 3 | **Inspect** | `head()`, `dtypes`, `info()`, `describe()`, `value_counts()` |
| 4 | **Clean** | Missing-value check + `drop_duplicates()` → `(8124, 23)` |
| 5 | **EDA** | 4 plots: class balance, feature distributions, odor vs class, correlation heatmap |
| 6 | **Preprocess** | `LabelEncoder` on all features + `StandardScaler` |
| 7 | **Split** | 80/20 stratified split, `random_state=42` → **6,499 train / 1,625 test** |
| 8–9 | **Train** | 7 classifiers fitted and predicted |
| 10 | **Evaluate** | Accuracy
