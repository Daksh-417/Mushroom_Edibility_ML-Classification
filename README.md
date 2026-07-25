# 🍄 Mushroom Edibility Classification — ML Pipeline

> **Binary classification pipeline that predicts whether a mushroom is `EDIBLE` or `POISONOUS` from 22 morphological features.**

A complete, end-to-end machine learning pipeline built as a single Jupyter notebook — from raw CSV to a fully evaluated, multi-model comparison.

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Pipeline Walkthrough](#pipeline-walkthrough)
- [Models Evaluated](#models-evaluated)
- [Results](#results)
- [Requirements](#requirements)
- [How to Run](#how-to-run)

---

## 🎯 Project Overview

Mushroom foraging is high-stakes — misclassifying a poisonous mushroom as edible can be fatal. This project builds a supervised binary classifier to automatically classify mushrooms as **EDIBLE** or **POISONOUS** using 22 physical characteristics (cap shape, odor, gill features, stalk features, etc.).

The pipeline covers the **full ML workflow**:

```
Load → Inspect → Clean → EDA → Encode → Scale → Split → Train → Evaluate → Compare
```

---

## 📊 Dataset

**File:** `Mushroom_Edibility.csv`

| Property | Value |
|---|---|
| Rows | **8,416** (8,124 after deduplication) |
| Features | **22 features + 1 target** |
| Target | `class` → `EDIBLE` / `POISONOUS` |
| Feature type | All categorical (string) |
| Missing values | None |

### Feature Dictionary

| # | Feature | Description |
|---|---|---|
| 1 | `class` | **Target** — `EDIBLE` or `POISONOUS` |
| 2 | `cap-shape` | Cap shape (convex, flat, …) |
| 3 | `cap-surface` | Cap surface texture |
| 4 | `cap-color` | Cap color |
| 5 | `bruises` | Bruising presence |
| 6 | `odor` | Odor (almond, foul, spicy, …) |
| 7 | `gill-attachment` | Gill attachment (free, attached) |
| 8 | `gill-spacing` | Gill spacing (close, crowded) |
| 9 | `gill-size` | Gill size (broad, narrow) |
| 10 | `gill-color` | Gill color |
| 11 | `stalk-shape` | Stalk shape (enlarging, tapering) |
| 12 | `stalk-root` | Stalk root (bulbous, …) |
| 13 | `stalk-surface-above-ring` | Stalk surface above the ring |
| 14 | `stalk-surface-below-ring` | Stalk surface below the ring |
| 15 | `stalk-color-above-ring` | Stalk color above the ring |
| 16 | `stalk-color-below-ring` | Stalk color below the ring |
| 17 | `veil-type` | Veil type (partial) |
| 18 | `veil-color` | Veil color |
| 19 | `ring-number` | Number of rings (one, …) |
| 20 | `ring-type` | Ring type (pendant, evanescent, large) |
| 21 | `spore-print-color` | Spore print color |
| 22 | `population` | Population density (several, clustered, …) |
| 23 | `habitat` | Habitat (woods, leaves, grasses, paths) |

---

## 🗂️ Project Structure

```
Mushroom_Edibility/
│
├── Mushroom_Edibility.csv        # Raw dataset (8,416 rows × 23 cols)
├── Mushroom_Edibility.ipynb      # Complete ML pipeline (single notebook)
├── model_comparison_results.csv  # Output: model accuracy comparison table
└── README.md                     # This file
```

**Generated plots** (saved by the notebook):

| File | Description |
|---|---|
| `01_target_distribution.png` | Class balance — EDIBLE vs POISONOUS |
| `02_feature_distributions.png` | Distribution of key features |
| `03_odor_vs_class.png` | Odor vs class — a strong predictor |
| `04_correlation_heatmap.png` | Feature correlation heatmap (post-encoding) |
| `06_confusion_matrix_best.png` | Confusion matrix of the best model |

---

## 🔄 Pipeline Walkthrough

The notebook is organized into clearly-labeled steps:

| Step | Step | What happens |
|---|---|---|
| **1** | Import Libraries | `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn` |
| **2** | Load Dataset | Load `Mushroom_Edibility.csv` → `(8416, 23)` |
| **3** | Data Inspection | `.head()`, `.dtypes`, `.info()`, `.describe()` |
| **4** | Data Cleaning | Check missing values (none) + drop duplicates → `(8124, 23)` |
| **5** | EDA | Class balance, feature distributions, odor-vs-class, correlation heatmap |
| **6** | Encoding + Scaling | `LabelEncoder` for all categorical columns + `StandardScaler` |
| **7** | Train/Test Split | `80/20` split, `stratified`, `random_state=42` |
| **8** | Train Models | Train 7 classifiers |
| **9** | Predict | Predict on the held-out test set |
| **10** | Evaluate | Accuracy + confusion matrix (best model plotted) |
| **11** | Compare | Results table → saved to `model_comparison_results.csv` |

**Split details:** `8,124` rows → **6,499 train / 1,625 test** (stratified, `random_state=42`)

---

## 🤖 Models Evaluated

Seven classifiers are trained and compared head-to-head:

| # | Model | Class |
|---|---|---|
| 1 | Logistic Regression | `LogisticRegression` |
| 2 | Decision Tree | `DecisionTreeClassifier` |
| 3 | Random Forest | `RandomForestClassifier` |
| 4 | Gradient Boosting | `GradientBoostingClassifier` |
| 5 | Support Vector Classifier | `SVC` |
| 6 | K-Nearest Neighbors | `KNeighborsClassifier` |
| 7 | Gaussian Naive Bayes | `GaussianNB` |

---

## 📈 Results

All models are evaluated on the held-out test set using **accuracy**, with a confusion matrix plotted for the best performer. The full comparison table is exported to **`model_comparison_results.csv`**, with the top model flagged ⭐.

> 📌 Open `Mushroom_Edibility.ipynb` and run all cells to regenerate the full comparison table and the best-model confusion matrix.

---

## ⚙️ Requirements

```
python >= 3.9
numpy
pandas
matplotlib
seaborn
scikit-learn
```

Install everything at once:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

---

## 🚀 How to Run

```bash
# 1. Clone / download the project
# 2. Install dependencies
pip install numpy pandas matplotlib seaborn scikit-learn

# 3. Launch the notebook
jupyter notebook Mushroom_Edibility.ipynb
```

Then **Run All** cells — the notebook runs the full pipeline, displays all plots, and writes `model_comparison_results.csv` to disk.

---

## 🔑 Key Takeaways

- **Odor is a powerful predictor** — the odor-vs-class plot shows strong class separation.
- **All 22 features are categorical** and are label-encoded before scaling.
- **Tree-based ensembles (Random Forest / Gradient Boosting) typically top the comparison**, as expected for tabular categorical data.
- The dataset is **clean (no missing values)** and only requires deduplication.

---

*Built as a single-notebook, end-to-end ML pipeline. 🍄*
