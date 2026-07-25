# ============================================================
# 🍄 MUSHROOM EDIBILITY - STREAMLIT APP
# Run in terminal:  streamlit run app.py
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# ============================================================
# PAGE SETUP + THEME (just paste, no need to fully understand)
# ============================================================
st.set_page_config(page_title="Mushroom Edibility", page_icon="🍄", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:wght@600;900&family=Nunito+Sans:wght@400;700&display=swap');

.stApp { background: linear-gradient(160deg, #0e2a1d 0%, #16392a 55%, #0b2418 100%); }
h1, h2, h3 { font-family: 'Fraunces', serif; color: #f5e9d0; }
p, label, span { font-family: 'Nunito Sans', sans-serif; }

.safe   { background:#1d5c38; border:2px solid #35c06f; border-radius:14px;
          padding:22px; text-align:center; font-size:26px; font-weight:bold; color:#eafff2; }
.danger { background:#6e1f1f; border:2px solid #ff6b57; border-radius:14px;
          padding:22px; text-align:center; font-size:26px; font-weight:bold; color:#ffecec; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD DATA (cached = runs once, then instant)
# ============================================================
@st.cache_data
def load_data():
    df = pd.read_csv('Mushroom_Edibility.csv')
    df.replace('?', np.nan, inplace=True)
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

# ============================================================
# TRAIN ALL MODELS (cached)
# ============================================================
@st.cache_resource
def train_models():
    df = load_data()

    X = df.drop('class', axis=1)
    y = df['class']

    # Encode target
    le_target = LabelEncoder()
    y_encoded = le_target.fit_transform(y)

    # Encode features (keep encoders for prediction later)
    encoders = {}
    X_encoded = X.copy()
    for col in X_encoded.columns:
        encoders[col] = LabelEncoder()
        X_encoded[col] = encoders[col].fit_transform(X_encoded[col].astype(str))

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_encoded)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

    # 7 models
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVC': SVC(probability=True, random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'Gaussian Naive Bayes': GaussianNB()
    }

    # Train + accuracy
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        results[name] = accuracy_score(y_test, model.predict(X_test))

    best = max(results, key=results.get)

    return {'models': models, 'results': results, 'best': best,
            'encoders': encoders, 'scaler': scaler,
            'le_target': le_target, 'df': df, 'X_encoded': X_encoded}

# ============================================================
# START APP
# ============================================================
df = load_data()
data = train_models()

st.sidebar.title("🍄 Mushroom Lab")
page = st.sidebar.radio("Go to", ["🔮 Prediction", "📊 Visualisation", "🏆 Model Comparison"])

# ============================================================
# PAGE 1: PREDICTION
# ============================================================
if page == "🔮 Prediction":
    st.title("🔮 Is this mushroom edible?")
    st.write("Pick the features you observe, then press **Predict**.")

    feature_cols = [c for c in df.columns if c != 'class']

    # 3 dropdowns per row
    user_inputs = {}
    cols = st.columns(3)
    for i, col_name in enumerate(feature_cols):
        with cols[i % 3]:
            user_inputs[col_name] = st.selectbox(
                col_name, sorted(df[col_name].unique()), key=col_name)

    if st.button("🍄 Predict!", type="primary"):
        with st.spinner("Checking the mushroom..."):
            # Encode user input using saved encoders
            input_row = [data['encoders'][c].transform([user_inputs[c]])[0]
                         for c in feature_cols]
            input_scaled = data['scaler'].transform([input_row])

            model = data['models'][data['best']]
            prediction = model.predict(input_scaled)[0]
            proba = model.predict_proba(input_scaled)[0]
            label = data['le_target'].inverse_transform([prediction])[0]
            confidence = max(proba)

            # Big result box
            if label == 'EDIBLE':
                st.markdown(f'<div class="safe">✅ EDIBLE — safe to eat! '
                            f'({confidence*100:.1f}% confident)</div>',
                            unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="danger">☠️ POISONOUS — do NOT eat! '
                            f'({confidence*100:.1f}% confident)</div>',
                            unsafe_allow_html=True)

            st.write("**Model confidence**")
            st.progress(confidence)

        # --- Visuals after prediction ---
        st.subheader("Why? Top 10 most important features")
        rf = data['models']['Random Forest']
        imp = pd.DataFrame({'Feature': feature_cols,
                            'Importance': rf.feature_importances_})
        imp = imp.sort_values('Importance').tail(10)

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.barh(imp['Feature'], imp['Importance'], color='#e8a33d', edgecolor='black')
        ax.set_xlabel('Importance')
        ax.set_title('Feature Importance (Random Forest)')
        st.pyplot(fig)

        st.info(f"🤖 Prediction made by **{data['best']}** "
                f"(accuracy {data['results'][data['best']]*100:.1f}%)")

# ============================================================
# PAGE 2: VISUALISATION (same 4 plots as notebook)
# ============================================================
elif page == "📊 Visualisation":
    st.title("📊 Explore the dataset")

    # PLOT 1: Target Balance
    st.subheader("1. Class Distribution")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    df['class'].value_counts().plot(kind='bar', ax=axes[0],
                                    color=['green', 'red'], edgecolor='black')
    axes[0].set_title('Class Distribution')
    axes[0].set_xticklabels(['EDIBLE', 'POISONOUS'], rotation=0)
    df['class'].value_counts().plot(kind='pie', ax=axes[1],
                                    color=['green', 'red'], autopct='%1.1f%%')
    axes[1].set_title('Class Proportion')
    axes[1].set_ylabel('')
    plt.tight_layout()
    st.pyplot(fig)

    # PLOT 2: Feature Distributions
    st.subheader("2. Feature Distributions")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    features_to_plot = ['cap-shape', 'cap-color', 'odor',
                        'gill-color', 'stalk-shape', 'habitat']
    for idx, feature in enumerate(features_to_plot):
        ax = axes[idx // 3, idx % 3]
        df[feature].value_counts().head(10).plot(kind='bar', ax=ax,
                                                 color='steelblue', edgecolor='black')
        ax.set_title(feature)
        ax.tick_params(axis='x', rotation=45)
    plt.suptitle('Feature Distributions')
    plt.tight_layout()
    st.pyplot(fig)

    # PLOT 3: Odor vs Class
    st.subheader("3. Odor vs Class")
    fig, ax = plt.subplots(figsize=(10, 6))
    pd.crosstab(df['odor'], df['class']).plot(kind='bar', ax=ax,
                                              color=['green', 'red'], edgecolor='black')
    ax.set_title('Odor vs Class')
    ax.set_xlabel('Odor')
    ax.set_ylabel('Count')
    ax.tick_params(axis='x', rotation=45)
    ax.legend(title='Class')
    plt.tight_layout()
    st.pyplot(fig)

    # PLOT 4: Correlation Heatmap (needs encoded data)
    st.subheader("4. Feature Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(14, 10))
    sns.heatmap(data['X_encoded'].corr(), cmap='coolwarm', ax=ax)
    ax.set_title('Feature Correlation Heatmap')
    plt.tight_layout()
    st.pyplot(fig)

# ============================================================
# PAGE 3: MODEL COMPARISON
# ============================================================
elif page == "🏆 Model Comparison":
    st.title("🏆 Model Comparison")

    results = data['results']
    best = data['best']

    # Bar chart (best model highlighted)
    names = list(results.keys())
    accs = list(results.values())
    colors = ['gold' if n == best else 'steelblue' for n in names]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(names, accs, color=colors, edgecolor='black')
    ax.set_xlabel('Accuracy')
    ax.set_title('Model Accuracy Comparison')
    ax.set_xlim(0.8, 1.02)
    for i, v in enumerate(accs):
        ax.text(v + 0.003, i, f'{v:.4f}', va='center')
    plt.tight_layout()
    st.pyplot(fig)

    # Table
    st.subheader("Results Table")
    results_df = pd.DataFrame(list(results.items()), columns=['Model', 'Accuracy'])
    results_df = results_df.sort_values('Accuracy', ascending=False).reset_index(drop=True)
    results_df.index += 1
    results_df['Best'] = results_df['Model'].apply(lambda x: '⭐' if x == best else '')
    st.dataframe(results_df, use_container_width=True)

    st.success(f"🏆 Best model: **{best}** with accuracy **{results[best]*100:.2f}%**")
