# ============================================================
# 🍄 MUSHROOM EDIBILITY PREDICTOR - STREAMLIT APP
# ============================================================
# Run with:  streamlit run app.py
# ============================================================

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# ============================================================
# PAGE SETUP
# ============================================================
st.set_page_config(page_title="Mushroom Edibility Predictor",
                   page_icon="🍄", layout="wide")

# Simple custom style
st.markdown("""
<style>
    .main-title { font-size: 2.5rem; font-weight: 800; }
    .sub-title  { color: gray; margin-top: -10px; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD DATA + TRAIN MODEL (runs only once, then cached)
# ============================================================
@st.cache_resource
def load_data_and_model():
    df = pd.read_csv('Mushroom_Edibility.csv')
    df = df.drop_duplicates().reset_index(drop=True)

    X = df.drop('class', axis=1)
    y = df['class']

    # Encode all features (save encoders for prediction)
    encoders = {}
    X_encoded = X.copy()
    for col in X_encoded.columns:
        le = LabelEncoder()
        X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
        encoders[col] = le

    # Train best model (Random Forest from your notebook)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_encoded, y)

    return df, model, encoders

df, model, encoders = load_data_and_model()
feature_columns = df.drop('class', axis=1).columns.tolist()

# ============================================================
# SIDEBAR - NAVIGATION
# ============================================================
with st.sidebar:
    st.title("🍄 Menu")
    page = st.radio("Go to:", ["🔮 Prediction", "📊 Visualizations", "📋 Dataset"])
    st.divider()
    st.info(f"**Model:** Random Forest\n\n**Features:** {len(feature_columns)}\n\n**Samples:** {len(df)}")

# ============================================================
# PAGE 1: PREDICTION
# ============================================================
if page == "🔮 Prediction":
    st.markdown('<p class="main-title">🔮 Is it Safe to Eat?</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Select mushroom features below to predict edibility</p>',
                unsafe_allow_html=True)

    # One selectbox per feature (3 per row)
    user_input = {}
    cols = st.columns(3)
    for i, feature in enumerate(feature_columns):
        with cols[i % 3]:
            user_input[feature] = st.selectbox(
                feature, sorted(df[feature].unique())
            )

    st.divider()

    # Predict button
    if st.button("🔍 Predict Edibility", type="primary", use_container_width=True):
        # Convert user input to encoded numbers
        input_df = pd.DataFrame([user_input])
        for col in feature_columns:
            input_df[col] = encoders[col].transform(input_df[col].astype(str))

        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0].max() * 100

        if prediction == 'EDIBLE':
            st.success(f"✅ EDIBLE — safe to eat! (confidence: {probability:.1f}%)")
            st.progress(probability / 100)
        else:
            st.error(f"☠️ POISONOUS — do NOT eat! (confidence: {probability:.1f}%)")
            st.progress(probability / 100)

# ============================================================
# PAGE 2: VISUALIZATIONS (same 4 plots as notebook)
# ============================================================
elif page == "📊 Visualizations":
    st.markdown('<p class="main-title">📊 Data Visualizations</p>', unsafe_allow_html=True)

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
                                              color=['green', 'red'],
                                              edgecolor='black')
    ax.set_title('Odor vs Class')
    ax.set_xlabel('Odor')
    ax.set_ylabel('Count')
    plt.xticks(rotation=45)
    ax.legend(title='Class')
    plt.tight_layout()
    st.pyplot(fig)

    # PLOT 4: Correlation Heatmap
    st.subheader("4. Feature Correlation Heatmap")
    X_encoded = df.drop('class', axis=1).apply(
        lambda col: LabelEncoder().fit_transform(col.astype(str))
    )
    fig, ax = plt.subplots(figsize=(14, 10))
    sns.heatmap(X_encoded.corr(), cmap='coolwarm', ax=ax)
    ax.set_title('Feature Correlation Heatmap')
    plt.tight_layout()
    st.pyplot(fig)

# ============================================================
# PAGE 3: DATASET
# ============================================================
else:
    st.markdown('<p class="main-title">📋 Dataset Preview</p>', unsafe_allow_html=True)
    st.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
    st.dataframe(df.head(20), use_container_width=True)