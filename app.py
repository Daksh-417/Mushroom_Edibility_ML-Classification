import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------
st.set_page_config(
    page_title="Mushroom Edibility Predictor",
    page_icon="🍄",
    layout="wide"
)

# ------------------------------------------------------------
# LOAD DATA + TRAIN MODEL (cached so it only runs once)
# ------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("Mushroom_Edibility.csv")
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

@st.cache_resource
def train_model(df):
    X = df.drop("class", axis=1)
    y = df["class"]

    le_target = LabelEncoder()
    y_encoded = le_target.fit_transform(y)  # EDIBLE=0, POISONOUS=1 (alphabetical)

    encoders = {}
    X_encoded = X.copy()
    for col in X_encoded.columns:
        le = LabelEncoder()
        X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
        encoders[col] = le

    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))

    importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)

    return model, encoders, le_target, acc, importances

df = load_data()
model, encoders, le_target, accuracy, importances = train_model(df)
feature_cols = [c for c in df.columns if c != "class"]

# ------------------------------------------------------------
# HEADER
# ------------------------------------------------------------
st.title("🍄 Mushroom Edibility Predictor")
st.caption("Pick a mushroom's features on the left, and instantly find out if it's likely EDIBLE or POISONOUS.")

top_c1, top_c2, top_c3 = st.columns(3)
top_c1.metric("Model", "Random Forest")
top_c2.metric("Test Accuracy", f"{accuracy*100:.2f}%")
top_c3.metric("Mushrooms in Dataset", f"{len(df):,}")

st.divider()

# ------------------------------------------------------------
# MAIN LAYOUT: LEFT = INPUT FORM | RIGHT = PREDICTION + CHARTS
# ------------------------------------------------------------
left, right = st.columns([1, 1.3], gap="large")

with left:
    st.subheader("🔧 Describe the Mushroom")
    st.write("Not sure what a term means? Just pick the option that looks/sounds closest — every field has friendly labels.")

    user_input = {}
    with st.form("prediction_form"):
        f1, f2 = st.columns(2)
        for i, col in enumerate(feature_cols):
            options = sorted(df[col].unique().tolist())
            label = col.replace("-", " ").title()
            target_col = f1 if i % 2 == 0 else f2
            user_input[col] = target_col.selectbox(label, options, key=col)

        submitted = st.form_submit_button("🔮 Predict Edibility", use_container_width=True, type="primary")

with right:
    st.subheader("🎯 Prediction")

    if submitted:
        input_df = pd.DataFrame([user_input])[feature_cols]
        input_encoded = input_df.copy()
        for col in feature_cols:
            le = encoders[col]
            val = input_df[col].iloc[0]
            if val in le.classes_:
                input_encoded[col] = le.transform([val])
            else:
                input_encoded[col] = 0  # fallback for unseen category

        pred = model.predict(input_encoded)[0]
        proba = model.predict_proba(input_encoded)[0]
        pred_label = le_target.inverse_transform([pred])[0]
        classes = le_target.classes_  # ['EDIBLE', 'POISONOUS']
        prob_dict = dict(zip(classes, proba))

        if pred_label == "EDIBLE":
            st.success(f"✅ This mushroom is likely **EDIBLE**  ({prob_dict['EDIBLE']*100:.1f}% confidence)")
        else:
            st.error(f"☠️ This mushroom is likely **POISONOUS**  ({prob_dict['POISONOUS']*100:.1f}% confidence)")

        prob_df = pd.DataFrame({
            "Class": classes,
            "Probability": [prob_dict[c] for c in classes]
        })
        fig_prob = px.bar(
            prob_df, x="Class", y="Probability", color="Class",
            color_discrete_map={"EDIBLE": "#2ecc71", "POISONOUS": "#e74c3c"},
            text_auto=".1%", range_y=[0, 1]
        )
        fig_prob.update_layout(showlegend=False, height=300, margin=dict(t=20, b=20, l=20, r=20))
        st.plotly_chart(fig_prob, use_container_width=True)

        st.warning("⚠️ For fun/learning only — never eat a wild mushroom based on this app!")
    else:
        st.info("👈 Fill in the mushroom's features and click **Predict Edibility** to see the result here.")

st.divider()

# ------------------------------------------------------------
# VISUALS SECTION (always visible, compact)
# ------------------------------------------------------------
st.subheader("📊 Dataset Insights")

v1, v2, v3 = st.columns(3)

with v1:
    class_counts = df["class"].value_counts().reset_index()
    class_counts.columns = ["Class", "Count"]
    fig1 = px.pie(
        class_counts, names="Class", values="Count", hole=0.5,
        color="Class", color_discrete_map={"EDIBLE": "#2ecc71", "POISONOUS": "#e74c3c"},
        title="Edible vs Poisonous"
    )
    fig1.update_layout(height=300, margin=dict(t=40, b=0, l=0, r=0))
    st.plotly_chart(fig1, use_container_width=True)

with v2:
    top_feat = importances.head(8).reset_index()
    top_feat.columns = ["Feature", "Importance"]
    fig2 = px.bar(
        top_feat.sort_values("Importance"), x="Importance", y="Feature", orientation="h",
        title="Top Predictive Features", color="Importance", color_continuous_scale="Viridis"
    )
    fig2.update_layout(height=300, margin=dict(t=40, b=0, l=0, r=0), coloraxis_showscale=False)
    st.plotly_chart(fig2, use_container_width=True)

with v3:
    odor_ct = pd.crosstab(df["odor"], df["class"]).reset_index()
    odor_melt = odor_ct.melt(id_vars="odor", var_name="Class", value_name="Count")
    fig3 = px.bar(
        odor_melt, x="odor", y="Count", color="Class", barmode="stack",
        color_discrete_map={"EDIBLE": "#2ecc71", "POISONOUS": "#e74c3c"},
        title="Odor vs Class"
    )
    fig3.update_layout(height=300, margin=dict(t=40, b=0, l=0, r=0), xaxis_title=None)
    st.plotly_chart(fig3, use_container_width=True)

st.caption("Built with Streamlit • Model: Random Forest Classifier • Dataset: Mushroom Edibility")
