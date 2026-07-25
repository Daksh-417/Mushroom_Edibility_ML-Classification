@@ -1,13 +1,7 @@
# Machine Learning
# Aim - To classify Mushrooms as Edible (e) or Poisonous (p) based on their
# physical characteristics such as Cap Shape, Cap Surface, Cap Color, Odor,
# Gill Attachment, Stalk Shape, Habitat, etc.

# importing libraries
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

import matplotlib.patches as mpatches
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
@@ -18,257 +12,209 @@
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score


# Streamlit page
st.set_page_config(
    page_title='Mushroom Edibility Prediction',
    layout='wide'
)


# loading data set and training models
# ─── Page Config ───────────────────────────────────────────────
st.set_page_config(page_title='🍄 Mushroom Classifier', layout='wide', initial_sidebar_state='collapsed')

# ─── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    .block-container { padding-top: 1rem; padding-bottom: 0rem; max-width: 1400px; }
    .stForm { border: 1px solid #e0e0e0; border-radius: 12px; padding: 0.8rem; background: #fafbfc; }
    div[data-testid="stMetric"] { background: #f8f9fa; border-radius: 10px; padding: 0.6rem 1rem; border: 1px solid #eee; }
    div[data-testid="stMetricLabel"] { font-size: 0.75rem; }
    div[data-testid="stMetricValue"] { font-size: 1.1rem; }
    .pred-card { border-radius: 10px; padding: 0.7rem 1rem; text-align: center; font-weight: 600; font-size: 0.95rem; margin-bottom: 4px; }
    .pred-e { background: linear-gradient(135deg, #d4edda, #c3e6cb); color: #155724; border: 1px solid #b1dfbb; }
    .pred-p { background: linear-gradient(135deg, #f8d7da, #f5c6cb); color: #721c24; border: 1px solid #f1b0b7; }
    .header-bar { background: linear-gradient(90deg, #2d5016, #4a7c23); padding: 1rem 1.5rem; border-radius: 12px; margin-bottom: 1rem; }
    .header-bar h1 { color: white; margin: 0; font-size: 1.6rem; }
    .header-bar p { color: #d4edda; margin: 0.2rem 0 0 0; font-size: 0.85rem; }
    .section-title { font-size: 0.9rem; font-weight: 700; color: #333; margin: 0.8rem 0 0.4rem 0; border-bottom: 2px solid #4a7c23; padding-bottom: 0.2rem; }
    div[data-testid="stSelectbox"] label { font-size: 0.72rem !important; font-weight: 600; }
    div[data-testid="stSelectbox"] { margin-bottom: 0.3rem; }
    .stButton > button { border-radius: 8px; font-weight: 600; }
    .history-entry { border: 1px solid #e8e8e8; border-radius: 8px; padding: 0.5rem; margin-bottom: 0.4rem; background: #fdfdfd; }
</style>
""", unsafe_allow_html=True)

# ─── Train Models (cached) ────────────────────────────────────
@st.cache_resource
def train_models():

    # loading data set
    df = pd.read_csv('mushroom_classification.csv')

    # Data Cleaning
    df.drop_duplicates(inplace=True)
    df = df.reset_index(drop=True)
    df.reset_index(drop=True, inplace=True)

    # Encoding Categorical Data
    le = {}
    df_encoded = df.copy()

    for col in df.columns:
        le[col] = LabelEncoder()
        df_encoded[col] = le[col].fit_transform(df[col])

    # Split X and Y
    X = df_encoded.drop('class', axis=1)
    Y = df_encoded['class']
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=42)

    # Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        Y,
        test_size=0.20,
        random_state=42
    )

    # Scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Building, Training and Testing Model
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'LogReg': LogisticRegression(max_iter=1000),
        'DecTree': DecisionTreeClassifier(random_state=42),
        'RandForest': RandomForestClassifier(random_state=42),
        'SVC': SVC(random_state=42),
        'KNN': KNeighborsClassifier(),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'GaussianNB': GaussianNB()
        'GradBoost': GradientBoostingClassifier(random_state=42),
        'GaussNB': GaussianNB()
    }

    scores = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        scores[name] = accuracy_score(y_test, model.predict(X_test))

    feature_cols = list(X.columns)

    return df, models, scores, scaler, le, feature_cols
        model.fit(X_train_s, y_train)
        scores[name] = accuracy_score(y_test, model.predict(X_test_s))

    return df, models, scores, scaler, le, list(X.columns)

# train models when app starts
df, models, scores, scaler, le, feature_cols = train_models()


# session state for past predictions
# ─── Session State ─────────────────────────────────────────────
if 'history' not in st.session_state:
    st.session_state.history = []


# prediction function
# ─── Predict Function ─────────────────────────────────────────
def predict_all(inputs):

    encoded = {}

    for col in feature_cols:
        encoded[col] = le[col].transform([inputs[col]])[0]

    encoded = {col: le[col].transform([inputs[col]])[0] for col in feature_cols}
    input_df = pd.DataFrame([encoded], columns=feature_cols)
    input_scaled = scaler.transform(input_df)

    preds = {}

    for name, model in models.items():
        pred_encoded = int(model.predict(input_scaled)[0])
        preds[name] = le['class'].inverse_transform([pred_encoded])[0]

        pred_enc = int(model.predict(input_scaled)[0])
        preds[name] = le['class'].inverse_transform([pred_enc])[0]
    return preds

# ─── Header ────────────────────────────────────────────────────
st.markdown("""
<div class="header-bar">
    <h1>🍄 Mushroom Edibility Classifier</h1>
    <p>7 ML models trained on 8,124 samples · 22 physical features · Select features → Predict</p>
</div>
""", unsafe_allow_html=True)

# one visualization for each prediction
def make_chart(i, preds, scores, actual=None):

    algos = list(preds.keys())
    pred_vals = list(preds.values())
    accs = [scores[a] for a in algos]

    best_by_acc = max(scores, key=scores.get)

    if actual is not None:

        correct = [p == actual for p in pred_vals]

        if any(correct):
            correct_algos = [a for a, ok in zip(algos, correct) if ok]
            best_this = max(correct_algos, key=lambda a: scores[a])
            title = f'P{i} | Best correct: {best_this} | Best Acc: {best_by_acc}'
        else:
            title = f'P{i} | No correct prediction | Best Acc: {best_by_acc}'

        edgecolors = ['black' if ok else 'none' for ok in correct]
        linewidths = [1.3 if ok else 0 for ok in correct]

    else:

        title = f'P{i} | Best: {best_by_acc} (Acc={scores[best_by_acc]:.2f})'
        edgecolors = ['none'] * len(algos)
        linewidths = [0] * len(algos)

    colors = ['#2ca02c' if p == 'e' else '#d62728' for p in pred_vals]

    fig, ax = plt.subplots(figsize=(6, 2.8))

    bars = ax.barh(
        algos,
        accs,
        color=colors,
        edgecolor=edgecolors,
        linewidth=linewidths
    )

    ax.set_xlim(0, 1.25)
    ax.invert_yaxis()

    for y, (p, acc) in enumerate(zip(pred_vals, accs)):
        ax.text(
            acc,
            y,
            f' {p} ({acc:.2f})',
            va='center',
            fontsize=6
        )

    if actual is not None:
        title = title + f' | Actual: {actual}'
# ─── Main Layout: 3 columns ───────────────────────────────────
col_input, col_result, col_history = st.columns([2.2, 1.8, 1.5])

    ax.set_title(title, fontsize=7)
    ax.set_xlabel('Accuracy')

    fig.tight_layout()

    return fig


# title
st.title('Mushroom Edibility Prediction')
st.subheader('All models are trained at startup')


# inputs in sidebar to keep main area compact
with st.sidebar:

    st.header('User Inputs')
# ─── INPUT COLUMN ──────────────────────────────────────────────
with col_input:
    st.markdown('<div class="section-title">🔬 Feature Input</div>', unsafe_allow_html=True)

    with st.form('mushroom_form'):

        cols = st.columns(2)
        c1, c2, c3 = st.columns(3)
        inputs = {}

        for i, col in enumerate(feature_cols):
            with cols[i % 2]:
            with [c1, c2, c3][i % 3]:
                options = sorted(df[col].unique().tolist())
                default = df.iloc[0][col]

                inputs[col] = st.selectbox(
                    col,
                    options,
                    index=options.index(default)
                )

        actual = st.selectbox(
            'Actual Class (optional)',
            ['Unknown', 'e', 'p']
        )

        submitted = st.form_submit_button(
            'Predict',
            use_container_width=True
        )

    clear = st.button(
        'Clear History',
        use_container_width=True
    )


# clear history
if clear:
    st.session_state.history = []
    st.rerun()


# prediction workflow
if submitted:

    preds = predict_all(inputs)

    st.session_state.history.append({
        'inputs': inputs,
        'preds': preds,
        'actual': None if actual == 'Unknown' else actual
    })


# stacked prediction visualizations
if st.session_state.history:

    st.markdown(f'#### Prediction History ({len(st.session_state.history)})')

    with st.container(height=540):

        for idx in range(len(st.session_state.history) - 1, -1, -1):

            rec = st.session_state.history[idx]

            fig = make_chart(
                idx + 1,
                rec['preds'],
                scores,
                rec['actual']
            )

            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

else:

    st.info('Select mushroom features and click Predict.')


# model scores
with st.expander('Model Accuracy Scores'):

    score_df = pd.DataFrame({
        'Algorithm': list(scores.keys()),
        'Accuracy': list(scores.values())
    })
                inputs[col] = st.selectbox(col, options, index=0)

        c_a, c_b = st.columns([2, 1])
        with c_a:
            actual = st.selectbox('Actual Class (optional)', ['Unknown', 'e', 'p'])
        with c_b:
            st.markdown('<br>', unsafe_allow_html=True)
            submitted = st.form_submit_button('🔍 Predict', use_container_width=True, type='primary')

# ─── RESULT COLUMN ─────────────────────────────────────────────
with col_result:
    st.markdown('<div class="section-title">📊 Prediction Results</div>', unsafe_allow_html=True)

    if submitted:
        preds = predict_all(inputs)
        st.session_state.history.append({'inputs': inputs, 'preds': preds, 'actual': None if actual == 'Unknown' else actual})

    if st.session_state.history:
        latest = st.session_state.history[-1]
        preds = latest['preds']
        actual_val = latest['actual']

        # Consensus
        e_count = sum(1 for v in preds.values() if v == 'e')
        p_count = len(preds) - e_count
        consensus = 'e' if e_count >= p_count else 'p'

        # Big verdict
        if consensus == 'e':
            st.markdown('<div class="pred-card pred-e" style="font-size:1.2rem; padding:1rem;">✅ EDIBLE (Consensus)</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="pred-card pred-p" style="font-size:1.2rem; padding:1rem;">☠️ POISONOUS (Consensus)</div>', unsafe_allow_html=True)

        # Per-model cards in 2 columns
        mc1, mc2 = st.columns(2)
        for i, (name, pred) in enumerate(preds.items()):
            with mc1 if i % 2 == 0 else mc2:
                acc = scores[name]
                cls = 'pred-e' if pred == 'e' else 'pred-p'
                icon = '✅' if pred == 'e' else '☠️'
                correct_mark = ''
                if actual_val:
                    correct_mark = ' ✓' if pred == actual_val else ' ✗'
                st.markdown(f'<div class="pred-card {cls}">{icon} {name}: <b>{pred.upper()}</b> ({acc:.0%}){correct_mark}</div>', unsafe_allow_html=True)

        # Bar chart
        fig, ax = plt.subplots(figsize=(4.5, 2.6))
        algos = list(preds.keys())
        accs = [scores[a] for a in algos]
        colors = ['#2d8a4e' if preds[a] == 'e' else '#c0392b' for a in algos]

        bars = ax.barh(algos, accs, color=colors, height=0.6, zorder=3)
        ax.set_xlim(0, 1.15)
        ax.invert_yaxis()
        ax.set_xlabel('Accuracy', fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(axis='x', alpha=0.3, zorder=0)
        ax.spines[['top', 'right']].set_visible(False)

        for y, (a, acc) in enumerate(zip(algos, accs)):
            ax.text(acc + 0.01, y, f'{preds[a]} ({acc:.2f})', va='center', fontsize=7, fontweight='bold')

        legend_e = mpatches.Patch(color='#2d8a4e', label='Edible')
        legend_p = mpatches.Patch(color='#c0392b', label='Poisonous')
        ax.legend(handles=[legend_e, legend_p], fontsize=7, loc='lower right', framealpha=0.9)

        title = f'Latest Prediction (P{len(st.session_state.history)})'
        if actual_val:
            title += f' | Actual: {actual_val.upper()}'
        ax.set_title(title, fontsize=8, fontweight='bold')

        fig.tight_layout(pad=0.5)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    else:
        st.info('👈 Select features and click **Predict**')

# ─── HISTORY COLUMN ────────────────────────────────────────────
with col_history:
    st.markdown('<div class="section-title">📜 History</div>', unsafe_allow_html=True)

    if st.session_state.history:
        st.markdown(f'<small>{len(st.session_state.history)} prediction(s)</small>', unsafe_allow_html=True)

        with st.container(height=320):
            for idx in range(len(st.session_state.history) - 1, -1, -1):
                rec = st.session_state.history[idx]
                e_c = sum(1 for v in rec['preds'].values() if v == 'e')
                p_c = len(rec['preds']) - e_c
                verdict = '✅ e' if e_c >= p_c else '☠️ p'
                act_str = f" | Actual: {rec['actual']}" if rec['actual'] else ''
                st.markdown(f'<div class="history-entry"><b>P{idx+1}</b> → {verdict} (e:{e_c} p:{p_c}){act_str}</div>', unsafe_allow_html=True)

        if st.button('🗑️ Clear', use_container_width=True):
            st.session_state.history = []
            st.rerun()
    else:
        st.caption('No predictions yet.')

    st.dataframe(score_df)
# ─── Footer: Model Scores ─────────────────────────────────────
with st.expander('📈 Model Accuracy Comparison', expanded=False):
    sc1, sc2, sc3, sc4, sc5, sc6, sc7 = st.columns(7)
    for col_widget, (name, acc) in zip([sc1, sc2, sc3, sc4, sc5, sc6, sc7], scores.items()):
        col_widget.metric(name, f'{acc:.1%}')
