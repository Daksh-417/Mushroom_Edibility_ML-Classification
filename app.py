# ============================================================
# 🍄 SPORE LAB — MUSHROOM EDIBILITY CLASSIFIER
# ============================================================
# Run with:  streamlit run app.py
# Model:     Random Forest (best-in-class from pipeline)
# Target:    class — EDIBLE vs POISONOUS
# ============================================================
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ============================================================
# PAGE SETUP
# ============================================================
st.set_page_config(page_title="Spore Lab — Mushroom Edibility",
                   page_icon="🍄", layout="wide")

# ============================================================
# THEME — mycology field-station styling
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600;9..144,700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@500;600&display=swap');

/* ---- layered ambient background: spore-dot print + forest grid + glows ---- */
.stApp {
  background:
    radial-gradient(1000px 520px at 85% -10%, rgba(116,227,154,.08), transparent 60%),
    radial-gradient(900px 500px at 5% 110%, rgba(232,176,75,.06), transparent 60%),
    radial-gradient(rgba(116,227,154,.05) 1.2px, transparent 1.2px),
    linear-gradient(rgba(148,200,170,.03) 1px, transparent 1px),
    linear-gradient(90deg, rgba(148,200,170,.03) 1px, transparent 1px),
    #0b120d;
  background-size: auto, auto, 26px 26px, 44px 44px, 44px 44px, auto;
}
html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; color: #cfe3d4; }

/* ---- typography ---- */
.lab-title {
  font-family: 'Fraunces', serif; font-size: 2.7rem; font-weight: 700;
  letter-spacing: .5px; color: #eef7f0; margin-bottom: 2px; line-height: 1.1;
}
.lab-title .amber { color: #e8b04b; text-shadow: 0 0 24px rgba(232,176,75,.45); }
.sub-title { color: #7d9a86; margin-top: -6px; margin-bottom: 18px; font-size: 1.02rem; }
h2, h3 { font-family: 'Fraunces', serif; color: #e8f4ec !important; letter-spacing: .3px; }
.subhead {
  font-family: 'Fraunces', serif; font-size: 1.15rem; font-weight: 600;
  color: #e8f4ec; border-left: 4px solid #e8b04b; padding-left: 12px; margin: 26px 0 10px 0;
}

/* ---- status strip chips ---- */
.chip-row { display: flex; flex-wrap: wrap; gap: 12px; margin: 14px 0 24px 0; }
.chip {
  font-family: 'IBM Plex Mono', monospace; font-size: .82rem;
  background: #101a12; border: 1px solid #24382a; border-radius: 8px;
  padding: 10px 16px; color: #7d9a86; transition: all .22s ease;
}
.chip b { color: #e8b04b; font-size: 1rem; display: block; margin-top: 2px; }
.chip .green { color: #5fd98a; }
.chip:hover { transform: translateY(-3px); border-color: #e8b04b; box-shadow: 0 8px 20px rgba(0,0,0,.35); }
.live-dot {
  display: inline-block; width: 9px; height: 9px; border-radius: 50%;
  background: #5fd98a; margin-right: 7px;
  box-shadow: 0 0 0 0 rgba(95,217,138,.6); animation: pulse 1.8s infinite;
}
@keyframes pulse { to { box-shadow: 0 0 0 11px rgba(95,217,138,0); } }

/* ---- panels ---- */
.panel {
  background: #101a12; border: 1px solid #24382a; border-radius: 10px;
  padding: 22px; transition: border-color .25s ease, box-shadow .25s ease;
  margin-bottom: 16px;
}
.panel:hover { border-color: #3a5a44; box-shadow: 0 10px 30px rgba(0,0,0,.35); }
.panel-label {
  font-family: 'IBM Plex Mono', monospace; font-size: .8rem; font-weight: 600;
  text-transform: uppercase; letter-spacing: 2px; color: #5f7a68; margin-bottom: 14px;
}

/* ---- verdict gauge ---- */
.verdict-panel { text-align: center; }
.gauge-wrap { display: flex; flex-direction: column; align-items: center; padding: 10px 0 4px 0; }
.gauge {
  width: 270px; height: 135px; border-radius: 270px 270px 0 0; position: relative; overflow: hidden;
  box-shadow: 0 0 34px rgba(95,217,138,.12);
}
.gauge::after {
  content: ''; position: absolute; left: 30px; right: 30px; top: 30px; bottom: -32px;
  background: #101a12; border-radius: 999px 999px 0 0; border-top: 1px solid #24382a;
}
.verdict-word {
  font-family: 'Fraunces', serif; font-size: 3rem; font-weight: 700;
  letter-spacing: 3px; margin-top: -46px; z-index: 2;
}
.verdict-edible { color: #5fd98a; text-shadow: 0 0 30px rgba(95,217,138,.55); }
.verdict-toxic  { color: #ff6f61; text-shadow: 0 0 30px rgba(255,111,97,.55); }
.verdict-conf { font-family: 'IBM Plex Mono', monospace; color: #7d9a86; font-size: .9rem; z-index: 2; margin-top: 4px; }
.gauge-range { display: flex; justify-content: space-between; width: 270px;
  font-family: 'IBM Plex Mono', monospace; font-size: .72rem; color: #5f7a68; margin-top: 6px; }

/* ---- diagnostic marker bars ---- */
.fi-row { display: flex; align-items: center; gap: 12px; margin: 11px 0; }
.fi-label { font-family: 'IBM Plex Mono', monospace; width: 150px; color: #7d9a86; font-size: .8rem; }
.fi-track { flex: 1; height: 14px; background: #16241a; border-radius: 7px; overflow: hidden; }
.fi-fill { height: 100%; border-radius: 7px; background: linear-gradient(90deg, #2f9e5f, #5fd98a);
  transition: width .6s ease, filter .2s ease; }
.fi-row:hover .fi-fill { filter: brightness(1.3); }
.fi-pct { font-family: 'IBM Plex Mono', monospace; width: 56px; text-align: right;
  color: #e8b04b; font-size: .82rem; }

/* ---- widgets ---- */
.stSelectbox > label, .stMultiSelect > label {
  font-family: 'IBM Plex Mono', monospace; font-size: .75rem; text-transform: uppercase;
  letter-spacing: 1px; color: #7d9a86 !important;
}
[data-baseweb="select"] > div { background-color: #131f16 !important; border-color: #24382a !important; }
.stDataFrame, .stTable { border-radius: 10px; overflow: hidden; }

/* ---- hide streamlit chrome ---- */
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ---- dark matplotlib theme (matches app) ----
plt.rcParams.update({
    'figure.facecolor': '#0b120d', 'axes.facecolor': '#0b120d',
    'axes.edgecolor': '#2a4232', 'axes.labelcolor': '#cfe3d4',
    'text.color': '#cfe3d4', 'xtick.color': '#7d9a86', 'ytick.color': '#7d9a86',
    'grid.color': '#1c2c20', 'axes.grid': True, 'grid.alpha': .35,
    'axes.spines.top': False, 'axes.spines.right': False,
    'font.family': 'sans-serif',
})
GREEN, AMBER, RED = '#5fd98a', '#e8b04b', '#ff6f61'

# ============================================================
# LOAD DATA + TRAIN MODEL (cached — runs once)
# ============================================================
FEATURE_GROUPS = {
    'Cap Morphology':   ['cap-shape', 'cap-surface', 'cap-color'],
    'Gill Structure':   ['gill-attachment', 'gill-spacing', 'gill-size', 'gill-color'],
    'Stalk / Stipe':    ['stalk-shape', 'stalk-root', 'stalk-surface-above-ring',
                         'stalk-surface-below-ring', 'stalk-color-above-ring',
                         'stalk-color-below-ring'],
    'Veil & Ring':      ['veil-type', 'veil-color', 'ring-number', 'ring-type'],
    'Spore & Field':    ['spore-print-color', 'odor', 'bruises', 'population', 'habitat'],
}
GROUP_COLS = {
    'Cap Morphology': 3, 'Gill Structure': 2, 'Stalk / Stipe': 3,
    'Veil & Ring': 2, 'Spore & Field': 3,
}

@st.cache_resource
def load_data_and_model():
    df = pd.read_csv('Mushroom_Edibility.csv')
    df = df.drop_duplicates().reset_index(drop=True)

    X = df.drop('class', axis=1)
    y = df['class']

    encoders = {}
    X_enc = X.copy()
    for col in X_enc.columns:
        le = LabelEncoder()
        X_enc[col] = le.fit_transform(X_enc[col].astype(str))
        encoders[col] = le

    y_enc = LabelEncoder().fit(y)          # EDIBLE=0, POISONOUS=1
    y_t = y_enc.transform(y)

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X_enc)

    # holdout accuracy for the status readout
    X_tr, X_te, y_tr, y_te = train_test_split(X_s, y_t, test_size=0.2,
                                              random_state=42, stratify=y_t)
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_tr, y_tr)
    acc = accuracy_score(y_te, model.predict(X_te))

    # refit on full data for production predictions
    model.fit(X_s, y_t)
    return df, X_enc, model, scaler, encoders, y_enc, acc

df, X_enc, model, scaler, encoders, y_enc, ACC = load_data_and_model()
FEATURES = list(X_enc.columns)
N_EDIBLE = int((df['class'] == 'EDIBLE').sum())
N_POISON = int((df['class'] == 'POISONOUS').sum())

# ============================================================
# SIDEBAR — NAVIGATION + MODEL CARD
# ============================================================
with st.sidebar:
    st.markdown("## 🍄 SPORE·LAB")
    page = st.radio("Navigate", ["🧪 Classify", "📊 Field Charts", "🗂️ Specimens"])
    st.divider()
    st.info(f"**Model:** Random Forest\n\n"
            f"**Accuracy:** {ACC:.4f}\n\n"
            f"**Specimens:** {len(df):,}\n\n"
            f"**Markers:** {len(FEATURES)}")
    st.caption(f"{N_EDIBLE} edible · {N_POISON} poisonous · 22 markers → class")

# ============================================================
# SHARED HEADER — lab status strip
# ============================================================
def status_strip():
    st.markdown(
        '<p class="lab-title">🍄 SPORE <span class="amber">LAB</span></p>'
        '<p class="sub-title">Mushroom edibility classifier — 22 morphological field markers</p>',
        unsafe_allow_html=True)
    st.markdown(f"""
    <div class="chip-row">
      <div class="chip"><span class="live-dot"></span>MYCOLOGY LAB <b>ACTIVE</b></div>
      <div class="chip">SPECIMENS <b>{len(df):,}</b></div>
      <div class="chip">MARKERS <b>{len(FEATURES)}</b></div>
      <div class="chip">MODEL ACC <b class="green">{ACC:.4f}</b></div>
      <div class="chip">CLASS SPLIT <b>{N_EDIBLE}E / {N_POISON}P</b></div>
    </div>""", unsafe_allow_html=True)

# ============================================================
# PAGE 1: CLASSIFY
# ============================================================
if page == "🧪 Classify":
    status_strip()
    col_in, col_out = st.columns([1.15, 1])

    # ---- LEFT: grouped marker panels ----
    with col_in:
        vals = {}
        for group, feats in FEATURE_GROUPS.items():
            st.markdown(f'<div class="panel"><div class="panel-label">▸ {group}</div>',
                        unsafe_allow_html=True)
            cols = st.columns(GROUP_COLS[group])
            for i, f in enumerate(feats):
                with cols[i % GROUP_COLS[group]]:
                    options = list(encoders[f].classes_)
                    default = df[f].mode()[0]
                    vals[f] = st.selectbox(f.replace('-', ' ').title(),
                                           options, options.index(default), key=f)
            st.markdown('</div>', unsafe_allow_html=True)

    # ---- RIGHT: live verdict ----
    with col_out:
        # assemble + encode input in training column order
        input_df = pd.DataFrame([[vals[f] for f in FEATURES]], columns=FEATURES)
        for col in FEATURES:
            input_df[col] = encoders[col].transform([vals[col]])[0]
        proba = model.predict_proba(scaler.transform(input_df))[0]

        is_edible = proba[0] >= proba[1]
        conf = float(proba.max())
        deg = conf * 180
        color = GREEN if is_edible else RED
        gauge_bg = (f"conic-gradient(from 270deg at 50% 100%, {color} 0deg, {color} {deg:.1f}deg, "
                    f"rgba(255,255,255,.05) {deg:.1f}deg 180deg, transparent 180deg)")
        word = 'EDIBLE' if is_edible else 'POISONOUS'
        wclass = 'verdict-edible' if is_edible else 'verdict-toxic'

        st.markdown(f"""
        <div class="panel verdict-panel">
          <div class="panel-label">▸ Field Verdict</div>
          <div class="gauge-wrap">
            <div class="gauge" style="background: {gauge_bg};"></div>
            <div class="verdict-word {wclass}">{word}</div>
            <div class="verdict-conf">{conf*100:.1f}% model confidence</div>
            <div class="gauge-range"><span>EDIBLE</span><span>POISONOUS</span></div>
          </div>
        </div>""", unsafe_allow_html=True)

        if is_edible:
            st.success("✅ Cleared as edible — always confirm with an expert mycologist before consumption.")
        else:
            st.error("☠️ Flagged as POISONOUS — do not consume. Seek expert verification immediately.")

        # ---- diagnostic marker importance ----
        imps = model.feature_importances_
        order = np.argsort(imps)[::-1][:10]
        max_imp = imps.max()
        rows = "".join(
            f'<div class="fi-row"><span class="fi-label">{FEATURES[i]}</span>'
            f'<div class="fi-track"><div class="fi-fill" style="width:{imps[i]/max_imp*100:.1f}%"></div></div>'
            f'<span class="fi-pct">{imps[i]*100:.1f}%</span></div>'
            for i in order)
        st.markdown(f'<div class="panel"><div class="panel-label">▸ Diagnostic Markers</div>{rows}</div>',
                    unsafe_allow_html=True)

# ============================================================
# PAGE 2: FIELD CHARTS
# ============================================================
elif page == "📊 Field Charts":
    status_strip()

    st.markdown('<div class="subhead">1 · Class Balance</div>', unsafe_allow_html=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    df['class'].value_counts().plot(kind='bar', ax=axes[0], color=[GREEN, RED], edgecolor='#0b120d')
    axes[0].set_title('Specimen Class Distribution'); axes[0].set_xticklabels(['EDIBLE', 'POISONOUS'], rotation=0)
    df['class'].value_counts().plot(kind='pie', ax=axes[1], color=[GREEN, RED], autopct='%1.1f%%')
    axes[1].set_title('Class Proportion'); axes[1].set_ylabel('')
    plt.tight_layout(); st.pyplot(fig, use_container_width=True)

    st.markdown('<div class="subhead">2 · Marker Distributions</div>', unsafe_allow_html=True)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for idx, f in enumerate(['cap-shape', 'cap-color', 'odor', 'gill-color', 'stalk-shape', 'habitat']):
        ax = axes[idx // 3, idx % 3]
        df[f].value_counts().head(10).plot(kind='bar', ax=ax, color=AMBER, edgecolor='#0b120d')
        ax.set_title(f); ax.tick_params(axis='x', rotation=45)
    plt.suptitle('Key Marker Distributions', color='#e8f4ec', fontsize=14)
    plt.tight_layout(); st.pyplot(fig, use_container_width=True)

    st.markdown('<div class="subhead">3 · Odor vs. Edibility — the strongest tell</div>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    pd.crosstab(df['odor'], df['class']).plot(kind='bar', color=[GREEN, RED], edgecolor='#0b120d', ax=ax)
    ax.set_title('Odor Profile vs. Class'); ax.set_xlabel('Odor'); ax.set_ylabel('Count')
    plt.xticks(rotation=45); ax.legend(title='Class')
    plt.tight_layout(); st.pyplot(fig, use_container_width=True)

    st.markdown('<div class="subhead">4 · Marker Correlation Map</div>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(X_enc.corr(), cmap='viridis', square=True, linewidths=.5,
                linecolor='#0b120d', cbar_kws={'shrink': .8}, ax=ax)
    ax.set_title('Encoded Marker Correlation')
    plt.tight_layout(); st.pyplot(fig, use_container_width=True)

# ============================================================
# PAGE 3: SPECIMENS
# ============================================================
else:
    status_strip()

    st.markdown('<div class="subhead">Specimen Ledger</div>', unsafe_allow_html=True)
    st.write(f"Shape: **{df.shape[0]:,} rows × {df.shape[1]} columns**")
    st.dataframe(df.head(20), use_container_width=True)

    st.markdown('<div class="subhead">Marker Glossary</div>', unsafe_allow_html=True)
    GLOSSARY = {
        'cap-shape': 'Cap silhouette (bell / convex / flat / knobbed / sunken / conical)',
        'cap-surface': 'Cap texture (smooth / fibrous / scaly / grooved)',
        'cap-color': 'Cap pigmentation',
        'bruises': 'Whether flesh bruises when cut',
        'odor': 'Scent profile — the single strongest edibility tell',
        'gill-attachment': 'How gills attach to the stalk (free / attached)',
        'gill-spacing': 'Gill density (crowded / close)',
        'gill-size': 'Gill breadth (broad / narrow)',
        'gill-color': 'Gill pigmentation',
        'stalk-shape': 'Stalk form (tapering / enlarging)',
        'stalk-root': 'Stalk base (bulbous / club / rooted / equal)',
        'stalk-surface-above-ring': 'Stalk texture above the ring',
        'stalk-surface-below-ring': 'Stalk texture below the ring',
        'stalk-color-above-ring': 'Stalk color above the ring',
        'stalk-color-below-ring': 'Stalk color below the ring',
        'veil-type': 'Veil type (partial / universal)',
        'veil-color': 'Veil color',
        'ring-number': 'Number of rings',
        'ring-type': 'Ring form (pendant / evanescent / large)',
        'spore-print-color': 'Spore print color — key taxonomic marker',
        'population': 'Growth density pattern',
        'habitat': 'Ecological setting',
    }
    glossary = pd.DataFrame([
        {'Marker': f, 'Description': GLOSSARY[f],
         'Unique': df[f].nunique(), 'Top Value': df[f].mode()[0]}
        for f in FEATURES])
    st.table(glossary)

    st.markdown('<div class="subhead">Statistical Summary (encoded)</div>', unsafe_allow_html=True)
    st.dataframe(X_enc.describe().round(2), use_container_width=True)
