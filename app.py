# Machine Learning
# Aim - To classify Mushrooms as Edible (e) or Poisonous (p) based on their
# physical characteristics such as Cap Shape, Cap Surface, Cap Color, Odor,
# Gill Attachment, Stalk Shape, Habitat, etc.

# importing libraries
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score


# Streamlit page
st.set_page_config(
    page_title='Mushroom Edibility Prediction',
    layout='wide'
)


# loading data set and training models
@st.cache_resource
def train_models():

    # loading data set
    df = pd.read_csv('mushroom_classification.csv')

    # Data Cleaning
    df.drop_duplicates(inplace=True)
    df = df.reset_index(drop=True)

    # Encoding Categorical Data
    le = {}
    df_encoded = df.copy()

    for col in df.columns:
        le[col] = LabelEncoder()
        df_encoded[col] = le[col].fit_transform(df[col])

    # Split X and Y
    X = df_encoded.drop('class', axis=1)
    Y = df_encoded['class']

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

    # Building, Training and Testing Model
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'SVC': SVC(random_state=42),
        'KNN': KNeighborsClassifier(),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'GaussianNB': GaussianNB()
    }

    scores = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        scores[name] = accuracy_score(y_test, model.predict(X_test))

    feature_cols = list(X.columns)

    return df, models, scores, scaler, le, feature_cols


# train models when app starts
df, models, scores, scaler, le, feature_cols = train_models()


# session state for past predictions
if 'history' not in st.session_state:
    st.session_state.history = []


# prediction function
def predict_all(inputs):

    encoded = {}

    for col in feature_cols:
        encoded[col] = le[col].transform([inputs[col]])[0]

    input_df = pd.DataFrame([encoded], columns=feature_cols)
    input_scaled = scaler.transform(input_df)

    preds = {}

    for name, model in models.items():
        pred_encoded = int(model.predict(input_scaled)[0])
        preds[name] = le['class'].inverse_transform([pred_encoded])[0]

    return preds


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

    with st.form('mushroom_form'):

        cols = st.columns(2)
        inputs = {}

        for i, col in enumerate(feature_cols):
            with cols[i % 2]:
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

    st.dataframe(score_df)