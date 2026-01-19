import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# ===================== PAGE CONFIG =====================
st.set_page_config(
    page_title="AI COVID-19 Health Assistant",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ===================== CUSTOM CSS =====================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.main {
    background-color: #f4f6f8;
}

.stButton>button {
    width: 100%;
    height: 3em;
    border-radius: 8px;
    background-color: #2563eb;
    color: white;
    font-weight: 600;
    border: none;
}

.stButton>button:hover {
    background-color: #1e40af;
}

.metric-card {
    background-color: white;
    padding: 18px;
    border-radius: 12px;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.05);
}
</style>
""", unsafe_allow_html=True)

# ===================== SIDEBAR =====================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2966/2966327.png", width=120)
    st.title("AI Health Project")
    st.markdown("### COVID-19 Risk Prediction")
    st.info("""
    **Objective:**  
    Predict COVID-19 risk using Machine Learning (KNN).

    **Workflow:**
    - Data Cleaning  
    - Data Visualization  
    - Model Training  
    - Web Deployment
    """)
    st.divider()
    st.caption("👨‍💻 Developed by **Faizan Ali Siddiqui**")

# ===================== DATA LOADING =====================
@st.cache_data
def load_and_clean():
    df = pd.read_csv("patient.csv")
    cols = ['sex', 'pneumonia', 'age', 'diabetes', 'asthma', 'outcome']
    df = df[cols]

    for col in ['sex', 'pneumonia', 'diabetes', 'asthma', 'outcome']:
        df[col] = df[col].replace({2: 0})
        df = df[df[col].isin([0, 1])]

    return df

@st.cache_resource
def train_model(df):
    X = df.drop('outcome', axis=1)
    y = df['outcome']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train, y_train)

    acc = accuracy_score(y_test, model.predict(X_test))
    return model, acc

try:
    data = load_and_clean()
    model, accuracy = train_model(data)
except FileNotFoundError:
    st.error("❌ patient.csv file not found. Please upload it.")
    st.stop()

# ===================== MAIN DASHBOARD =====================
st.title("🩺 AI-Powered COVID-19 Health Assessment")
st.markdown("Predict COVID-19 risk using machine learning insights.")
st.divider()

c1, c2, c3 = st.columns(3)
c1.metric("Model Used", "K-Nearest Neighbors")
c2.metric("Train/Test Split", "80% / 20%")
c3.metric("Accuracy", f"{accuracy:.1%}")

# ===================== VISUALIZATION =====================
st.subheader("📊 Health Data Insights")
v1, v2 = st.columns(2)

with v1:
    fig1, ax1 = plt.subplots()
    sns.countplot(x=data['outcome'], palette="Blues", ax=ax1)
    ax1.set_xticklabels(["Negative", "Positive"])
    ax1.set_ylabel("Patients Count")
    st.pyplot(fig1)

with v2:
    fig2, ax2 = plt.subplots()
    sns.histplot(data[data['outcome'] == 1]['age'], bins=15, kde=True, ax=ax2)
    ax2.set_xlabel("Age")
    st.pyplot(fig2)

st.divider()

# ===================== USER PREDICTION =====================
st.subheader("🔍 Personal Risk Evaluation")
st.write("Provide patient details below:")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 0, 110, 25)
    gender = st.radio("Gender", ["Male", "Female"], horizontal=True)
    pneumonia = st.selectbox("Pneumonia Diagnosis", ["No", "Yes"])

with col2:
    diabetes = st.selectbox("Diabetes History", ["No", "Yes"])
    asthma = st.selectbox("Asthma History", ["No", "Yes"])

if st.button("🔎 Predict Risk"):
    input_data = [[
        1 if gender == "Female" else 0,
        1 if pneumonia == "Yes" else 0,
        age,
        1 if diabetes == "Yes" else 0,
        1 if asthma == "Yes" else 0
    ]]

    result = model.predict(input_data)

    st.divider()

    if result[0] == 1:
        st.error("🚩 High Risk Detected — Please consult a healthcare professional.")
    else:
        st.success("✅ Low Risk Detected — No immediate risk identified.")



# ===================== FOOTER =====================
st.markdown("---")
st.caption(
    "⚠️ **Disclaimer:** This educational project is developed by **Faizan Ali Siddiqui**. "
    "Predictions are based on historical data and should not replace medical advice."
)
