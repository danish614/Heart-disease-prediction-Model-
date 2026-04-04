import sys
import os
import subprocess

# Agar python se directly run kiya toh automatically streamlit launch karo
if not os.environ.get("STREAMLIT_RUNNING"):
    env = os.environ.copy()
    env["STREAMLIT_RUNNING"] = "1"
    subprocess.run([sys.executable, "-m", "streamlit", "run", os.path.abspath(__file__)], env=env)
    sys.exit()

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

# ─── Page Config ───
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="&#10084;&#65039;",
    layout="wide"
)

# ─── Custom CSS ───
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
    }
    .result-box {
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 1rem 0;
    }
    .high-risk {
        background: linear-gradient(135deg, #ff4b4b22, #ff000011);
        border: 2px solid #ff4b4b;
    }
    .moderate-risk {
        background: linear-gradient(135deg, #ffa50022, #ffff0011);
        border: 2px solid #ffa500;
    }
    .low-risk {
        background: linear-gradient(135deg, #00cc0022, #00ff0011);
        border: 2px solid #00cc00;
    }
    .risk-factor-item {
        padding: 0.5rem 1rem;
        margin: 0.3rem 0;
        border-left: 4px solid #ff4b4b;
        background: #ff4b4b11;
        border-radius: 0 8px 8px 0;
    }
    .model-info {
        background: #f0f2f6;
        padding: 1rem 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #3498db;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ─── Resolve paths relative to this script ───
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ─── Load Model ───
@st.cache_resource
def load_model():
    model = joblib.load(os.path.join(SCRIPT_DIR, "best_model.pkl"))
    scaler = joblib.load(os.path.join(SCRIPT_DIR, "scaler.pkl"))
    feature_names = joblib.load(os.path.join(SCRIPT_DIR, "feature_names.pkl"))
    feature_importance = joblib.load(os.path.join(SCRIPT_DIR, "feature_importance.pkl"))
    return model, scaler, feature_names, feature_importance

@st.cache_data
def load_comparison():
    path = os.path.join(SCRIPT_DIR, "model_comparison.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


try:
    model, scaler, feature_names, feature_importance = load_model()
except FileNotFoundError:
    st.error("Model files nahi mil rahe! Pehle `heart_disease_training.py` run karein.")
    st.code("python heart_disease_training.py", language="bash")
    st.stop()

comparison_df = load_comparison()

# ─── Mappings ───
cp_map = {"Typical Angina": 0, "Atypical Angina": 1, "Non-Anginal Pain": 2, "Asymptomatic": 3}
restecg_map = {"Normal": 0, "ST-T Abnormality": 1, "LV Hypertrophy": 2}
slope_map = {"Upsloping": 0, "Flat": 1, "Downsloping": 2}
thal_map = {"Normal": 1, "Fixed Defect": 2, "Reversable Defect": 3}


# ─── Risk Calculation ───
def calculate_manual_risk_score(inputs):
    score = 0
    risk_factors = []

    if inputs["age"] > 60:
        score += 1
        risk_factors.append(f"Umar ({inputs['age']} saal) zyada hai")

    if inputs["trestbps"] > 140:
        score += 2
        risk_factors.append(f"Blood pressure high hai ({inputs['trestbps']} mm Hg)")

    if inputs["chol"] > 240:
        score += 1
        risk_factors.append(f"Cholesterol high hai ({inputs['chol']} mg/dL)")
    elif inputs["chol"] > 200:
        score += 0.5

    if inputs["cp"] in ["Atypical Angina", "Asymptomatic"]:
        score += 2
        risk_factors.append("Chest pain ka type risk indicate kar raha hai")

    if inputs["exang"] == 1:
        score += 3
        risk_factors.append("Exercise ke dauran chest pain hua")

    if inputs["oldpeak"] > 2:
        score += 2
        risk_factors.append(f"ST depression high hai ({inputs['oldpeak']})")
    elif inputs["oldpeak"] > 1:
        score += 1

    if inputs["ca"] >= 2:
        score += 3
        risk_factors.append(f"Multiple vessels affected hain ({inputs['ca']})")
    elif inputs["ca"] == 1:
        score += 1

    if inputs["restecg"] in ["ST-T Abnormality", "LV Hypertrophy"]:
        score += 2
        risk_factors.append("ECG abnormality detected")

    if inputs["fbs"] == 1:
        score += 1
        risk_factors.append("Fasting blood sugar high hai")

    if inputs["thal"] in ["Fixed Defect", "Reversable Defect"]:
        score += 2
        risk_factors.append("Blood flow defect detected")

    score = min(score, 15)
    return score, risk_factors


def predict(inputs):
    mapped = {
        "age": float(inputs["age"]),
        "sex": 1 if inputs["sex"] == "Male" else 0,
        "cp": cp_map[inputs["cp"]],
        "trestbps": float(inputs["trestbps"]),
        "chol": float(inputs["chol"]),
        "fbs": inputs["fbs"],
        "restecg": restecg_map[inputs["restecg"]],
        "thalach": float(inputs["thalach"]),
        "exang": inputs["exang"],
        "oldpeak": float(inputs["oldpeak"]),
        "slope": slope_map[inputs["slope"]],
        "ca": float(inputs["ca"]),
        "thal": thal_map[inputs["thal"]],
    }

    # Feature engineering
    mapped["age_chol"] = mapped["age"] * mapped["chol"]
    mapped["trestbps_thalach_ratio"] = mapped["trestbps"] / (mapped["thalach"] + 1)
    mapped["oldpeak_slope"] = mapped["oldpeak"] * mapped["slope"]
    mapped["age_trestbps"] = mapped["age"] * mapped["trestbps"]
    mapped["chol_thalach_ratio"] = mapped["chol"] / (mapped["thalach"] + 1)

    user_df = pd.DataFrame([mapped])
    for col in feature_names:
        if col not in user_df.columns:
            user_df[col] = 0
    user_df = user_df[feature_names]

    user_scaled = scaler.transform(user_df)
    pred = model.predict(user_scaled)[0]
    prob = model.predict_proba(user_scaled)[0][1]

    manual_score, risk_factors = calculate_manual_risk_score(inputs)

    if manual_score >= 7:
        final_risk = "HIGH"
    elif manual_score >= 4:
        final_risk = "HIGH" if prob >= 0.5 or manual_score >= 6 else "MODERATE"
    else:
        if prob >= 0.7:
            final_risk = "HIGH"
        elif prob >= 0.4:
            final_risk = "MODERATE"
        else:
            final_risk = "LOW"

    return prob, manual_score, risk_factors, final_risk


# ══════════════════════════════════════════════════════════
# UI - SIDEBAR (Model Info)
# ══════════════════════════════════════════════════════════
with st.sidebar:
    st.header("Model Information")

    if comparison_df is not None:
        st.subheader("Model Comparison")
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)

    # Feature Importance
    st.subheader("Feature Importance")
    if feature_importance:
        sorted_fi = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        top_features = sorted_fi[:10]
        fi_names = [f[0] for f in top_features]
        fi_values = [f[1] for f in top_features]

        fig_fi = go.Figure(go.Bar(
            x=fi_values,
            y=fi_names,
            orientation="h",
            marker_color="#3498db",
        ))
        fig_fi.update_layout(
            height=350,
            margin=dict(l=0, r=0, t=10, b=0),
            yaxis=dict(autorange="reversed"),
            xaxis_title="Importance",
        )
        st.plotly_chart(fig_fi, use_container_width=True)

    # EDA Images
    st.subheader("Training Visualizations")
    eda_images = [
        ("Target & Age Distribution", "eda_target_age.png"),
        ("Feature Boxplots", "eda_boxplots.png"),
        ("Correlation Heatmap", "eda_correlation.png"),
        ("Categorical Features", "eda_categorical.png"),
        ("Model Comparison", "model_comparison.png"),
        ("Confusion Matrices", "confusion_matrices.png"),
        ("ROC Curves", "roc_curves.png"),
        ("Feature Importance", "feature_importance.png"),
    ]
    for title, filename in eda_images:
        img_path = os.path.join(SCRIPT_DIR, filename)
        if os.path.exists(img_path):
            with st.expander(title):
                st.image(img_path, use_container_width=True)


# ══════════════════════════════════════════════════════════
# UI - MAIN (Input Form + Results)
# ══════════════════════════════════════════════════════════
st.markdown("<h1 class='main-header'>&#10084;&#65039; Heart Disease Prediction System</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:gray;'>Apni medical information darj karein aur heart disease ka risk janein</p>", unsafe_allow_html=True)

st.divider()

# ─── Input Form ───
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Basic Info")
    age = st.number_input("Age (Umar)", min_value=1, max_value=120, value=45, step=1)
    sex = st.selectbox("Gender (Jins)", ["Male", "Female"])
    cp = st.selectbox("Chest Pain Type", list(cp_map.keys()),
                      help="Typical Angina: classic dil ka dard | Asymptomatic: koi dard nahi")

with col2:
    st.subheader("Medical Measurements")
    trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=250, value=120, step=1,
                               help="Normal: <120 | High: >=140")
    chol = st.number_input("Cholesterol (mg/dL)", min_value=100, max_value=600, value=200, step=1,
                           help="Normal: <200 | High: >=240")
    thalach = st.number_input("Max Heart Rate (bpm)", min_value=60, max_value=220, value=150, step=1)
    oldpeak = st.number_input("ST Depression", min_value=0.0, max_value=10.0, value=0.0, step=0.1)

with col3:
    st.subheader("Clinical Tests")
    fbs_label = st.selectbox("Fasting Blood Sugar > 120 mg/dL?", ["No", "Yes"])
    fbs = 1 if fbs_label == "Yes" else 0
    restecg = st.selectbox("Resting ECG", list(restecg_map.keys()))
    exang_label = st.selectbox("Exercise Induced Angina?", ["No", "Yes"],
                               help="Kya exercise ke dauran chest pain hota hai?")
    exang = 1 if exang_label == "Yes" else 0
    slope = st.selectbox("ST Slope", list(slope_map.keys()))
    ca = st.selectbox("Major Vessels Colored (0-3)", [0, 1, 2, 3])
    thal = st.selectbox("Thalassemia", list(thal_map.keys()))

st.divider()

# ─── Predict Button ───
if st.button("Predict Karo", type="primary", use_container_width=True):

    inputs = {
        "age": age, "sex": sex, "cp": cp,
        "trestbps": trestbps, "chol": chol, "fbs": fbs,
        "restecg": restecg, "thalach": thalach, "exang": exang,
        "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal,
    }

    prob, manual_score, risk_factors, final_risk = predict(inputs)

    # ─── Results ───
    st.divider()

    if final_risk == "HIGH":
        risk_class = "high-risk"
        risk_emoji = "&#128308;"
        risk_color = "#ff4b4b"
    elif final_risk == "MODERATE":
        risk_class = "moderate-risk"
        risk_emoji = "&#128992;"
        risk_color = "#ffa500"
    else:
        risk_class = "low-risk"
        risk_emoji = "&#128994;"
        risk_color = "#00cc00"

    st.markdown(f"""
    <div class='result-box {risk_class}'>
        <h1>{risk_emoji} {final_risk} RISK</h1>
        <h3>Heart Disease Risk Assessment</h3>
    </div>
    """, unsafe_allow_html=True)

    # ─── Metrics Row ───
    m1, m2, m3 = st.columns(3)
    m1.metric("Model Probability", f"{prob * 100:.1f}%")
    m2.metric("Clinical Risk Score", f"{manual_score:.0f} / 15")
    m3.metric("Risk Level", final_risk)

    # ─── Charts ───
    chart1, chart2 = st.columns(2)

    with chart1:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            title={"text": "Model Probability (%)"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": risk_color},
                "steps": [
                    {"range": [0, 40], "color": "rgba(0,204,0,0.2)"},
                    {"range": [40, 70], "color": "rgba(255,165,0,0.2)"},
                    {"range": [70, 100], "color": "rgba(255,75,75,0.2)"},
                ],
                "threshold": {"line": {"color": "red", "width": 3}, "value": 50},
            },
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

    with chart2:
        fig2 = go.Figure(go.Indicator(
            mode="gauge+number",
            value=manual_score,
            title={"text": "Clinical Risk Score"},
            gauge={
                "axis": {"range": [0, 15]},
                "bar": {"color": risk_color},
                "steps": [
                    {"range": [0, 4], "color": "rgba(0,204,0,0.2)"},
                    {"range": [4, 7], "color": "rgba(255,165,0,0.2)"},
                    {"range": [7, 15], "color": "rgba(255,75,75,0.2)"},
                ],
                "threshold": {"line": {"color": "red", "width": 3}, "value": 7},
            },
        ))
        fig2.update_layout(height=300)
        st.plotly_chart(fig2, use_container_width=True)

    # ─── Risk Factors ───
    if risk_factors:
        st.subheader("Risk Factors Detected")
        for factor in risk_factors:
            st.markdown(f"<div class='risk-factor-item'>&#9888;&#65039; {factor}</div>", unsafe_allow_html=True)

    # ─── Recommendations ───
    st.divider()
    st.subheader("Recommendations")

    if final_risk == "HIGH":
        st.error("**URGENT: Jaldi se cardiologist se consult karein!**")
        recs = [
            "Complete cardiac evaluation karwayein",
            "Regular check-ups karein (monthly)",
            "Healthy lifestyle follow karein",
            "Medications doctor ke advice se lein",
            "Stress management karein",
        ]
    elif final_risk == "MODERATE":
        st.warning("**Doctor se consult karna recommended hai**")
        recs = [
            "Cardiologist se consult karein",
            "Regular monitoring zaroori hai",
            "Risk factors ko control karein",
            "Healthy lifestyle maintain karein",
            "3-6 months mein follow-up karein",
        ]
    else:
        st.success("**Risk kam hai, lekin regular check-ups zaroori hain**")
        recs = [
            "Healthy lifestyle maintain karein",
            "Balanced diet aur regular exercise",
            "Smoking avoid karein",
            "Yearly health check-up karein",
        ]

    for i, rec in enumerate(recs, 1):
        st.markdown(f"**{i}.** {rec}")

    st.divider()
    st.caption("**Disclaimer:** Ye system sirf prediction ke liye hai. Actual medical diagnosis ke liye hamesha qualified doctor se consult karein.")
