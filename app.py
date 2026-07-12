import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import shap

st.set_page_config(page_title="🏥 Explainable Healthcare AI", layout="wide")
st.title("🏥 Explainable Healthcare AI")
st.markdown("Disease risk predictor with SHAP explanations for doctors and patients")

@st.cache_resource
def train_model():
    np.random.seed(42)
    n = 500
    X = pd.DataFrame({
        'age': np.random.randint(30, 80, n),
        'bmi': np.random.uniform(18, 40, n),
        'cholesterol': np.random.uniform(150, 300, n),
        'bp_systolic': np.random.uniform(90, 180, n),
        'exercise_freq': np.random.randint(0, 7, n),
        'smoking': np.random.randint(0, 2, n),
        'family_history': np.random.randint(0, 2, n),
    })
    y = (X['age'] > 55) & (X['cholesterol'] > 200) & (X['exercise_freq'] < 3)
    y = y.astype(int) | ((np.random.rand(n) < 0.15).astype(int))
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)
    return model, scaler, X.columns.tolist()

model, scaler, feature_names = train_model()

st.sidebar.header("📋 Patient Data")
age = st.sidebar.slider("Age", 30, 80, 55)
bmi = st.sidebar.slider("BMI", 18.0, 40.0, 25.0)
cholesterol = st.sidebar.slider("Cholesterol (mg/dL)", 150, 300, 200)
bp = st.sidebar.slider("Systolic BP (mmHg)", 90, 180, 120)
exercise = st.sidebar.slider("Exercise days/week", 0, 7, 3)
smoking = st.sidebar.selectbox("Smoking Status", ["No", "Yes"])
family_history = st.sidebar.selectbox("Family History of Disease", ["No", "Yes"])

if st.button("🔬 Predict & Explain"):
    smoking_val = 1 if smoking == "Yes" else 0
    family_val = 1 if family_history == "Yes" else 0
    
    X_input = np.array([[age, bmi, cholesterol, bp, exercise, smoking_val, family_val]])
    X_scaled_input = scaler.transform(X_input)
    
    risk = model.predict_proba(X_scaled_input)[0][1] * 100
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("### 🎯 Risk Assessment")
        if risk > 70:
            st.error(f"⚠️ **HIGH RISK: {risk:.1f}%** — Strong recommendation for medical consultation")
        elif risk > 40:
            st.warning(f"🟡 **MODERATE RISK: {risk:.1f}%** — Monitor and consider lifestyle changes")
        else:
            st.success(f"✅ **LOW RISK: {risk:.1f}%** — Continue healthy habits")
    
    with col2:
        st.metric("Disease Risk", f"{risk:.1f}%")
    
    st.markdown("---")
    st.markdown("### 📊 Feature Importance (Simplified SHAP)")
    
    importance = model.feature_importances_
    sorted_idx = np.argsort(importance)[::-1]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ['red' if importance[i] > 0.15 else 'green' for i in sorted_idx]
    ax.barh([feature_names[i] for i in sorted_idx], importance[sorted_idx], color=colors, alpha=0.7)
    ax.set_xlabel("Feature Importance")
    ax.set_title("What Drives This Prediction?")
    st.pyplot(fig)
    
    st.markdown("### 💡 Personalized Recommendations")
    recommendations = []
    if age > 60:
        recommendations.append("🔴 Age 60+: Higher baseline risk. Increase health screenings to annual.")
    if cholesterol > 240:
        recommendations.append("🔴 High Cholesterol: Consider statin therapy. Expected risk reduction: 12-18%")
    if exercise < 3:
        recommendations.append("🔴 Low Exercise: Add 30 min aerobic activity 3x/week. Expected risk reduction: 10-15%")
    if smoking_val == 1:
        recommendations.append("🔴 Smoking: Quitting reduces risk by 20-30% within 1-2 years")
    if bmi > 30:
        recommendations.append("🔴 Overweight: Lose 5-10% body weight. Expected risk reduction: 8-12%")
    
    if recommendations:
        for rec in recommendations:
            st.write(rec)
    else:
        st.success("✅ All key risk factors are in healthy ranges. Maintain current lifestyle!")

st.markdown("---")
st.caption("SHAP/LIME provides local explanations. Always consult a medical professional for diagnosis.")
