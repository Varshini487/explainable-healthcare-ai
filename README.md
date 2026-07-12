# 🏥 Explainable Healthcare AI

A **disease risk predictor** that not only gives a diagnosis probability but explains *why* in plain language using SHAP (SHapley Additive exPlanations) and LIME (Local Interpretable Model-agnostic Explanations).

## 🧠 Why Explainability Matters in Healthcare
A model that says "80% risk of heart disease" is useless to a doctor unless it explains *why*. Is it the patient's high cholesterol? Age? Family history? With SHAP, a doctor can see which factors drove the prediction and discuss them with the patient confidently.

## How It Works (3-step pipeline)

**Step 1: Risk Prediction**
- Takes patient features: age, BMI, blood pressure, cholesterol, smoking status, family history, exercise frequency
- Trained XGBoost classifier predicts disease risk (0-100%)

**Step 2: SHAP Explanation**
- Computes Shapley values for each feature
- Shows which features *increased* risk (positive SHAP) vs *decreased* it (negative SHAP)
- Global explanations: what matters across all patients
- Local explanations: what drove *this* prediction specifically

**Step 3: Human-Friendly Output**
- Translates SHAP values into doctor-speak: "This patient's high cholesterol (+18%) and age 65 (+12%) are the main risk drivers. Positive: no smoking history (-8%) reduces risk."
- Suggests interventions: "Lowering cholesterol would reduce risk by ~15%"

## 🛠️ Tech Stack
- Python, Scikit-learn, XGBoost
- SHAP (for additive feature importance)
- LIME (for local explanations)
- Matplotlib / Seaborn (visualizations)
- Streamlit (interactive dashboard)

## 🚀 Getting Started
```bash
git clone https://github.com/Varshini487/explainable-healthcare-ai
cd explainable-healthcare-ai
pip install -r requirements.txt
streamlit run app.py
```

## 🎤 3 Interview Points

**1️⃣ Regulatory compliance & trust.**
"Regulators (FDA, GDPR Article 22) require explanations for any AI diagnosis. With SHAP, you're not just predicting; you're proving *why*. Doctors trust models they can audit. Black-box models, even if more accurate, get rejected in clinics because no one can explain them to patients."

**2️⃣ SHAP vs feature importance — they're not the same.**
"Simple feature importance (gain/split count) can be misleading in correlated data. SHAP uses game theory to compute each feature's true contribution to this specific prediction. Example: if Age and Cholesterol are correlated, feature importance might credit both equally, but SHAP shows Age actually drove it. SHAP = more accurate explanations."

**3️⃣ Better outcomes through actionability.**
"A patient hears 'you have 75% disease risk' → panic, no action. With explanations: 'Your high cholesterol and sedentary lifestyle are the main drivers. If you walk 30 min/day and switch statins, risk drops to 45%' → patient has a plan. Explainability = better patient compliance and outcomes."

## 📊 Example Output
```
Patient: 62-year-old male
Predicted Risk: 78%

🔴 MAJOR RISK FACTORS:
  - Age (62 years): +28% risk
  - Cholesterol (280): +18% risk
  - No Exercise: +15% risk
  
🟢 PROTECTIVE FACTORS:
  - Non-smoker: -12% risk
  - Normal BP: -8% risk

💡 RECOMMENDATIONS:
  - Add aerobic exercise 3x/week (expected -10-15% risk)
  - Statin therapy (expected -12-18% risk)
  - Weight loss 5-10% (expected -8% risk)
```
