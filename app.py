import streamlit as st
import joblib
import pandas as pd

# Load sklearn model
model = joblib.load("logreg_sklearn_model.pkl")

# Define feature order
feature_names = ['GRE Score', 'University Rating', 'CGPA']

# Streamlit UI
st.title("📘 Admission Predictor (scikit-learn)")
st.markdown("This app predicts whether a student will be **Admitted** or **Rejected** based on GRE, Rating, and CGPA.")

# Inputs
gre = st.number_input("GRE Score", min_value=200, max_value=340, value=320)
rating = st.selectbox("University Rating", [1, 2, 3, 4, 5], index=3)
cgpa = st.slider("CGPA", min_value=0.0, max_value=10.0, step=0.1, value=8.5)

if st.button("Predict Admission"):
    input_df = pd.DataFrame([[gre, rating, cgpa]], columns=feature_names)
    prob = model.predict_proba(input_df)[0][1]  # Probability of class 1
    label = "Admit" if prob >= 0.6 else "Reject"
    
    st.subheader(f"🎯 Result: **{label}**")
    st.write(f"📊 Probability of Admission: **{prob:.4f}**")
