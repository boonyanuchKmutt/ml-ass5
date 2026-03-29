import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model = joblib.load('best_model.pkl')
scaler = joblib.load('scaler.pkl')

st.set_page_config(page_title="US Crime Predictor", page_icon="🔍", layout="centered")

st.title("🔍 US Crime Rate Predictor")
st.markdown("""This app uses a **Logistic Regression model trained with SMOTE oversampling** 
to predict whether a community has a **high violent crime rate**.""")

st.header("Enter Community Feature Values")
st.markdown("Input the 100 socio-economic feature values for the community:")

n_features = model.n_features_in_

# Layout: 4 columns for 100 features
cols = st.columns(4)
input_values = []

for i in range(n_features):
    col = cols[i % 4]
    with col:
        val = st.number_input(f"F{i+1}", value=0.0, format="%.4f", key=f"f{i}")
        input_values.append(val)

if st.button("🔍 Predict", use_container_width=True):
    input_array = np.array(input_values).reshape(1, -1)
    input_scaled = scaler.transform(input_array)

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0]
    crime_prob = probability[list(model.classes_).index(1)] * 100

    st.divider()
    if prediction == 1:
        st.error(f"⚠️ **HIGH CRIME RATE** — Probability: {crime_prob:.2f}%")
    else:
        st.success(f"✅ **LOW CRIME RATE** — Probability: {crime_prob:.2f}%")

    st.progress(int(crime_prob))
    st.caption(f"Model: Logistic Regression + SMOTE | Confidence: {max(probability)*100:.2f}%")

st.divider()
st.markdown("**Assignment 5 | Imbalanced Data Classification | KMU
