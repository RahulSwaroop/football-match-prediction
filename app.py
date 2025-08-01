
import streamlit as st
import pandas as pd
import joblib

# Load model and encoder
model = joblib.load("football_model.joblib")
label_encoder = joblib.load("label_encoder.joblib")

# UI
st.set_page_config(page_title="Football Match Predictor", layout="centered")
st.title("⚽ Football Match Outcome Predictor")
st.markdown("Enter the Elo ratings to predict the outcome.")

# Input sliders
home_elo = st.slider("Home Team Elo Rating", min_value=1000, max_value=2500, value=1500)
away_elo = st.slider("Away Team Elo Rating", min_value=1000, max_value=2500, value=1500)

# Predict
if st.button("Predict Outcome"):
    input_df = pd.DataFrame([[home_elo, away_elo]], columns=['home_elo', 'away_elo'])
    prediction = model.predict(input_df)
    result = label_encoder.inverse_transform(prediction)[0]
    
    st.success(f"🏆 Predicted Match Outcome: **{result}**")
