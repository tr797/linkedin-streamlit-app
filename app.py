import streamlit as st
import joblib
import pandas as pd

# Load model
logit_model = joblib.load("linkedin_model.pkl")

st.title("LinkedIn Usage Predictor")

st.write("Enter the person's characteristics:")

# User inputs
income = st.number_input("Income (1-9)", min_value=1, max_value=9)
education = st.number_input("Education (1-8)", min_value=1, max_value=8)
parent = st.selectbox("Parent?", [0, 1])
married = st.selectbox("Married?", [0, 1])
female = st.selectbox("Female?", [0, 1])
age = st.number_input("Age", min_value=0, max_value=98)

# Input dataframe
person = pd.DataFrame({
    'income': [income],
    'education': [education],
    'parent': [parent],
    'married': [married],
    'female': [female],
    'age': [age]
})

# Prediction
prediction = logit_model.predict(person)[0]
probability = logit_model.predict_proba(person)[0][1]

# Display results (NO Markdown)
if prediction == 1:
    st.write("Prediction: LinkedIn User")
else:
    st.write("Prediction: Not a LinkedIn User")

st.write("Probability of being a LinkedIn user:", round(probability, 2))

