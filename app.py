import streamlit as st
import numpy as np
from sklearn.linear_model import LinearRegression

st.title("AI Exam Score Predictor")

st.write("Predict your exam score based on study hours and sleep hours.")

# dataset
X = np.array([
    [1,7],
    [2,6],
    [3,6],
    [4,5],
    [5,5],
    [6,4],
    [7,4],
    [8,3]
])

y = np.array([40,45,50,55,60,65,70,75])

# train model
model = LinearRegression()
model.fit(X, y)

# user input
study = st.slider("Study Hours", 0, 12)
sleep = st.slider("Sleep Hours", 0, 12)

prediction = model.predict([[study, sleep]])

st.subheader("Predicted Exam Score")
st.write(round(prediction[0],2))