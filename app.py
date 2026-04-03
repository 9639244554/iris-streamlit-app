import streamlit as st
import numpy as np
import joblib

st.write("This app predict Iris flower species using Mechine Learning.")
st.image("https://upload.wikimedia.org/wikipedia/commons/4/41/Iris_setosa.jpg")
model=joblib.load('iris_model.pkl')
st.set_page_config(page_title="Iris Predictor", page_icon ="🌸")

st.sidebar.title("About")
st.sidebar.write("Created by Mohd Rizwan")
st.title("🌸 Iris Flower Prediction App")
st.write("Enter the flower measurement below:")

# Inputs with better labels
sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0)
sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0)
petal_length = st.number_input("Petal Length (cm)", min_value=0.0)
petal_width = st.number_input("Petal Width (cm)", min_value=0.0)

#Prediction
if st.button("🔍 Predict"):
    data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(data)

    st.success(f"🌼 Predicted Species: {prediction[0]}")

if petal_length == 0 or petal_width == 0:
    st.warning("Please enter valid values!")

col1, col2 = st.columns()
with col1:
    sepal_length = st.number_input('Sepal Length')
with col2:
    sepal_width = st.number_input('Sepal Width')