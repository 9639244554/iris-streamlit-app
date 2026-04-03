import streamlit as st
import numpy as np
import joblib

# Load model
model = joblib.load('iris_model.pkl')

st.set_page_config(page_title="Iris Predictor", page_icon="🌸")

st.title("🌸 Iris Flower Prediction App")
st.write("This app predicts Iris flower species using Machine Learning.")

# Image (safe image link)
st.image("https://upload.wikimedia.org/wikipedia/commons/5/56/Iris_setosa_2.jpg")

# Columns layout
col1, col2 = st.columns(2)

with col1:
    sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0)
    petal_length = st.number_input("Petal Length (cm)", min_value=0.0)

with col2:
    sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0)
    petal_width = st.number_input("Petal Width (cm)", min_value=0.0)

# Prediction
if st.button("🔍 Predict"):

    # Validation
    if sepal_length == 0 or sepal_width == 0 or petal_length == 0 or petal_width == 0:
        st.warning("⚠️ Please enter all values correctly!")
    else:
        data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        prediction = model.predict(data)

        # If model returns string
        name = prediction[0].replace('Iris-', '')

        st.success(f"🌼 Predicted Species: {name}")

# Sidebar
st.sidebar.title("About")
st.sidebar.write("Created by Mohd Rizwan 🚀")