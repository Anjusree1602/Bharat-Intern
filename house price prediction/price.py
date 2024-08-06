import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import tensorflow as tf
from keras import models

# Load the models
linear_regression_model = joblib.load('linear_regression_model.joblib')
neural_network_model = models.load_model('neural_network_model.h5')

# Create a StandardScaler instance and fit it to some sample data
scaler = StandardScaler()
sample_data = pd.DataFrame({
    'longitude': [-122.5],
    'latitude': [37.5],
    'housing_median_age': [20.0],
    'total_rooms': [5.0],
    'total_bedrooms': [2.0],
    'population': [1000.0],
    'households': [500.0],
    'median_income': [3.0]
})
scaler.fit(sample_data)

# Create a Streamlit app
st.title("House Price Prediction")

# Create a form for the user to input the features
st.header("Enter the features of the house:")
form = st.form("house_features")
longitude = form.number_input("Longitude:", value=-122.5)
latitude = form.number_input("Latitude:", value=37.5)
housing_median_age = form.number_input("Housing Median Age:", value=20.0)
total_rooms = form.number_input("Total Rooms:", value=5.0)
total_bedrooms = form.number_input("Total Bedrooms:", value=2.0)
population = form.number_input("Population:", value=1000.0)
households = form.number_input("Households:", value=500.0)
median_income = form.number_input("Median Income:", value=3.0)

# Create a button to submit the form
submit_button = form.form_submit_button("Predict")

# Create a section for the predictions
st.header("Predicted House Price:")

# Create a function to make the predictions
def make_predictions(longitude, latitude, housing_median_age, total_rooms, total_bedrooms, population, households, median_income):
    # Create a new house entry
    new_house = pd.DataFrame({
        'longitude': [longitude],
        'latitude': [latitude],
        'housing_median_age': [housing_median_age],
        'total_rooms': [total_rooms],
        'total_bedrooms': [total_bedrooms],
        'population': [population],
        'households': [households],
        'median_income': [median_income]
    })

    # Scale the new house entry
    new_house = scaler.transform(new_house)

    # Make predictions using the Linear Regression model
    predicted_price_lr = linear_regression_model.predict(new_house)
    predicted_price_lr = predicted_price_lr[0]

    # Make predictions using the Neural Network model
    predicted_price_nn = neural_network_model.predict(new_house)
    predicted_price_nn = predicted_price_nn[0][0]

    return predicted_price_lr, predicted_price_nn

# Make the predictions when the form is submitted
if submit_button:
    predicted_price_lr, predicted_price_nn = make_predictions(longitude, latitude, housing_median_age, total_rooms, total_bedrooms, population, households, median_income)
    st.write(f"Predicted Price using Linear Regression: ${predicted_price_lr:.2f}")
    st.write(f"Predicted Price using Neural Network: ${predicted_price_nn:.2f}")
