import sklearn
import xgboost
import streamlit as st
import numpy as np
import pickle

# Load the trained XGBoost model
with open('xgb_regressor_model.pkl', 'rb') as file:
    m = pickle.load(file)

# Set the title of the app
st.title("Price Prediction with XGBoost")

# Add a description of the app
st.write("🔍 This app uses a XGBRegressor to predict Car Price.")


# Collect input from the user for the features
# You can adjust the input fields as per the features used in your model
feature_1 = st.number_input("Feature 1", min_value=0.0, max_value=10000.0, step=0.1)
feature_2 = st.number_input("Feature 2", min_value=0, max_value=100, step=1)
feature_3 = st.number_input("Feature 3", min_value=0.0, max_value=100.0, step=0.1)
feature_4 = st.number_input("Feature 4", min_value=0, max_value=1000000, step=1000)
feature_5 = st.number_input("Feature 5", min_value=0, max_value=100, step=1)
feature_6 = st.number_input("Feature 1", min_value=0.0, max_value=10000.0, step=0.1)
feature_7 = st.number_input("Feature 2", min_value=0, max_value=100, step=1)
feature_8 = st.number_input("Feature 3", min_value=0.0, max_value=100.0, step=0.1)
feature_9 = st.number_input("Feature 4", min_value=0, max_value=1000000, step=1000)

# Collect the features into an array
input_features = np.array([[feature_1, feature_2, feature_3, feature_4, feature_5,feature_6, feature_7, feature_8, feature_9]])

# When the user clicks the button, predict the price
if st.button("Predict Price"):
    # Make prediction using the loaded model
    predicted_price = m.predict(input_features)[0]
    
    # Display the result
    st.write(f"Predicted Price: ${predicted_price:,.2f}")
