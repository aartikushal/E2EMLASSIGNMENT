{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "c33844d8-4395-4f05-979c-1a5f48ca2edf",
   "metadata": {},
   "outputs": [],
   "source": [
    "import streamlit as st\n",
    "import numpy as np\n",
    "import pickle\n",
    "\n",
    "# Load the model\n",
    "with open('xgb_regressor_model.pkl', 'rb') as f:\n",
    "    model = pickle.load(f)\n",
    "\n",
    "# Set the title of the app\n",
    "st.title('Price Prediction with XGBoost')\n",
    "\n",
    "# Add a description of the app\n",
    "st.write(\"\"\"\n",
    "This is a price prediction app built using **XGBoost**. \n",
    "Please input the features to predict the price.\n",
    "\"\"\")\n",
    "\n",
    "# Collect input from the user for the features\n",
    "# You can adjust the input fields as per the features used in your model\n",
    "feature_1 = st.number_input('Feature 1', min_value=0.0, max_value=10000.0, step=0.1)\n",
    "feature_2 = st.number_input('Feature 2', min_value=0, max_value=100, step=1)\n",
    "feature_3 = st.number_input('Feature 3', min_value=0.0, max_value=100.0, step=0.1)\n",
    "feature_4 = st.number_input('Feature 4', min_value=0, max_value=1000000, step=1000)\n",
    "feature_5 = st.number_input('Feature 5', min_value=0, max_value=100, step=1)\n",
    "\n",
    "# Collect the features into an array\n",
    "input_features = np.array([[feature_1, feature_2, feature_3, feature_4, feature_5]])\n",
    "\n",
    "# When the user clicks the button, predict the price\n",
    "if st.button('Predict Price'):\n",
    "    # Make prediction using the loaded model\n",
    "    predicted_price = model.predict(input_features)[0]\n",
    "    \n",
    "    # Display the result\n",
    "    st.write(f'Predicted Price: ${predicted_price:,.2f}')"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
