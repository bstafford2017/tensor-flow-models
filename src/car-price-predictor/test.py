import pickle
import pandas as pd
import tensorflow as tf
import os

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Load model
model = tf.keras.models.load_model(os.path.join(script_dir, 'car-price-model.keras'))

# Load scaler
with open(os.path.join(script_dir, 'scaler.pkl'), 'rb') as f:
    scaler = pickle.load(f)

# Hardcode feature columns (from training data)
train_columns = ['Year', 'Engine Size', 'Mileage', 'Fuel Type_Diesel', 
                 'Fuel Type_Electric', 'Fuel Type_Petrol', 
                 'Transmission_Automatic', 'Transmission_Manual']

# Create test input
new_car = pd.DataFrame({
    'Year': [2024],
    'Engine Size': [2.0],
    'Mileage': [1000],
    'Fuel Type': ['Petrol'],
    'Transmission': ['Automatic']
})

# Preprocess: convert categorical to dummies and match training columns
new_car_dummies = pd.get_dummies(new_car)
new_car_dummies = new_car_dummies.reindex(columns=train_columns, fill_value=0)

# Normalize using saved scaler
new_car_normalized = scaler.transform(new_car_dummies)

# Make prediction
predicted_price = model.predict(new_car_normalized)
print(f"Predicted price: ${predicted_price[0][0]:.2f}")