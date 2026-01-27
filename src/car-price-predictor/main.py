import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler

def create_model(input_dim): 
    """Create and compile the neural network model"""
    model = Sequential([
        # Input layer with 32 neurons
        Dense(units=32, activation='relu', input_dim=input_dim),
        # Second hidden layer with 16 neurons
        Dense(units=16, activation='relu'),
        # Output layer for regression
        Dense(1)
    ])

    # Optimize with loss function - higher learning rate for faster convergence
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mean_absolute_error'])
    return model

def load_data(csv_path):
    """Load data from CSV file"""
    df = pd.read_csv(csv_path)
    return df

def preprocess_data(df):
    """Preprocess data: clean, remove columns, create dummy variables, and normalize"""
    # Removing rows with missing values
    df_cleaned = df.dropna()

    # Remove unnecessary columns and convert categorical to dummy variables
    X = pd.get_dummies(df_cleaned.drop(['Price', 'Make', 'Model'], axis=1))

    # Identity output column
    y = df_cleaned['Price']

    # Normalize the numerical features
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X)

    return X_normalized, y, scaler

def train_model(X_train, y_train, input_dim, epochs=50, batch_size=32):
    """Train the model"""
    model = create_model(input_dim)
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""
    test_loss = model.evaluate(X_test, y_test)
    y_pred = model.predict(X_test)
    
    predictions = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred.flatten()})
    print(f"Test Loss (MSE): {test_loss}")
    print(predictions.head())
    
    return test_loss, predictions

def save_model(model, model_path):
    """Save the trained model"""
    model.save(model_path)
    print(f"Model saved to {model_path}")

def save_scaler(scaler, scaler_path):
    """Save the fitted scaler"""
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Scaler saved to {scaler_path}")

if __name__ == "__main__":
    # Load data
    df = load_data("hf://datasets/Yash007001/Car-Price-Prediction/Car_Price_Prediction.csv")
    
    # Preprocess data
    X_normalized, y, scaler = preprocess_data(df)
    
    # Split data into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.2)
    
    # Train model
    model = train_model(X_train, y_train, X_train.shape[1])
    
    # Evaluate model
    evaluate_model(model, X_test, y_test)
    
    # Save model and scaler
    save_model(model, 'src/car-price-predictor/car-price-model.keras')
    save_scaler(scaler, 'src/car-price-predictor/scaler.pkl')