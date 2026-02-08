import kagglehub
import pandas as pd
import pickle
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

print("Downloading dataset...")

path = kagglehub.dataset_download("jacksoncrow/stock-market-dataset")

full_path = path + "/stocks/GOOGL.csv"

print("Dataset downloaded to:", full_path)
print("Reading full path...")

df = pd.read_csv(full_path)

print("Cleaning data and creating features...")

df_populated = df.dropna()
y = df_populated["Close"]
x = df_populated.drop(columns=["Date", "Adj Close", "Volume", "Close"])

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)
y_scaled = scaler.fit_transform(y.values.reshape(-1, 1))

x_train, x_test, y_train, y_test = train_test_split(x_scaled, y_scaled, test_size=0.2)

df_scaled = pd.DataFrame(x_scaled, columns=x.columns)

print("Creating model...")

model = Sequential([
    Dense(units=32, activation='relu', input_dim=x_train.shape[1]),
    Dense(units=16, activation='relu'),
    Dense(1)
])

optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mean_absolute_error'])

print("Training model...")

model.fit(x_train, y_train, epochs=10, batch_size=32)

print("Evaluating model...")

test_loss = model.evaluate(x_test, y_test)
y_pred = model.predict(x_test)

y_pred_real = scaler.inverse_transform(y_pred.reshape(-1, 1))
y_test_inverse = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

predictions = pd.DataFrame({'Actual': y_test_inverse, 'Predicted': y_pred_real.flatten()})
print(f"Test Loss (MSE): {test_loss}")
print(predictions.head())

print("Saving model...")

model_path = './stock-trader-model.keras'
model.save(model_path)

x_scaler_path = './x_scaler.pkl'
joblib.dump(scaler, x_scaler_path)

y_scaler_path = './y_scaler.pkl'
joblib.dump(scaler, y_scaler_path)

print("Completed!")



