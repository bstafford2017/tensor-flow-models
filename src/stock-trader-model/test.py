import joblib
import tensorflow as tf
import pandas as pd

x_scaler = joblib.load("x_scaler.pkl")
y_scaler = joblib.load("y_scaler.pkl")
model = tf.keras.models.load_model("stock-trader-model.keras")

x = [[1124.0, 1129.4200439453125, 1093.489990234375]]
print("Actual:", x)

x_infer = pd.DataFrame(x, columns=["Open", "High", "Low"])

X_new_scaled = x_scaler.fit_transform(x_infer)

pred_scaled = model.predict(X_new_scaled)

pred_real = y_scaler.inverse_transform(pred_scaled)
y_test_inverse = y_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()

print("Predicted:", y_test_inverse)