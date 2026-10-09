
import json
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
df = pd.read_csv("house_prices_practice.csv")

print("Dataset Shape:", df.shape)
print("\nDataset Preview:")
print(df.head())

# Select features and target
features = [
    "OverallQual",
    "GrLivArea",
    "GarageCars",
    "TotalBsmtSF",
    "YearBuilt",
    "FullBath",
    "BedroomAbvGr",
    "LotArea"
]

target = "SalePrice"

X = df[features]
y = df[target]

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# Build the regression pipeline
model = Pipeline([
    ("scaler", StandardScaler()),
    ("regressor", RandomForestRegressor(
        n_estimators=100,
        random_state=42
    ))
])

# Train the model
model.fit(X_train, y_train)

# Generate predictions
predictions = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

print("\nModel Evaluation Results")
print("Training Records:", len(X_train))
print("Testing Records:", len(X_test))
print("MAE:", round(mae, 2))
print("RMSE:", round(rmse, 2))
print("R2 Score:", round(r2, 4))

# Save the trained model
joblib.dump(model, "house_price_model.pkl")

# Save metrics for the quality gate
metrics = {
    "mae": float(mae),
    "rmse": float(rmse),
    "r2_score": float(r2),
    "training_records": int(len(X_train)),
    "testing_records": int(len(X_test))
}

with open("metrics.json", "w") as file:
    json.dump(metrics, file, indent=4)

print("\nModel saved successfully.")
print("Metrics saved to metrics.json.")
