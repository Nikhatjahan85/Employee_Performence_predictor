import os
import pandas as pd
from sklearn.model_selection import train_test_split
import joblib

from src.preprocessing import load_and_preprocess
from src.model import train_model
from src.evaluate import evaluate_model

# Load data
X, y, le = load_and_preprocess("data/employee_data.csv")

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train
model = train_model(X_train, y_train)

# Evaluate
acc, cm, report, y_pred = evaluate_model(model, X_test, y_test)

# Print results
print("Accuracy:", acc)
print("Confusion Matrix:\n", cm)
print("Classification Report:\n", report)

# Save model
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/model.pkl")

# Save outputs
os.makedirs("outputs", exist_ok=True)

pd.DataFrame(y_pred, columns=["Predictions"]).to_csv("outputs/predictions.csv", index=False)

with open("outputs/results.txt", "w") as f:
    f.write(f"Accuracy: {acc}\n\n")
    f.write(str(cm))
    f.write("\n\n")
    f.write(report)

print("✅ Model & outputs saved!")