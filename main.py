import pandas as pd
from sklearn.model_selection import train_test_split

from src.preprocessing import load_and_preprocess
from src.model import train_model
from src.evaluate import evaluate_model


def main():
    print("🚀 Starting Employee Performance Pipeline...")

    # 🔹 Load dataset
    df = pd.read_csv("data/employee_data.csv")
    print("✅ Data loaded successfully!")

    # 🔹 Preprocessing
    X, y, le = load_and_preprocess("data/employee_data.csv")
    print("✅ Data preprocessing completed!")

    # 🔹 Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print("✅ Data split into train and test!")

    # 🔹 Train model
    model = train_model(X_train, y_train)   # IMPORTANT
    print("✅ Model trained successfully!")

    # 🔹 Prediction
    y_pred = model.predict(X_test)

    # 🔹 Evaluation
    accuracy, cm = evaluate_model(y_test, y_pred)

    print(f"\n📊 Accuracy: {accuracy:.2f}")
    print("📉 Confusion Matrix:\n", cm)

    # 🔹 Save predictions
    results_df = X_test.copy()
    results_df["Actual"] = y_test
    results_df["Predicted"] = y_pred

    results_df.to_csv("outputs/predictions.csv", index=False)

    # 🔹 Save results
    with open("outputs/results.txt", "w") as f:
        f.write(f"Accuracy: {accuracy:.2f}\n")
        f.write("Confusion Matrix:\n")
        f.write(str(cm))

    print("✅ Predictions and results saved in outputs/ folder")


if __name__ == "__main__":
    main()