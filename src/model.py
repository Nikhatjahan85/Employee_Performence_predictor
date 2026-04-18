from sklearn.ensemble import RandomForestClassifier
import joblib
import os

def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Save model
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/model.pkl")

    print("✅ Model saved successfully!")

    return model