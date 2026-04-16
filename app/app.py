import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="Employee Performance Dashboard", layout="wide")

st.title("📊 Employee Performance Analytics Dashboard")

# Load model
model = joblib.load("../models/model.pkl")

# Upload file
uploaded_file = st.file_uploader("📁 Upload Employee CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("📋 Uploaded Data")
    st.dataframe(df.head())

    # Preprocess
    if 'performance' in df.columns:
        X = df.drop('performance', axis=1)
    else:
        X = df.copy()

    X_encoded = pd.get_dummies(X, drop_first=True)

    # Align columns (important fix)
    model_features = model.feature_names_in_
    for col in model_features:
        if col not in X_encoded.columns:
            X_encoded[col] = 0

    X_encoded = X_encoded[model_features]

    # Prediction
    predictions = model.predict(X_encoded)
    df['Predicted Performance'] = predictions

    st.subheader("📊 Prediction Results")
    st.dataframe(df)

    # 🔥 Chart 1: Performance Distribution
    st.subheader("📈 Performance Distribution")
    fig1, ax1 = plt.subplots()
    df['Predicted Performance'].value_counts().plot(kind='bar', ax=ax1)
    st.pyplot(fig1)

    # 🔥 Chart 2: Feature Importance
    st.subheader("📊 Feature Importance")

    importances = model.feature_importances_
    features = model.feature_names_in_

    fig2, ax2 = plt.subplots()
    ax2.barh(features, importances)
    st.pyplot(fig2)

    # 🔥 HR Recommendations
    st.subheader("🧠 HR Recommendations")

    def recommendation(row):
        if row['Predicted Performance'] == 0:
            return "⚠️ Needs training & monitoring"
        elif row['Predicted Performance'] == 1:
            return "📚 Moderate improvement required"
        else:
            return "🏆 Eligible for promotion"

    df['Recommendation'] = df.apply(recommendation, axis=1)

    st.dataframe(df[['Predicted Performance', 'Recommendation']])

    # Download
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download Results", csv, "results.csv", "text/csv")