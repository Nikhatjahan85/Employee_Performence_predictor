import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="Employee Dashboard", layout="wide")

# -------------------------------
# CUSTOM CSS (PREMIUM UI)
# -------------------------------
st.markdown("""
<style>
.main {
    background-color: #0E1117;
    color: white;
}
.metric-card {
    background-color: #1c1f26;
    padding: 20px;
    border-radius: 10px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

st.title("📊 Employee Performance Analytics Dashboard")

# -------------------------------
# LOAD MODEL
# -------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, "models", "model.pkl")

if not os.path.exists(model_path):
    st.error("❌ Run main.py first to generate model.pkl")
    st.stop()

model = joblib.load(model_path)

# -------------------------------
# SIDEBAR
# -------------------------------
st.sidebar.header("⚙ Controls")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

# -------------------------------
# MAIN LOGIC
# -------------------------------
if uploaded_file:

    df = pd.read_csv(uploaded_file)

    st.subheader("📄 Uploaded Data")
    st.dataframe(df, use_container_width=True)

    # -------------------------------
    # FILTERS
    # -------------------------------
    st.sidebar.subheader("🔍 Filters")

    if "department" in df.columns:
        dept = st.sidebar.multiselect("Department", df["department"].unique(), default=df["department"].unique())
        df = df[df["department"].isin(dept)]

    if "salary" in df.columns:
        min_sal, max_sal = int(df["salary"].min()), int(df["salary"].max())
        sal_range = st.sidebar.slider("Salary Range", min_sal, max_sal, (min_sal, max_sal))
        df = df[(df["salary"] >= sal_range[0]) & (df["salary"] <= sal_range[1])]

    # -------------------------------
    # PREPROCESS
    # -------------------------------
    df_copy = df.copy()

    if "performance" in df_copy.columns:
        df_copy = df_copy.drop("performance", axis=1)

    df_processed = pd.get_dummies(df_copy, drop_first=True)

    try:
        df_processed = df_processed.reindex(columns=model.feature_names_in_, fill_value=0)
    except:
        pass

    # -------------------------------
    # PREDICTIONS
    # -------------------------------
    predictions = model.predict(df_processed)
    df["Predicted Performance"] = predictions

    # -------------------------------
    # KPI CARDS
    # -------------------------------
    st.subheader("📊 Key Metrics")

    col1, col2, col3 = st.columns(3)

    col1.metric("👥 Total Employees", len(df))
    col2.metric("💰 Avg Salary", f"{int(df['salary'].mean())}")
    col3.metric("📈 Avg Experience", f"{round(df['experience'].mean(),1)} yrs")

    # -------------------------------
    # PREDICTION TABLE
    # -------------------------------
    st.subheader("🔮 Prediction Results")
    st.dataframe(df, use_container_width=True)

    # -------------------------------
    # VISUALS
    # -------------------------------
    st.subheader("📈 Insights")

    col1, col2 = st.columns(2)

    with col1:
        fig1, ax1 = plt.subplots()
        sns.histplot(df["salary"], kde=True, ax=ax1)
        st.pyplot(fig1)

    with col2:
        fig2, ax2 = plt.subplots()
        sns.countplot(x="Predicted Performance", data=df, ax=ax2)
        st.pyplot(fig2)

    # -------------------------------
    # HR RECOMMENDATION
    # -------------------------------
    st.subheader("🧠 HR Insights")

    def recommendation(p):
        if p == 0:
            return "⚠ Needs Training"
        elif p == 1:
            return "🟡 Moderate Performer"
        else:
            return "🟢 High Performer"

    df["Recommendation"] = df["Predicted Performance"].apply(recommendation)

    st.dataframe(df[["Predicted Performance", "Recommendation"]])

    # -------------------------------
    # DOWNLOAD
    # -------------------------------
    st.download_button(
        "⬇ Download Results",
        df.to_csv(index=False),
        "predictions.csv",
        "text/csv"
    )

else:
    st.info("👈 Upload employee dataset to start analysis")