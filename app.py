import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Student Academic Performance AI",
    layout="centered"
)

# ---------------- LOAD MODEL ----------------
model = joblib.load("student_model.pkl")
model_columns = joblib.load("model_columns.pkl")

# ---------------- CLEAN CSS ----------------
st.markdown("""
<style>

.stApp {
     background:
        radial-gradient(circle at 20% 20%, rgba(255,255,255,0.7), transparent 40%),
        radial-gradient(circle at 80% 80%, rgba(255,255,255,0.6), transparent 40%),
        linear-gradient(180deg, #eef1f5 0%, #d8dde5 100%);
}

h1 {
    font-size: 22px;
    font-weight: 600;
    margin-bottom: 5px;
}

.subtitle {
    font-size: 14px;
    color: #6b7280;
    margin-bottom: 25px;
}

/* Button */
.stButton>button {
    width: 100%;
    height: 42px;
    border-radius: 6px;
    background-color: #2563eb;
    color: white;
    font-weight: 500;
    border: none;
}

.stButton>button:hover {
    background-color: #1e40af;
}

/* Result box */
.result-box {
    padding: 18px;
    border-radius: 8px;
    background-color: #eef2ff;
    border: 1px solid #dbeafe;
    margin-top: 20px;
}

.mac-titlebar {
    height: 46px;
    display: flex;
    align-items: center;
    padding: 0 18px;
    background: rgba(255,255,255,0.55);
    border-bottom: 1px solid rgba(0,0,0,0.05);
}

.mac-controls {
    display: flex;
    gap: 8px;
}

.mac-dot {
    width: 13px;
    height: 13px;
    border-radius: 50%;
}

.red { background: #ff5f57; }
.yellow { background: #febc2e; }
.green { background: #28c840; }

.mac-content {
    padding: 35px;
}

</style>
""", unsafe_allow_html=True)

# ---------------- TRAFFIC BAR ----------------
st.markdown("""
<div class="mac-titlebar">
    <div class="mac-controls">
        <div class="mac-dot red"></div>
        <div class="mac-dot yellow"></div>
        <div class="mac-dot green"></div>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.title("Student Academic Performance")
st.markdown(
    '<div class="subtitle">Machine learning powered CGPA prediction system</div>',
    unsafe_allow_html=True
)

# ---------------- INPUTS ----------------
st.subheader("Student Details")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    devices = st.slider("Number of Devices", 1, 5)

with col2:
    academic_hours = st.slider("Academic Hours per Day", 1, 15)
    study_year = st.selectbox("Study Year", ["1", "2", "3", "4", "Post Graduate"])

cgpa_trend = st.selectbox("CGPA Trend", ["Increase", "Decrease", "None"])

predict = st.button("Predict CGPA")

# ---------------- RESULT AT BOTTOM ----------------
if predict:

    input_dict = {
        "Gender": 1 if gender == "Male" else 0,
        "Devices": devices,
        "Academic Hours": academic_hours,
        "CGPA Trend": {"Increase": 1, "Decrease": 0, "None": 2}[cgpa_trend]
    }

    input_df = pd.DataFrame([input_dict])
    input_df = pd.get_dummies(input_df)

    for col in model_columns:
        if col not in input_df:
            input_df[col] = 0

    input_df = input_df[model_columns]
    prediction = model.predict(input_df)[0]

    # Performance Level
    if prediction >= 3.5:
        performance = "Excellent"
    elif prediction >= 3.0:
        performance = "Good"
    else:
        performance = "Needs Improvement"

    # ---------------- RESULT BOX ----------------
    st.markdown(f"""
    <div class="result-box">
        <strong>Predicted CGPA:</strong> {prediction:.2f} <br>
        <strong>Performance Level:</strong> {performance}
    </div>
    """, unsafe_allow_html=True)

    # ---------------- EXPLANATION ----------------
    st.markdown("### Why this CGPA?")
    st.write(
        f"The model prediction is influenced mainly by Academic Hours ({academic_hours}) "
        f"and CGPA Trend ({cgpa_trend})."
    )

    # ---------------- RECOMMENDATIONS ----------------
    st.markdown("### Recommendations")

    if academic_hours < 4:
        st.write(" Increase daily academic hours.")
    if devices > 3:
        st.write(" Reduce device usage for better focus.")
    if cgpa_trend == "Decrease":
        st.write(" Review weak subjects and revise regularly.")

    st.divider()

    # ---------------- MODEL INSIGHTS ----------------
    st.subheader("Model Insights")

    importances = model.feature_importances_

    feature_df = (
        pd.DataFrame({
            "Feature": model_columns,
            "Importance": importances
        })
        .sort_values(by="Importance", ascending=False)
        .head(10)
    )

    fig, ax = plt.subplots()
    ax.barh(feature_df["Feature"], feature_df["Importance"])
    ax.invert_yaxis()
    ax.set_xlabel("Importance Score")

    st.pyplot(fig)

st.caption("Built with Streamlit • Random Forest Regression")