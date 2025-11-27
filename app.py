import streamlit as st
import pandas as pd
import joblib
import EDA  # ✅ EDA module for charts

# Same feature list
feature_cols = [
    "Age",
    "MonthlyIncome",
    "JobSatisfaction",
    "YearsAtCompany",
    "Education",
    "JobInvolvement",
    "JobLevel",
    "YearsInCurrentRole",
]

# Load data
clean_df = pd.read_csv("clean_employee.csv")
df_eda = EDA.load_data()

# Load models
attr_model = joblib.load("attrition_model.pkl")
perf_model = joblib.load("performance_model.pkl")

# ---- Title  ----
st.title("Employee Attrition")
st.write("- by Ashwin")
st.write("---")

# ---- Tiles ----
col1, col2 = st.columns(2)

# --------- EDA TILE (LEFT) ----------
with col1:
    st.markdown(
        """
        <div style='border:2px solid #4CAF50; padding:20px; border-radius:10px;height:200px;'>
            <h3 style='text-align:center;'>📊 EDA</h3>
            <p style='text-align:center;'>Select an option to view EDA chart.</p>
        """,
        unsafe_allow_html=True,
    )
st.write("---")

# EDA selection inside tile
eda_option = st.selectbox(
    "Choose EDA View",
    [
        "Attrition Count",
        "Age Distribution",
        "Monthly Income Distribution",
        "Attrition by Department",
        "Attrition by Job Role",
    ],
)

if eda_option == "Attrition Count":
    fig = EDA.attrition_count(df_eda)
    st.pyplot(fig)

elif eda_option == "Age Distribution":
    fig = EDA.age_distribution(df_eda)
    st.pyplot(fig)

elif eda_option == "Monthly Income Distribution":
    fig = EDA.income_distribution(df_eda)
    st.pyplot(fig)

elif eda_option == "Attrition by Department":
    fig = EDA.department_attrition(df_eda)
    st.pyplot(fig)

elif eda_option == "Attrition by Job Role":
    fig = EDA.jobrole_attrition(df_eda)
    st.pyplot(fig)

st.markdown("</div>", unsafe_allow_html=True)

# --------- PREDICTION TILE (RIGHT) ----------
with col2:
    st.markdown(
        """
        <div style='border:2px solid #2196F3; padding:20px; border-radius:10px; height:200px;'>
            <h3 style='text-align:center;'>🤖 Prediction</h3>
            <p style='text-align:center;'>Employee Attrition & Performance Prediction.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.write("---")

# Defaults for form

num_defaults = clean_df.median(numeric_only=True)
age_default = int(num_defaults["Age"])
income_default = int(num_defaults["MonthlyIncome"])
jobsat_default = int(num_defaults["JobSatisfaction"])
yearscomp_default = int(num_defaults["YearsAtCompany"])
education_default = int(num_defaults["Education"])
jobinv_default = int(num_defaults["JobInvolvement"])
joblevel_default = int(num_defaults["JobLevel"])
yearscurrole_default = int(num_defaults["YearsInCurrentRole"])

# ----- Prediction Form -----
"Prediction Arena"
age = st.number_input("Age", min_value=18, max_value=60, value=age_default, step=1)
monthly_income = st.number_input(
    "Monthly Income", min_value=1000, max_value=100000, value=income_default, step=500
)
job_satisfaction = st.selectbox("Job Satisfaction", [1, 2, 3, 4], index=jobsat_default - 1)
education = st.selectbox("Education", [1, 2, 3, 4, 5], index=education_default - 1)
job_involvement = st.selectbox("Job Involvement", [1, 2, 3, 4], index=jobinv_default - 1)
job_level = st.selectbox("Job Level", [1, 2, 3, 4, 5], index=joblevel_default - 1)
years_at_company = st.number_input(
    "Years at Company", min_value=0, max_value=40, value=yearscomp_default, step=1
)
years_in_current_role = st.number_input(
    "Years in Current Role", min_value=0, max_value=40, value=yearscurrole_default, step=1
)

if st.button("Predict"):
    data = {
        "Age": age,
        "MonthlyIncome": monthly_income,
        "JobSatisfaction": job_satisfaction,
        "YearsAtCompany": years_at_company,
        "Education": education,
        "JobInvolvement": job_involvement,
        "JobLevel": job_level,
        "YearsInCurrentRole": years_in_current_role,
    }

    user_df = pd.DataFrame([data])
    user_df = user_df[feature_cols]

    st.subheader("Prediction Results :")

    # Attrition
    attr_pred = attr_model.predict(user_df)[0]
    attr_prob = attr_model.predict_proba(user_df)[0][1]

    if attr_pred == 1:
        st.write("🔴 Employee is **likely to LEAVE**.")
    else:
        st.write("🟢 Employee is **likely to STAY**.")

    st.write("⭐  Probability of leaving:", round(attr_prob, 2))

    # Performance rating
    perf_pred = perf_model.predict(user_df)[0]
    st.write("⭐ Predicted Performance Rating:", round(perf_pred, 2))

# ---- Footer line ----
st.write("---")
