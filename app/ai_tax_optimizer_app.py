import streamlit as st
import pandas as pd
import numpy as np
import joblib
import oracledb


# Load models and scaler
tax_model = joblib.load("models/tax_estimator_rf.pkl")
cf_model = joblib.load("models/cf_knn_model.pkl")
cf_reference = joblib.load("models/cf_reference_matrix.pkl")
scaler = joblib.load("models/input_scaler.pkl")

st.set_page_config(page_title="AI Tax Optimizer", layout="centered", page_icon="🧾")

st.markdown("""
    <h1 style='text-align:center;'>🧠 AI-Powered Tax Optimizer</h1>
    <p style='text-align:center;'>Estimate your tax & get smart deduction + investment suggestions based on your income & profile.</p>
    <hr style="border:1px solid #ccc">
""", unsafe_allow_html=True)

# 📋 Use a form to allow Enter-key navigation
with st.form(key="tax_form"):
    st.markdown("### 👤 Personal Details")
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=30, step=1)
    with col2:
        total_income = st.number_input("Yearly Total Income (₹)", min_value=0.0, step=1000.0)

    st.markdown("### 💼 Income & Deductions (Yearly)")
    salary_income = st.number_input("Salary Income (₹)", min_value=0.0, step=500.0)
    business_income = st.number_input("Business Income (₹)", min_value=0.0, step=500.0)
    capital_gains = st.number_input("Capital Gains (₹)", min_value=0.0, step=500.0)

    st.markdown("### 🏠 Loan & Rent (Yearly)")
    home_loan_interest = st.number_input("Home Loan Interest Paid (₹)", min_value=0.0, step=500.0)
    education_loan_interest = st.number_input("Education Loan Interest (₹)", min_value=0.0, step=500.0)
    rent_paid = st.number_input("Rent Paid (₹)", min_value=0.0, step=500.0)

    st.markdown("### 📈 Investments (Yearly)")
    mutual_funds = st.number_input("Mutual Fund Investments (₹)", min_value=0.0, step=500.0)
    fixed_deposits = st.number_input("Fixed Deposits (₹)", min_value=0.0, step=500.0)
    real_estate = st.number_input("Real Estate Investment (₹)", min_value=0.0, step=500.0)
    nps_contribution = st.number_input("NPS Contribution (₹)", min_value=0.0, step=500.0)

    st.markdown("### 🧾 Tax Deduction Declarations (Monthly Values)")
    section_80C = st.number_input("Section 80C Deduction (₹)", min_value=0.0, step=500.0)
    section_80D = st.number_input("Section 80D Deduction (₹)", min_value=0.0, step=500.0)

    submit_button = st.form_submit_button(label="📊 Optimize My Tax")

# 🧮 Handle submission
if submit_button:
    try:
        # Convert monthly to yearly
        def to_yearly(val): return round(val * 12, 2)
        input_data = {
            'age': age,
            'total_income': to_yearly(total_income),
            'salary_income': to_yearly(salary_income),
            'business_income': to_yearly(business_income),
            'capital_gains': to_yearly(capital_gains),
            'home_loan_interest': to_yearly(home_loan_interest),
            'education_loan_interest': to_yearly(education_loan_interest),
            'rent_paid': to_yearly(rent_paid),
            'mutual_funds': to_yearly(mutual_funds),
            'fixed_deposits': to_yearly(fixed_deposits),
            'real_estate': to_yearly(real_estate),
            'nps_contribution': to_yearly(nps_contribution),
            'section_80C': to_yearly(section_80C),
            'section_80D': to_yearly(section_80D)
        }

        total_sum = sum(v for k, v in input_data.items() if k != 'age')
        if total_sum == 0:
            estimated_tax = 0.0
            recommended_deduction = 0.0
            recommended_investment = 0.0
        else:
            input_df = pd.DataFrame([input_data])
            estimated_tax = tax_model.predict(input_df)[0]

            scaled_input = scaler.transform(input_df)
            distances, indices = cf_model.kneighbors(scaled_input)
            similar_users = cf_reference.iloc[indices[0]]
            recommended_deduction = similar_users['total_deductions'].mean()
            recommended_investment = similar_users['total_investments'].mean()

        # 🧾 Results
        st.markdown("## ✅ Results")
        st.success(f"**Estimated Annual Tax**: ₹{estimated_tax:,.2f}")
        st.info(f"**Recommended Deductions**: ₹{recommended_deduction:,.2f}")
        st.info(f"**Recommended Investments**: ₹{recommended_investment:,.2f}")

        # 🎯 Investment Suggestions
        st.markdown("---")
        st.markdown("### 🎯 Personalized Investment Suggestions")
        if recommended_investment == 0:
            st.write("📌 Consider starting with **ELSS Mutual Funds** or **Tax-saving Fixed Deposits (FDs)** for low-risk options.")
        elif recommended_investment < 100000:
            st.write("📌 Start with **ELSS Mutual Funds** or **Tax-saving FDs** for safe and low-minimum investment options.")
        elif 100000 <= recommended_investment <= 250000:
            st.write("📌 Diversify: Allocate into **NPS**, **Mutual Funds**, and optionally start a **Real Estate SIP**.")
        else:
            st.write("📌 Go big: Combine **NPS**, **Real Estate**, and **ELSS**. Aim to use the ₹1.5L 80C limit and invest beyond for long-term growth.")
        # 📘 Deduction Tips
        st.markdown("### 📘 Deduction Maximization Tips")
        if recommended_deduction == 0:
            st.write("🧠 Start with **Section 80C** investments like **LIC, ELSS, PPF**, and **Tuition Fees**.")
        elif recommended_deduction < 100000:
            st.write("🧠 Use Section 80C instruments like **LIC, ELSS, PPF**, and **Tuition Fees**.")
        elif 100000 <= recommended_deduction <= 200000:
            st.write("🧠 Include **health insurance (80D)**, **education/home loan interest**, and **NPS contributions**.")
        else:
            st.write("🧠 Great job! You've covered most deductions. Check niche ones like **80E (education)** or **80G (donations)**.")

        # EDA Image
        st.markdown("### 📈 EDA Correlation Heatmap")
        st.image("eda/eda_correlation_heatmap.png")

    except Exception as e:
        st.error(f"❌ Error: {e}")
    

