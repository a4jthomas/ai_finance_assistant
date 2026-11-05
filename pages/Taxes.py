import streamlit as st

# ----------------------------------------------------
# 1. PAGE CONFIGURATION
# ----------------------------------------------------
st.set_page_config(
    page_title="Tax Calculator",
    page_icon="📊",
    layout="centered",
)

st.title("📊 Tax Calculator")
st.caption("Estimate your net income based on your annual salary and tax rate.")


# ----------------------------------------------------
# 2. INPUTS AND CALCULATIONS
# ----------------------------------------------------      
# Create two columns for inputs
col_salary, col_tax = st.columns(2)

with col_salary:
  annual_salary = st.number_input(
      "Annual Salary ($)", min_value=0.0, value=75000.0, step=1000.0, format="%.2f"
  )

with col_tax:
  tax_rate = st.number_input(
      "Tax Rate (%)", min_value=0.0, max_value=100.0, value=20.0, step=0.5
  )

# Calculations
tax_amount = annual_salary * (tax_rate / 100.0)
net_income = annual_salary - tax_amount

# Display Results using metrics
st.divider()
c1, c2, c3 = st.columns(3)
c1.metric("Gross Salary", f"${annual_salary:,.2f}")
c2.metric("Estimated Tax", f"${tax_amount:,.2f}")
c3.metric("Net Income", f"${net_income:,.2f}")