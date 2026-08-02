import streamlit as st
import pandas as pd
from core_engine import run_audit_pipeline, simulate_hotel_data, generate_true_revenue

st.set_page_config(page_title="Hotel Revenue Fraud Dashboard", layout="wide")

@st.cache_data
def get_data():
    return generate_true_revenue(simulate_hotel_data())

st.sidebar.title("⚙️ Control Panel")
uploaded_file = st.sidebar.file_uploader("1. Upload Data (Excel/CSV)", type=["xlsx", "csv"])
user_train_ratio = st.sidebar.slider("Train/Test Split:", 0.50, 0.90, 0.70, 0.05)
user_fraud_mag = st.sidebar.slider("Fraud Magnitude:", 0.01, 0.50, 0.10, 0.01)

raw_data = pd.read_excel(uploaded_file) if uploaded_file else get_data()

causal_res, scored_data, eval_res, action_sum = run_audit_pipeline(raw_data, user_fraud_mag, user_train_ratio)

st.title("🏨 Hotel Revenue Fraud Detection Dashboard")
st.markdown("Automated anomaly detection with human-in-the-loop action workflows.")

r1c1, r1c2 = st.columns(2)
with r1c1:
    st.subheader("1. Causal Recovery (Explanation)")
    st.dataframe(causal_res.round(3), use_container_width=True)
    st.info("💡 **Hotel Fixed-Effects** explicitely blocks unobserved confounders to eliminate bias.")

with r1c2:
    st.subheader("2. Model Detection Metrics")
    st.dataframe(eval_res.round(3), use_container_width=True)

r2c1, r2c2 = st.columns(2)
with r2c1:
    st.subheader("3. High-Risk Audit Queue")
    st.dataframe(scored_data.nlargest(10, "posterior_fraud_probability")[["hotel_id", "day", "observed_revenue", "posterior_fraud_probability", "audit_action"]].round(3), use_container_width=True)

with r2c2:
    st.subheader("4. Action Layering (B's Design)")
    st.markdown("Directive mapping of probability bounds to exact audit workflows.")
    st.dataframe(action_sum.style.format({"mean_probability": "{:.2%}", "max_probability": "{:.2%}"}), use_container_width=True)