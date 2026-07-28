import streamlit as st
import pandas as pd
import backend  

# -----------------------------------------------------------------------------
# Configuration & Page Setup
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Hotel Revenue Fraud Detection Dashboard",
    page_icon="🏨",
    layout="wide"
)

# -----------------------------------------------------------------------------
# Data Loading & Processing Functions
# -----------------------------------------------------------------------------
@st.cache_data
def get_default_simulation_data():
    """
    Default logic (Requirement 6): Generates simulation data if no file is uploaded.
    """
    hotel_data = backend.simulate_hotel_data(n_hotels=50, n_days=365, seed=100)
    base_data = backend.generate_true_revenue(
        hotel_data, seed=100, true_price_effect=500, revenue_noise_sd=2750
    )
    return base_data

@st.cache_data
def process_data(base_data, fraud_magnitude, train_ratio):
    """
    Executes the algorithmic pipeline from the backend module using provided data and parameters.
    """
    # 1. Inject fraud using the UI-controlled magnitude (Requirement 4)
    scenario_data = backend.inject_fraud(
        base_data, fraud_magnitude=fraud_magnitude, seed=190, intercept=-3.35
    )
    
    # 2. Estimate causal recovery (on base true revenue)
    causal_results = backend.estimate_causal_recovery(
        base_data, true_price_effect=500
    )
    causal_results["relative_bias"] = causal_results["relative_bias"] * 100
    
    # 3. Train model & score expected revenue using UI-controlled split (Requirement 5)
    train_data, test_data, split_day = backend.split_train_test(scenario_data, train_ratio=train_ratio)
    expected_rev_model, test_data = backend.add_expected_revenue(train_data, test_data)
    
    # 4. Standardize residuals and compute Bayesian scores
    test_data, center, scale = backend.standardize_residuals(test_data)
    scored_data = backend.add_bayesian_score(test_data, prior=0.05)
    
    # 5. Evaluate detection performance thresholds
    eval_results = backend.evaluate_thresholds(
        scored_data, 
        score_column="posterior_fraud_probability", 
        thresholds=[0.10, 0.20, 0.30, 0.50], 
        method_name="Bayesian posterior score"
    )
    
    # 6. Compute Top-K prioritizing efficiency
    top5_pool = backend.summarize_top_review_pool(
        scored_data, 
        score_column="posterior_fraud_probability", 
        top_fraction=0.05
    )
    
    return causal_results, scored_data, eval_results, top5_pool


# -----------------------------------------------------------------------------
# Sidebar: User Inputs & Controls
# -----------------------------------------------------------------------------
st.sidebar.title("⚙️ Control Panel")
st.sidebar.markdown("Upload your own data or use the default simulation.")

# Requirement 1: Excel data upload component
uploaded_file = st.sidebar.file_uploader("1. Upload Hotel Data (Excel)", type=["xlsx", "xls"])

st.sidebar.markdown("---")
st.sidebar.subheader("2. Model Parameters")

# Requirement 5: Train/Test split slider
user_train_ratio = st.sidebar.slider(
    "Train/Test Split Ratio:", 
    min_value=0.50, max_value=0.90, value=0.70, step=0.05,
    help="Determines the percentage of historical data used to train the baseline revenue model."
)

# Requirement 4: Fraud Magnitude slider
user_fraud_magnitude = st.sidebar.slider(
    "Fraud Magnitude:", 
    min_value=0.01, max_value=0.50, value=0.10, step=0.01,
    help="Adjust the simulated intensity of revenue suppression/inflation."
)

# -----------------------------------------------------------------------------
# Execute Pipeline based on User Input
# -----------------------------------------------------------------------------
# Requirement 6 & 1: Data parsing and default fallback
if uploaded_file is not None:
    try:
        # Parse incoming Excel data
        raw_data = pd.read_excel(uploaded_file)
        st.sidebar.success("✅ Excel file loaded successfully!")
    except Exception as e:
        st.sidebar.error(f"Error reading file: {e}")
        st.stop()
else:
    # Default fallback to simulation
    raw_data = get_default_simulation_data()
    st.sidebar.info("ℹ️ No file uploaded. Running default simulation.")

# Requirement 2: Process data using backend logic
causal_results, scored_data, eval_results, top5_pool = process_data(
    base_data=raw_data, 
    fraud_magnitude=user_fraud_magnitude, 
    train_ratio=user_train_ratio
)


# -----------------------------------------------------------------------------
# UI / Dashboard Layout (Requirement 3: 2x2 Grid)
# -----------------------------------------------------------------------------
st.title("🏨 RegTech Audit Dashboard: Hotel Revenue Fraud")
st.markdown("Automated anomaly detection, causal recovery modeling, and Bayesian risk scoring for compliance auditing.")
st.markdown("---")

row1_col1, row1_col2 = st.columns(2)
row2_col1, row2_col2 = st.columns(2)

# ==========================================
# Quadrant 1: Causal Recovery
# ==========================================
with row1_col1:
    st.subheader("1. Causal Recovery & Bias Reduction")
    st.markdown("Comparing true price effect recovery across statistical models.")
    st.dataframe(
        causal_results[["model", "estimated_price_effect", "bias", "relative_bias", "ci_covers_true_effect"]].round(3),
        use_container_width=True
    )
    
    with st.expander("🔍 Explainability & Causal Defense", expanded=False):
        st.write(
            "**Why was this model chosen and how does it prevent false alerts?**\n\n"
            "The **Hotel Fixed-Effects Model** explicitly blocks backdoor paths generated by unobserved, time-invariant hotel-level confounders (such as static hotel quality and location advantages).\n\n"
            "By using a panel fixed-effects estimator (`C(hotel_id)`), we isolate the intra-hotel variation. This ensures that the algorithm captures true demand elasticity rather than spurious correlations, effectively eliminating estimation bias and establishing a legally and mathematically defensible baseline for 'normal' revenue."
        )

# ==========================================
# Quadrant 2: Detection Performance
# ==========================================
with row1_col2:
    st.subheader("2. Model Detection Performance")
    st.markdown("Bayesian anomaly classification metrics across posterior risk thresholds.")
    
    st.dataframe(
        eval_results[["threshold", "precision", "recall", "f1_score", "number_flagged", "fraud_found"]].round(3),
        use_container_width=True
    )
    
    st.info("💡 **Analyst Note:** A 0.3 risk threshold historically balances precision and recall, optimizing the F1-score for this hotel corpus.")

st.markdown("---")

# ==========================================
# Quadrant 3: Top-K Prioritization
# ==========================================
with row2_col1:
    st.subheader("3. High-Risk Audit Queue (Top 5%)")
    st.markdown("Prioritizing the highest-probability anomalies maximizes auditor efficiency.")
    
    st.dataframe(top5_pool.round(3), use_container_width=True)
    
    st.markdown("**Top 10 Flagged Hotel-Days for Immediate Review:**")
    high_risk_queue = scored_data.nlargest(10, "posterior_fraud_probability")
    st.dataframe(
        high_risk_queue[["hotel_id", "day", "observed_revenue", "expected_revenue", "z_residual", "posterior_fraud_probability"]].round(3),
        use_container_width=True
    )

# ==========================================
# Quadrant 4: Action Layering & Human-in-the-Loop
# ==========================================
with row2_col2:
    st.subheader("4. Action Layering & Workflow")
    st.markdown("Mapping flagged statistical anomalies to investigative stages.")
    
    st.warning(
        "⚠️ **Human-in-the-Loop Requirement:**\n\n"
        "Statistical flags identify high-probability anomalies. Human auditor vouching and substantive testing remain mandatory to definitively distinguish intentional fraud from system leaks or human data-entry errors."
    )
    
    st.markdown("**Select Risk Threshold for Case Assignment:**")
    selected_threshold = st.slider("Bayesian Risk Threshold:", min_value=0.10, max_value=0.90, value=0.30, step=0.05)
    
    flagged_cases = scored_data[scored_data["posterior_fraud_probability"] >= selected_threshold]
    
    st.metric(label="Cases Mapped to Tier-1 Audit Workflow", value=len(flagged_cases))
    
    st.write(f"Routing {len(flagged_cases)} cases with a > {selected_threshold * 100}% algorithmic fraud probability to the Substantive Testing Queue. Review teams will cross-reference these days with physical occupancy logs and folio receipts.")