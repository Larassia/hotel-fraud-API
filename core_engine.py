import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score

def simulate_hotel_data(n_hotels=50, n_days=365, seed=100):
    rng = np.random.default_rng(seed)
    hotel_ids = np.arange(n_hotels)
    days = np.arange(n_days)
    data = pd.MultiIndex.from_product([hotel_ids, days], names=["hotel_id", "day"]).to_frame(index=False)
    
    hotel_quality = rng.normal(0, 1, size=n_hotels)
    quality_map = dict(zip(hotel_ids, hotel_quality))
    data["hotel_quality"] = data["hotel_id"].map(quality_map)
    
    data["seasonality"] = 0.35 * np.sin(2 * np.pi * data["day"] / 365) + 0.15 * np.sin(2 * np.pi * data["day"] / 7) + rng.normal(0, 0.05, size=len(data))
    data["market_trend"] = 0.20 * np.sin(2 * np.pi * data["day"] / 180) + 0.001 * data["day"] + rng.normal(0, 0.03, size=len(data))
    data["competitor_price"] = 100 + 12 * data["seasonality"] + 8 * data["market_trend"] + 5 * data["hotel_quality"] + rng.normal(0, 5, size=len(data))
    data["price"] = 90 + 10 * data["seasonality"] + 6 * data["market_trend"] + 0.45 * data["competitor_price"] + 4 * data["hotel_quality"] + rng.normal(0, 5, size=len(data))
    return data

def generate_true_revenue(data, seed=100, true_price_effect=500, revenue_noise_sd=2750):
    rng = np.random.default_rng(seed)
    result = data.copy()
    result["true_revenue"] = 5000 + true_price_effect * result["price"] + 1200 * result["seasonality"] + 900 * result["market_trend"] + 200 * result["competitor_price"] + 1000 * result["hotel_quality"] + rng.normal(0, revenue_noise_sd, size=len(result))
    return result

def inject_fraud(data, fraud_magnitude, seed=190, intercept=-3.35):
    rng = np.random.default_rng(seed)
    result = data.copy()
    season_z = (result["seasonality"] - result["seasonality"].mean()) / result["seasonality"].std()
    trend_z = (result["market_trend"] - result["market_trend"].mean()) / result["market_trend"].std()
    risk_index = 1.2 * season_z + 0.8 * trend_z
    
    result["fraud_probability"] = 1 / (1 + np.exp(-(np.clip(intercept + risk_index, -500, 500))))
    result["fraud"] = rng.binomial(n=1, p=result["fraud_probability"], size=len(result))
    result["observed_revenue"] = result["true_revenue"].copy()
    
    fraud_mask = result["fraud"] == 1
    if fraud_mask.sum() > 0:
        fraud_types = rng.choice(["suppression", "inflation"], size=int(fraud_mask.sum()), p=[0.60, 0.40])
        magnitudes = rng.uniform(low=fraud_magnitude * 0.60, high=fraud_magnitude * 1.40, size=int(fraud_mask.sum()))
        result.loc[fraud_mask, "fraud_type"] = fraud_types
        result.loc[fraud_mask, "realized_magnitude"] = magnitudes
        
        supp_mask = fraud_mask & (result["fraud_type"] == "suppression")
        inf_mask = fraud_mask & (result["fraud_type"] == "inflation")
        result.loc[supp_mask, "observed_revenue"] *= (1 - result.loc[supp_mask, "realized_magnitude"])
        result.loc[inf_mask, "observed_revenue"] *= (1 + result.loc[inf_mask, "realized_magnitude"])
    return result

def estimate_causal_recovery(data, true_price_effect=500):
    formulas = {
        "Naive model": "true_revenue ~ price",
        "Adjusted controls": "true_revenue ~ price + seasonality + market_trend + competitor_price",
        "Hotel fixed effects": "true_revenue ~ price + seasonality + market_trend + competitor_price + C(hotel_id)"
    }
    rows = []
    for model_name, formula in formulas.items():
        model = smf.ols(formula, data=data).fit()
        estimate = model.params["price"]
        rows.append({"model": model_name, "estimated_price_effect": estimate, "bias": estimate - true_price_effect, "relative_bias": (estimate - true_price_effect) / true_price_effect * 100})
    return pd.DataFrame(rows)

def split_train_test(data, train_ratio=0.70):
    split_day = int(data["day"].max() * train_ratio)
    return data[data["day"] <= split_day].copy(), data[data["day"] > split_day].copy(), split_day

def add_expected_revenue(train_data, test_data):
    model = smf.ols("observed_revenue ~ price + seasonality + market_trend + competitor_price + C(hotel_id)", data=train_data).fit()
    result = test_data.copy()
    result["expected_revenue"] = model.predict(result)
    result["residual"] = result["observed_revenue"] - result["expected_revenue"]
    return model, result

def standardize_residuals(data):
    result = data.copy()
    center = np.median(result["residual"])
    scale = 1.4826 * np.median(np.abs(result["residual"] - center))
    scale = scale if scale > 0 else 1.0
    result["z_residual"] = (result["residual"] - center) / scale
    result["abs_z_residual"] = result["z_residual"].abs()
    return result, center, scale

def add_bayesian_score(data, prior=0.05):
    result = data.copy()
    z = result["z_residual"]
    likelihood_no_fraud = np.exp(-0.5 * z**2) / np.sqrt(2 * np.pi)
    likelihood_fraud = 0.60 * (np.exp(-0.5 * ((z + 1.5)/1.8)**2)/(1.8*np.sqrt(2*np.pi))) + 0.40 * (np.exp(-0.5 * ((z - 1.5)/1.8)**2)/(1.8*np.sqrt(2*np.pi)))
    num = likelihood_fraud * prior
    result["posterior_fraud_probability"] = num / (num + likelihood_no_fraud * (1 - prior) + 1e-12)
    return result

def add_action_layer(data):
    result = data.copy()
    conditions = [
        result["posterior_fraud_probability"] < 0.10,
        (result["posterior_fraud_probability"] >= 0.10) & (result["posterior_fraud_probability"] < 0.30),
        (result["posterior_fraud_probability"] >= 0.30) & (result["posterior_fraud_probability"] < 0.50),
        result["posterior_fraud_probability"] >= 0.50,
    ]
    labels = ["Routine documentation", "Screening review", "Targeted testing", "High-level escalation"]
    result["audit_action"] = np.select(conditions, labels, default=labels[0])
    return result

def get_action_summary(scored):
    order = ["Routine documentation", "Screening review", "Targeted testing", "High-level escalation"]
    table = scored.groupby("audit_action").agg(
        records=("audit_action", "size"),
        mean_probability=("posterior_fraud_probability", "mean"),
        max_probability=("posterior_fraud_probability", "max")
    ).reindex(order).fillna(0).reset_index()
    return table

def evaluate_thresholds(data):
    rows = []
    for threshold in [0.10, 0.20, 0.30, 0.50]:
        y_pred = data["posterior_fraud_probability"] >= threshold
        rows.append({"threshold": threshold, "precision": precision_score(data["fraud"], y_pred, zero_division=0), "recall": recall_score(data["fraud"], y_pred, zero_division=0), "f1_score": f1_score(data["fraud"], y_pred, zero_division=0), "number_flagged": int(y_pred.sum())})
    return pd.DataFrame(rows)

def run_audit_pipeline(base_data, fraud_magnitude=0.10, train_ratio=0.70):
    scenario_data = inject_fraud(base_data, fraud_magnitude)
    causal_res = estimate_causal_recovery(base_data)
    
    train_data, test_data, _ = split_train_test(scenario_data, train_ratio)
    _, test_data = add_expected_revenue(train_data, test_data)
    test_data, _, _ = standardize_residuals(test_data)
    
    scored_data = add_bayesian_score(test_data)
    scored_data = add_action_layer(scored_data) 
    
    eval_res = evaluate_thresholds(scored_data)
    action_sum = get_action_summary(scored_data) 
    
    return causal_res, scored_data, eval_res, action_sum