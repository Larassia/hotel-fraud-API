import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)

def simulate_hotel_data(n_hotels=50, n_days=365, seed=42):
    """Create the fraud-free hotel business environment."""
    rng = np.random.default_rng(seed)

    hotel_ids = np.arange(n_hotels)
    days = np.arange(n_days)

    data = pd.MultiIndex.from_product(
        [hotel_ids, days],
        names=["hotel_id", "day"]
    ).to_frame(index=False)

    hotel_quality = rng.normal(0, 1, size=n_hotels)
    quality_map = dict(zip(hotel_ids, hotel_quality))
    data["hotel_quality"] = data["hotel_id"].map(quality_map)

    annual_season = 0.35 * np.sin(2 * np.pi * data["day"] / 365)
    weekly_season = 0.15 * np.sin(2 * np.pi * data["day"] / 7)
    season_noise = rng.normal(0, 0.05, size=len(data))
    data["seasonality"] = annual_season + weekly_season + season_noise

    trend_noise = rng.normal(0, 0.03, size=len(data))
    data["market_trend"] = (
        0.20 * np.sin(2 * np.pi * data["day"] / 180)
        + 0.001 * data["day"]
        + trend_noise
    )

    competitor_noise = rng.normal(0, 5, size=len(data))
    data["competitor_price"] = (
        100
        + 12 * data["seasonality"]
        + 8 * data["market_trend"]
        + 5 * data["hotel_quality"]
        + competitor_noise
    )

    price_noise = rng.normal(0, 5, size=len(data))
    data["price"] = (
        90
        + 10 * data["seasonality"]
        + 6 * data["market_trend"]
        + 0.45 * data["competitor_price"]
        + 4 * data["hotel_quality"]
        + price_noise
    )
    return data

def generate_true_revenue(data, seed=42, true_price_effect=500, revenue_noise_sd=2750):
    """Add fraud-free revenue to the simulated hotel environment."""
    rng = np.random.default_rng(seed)
    result = data.copy()

    revenue_noise = rng.normal(0, revenue_noise_sd, size=len(result))

    result["true_revenue"] = (
        5000
        + true_price_effect * result["price"]
        + 1200 * result["seasonality"]
        + 900 * result["market_trend"]
        + 200 * result["competitor_price"]
        + 1000 * result["hotel_quality"]
        + revenue_noise
    )
    return result

def sigmoid(x):
    """Convert any real number into a probability between 0 and 1."""
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))

def inject_fraud(data, fraud_magnitude, seed=42, intercept=-3.35):
    """Create observed revenue by injecting suppression or inflation fraud."""
    rng = np.random.default_rng(seed)
    result = data.copy()

    season_z = (result["seasonality"] - result["seasonality"].mean()) / result["seasonality"].std()
    trend_z = (result["market_trend"] - result["market_trend"].mean()) / result["market_trend"].std()

    risk_index = 1.2 * season_z + 0.8 * trend_z
    result["fraud_probability"] = sigmoid(intercept + risk_index)

    result["fraud"] = rng.binomial(n=1, p=result["fraud_probability"], size=len(result))

    result["observed_revenue"] = result["true_revenue"].copy()
    result["fraud_type"] = "none"
    result["realized_magnitude"] = 0.0

    fraud_mask = result["fraud"] == 1
    number_of_fraud_records = int(fraud_mask.sum())

    if number_of_fraud_records > 0:
        fraud_types = rng.choice(
            ["suppression", "inflation"],
            size=number_of_fraud_records,
            p=[0.60, 0.40]
        )
        realized_magnitudes = rng.uniform(
            low=fraud_magnitude * 0.60,
            high=fraud_magnitude * 1.40,
            size=number_of_fraud_records
        )

        result.loc[fraud_mask, "fraud_type"] = fraud_types
        result.loc[fraud_mask, "realized_magnitude"] = realized_magnitudes

        suppression_mask = fraud_mask & (result["fraud_type"] == "suppression")
        inflation_mask = fraud_mask & (result["fraud_type"] == "inflation")

        result.loc[suppression_mask, "observed_revenue"] = (
            result.loc[suppression_mask, "true_revenue"]
            * (1 - result.loc[suppression_mask, "realized_magnitude"])
        )
        result.loc[inflation_mask, "observed_revenue"] = (
            result.loc[inflation_mask, "true_revenue"]
            * (1 + result.loc[inflation_mask, "realized_magnitude"])
        )

    result["fraud_magnitude_target"] = fraud_magnitude
    result["revenue_difference"] = result["observed_revenue"] - result["true_revenue"]
    return result

def estimate_causal_recovery(data, true_price_effect=500):
    """Fit the three required causal models and compare their price estimates."""
    formulas = {
        "Naive model": "true_revenue ~ price",
        "Adjusted controls": "true_revenue ~ price + seasonality + market_trend + competitor_price",
        "Hotel fixed effects": "true_revenue ~ price + seasonality + market_trend + competitor_price + C(hotel_id)"
    }

    rows = []
    for model_name, formula in formulas.items():
        model = smf.ols(formula, data=data).fit()
        estimate = model.params["price"]
        ci_low, ci_high = model.conf_int().loc["price"]

        rows.append({
            "model": model_name,
            "estimated_price_effect": estimate,
            "true_price_effect": true_price_effect,
            "bias": estimate - true_price_effect,
            "relative_bias": (estimate - true_price_effect) / true_price_effect,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "ci_covers_true_effect": (ci_low <= true_price_effect <= ci_high)
        })
    return pd.DataFrame(rows)

def split_train_test(data, train_ratio=0.70):
    """Split panel data by time instead of randomly mixing past and future days."""
    split_day = int(data["day"].max() * train_ratio)
    train_data = data[data["day"] <= split_day].copy()
    test_data = data[data["day"] > split_day].copy()
    return train_data, test_data, split_day

def add_expected_revenue(train_data, test_data):
    """Estimate normal reported revenue and calculate test-period residuals."""
    model = smf.ols(
        "observed_revenue ~ price + seasonality + market_trend + competitor_price + C(hotel_id)",
        data=train_data
    ).fit()
    result = test_data.copy()
    result["expected_revenue"] = model.predict(result)
    result["residual"] = result["observed_revenue"] - result["expected_revenue"]
    return model, result

def standardize_residuals(data):
    """Convert residuals into robust z-scores."""
    result = data.copy()
    residuals = result["residual"]
    center = np.median(residuals)
    mad = np.median(np.abs(residuals - center))
    scale = 1.4826 * mad

    if scale == 0:
        scale = residuals.std()
    if scale == 0:
        scale = 1.0

    result["z_residual"] = (result["residual"] - center) / scale
    result["abs_z_residual"] = result["z_residual"].abs()
    return result, center, scale

def normal_pdf(x, mean=0, sd=1):
    """Normal probability density function."""
    return np.exp(-0.5 * ((x - mean) / sd) ** 2) / (sd * np.sqrt(2 * np.pi))

def add_bayesian_score(data, prior=0.05):
    """Convert standardized residual evidence into posterior fraud risk."""
    result = data.copy()
    z = result["z_residual"]

    likelihood_no_fraud = normal_pdf(z, mean=0, sd=1)
    likelihood_fraud_negative = normal_pdf(z, mean=-1.5, sd=1.8)
    likelihood_fraud_positive = normal_pdf(z, mean=1.5, sd=1.8)

    likelihood_fraud = 0.60 * likelihood_fraud_negative + 0.40 * likelihood_fraud_positive
    numerator = likelihood_fraud * prior
    denominator = numerator + likelihood_no_fraud * (1 - prior)

    result["posterior_fraud_probability"] = numerator / (denominator + 1e-12)
    return result

def evaluate_thresholds(data, score_column, thresholds, method_name):
    """Calculate classification metrics at several decision thresholds."""
    y_true = data["fraud"]
    scores = data[score_column]

    auc = roc_auc_score(y_true, scores)
    average_precision = average_precision_score(y_true, scores)

    rows = []
    for threshold in thresholds:
        y_pred = scores >= threshold
        rows.append({
            "method": method_name,
            "threshold": threshold,
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1_score": f1_score(y_true, y_pred, zero_division=0),
            "auc": auc,
            "average_precision": average_precision,
            "number_flagged": int(y_pred.sum()),
            "fraud_found": int(((y_pred == 1) & (y_true == 1)).sum()),
            "total_fraud": int(y_true.sum())
        })
    return pd.DataFrame(rows)

def summarize_top_review_pool(data, score_column, top_fraction=0.05):
    """Measure audit efficiency inside the highest-risk review pool."""
    number_to_review = int(np.ceil(len(data) * top_fraction))
    review_pool = data.nlargest(number_to_review, score_column)

    total_fraud = int(data["fraud"].sum())
    fraud_found = int(review_pool["fraud"].sum())
    overall_fraud_rate = data["fraud"].mean()
    review_pool_fraud_rate = review_pool["fraud"].mean()

    return pd.DataFrame([{
        "top_fraction": top_fraction,
        "records_reviewed": number_to_review,
        "fraud_captured": fraud_found / total_fraud if total_fraud > 0 else np.nan,
        "fraud_rate_in_review_pool": review_pool_fraud_rate,
        "lift": review_pool_fraud_rate / overall_fraud_rate if overall_fraud_rate > 0 else np.nan,
        "records_per_fraud_found": number_to_review / fraud_found if fraud_found > 0 else np.nan
    }])