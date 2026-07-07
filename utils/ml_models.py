"""
ML Models Module — delegates to src.models for core logic.
Maintains backward-compatible function signatures for page imports.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

from src.models.clustering import CountyClustering
from src.models.forecasting import HoltWintersForecaster, ARIMAForecaster
from src.models.regression import train_poverty_model, train_youth_model

CLUSTER_NAMES = {
    0: "\U0001f33f Emerging (Low Poverty)",
    1: "\U0001f4c8 Developing",
    2: "\u26a1 Transitioning",
    3: "\u26a0\ufe0f Vulnerable",
    4: "\U0001f534 Critical Need",
}


def cluster_counties(county_df: pd.DataFrame, n_clusters: int = 5) -> pd.DataFrame:
    """KMeans clustering on Kenya counties. Delegates to src.models.clustering."""
    clusterer = CountyClustering(n_clusters=n_clusters)
    return clusterer.fit_predict(county_df)


def forecast_indicator(
    series: pd.Series, years: list, forecast_years: int = 5
) -> pd.DataFrame:
    """Holt-Winters forecast. Delegates to src.models.forecasting."""
    clean = series.dropna()
    if len(clean) < 5:
        return pd.DataFrame()
    try:
        forecaster = HoltWintersForecaster()
        forecaster.fit(clean.values)
        forecast = forecaster.forecast(forecast_years)

        all_years = list(years[: len(clean)]) + list(
            range(
                max(years[: len(clean)]) + 1,
                max(years[: len(clean)]) + 1 + forecast_years,
            )
        )
        all_values = list(clean.values) + list(forecast)
        is_forecast = [False] * len(clean) + [True] * forecast_years

        return pd.DataFrame(
            {
                "Year": all_years,
                "Value": all_values,
                "Is_Forecast": is_forecast,
            }
        )
    except Exception:
        return pd.DataFrame()


def arima_forecast(
    series: pd.Series, steps: int = 5
) -> tuple:
    """ARIMA forecast with confidence intervals. Delegates to src.models.forecasting."""
    clean = series.dropna()
    if len(clean) < 8:
        return None, None
    try:
        forecaster = ARIMAForecaster()
        forecaster.fit(clean.values)
        mean, lo, hi = forecaster.forecast(steps, alpha=0.2)
        conf_int = pd.DataFrame({0: lo, 1: hi})
        return mean, conf_int
    except Exception:
        return None, None


def mobile_money_poverty_model(mm_df: pd.DataFrame) -> dict:
    """Regression: mobile money → poverty. Delegates to src.models.regression."""
    result = train_poverty_model(mm_df)
    return {
        "model_results": result["results"],
        "best_r2": result["best_r2"],
        "feature_importance": result["importance"],
        "predictions_df": result["predictions"].rename(columns={"Predicted": "Predicted_Poverty"}),
        "best_model": None,
        "features": result["features"],
    }


def youth_unemployment_model(yu_df: pd.DataFrame) -> dict:
    """Youth unemployment GBM model. Delegates to src.models.regression."""
    features = ["GDP_Growth", "University_Enrollment_K",
                "FDI_Inflows_B_USD", "Internet_Users_Pct", "Inflation_Rate"]
    target = "Youth_Unemployment_Pct"

    df = yu_df.dropna().copy()
    X = df[features]
    y = df[target]

    model = GradientBoostingRegressor(
        n_estimators=200, learning_rate=0.05, max_depth=3, random_state=42
    )
    model.fit(X, y)
    preds = model.predict(X)
    importance = dict(zip(features, np.round(model.feature_importances_ * 100, 2)))

    last_row = df[features].iloc[-1].copy()
    trend = df[features].diff().mean()
    future_rows = []
    for i in range(1, 6):
        row = last_row + trend * i
        future_rows.append(row.values)
    future_X = pd.DataFrame(future_rows, columns=features)
    future_preds = model.predict(future_X)
    future_years = list(range(df["Year"].max() + 1, df["Year"].max() + 6))

    return {
        "model": model,
        "features": features,
        "feature_importance": importance,
        "historical_preds": preds,
        "forecast_years": future_years,
        "forecast_values": future_preds,
        "df": df,
    }


def simulate_scenario(
    yu_model_result: dict,
    gdp_growth: float,
    fdi: float,
    university_k: float,
    internet_pct: float,
    inflation: float,
) -> float:
    """Scenario simulation: predict youth unemployment given parameters."""
    model = yu_model_result["model"]
    features = yu_model_result["features"]
    X = pd.DataFrame(
        [[gdp_growth, university_k, fdi, internet_pct, inflation]],
        columns=features,
    )
    return round(float(model.predict(X)[0]), 1)


def compute_regional_stats(county_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate county data by region."""
    return (
        county_df.groupby("Region")
        .agg(
            Counties=("County", "count"),
            Avg_Poverty=("Poverty_Rate", "mean"),
            Avg_Unemployment=("Unemployment_Rate", "mean"),
            Avg_Mobile=("Mobile_Penetration", "mean"),
            Avg_Electricity=("Electricity_Access", "mean"),
            Avg_HDI=("HDI_Score", "mean"),
            Total_Population=("Population_2019", "sum"),
        )
        .round(2)
        .reset_index()
    )
