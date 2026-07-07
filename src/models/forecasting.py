"""
src.models.forecasting
=======================
Time-series forecasting: Holt-Winters (damped trend) + ARIMA(2,1,2).
Author: Stephen Muema
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model  import ARIMA
from statsmodels.tsa.stattools    import adfuller
import warnings
warnings.filterwarnings("ignore")


class HoltWintersForecaster:
    """Additive Holt-Winters (damped trend) forecaster."""

    def __init__(self):
        self._fit = None

    def fit(self, series: np.ndarray):
        model = ExponentialSmoothing(
            series, trend="add", seasonal=None, damped_trend=True
        )
        self._fit = model.fit(optimized=True)
        return self

    def forecast(self, steps: int) -> np.ndarray:
        if self._fit is None:
            raise RuntimeError("Call .fit() first.")
        return self._fit.forecast(steps)


class ARIMAForecaster:
    """ARIMA forecaster with auto order selection and confidence intervals."""

    def __init__(self, order: tuple = None):
        self.order = order
        self._fit  = None
        self._selected_order = None

    def _select_order(self, series: np.ndarray, max_p: int = 3, max_d: int = 2, max_q: int = 3) -> tuple:
        """Select best ARIMA order by minimizing AIC."""
        best_aic = np.inf
        best_order = (1, 1, 1)
        for p in range(max_p + 1):
            for d in range(max_d + 1):
                for q in range(max_q + 1):
                    if p == 0 and q == 0:
                        continue
                    try:
                        model = ARIMA(series, order=(p, d, q))
                        result = model.fit()
                        if result.aic < best_aic:
                            best_aic = result.aic
                            best_order = (p, d, q)
                    except Exception:
                        continue
        return best_order

    def fit(self, series: np.ndarray):
        order = self.order or self._select_order(series)
        self._selected_order = order
        model = ARIMA(series, order=order)
        self._fit = model.fit()
        return self

    def forecast(self, steps: int, alpha: float = 0.1):
        """Returns (mean_forecast, lower_ci, upper_ci)."""
        if self._fit is None:
            raise RuntimeError("Call .fit() first.")
        fc_obj = self._fit.get_forecast(steps=steps)
        mean   = fc_obj.predicted_mean
        ci     = fc_obj.conf_int(alpha=alpha)
        if hasattr(ci, "iloc"):
            lo = ci.iloc[:, 0].values
            hi = ci.iloc[:, 1].values
        else:
            lo = ci[:, 0]
            hi = ci[:, 1]
        return mean, lo, hi

    def summary(self) -> dict:
        """Return model diagnostics."""
        if self._fit is None:
            return {}
        return {
            "order": self._selected_order or self.order,
            "aic": round(self._fit.aic, 2),
            "bic": round(self._fit.bic, 2),
        }


def evaluate_forecast(series: pd.Series, test_size: int = 4) -> dict:
    """Evaluate forecast model on held-out test set. Returns RMSE and MAPE."""
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    if len(series) < test_size + 4:
        return {"rmse": None, "mae": None, "mape": None}

    train = series.iloc[:-test_size]
    test = series.iloc[-test_size:]

    forecaster = HoltWintersForecaster()
    forecaster.fit(train.values)
    preds = forecaster.forecast(test_size)

    rmse = np.sqrt(mean_squared_error(test.values, preds))
    mae = mean_absolute_error(test.values, preds)
    mape = np.mean(np.abs((test.values - preds) / (test.values + 1e-10))) * 100

    return {
        "rmse": round(rmse, 3),
        "mae": round(mae, 3),
        "mape": round(mape, 2),
        "test_size": test_size,
    }


def adf_test(series: pd.Series) -> dict:
    """Augmented Dickey-Fuller stationarity test."""
    result = adfuller(series.dropna(), autolag="AIC")
    return {
        "statistic":  round(result[0], 4),
        "p_value":    round(result[1], 4),
        "stationary": result[1] < 0.05,
        "critical_values": result[4],
    }


def forecast_series(
    series: pd.Series,
    horizon: int = 5,
    method: str = "holt_winters"
) -> pd.DataFrame:
    """
    Convenience wrapper: forecast a pandas Series forward by `horizon` steps.
    Returns a DataFrame with columns: Year, Value, Lower_CI, Upper_CI, Is_Forecast.
    """
    clean = series.dropna()
    if len(clean) < 5:
        return pd.DataFrame()

    last_year = int(clean.index[-1]) if hasattr(clean.index, "__iter__") else len(clean) - 1
    fc_years  = list(range(last_year + 1, last_year + 1 + horizon))

    try:
        if method == "arima":
            fc = ARIMAForecaster()
            fc.fit(clean.values)
            mean, lo, hi = fc.forecast(horizon)
        else:
            fc = HoltWintersForecaster()
            fc.fit(clean.values)
            mean = fc.forecast(horizon)
            lo = hi = mean  # HW doesn't return CI in this wrapper

        hist_df = pd.DataFrame({
            "Year":        list(clean.index) if hasattr(clean.index, "__iter__") else range(len(clean)),
            "Value":       clean.values,
            "Lower_CI":    clean.values,
            "Upper_CI":    clean.values,
            "Is_Forecast": False,
        })
        fc_df = pd.DataFrame({
            "Year":        fc_years,
            "Value":       mean,
            "Lower_CI":    lo,
            "Upper_CI":    hi,
            "Is_Forecast": True,
        })
        return pd.concat([hist_df, fc_df], ignore_index=True)

    except Exception:
        return pd.DataFrame()
