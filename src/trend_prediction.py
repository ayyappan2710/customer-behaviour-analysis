"""
trend_prediction.py — Sales trend analysis + short-term revenue forecasting.

Model used: Random Forest Regressor (robust, no stationarity assumptions).
Features: month index (1, 2, 3, …), month-of-year, year.
Target : monthly total revenue.

LIMITATION NOTICE (displayed in the UI):
  This model is trained on synthetic/limited historical data.
  Predictions are indicative trend estimates only, not financial forecasts.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from src.utils import safe_empty


# ──────────────────────────────────────────────────────────────────────────────
# Aggregation helpers
# ──────────────────────────────────────────────────────────────────────────────

def daily_trend(df: pd.DataFrame) -> pd.DataFrame:
    """Daily revenue and quantity."""
    if safe_empty(df):
        return pd.DataFrame()
    day = df.copy()
    day['Date'] = day['Purchase Date'].dt.date
    return (day.groupby('Date').agg(
        Daily_Revenue=('Total Amount', 'sum'),
        Daily_Quantity=('Quantity', 'sum'),
        Daily_Orders=('Customer ID', 'count'),
    ).reset_index())


def monthly_trend(df: pd.DataFrame) -> pd.DataFrame:
    """Monthly revenue, quantity, and order count."""
    if safe_empty(df):
        return pd.DataFrame()
    m = df.copy()
    m['YearMonth'] = m['Purchase Date'].dt.to_period('M')
    agg = (m.groupby('YearMonth').agg(
        Monthly_Revenue=('Total Amount', 'sum'),
        Monthly_Quantity=('Quantity', 'sum'),
        Monthly_Orders=('Customer ID', 'count'),
    ).reset_index())
    agg['YearMonth'] = agg['YearMonth'].astype(str)
    return agg


def category_monthly_trend(df: pd.DataFrame) -> pd.DataFrame:
    """Monthly revenue broken down by Product Category."""
    if safe_empty(df):
        return pd.DataFrame()
    m = df.copy()
    m['YearMonth'] = m['Purchase Date'].dt.to_period('M').astype(str)
    return (m.groupby(['YearMonth', 'Product Category'])
              .agg(Revenue=('Total Amount', 'sum'))
              .reset_index())


# ──────────────────────────────────────────────────────────────────────────────
# Forecasting
# ──────────────────────────────────────────────────────────────────────────────

def build_forecast(df: pd.DataFrame, periods_ahead: int = 3):
    """
    Train a Random Forest on monthly revenue and predict `periods_ahead` months.

    Returns
    -------
    history_df : DataFrame of historical monthly data with predictions on train set
    forecast_df : DataFrame of future period predictions
    metrics : dict with MAE, R2
    """
    if safe_empty(df):
        return pd.DataFrame(), pd.DataFrame(), {}

    monthly = monthly_trend(df)
    if len(monthly) < 4:
        return monthly, pd.DataFrame(), {'error': 'Not enough monthly data (need ≥4 months).'}

    # Feature engineering on historical data
    monthly = monthly.sort_values('YearMonth').reset_index(drop=True)
    monthly['Month_Index'] = np.arange(1, len(monthly) + 1)
    monthly['Month_Of_Year'] = monthly['YearMonth'].str[-2:].astype(int)

    X = monthly[['Month_Index', 'Month_Of_Year']].values
    y = monthly['Monthly_Revenue'].values

    # Train / test split (last 20% for test, min 1 point)
    split = max(1, int(len(X) * 0.8))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    # Metrics on test set (only if we have test data)
    metrics = {}
    if len(X_test) > 0:
        y_pred_test = model.predict(X_test)
        metrics['MAE'] = round(mean_absolute_error(y_test, y_pred_test), 2)
        metrics['R2'] = round(r2_score(y_test, y_pred_test), 4)
    else:
        metrics['note'] = 'Insufficient data for test split.'

    # Predictions on full history (for the chart)
    monthly['Predicted_Revenue'] = model.predict(X)

    # Future predictions
    last_idx = monthly['Month_Index'].max()
    last_ym = pd.Period(monthly['YearMonth'].iloc[-1], freq='M')

    future_rows = []
    for i in range(1, periods_ahead + 1):
        future_period = last_ym + i
        month_idx = last_idx + i
        month_of_year = future_period.month
        pred = model.predict([[month_idx, month_of_year]])[0]
        future_rows.append({
            'YearMonth': str(future_period),
            'Month_Index': month_idx,
            'Month_Of_Year': month_of_year,
            'Predicted_Revenue': max(0, round(pred, 2)),
        })

    forecast_df = pd.DataFrame(future_rows)

    return monthly, forecast_df, metrics
