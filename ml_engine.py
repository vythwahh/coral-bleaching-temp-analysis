"""
EcoCast - ml_engine.py
 
Coral Bleaching Prediction Engine
 
Pipeline:
  1. Load & preprocess historical SST + bleaching survey data
  2. Forecast SST for next 30 days (ARIMA preferred, XGBoost fallback)
  3. Compute Degree Heating Weeks (DHW) from forecasted SST
  4. Convert DHW → bleaching risk probability
  5. Generate NLA (Natural Language Analysis) summary
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Tuple, Dict, Optional

# Try importing model backends 
try:
    from statsmodels.tsa.arima.model import ARIMA
    ARIMA_AVAILABLE = True
except ImportError:
    ARIMA_AVAILABLE = False

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False


 
# CONSTANTS
 

# Maximum Monthly Mean (MMM) baseline per region — °C
REGION_MMM = {
    "Great Barrier Reef":   28.5,
    "Coral Sea":            28.0,
    "Indonesia":            29.2,
    "Philippines":          29.0,
    "Red Sea":              30.1,
    "Caribbean":            29.5,
    "Maldives":             29.8,
    "Coral Triangle":       29.3,
}

# DHW thresholds → bleaching category
DHW_THRESHOLDS = {
    "Watch":    2.0,   
    "Warning":  4.0,   
    "Alert 1":  8.0,   
    "Alert 2": 12.0,   
}


 
# DATA LOADING & PREPROCESSING
 

def load_and_preprocess(csv_path: str, region: str) -> pd.Series:
    """
    Load the Donner et al. coral survey CSV and return a clean
    daily/weekly SST time-series for the given region.
    """
    df = pd.read_csv("/Users/vythu/Documents/Coral Project /global_bleaching_environmental.csv")

    # Standardise column names
    df.columns = (
        df.columns
        .str.strip()
        .str.replace(r"\s+", "_", regex=True)
        .str.replace(r"\W+", "_", regex=True)
        .str.lower()
    )

    # Map region label → country filter
    country_map = {
        "Indonesia":  ["Indonesia"],
        "Philippines": ["Philippines"],
        "Great Barrier Reef": ["Australia"],
        "Coral Sea":  ["Australia"],
        "Red Sea":    ["Egypt", "Saudi Arabia", "Yemen"],
        "Caribbean":  ["Mexico", "Belize", "Cuba", "Jamaica"],
        "Maldives":   ["Maldives"],
        "Coral Triangle": ["Indonesia", "Philippines", "Papua New Guinea",
                           "Solomon Islands", "Timor-Leste", "Malaysia"],
    }
    countries = country_map.get(region, [region])

    # Parse date & numeric columns
    for col in ["date", "date_collected"]:
        if col in df.columns:
            df["date"] = pd.to_datetime(df[col], errors="coerce")
            break

    for col in ["temperature_mean", "temperature_mean_c"]:
        if col in df.columns:
            df["sst"] = pd.to_numeric(df[col], errors="coerce")
            break

    if "country_name" in df.columns:
        df = df[df["country_name"].isin(countries)]

    if "depth_m" in df.columns:
        df["depth_m"] = pd.to_numeric(df["depth_m"], errors="coerce")
        df = df[(df["depth_m"] >= 0) & (df["depth_m"] <= 5)]

    df = df.dropna(subset=["date", "sst"])
    df = df.sort_values("date")

    # Aggregate to daily mean SST
    ts = (
        df.groupby("date")["sst"]
        .mean()
        .reset_index()
        .rename(columns={"date": "date", "sst": "sst"})
    )
    ts = ts.set_index("date").sort_index()

    # Fill gaps with forward-fill (max 7 days)
    ts = ts.resample("D").mean().ffill(limit=7)
    
     
    ts["sst"] = np.clip(ts["sst"], 15.0, 35.0)

    return ts["sst"]


 
# FORECASTING ENGINE
 

def _make_xgb_features(series: pd.Series, lags: int = 14) -> Tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    vals = series.values
    for i in range(lags, len(vals)):
        X.append(vals[i - lags:i])
        y.append(vals[i])
    return np.array(X), np.array(y)


def forecast_sst(
    ts: pd.Series,
    horizon: int = 30,
    backend: str = "auto",
) -> pd.DataFrame:
    """
    Forecast SST for the next `horizon` days with safety limits.
    """
    if len(ts) < 30:
        raise ValueError(f"Time series too short ({len(ts)} obs). Need at least 30 data points.")

    use_arima = (backend == "arima" or backend == "auto") and ARIMA_AVAILABLE
    use_xgb   = (backend == "xgboost" or (backend == "auto" and not ARIMA_AVAILABLE)) and XGB_AVAILABLE

    last_date = ts.index[-1]
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=horizon, freq="D")

    if use_arima:
        model = ARIMA(ts, order=(5, 1, 0))
        result = model.fit()
        forecast_obj = result.get_forecast(steps=horizon)
        forecast_mean = forecast_obj.predicted_mean.values
        conf_int = forecast_obj.conf_int(alpha=0.1)
        lower = conf_int.iloc[:, 0].values
        upper = conf_int.iloc[:, 1].values

    elif use_xgb:
        lags = min(14, len(ts) // 2)
        X_train, y_train = _make_xgb_features(ts, lags=lags)
        model = xgb.XGBRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            objective="reg:squarederror", verbosity=0
        )
        model.fit(X_train, y_train)

        history = list(ts.values)
        forecast_mean = []
        for _ in range(horizon):
            x = np.array(history[-lags:]).reshape(1, -1)
            pred = float(model.predict(x)[0])
            forecast_mean.append(pred)
            history.append(pred)

        y_pred_train = model.predict(X_train)
        residual_std = np.std(y_train - y_pred_train)
        forecast_mean = np.array(forecast_mean)
        lower = forecast_mean - 1.645 * residual_std
        upper = forecast_mean + 1.645 * residual_std

    else:
        pattern = ts.values[-30:]
        trend   = (ts.values[-1] - ts.values[-30]) / 30
        forecast_mean = np.array([pattern[i % 30] + trend * (i + 1) for i in range(horizon)])
        residual_std = ts.std()
        lower = forecast_mean - 1.645 * residual_std
        upper = forecast_mean + 1.645 * residual_std

     
    forecast_mean = np.clip(forecast_mean, 15.0, 31.5)
    lower = np.clip(lower, 15.0, 31.5)
    upper = np.clip(upper, 15.0, 31.5)

    return pd.DataFrame({
        "date":         future_dates,
        "sst_forecast": np.round(forecast_mean, 3),
        "lower_ci":     np.round(lower, 3),
        "upper_ci":     np.round(upper, 3),
    }).set_index("date")


 
 
def compute_dhw(
    historical_ts: pd.Series,
    forecast_df: pd.DataFrame,
    region: str,
    rolling_window: int = 84,   # 12 weeks × 7 days
) -> pd.DataFrame:
    """
    Compute Degree Heating Weeks (DHW) using NOAA methodology.
    """
    mmm = REGION_MMM.get(region, 28.5)

    # Chỉ bốc đúng 84 ngày lịch sử cuối cùng
    recent_history = historical_ts.tail(rolling_window)

    combined = pd.concat([
        recent_history.rename("sst"),
        forecast_df["sst_forecast"].rename("sst"),
    ])
    combined = combined[~combined.index.duplicated(keep="last")]
    combined = combined.sort_index()

     
    combined = np.clip(combined, 15.0, 31.5)

     
    hotspot = (combined - mmm).clip(lower=0)
    hotspot[hotspot < 1.0] = 0  

     
    dhw = hotspot.rolling(window=rolling_window, min_periods=1).sum() / 7.0

     
    forecast_dates = forecast_df.index
    result = pd.DataFrame({
        "sst":       combined.loc[forecast_dates].values,
        "hotspot":   hotspot.loc[forecast_dates].values,
        "dhw":       dhw.loc[forecast_dates].values,
    }, index=forecast_dates)
    result.index.name = "date"

    result["bleaching_risk"] = result["dhw"].apply(_dhw_to_risk)
    result["alert_level"]    = result["dhw"].apply(_dhw_to_alert)

    return result.round(3)


def _dhw_to_risk(dhw: float) -> float:
    k  = 0.55
    x0 = 4.0
    return round(1 / (1 + np.exp(-k * (dhw - x0))), 4)


def _dhw_to_alert(dhw: float) -> str:
    if dhw >= DHW_THRESHOLDS["Alert 2"]:
        return "Alert Level 2"
    elif dhw >= DHW_THRESHOLDS["Alert 1"]:
        return "Alert Level 1"
    elif dhw >= DHW_THRESHOLDS["Warning"]:
        return "Warning"
    elif dhw >= DHW_THRESHOLDS["Watch"]:
        return "Watch"
    else:
        return "No Stress"


 
# NATURAL LANGUAGE ANALYSIS (NLA)
 

def generate_nla(
    region: str,
    dhw_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
) -> Dict[str, str]:
    """
    Generate a Natural Language Analysis summary of the forecast results.
    """
    peak_dhw      = dhw_df["dhw"].max()
    peak_date     = dhw_df["dhw"].idxmax().strftime("%d %b %Y")
    peak_risk_pct = round(dhw_df["bleaching_risk"].max() * 100, 1)
    alert_level   = dhw_df.loc[dhw_df["dhw"].idxmax(), "alert_level"]
    avg_sst_fore  = forecast_df["sst_forecast"].mean()
    mmm           = REGION_MMM.get(region, 28.5)
    sst_anomaly   = round(avg_sst_fore - mmm, 2)

    sst_vals  = forecast_df["sst_forecast"].values
    sst_trend = np.polyfit(range(len(sst_vals)), sst_vals, 1)[0]
    if sst_trend > 0.02:
        trend_desc = f"rising at approximately {abs(sst_trend):.2f}°C/day"
    elif sst_trend < -0.02:
        trend_desc = f"declining at approximately {abs(sst_trend):.2f}°C/day"
    else:
        trend_desc = "relatively stable"

    if peak_risk_pct >= 85:
        headline = f" CRITICAL — {region}: Severe bleaching event highly probable"
    elif peak_risk_pct >= 60:
        headline = f" HIGH RISK — {region}: Significant bleaching likely"
    elif peak_risk_pct >= 35:
        headline = f" ELEVATED — {region}: Thermal stress building"
    else:
        headline = f" LOW RISK — {region}: Conditions within normal range"

    summary = (
        f"Over the next 30 days, sea surface temperatures in {region} are projected to average "
        f"{avg_sst_fore:.2f}°C — a {'+' if sst_anomaly >= 0 else ''}{sst_anomaly}°C anomaly relative to "
        f"the Maximum Monthly Mean baseline of {mmm}°C. "
        f"Degree Heating Weeks are forecast to peak at {peak_dhw:.1f}°C-weeks on {peak_date}, "
        f"corresponding to a bleaching probability of {peak_risk_pct}% ({alert_level.replace('Alert Level 2','').replace('Alert Level 1','').replace('Warning','').replace('Watch','').replace('No Stress','').strip()}). "
        f"SST is currently {trend_desc} across the forecast window."
    )

    if peak_risk_pct >= 85:
        action = (
            "Immediate field survey recommended. Notify reef monitoring networks "
            "(GBRMPA, NOAA CRW) and activate emergency response protocols."
        )
    elif peak_risk_pct >= 60:
        action = (
            "Increase monitoring frequency to weekly surveys. "
            "Pre-position assessment teams and prepare rapid-response data collection."
        )
    elif peak_risk_pct >= 35:
        action = (
            "Continue standard bi-weekly monitoring. "
            "Flag region for elevated attention in next monthly report."
        )
    else:
        action = "Maintain routine monitoring schedule. No immediate intervention required."

    return {
        "headline": headline,
        "summary":  summary,
        "trend":    f"SST trend: {trend_desc}",
        "action":   f"Recommended action: {action}",
    }


 
# MAIN PIPELINE
 

def run_pipeline(
    csv_path: str,
    region: str = "Indonesia",
    horizon: int = 30,
    backend: str = "auto",
) -> Dict:
    """
    End-to-end pipeline: load → forecast → DHW → NLA.
    """
    print(f"[EcoCast] Loading data for region: {region}")
    ts = load_and_preprocess(csv_path, region)
    print(f"[EcoCast] {len(ts)} daily records loaded. Date range: {ts.index[0].date()} → {ts.index[-1].date()}")

    print(f"[EcoCast] Forecasting SST {horizon} days ahead using backend='{backend}'...")
    forecast_df = forecast_sst(ts, horizon=horizon, backend=backend)

    print("[EcoCast] Computing Degree Heating Weeks (DHW)...")
    dhw_df = compute_dhw(ts, forecast_df, region)

    print("[EcoCast] Generating Natural Language Analysis...")
    nla = generate_nla(region, dhw_df, forecast_df)

    print("\n" + "="*60)
    print(nla["headline"])
    print("="*60)
    print(nla["summary"])
    print(nla["trend"])
    print(nla["action"])
    print("="*60)

    return {
        "ts":       ts,
        "forecast": forecast_df,
        "dhw":      dhw_df,
        "nla":      nla,
    }


if __name__ == "__main__":
    import sys
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/coral_bleaching.csv"
    region   = sys.argv[2] if len(sys.argv) > 2 else "Indonesia"
    run_pipeline(csv_path, region)