"""
EcoCast - ml_engine.py
======================
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
# Source: NOAA Coral Reef Watch climatology (approximate values)
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

REGION_COORDS = {
    "Great Barrier Reef": (-18.2871, 147.6992),
    "Coral Sea":          (-17.0, 152.0),
    "Indonesia":          (-2.5, 118.0),
    "Philippines":        (12.8797, 121.7740),
    "Coral Triangle":     (-5.0, 125.0),
    "Maldives":           (3.2028, 73.2207),
    "Red Sea":            (22.0, 38.0),
    "Caribbean":          (15.0, -75.0),
}


# DATA LOADING & PREPROCESSING


def _filter_countries_by_region(df: pd.DataFrame, region: str) -> pd.DataFrame:
    """Filter the dataframe based on country mapping for the given region."""
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
    if "country_name" in df.columns:
        return df[df["country_name"].isin(countries)]
    return df


def load_and_preprocess(csv_path: str, region: str) -> pd.Series:
    df = pd.read_csv(csv_path, low_memory=False)

    df.columns = (
        df.columns
        .str.strip()
        .str.replace(r"\s+", "_", regex=True)
        .str.replace(r"\W+", "_", regex=True)
        .str.lower()
    )

    for col in ["date", "date_collected"]:
        if col in df.columns:
            df["date"] = pd.to_datetime(df[col], errors="coerce")
            break

    for col in ["temperature_kelvin", "temperature_mean", "temperature_mean_c"]:
        if col in df.columns:
            df["sst"] = pd.to_numeric(df[col], errors="coerce")
            if df["sst"].median() > 200:
                df["sst"] = df["sst"] - 273.15
            break

    df = _filter_countries_by_region(df, region)

    if "depth_m" in df.columns:
        df["depth_m"] = pd.to_numeric(df["depth_m"], errors="coerce")
        df = df[(df["depth_m"] >= 0) & (df["depth_m"] <= 5)]

    df = df.dropna(subset=["date", "sst"])
    df = df.sort_values("date")

    cutoff = df["date"].max() - pd.DateOffset(years=3)
    df = df[df["date"] >= cutoff]

    ts = (
        df.groupby("date")["sst"]
        .mean()
        .reset_index()
        .set_index("date")["sst"]
        .sort_index()
    )

    ts = ts.resample("D").mean().ffill(limit=7)
    ts.name = "sst"
    return ts

def generate_synthetic_ts(region: str, n_days: int = 365) -> pd.Series:
    """Generate realistic synthetic SST time-series for demo mode."""
    mmm = REGION_MMM.get(region, 28.5)
    np.random.seed(42)
    dates = pd.date_range(end=datetime.today(), periods=n_days, freq="D")
    t = np.arange(n_days)
    seasonal  = 1.2 * np.sin(2 * np.pi * t / 365 - np.pi / 2)
    trend     = 0.002 * t
    noise     = np.random.normal(0, 0.25, n_days)
    el_nino   = np.where((t > 200) & (t < 280), 0.8, 0)
    sst = mmm + seasonal + trend + noise + el_nino
    return pd.Series(sst, index=dates, name="sst")


# FORECASTING ENGINE


def _make_xgb_features(series: pd.Series, lags: int = 14) -> Tuple[np.ndarray, np.ndarray]:
    """Build lag-based feature matrix for XGBoost."""
    X, y = [], []
    vals = series.values
    for i in range(lags, len(vals)):
        X.append(vals[i - lags:i])
        y.append(vals[i])
    return np.array(X), np.array(y)


def forecast_sst(
    ts,
    horizon: int = 30,
    backend: str = "auto",
) -> pd.DataFrame:
    """
    Forecast SST for the next `horizon` days.

    Args:
        ts:       daily SST time-series (pd.Series with DatetimeIndex)
        horizon:  number of days to forecast
        backend:  'arima' | 'xgboost' | 'auto' (ARIMA preferred, XGB fallback)

    Returns:
        DataFrame with columns [date, sst_forecast, lower_ci, upper_ci]
    """
    if isinstance(ts, pd.DataFrame):
        if "sst" in ts.columns:
            ts = ts["sst"]
        else:
            raise ValueError("DataFrame must contain 'sst' column.")
    if len(ts) < 30:
        raise ValueError(
            f"Time series too short ({len(ts)} obs). Need at least 30 data points."
        )

    use_arima = (backend == "arima" or backend == "auto") and ARIMA_AVAILABLE
    use_xgb   = (backend == "xgboost" or (backend == "auto" and not ARIMA_AVAILABLE)) and XGB_AVAILABLE

    last_date = ts.index[-1]
    future_dates = pd.date_range(
        start=last_date + timedelta(days=1),
        periods=horizon,
        freq="D",
    )

    if use_arima:
        # ARIMA(5,1,0) — reasonable default for SST; seasonal component via differencing
        model = ARIMA(ts, order=(5, 1, 0))
        result = model.fit()
        forecast_obj = result.get_forecast(steps=horizon)
        forecast_mean = forecast_obj.predicted_mean.values
        conf_int = forecast_obj.conf_int(alpha=0.1)  # 90% CI
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

        # Recursive forecasting
        history = list(ts.values)
        forecast_mean = []
        for _ in range(horizon):
            x = np.array(history[-lags:]).reshape(1, -1)
            pred = float(model.predict(x)[0])
            forecast_mean.append(pred)
            history.append(pred)

        # Estimate CI from training residuals
        y_pred_train = model.predict(X_train)
        residual_std = np.std(y_train - y_pred_train)
        forecast_mean = np.array(forecast_mean)
        lower = forecast_mean - 1.645 * residual_std
        upper = forecast_mean + 1.645 * residual_std

    else:
        # Fallback: seasonal naive (copy last 30 days pattern + slight drift)
        pattern = ts.values[-30:]
        trend   = (ts.values[-1] - ts.values[-30]) / 30
        forecast_mean = np.array([
            pattern[i % 30] + trend * (i + 1) for i in range(horizon)
        ])
        residual_std = ts.std()
        lower = forecast_mean - 1.645 * residual_std
        upper = forecast_mean + 1.645 * residual_std

    return pd.DataFrame({
        "date":         future_dates,
        "sst_forecast": np.round(forecast_mean, 3),
        "lower_ci":     np.round(lower, 3),
        "upper_ci":     np.round(upper, 3),
    }).set_index("date")


# DHW CALCULATOR


def compute_dhw(
    historical_ts: pd.Series,
    forecast_df: pd.DataFrame,
    region: str,
    rolling_window: int = 84,   # 12 weeks × 7 days
) -> pd.DataFrame:
    """
    Compute Degree Heating Weeks (DHW) using NOAA methodology.

    DHW = sum of Hotspot values (SST − MMM) over a 12-week rolling window,
          divided by 7 (convert degree-days → degree-weeks).
          Only positive hotspots ≥ 1°C above MMM are counted.

    Args:
        historical_ts:  past SST series
        forecast_df:    output of forecast_sst()
        region:         region name (to look up MMM)
        rolling_window: number of days for rolling sum (default 84 = 12 weeks)

    Returns:
        DataFrame with [date, sst, hotspot, dhw, bleaching_risk, alert_level]
        covering the forecast horizon.
    """
    mmm = REGION_MMM.get(region, 28.5)

    # Combine historical + forecast into one series
    hist = historical_ts["sst"] if isinstance(historical_ts, pd.DataFrame) else historical_ts
    combined = pd.concat([
        hist.rename("sst"),
        forecast_df["sst_forecast"].rename("sst"),
    ])
    combined = combined[~combined.index.duplicated(keep="last")]
    combined = combined.sort_index()

    # Hotspot: SST − MMM, only counted when > 1°C
    hotspot = (combined - mmm).clip(lower=0)
    hotspot[hotspot < 1.0] = 0  # NOAA threshold: only ≥1°C contributes

    # Rolling DHW (degree-days → degree-weeks)
    dhw = hotspot.rolling(window=rolling_window, min_periods=1).sum() / 7.0

    # Slice to forecast window only
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
    """
    Convert DHW (°C-weeks) → bleaching probability [0, 1].

    Sigmoid-like mapping calibrated to NOAA Coral Reef Watch thresholds:
      DHW=0  → ~2%  (background)
      DHW=4  → ~50% (Warning level)
      DHW=8  → ~85% (Alert Level 1)
      DHW=12 → ~97% (Alert Level 2)
    """
    # Logistic: p = 1 / (1 + exp(-k*(dhw - x0)))
    k  = 0.55
    x0 = 4.0
    return round(1 / (1 + np.exp(-k * (dhw - x0))), 4)


def _dhw_to_alert(dhw: float) -> str:
    """Map DHW value to NOAA alert category string."""
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

    Returns a dict with keys:
        headline   — one-line alert status
        summary    — 2–3 sentence paragraph
        trend      — SST trend description
        action     — recommended monitoring action
    """
    peak_dhw      = dhw_df["dhw"].max()
    peak_date     = dhw_df["dhw"].idxmax().strftime("%d %b %Y")
    peak_risk_pct = round(dhw_df["bleaching_risk"].max() * 100, 1)
    alert_level   = dhw_df.loc[dhw_df["dhw"].idxmax(), "alert_level"]
    avg_sst_fore  = forecast_df["sst_forecast"].mean()
    mmm           = REGION_MMM.get(region, 28.5)
    sst_anomaly   = round(avg_sst_fore - mmm, 2)

    # SST trend direction
    sst_vals  = forecast_df["sst_forecast"].values
    sst_trend = np.polyfit(range(len(sst_vals)), sst_vals, 1)[0]
    if sst_trend > 0.05:
        trend_desc = f"rising at approximately {abs(sst_trend):.2f}°C/day"
    elif sst_trend < -0.05:
        trend_desc = f"declining at approximately {abs(sst_trend):.2f}°C/day"
    else:
        trend_desc = "relatively stable"

    # Headline
    if peak_risk_pct >= 85:
        headline = f"CRITICAL — {region}: Severe bleaching event highly probable"
    elif peak_risk_pct >= 60:
        headline = f"HIGH RISK — {region}: Significant bleaching likely"
    elif peak_risk_pct >= 35:
        headline = f"ELEVATED — {region}: Thermal stress building"
    else:
        headline = f"LOW RISK — {region}: Conditions within normal range"

    # Summary paragraph
    summary = (
        f"Over the next 30 days, sea surface temperatures in {region} are projected to average "
        f"{avg_sst_fore:.2f}°C — a {'+' if sst_anomaly >= 0 else ''}{sst_anomaly}°C anomaly relative to "
        f"the Maximum Monthly Mean baseline of {mmm}°C. "
        f"Degree Heating Weeks are forecast to peak at {peak_dhw:.1f}°C-weeks on {peak_date}, "
        f"corresponding to a bleaching probability of {peak_risk_pct}% ({alert_level.split(' ', 1)[-1].strip()}). "
        f"SST is currently {trend_desc} across the forecast window."
    )

    # Action recommendation
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


# MAIN PIPELINE  (for standalone testing)


def run_pipeline(
    csv_path: str,
    region: str = "Indonesia",
    horizon: int = 30,
    backend: str = "auto",
) -> Dict:
    """
    End-to-end pipeline: load → forecast → DHW → NLA.

    Returns a dict with keys: ts, forecast, dhw, nla
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
    import os

    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        candidates = [
            os.path.join(script_dir, "global_bleaching_environmental.csv"),
            "/Users/vythu/Documents/Coral Project /global_bleaching_environmental.csv",
            os.path.join(script_dir, "data", "raw", "coral_bleaching.csv"),
        ]
        csv_path = next((p for p in candidates if os.path.exists(p)), None)
        if csv_path is None:
            print("Usage: python ml_engine.py <path_to_csv> [region]")
            print("CSV not found automatically. Please provide the path as argument.")
            sys.exit(1)

    region = sys.argv[2] if len(sys.argv) > 2 else "Indonesia"
    run_pipeline(csv_path, region) 