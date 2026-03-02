"""
Kerala Weather Forecast — Fully Automated (No Colab)
Runs daily via GitHub Actions cron job.
Fetches historical data automatically from Open-Meteo Archive API.
Saves outputs as CSV + PNG files (picked up by GitHub Actions artifacts).
"""

import json
import warnings
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import optuna
import requests
import requests_cache
from retry_requests import retry

from prophet import Prophet
from prophet.serialize import model_to_json
from xgboost import XGBRegressor, XGBClassifier
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
pd.set_option("display.float_format", "{:.2f}".format)
tf.random.set_seed(42)
np.random.seed(42)

# ─────────────────────────────────────────────────────────────
#  CONFIG  — edit these if needed
# ─────────────────────────────────────────────────────────────
LATITUDE       = 10.8505
LONGITUDE      = 76.2711
LOCATION_NAME  = "Kerala"
HISTORY_START  = "2015-01-01"          # how far back to pull history
FORECAST_DAYS  = 365                   # days to forecast ahead
LOOKBACK       = 60                    # LSTM lookback window
N_TRIALS       = 15                    # Optuna trials (keep low for CI speed)
LSTM_EPOCHS    = 80                    # max LSTM epochs
ALL_VARS       = ["temp", "tempmax", "tempmin", "humidity", "pressure", "precip"]

print("=" * 60)
print(f"  Kerala Weather Forecast Pipeline")
print(f"  Run date : {pd.Timestamp.today().date()}")
print("=" * 60)

# ─────────────────────────────────────────────────────────────
#  STEP 1 — FETCH HISTORICAL DATA FROM OPEN-METEO ARCHIVE API
# ─────────────────────────────────────────────────────────────
print("\n📥 Fetching historical weather data from Open-Meteo Archive API...")

end_date   = pd.Timestamp.today().strftime("%Y-%m-%d")
start_date = HISTORY_START

# Daily variables
daily_url    = "https://archive-api.open-meteo.com/v1/archive"
daily_params = {
    "latitude"   : LATITUDE,
    "longitude"  : LONGITUDE,
    "start_date" : start_date,
    "end_date"   : end_date,
    "daily"      : [
        "temperature_2m_max",
        "temperature_2m_min",
        "precipitation_sum",
    ],
    "timezone"   : "Asia/Kolkata",
    "format"     : "json",
}
daily_resp = requests.get(daily_url, params=daily_params, timeout=60).json()
if daily_resp.get("error"):
    raise RuntimeError(f"Open-Meteo daily error: {daily_resp.get('reason')}")

# Hourly variables (humidity + pressure) → aggregate to daily
hourly_params = {
    "latitude"   : LATITUDE,
    "longitude"  : LONGITUDE,
    "start_date" : start_date,
    "end_date"   : end_date,
    "hourly"     : ["relative_humidity_2m", "surface_pressure"],
    "timezone"   : "Asia/Kolkata",
    "format"     : "json",
}
hourly_resp = requests.get(daily_url, params=hourly_params, timeout=120).json()
if hourly_resp.get("error"):
    raise RuntimeError(f"Open-Meteo hourly error: {hourly_resp.get('reason')}")

# Build daily DataFrame
d        = daily_resp["daily"]
dates    = pd.to_datetime(d["time"])
tempmax  = np.array(d["temperature_2m_max"],  dtype=float)
tempmin  = np.array(d["temperature_2m_min"],  dtype=float)
temp     = (tempmax + tempmin) / 2
precip   = np.clip(np.array(d["precipitation_sum"], dtype=float), 0, None)

hourly_df = pd.DataFrame({
    "time"     : pd.to_datetime(hourly_resp["hourly"]["time"]),
    "humidity" : np.array(hourly_resp["hourly"]["relative_humidity_2m"], dtype=float),
    "pressure" : np.array(hourly_resp["hourly"]["surface_pressure"],     dtype=float),
}).set_index("time")
daily_hourly = hourly_df.resample("D").mean()
daily_hourly.index = daily_hourly.index.normalize()

df = pd.DataFrame({
    "temp"    : temp,
    "tempmax" : tempmax,
    "tempmin" : tempmin,
    "precip"  : precip,
    "humidity": daily_hourly["humidity"].reindex(dates).values,
    "pressure": daily_hourly["pressure"].reindex(dates).values,
}, index=dates)
df.index.name = "date"
df.sort_index(inplace=True)
df = df.asfreq("D").interpolate(method="time")
df["humidity"] = df["humidity"].clip(0, 100)
df["precip"]   = df["precip"].clip(lower=0)

print(f"   ✅ Historical data fetched!")
print(f"   Date range : {df.index.min().date()} → {df.index.max().date()}")
print(f"   Total days : {len(df)}")

# ─────────────────────────────────────────────────────────────
#  STEP 2 — FETCH NWP FORECAST (next 16 days from Open-Meteo)
# ─────────────────────────────────────────────────────────────
print("\n🌐 Fetching NWP forecast data...")

NWP_AVAILABLE = False
nwp_df        = pd.DataFrame()

try:
    nwp_daily_params = {
        "latitude"      : LATITUDE,
        "longitude"     : LONGITUDE,
        "daily"         : ["temperature_2m_max", "temperature_2m_min",
                           "precipitation_sum", "rain_sum"],
        "timezone"      : "Asia/Kolkata",
        "forecast_days" : 16,
        "format"        : "json",
    }
    nwp_daily_resp = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params=nwp_daily_params, timeout=30
    ).json()
    if nwp_daily_resp.get("error"):
        raise ValueError(nwp_daily_resp.get("reason"))

    nwp_hourly_params = {
        "latitude"      : LATITUDE,
        "longitude"     : LONGITUDE,
        "hourly"        : ["relative_humidity_2m", "surface_pressure"],
        "timezone"      : "Asia/Kolkata",
        "forecast_days" : 16,
        "format"        : "json",
    }
    nwp_hourly_resp = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params=nwp_hourly_params, timeout=30
    ).json()
    if nwp_hourly_resp.get("error"):
        raise ValueError(nwp_hourly_resp.get("reason"))

    nd          = nwp_daily_resp["daily"]
    nwp_dates   = pd.to_datetime(nd["time"])
    nwp_tempmax = np.array(nd["temperature_2m_max"],  dtype=float)
    nwp_tempmin = np.array(nd["temperature_2m_min"],  dtype=float)
    nwp_temp    = (nwp_tempmax + nwp_tempmin) / 2
    nwp_precip  = np.clip(np.array(nd["precipitation_sum"], dtype=float), 0, None)

    nh_df = pd.DataFrame({
        "time"     : pd.to_datetime(nwp_hourly_resp["hourly"]["time"]),
        "humidity" : np.array(nwp_hourly_resp["hourly"]["relative_humidity_2m"], dtype=float),
        "pressure" : np.array(nwp_hourly_resp["hourly"]["surface_pressure"],     dtype=float),
    }).set_index("time")
    nh_daily = nh_df.resample("D").mean()
    nh_daily.index = nh_daily.index.normalize()

    nwp_df = pd.DataFrame({
        "nwp_temp"    : nwp_temp,
        "nwp_tempmax" : nwp_tempmax,
        "nwp_tempmin" : nwp_tempmin,
        "nwp_humidity": np.clip(nh_daily["humidity"].reindex(nwp_dates).values, 0, 100),
        "nwp_pressure": nh_daily["pressure"].reindex(nwp_dates).values,
        "nwp_precip"  : nwp_precip,
    }, index=nwp_dates)
    nwp_df.index.name = "date"
    nwp_df.index      = nwp_df.index.normalize()
    NWP_AVAILABLE     = True
    print(f"   ✅ NWP data ready ({len(nwp_df)} days)")

except Exception as e:
    print(f"   ⚠️  NWP fetch failed: {e}. Continuing without NWP.")

# ─────────────────────────────────────────────────────────────
#  STEP 3 — FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────
print("\n🔧 Building features...")

def build_features(df, nwp_df=None):
    df    = df.copy()
    month = df.index.month
    df["day_of_year"]  = df.index.dayofyear
    df["month"]        = month
    df["sin_doy"]      = np.sin(2 * np.pi * df["day_of_year"] / 365)
    df["cos_doy"]      = np.cos(2 * np.pi * df["day_of_year"] / 365)
    df["sin_month"]    = np.sin(2 * np.pi * month / 12)
    df["cos_month"]    = np.cos(2 * np.pi * month / 12)

    for col in ALL_VARS:
        for lag in [1, 7, 14, 30, 365]:
            df[f"{col}_lag_{lag}"] = df[col].shift(lag)
        for window in [7, 30]:
            df[f"{col}_roll_{window}"] = df[col].rolling(window).mean()

    df["is_sw_monsoon"]  = month.isin([6, 7, 8, 9]).astype(float)
    df["is_ne_monsoon"]  = month.isin([10, 11, 12]).astype(float)
    df["is_dry_season"]  = month.isin([1, 2, 3, 4, 5]).astype(float)
    df["is_pre_monsoon"] = month.isin([3, 4, 5]).astype(float)
    df["precip_roll_7"]  = df["precip"].rolling(7).mean()
    df["precip_roll_30"] = df["precip"].rolling(30).mean()
    df["rain_day"]       = (df["precip"] > 0.1).astype(float)
    df["rain_streak"]    = df["rain_day"].rolling(7).sum()
    df["temp_range"]     = df["tempmax"] - df["tempmin"]
    df["humidity_x_temp"]= df["humidity"] * df["temp"] / 100

    if nwp_df is not None and len(nwp_df) > 0:
        nwp_col_map = {
            "nwp_temp"    : "temp",
            "nwp_tempmax" : "tempmax",
            "nwp_tempmin" : "tempmin",
            "nwp_humidity": "humidity",
            "nwp_pressure": "pressure",
            "nwp_precip"  : "precip",
        }
        for nwp_col, raw_col in nwp_col_map.items():
            df[nwp_col] = nwp_df[nwp_col].reindex(df.index)
            df[nwp_col] = df[nwp_col].fillna(df[raw_col].median())
    return df


df_feat = build_features(df, nwp_df if NWP_AVAILABLE else None)
df_feat.dropna(inplace=True)
print(f"   ✅ Features built. Shape: {df_feat.shape}")

FEATURE_COLS = [c for c in df_feat.columns if c not in ALL_VARS]

# ─────────────────────────────────────────────────────────────
#  STEP 4 — HISTORICAL OVERVIEW CHART
# ─────────────────────────────────────────────────────────────
print("\n📊 Saving historical overview chart...")

fig, axes = plt.subplots(6, 1, figsize=(16, 18), sharex=True)
plot_cfg = [
    ("temp",     "Average Temperature (°C)", "darkorange"),
    ("tempmax",  "Max Temperature (°C)",     "red"),
    ("tempmin",  "Min Temperature (°C)",     "steelblue"),
    ("humidity", "Humidity (%)",             "purple"),
    ("pressure", "Pressure (hPa)",           "green"),
    ("precip",   "Precipitation (mm)",       "navy"),
]
for ax, (col, title, color) in zip(axes, plot_cfg):
    df[col].plot(ax=ax, color=color, linewidth=0.9, title=title)
    ax.grid(alpha=0.3)
    ax.set_xlabel("")
plt.suptitle(f"Historical Weather Overview — {LOCATION_NAME}", fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig("historical_overview.png", dpi=150, bbox_inches="tight")
plt.close()
print("   ✅ historical_overview.png saved.")

# ─────────────────────────────────────────────────────────────
#  STEP 5 — PROPHET MODELS
# ─────────────────────────────────────────────────────────────
print("\n🔮 Training Prophet models...")

def add_monsoon_flags(pdf):
    pdf   = pdf.copy()
    month = pd.to_datetime(pdf["ds"]).dt.month
    pdf["is_sw_monsoon"] = month.isin([6, 7, 8, 9]).astype(float)
    pdf["is_ne_monsoon"] = month.isin([10, 11, 12]).astype(float)
    pdf["is_dry_season"] = month.isin([1, 2, 3, 4, 5]).astype(float)
    return pdf


def train_prophet(df, target_col, periods=365,
                  regressors=None, seasonality_mode="multiplicative"):
    print(f"   Prophet → {target_col}")
    pdf = df[[target_col]].reset_index()
    pdf.columns = ["ds", "y"]
    if regressors:
        for r in regressors:
            pdf[r] = df[r].values
    pdf = add_monsoon_flags(pdf)

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode=seasonality_mode,
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=15.0,
    )
    model.add_seasonality(name="southwest_monsoon", period=365.25,
                          fourier_order=8, condition_name="is_sw_monsoon")
    model.add_seasonality(name="northeast_monsoon",  period=365.25,
                          fourier_order=5, condition_name="is_ne_monsoon")
    model.add_seasonality(name="dry_summer_season",  period=365.25,
                          fourier_order=5, condition_name="is_dry_season")
    if regressors:
        for r in regressors:
            model.add_regressor(r)

    model.fit(pdf)
    future   = model.make_future_dataframe(periods=periods)
    if regressors:
        for r in regressors:
            future[r] = pdf[r].median()
    future   = add_monsoon_flags(future)
    forecast = model.predict(future)
    mae      = mean_absolute_error(
        pdf["y"], forecast[forecast["ds"].isin(pdf["ds"])]["yhat"]
    )
    print(f"      In-sample MAE: {mae:.4f}")
    return model, forecast


prophet_temp_model, forecast_temp  = train_prophet(df, "temp",     regressors=["humidity", "pressure"])
prophet_max_model,  forecast_max   = train_prophet(df, "tempmax",  regressors=["humidity", "pressure"])
prophet_min_model,  forecast_min   = train_prophet(df, "tempmin",  regressors=["humidity", "pressure"])
prophet_hum_model,  forecast_hum   = train_prophet(df, "humidity", regressors=["pressure", "precip"],  seasonality_mode="additive")
prophet_pres_model, forecast_pres  = train_prophet(df, "pressure", regressors=["temp", "humidity"],    seasonality_mode="additive")
_,                  forecast_prec  = train_prophet(df, "precip",   regressors=["humidity", "pressure"], seasonality_mode="additive")
print("   ✅ Prophet done.")

# ─────────────────────────────────────────────────────────────
#  STEP 6 — XGBOOST + OPTUNA
# ─────────────────────────────────────────────────────────────
print("\n🔍 Training XGBoost + Optuna models...")

def optuna_tune_xgb(df_feat, target_col, n_trials=N_TRIALS):
    X    = df_feat[FEATURE_COLS]
    y    = df_feat[target_col]
    tscv = TimeSeriesSplit(n_splits=5)

    def objective(trial):
        params = {
            "n_estimators"     : trial.suggest_int("n_estimators", 200, 600),
            "learning_rate"    : trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "max_depth"        : trial.suggest_int("max_depth", 3, 7),
            "subsample"        : trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree" : trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight" : trial.suggest_int("min_child_weight", 1, 10),
            "gamma"            : trial.suggest_float("gamma", 0.0, 0.5),
            "reg_alpha"        : trial.suggest_float("reg_alpha", 0.0, 1.0),
            "reg_lambda"       : trial.suggest_float("reg_lambda", 0.5, 2.0),
            "random_state"     : 42,
            "verbosity"        : 0,
        }
        fold_maes = []
        for tr, val in tscv.split(X):
            m = XGBRegressor(**params)
            m.fit(X.iloc[tr], y.iloc[tr],
                  eval_set=[(X.iloc[val], y.iloc[val])], verbose=False)
            fold_maes.append(mean_absolute_error(y.iloc[val], m.predict(X.iloc[val])))
        return np.mean(fold_maes)

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    final = XGBRegressor(**study.best_params, random_state=42, verbosity=0)
    final.fit(X, y)
    print(f"   XGBoost → {target_col}  CV MAE: {study.best_value:.4f}")
    return final, study.best_params, study.best_value


xgb_temp,  best_temp,  mae_temp  = optuna_tune_xgb(df_feat, "temp")
xgb_max,   best_max,   mae_max   = optuna_tune_xgb(df_feat, "tempmax")
xgb_min,   best_min,   mae_min   = optuna_tune_xgb(df_feat, "tempmin")
xgb_hum,   best_hum,   mae_hum   = optuna_tune_xgb(df_feat, "humidity")
xgb_pres,  best_pres,  mae_pres  = optuna_tune_xgb(df_feat, "pressure")
print("   ✅ XGBoost done.")

# ─────────────────────────────────────────────────────────────
#  STEP 7 — BIDIRECTIONAL LSTM
# ─────────────────────────────────────────────────────────────
print("\n🧠 Training Bidirectional LSTM models...")

def build_lstm_sequences(data, lookback=LOOKBACK):
    X_seq, y_seq = [], []
    for i in range(lookback, len(data)):
        X_seq.append(data[i - lookback:i])
        y_seq.append(data[i])
    return np.array(X_seq), np.array(y_seq)


def train_lstm(df_feat, target_col, lookback=LOOKBACK):
    lstm_features = [target_col] + [
        f"{target_col}_lag_1", f"{target_col}_lag_7",
        f"{target_col}_roll_7", f"{target_col}_roll_30",
        "sin_doy", "cos_doy", "sin_month", "cos_month",
        "is_sw_monsoon", "is_ne_monsoon", "is_dry_season",
        "humidity_lag_1" if target_col != "humidity" else "pressure_lag_1",
        "pressure_lag_1" if target_col != "pressure" else "temp_lag_1",
        "precip_roll_7",
    ]
    lstm_features = [c for c in lstm_features if c in df_feat.columns]

    data_raw    = df_feat[lstm_features].values
    scaler      = MinMaxScaler()
    data_scaled = scaler.fit_transform(data_raw)
    target_idx  = 0

    X_seq, y_seq = build_lstm_sequences(data_scaled, lookback)
    split        = int(len(X_seq) * 0.90)
    X_train, X_val = X_seq[:split], X_seq[split:]
    y_train, y_val = y_seq[:split, target_idx], y_seq[split:, target_idx]

    model = Sequential([
        Bidirectional(LSTM(128, return_sequences=True),
                      input_shape=(lookback, X_seq.shape[2])),
        Dropout(0.25),
        LSTM(64, return_sequences=False),
        Dropout(0.20),
        Dense(32, activation="relu"),
        Dense(1),
    ])
    model.compile(optimizer=Adam(learning_rate=1e-3), loss="huber")

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=15,
                      restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                          patience=7, min_lr=1e-5, verbose=0),
    ]
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=LSTM_EPOCHS,
        batch_size=32,
        callbacks=callbacks,
        verbose=0,
    )

    def inverse_target(scaled_vals, scaler, target_idx, n_features):
        dummy = np.zeros((len(scaled_vals), n_features))
        dummy[:, target_idx] = scaled_vals
        return scaler.inverse_transform(dummy)[:, target_idx]

    n_feat     = data_scaled.shape[1]
    val_preds  = inverse_target(model.predict(X_val, verbose=0).flatten(), scaler, target_idx, n_feat)
    val_true   = inverse_target(y_val, scaler, target_idx, n_feat)
    val_mae    = mean_absolute_error(val_true, val_preds)
    print(f"   LSTM → {target_col}  Val MAE: {val_mae:.4f}  Epochs: {len(history.history['loss'])}")
    return model, scaler, lstm_features, val_mae


lstm_temp,  scaler_temp,  feats_temp,  lstm_mae_temp  = train_lstm(df_feat, "temp")
lstm_max,   scaler_max,   feats_max,   lstm_mae_max   = train_lstm(df_feat, "tempmax")
lstm_min,   scaler_min,   feats_min,   lstm_mae_min   = train_lstm(df_feat, "tempmin")
lstm_hum,   scaler_hum,   feats_hum,   lstm_mae_hum   = train_lstm(df_feat, "humidity")
lstm_pres,  scaler_pres,  feats_pres,  lstm_mae_pres  = train_lstm(df_feat, "pressure")
print("   ✅ LSTM done.")

# ─────────────────────────────────────────────────────────────
#  STEP 8 — SARIMA
# ─────────────────────────────────────────────────────────────
print("\n📈 Training SARIMA models...")

def train_sarima(series, exog, order=(2, 1, 2), seasonal_order=(2, 1, 1, 12)):
    result = SARIMAX(
        series, exog=exog,
        order=order, seasonal_order=seasonal_order,
        enforce_stationarity=False, enforce_invertibility=False,
    ).fit(disp=False)
    print(f"   SARIMA → {series.name}  AIC: {result.aic:.2f}")
    return result


sarima_temp = train_sarima(df["temp"],     exog=df[["humidity", "pressure"]])
sarima_max  = train_sarima(df["tempmax"],  exog=df[["humidity", "pressure"]])
sarima_min  = train_sarima(df["tempmin"],  exog=df[["humidity", "pressure"]])
sarima_hum  = train_sarima(df["humidity"], exog=df[["pressure", "precip"]])
sarima_pres = train_sarima(df["pressure"], exog=df[["temp", "humidity"]])
print("   ✅ SARIMA done.")

# ─────────────────────────────────────────────────────────────
#  STEP 9 — PRECIPITATION MODEL
# ─────────────────────────────────────────────────────────────
print("\n🌧️  Training Precipitation model...")

X_all      = df_feat[FEATURE_COLS]
y_precip   = df_feat["precip"]
y_rain_bin = (y_precip > 0.1).astype(int)
y_rain_amt = y_precip[y_precip > 0.1]
X_rain_amt = X_all.loc[y_rain_amt.index]


def optuna_tune_clf(X, y, n_trials=N_TRIALS):
    tscv       = TimeSeriesSplit(n_splits=5)
    pos_weight = (y == 0).sum() / (y == 1).sum()

    def objective(trial):
        params = {
            "n_estimators"     : trial.suggest_int("n_estimators", 200, 600),
            "learning_rate"    : trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "max_depth"        : trial.suggest_int("max_depth", 3, 7),
            "subsample"        : trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree" : trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "scale_pos_weight" : pos_weight,
            "eval_metric"      : "logloss",
            "random_state"     : 42,
            "verbosity"        : 0,
        }
        accs = []
        for tr, val in tscv.split(X):
            m = XGBClassifier(**params)
            m.fit(X.iloc[tr], y.iloc[tr], verbose=False)
            accs.append((m.predict(X.iloc[val]) == y.iloc[val]).mean())
        return -np.mean(accs)

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best_clf = XGBClassifier(**study.best_params, scale_pos_weight=pos_weight,
                             eval_metric="logloss", random_state=42, verbosity=0)
    best_clf.fit(X, y)
    print(f"   Classifier accuracy: {-study.best_value:.4f}")
    return best_clf


rain_clf       = optuna_tune_clf(X_all, y_rain_bin)
rain_amt_reg, _, rain_amt_mae = optuna_tune_xgb(
    df_feat.loc[y_rain_amt.index], "precip", n_trials=N_TRIALS
)
print("   ✅ Precipitation model done.")

# ─────────────────────────────────────────────────────────────
#  STEP 10 — BUILD FUTURE FEATURES
# ─────────────────────────────────────────────────────────────
print("\n📅 Building future feature matrix...")

def build_future_features(df, df_feat, nwp_df=None, periods=FORECAST_DAYS):
    future_dates = pd.date_range(
        start=df_feat.index[-1] + pd.Timedelta("1D"), periods=periods
    )
    fdf   = pd.DataFrame(index=future_dates)
    doy   = fdf.index.dayofyear
    month = fdf.index.month

    fdf["day_of_year"]  = doy
    fdf["month"]        = month
    fdf["sin_doy"]      = np.sin(2 * np.pi * doy / 365)
    fdf["cos_doy"]      = np.cos(2 * np.pi * doy / 365)
    fdf["sin_month"]    = np.sin(2 * np.pi * month / 12)
    fdf["cos_month"]    = np.cos(2 * np.pi * month / 12)

    for col in ALL_VARS:
        for lag in [1, 7, 14, 30, 365]:
            fdf[f"{col}_lag_{lag}"] = df[col].shift(lag).reindex(
                future_dates, method="nearest", tolerance=pd.Timedelta("30D")
            ).values
        for window in [7, 30]:
            fdf[f"{col}_roll_{window}"] = df[col].rolling(window).mean().iloc[-1]

    fdf["is_sw_monsoon"]  = month.isin([6, 7, 8, 9]).astype(float)
    fdf["is_ne_monsoon"]  = month.isin([10, 11, 12]).astype(float)
    fdf["is_dry_season"]  = month.isin([1, 2, 3, 4, 5]).astype(float)
    fdf["is_pre_monsoon"] = month.isin([3, 4, 5]).astype(float)
    fdf["precip_roll_7"]  = df["precip"].rolling(7).mean().iloc[-1]
    fdf["precip_roll_30"] = df["precip"].rolling(30).mean().iloc[-1]
    fdf["rain_day"]       = fdf["is_sw_monsoon"] * 0.7 + fdf["is_ne_monsoon"] * 0.4 + fdf["is_dry_season"] * 0.05
    fdf["rain_streak"]    = fdf["is_sw_monsoon"] * 5   + fdf["is_ne_monsoon"] * 2.5 + fdf["is_dry_season"] * 0.3
    fdf["temp_range"]     = (df["tempmax"] - df["tempmin"]).mean()
    fdf["humidity_x_temp"]= df["humidity"].median() * df["temp"].median() / 100

    if nwp_df is not None and len(nwp_df) > 0:
        for col in nwp_df.columns:
            fdf[col] = nwp_df[col].reindex(future_dates)
        for col in nwp_df.columns:
            fdf[col] = fdf[col].ffill().bfill()
    else:
        nwp_cols = ["nwp_temp", "nwp_tempmax", "nwp_tempmin",
                    "nwp_humidity", "nwp_pressure", "nwp_precip"]
        for col in nwp_cols:
            if col in FEATURE_COLS:
                fdf[col] = df_feat[col].mean()

    for col in FEATURE_COLS:
        if col not in fdf.columns:
            fdf[col] = 0

    return fdf[FEATURE_COLS]


future_feat  = build_future_features(df, df_feat, nwp_df if NWP_AVAILABLE else None)
future_dates = future_feat.index
print(f"   ✅ Future features: {future_feat.shape}  ({future_dates[0].date()} → {future_dates[-1].date()})")

# ─────────────────────────────────────────────────────────────
#  STEP 11 — GENERATE PREDICTIONS
# ─────────────────────────────────────────────────────────────
print("\n🔄 Generating predictions...")

# LSTM rolling forecast
def lstm_rolling_forecast(model, scaler, lstm_feature_cols, df_feat,
                           periods=FORECAST_DAYS, target_col="temp", lookback=LOOKBACK):
    n_features  = len(lstm_feature_cols)
    target_idx  = 0
    seed_data   = df_feat[lstm_feature_cols].values[-lookback:]
    seed_scaled = scaler.transform(seed_data)
    predictions_scaled = []
    current_seq = seed_scaled.copy()

    for _ in range(periods):
        X_input    = current_seq[np.newaxis, :, :]
        pred_scaled = model.predict(X_input, verbose=0)[0, 0]
        predictions_scaled.append(pred_scaled)
        next_row   = current_seq[-1].copy()
        next_row[target_idx] = pred_scaled
        current_seq = np.vstack([current_seq[1:], next_row])

    dummy = np.zeros((periods, n_features))
    dummy[:, target_idx] = predictions_scaled
    return scaler.inverse_transform(dummy)[:, target_idx]


lstm_pred_temp = lstm_rolling_forecast(lstm_temp,  scaler_temp,  feats_temp,  df_feat, target_col="temp")
lstm_pred_max  = lstm_rolling_forecast(lstm_max,   scaler_max,   feats_max,   df_feat, target_col="tempmax")
lstm_pred_min  = lstm_rolling_forecast(lstm_min,   scaler_min,   feats_min,   df_feat, target_col="tempmin")
lstm_pred_hum  = np.clip(lstm_rolling_forecast(lstm_hum,  scaler_hum,  feats_hum,  df_feat, target_col="humidity"), 0, 100)
lstm_pred_pres = lstm_rolling_forecast(lstm_pres,  scaler_pres,  feats_pres,  df_feat, target_col="pressure")
print("   ✅ LSTM forecasts done.")

# XGBoost predictions
Xf            = future_feat
xgb_pred_temp = xgb_temp.predict(Xf)
xgb_pred_max  = xgb_max.predict(Xf)
xgb_pred_min  = xgb_min.predict(Xf)
xgb_pred_hum  = np.clip(xgb_hum.predict(Xf), 0, 100)
xgb_pred_pres = xgb_pres.predict(Xf)
print("   ✅ XGBoost forecasts done.")

# Precipitation 2-step
monthly_rain_prob = {}
monthly_rain_amt  = {}
for m in range(1, 13):
    mask      = df.index.month == m
    rain_mask = mask & (df["precip"] > 0.1)
    monthly_rain_prob[m] = (df.loc[mask, "precip"] > 0.1).mean()
    monthly_rain_amt[m]  = df.loc[rain_mask, "precip"].mean() if rain_mask.sum() > 0 else 0.0

np.random.seed(42)
future_months      = future_dates.month
rain_occurs        = np.array([np.random.random() < monthly_rain_prob[m] for m in future_months])
monthly_correction = np.array([monthly_rain_amt[m] for m in future_months])
prophet_precip_raw = np.clip(forecast_prec.tail(FORECAST_DAYS)["yhat"].values, 0, None)
rain_amount_all    = np.clip(rain_amt_reg.predict(Xf), 0, None)
blended_amount     = 0.55 * rain_amount_all + 0.45 * prophet_precip_raw
final_amount       = 0.65 * blended_amount  + 0.35 * monthly_correction
xgb_pred_prec      = np.where(rain_occurs, np.clip(final_amount, 0, None), 0.0)
print(f"   ✅ Precipitation: {(xgb_pred_prec > 0).sum()} rainy days | {xgb_pred_prec.sum():.1f} mm annual total")

# Prophet predictions
p_temp = forecast_temp.tail(FORECAST_DAYS)["yhat"].values
p_max  = forecast_max.tail(FORECAST_DAYS)["yhat"].values
p_min  = forecast_min.tail(FORECAST_DAYS)["yhat"].values
p_hum  = np.clip(forecast_hum.tail(FORECAST_DAYS)["yhat"].values, 0, 100)
p_pres = forecast_pres.tail(FORECAST_DAYS)["yhat"].values

# SARIMA predictions
n = FORECAST_DAYS
def sarima_fc(model, exog_dict):
    return model.forecast(n, exog=pd.DataFrame({k: [v]*n for k, v in exog_dict.items()})).values

s_temp = sarima_fc(sarima_temp, {"humidity": df["humidity"].median(), "pressure": df["pressure"].median()})
s_max  = sarima_fc(sarima_max,  {"humidity": df["humidity"].median(), "pressure": df["pressure"].median()})
s_min  = sarima_fc(sarima_min,  {"humidity": df["humidity"].median(), "pressure": df["pressure"].median()})
s_hum  = np.clip(sarima_fc(sarima_hum,  {"pressure": df["pressure"].median(), "precip": df["precip"].median()}), 0, 100)
s_pres = sarima_fc(sarima_pres, {"temp": df["temp"].median(), "humidity": df["humidity"].median()})
print("   ✅ SARIMA forecasts done.")

# NWP arrays
if NWP_AVAILABLE:
    nwp_temp_arr = nwp_df["nwp_temp"].reindex(future_dates).values
    nwp_max_arr  = nwp_df["nwp_tempmax"].reindex(future_dates).values
    nwp_min_arr  = nwp_df["nwp_tempmin"].reindex(future_dates).values
    nwp_hum_arr  = nwp_df["nwp_humidity"].reindex(future_dates).values
    nwp_pres_arr = nwp_df["nwp_pressure"].reindex(future_dates).values
    nwp_mask     = ~np.isnan(nwp_temp_arr)
else:
    nwp_mask     = np.zeros(FORECAST_DAYS, dtype=bool)
    nwp_temp_arr = nwp_max_arr = nwp_min_arr = nwp_hum_arr = nwp_pres_arr = np.zeros(FORECAST_DAYS)

# ─────────────────────────────────────────────────────────────
#  STEP 12 — DYNAMIC ENSEMBLE
# ─────────────────────────────────────────────────────────────
print("\n🎯 Running dynamic ensemble...")

def dynamic_ensemble(p, lstm, xgb, sarima, nwp, nwp_mask,
                     clip_min=None, clip_max=None):
    result = np.zeros(FORECAST_DAYS)
    for i in range(FORECAST_DAYS):
        if nwp_mask[i] and not np.isnan(nwp[i]):
            result[i] = (0.40 * nwp[i]   + 0.25 * p[i]    +
                         0.20 * lstm[i]   + 0.10 * xgb[i]  + 0.05 * sarima[i])
        else:
            result[i] = (0.30 * p[i]      + 0.30 * lstm[i] +
                         0.25 * xgb[i]    + 0.15 * sarima[i])
    if clip_min is not None:
        result = np.clip(result, clip_min, clip_max)
    return np.round(result, 2)


final_temp = dynamic_ensemble(p_temp, lstm_pred_temp, xgb_pred_temp, s_temp, nwp_temp_arr, nwp_mask)
final_max  = dynamic_ensemble(p_max,  lstm_pred_max,  xgb_pred_max,  s_max,  nwp_max_arr,  nwp_mask)
final_min  = dynamic_ensemble(p_min,  lstm_pred_min,  xgb_pred_min,  s_min,  nwp_min_arr,  nwp_mask)
final_hum  = dynamic_ensemble(p_hum,  lstm_pred_hum,  xgb_pred_hum,  s_hum,  nwp_hum_arr,  nwp_mask, 0, 100)
final_pres = dynamic_ensemble(p_pres, lstm_pred_pres, xgb_pred_pres, s_pres, nwp_pres_arr, nwp_mask)
final_prec = np.round(xgb_pred_prec, 2)

results = pd.DataFrame({
    "avg_temp" : final_temp,
    "max_temp" : final_max,
    "min_temp" : final_min,
    "humidity" : final_hum,
    "pressure" : final_pres,
    "precip"   : final_prec,
}, index=future_dates)
results.index.name = "date"
print("   ✅ Ensemble complete.")

# Monthly summary
monthly = results.resample("ME").agg({
    "avg_temp" : "mean",
    "max_temp" : "max",
    "min_temp" : "min",
    "humidity" : "mean",
    "pressure" : "mean",
    "precip"   : "sum",
}).round(2)
monthly.index = monthly.index.strftime("%b %Y")

# ─────────────────────────────────────────────────────────────
#  STEP 13 — CHARTS
# ─────────────────────────────────────────────────────────────
print("\n📊 Generating forecast charts...")

sw_p  = mpatches.Patch(color="blue",  alpha=0.35, label="SW Monsoon (Jun–Sep)")
ne_p  = mpatches.Patch(color="teal",  alpha=0.35, label="NE Monsoon (Oct–Dec)")
pre_p = mpatches.Patch(color="red",   alpha=0.35, label="Pre-Monsoon (Mar–May)")
nwp_p = mpatches.Patch(color="gold",  alpha=0.50, label="NWP direct range (ECMWF)")


def shade_monsoon(ax, results):
    for year in sorted(set(results.index.year)):
        for (ms, me_month, me_day, col) in [
            (6, 9, 30, "blue"), (10, 12, 31, "teal"), (3, 5, 31, "red")
        ]:
            s = max(pd.Timestamp(f"{year}-{ms:02d}-01"), results.index[0])
            e = min(pd.Timestamp(f"{year}-{me_month:02d}-{me_day}"), results.index[-1])
            if s < e:
                ax.axvspan(s, e, alpha=0.07, color=col)


def add_nwp_band(ax):
    if NWP_AVAILABLE and nwp_mask.any():
        nwp_end = future_dates[nwp_mask][-1]
        ax.axvspan(future_dates[0], nwp_end, alpha=0.08, color="gold")


# Temperature
fig, ax = plt.subplots(figsize=(18, 6))
ax.plot(df["temp"],         color="gray",       lw=0.7, alpha=0.4, label="Historical Avg")
ax.plot(df["tempmax"],      color="lightcoral", lw=0.6, alpha=0.3, label="Historical Max")
ax.plot(df["tempmin"],      color="lightblue",  lw=0.6, alpha=0.3, label="Historical Min")
ax.plot(results["avg_temp"],color="darkorange", lw=2.2, label="Forecast Avg")
ax.plot(results["max_temp"],color="red",        lw=1.2, ls="--",   label="Forecast Max")
ax.plot(results["min_temp"],color="steelblue",  lw=1.2, ls="--",   label="Forecast Min")
ax.fill_between(results.index, results["min_temp"], results["max_temp"], alpha=0.1, color="orange")
ax.axvline(x=df.index[-1], color="black", ls=":", lw=1.5, label="Forecast Start")
add_nwp_band(ax)
shade_monsoon(ax, results)
h, _ = ax.get_legend_handles_labels()
ax.legend(handles=h + [sw_p, ne_p, pre_p, nwp_p], fontsize=8, ncol=3, loc="upper left")
ax.set_title(f"{LOCATION_NAME} Temperature Forecast — Next {FORECAST_DAYS} Days\n"
             "Ensemble: Prophet + LSTM + XGBoost(Optuna) + SARIMA + NWP(ECMWF)", fontsize=12)
ax.set_ylabel("Temperature (°C)")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("forecast_temperature.png", dpi=150, bbox_inches="tight")
plt.close()

# Humidity
fig, ax = plt.subplots(figsize=(18, 5))
ax.plot(df["humidity"],      color="gray",   lw=0.7, alpha=0.4, label="Historical")
ax.plot(results["humidity"], color="purple", lw=2.0, label="Forecast")
ax.axhline(60, color="orange", ls="--", lw=0.8, alpha=0.6, label="Comfort (60%)")
ax.axhline(90, color="red",    ls="--", lw=0.8, alpha=0.6, label="High (90%)")
ax.axvline(x=df.index[-1], color="black", ls=":", lw=1.5)
add_nwp_band(ax)
shade_monsoon(ax, results)
h, _ = ax.get_legend_handles_labels()
ax.legend(handles=h + [sw_p, ne_p, pre_p, nwp_p], fontsize=8, loc="upper left")
ax.set_title(f"{LOCATION_NAME} Humidity Forecast — Next {FORECAST_DAYS} Days", fontsize=12)
ax.set_ylabel("Relative Humidity (%)")
ax.set_ylim(0, 105)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("forecast_humidity.png", dpi=150, bbox_inches="tight")
plt.close()

# Pressure
fig, ax = plt.subplots(figsize=(18, 5))
ax.plot(df["pressure"],      color="gray",  lw=0.7, alpha=0.4, label="Historical")
ax.plot(results["pressure"], color="green", lw=2.0, label="Forecast")
ax.axvline(x=df.index[-1], color="black", ls=":", lw=1.5)
add_nwp_band(ax)
shade_monsoon(ax, results)
h, _ = ax.get_legend_handles_labels()
ax.legend(handles=h + [sw_p, ne_p, pre_p, nwp_p], fontsize=8, loc="upper left")
ax.set_title(f"{LOCATION_NAME} Pressure Forecast — Next {FORECAST_DAYS} Days", fontsize=12)
ax.set_ylabel("Pressure (hPa)")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("forecast_pressure.png", dpi=150, bbox_inches="tight")
plt.close()

# Precipitation
fig, axes = plt.subplots(2, 1, figsize=(18, 9), sharex=True,
                         gridspec_kw={"height_ratios": [2, 1]})
axes[0].bar(df.index,      df["precip"],      color="lightblue", alpha=0.6, width=1, label="Historical")
axes[0].bar(results.index, results["precip"], color="navy",      alpha=0.7, width=1, label="Forecast")
axes[0].axvline(x=df.index[-1], color="black", ls=":", lw=1.5)
add_nwp_band(axes[0])
shade_monsoon(axes[0], results)
axes[0].set_title(f"{LOCATION_NAME} Precipitation Forecast — Next {FORECAST_DAYS} Days", fontsize=12)
axes[0].set_ylabel("Daily Precipitation (mm)")
axes[0].grid(alpha=0.3)
h0, _ = axes[0].get_legend_handles_labels()
axes[0].legend(handles=h0 + [sw_p, ne_p, pre_p], fontsize=8, loc="upper left")

axes[1].plot(df["precip"].rolling(30).sum(),      color="steelblue", lw=1.0, alpha=0.5, label="Historical 30d total")
axes[1].plot(results["precip"].rolling(30).sum(), color="navy",      lw=1.8, label="Forecast 30d rolling total")
axes[1].axvline(x=df.index[-1], color="black", ls=":", lw=1.5)
shade_monsoon(axes[1], results)
axes[1].set_ylabel("30-Day Rolling Total (mm)")
axes[1].set_xlabel("Date")
axes[1].grid(alpha=0.3)
axes[1].legend(fontsize=8, loc="upper left")
plt.tight_layout()
plt.savefig("forecast_precipitation.png", dpi=150, bbox_inches="tight")
plt.close()

print("   ✅ All charts saved.")

# ─────────────────────────────────────────────────────────────
#  STEP 14 — SAVE CSVs
# ─────────────────────────────────────────────────────────────
results.to_csv("daily_forecast_365days.csv")
monthly.to_csv("monthly_forecast_summary.csv")
print("\n✅ CSVs saved.")

# ─────────────────────────────────────────────────────────────
#  STEP 15 — PRINT SUMMARY
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("   MODEL PERFORMANCE SUMMARY")
print("=" * 60)
print(f"\n   XGBoost + Optuna (CV MAE):")
for name, mae in [("temp", mae_temp), ("tempmax", mae_max), ("tempmin", mae_min),
                  ("humidity", mae_hum), ("pressure", mae_pres)]:
    print(f"     {name:<10} : {mae:.4f}")
print(f"\n   Bidirectional LSTM (Val MAE):")
for name, mae in [("temp", lstm_mae_temp), ("tempmax", lstm_mae_max),
                  ("tempmin", lstm_mae_min), ("humidity", lstm_mae_hum),
                  ("pressure", lstm_mae_pres)]:
    print(f"     {name:<10} : {mae:.4f}")
print(f"\n   NWP (ECMWF) : {'✅ Active (first 16 days)' if NWP_AVAILABLE else '❌ Not available'}")
print(f"\n   Ensemble weights:")
print(f"     Days 1-16 : NWP 40% | Prophet 25% | LSTM 20% | XGBoost 10% | SARIMA 5%")
print(f"     Days 17+  : Prophet 30% | LSTM 30% | XGBoost 25% | SARIMA 15%")
print("=" * 60)

print("\n📅 Monthly Forecast Summary:")
print("─" * 82)
print(f"{'Month':<12}  {'Avg°C':>6}  {'Max°C':>6}  {'Min°C':>6}  "
      f"{'Humidity':>9}  {'Pressure':>9}  {'Precip(mm)':>11}")
print("─" * 82)
for m_label, row in monthly.iterrows():
    print(f"{m_label:<12}  {row['avg_temp']:>6.1f}  {row['max_temp']:>6.1f}  "
          f"{row['min_temp']:>6.1f}  {row['humidity']:>9.1f}  "
          f"{row['pressure']:>9.1f}  {row['precip']:>11.1f}")
print("─" * 82)

print("\n🎉 Pipeline complete! Output files:")
for f in ["daily_forecast_365days.csv", "monthly_forecast_summary.csv",
          "forecast_temperature.png", "forecast_humidity.png",
          "forecast_pressure.png", "forecast_precipitation.png",
          "historical_overview.png"]:
    exists = "✅" if os.path.exists(f) else "❌"
    print(f"   {exists} {f}")
