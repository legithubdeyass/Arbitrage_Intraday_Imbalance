import pandas as pd
import numpy as np

def missing_values_summary(df, name = "Dataframe"):
    mv_df = df.isnull().sum().reset_index()
    mv_df.columns = ["Variable", "Missing Count"]
    mv_df["Missing Value (%)"] = mv_df["Missing Count"]/ df.shape[0] * 100
    mv_df = mv_df.sort_values("Missing Value (%)", ascending=False)

    print(f"\nMissing Value summary for {name} (Total rows : {df.shape[0]})")
    print(mv_df.to_string(index=False))
  
 
def interpolate_time(df):
    df = df.copy()
    df.index = pd.to_datetime(df.index)

    for col in df.columns:
        if df[col].isna().any():
            df[col] = df[col].interpolate(method='time', limit_direction='both')

    return df


def compute_coverage_ratios(df, epsilon=1e-5):
    def afrr_ratio(row):
        imb = row['imb_volume']
        if imb > 0:
            return row['afrr_down'] / (abs(imb) + epsilon)
        elif imb < 0:
            return row['afrr_up'] / (abs(imb) + epsilon)
        else:
            return 0

    def mfrr_ratio(row):
        imb = row['imb_volume']
        if imb > 0:
            residual = max(0, abs(imb) - row['afrr_down'])
            return row['mfrr_down'] / (residual + epsilon)
        elif imb < 0:
            residual = max(0, abs(imb) - row['afrr_up'])
            return row['mfrr_up'] / (residual + epsilon)
        else:
            return 0

    df['afrr_cover_ratio'] = df.apply(afrr_ratio, axis=1)
    df['mfrr_cover_ratio'] = df.apply(mfrr_ratio, axis=1)
    return df


def coverage_status(row):
    imb = row["imb_volume"]
    if abs(imb) < 1e-3:
        return 0
    if imb > 0:
        residual = max(0, abs(imb) - row["afrr_down"])
        return 1 if residual <= row["mfrr_down"] else -1
    else:
        residual = max(0, abs(imb) - row["afrr_up"])
        return 1 if residual <= row["mfrr_up"] else -1


def classify_spread(row, threshold=0):
    long_spread = row["spread_long"]
    short_spread = row["spread_short"]
    if long_spread > threshold and short_spread < -threshold:
        return 1
    elif short_spread > threshold and long_spread < -threshold:
        return -1
    elif long_spread < 0 and short_spread < 0:
        return 0


def compute_historical_spread(row, df, threshold=5):
    n = len(df)
    position = df.index.get_loc(row.name)
    weight = (position - 1) / n
    spread_buy = row["imb_price_pos"] - row["ID_QH_VWAP"]
    spread_sell = row["ID_QH_VWAP"] - row["imb_price_neg"]
    if spread_buy > threshold:
        return weight
    elif spread_sell > threshold:
        return -weight
    else:
        return 0


def map_position_to_volume(row, threshold=5):
    if row["position"] == 1:
        return 10
    elif row["position"] == -1:
        return -10
    return 0


def compute_PnL_optimal(row):
    if row["position"] == 1:
        return abs(row["spread_long"] * row["target_volume"])
    elif row["position"] == -1:
        return abs(row["spread_short"] * row["target_volume"])
    return 0


def compute_realized_PnL(row):
    if row["prediction"] > 0 and row["prediction"] <= 10:
        return row["spread_long"] * row["prediction"]
    elif row["prediction"] >= -10 and row["prediction"] < 0:
        return -row["spread_short"] * row["prediction"]
    return 0


def add_temporal_features(df):
    df['hour'] = df.index.hour.astype('float64')
    df['dayofweek'] = df.index.dayofweek.astype('float64')
    df['quarter'] = df.index.quarter.astype('float64')
    df['month'] = df.index.month.astype('float64')
    df['year'] = df.index.year.astype('float64')
    df['dayofyear'] = df.index.dayofyear.astype('float64')
    df['dayofmonth'] = df.index.day.astype('float64')
    df['weekofyear'] = df.index.isocalendar().week.astype('float64')
    return df


def lag_features(df, lag_dict, drop=False):
    for name, lags in lag_dict.items():
        for lag in lags:
            lagged_col = f"{name}_lagged_{lag}"
            df[lagged_col] = df[name].shift(lag)
            df[lagged_col] = df[lagged_col].fillna(df[name])
        if drop:
            df.drop(columns=name, inplace=True)
    return df


def add_all_features(df):
    df = df.copy()

    lag_dict = {  # dictionnaire lag tel quel
        "imb_price_pos": [4, 5, 6],
        "imb_price_neg": [4, 5, 6],
        "nuclear_real": [4, 5, 6],
        "solar_fcst": [4, 5, 6],
        "wind_fcst": [4, 5, 6],
        "fossil_gas_real": [4, 5, 6],
        "imb_volume": [4, 5, 6],
        "load_real": [4, 5, 6],
        'load_err': [4, 5, 6],
        "solar_real": [4, 5, 6],
        'solar_err': [4, 5, 6],
        "wind_real": [4, 5, 6],
        'wind_err': [4, 5, 6],
        "afrr_up": [4, 5, 6],
        "afrr_down": [4, 5, 6],
        "mfrr_up": [4, 5, 6],
        "mfrr_down": [4, 5, 6],
        "historical_spread": [4, 5, 6],
        "afrr_cover_ratio": [4, 5, 6],
        "mfrr_cover_ratio": [4, 5, 6],
        "imbalance_status": [4, 5, 6],
        "spread_long": [4, 5, 6],
        "spread_short": [4, 5, 6],
        "total_reserves": [4, 5, 6],
        'total_diff': [4, 5, 6],
        'total_prod': [4, 5, 6],
        "target_volume": [4, 5, 6]
    }

    df["load_err"] = df["load_fcst"] - df["load_real"]
    df["solar_err"] = df["solar_fcst"] - df["solar_real"]
    df["wind_err"] = df["wind_fcst"] - df["wind_real"]
    df["total_prod"] = df["nuclear_real"] + df["fossil_gas_real"] + df["solar_real"] + df["wind_real"]
    df["total_reserves"] = df["afrr_up"] + df["afrr_down"] + df["mfrr_up"] + df["mfrr_down"]
    df["total_diff"] = df["total_prod"] - df["load_real"]
    df["spread_long"] = df["imb_price_pos"] - df["ID_QH_VWAP"]
    df["spread_short"] = df["ID_QH_VWAP"] - df["imb_price_neg"]
    df["position"] = df.apply(classify_spread, axis=1)
    df["historical_spread"] = df.apply(lambda row: compute_historical_spread(row, df), axis=1)
    df["target_volume"] = df.apply(map_position_to_volume, axis=1)
    df["imbalance_status"] = df.apply(coverage_status, axis=1)
    df["PnL_optimal"] = df.apply(compute_PnL_optimal, axis=1)

    df = compute_coverage_ratios(df)
    df = add_temporal_features(df)
    df = lag_features(df, lag_dict)
    df = df.ffill()
    return df


def split_train_test(df, start='2024-01-01', end='2025-01-01'):
    train = df[df.index < end]
    test = df[df.index >= start]
    return train, test
