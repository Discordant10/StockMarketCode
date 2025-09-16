import sqlite3
import pandas as pd
import numpy as np
from config import FEATURE_ENGINEERING

def run_feature_engineering():
    conn = sqlite3.connect("db/market_data.db")

    # Load raw equity data
    df = pd.read_sql("SELECT * FROM equity_data", conn)

    if df.empty:
        print("⚠️ No equity data found for feature engineering.")
        return

    features = []

    # Daily returns
    if "returns" in FEATURE_ENGINEERING["INCLUDE_FEATURES"]:
        df["daily_return"] = df.groupby("ticker")["adj_close"].pct_change()
        features.append("daily_return")

    # Moving averages
    if "moving_avg" in FEATURE_ENGINEERING["INCLUDE_FEATURES"]:
        for window in FEATURE_ENGINEERING["LOOKBACK_WINDOWS"]:
            col = f"ma_{window}"
            df[col] = df.groupby("ticker")["adj_close"].transform(
                lambda x: x.rolling(window).mean()
            )
            features.append(col)

    # Bollinger Bands (20-day)
    if "bollinger_bands" in FEATURE_ENGINEERING["INCLUDE_FEATURES"]:
        window = 20
        df["bb_mid"] = df.groupby("ticker")["adj_close"].transform(
            lambda x: x.rolling(window).mean()
        )
        df["bb_std"] = df.groupby("ticker")["adj_close"].transform(
            lambda x: x.rolling(window).std()
        )
        df["bb_upper"] = df["bb_mid"] + (2 * df["bb_std"])
        df["bb_lower"] = df["bb_mid"] - (2 * df["bb_std"])
        features += ["bb_mid", "bb_std", "bb_upper", "bb_lower"]

    # RSI (14-day)
    if "rsi" in FEATURE_ENGINEERING["INCLUDE_FEATURES"]:
        window = 14
        delta = df.groupby("ticker")["adj_close"].diff()
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)

        df["avg_gain"] = (
            pd.Series(gain).groupby(df["ticker"]).transform(
                lambda x: pd.Series(x).rolling(window).mean()
            )
        )
        df["avg_loss"] = (
            pd.Series(loss).groupby(df["ticker"]).transform(
                lambda x: pd.Series(x).rolling(window).mean()
            )
        )
        rs = df["avg_gain"] / df["avg_loss"].replace(0, np.nan)
        df["rsi"] = 100 - (100 / (1 + rs))
        features.append("rsi")

    # MACD (12-26 EMA difference + Signal line 9 EMA)
    if "macd" in FEATURE_ENGINEERING["INCLUDE_FEATURES"]:
        short_ema = df.groupby("ticker")["adj_close"].transform(
            lambda x: x.ewm(span=12, adjust=False).mean()
        )
        long_ema = df.groupby("ticker")["adj_close"].transform(
            lambda x: x.ewm(span=26, adjust=False).mean()
        )
        df["macd"] = short_ema - long_ema
        df["macd_signal"] = df.groupby("ticker")["macd"].transform(
            lambda x: x.ewm(span=9, adjust=False).mean()
        )
        features += ["macd", "macd_signal"]

    # Volatility (30-day rolling std of returns)
    if "volatility" in FEATURE_ENGINEERING["INCLUDE_FEATURES"]:
        df["volatility"] = df.groupby("ticker")["daily_return"].transform(
            lambda x: x.rolling(30).std()
        )
        features.append("volatility")

    # Drop helper columns
    df = df.drop(columns=["avg_gain", "avg_loss"], errors="ignore")

    # Save features
    df.to_sql("equity_features", conn, if_exists="replace", index=False)
    conn.close()

    print(f"✅ Feature engineering complete. Features saved: {features}")
