# modules/data_download.py

import os
import pandas as pd
import yfinance as yf
import sqlite3
from tqdm import tqdm
from datetime import datetime, timedelta
from config import FEATURE_ENGINEERING
from modules import feature_engineering as fe
from config import DB_PATH

def read_watchlist(file_path: str) -> list:
    """Read watchlist CSV and return list of tickers."""
    if not os.path.exists(file_path):
        print(f"⚠️ Watchlist {file_path} not found.")
        return []
    df = pd.read_csv(file_path, dtype=str)
    tickers = df['ticker'].astype(str).str.strip().tolist()
    return tickers


def get_last_date(ticker: str) -> str | None:
    """Get last available date in DB for given ticker."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT MAX(date) FROM equities WHERE ticker=?", (ticker,))
    result = cursor.fetchone()[0]
    conn.close()
    return result


def download_equity_data(tickers: list, lookback_years: int = 10):
    """Download and update equity data incrementally into SQLite DB."""
    if not tickers:
        print("⚠️ No tickers provided.")
        return

    conn = sqlite3.connect(DB_PATH)

    for ticker in tqdm(tickers, desc="Downloading equities"):
        try:
            last_date = get_last_date(ticker)
            if last_date:
                start = (datetime.strptime(last_date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
            else:
                start = (datetime.now() - timedelta(days=365 * lookback_years)).strftime("%Y-%m-%d")

            end = datetime.now().strftime("%Y-%m-%d")

            df = yf.download(ticker, start=start, end=end, progress=False)
            if df.empty:
                print(f"⚠️ No data for {ticker}.")
                continue

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [col[0] for col in df.columns]

            df.reset_index(inplace=True)
            df["ticker"] = ticker
            df.rename(columns={
                "Date": "date",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume"
            }, inplace=True)

            df[["ticker", "date", "open", "high", "low", "close", "volume"]].to_sql(
                "equities", conn, if_exists="append", index=False
            )
            # Run feature engineering if enabled
            if FEATURE_ENGINEERING["AUTO_RUN"]:
                print("⚙️ Running feature engineering after DB update...")
                fe.run_feature_engineering()

        except Exception as e:
            print(f"❌ Error downloading {ticker}: {e}")

    conn.commit()
    conn.close()
