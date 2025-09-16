# modules/evaluation_module.py

import pandas as pd
from datetime import datetime
from modules import portfolio_module as pm
from modules import db_utils
from config import EVAL_CONFIG
"""
Evaluation Module

- Accepts DataFrame of equity features with columns:
  [ticker, date, open, high, low, close, volume, ...features]
- Runs a simple backtesting loop using configurable buy/sell logic
- Records transactions in the portfolio DB if write_to_db=True
- Returns a DataFrame with signals and PnL
"""


def run_backtest(features_df: pd.DataFrame,
                 portfolio_id: int,
                 config: dict = None) -> pd.DataFrame:
    """
    Run a backtest on features_df using simple rules.
    """
    if config is None:
        config = EVAL_CONFIG

    results = []
    position = None  # current open position dict
    cash = config["initial_cash"]

    for _, row in features_df.iterrows():
        ticker = row["ticker"]
        date = pd.to_datetime(row["date"])
        close = row["close"]

        signal = 0  # default: hold

        # --- Simple Buy Logic ---
        if position is None and row.get("returns", 0) > config["buy_threshold"]:
            units = cash // close
            if units > 0:
                buy_cost = units * close
                cash -= buy_cost
                position = {
                    "ticker": ticker,
                    "buy_date": date,
                    "buy_price": close,
                    "units": units,
                    "total_cost": buy_cost
                }
                signal = 1
                if config["write_to_db"]:
                    pm.add_transaction(
                        portfolio_id,
                        ticker=ticker,
                        buy_date=date.strftime("%Y-%m-%d"),
                        units=units,
                        buy_price=close,
                        fees=0
                    )

        # --- Sell Logic ---
        elif position is not None:
            profit_pct = (close - position["buy_price"]) / position["buy_price"]

            if profit_pct >= config["sell_threshold"] or profit_pct <= -config["stop_loss"]:
                revenue = position["units"] * close
                cash += revenue
                signal = -1
                if config["write_to_db"]:
                    pm.close_transaction(
                        portfolio_id,
                        ticker=position["ticker"],
                        sell_date=date.strftime("%Y-%m-%d"),
                        units=position["units"],
                        sell_price=close,
                        fees=0
                    )
                position = None

        results.append({
            "date": date,
            "ticker": ticker,
            "close": close,
            "signal": signal,
            "cash": cash,
            "position": position["units"] if position else 0
        })

    return pd.DataFrame(results)


# --- Utility for writing evaluation results ---
def save_eval_results(df: pd.DataFrame, db_path="db/market_data.db"):
    """
    Save evaluation results to DB (table: eval_results).
    """
    conn = db_utils.get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS eval_results (
            ticker TEXT,
            date TEXT,
            signal INTEGER,
            close REAL,
            cash REAL,
            position INTEGER,
            PRIMARY KEY (ticker, date)
        )
    """)
    df.to_sql("eval_results", conn, if_exists="append", index=False)
    conn.commit()
    conn.close()
