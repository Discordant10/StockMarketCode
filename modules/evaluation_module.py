"""
Evaluation Module

Primary function:
    evaluate_trades(trades_df, price_df=None, db_path="db/market_data.db",
                    portfolio_limit=5, per_trade_amount=None,
                    default_units=1, write_to_db=False)

Description:
- Accepts a DataFrame `trades_df` containing proposed BUY/SELL rows.
- Simulates execution: checks if proposed price is reached on the provided date
  (or the next available trading day).
- If a BUY is executed it opens a simulated position (if portfolio limit allows).
- For each open position we monitor price history from buy_date -> sell_date:
    - if low <= stop_loss at any day before the sell_date => close at stop_loss (stop-loss triggered)
    - else if a SELL exists and is executed on its sell_date => close at sell_price
    - else trade remains open (unrealized)
- Does not modify portfolio DB. Optionally writes summary to an `eval_trades` table.
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional

DEFAULT_DB_PATH = "db/market_data.db"


# -----------------------
# Helpers
# -----------------------
def _load_price_data_from_db(db_path: str = DEFAULT_DB_PATH) -> pd.DataFrame:
    """Load OHLC price data from the equities table in SQLite db."""
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query("SELECT ticker, date, open, high, low, close, adj_close, volume FROM equities", conn)
    finally:
        conn.close()
    if df.empty:
        return df
    df['date'] = pd.to_datetime(df['date'])
    # ensure numeric
    for c in ['open', 'high', 'low', 'close', 'adj_close', 'volume']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    return df


def _get_next_trading_date(df_ticker: pd.DataFrame, target_date: pd.Timestamp) -> Optional[pd.Timestamp]:
    """
    Return the earliest trading date in df_ticker >= target_date.
    df_ticker must have 'date' column of dtype datetime.
    """
    mask = df_ticker['date'] >= target_date
    s = df_ticker.loc[mask, 'date']
    if s.empty:
        return None
    return s.iloc[0]


def _row_for_date(df_ticker: pd.DataFrame, date: pd.Timestamp) -> Optional[pd.Series]:
    """Return the row (Series) for exact date if present, else None."""
    r = df_ticker.loc[df_ticker['date'] == date]
    if r.empty:
        return None
    return r.iloc[0]


# -----------------------
# Core evaluation function
# -----------------------
def evaluate_trades(
    trades_df: pd.DataFrame,
    price_df: pd.DataFrame = None,
    db_path: str = DEFAULT_DB_PATH,
    portfolio_limit: int = 5,
    per_trade_amount: Optional[float] = None,
    default_units: float = 1.0,
    write_to_db: bool = False,
    db_table_name: str = "eval_trades",
    verbose: bool = False
) -> pd.DataFrame:
    """
    Evaluate a list of proposed trades.

    trades_df columns required:
        - ticker (str)
        - action (str): "BUY" or "SELL"
        - date (str or datetime) : date when action is proposed (interpreted as business/trading date)
        - price (float): proposed target price for buy/sell
        - stop_loss (float or NaN): stop loss price (for BUY, typically < buy price)

    Parameters:
        trades_df: DataFrame of proposals (can include multiple tickers & dates).
        price_df: optional price history DataFrame (ticker, date, open, high, low, close, ...)
                  If None will be loaded from db_path (table 'equities').
        portfolio_limit: max concurrent open positions (simulation only).
        per_trade_amount: if set, units = per_trade_amount / buy_price; else uses default_units.
        default_units: units to assume if per_trade_amount not given.
        write_to_db: if True, append summary to `db_path` table `db_table_name`.
        verbose: if True, prints progress info.

    Returns:
        summary_df: DataFrame summarizing trades & results with columns:
            ['ticker','buy_date','buy_price','buy_executed','buy_exec_date','buy_exec_price',
             'sell_date','sell_price','sell_executed','sell_exec_date','sell_exec_price',
             'stop_loss','stop_loss_triggered','stop_loss_date','stop_price',
             'units','profit','profit_pct','status']
    """
    # Validate trades_df
    required_cols = {'ticker', 'action', 'date', 'price'}
    if not required_cols.issubset(set(trades_df.columns)):
        raise ValueError(f"trades_df must include columns: {required_cols}")

    df = trades_df.copy()
    df['action'] = df['action'].str.upper().str.strip()
    df['date'] = pd.to_datetime(df['date'])
    # normalize stop_loss column
    if 'stop_loss' not in df.columns:
        df['stop_loss'] = np.nan

    # Load price data if not provided
    if price_df is None:
        price_df = _load_price_data_from_db(db_path)
    if price_df is None or price_df.empty:
        raise ValueError("No price data available (price_df empty and DB read failed).")

    # index price_df by ticker for fast access
    price_by_ticker = {t: g.sort_values('date').reset_index(drop=True) for t, g in price_df.groupby('ticker')}

    # Sort events chronologically (stable)
    df = df.sort_values(['date', 'ticker', 'action']).reset_index(drop=True)

    open_positions = {}  # ticker -> position dict
    open_count = 0

    results = []  # will contain dicts per completed trade or event summary

    # We'll process rows in chronological order; when BUY executes we open position,
    # when SELL executes for that ticker we attempt to close existing position.
    for idx, row in df.iterrows():
        ticker = row['ticker']
        action = row['action']
        prop_date = pd.to_datetime(row['date'])
        prop_price = float(row['price']) if not pd.isna(row['price']) else np.nan
        stop_loss = float(row['stop_loss']) if not pd.isna(row['stop_loss']) else np.nan

        if verbose:
            print(f"[{prop_date.date()}] {action} {ticker} @ {prop_price} stop:{stop_loss}")

        # price history for this ticker
        if ticker not in price_by_ticker:
            # no price data for ticker
            if verbose:
                print(f"  ⚠️ No price history for {ticker}. Skipping.")
            # record as skipped
            results.append({
                'ticker': ticker,
                'buy_date': prop_date if action == 'BUY' else pd.NaT,
                'buy_price': prop_price if action == 'BUY' else np.nan,
                'buy_executed': False if action == 'BUY' else np.nan,
                'buy_exec_date': pd.NaT if action == 'BUY' else pd.NaT,
                'buy_exec_price': np.nan,
                'sell_date': prop_date if action == 'SELL' else pd.NaT,
                'sell_price': prop_price if action == 'SELL' else np.nan,
                'sell_executed': False if action == 'SELL' else np.nan,
                'sell_exec_date': pd.NaT,
                'sell_exec_price': np.nan,
                'stop_loss': stop_loss,
                'stop_loss_triggered': False,
                'stop_loss_date': pd.NaT,
                'stop_price': np.nan,
                'units': 0,
                'profit': 0.0,
                'profit_pct': 0.0,
                'status': 'no_price_data'
            })
            continue

        hist = price_by_ticker[ticker]

        # Find the trading date to evaluate this action:
        exec_date = _get_next_trading_date(hist, prop_date)
        if exec_date is None:
            # no trading days on or after prop_date
            if verbose:
                print(f"  ⚠️ No trading days on/after {prop_date.date()} for {ticker}. Skipping.")
            results.append({
                'ticker': ticker,
                'buy_date': prop_date if action == 'BUY' else pd.NaT,
                'buy_price': prop_price if action == 'BUY' else np.nan,
                'buy_executed': False if action == 'BUY' else np.nan,
                'buy_exec_date': pd.NaT,
                'buy_exec_price': np.nan,
                'sell_date': prop_date if action == 'SELL' else pd.NaT,
                'sell_price': prop_price if action == 'SELL' else np.nan,
                'sell_executed': False if action == 'SELL' else np.nan,
                'sell_exec_date': pd.NaT,
                'sell_exec_price': np.nan,
                'stop_loss': stop_loss,
                'stop_loss_triggered': False,
                'stop_loss_date': pd.NaT,
                'stop_price': np.nan,
                'units': 0,
                'profit': 0.0,
                'profit_pct': 0.0,
                'status': 'no_trading_days'
            })
            continue

        # get the row for exec_date
        price_row = _row_for_date(hist, exec_date)
        if price_row is None:
            # should not happen because we used _get_next_trading_date, but guard anyway
            if verbose:
                print(f"  ⚠️ no exact price row for {exec_date} (ticker {ticker}).")
            results.append({
                'ticker': ticker,
                'status': 'no_price_row'
            })
            continue

        day_high = price_row['high']
        day_low = price_row['low']

        if action == 'BUY':
            # If already have an open position for this ticker, skip opening (no averaging)
            if ticker in open_positions:
                results.append({
                    'ticker': ticker,
                    'buy_date': prop_date,
                    'buy_price': prop_price,
                    'buy_executed': False,
                    'buy_exec_date': pd.NaT,
                    'buy_exec_price': np.nan,
                    'sell_date': pd.NaT,
                    'sell_price': np.nan,
                    'sell_executed': np.nan,
                    'sell_exec_date': pd.NaT,
                    'sell_exec_price': np.nan,
                    'stop_loss': stop_loss,
                    'stop_loss_triggered': False,
                    'stop_loss_date': pd.NaT,
                    'stop_price': np.nan,
                    'units': 0,
                    'profit': 0.0,
                    'profit_pct': 0.0,
                    'status': 'already_open'
                })
                continue

            # Check if buy price was reached on exec_date
            buy_reached = (day_low <= prop_price <= day_high)
            if not buy_reached:
                results.append({
                    'ticker': ticker,
                    'buy_date': prop_date,
                    'buy_price': prop_price,
                    'buy_executed': False,
                    'buy_exec_date': pd.NaT,
                    'buy_exec_price': np.nan,
                    'sell_date': pd.NaT,
                    'sell_price': np.nan,
                    'sell_executed': np.nan,
                    'sell_exec_date': pd.NaT,
                    'sell_exec_price': np.nan,
                    'stop_loss': stop_loss,
                    'stop_loss_triggered': False,
                    'stop_loss_date': pd.NaT,
                    'stop_price': np.nan,
                    'units': 0,
                    'profit': 0.0,
                    'profit_pct': 0.0,
                    'status': 'buy_not_reached'
                })
                continue

            # If buy_reached, attempt to open if portfolio limit allows
            if open_count >= portfolio_limit:
                results.append({
                    'ticker': ticker,
                    'buy_date': prop_date,
                    'buy_price': prop_price,
                    'buy_executed': True,
                    'buy_exec_date': exec_date,
                    'buy_exec_price': prop_price,
                    'sell_date': pd.NaT,
                    'sell_price': np.nan,
                    'sell_executed': np.nan,
                    'sell_exec_date': pd.NaT,
                    'sell_exec_price': np.nan,
                    'stop_loss': stop_loss,
                    'stop_loss_triggered': False,
                    'stop_loss_date': pd.NaT,
                    'stop_price': np.nan,
                    'units': 0,
                    'profit': 0.0,
                    'profit_pct': 0.0,
                    'status': 'skipped_due_to_portfolio_limit'
                })
                continue

            # open position
            units = default_units
            if per_trade_amount is not None and per_trade_amount > 0:
                units = float(per_trade_amount) / float(prop_price)

            position = {
                'ticker': ticker,
                'buy_date': exec_date,
                'buy_price': prop_price,
                'stop_loss': stop_loss,
                'units': units,
                'buy_event_row_idx': idx  # helpful for tracing
            }
            open_positions[ticker] = position
            open_count += 1

            # If stop_loss already hit on the same exec_date (e.g., market gaps), close immediately
            if not np.isnan(stop_loss) and day_low <= stop_loss:
                # close at stop_loss on exec_date
                closed_price = stop_loss
                profit = (closed_price - position['buy_price']) * position['units']
                profit_pct = (closed_price - position['buy_price']) / position['buy_price'] * 100.0
                # record result and remove position
                open_positions.pop(ticker, None)
                open_count = max(0, open_count - 1)
                results.append({
                    'ticker': ticker,
                    'buy_date': prop_date,
                    'buy_price': prop_price,
                    'buy_executed': True,
                    'buy_exec_date': exec_date,
                    'buy_exec_price': prop_price,
                    'sell_date': exec_date,
                    'sell_price': closed_price,
                    'sell_executed': True,
                    'sell_exec_date': exec_date,
                    'sell_exec_price': closed_price,
                    'stop_loss': stop_loss,
                    'stop_loss_triggered': True,
                    'stop_loss_date': exec_date,
                    'stop_price': closed_price,
                    'units': position['units'],
                    'profit': profit,
                    'profit_pct': profit_pct,
                    'status': 'closed_same_day_stop_loss'
                })
            else:
                # opened successfully, not closed yet
                results.append({
                    'ticker': ticker,
                    'buy_date': prop_date,
                    'buy_price': prop_price,
                    'buy_executed': True,
                    'buy_exec_date': exec_date,
                    'buy_exec_price': prop_price,
                    'sell_date': pd.NaT,
                    'sell_price': np.nan,
                    'sell_executed': np.nan,
                    'sell_exec_date': pd.NaT,
                    'sell_exec_price': np.nan,
                    'stop_loss': stop_loss,
                    'stop_loss_triggered': False,
                    'stop_loss_date': pd.NaT,
                    'stop_price': np.nan,
                    'units': position['units'],
                    'profit': np.nan,
                    'profit_pct': np.nan,
                    'status': 'opened'
                })

        elif action == 'SELL':
            # If there is no open position for this ticker, record and skip
            if ticker not in open_positions:
                # Sell without an open position (could be a short or closing earlier outside simulated buys)
                sell_reached = (day_low <= prop_price <= day_high)
                results.append({
                    'ticker': ticker,
                    'buy_date': pd.NaT,
                    'buy_price': np.nan,
                    'buy_executed': np.nan,
                    'buy_exec_date': pd.NaT,
                    'buy_exec_price': np.nan,
                    'sell_date': prop_date,
                    'sell_price': prop_price,
                    'sell_executed': sell_reached,
                    'sell_exec_date': exec_date if sell_reached else pd.NaT,
                    'sell_exec_price': prop_price if sell_reached else np.nan,
                    'stop_loss': stop_loss,
                    'stop_loss_triggered': False,
                    'stop_loss_date': pd.NaT,
                    'stop_price': np.nan,
                    'units': 0,
                    'profit': 0.0,
                    'profit_pct': 0.0,
                    'status': 'sell_without_open'
                })
                continue

            # There is an open position for this ticker — attempt to close it
            pos = open_positions[ticker]

            # Determine the interval to monitor: buy_exec_date -> sell_exec_date (exec_date)
            buy_dt = pos['buy_date']
            sell_dt = exec_date

            # Get price slice between buy_dt and sell_dt inclusive
            mask = (hist['date'] >= buy_dt) & (hist['date'] <= sell_dt)
            period = hist.loc[mask].sort_values('date')

            stop_triggered = False
            stop_triggered_date = pd.NaT
            stop_exec_price = np.nan

            # Check for stop-loss first (scan day-by-day)
            if not np.isnan(pos.get('stop_loss', np.nan)):
                sl = pos['stop_loss']
                for _, per_row in period.iterrows():
                    if per_row['low'] <= sl:
                        stop_triggered = True
                        stop_triggered_date = per_row['date']
                        stop_exec_price = sl
                        break

            if stop_triggered:
                # Close at stop price on stop_triggered_date
                profit = (stop_exec_price - pos['buy_price']) * pos['units']
                profit_pct = (stop_exec_price - pos['buy_price']) / pos['buy_price'] * 100.0
                open_positions.pop(ticker, None)
                open_count = max(0, open_count - 1)
                results.append({
                    'ticker': ticker,
                    'buy_date': pos['buy_date'],
                    'buy_price': pos['buy_price'],
                    'buy_executed': True,
                    'buy_exec_date': pos['buy_date'],
                    'buy_exec_price': pos['buy_price'],
                    'sell_date': prop_date,
                    'sell_price': prop_price,
                    'sell_executed': False,   # sell target did not execute since close happened by stop
                    'sell_exec_date': pd.NaT,
                    'sell_exec_price': np.nan,
                    'stop_loss': pos['stop_loss'],
                    'stop_loss_triggered': True,
                    'stop_loss_date': stop_triggered_date,
                    'stop_price': stop_exec_price,
                    'units': pos['units'],
                    'profit': profit,
                    'profit_pct': profit_pct,
                    'status': 'closed_by_stop_loss'
                })
                continue

            # If no stop-loss triggered, check whether sell target was reached on its exec_date
            if len(period) == 0:
                # no price rows in the period (shouldn't happen), treat as sell not executed
                results.append({
                    'ticker': ticker,
                    'buy_date': pos['buy_date'],
                    'buy_price': pos['buy_price'],
                    'buy_executed': True,
                    'buy_exec_date': pos['buy_date'],
                    'buy_exec_price': pos['buy_price'],
                    'sell_date': prop_date,
                    'sell_price': prop_price,
                    'sell_executed': False,
                    'sell_exec_date': pd.NaT,
                    'sell_exec_price': np.nan,
                    'stop_loss': pos['stop_loss'],
                    'stop_loss_triggered': False,
                    'stop_loss_date': pd.NaT,
                    'stop_price': np.nan,
                    'units': pos['units'],
                    'profit': np.nan,
                    'profit_pct': np.nan,
                    'status': 'sell_no_price_period'
                })
                continue

            # Check sell exec_date row specifically
            sell_day_row = _row_for_date(hist, sell_dt)
            sell_reached = False
            if sell_day_row is not None:
                if sell_day_row['low'] <= prop_price <= sell_day_row['high']:
                    sell_reached = True

            if sell_reached:
                # Close at sell price
                close_price = prop_price
                profit = (close_price - pos['buy_price']) * pos['units']
                profit_pct = (close_price - pos['buy_price']) / pos['buy_price'] * 100.0
                open_positions.pop(ticker, None)
                open_count = max(0, open_count - 1)
                results.append({
                    'ticker': ticker,
                    'buy_date': pos['buy_date'],
                    'buy_price': pos['buy_price'],
                    'buy_executed': True,
                    'buy_exec_date': pos['buy_date'],
                    'buy_exec_price': pos['buy_price'],
                    'sell_date': prop_date,
                    'sell_price': prop_price,
                    'sell_executed': True,
                    'sell_exec_date': sell_dt,
                    'sell_exec_price': close_price,
                    'stop_loss': pos['stop_loss'],
                    'stop_loss_triggered': False,
                    'stop_loss_date': pd.NaT,
                    'stop_price': np.nan,
                    'units': pos['units'],
                    'profit': profit,
                    'profit_pct': profit_pct,
                    'status': 'closed_by_sell'
                })
            else:
                # Sell target not reached on its execute date; position remains open (unrealized)
                results.append({
                    'ticker': ticker,
                    'buy_date': pos['buy_date'],
                    'buy_price': pos['buy_price'],
                    'buy_executed': True,
                    'buy_exec_date': pos['buy_date'],
                    'buy_exec_price': pos['buy_price'],
                    'sell_date': prop_date,
                    'sell_price': prop_price,
                    'sell_executed': False,
                    'sell_exec_date': pd.NaT,
                    'sell_exec_price': np.nan,
                    'stop_loss': pos['stop_loss'],
                    'stop_loss_triggered': False,
                    'stop_loss_date': pd.NaT,
                    'stop_price': np.nan,
                    'units': pos['units'],
                    'profit': np.nan,
                    'profit_pct': np.nan,
                    'status': 'sell_not_reached_still_open'
                })
        else:
            # unknown action
            results.append({
                'ticker': ticker,
                'status': 'unknown_action'
            })

    # Final: consolidate results into DataFrame
    summary_df = pd.DataFrame(results)
    # normalize dtypes
    if 'buy_exec_date' in summary_df.columns:
        summary_df['buy_exec_date'] = pd.to_datetime(summary_df['buy_exec_date'])
    if 'sell_exec_date' in summary_df.columns:
        summary_df['sell_exec_date'] = pd.to_datetime(summary_df['sell_exec_date'])
    if 'stop_loss_date' in summary_df.columns:
        summary_df['stop_loss_date'] = pd.to_datetime(summary_df['stop_loss_date'])

    # Optionally persist summary to DB (separate table)
    if write_to_db:
        conn = sqlite3.connect(db_path)
        try:
            # create table if not exists
            conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {db_table_name} (
                ticker TEXT,
                buy_date TEXT,
                buy_price REAL,
                buy_executed INTEGER,
                buy_exec_date TEXT,
                buy_exec_price REAL,
                sell_date TEXT,
                sell_price REAL,
                sell_executed INTEGER,
                sell_exec_date TEXT,
                sell_exec_price REAL,
                stop_loss REAL,
                stop_loss_triggered INTEGER,
                stop_loss_date TEXT,
                stop_price REAL,
                units REAL,
                profit REAL,
                profit_pct REAL,
                status TEXT
            )
            """)
            # convert date cols to ISO strings for DB
            df_to_write = summary_df.copy()
            for col in ['buy_date', 'buy_exec_date', 'sell_date', 'sell_exec_date', 'stop_loss_date']:
                if col in df_to_write.columns:
                    df_to_write[col] = df_to_write[col].apply(lambda x: pd.to_datetime(x).strftime("%Y-%m-%d") if pd.notna(x) else None)
            df_to_write.to_sql(db_table_name, conn, if_exists='append', index=False)
        finally:
            conn.close()

    return summary_df
