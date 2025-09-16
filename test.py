import pandas as pd
from modules import evaluation_module as em

# Example trades (BUY then SELL)
trades = pd.DataFrame([
    {"ticker": "INFY.NS", "action": "BUY",  "date": "2024-12-26", "price": 1880.0, "stop_loss": 1820.0},
    {"ticker": "INFY.NS", "action": "SELL", "date": "2024-12-30", "price": 1879.0, "stop_loss": 1910.0},
])

# If you already have price history in DB, omit price_df and use db_path default
summary = em.evaluate_trades(trades, price_df=None, db_path="db/market_data.db",
                             portfolio_limit=5, per_trade_amount=100000,
                             write_to_db=False, verbose=True)

print(summary)
