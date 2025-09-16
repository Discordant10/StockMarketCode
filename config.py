# config.py
import os

# Base directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(BASE_DIR, "db")
INPUTS_DIR = os.path.join(BASE_DIR, "inputs")
DATA_DIR = os.path.join(BASE_DIR, "data")

# Ensure required folders exist
os.makedirs(DB_DIR, exist_ok=True)
os.makedirs(INPUTS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# Database
DB_PATH = os.path.join(DB_DIR, "market_data.db")

# Watchlist file
WATCHLIST_FILE = os.path.join(INPUTS_DIR, "tickers.csv")

# Default settings
LOOKBACK_YEARS = 10

# Feature engineering settings
FEATURE_ENGINEERING = {
    "AUTO_RUN": True,        # If True, runs automatically after DB update
    "LOOKBACK_WINDOWS": [14, 30, 50, 200],  # For moving averages
    "INCLUDE_FEATURES": [
        "returns",
        "moving_avg",
        "bollinger_bands",
        "rsi",
        "macd",
        "volatility"
    ],
    "target_threshold": 0.005,   # 0.5% move for buy/sell
    "include_target": True       # whether to add a target column to equity_features
}

# Evaluation Module config
EVAL_CONFIG = {
    "buy_threshold": 0.03,       # Placeholder: e.g. RSI/Signal-based
    "sell_threshold": -0.02,     # Placeholder
    "stop_loss": 0.05,           # 5% stop loss
    "initial_cash": 100000,
    "write_to_db": True
}