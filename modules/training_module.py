"""
training_module.py

Train models (tabular or LSTM) on engineered features and evaluate via evaluation_module.

Usage:
    from modules import training_module as tm
    # train and evaluate
    best_model, metrics = tm.train_and_evaluate(model_type="rf", eval_after_train=True)

Key functions:
- load_feature_data: read equity_features table (or accept df)
- make_targets: create hybrid targets (classification: buy/sell/hold, regression: next-day return/price)
- train_model: trains RF or (optional) LSTM
- predict_next_day: produce buy/sell list with indicative price + stop loss
- train_and_evaluate: full pipeline, calls evaluation_module.evaluate_trades
- write_predictions_to_db: optional
"""

import sqlite3
import pandas as pd
import numpy as np
import os
from typing import Tuple, Dict, Any, Optional
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, mean_squared_error
import joblib
from datetime import datetime, timedelta

# Local modules
from modules import evaluation_module as em
from modules import portfolio_module as pm
import torch
import torch.nn as nn
import torch.optim as optim
from config import DB_PATH

# config defaults (try to import user's config.py if present)
try:
    from config import FEATURE_ENGINEERING as FE_CFG
    from config import LOOKBACK_YEARS, WATCHLIST_FILE
    TRAINING_CFG = getattr(__import__("config"), "TRAINING_CFG", {})
except Exception:
    FE_CFG = {"INCLUDE_FEATURES": ["returns", "moving_avg", "bollinger_bands", "rsi", "macd", "volatility"],
             "LOOKBACK_WINDOWS": [14, 30, 50, 200]}
    LOOKBACK_YEARS = 10
    WATCHLIST_FILE = "inputs/tickers.csv"
    TRAINING_CFG = {}



# Default training config
DEFAULT_CFG = {
    "model_type": "rf",          # 'rf' (RandomForest) or 'lstm' (requires torch)
    "rf_params": {"n_estimators": 200, "max_depth": 8, "random_state": 42},
    "test_size": 0.2,
    "random_state": 42,
    "min_return_for_buy": 0.01,  # 1% expected return -> label BUY
    "min_return_for_sell": -0.01, # <= -1% -> label SELL
    "per_trade_amount": 100000,
    "portfolio_limit": 5,
    "stop_loss_pct": 0.05,       # default stop loss 5% below buy
    "model_artifact_path": "models"
}

# Merge user-specified TRAINING_CFG if present
CFG = DEFAULT_CFG.copy()
CFG.update(TRAINING_CFG if isinstance(TRAINING_CFG, dict) else {})

# Try import torch for LSTM option
_HAS_TORCH = False
try:
    import torch
    from torch import nn, optim
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False


# ---------------------------
# Utilities: DB read/write
# ---------------------------
def _get_db_connection(db_path: str = DB_PATH):
    conn = sqlite3.connect(db_path)
    return conn


def load_feature_data(db_path: str = DB_PATH, tickers: Optional[list] = None) -> pd.DataFrame:
    """
    Load engineered features from equity_features table. If not present raise error.
    Returns a DataFrame with ticker/date and feature columns.
    """
    conn = _get_db_connection(db_path)
    try:
        df = pd.read_sql_query("SELECT * FROM equity_features", conn)
    finally:
        conn.close()
    if df.empty:
        raise ValueError("No equity_features found in DB. Run feature_engineering first.")
    df['date'] = pd.to_datetime(df['date'])
    if tickers:
        df = df[df['ticker'].isin(tickers)].copy()
    # sort
    df = df.sort_values(['ticker', 'date']).reset_index(drop=True)
    return df


def make_targets(df: pd.DataFrame, target_horizon_days: int = 1,
                 min_buy_return: float = None, min_sell_return: float = None
                 ) -> pd.DataFrame:
    """
    Create hybrid targets:
      - regression target: next_day_return (adj_close_pct change)
      - classification target: buy(1)/sell(-1)/hold(0) based on next_day_return thresholds
    """
    df = df.copy()
    # require adj_close column
    if 'adj_close' not in df.columns and 'close' in df.columns:
        df['adj_close'] = df['close']

    # compute next day return per ticker
    df['next_adj_close'] = df.groupby('ticker')['adj_close'].shift(-target_horizon_days)
    df['next_return'] = (df['next_adj_close'] - df['adj_close']) / df['adj_close']

    mb = CFG['min_return_for_buy'] if min_buy_return is None else min_buy_return
    ms = CFG['min_return_for_sell'] if min_sell_return is None else min_sell_return

    def label_fn(r):
        if pd.isna(r):
            return np.nan
        if r >= mb:
            return 1
        if r <= ms:
            return -1
        return 0

    df['label'] = df['next_return'].apply(label_fn)
    return df


# ---------------------------
# Feature matrix builder
# ---------------------------
def build_X_y(df: pd.DataFrame, feature_cols: Optional[list] = None):
    """
    Build X (features) and y (labels) for ML training.
    Drops rows with NaN label.
    """
    df = df.copy()
    if feature_cols is None:
        # automatically pick numeric columns except targets
        exclude = {'ticker', 'date', 'next_adj_close', 'next_return', 'label'}
        feature_cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    # drop rows missing label
    df = df[~df['label'].isna()].copy()
    X = df[feature_cols].fillna(0.0).values
    y_cls = df['label'].values.astype(int)
    y_reg = df['next_return'].fillna(0.0).values.astype(float)
    return X, y_cls, y_reg, df, feature_cols


# ---------------------------
# Model training
# ---------------------------
def train_rf(X_train, y_train, X_valid=None, y_valid=None, params=None):
    params = params or CFG['rf_params']
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    if X_valid is not None and y_valid is not None:
        preds = model.predict(X_valid)
        report = classification_report(y_valid, preds, output_dict=True)
    else:
        report = None
    return model, report


# Optional LSTM (sequence) implementation (simple)
class SimpleLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=1, out_dim=3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, out_dim)  # classification: 3 classes
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)


def train_lstm(df, feature_cols, epochs=20, lr=1e-3, device='cpu'):
    """
    Train a very simple LSTM classifier on sequences of features.
    Expects df sorted by ticker/date and with label column.
    """
    if not _HAS_TORCH:
        raise RuntimeError("PyTorch not available. Install torch to use LSTM.")

    # prepare sequences per ticker: last N days => label of next day
    seq_len = 20
    X_seq = []
    y_seq = []
    for t, g in df.groupby('ticker'):
        arr = g[feature_cols].fillna(0.0).values
        labels = g['label'].values
        for i in range(len(arr) - seq_len):
            X_seq.append(arr[i:i + seq_len])
            y_seq.append(labels[i + seq_len])
    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq).astype(int)

    # train/test split
    X_train, X_val, y_train, y_val = train_test_split(X_seq, y_seq, test_size=CFG['test_size'], random_state=CFG['random_state'])
    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.long).to(device)
    X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val = torch.tensor(y_val, dtype=torch.long).to(device)

    model = SimpleLSTM(input_dim=X_seq.shape[2], hidden_dim=64)
    model.to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for e in range(epochs):
        model.train()
        opt.zero_grad()
        out = model(X_train)
        loss = loss_fn(out, y_train)
        loss.backward()
        opt.step()
        # validation
        model.eval()
        with torch.no_grad():
            val_out = model(X_val)
            val_loss = loss_fn(val_out, y_val)
        # You may want to print progress
    return model


# ---------------------------
# Prediction & Signal generation
# ---------------------------
def predict_and_generate_signals(model, feature_cols, scaler, recent_df: pd.DataFrame,
                                 model_type: str = "rf", per_trade_amount: float = None,
                                 stop_loss_pct: float = None) -> pd.DataFrame:
    """
    recent_df: DataFrame of latest rows per ticker representing the most recent date to predict for.
    It must contain the same feature_cols and 'ticker' and last known 'adj_close' (or 'close').
    Returns DataFrame of proposed trades: columns [ticker, action, date, price, stop_loss]
    """
    df = recent_df.copy()
    X = df[feature_cols].fillna(0.0).values
    if scaler:
        Xs = scaler.transform(X)
    else:
        Xs = X

    if model_type == "rf":
        preds = model.predict(Xs)
        # For regression price estimate, not provided by RF classifier. We'll estimate target price as
        # current_price * (1 + expected_return). We can approximate expected_return via a small regressor or model.predict_proba.
        # For now we'll keep indicative price = last close (user requested indicative price).
        scores = None
    else:
        # LSTM path not implemented in detail here
        preds = model.predict(Xs)
        scores = None

    proposals = []
    for i, row in df.reset_index(drop=True).iterrows():
        action_label = preds[i]
        if action_label == 1:
            action = "BUY"
            price = row.get('adj_close', row.get('close'))
            if pd.isna(price):
                price = row.get('close')
            # set stop loss as pct below price
            sl_pct = stop_loss_pct if stop_loss_pct is not None else CFG['stop_loss_pct']
            stop = price * (1 - sl_pct)
        elif action_label == -1:
            action = "SELL"
            price = row.get('adj_close', row.get('close'))
            sl_pct = stop_loss_pct if stop_loss_pct is not None else CFG['stop_loss_pct']
            stop = price * (1 + sl_pct)  # for sell (closing long), stop here is above; kept for completeness
        else:
            continue  # hold -> no proposal

        # compute units based on per_trade_amount
        units = CFG['per_trade_amount'] / price if per_trade_amount is None else per_trade_amount / price
        proposals.append({
            'ticker': row['ticker'],
            'action': action,
            'date': row['date'],
            'price': float(price),
            'stop_loss': float(stop),
            'units': float(units)
        })

    return pd.DataFrame(proposals)


# ---------------------------
# Main training & evaluation pipeline
# ---------------------------
def train_and_evaluate(model_type: str = None,
                       db_path: str = DB_PATH,
                       tickers: Optional[list] = None,
                       write_predictions: bool = False,
                       save_model: bool = True
                       ) -> Tuple[Any, Dict[str, Any]]:
    """
    Full pipeline:
      - load features
      - create targets
      - train model (rf or lstm)
      - generate next-day proposals (on last available date in features)
      - call evaluation_module.evaluate_trades on proposals (simulation)
      - return trained_model, metrics (classification report + evaluation summary)
    """
    cfg = CFG.copy()
    if model_type:
        cfg['model_type'] = model_type

    # load features
    df = load_feature_data(db_path=db_path, tickers=tickers)

    df_targets = make_targets(df)

    # prepare X,y
    X, y_cls, y_reg, df_model, feature_cols = build_X_y(df_targets)
    if len(X) == 0:
        raise ValueError("No labeled rows found for training. Check your features and target creation.")

    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y_cls, test_size=cfg['test_size'], random_state=cfg['random_state'])

    # scaler + model pipeline for RF
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = None
    cls_report = None

    if cfg['model_type'] == 'rf':
        model, cls_report = train_rf(X_train_scaled, y_train, X_test_scaled, y_test, params=cfg.get('rf_params'))
    elif cfg['model_type'] == 'lstm':
        if not _HAS_TORCH:
            raise RuntimeError("LSTM requested but torch not available.")
        # simple LSTM training (sequence-based) - not fully integrated with above pipeline
        model = train_lstm(df_model, feature_cols)
    else:
        raise ValueError("Unknown model_type in cfg")

    # Save model & scaler
    if save_model:
        os.makedirs(cfg.get('model_artifact_path', 'models'), exist_ok=True)
        joblib.dump({'model': model, 'scaler': scaler, 'feature_cols': feature_cols}, os.path.join(cfg.get('model_artifact_path', 'models'), f"model_{cfg['model_type']}.joblib"))

    # Generate proposals for next day using most recent rows per ticker
    last_rows = df_model.sort_values(['ticker', 'date']).groupby('ticker').tail(1).reset_index(drop=True)
    proposals = predict_and_generate_signals(model, feature_cols, scaler, last_rows, model_type=cfg['model_type'], per_trade_amount=cfg.get('per_trade_amount'), stop_loss_pct=cfg.get('stop_loss_pct'))

    # Run evaluation module on proposals (it will not modify portfolio DB; pass portfolio limit)
    if not proposals.empty:
        eval_df = em.evaluate_trades(proposals[['ticker', 'action', 'date', 'price', 'stop_loss']], price_df=None,
                                     db_path=db_path,
                                     portfolio_limit=cfg.get('portfolio_limit', 5),
                                     per_trade_amount=cfg.get('per_trade_amount'),
                                     write_to_db=False,
                                     verbose=False)
    else:
        eval_df = pd.DataFrame()

    metrics = {
        'classification_report': cls_report,
        'eval_summary': eval_df
    }

    # Optionally write proposals to db
    if write_predictions and not proposals.empty:
        conn = _get_db_connection(db_path)
        try:
            proposals_to_write = proposals.copy()
            proposals_to_write['date'] = proposals_to_write['date'].apply(lambda x: pd.to_datetime(x).strftime("%Y-%m-%d"))
            proposals_to_write.to_sql('model_proposals', conn, if_exists='append', index=False)
        finally:
            conn.close()

    return model, metrics
