import os
import sqlite3
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from modules import evaluation_module
from config import FEATURE_ENGINEERING, DB_PATH

# -----------------------------
# Model: Hybrid LSTM
# -----------------------------
class HybridLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, output_dim=3):
        super(HybridLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # last hidden state
        return self.fc(out)

# -----------------------------
# Load data from SQLite
# -----------------------------
def load_data():
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql("SELECT * FROM equity_features", conn)
    except Exception as e:
        print(f"[ERROR] Could not read equity_features table: {e}")
        return None
    finally:
        conn.close()
    return df

# -----------------------------
# Training
# -----------------------------
def train_model(epochs=20, lr=0.001, test_size=0.2):
    df = load_data()
    if df is None or df.empty:
        print("[WARNING] No training data available.")
        return

    if "target" not in df.columns:
        raise RuntimeError("equity_features table missing 'target' column. Please run feature_engineering with include_target=True.")

    # Only use numeric columns for features
    feature_cols = df.select_dtypes(include=[np.number]).columns.drop("target")
    X = df[feature_cols].values
    y = df["target"].map({-1: 0, 0: 1, 1: 2}).values
    
    # Check for NaN/Inf
    if np.isnan(X).any() or np.isinf(X).any():
        raise ValueError("Features contain NaN or Inf values.")
    if np.isnan(y).any() or np.isinf(y).any():
        raise ValueError("Targets contain NaN or Inf values.")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=True)

    # Get indices for train/test split
    indices = np.arange(len(df))
    train_idx, test_idx = train_test_split(indices, test_size=test_size, shuffle=True, stratify=y)
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Torch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)
    y_test = torch.tensor(y_test, dtype=torch.long)

    # Model setup
    model = HybridLSTM(input_dim=X_train.shape[2])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        if epoch % 5 == 0:
            print(f"[INFO] Epoch {epoch}/{epochs}, Loss: {loss.item():.4f}")

    # Evaluation
    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        preds = outputs.argmax(dim=1).cpu().numpy()
        y_test_np = y_test.cpu().numpy()

    # Classification report
    report = classification_report(y_test_np, preds, output_dict=True)

    # Portfolio-style evaluation using evaluation_module
    # Prepare trades_df for evaluation: reconstruct BUY/SELL proposals from predictions
    trades_df = df.iloc[test_idx].copy() if "ticker" in df.columns and "date" in df.columns else None
    if trades_df is not None:
        trades_df = trades_df[["ticker", "date"]].copy()
        trades_df["action"] = np.where(preds == 2, "BUY", np.where(preds == 0, "SELL", "HOLD"))
        trades_df["price"] = np.nan  # You may want to fill this with actual price info
        trades_df["stop_loss"] = np.nan
        # Only keep BUY/SELL actions for evaluation
        trades_df = trades_df[trades_df["action"].isin(["BUY", "SELL"])]
        try:
            eval_results_df = evaluation_module.evaluate_trades(trades_df)
            eval_results = eval_results_df.describe(include="all").to_dict()
        except Exception as e:
            print(f"[WARNING] Evaluation module failed: {e}")
            eval_results = {}
    else:
        eval_results = {}

    # Save formatted report
    save_training_report(report, eval_results)

    return model

# -----------------------------
# Save training results
# -----------------------------
def save_training_report(report, eval_results):
    conn = sqlite3.connect(DB_PATH)
    report_df = pd.DataFrame(report).transpose()
    report_df["training_date"] = pd.Timestamp.now()
    report_df.to_sql("training_results", conn, if_exists="append", index=True)
    conn.close()

    print("\n=== Training Report ===")
    print(report_df.round(3).to_string())

    if eval_results:
        print("\n=== Evaluation Results ===")
        for k, v in eval_results.items():
            print(f"{k}: {v}")

# -----------------------------
# Entry point
# -----------------------------
if __name__ == "__main__":
    train_model()