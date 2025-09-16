import sqlite3
from datetime import datetime
from config import DB_PATH

def init_portfolio_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # 1. Portfolios
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS portfolios (
        portfolio_id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT DEFAULT 'Master',
        type TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        cash_balance REAL DEFAULT 0
    )
    """)

    # 2. Cash Flows
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS cash_flows (
        cashflow_id INTEGER PRIMARY KEY AUTOINCREMENT,
        portfolio_id INTEGER,
        date TEXT,
        amount REAL,
        note TEXT,
        FOREIGN KEY (portfolio_id) REFERENCES portfolios (portfolio_id)
    )
    """)

    # 3. Transactions
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS transactions (
        transaction_id INTEGER PRIMARY KEY AUTOINCREMENT,
        portfolio_id INTEGER,
        ticker TEXT,
        trade_date TEXT,
        action TEXT CHECK(action IN ('BUY','SELL')),
        units REAL,
        price REAL,
        fees REAL,
        total_value REAL,
        profit REAL,
        profit_pct REAL,
        FOREIGN KEY (portfolio_id) REFERENCES portfolios (portfolio_id)
    )
    """)

    # 4. Holdings (current open positions)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS holdings (
        holding_id INTEGER PRIMARY KEY AUTOINCREMENT,
        portfolio_id INTEGER,
        ticker TEXT,
        units REAL,
        avg_buy_price REAL,
        total_invested REAL,
        FOREIGN KEY (portfolio_id) REFERENCES portfolios (portfolio_id)
    )
    """)

    # 5. Performance (daily snapshot)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS performance (
        perf_id INTEGER PRIMARY KEY AUTOINCREMENT,
        portfolio_id INTEGER,
        date TEXT,
        total_value REAL,
        FOREIGN KEY (portfolio_id) REFERENCES portfolios (portfolio_id)
    )
    """)

    conn.commit()
    conn.close()


# -------------------------
# Portfolio Management
# -------------------------
def create_portfolio(name="Master", type="longterm", corpus=0):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO portfolios (name, type, cash_balance)
        VALUES (?, ?, ?)
    """, (name, type, corpus))
    portfolio_id = cursor.lastrowid

    # Add initial corpus as cashflow entry
    if corpus > 0:
        cursor.execute("""
            INSERT INTO cash_flows (portfolio_id, date, amount, note)
            VALUES (?, ?, ?, ?)
        """, (portfolio_id, datetime.today().strftime("%Y-%m-%d"), corpus, "Initial corpus"))

    conn.commit()
    conn.close()
    return portfolio_id


def modify_cash_balance(portfolio_id, amount, date=None, note=None):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    if date is None:
        date = datetime.today().strftime("%Y-%m-%d")

    # Update cash balance
    cursor.execute("UPDATE portfolios SET cash_balance = cash_balance + ? WHERE portfolio_id = ?", 
                   (amount, portfolio_id))

    # Log in cash_flows
    cursor.execute("""
        INSERT INTO cash_flows (portfolio_id, date, amount, note)
        VALUES (?, ?, ?, ?)
    """, (portfolio_id, date, amount, note))

    conn.commit()
    conn.close()


def get_cash_flows(portfolio_id):
    conn = sqlite3.connect(DB_PATH)
    df = None
    try:
        import pandas as pd
        df = pd.read_sql_query("""
            SELECT * FROM cash_flows WHERE portfolio_id = ? ORDER BY date ASC
        """, conn, params=(portfolio_id,))
    finally:
        conn.close()
    return df
