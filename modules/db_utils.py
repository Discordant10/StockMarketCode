# modules/db_utils.py

import os
import sqlite3
from config import DB_PATH

# Ensure directories exist
os.makedirs("db", exist_ok=True)
os.makedirs("inputs", exist_ok=True)

def init_db():
    """Initialize SQLite DB with required tables if not exist."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS equities (
        ticker TEXT,
        date TEXT,
        open REAL,
        high REAL,
        low REAL,
        close REAL,
        volume INTEGER,
        PRIMARY KEY (ticker, date)
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS fo_data (
        ticker TEXT,
        date TEXT,
        open_interest REAL,
        volume REAL,
        PRIMARY KEY (ticker, date)
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS mf_data (
        fund TEXT,
        date TEXT,
        nav REAL,
        PRIMARY KEY (fund, date)
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS block_deals (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT,
        date TEXT,
        buy_sell TEXT,
        volume REAL,
        price REAL
    )
    """)

    conn.commit()
    conn.close()
