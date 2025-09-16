# main.py

import os
from config import WATCHLIST_FILE, LOOKBACK_YEARS
from modules import init_db, read_watchlist, download_equity_data
from modules import portfolio_module as pm


def main():
    print("🚀 Initializing Stock Market Forecasting Project...")

    # ------------------ Equity Data Setup ------------------
    init_db()

    # Load watchlist
    watchlist_path = os.path.join("inputs", "tickers.csv")
    equity_watchlist = read_watchlist(watchlist_path)
    print(f"📈 Found {len(equity_watchlist)} equity tickers.")

    # Download equity data
    download_equity_data(equity_watchlist, lookback_years=LOOKBACK_YEARS)
    print("✅ Equity data download complete.")

    # ------------------ Portfolio Setup ------------------
    pm.init_portfolio_db()

    # # Create portfolio with initial cash
    # pid = pm.create_portfolio("Master", "longterm", 100000)
    # print(f"💼 Portfolio created with ID: {pid}")

    # # Add cash
    # pm.modify_cash_balance(pid, 25000, note="Top-up")

    # # Withdraw cash
    # pm.modify_cash_balance(pid, -5000, note="Personal withdrawal")

    # # Show cash flows
    # print("📊 Cash Flow Log:")
    # print(pm.get_cash_flows(pid))


if __name__ == "__main__":
    main()
