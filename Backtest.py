import os
from dotenv import load_dotenv
import time
import requests
import pandas as pd
import hmac
import hashlib
import json

# Load environment variables
load_dotenv()

api_key = os.getenv("API_KEY")
secret_key = os.getenv("SECRET_KEY")
base_url = "https://fapi.pi42.com"

# Constants
symbol = "BTCUSDT"
timeframe = "1d"
bollinger_period = 20
bollinger_std_dev = 2
rsi_period = 14
paper_capital = 10000  # Initial paper capital
profit_target_percent = 0.002  # 0.2% profit
stop_loss_percent = 0.005  # 0.5% stop loss
max_holding_candles = 15  # Exit after 15 candles if no mean reversion

def get_kline_data(symbol, timeframe, limit=500):
    """Fetch kline data and return as a pandas DataFrame."""
    params = {
        "pair": symbol.upper(),
        "interval": timeframe.lower(),
        "limit": limit
    }
    headers = {"Content-Type": "application/json"}
    kline_url = "https://api.pi42.com/v1/market/klines"

    response = requests.post(kline_url, json=params, headers=headers)
    response.raise_for_status()
    data = response.json()
    df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].astype(float)
    return df

def calculate_rsi(df, period=14):
    df["delta"] = df["close"].diff()
    df["gain"] = df["delta"].clip(lower=0)
    df["loss"] = -df["delta"].clip(upper=0)
    df["avg_gain"] = df["gain"].rolling(window=period).mean()
    df["avg_loss"] = df["loss"].rolling(window=period).mean()
    df["rs"] = df["avg_gain"] / df["avg_loss"]
    df["rsi"] = 100 - (100 / (1 + df["rs"]))
    return df.drop(["delta", "gain", "loss", "avg_gain", "avg_loss", "rs"], axis=1)

def calculate_bollinger_bands(df, period=20, std_dev=2):
    df["mean"] = df["close"].rolling(window=period).mean()
    df["std"] = df["close"].rolling(window=period).std()
    df["upper_band"] = df["mean"] + (std_dev * df["std"])
    df["lower_band"] = df["mean"] - (std_dev * df["std"])
    return df

def calculate_intraday_vwap(df):
    """
    Calculate intraday VWAP, resetting at the start of each trading day.
    Assumes the DataFrame has a 'timestamp' column.
    """
    df["date"] = df["timestamp"].dt.date  # Extract date from timestamp
    df["cum_price_volume"] = (df["close"] * df["volume"]).groupby(df["date"]).cumsum()
    df["cum_volume"] = df["volume"].groupby(df["date"]).cumsum()
    df["vwap"] = df["cum_price_volume"] / df["cum_volume"]
    return df.drop(["cum_price_volume", "cum_volume", "date"], axis=1)

def simulate_trades(df):
    trades = []
    capital = paper_capital
    position = 0
    entry_price = 0
    entry_time = None

    for i in range(bollinger_period, len(df)):
        price = df["close"].iloc[i]
        mean = df["mean"].iloc[i]
        lower_band = df["lower_band"].iloc[i]
        upper_band = df["upper_band"].iloc[i]
        rsi = df["rsi"].iloc[i]
        vwap = df["vwap"].iloc[i]

        # Entry Rules
        if position == 0:  # No position
            trade_capital = capital * 0.3  # 30% of current capital
            position_size = trade_capital / price

            if price <= lower_band and rsi < 30:
                # Long Entry (Above VWAP)
                entry_price = price
                position = position_size
                entry_time = i
                trades.append({
                    "action": "BUY",
                    "price": entry_price,
                    "capital": capital
                })

            elif price >= upper_band and rsi > 70:
                # Short Entry (Below VWAP)
                entry_price = price
                position = -position_size
                entry_time = i
                trades.append({
                    "action": "SELL",
                    "price": entry_price,
                    "capital": capital
                })

        # Exit Rules
        elif position != 0:
            is_long = position > 0
            profit_target = entry_price * (1 + profit_target_percent if is_long else 1 - profit_target_percent)
            stop_loss = entry_price * (1 - stop_loss_percent if is_long else 1 + stop_loss_percent)

            # Profit Target
            if (is_long and price >= profit_target) or (not is_long and price <= profit_target):
                profit = abs(position) * (price - entry_price if is_long else entry_price - price)
                capital += profit
                trades.append({
                    "action": "CLOSE",
                    "price": price,
                    "capital": capital
                })
                position = 0

            # Stop Loss
            elif (is_long and price <= stop_loss) or (not is_long and price >= stop_loss):
                loss = abs(position) * (price - entry_price if is_long else entry_price - price)
                capital += loss
                trades.append({
                    "action": "STOP LOSS",
                    "price": price,
                    "capital": capital
                })
                position = 0

            # Time-Based Exit
            elif i - entry_time >= max_holding_candles:
                time_exit_profit = abs(position) * (price - entry_price if is_long else entry_price - price)
                capital += time_exit_profit
                trades.append({
                    "action": "TIME EXIT",
                    "price": price,
                    "capital": capital
                })
                position = 0

    # Save trades to CSV without the timestamp
    trades_df = pd.DataFrame(trades)
    trades_df.to_csv("trades1d.csv", index=False)
    print("Trades saved to trades1d.csv")
    
    # Calculate overall percentage profit/loss
    profit_loss_percent = ((capital - paper_capital) / paper_capital) * 100
    print(f"Overall Percentage Profit/Loss: {profit_loss_percent:.2f}%")

def main():
    df = get_kline_data(symbol, timeframe)
    df = calculate_rsi(df, rsi_period)
    df = calculate_bollinger_bands(df, bollinger_period, bollinger_std_dev)
    df = calculate_intraday_vwap(df)
    simulate_trades(df)

if __name__ == "__main__":
    main()
