from __future__ import annotations

import sqlite3
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from config import AppConfig
from core.mt5_client import create_mt5_client

# --- Page config ---
st.set_page_config(
    page_title="XAU/USD Scalping Dashboard",
    page_icon="\U0001f4b0",
    layout="wide",
)

st.title("XAU/USD Scalping Dashboard")


# --- Data loading ---
@st.cache_resource
def get_config() -> AppConfig:
    return AppConfig()


@st.cache_data(ttl=30)
def load_trades(db_path: str) -> pd.DataFrame:
    """Load all trades from the database."""
    try:
        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query("SELECT * FROM trades ORDER BY opened_at DESC", conn)
        conn.close()
        if not df.empty:
            df["opened_at"] = pd.to_datetime(df["opened_at"])
            df["closed_at"] = pd.to_datetime(df["closed_at"], errors="coerce")
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=10)
def load_candles(symbol: str, timeframe: str, count: int) -> pd.DataFrame:
    """Load candle data from MT5 client."""
    config = get_config()
    client = create_mt5_client(config)
    client.connect()
    try:
        df = client.get_rates(symbol, timeframe, count)
        return df
    finally:
        client.disconnect()


# --- Sidebar controls ---
with st.sidebar:
    st.header("Controls")
    candle_count = st.slider("Candles to display", 50, 500, 200, step=50)
    refresh_interval = st.selectbox("Auto-refresh", ["Off", "10s", "30s", "60s"], index=0)

# Handle auto-refresh with a placeholder that counts down
if refresh_interval != "Off":
    seconds = int(refresh_interval.replace("s", ""))
    refresh_placeholder = st.sidebar.empty()
    for remaining in range(seconds, 0, -1):
        refresh_placeholder.caption(f"Refreshing in {remaining}s...")
        time.sleep(1)
    refresh_placeholder.empty()
    st.rerun()

config = get_config()

# --- Load data ---
candles = load_candles(config.mt5.symbol, config.mt5.timeframe, candle_count)
trades_df = load_trades(config.db_path)

# --- Compute metrics ---
closed_trades = trades_df[trades_df["status"] != "open"] if not trades_df.empty else pd.DataFrame()
open_trades = trades_df[trades_df["status"] == "open"] if not trades_df.empty else pd.DataFrame()

total_pnl = closed_trades["pnl"].sum() if not closed_trades.empty else 0.0
total_trades_count = len(closed_trades)
winning = closed_trades[closed_trades["pnl"] > 0] if not closed_trades.empty else pd.DataFrame()
win_rate = (len(winning) / total_trades_count * 100) if total_trades_count > 0 else 0.0

# Max drawdown from equity curve
if not closed_trades.empty:
    equity_curve = closed_trades["pnl"].cumsum()
    running_max = equity_curve.cummax()
    drawdown = running_max - equity_curve
    max_drawdown = drawdown.max()
else:
    max_drawdown = 0.0

# Balance (mock starting balance + P&L)
client_temp = create_mt5_client(config)
client_temp.connect()
account_info = client_temp.get_account_info()
client_temp.disconnect()
balance = account_info["balance"] + total_pnl

# --- KPI Row ---
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("Balance", f"${balance:,.2f}")
with col2:
    st.metric("Total P&L", f"${total_pnl:,.2f}", delta=f"{total_pnl:+,.2f}")
with col3:
    st.metric("Max Drawdown", f"${max_drawdown:,.2f}")
with col4:
    st.metric("Win Rate", f"{win_rate:.1f}%")
with col5:
    st.metric("Closed Trades", str(total_trades_count))

st.divider()

# --- Candlestick Chart ---
st.subheader(f"{config.mt5.symbol} - {config.mt5.timeframe} Candles")

if not candles.empty:
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.8, 0.2],
        subplot_titles=("Price", "Volume"),
    )

    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=candles["datetime"],
            open=candles["open"],
            high=candles["high"],
            low=candles["low"],
            close=candles["close"],
            name="XAUUSD",
            increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350",
        ),
        row=1, col=1,
    )

    # Trade markers on chart
    if not closed_trades.empty:
        buy_trades = closed_trades[closed_trades["direction"].str.contains("BUY", case=False, na=False)]
        sell_trades = closed_trades[closed_trades["direction"].str.contains("SELL", case=False, na=False)]

        if not buy_trades.empty:
            fig.add_trace(
                go.Scatter(
                    x=buy_trades["opened_at"],
                    y=buy_trades["entry_price"],
                    mode="markers",
                    marker=dict(symbol="triangle-up", size=12, color="#26a69a"),
                    name="Buy Entry",
                ),
                row=1, col=1,
            )
        if not sell_trades.empty:
            fig.add_trace(
                go.Scatter(
                    x=sell_trades["opened_at"],
                    y=sell_trades["entry_price"],
                    mode="markers",
                    marker=dict(symbol="triangle-down", size=12, color="#ef5350"),
                    name="Sell Entry",
                ),
                row=1, col=1,
            )

    # Volume bars
    colors = [
        "#26a69a" if c >= o else "#ef5350"
        for c, o in zip(candles["close"], candles["open"])
    ]
    fig.add_trace(
        go.Bar(
            x=candles["datetime"],
            y=candles["volume"],
            marker_color=colors,
            name="Volume",
            showlegend=False,
        ),
        row=2, col=1,
    )

    fig.update_layout(
        height=600,
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        margin=dict(l=50, r=50, t=30, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(type="category", nticks=20, row=2, col=1)

    st.plotly_chart(fig, width="stretch")
else:
    st.warning("No candle data available")

# --- Equity Curve ---
if not closed_trades.empty:
    st.subheader("Equity Curve")
    eq_df = closed_trades.sort_values("opened_at").copy()
    eq_df["cumulative_pnl"] = eq_df["pnl"].cumsum()

    eq_fig = go.Figure()
    eq_fig.add_trace(
        go.Scatter(
            x=eq_df["opened_at"],
            y=eq_df["cumulative_pnl"],
            mode="lines+markers",
            fill="tozeroy",
            line=dict(color="#26a69a", width=2),
            marker=dict(size=4),
            name="Equity",
        )
    )
    eq_fig.update_layout(
        height=300,
        template="plotly_dark",
        yaxis_title="Cumulative P&L ($)",
        margin=dict(l=50, r=50, t=10, b=30),
    )
    st.plotly_chart(eq_fig, width="stretch")

st.divider()

# --- Closed Trades Table ---
st.subheader("Closed Trades")

if not closed_trades.empty:
    display_df = closed_trades[[
        "id", "strategy", "direction", "entry_price", "exit_price",
        "sl", "tp", "lot_size", "pnl", "opened_at", "closed_at", "status",
    ]].copy()

    # Color PnL column
    st.dataframe(
        display_df.style.applymap(
            lambda v: "color: #26a69a" if isinstance(v, (int, float)) and v > 0
            else ("color: #ef5350" if isinstance(v, (int, float)) and v < 0 else ""),
            subset=["pnl"],
        ),
        width="stretch",
        height=400,
    )
else:
    st.info("No closed trades yet. Run the bot in dry-run or live mode to generate trades.")

# --- Open Trades ---
if not open_trades.empty:
    st.subheader("Open Trades")
    st.dataframe(
        open_trades[[
            "id", "strategy", "direction", "entry_price", "sl", "tp",
            "lot_size", "opened_at", "status",
        ]],
        width="stretch",
    )

# --- Footer ---
st.divider()
st.caption(f"Last updated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')} | Mode: {config.mode}")
