#!/usr/bin/env python3
"""
TradingBot MCP Server — Real-time monitoring, trade analysis & diagnostics.

Gives Claude direct introspection into:
  - Live bot status, account info, open positions
  - All historical trades with filtering
  - Per-strategy and per-model performance stats
  - RL model comparison and training status
  - Automated diagnostics (overtrading, bad win rate, drawdown, etc.)

Run:  python3 mcp_server.py
Or:   mcp run mcp_server.py
"""

from __future__ import annotations

import json
import math
import sqlite3
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
from mcp.server.fastmcp import FastMCP

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

mcp = FastMCP(
    "TradingBot",
    instructions="Real-time monitoring and analysis for the XAU/USD trading bot",
    port=8100,
)

DB_PATH = Path(__file__).parent / "trades.db"
API_BASE = "http://localhost:8000"


def _db() -> sqlite3.Connection:
    """Open a read-only connection to the trades database."""
    conn = sqlite3.connect(str(DB_PATH), timeout=5)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


async def _api(path: str, method: str = "GET", body: dict | None = None) -> dict:
    """Call the FastAPI backend."""
    async with httpx.AsyncClient(base_url=API_BASE, timeout=10) as client:
        if method == "GET":
            r = await client.get(path)
        else:
            r = await client.post(path, json=body or {})
        r.raise_for_status()
        return r.json()


# ===================================================================
# LIVE STATUS
# ===================================================================

@mcp.tool()
async def bot_status() -> str:
    """Get the current bot status: running mode, account info, active model,
    open positions count, and risk config."""
    try:
        status = await _api("/api/bot/status")
    except Exception as e:
        return f"ERROR: Cannot reach bot API at {API_BASE} — {e}"

    # Get active model info
    try:
        metrics = await _api("/api/learning/training-metrics")
    except Exception:
        metrics = {}

    lines = [
        "=== BOT STATUS ===",
        f"Running: {status.get('running', '?')}",
        f"Mode: {status.get('mode', '?')}",
        f"Symbol: {status.get('symbol', '?')}",
        f"Timeframe: {status.get('timeframe', '?')}",
        f"Strategy mode: {status.get('strategy_mode', '?')}",
        f"Enabled strategies: {', '.join(status.get('enabled_strategies', []))}",
        "",
        "--- Account ---",
        f"Balance: ${status.get('account', {}).get('balance', 0):,.2f}",
        f"Equity: ${status.get('account', {}).get('equity', 0):,.2f}",
        f"Profit: ${status.get('account', {}).get('profit', 0):,.2f}",
        "",
        f"Open positions: {status.get('open_positions_count', '?')}",
    ]

    if metrics:
        lines += [
            "",
            "--- Active Model ---",
            f"Readiness: {metrics.get('readiness', '?')}",
            f"Model ID: {metrics.get('active_model_id', 'None')}",
            f"Model name: {metrics.get('active_model_name', 'None')}",
            f"Profile: {metrics.get('active_model_profile', 'None')}",
            f"Timeframe: {metrics.get('active_model_timeframe', 'None')}",
            f"Episode: {metrics.get('episode', 0)}",
            f"Epsilon: {metrics.get('epsilon', 0):.4f}",
            f"Win rate: {metrics.get('win_rate', 0) * 100:.1f}%",
        ]

    return "\n".join(lines)


@mcp.tool()
async def open_positions() -> str:
    """Get all currently open positions with entry price, SL, TP, current P&L,
    and how long they've been open."""
    conn = _db()
    try:
        rows = conn.execute(
            """SELECT * FROM trades
               WHERE status IN ('open', 'dry-run') AND closed_at IS NULL
               ORDER BY id DESC"""
        ).fetchall()

        if not rows:
            return "No open positions."

        lines = [f"=== {len(rows)} OPEN POSITION(S) ===", ""]
        now = datetime.utcnow()
        for r in rows:
            r = dict(r)
            opened = r.get("opened_at", "")
            duration = ""
            if opened:
                try:
                    dt = datetime.fromisoformat(opened.replace("Z", "+00:00"))
                    mins = (now - dt.replace(tzinfo=None)).total_seconds() / 60
                    duration = f"{int(mins)}m"
                except Exception:
                    pass
            lines.append(
                f"#{r['id']} | {r['direction']} {r['symbol']} @ {r['entry_price']:.2f} | "
                f"SL {r['sl']:.2f} | TP {r['tp']:.2f} | {r['lot_size']} lots | "
                f"Strategy: {r['strategy']} | Open {duration}"
            )
        return "\n".join(lines)
    finally:
        conn.close()


@mcp.tool()
async def account_info() -> str:
    """Get detailed account information from the broker."""
    try:
        status = await _api("/api/bot/status")
        acct = status.get("account", {})
        return (
            f"Balance: ${acct.get('balance', 0):,.2f}\n"
            f"Equity: ${acct.get('equity', 0):,.2f}\n"
            f"Margin: ${acct.get('margin', 0):,.2f}\n"
            f"Free margin: ${acct.get('free_margin', 0):,.2f}\n"
            f"Profit: ${acct.get('profit', 0):,.2f}\n"
            f"Leverage: {acct.get('leverage', 0)}"
        )
    except Exception as e:
        return f"ERROR: {e}"


# ===================================================================
# TRADE ANALYSIS
# ===================================================================

@mcp.tool()
async def get_trades(
    status: str = "all",
    strategy: str = "all",
    direction: str = "all",
    last_n: int = 50,
    date: str = "",
) -> str:
    """Query trades from the database.

    Args:
        status: Filter by status — "open", "closed", "dry-run", or "all"
        strategy: Filter by strategy name — "ema_crossover", "bos", "asian_breakout", "candle_pattern", or "all"
        direction: Filter by direction — "BUY", "SELL", or "all"
        last_n: Number of most recent trades to return (max 200)
        date: Filter by date (YYYY-MM-DD). Empty = all dates.
    """
    conn = _db()
    try:
        conditions = []
        params: list[Any] = []

        if status != "all":
            conditions.append("status = ?")
            params.append(status)
        if strategy != "all":
            conditions.append("strategy = ?")
            params.append(strategy)
        if direction != "all":
            conditions.append("direction = ?")
            params.append(direction)
        if date:
            conditions.append("opened_at LIKE ?")
            params.append(f"{date}%")

        where = " AND ".join(conditions) if conditions else "1=1"
        last_n = min(last_n, 200)
        params.append(last_n)

        rows = conn.execute(
            f"SELECT * FROM trades WHERE {where} ORDER BY id DESC LIMIT ?",
            params,
        ).fetchall()

        if not rows:
            return "No trades found matching the filters."

        lines = [f"=== {len(rows)} TRADE(S) ===", ""]
        for r in rows:
            r = dict(r)
            pnl_str = f"${r['pnl']:+.2f}" if r.get("pnl") else "open"
            exit_str = f"{r['exit_price']:.2f}" if r.get("exit_price") else "—"
            lines.append(
                f"#{r['id']} [{r['status']}] {r['direction']} {r['symbol']} | "
                f"Entry {r['entry_price']:.2f} -> Exit {exit_str} | "
                f"SL {r['sl']:.2f} TP {r['tp']:.2f} | {r['lot_size']} lots | "
                f"P&L: {pnl_str} | {r['strategy']} | {r['opened_at'][:16]}"
            )
        return "\n".join(lines)
    finally:
        conn.close()


@mcp.tool()
async def trade_stats(days: int = 0, strategy: str = "all") -> str:
    """Calculate trading performance statistics.

    Args:
        days: Look back N days (0 = all time)
        strategy: Filter by strategy name, or "all"

    Returns win rate, profit factor, avg win/loss, Sharpe ratio, max drawdown,
    total P&L, trade count, avg trade duration.
    """
    conn = _db()
    try:
        conditions = ["status = 'closed'"]
        params: list[Any] = []

        if days > 0:
            cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()
            conditions.append("closed_at >= ?")
            params.append(cutoff)
        if strategy != "all":
            conditions.append("strategy = ?")
            params.append(strategy)

        where = " AND ".join(conditions)
        rows = conn.execute(
            f"SELECT * FROM trades WHERE {where} ORDER BY id", params
        ).fetchall()

        if not rows:
            return "No closed trades found."

        pnls = [r["pnl"] for r in rows]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]
        total_pnl = sum(pnls)
        win_rate = len(wins) / len(pnls) * 100 if pnls else 0
        avg_win = sum(wins) / len(wins) if wins else 0
        avg_loss = sum(losses) / len(losses) if losses else 0
        profit_factor = sum(wins) / abs(sum(losses)) if losses else float("inf")

        # Sharpe ratio (daily)
        if len(pnls) > 1:
            mean_pnl = total_pnl / len(pnls)
            std_pnl = (sum((p - mean_pnl) ** 2 for p in pnls) / len(pnls)) ** 0.5
            sharpe = (mean_pnl / std_pnl * math.sqrt(252)) if std_pnl > 0 else 0
        else:
            sharpe = 0

        # Max drawdown
        equity = 0
        peak = 0
        max_dd = 0
        for p in pnls:
            equity += p
            if equity > peak:
                peak = equity
            dd = peak - equity
            if dd > max_dd:
                max_dd = dd

        # Avg trade duration
        durations = []
        for r in rows:
            if r["opened_at"] and r["closed_at"]:
                try:
                    opened = datetime.fromisoformat(r["opened_at"].replace("Z", "+00:00"))
                    closed = datetime.fromisoformat(r["closed_at"].replace("Z", "+00:00"))
                    durations.append((closed - opened).total_seconds())
                except Exception:
                    pass
        avg_duration_min = (sum(durations) / len(durations) / 60) if durations else 0

        # Consecutive wins/losses
        max_consec_wins = max_consec_losses = consec = 0
        last_dir = None
        for p in pnls:
            d = "W" if p > 0 else "L"
            if d == last_dir:
                consec += 1
            else:
                consec = 1
                last_dir = d
            if d == "W":
                max_consec_wins = max(max_consec_wins, consec)
            else:
                max_consec_losses = max(max_consec_losses, consec)

        period = f"Last {days} days" if days > 0 else "All time"
        strat_label = strategy if strategy != "all" else "All strategies"

        return (
            f"=== TRADE STATS ({period} | {strat_label}) ===\n"
            f"\n"
            f"Total trades: {len(pnls)}\n"
            f"Wins: {len(wins)} | Losses: {len(losses)} | Breakeven: {len(pnls) - len(wins) - len(losses)}\n"
            f"Win rate: {win_rate:.1f}%\n"
            f"Profit factor: {profit_factor:.2f}\n"
            f"\n"
            f"Total P&L: ${total_pnl:+,.2f}\n"
            f"Avg win: ${avg_win:+,.2f}\n"
            f"Avg loss: ${avg_loss:+,.2f}\n"
            f"Avg win/loss ratio: {abs(avg_win / avg_loss):.2f}x\n" if avg_loss != 0 else ""
            f"Sharpe (annualized): {sharpe:.2f}\n"
            f"\n"
            f"Max drawdown: ${max_dd:,.2f}\n"
            f"Max consecutive wins: {max_consec_wins}\n"
            f"Max consecutive losses: {max_consec_losses}\n"
            f"\n"
            f"Avg trade duration: {avg_duration_min:.1f} min\n"
        )
    finally:
        conn.close()


@mcp.tool()
async def strategy_performance(days: int = 0) -> str:
    """Compare performance across all strategies.

    Args:
        days: Look back N days (0 = all time)

    Shows trade count, win rate, total P&L, avg P&L, profit factor for each strategy.
    """
    conn = _db()
    try:
        conditions = ["status = 'closed'"]
        params: list[Any] = []
        if days > 0:
            cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()
            conditions.append("closed_at >= ?")
            params.append(cutoff)

        where = " AND ".join(conditions)
        rows = conn.execute(
            f"SELECT * FROM trades WHERE {where} ORDER BY id", params
        ).fetchall()

        if not rows:
            return "No closed trades found."

        by_strat: dict[str, list[float]] = defaultdict(list)
        for r in rows:
            by_strat[r["strategy"]].append(r["pnl"])

        lines = [
            f"=== STRATEGY PERFORMANCE ({'Last ' + str(days) + ' days' if days else 'All time'}) ===",
            "",
            f"{'Strategy':<20} {'Trades':>6} {'WR':>6} {'P&L':>10} {'Avg':>8} {'PF':>6}",
            "-" * 60,
        ]

        for strat in sorted(by_strat, key=lambda s: sum(by_strat[s]), reverse=True):
            pnls = by_strat[strat]
            wins = [p for p in pnls if p > 0]
            losses = [p for p in pnls if p < 0]
            wr = len(wins) / len(pnls) * 100
            total = sum(pnls)
            avg = total / len(pnls)
            pf = sum(wins) / abs(sum(losses)) if losses else float("inf")
            pf_str = f"{pf:.2f}" if pf < 100 else "inf"
            lines.append(
                f"{strat:<20} {len(pnls):>6} {wr:>5.1f}% ${total:>+9.2f} ${avg:>+7.2f} {pf_str:>6}"
            )

        lines.append("-" * 60)
        total_pnl = sum(r["pnl"] for r in rows)
        total_wr = sum(1 for r in rows if r["pnl"] > 0) / len(rows) * 100
        lines.append(f"{'TOTAL':<20} {len(rows):>6} {total_wr:>5.1f}% ${total_pnl:>+9.2f}")

        return "\n".join(lines)
    finally:
        conn.close()


@mcp.tool()
async def daily_summary(days: int = 7) -> str:
    """Show daily P&L, trade count, and win rate for the last N days.

    Args:
        days: Number of days to show (default 7)
    """
    conn = _db()
    try:
        cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()
        rows = conn.execute(
            "SELECT * FROM trades WHERE status = 'closed' AND closed_at >= ? ORDER BY closed_at",
            (cutoff,),
        ).fetchall()

        if not rows:
            return "No closed trades in the last {days} days."

        by_day: dict[str, list[float]] = defaultdict(list)
        for r in rows:
            day = r["closed_at"][:10] if r["closed_at"] else "unknown"
            by_day[day].append(r["pnl"])

        lines = [
            f"=== DAILY SUMMARY (Last {days} days) ===",
            "",
            f"{'Date':<12} {'Trades':>6} {'WR':>6} {'P&L':>10} {'Cum P&L':>10}",
            "-" * 50,
        ]

        cum_pnl = 0
        for day in sorted(by_day):
            pnls = by_day[day]
            wins = sum(1 for p in pnls if p > 0)
            wr = wins / len(pnls) * 100
            total = sum(pnls)
            cum_pnl += total
            lines.append(
                f"{day:<12} {len(pnls):>6} {wr:>5.1f}% ${total:>+9.2f} ${cum_pnl:>+9.2f}"
            )

        return "\n".join(lines)
    finally:
        conn.close()


@mcp.tool()
async def hourly_trade_distribution(days: int = 7) -> str:
    """Show how many trades were opened in each hour of the day.
    Helps identify overtrading during specific hours.

    Args:
        days: Look back N days (0 = all time)
    """
    conn = _db()
    try:
        conditions = ["status = 'closed'"]
        params: list[Any] = []
        if days > 0:
            cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()
            conditions.append("opened_at >= ?")
            params.append(cutoff)

        where = " AND ".join(conditions)
        rows = conn.execute(
            f"SELECT * FROM trades WHERE {where}", params
        ).fetchall()

        if not rows:
            return "No trades found."

        by_hour: dict[int, list[float]] = defaultdict(list)
        for r in rows:
            try:
                hour = int(r["opened_at"][11:13])
                by_hour[hour].append(r["pnl"])
            except (ValueError, IndexError):
                pass

        lines = [
            f"=== HOURLY TRADE DISTRIBUTION ===",
            "",
            f"{'Hour':<6} {'Trades':>6} {'WR':>6} {'P&L':>10} {'Avg':>8}",
            "-" * 40,
        ]

        for hour in range(24):
            if hour not in by_hour:
                continue
            pnls = by_hour[hour]
            wins = sum(1 for p in pnls if p > 0)
            wr = wins / len(pnls) * 100
            total = sum(pnls)
            avg = total / len(pnls)
            bar = "#" * min(len(pnls), 40)
            lines.append(
                f"{hour:02d}:00 {len(pnls):>6} {wr:>5.1f}% ${total:>+9.2f} ${avg:>+7.2f}  {bar}"
            )

        return "\n".join(lines)
    finally:
        conn.close()


# ===================================================================
# MODEL ANALYSIS
# ===================================================================

@mcp.tool()
async def list_models(top_n: int = 20) -> str:
    """List all saved RL models with their metrics (without loading model blobs).

    Args:
        top_n: Max number of models to return (default 20)
    """
    conn = _db()
    try:
        rows = conn.execute(
            """SELECT id, model_name, episode, epsilon, total_reward, win_rate,
                      profile, timeframe, created_at
               FROM rl_models ORDER BY id DESC LIMIT ?""",
            (top_n,),
        ).fetchall()

        if not rows:
            return "No saved models."

        lines = [
            f"=== {len(rows)} SAVED RL MODEL(S) ===",
            "",
            f"{'ID':>4} {'Name':<25} {'Episodes':>8} {'Eps':>6} {'WR':>6} {'Reward':>10} {'Profile':<12} {'TF':<4} {'Created':<16}",
            "-" * 100,
        ]

        for r in rows:
            r = dict(r)
            wr = r.get("win_rate", 0) or 0
            lines.append(
                f"{r['id']:>4} {(r.get('model_name') or '?'):<25} "
                f"{r.get('episode', 0):>8} "
                f"{r.get('epsilon', 0):>6.4f} "
                f"{wr * 100:>5.1f}% "
                f"{r.get('total_reward', 0):>+10.2f} "
                f"{(r.get('profile') or '-'):<12} "
                f"{(r.get('timeframe') or '-'):<4} "
                f"{(r.get('created_at') or '')[:16]}"
            )

        return "\n".join(lines)
    finally:
        conn.close()


@mcp.tool()
async def model_comparison() -> str:
    """Rank all models by win rate and identify the best performer.
    Groups by profile/timeframe and shows the top model in each group."""
    conn = _db()
    try:
        rows = conn.execute(
            """SELECT id, model_name, episode, epsilon, total_reward, win_rate,
                      profile, timeframe, created_at
               FROM rl_models ORDER BY win_rate DESC"""
        ).fetchall()

        if not rows:
            return "No saved models."

        # Group by profile+timeframe
        groups: dict[str, list[dict]] = defaultdict(list)
        for r in rows:
            r = dict(r)
            key = f"{r.get('profile') or 'default'}_{r.get('timeframe') or 'unknown'}"
            groups[key].append(r)

        lines = ["=== MODEL COMPARISON ===", ""]

        # Overall top 5
        lines.append("Top 5 by Win Rate:")
        for r in [dict(r) for r in rows[:5]]:
            wr = (r.get("win_rate") or 0) * 100
            lines.append(
                f"  #{r['id']} {r.get('model_name', '?')} | WR {wr:.1f}% | "
                f"eps {r.get('epsilon', 0):.4f} | episodes {r.get('episode', 0)} | "
                f"{r.get('profile', '-')} {r.get('timeframe', '-')}"
            )

        # Best per group
        lines += ["", "Best per Profile/Timeframe:"]
        for key in sorted(groups):
            best = groups[key][0]  # already sorted by WR desc
            wr = (best.get("win_rate") or 0) * 100
            lines.append(
                f"  {key}: #{best['id']} WR {wr:.1f}% | episodes {best.get('episode', 0)} | "
                f"reward {best.get('total_reward', 0):+.2f}"
            )

        return "\n".join(lines)
    finally:
        conn.close()


@mcp.tool()
async def training_status() -> str:
    """Get current RL training status: readiness, active model, epsilon,
    episode count, and pipeline training progress if running."""
    try:
        metrics = await _api("/api/learning/training-metrics")
    except Exception as e:
        return f"ERROR: Cannot reach API — {e}"

    lines = [
        "=== TRAINING STATUS ===",
        "",
        f"Readiness: {metrics.get('readiness', '?')}",
        f"Episode: {metrics.get('episode', 0)}",
        f"Epsilon: {metrics.get('epsilon', 0):.4f}",
        f"Win rate: {metrics.get('win_rate', 0) * 100:.1f}%",
        f"Total reward: {metrics.get('total_reward', 0):.2f}",
        "",
        f"Active model ID: {metrics.get('active_model_id', 'None')}",
        f"Active model name: {metrics.get('active_model_name', 'None')}",
        f"Active model profile: {metrics.get('active_model_profile', 'None')}",
        f"Active model timeframe: {metrics.get('active_model_timeframe', 'None')}",
    ]

    # Pipeline status
    try:
        pipe = await _api("/api/learning/pipeline-status")
        if pipe.get("running"):
            prog = pipe.get("progress", {}) or {}
            lines += [
                "",
                "--- Pipeline Training ACTIVE ---",
                f"Step {prog.get('step', 0) + 1}/{prog.get('total_steps', '?')} ({prog.get('step_timeframe', '?')})",
                f"Progress: {prog.get('pct', 0):.1f}%",
            ]
            logs = pipe.get("logs", [])
            if logs:
                lines.append(f"Last log: {logs[-1]}")
        else:
            lines.append("\nPipeline: not running")
    except Exception:
        pass

    return "\n".join(lines)


# ===================================================================
# DIAGNOSTICS
# ===================================================================

@mcp.tool()
async def diagnose(days: int = 1) -> str:
    """Auto-diagnose issues with the trading bot.

    Checks for:
    - Overtrading (too many trades per hour/day)
    - Poor win rate (below 45%)
    - Negative P&L trend
    - High drawdown
    - Unbalanced direction (too many buys or sells)
    - Short trade durations (rapid cycling)
    - Strategy underperformance

    Args:
        days: Look back N days (default 1 = today)
    """
    conn = _db()
    try:
        cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()
        rows = conn.execute(
            "SELECT * FROM trades WHERE opened_at >= ? ORDER BY id",
            (cutoff,),
        ).fetchall()

        if not rows:
            return f"No trades in the last {days} day(s) — nothing to diagnose."

        issues: list[str] = []
        warnings: list[str] = []
        info: list[str] = []

        total = len(rows)
        closed = [r for r in rows if r["status"] == "closed"]
        open_trades = [r for r in rows if r["status"] in ("open", "dry-run") and not r["closed_at"]]

        # --- Trade count ---
        info.append(f"Total trades: {total} ({len(closed)} closed, {len(open_trades)} open)")

        # --- Overtrading check ---
        trades_per_hour: dict[str, int] = defaultdict(int)
        for r in rows:
            hour_key = r["opened_at"][:13]  # YYYY-MM-DDTHH
            trades_per_hour[hour_key] += 1

        max_hour = max(trades_per_hour.values()) if trades_per_hour else 0
        max_hour_key = max(trades_per_hour, key=trades_per_hour.get) if trades_per_hour else ""
        avg_per_hour = total / max(len(trades_per_hour), 1)

        if max_hour > 15:
            issues.append(
                f"OVERTRADING: {max_hour} trades in hour {max_hour_key} "
                f"(avg {avg_per_hour:.1f}/hr). This is excessive for M1 scalping."
            )
        elif max_hour > 8:
            warnings.append(
                f"High trade frequency: {max_hour} trades in hour {max_hour_key} "
                f"(avg {avg_per_hour:.1f}/hr)."
            )

        if total > 50 * days:
            issues.append(f"OVERTRADING: {total} trades in {days} day(s) = {total/days:.0f}/day. Way too many.")
        elif total > 20 * days:
            warnings.append(f"High daily count: {total/days:.0f} trades/day.")

        # --- Win rate ---
        if closed:
            pnls = [r["pnl"] for r in closed]
            wins = sum(1 for p in pnls if p > 0)
            wr = wins / len(pnls) * 100
            total_pnl = sum(pnls)

            info.append(f"Win rate: {wr:.1f}% ({wins}/{len(pnls)})")
            info.append(f"Total P&L: ${total_pnl:+,.2f}")

            if wr < 35:
                issues.append(f"VERY LOW WIN RATE: {wr:.1f}% — model may not be suitable for live trading.")
            elif wr < 45:
                warnings.append(f"Low win rate: {wr:.1f}%")

            if total_pnl < -50:
                issues.append(f"SIGNIFICANT LOSS: ${total_pnl:,.2f} in {days} day(s).")

            # Avg win vs avg loss
            avg_win = sum(p for p in pnls if p > 0) / max(wins, 1)
            avg_loss = sum(p for p in pnls if p < 0) / max(len(pnls) - wins, 1)
            if avg_loss != 0:
                rr = abs(avg_win / avg_loss)
                info.append(f"Avg win: ${avg_win:+.2f} | Avg loss: ${avg_loss:+.2f} | R:R = {rr:.2f}")
                if rr < 0.8:
                    warnings.append(f"Poor risk/reward ratio: {rr:.2f}x (avg win too small vs avg loss)")

            # Drawdown
            equity = 0
            peak = 0
            max_dd = 0
            for p in pnls:
                equity += p
                peak = max(peak, equity)
                max_dd = max(max_dd, peak - equity)

            if max_dd > 100:
                issues.append(f"HIGH DRAWDOWN: ${max_dd:,.2f}")
            elif max_dd > 50:
                warnings.append(f"Notable drawdown: ${max_dd:,.2f}")

        # --- Direction balance ---
        buys = sum(1 for r in rows if r["direction"] in ("BUY", "Direction.BUY"))
        sells = total - buys
        if total > 10:
            ratio = max(buys, sells) / total * 100
            if ratio > 80:
                dominant = "BUY" if buys > sells else "SELL"
                warnings.append(
                    f"Direction imbalance: {ratio:.0f}% {dominant} ({buys} buys, {sells} sells). "
                    f"Bot may be biased."
                )

        # --- Trade duration ---
        durations = []
        for r in closed:
            if r["opened_at"] and r["closed_at"]:
                try:
                    opened = datetime.fromisoformat(r["opened_at"].replace("Z", "+00:00"))
                    closed_dt = datetime.fromisoformat(r["closed_at"].replace("Z", "+00:00"))
                    durations.append((closed_dt - opened).total_seconds())
                except Exception:
                    pass

        if durations:
            avg_dur = sum(durations) / len(durations)
            min_dur = min(durations)
            short_trades = sum(1 for d in durations if d < 30)

            info.append(f"Avg trade duration: {avg_dur/60:.1f} min (shortest: {min_dur:.0f}s)")

            if avg_dur < 60:
                issues.append(
                    f"RAPID CYCLING: Avg trade duration is only {avg_dur:.0f}s. "
                    f"SL may be too tight or signals are firing on noise."
                )
            elif short_trades > len(durations) * 0.5:
                warnings.append(
                    f"{short_trades}/{len(durations)} trades lasted under 30 seconds — "
                    f"possible SL too tight."
                )

        # --- Strategy breakdown ---
        by_strat: dict[str, list[float]] = defaultdict(list)
        for r in closed:
            by_strat[r["strategy"]].append(r["pnl"])

        for strat, pnls in by_strat.items():
            s_wr = sum(1 for p in pnls if p > 0) / len(pnls) * 100
            s_pnl = sum(pnls)
            if s_wr < 30 and len(pnls) >= 5:
                issues.append(f"Strategy '{strat}' has {s_wr:.0f}% WR over {len(pnls)} trades (P&L: ${s_pnl:+.2f})")
            elif s_pnl < -20 and len(pnls) >= 5:
                warnings.append(f"Strategy '{strat}' losing: ${s_pnl:+.2f} over {len(pnls)} trades")

        # --- Build report ---
        lines = [f"=== DIAGNOSTICS (Last {days} day(s)) ===", ""]

        if issues:
            lines.append("ISSUES (need attention):")
            for i, issue in enumerate(issues, 1):
                lines.append(f"  {i}. {issue}")
            lines.append("")

        if warnings:
            lines.append("WARNINGS:")
            for i, w in enumerate(warnings, 1):
                lines.append(f"  {i}. {w}")
            lines.append("")

        lines.append("INFO:")
        for i in info:
            lines.append(f"  - {i}")

        if not issues and not warnings:
            lines.append("\nNo issues detected. Bot appears to be operating normally.")

        return "\n".join(lines)
    finally:
        conn.close()


@mcp.tool()
async def trade_pnl_distribution(days: int = 7) -> str:
    """Show the distribution of trade P&L values to identify patterns.
    Groups trades into buckets and shows count per bucket.

    Args:
        days: Look back N days (0 = all time)
    """
    conn = _db()
    try:
        conditions = ["status = 'closed'"]
        params: list[Any] = []
        if days > 0:
            cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()
            conditions.append("closed_at >= ?")
            params.append(cutoff)

        where = " AND ".join(conditions)
        rows = conn.execute(
            f"SELECT pnl FROM trades WHERE {where} ORDER BY pnl", params
        ).fetchall()

        if not rows:
            return "No closed trades."

        pnls = [r["pnl"] for r in rows]
        min_pnl = min(pnls)
        max_pnl = max(pnls)

        # Create buckets
        buckets = [
            ("<-50", lambda p: p < -50),
            ("-50 to -20", lambda p: -50 <= p < -20),
            ("-20 to -10", lambda p: -20 <= p < -10),
            ("-10 to -5", lambda p: -10 <= p < -5),
            ("-5 to 0", lambda p: -5 <= p < 0),
            ("0 (BE)", lambda p: p == 0),
            ("0 to 5", lambda p: 0 < p <= 5),
            ("5 to 10", lambda p: 5 < p <= 10),
            ("10 to 20", lambda p: 10 < p <= 20),
            ("20 to 50", lambda p: 20 < p <= 50),
            (">50", lambda p: p > 50),
        ]

        lines = [
            f"=== P&L DISTRIBUTION ({len(pnls)} trades) ===",
            f"Range: ${min_pnl:+.2f} to ${max_pnl:+.2f}",
            "",
        ]

        for label, cond in buckets:
            count = sum(1 for p in pnls if cond(p))
            if count > 0:
                pct = count / len(pnls) * 100
                bar = "#" * min(count, 50)
                lines.append(f"  {label:>12}: {count:>4} ({pct:>5.1f}%) {bar}")

        return "\n".join(lines)
    finally:
        conn.close()


@mcp.tool()
async def get_config() -> str:
    """Get the current bot configuration — risk settings, strategy params, RL config."""
    try:
        status = await _api("/api/bot/status")
        config = status.get("config", {})
        if config:
            return f"=== CURRENT CONFIG ===\n\n{json.dumps(config, indent=2)}"

        # Fallback: show what we can from status
        return (
            f"=== BOT CONFIG ===\n"
            f"Mode: {status.get('mode', '?')}\n"
            f"Symbol: {status.get('symbol', '?')}\n"
            f"Timeframe: {status.get('timeframe', '?')}\n"
            f"Strategy mode: {status.get('strategy_mode', '?')}\n"
            f"Strategies: {', '.join(status.get('enabled_strategies', []))}\n"
            f"\n(Full config not available via API — check config.py)"
        )
    except Exception as e:
        return f"ERROR: {e}"


@mcp.tool()
async def recent_signals(last_n: int = 20) -> str:
    """Show recent trades with their features, confidence scores, and regime context.

    Args:
        last_n: Number of recent trades to show (default 20)
    """
    conn = _db()
    try:
        rows = conn.execute(
            """SELECT t.*, tf.regime as trade_regime, tf.confidence as trade_confidence,
                      tf.rl_action, tf.claude_reasoning
               FROM trades t
               LEFT JOIN trade_features tf ON t.id = tf.trade_id
               ORDER BY t.id DESC LIMIT ?""",
            (last_n,),
        ).fetchall()

        if not rows:
            return "No trades found."

        lines = [f"=== RECENT {len(rows)} TRADE(S) WITH CONTEXT ===", ""]

        for r in rows:
            r = dict(r)
            pnl_str = f"${r['pnl']:+.2f}" if r.get("pnl") else "open"
            conf = f"{r.get('trade_confidence', 0) or 0:.2f}" if r.get("trade_confidence") else "-"
            regime = r.get("trade_regime") or "-"
            rl_action = r.get("rl_action")
            rl_str = f"RL={rl_action}" if rl_action is not None else ""

            lines.append(
                f"#{r['id']} [{r['status']}] {r['direction']} @ {r['entry_price']:.2f} | "
                f"P&L: {pnl_str} | {r['strategy']} | "
                f"Conf: {conf} | Regime: {regime} {rl_str}"
            )
            if r.get("claude_reasoning"):
                lines.append(f"   Claude: {r['claude_reasoning'][:100]}")

        return "\n".join(lines)
    finally:
        conn.close()


@mcp.tool()
async def equity_curve_data(days: int = 7) -> str:
    """Get equity curve data points (cumulative P&L over time).

    Args:
        days: Look back N days (0 = all time)
    """
    conn = _db()
    try:
        conditions = ["status = 'closed'"]
        params: list[Any] = []
        if days > 0:
            cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()
            conditions.append("closed_at >= ?")
            params.append(cutoff)

        where = " AND ".join(conditions)
        rows = conn.execute(
            f"SELECT id, pnl, closed_at, strategy FROM trades WHERE {where} ORDER BY closed_at",
            params,
        ).fetchall()

        if not rows:
            return "No closed trades."

        lines = [f"=== EQUITY CURVE ({len(rows)} trades) ===", ""]
        cum_pnl = 0
        peak = 0
        for r in rows:
            cum_pnl += r["pnl"]
            peak = max(peak, cum_pnl)
            dd = peak - cum_pnl
            dd_str = f" (DD: ${dd:.2f})" if dd > 0 else ""
            lines.append(
                f"#{r['id']:>4} {r['closed_at'][:16]} | ${r['pnl']:>+8.2f} | "
                f"Cum: ${cum_pnl:>+9.2f}{dd_str}"
            )

        lines += [
            "",
            f"Final P&L: ${cum_pnl:+,.2f}",
            f"Peak: ${peak:+,.2f}",
            f"Max drawdown from peak: ${peak - min(0, cum_pnl):,.2f}",
        ]

        return "\n".join(lines)
    finally:
        conn.close()


# ===================================================================
# Run
# ===================================================================

if __name__ == "__main__":
    mcp.run()
