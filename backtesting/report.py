from __future__ import annotations

import csv
from dataclasses import dataclass
from typing import Any

import numpy as np
from loguru import logger


@dataclass
class BacktestReport:
    """Generates performance reports from backtest trades."""

    trades: list[Any]  # list of SimulatedTrade

    @property
    def total_trades(self) -> int:
        return len(self.trades)

    @property
    def winning_trades(self) -> list[Any]:
        return [t for t in self.trades if t.pnl > 0]

    @property
    def losing_trades(self) -> list[Any]:
        return [t for t in self.trades if t.pnl <= 0]

    @property
    def win_rate(self) -> float:
        """Win rate as percentage."""
        if not self.trades:
            return 0.0
        return len(self.winning_trades) / len(self.trades) * 100

    @property
    def profit_factor(self) -> float:
        """Ratio of gross profit to gross loss."""
        gross_profit = sum(t.pnl for t in self.winning_trades)
        gross_loss = abs(sum(t.pnl for t in self.losing_trades))
        if gross_loss == 0:
            return float("inf") if gross_profit > 0 else 0.0
        return gross_profit / gross_loss

    @property
    def total_pnl(self) -> float:
        return sum(t.pnl for t in self.trades)

    @property
    def avg_win(self) -> float:
        wins = self.winning_trades
        return sum(t.pnl for t in wins) / len(wins) if wins else 0.0

    @property
    def avg_loss(self) -> float:
        losses = self.losing_trades
        return sum(t.pnl for t in losses) / len(losses) if losses else 0.0

    @property
    def max_drawdown(self) -> float:
        """Maximum drawdown from equity curve."""
        if not self.trades:
            return 0.0
        equity = np.cumsum([t.pnl for t in self.trades])
        peak = np.maximum.accumulate(equity)
        drawdown = peak - equity
        return float(np.max(drawdown)) if len(drawdown) > 0 else 0.0

    @property
    def sharpe_ratio(self) -> float:
        """Annualized Sharpe ratio (assuming 5-min bars, ~252 trading days)."""
        if len(self.trades) < 2:
            return 0.0
        returns = [t.pnl for t in self.trades]
        mean_return = np.mean(returns)
        std_return = np.std(returns, ddof=1)
        if std_return == 0:
            return 0.0
        # Annualize: ~75,600 five-min bars per year (252 days * 24h * 12 bars/h)
        annualization = np.sqrt(min(len(self.trades), 75600))
        return float(mean_return / std_return * annualization)

    def equity_curve_stats(self) -> dict[str, float]:
        """Get equity curve statistics."""
        if not self.trades:
            return {"final_equity": 0, "max_equity": 0, "min_equity": 0, "max_drawdown": 0}
        equity = list(np.cumsum([t.pnl for t in self.trades]))
        return {
            "final_equity": equity[-1],
            "max_equity": max(equity),
            "min_equity": min(equity),
            "max_drawdown": self.max_drawdown,
        }

    def print_summary(self) -> None:
        """Print formatted backtest summary to console."""
        print("\n" + "=" * 50)
        print("         BACKTEST REPORT")
        print("=" * 50)
        print(f"  Total Trades:    {self.total_trades}")
        print(f"  Win Rate:        {self.win_rate:.1f}%")
        print(f"  Profit Factor:   {self.profit_factor:.2f}")
        print(f"  Total PnL:       {self.total_pnl:.2f}")
        print(f"  Avg Win:         {self.avg_win:.2f}")
        print(f"  Avg Loss:        {self.avg_loss:.2f}")
        print(f"  Max Drawdown:    {self.max_drawdown:.2f}")
        print(f"  Sharpe Ratio:    {self.sharpe_ratio:.2f}")
        stats = self.equity_curve_stats()
        print(f"  Final Equity:    {stats['final_equity']:.2f}")
        print(f"  Max Equity:      {stats['max_equity']:.2f}")
        print("=" * 50 + "\n")

    def export_csv(self, filepath: str) -> None:
        """Export trades to CSV file."""
        if not self.trades:
            logger.warning("No trades to export")
            return
        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "strategy", "direction", "entry_price", "exit_price",
                "sl", "tp", "pnl", "entry_time", "exit_time", "exit_reason",
            ])
            for t in self.trades:
                writer.writerow([
                    t.strategy, t.direction, t.entry_price, t.exit_price,
                    t.sl, t.tp, t.pnl, t.entry_time, t.exit_time, t.exit_reason,
                ])
        logger.info(f"Exported {len(self.trades)} trades to {filepath}")
