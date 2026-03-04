"""Backtesting endpoint."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from api.schemas import (
    BacktestReportResponse,
    BacktestRequest,
    EquityCurvePoint,
    SimulatedTradeResponse,
)
from backtesting.backtester import Backtester
from backtesting.report import BacktestReport
from config import AppConfig
from core.strategy import AsianRangeBreakoutStrategy, EMACrossoverStrategy

router = APIRouter(prefix="/api/backtest", tags=["backtest"])


def _create_strategy(name: str, config: AppConfig):
    """Instantiate a strategy by short name."""
    if name == "ema":
        cfg = config.ema_strategy
        return EMACrossoverStrategy(
            fast_ema=cfg.fast_ema,
            slow_ema=cfg.slow_ema,
            rsi_period=cfg.rsi_period,
            rsi_overbought=cfg.rsi_overbought,
            rsi_oversold=cfg.rsi_oversold,
            atr_period=cfg.atr_period,
            sl_atr_multiplier=cfg.sl_atr_multiplier,
            tp_sl_ratio=cfg.tp_sl_ratio,
            trailing_atr_trigger=cfg.trailing_atr_trigger,
        )
    if name == "breakout":
        cfg = config.breakout_strategy
        return AsianRangeBreakoutStrategy(
            asian_start_hour=cfg.asian_start_hour,
            asian_end_hour=cfg.asian_end_hour,
            active_start_hour=cfg.active_start_hour,
            active_end_hour=cfg.active_end_hour,
            atr_buffer_multiplier=cfg.atr_buffer_multiplier,
            min_range_pips=cfg.min_range_pips,
            max_range_pips=cfg.max_range_pips,
            tp_sl_ratio=cfg.tp_sl_ratio,
        )
    raise HTTPException(status_code=400, detail=f"Unknown strategy: {name}")


@router.post("", response_model=BacktestReportResponse)
async def run_backtest(
    body: BacktestRequest,
    request: Request,
) -> BacktestReportResponse:
    """Run a backtest for the requested strategy and return the report."""
    mt5_client = request.app.state.mt5_client
    config: AppConfig = request.app.state.config

    strategy = _create_strategy(body.strategy, config)

    backtester = Backtester(mt5_client=mt5_client)
    data = backtester.load_data(
        symbol=config.mt5.symbol,
        timeframe=config.mt5.timeframe,
        count=body.count,
    )
    sim_trades = backtester.run(strategy, data)

    report = BacktestReport(trades=sim_trades)

    # Build cumulative equity curve
    cumulative = 0.0
    equity_curve: list[EquityCurvePoint] = []
    for i, t in enumerate(sim_trades):
        cumulative += t.pnl
        equity_curve.append(EquityCurvePoint(trade_index=i, cumulative_pnl=round(cumulative, 4)))

    # Serialise simulated trades
    trade_responses = [
        SimulatedTradeResponse(
            strategy=t.strategy,
            direction=t.direction,
            entry_price=t.entry_price,
            exit_price=t.exit_price,
            sl=t.sl,
            tp=t.tp,
            pnl=t.pnl,
            entry_time=t.entry_time.isoformat()
            if hasattr(t.entry_time, "isoformat")
            else str(t.entry_time),
            exit_time=t.exit_time.isoformat()
            if hasattr(t.exit_time, "isoformat")
            else str(t.exit_time),
            exit_reason=t.exit_reason,
        )
        for t in sim_trades
    ]

    return BacktestReportResponse(
        total_trades=report.total_trades,
        win_rate=round(report.win_rate, 2),
        profit_factor=round(report.profit_factor, 4),
        total_pnl=round(report.total_pnl, 4),
        avg_win=round(report.avg_win, 4),
        avg_loss=round(report.avg_loss, 4),
        max_drawdown=round(report.max_drawdown, 4),
        sharpe_ratio=round(report.sharpe_ratio, 4),
        equity_curve_stats=report.equity_curve_stats(),
        equity_curve=equity_curve,
        trades=trade_responses,
    )
