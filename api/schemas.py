"""Pydantic response models for the TradingBot API."""

from __future__ import annotations

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Account
# ---------------------------------------------------------------------------

class AccountResponse(BaseModel):
    balance: float
    equity: float
    margin: float
    free_margin: float
    profit: float
    leverage: int


# ---------------------------------------------------------------------------
# Candles
# ---------------------------------------------------------------------------

class CandleResponse(BaseModel):
    datetime: str
    open: float
    high: float
    low: float
    close: float
    volume: int


# ---------------------------------------------------------------------------
# Trades
# ---------------------------------------------------------------------------

class TradeResponse(BaseModel):
    id: int
    strategy: str
    symbol: str
    direction: str
    entry_price: float
    exit_price: float | None = None
    sl: float
    tp: float
    lot_size: float
    pnl: float
    opened_at: str
    closed_at: str | None = None
    status: str


# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------

class BacktestRequest(BaseModel):
    strategy: str = Field(..., pattern=r"^(ema|breakout)$")
    count: int = Field(default=500, ge=50, le=5000)


class SimulatedTradeResponse(BaseModel):
    strategy: str
    direction: str
    entry_price: float
    exit_price: float
    sl: float
    tp: float
    pnl: float
    entry_time: str
    exit_time: str
    exit_reason: str


class EquityCurvePoint(BaseModel):
    trade_index: int
    cumulative_pnl: float


class BacktestReportResponse(BaseModel):
    total_trades: int
    win_rate: float
    profit_factor: float
    total_pnl: float
    avg_win: float
    avg_loss: float
    max_drawdown: float
    sharpe_ratio: float
    equity_curve_stats: dict[str, float]
    equity_curve: list[EquityCurvePoint]
    trades: list[SimulatedTradeResponse]


# ---------------------------------------------------------------------------
# Bot status
# ---------------------------------------------------------------------------

class BotStatusResponse(BaseModel):
    mode: str
    symbol: str
    timeframe: str
    use_mock: bool
    strategies: list[str]


# ---------------------------------------------------------------------------
# Learning system schemas
# ---------------------------------------------------------------------------

class StrategyWeight(BaseModel):
    name: str
    weight: float
    win_rate: float = 0.0
    trades: int = 0


class RLStats(BaseModel):
    episode: int = 0
    epsilon: float = 1.0
    total_reward: float = 0.0
    buffer_size: int = 0
    training: bool = False
    win_rate: float = 0.0
    torch_available: bool = False


class RegimeEntry(BaseModel):
    regime: str
    confidence: float
    adx: float = 0.0
    atr_ratio: float = 0.0
    timestamp: str | None = None


class ClaudeInsight(BaseModel):
    review_type: str
    timestamp: str | None = None
    analysis: str = ""
    recommendations: list[str] = []
    market_brief: str = ""
    score: float = 0.5


class LearningStatus(BaseModel):
    enabled: bool = False
    regime: str = "unknown"
    regime_confidence: float = 0.0
    session: str = "unknown"
    timeframe: str = "M1"
    context_timeframes: list[str] = []
    confidence_threshold: float = 0.4
    weights: list[StrategyWeight] = []
    rl_stats: RLStats = Field(default_factory=RLStats)


class StrategyPerformance(BaseModel):
    name: str
    trades: int = 0
    wins: int = 0
    losses: int = 0
    total_pnl: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0


class ToggleLearningRequest(BaseModel):
    enabled: bool


# ---------------------------------------------------------------------------
# RL Backtest schemas
# ---------------------------------------------------------------------------

class RLBacktestRequest(BaseModel):
    timeframe: str = Field(default="H1", pattern=r"^(M1|M5|M15|H1|H4|D1)$")
    count: int = Field(default=2000, ge=100, le=10000)
    epochs: int = Field(default=3, ge=1, le=20)
    train_every: int = Field(default=5, ge=1, le=50)


class RLBacktestTradeResponse(BaseModel):
    epoch: int
    action: str
    entry_price: float
    exit_price: float
    pnl: float
    reward: float
    hold_bars: int
    bar_index: int


class RLBacktestEpochStats(BaseModel):
    epoch: int
    trades: int
    wins: int
    win_rate: float
    total_reward: float
    avg_train_loss: float
    epsilon: float


class RLBacktestResponse(BaseModel):
    total_trades: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    total_reward: float = 0.0
    avg_reward_per_trade: float = 0.0
    max_drawdown: float = 0.0
    final_epsilon: float = 1.0
    episodes: int = 0
    epochs_completed: int = 0
    equity_curve: list[EquityCurvePoint] = []
    trades: list[RLBacktestTradeResponse] = []
    epoch_stats: list[RLBacktestEpochStats] = []
    claude_review: ClaudeInsight | None = None
