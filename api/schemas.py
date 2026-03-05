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
    strategy: str = Field(..., pattern=r"^(ema|breakout|bos|candle)$")
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
    strategy_mode: str = "independent"
    enabled_strategies: list[str] = []


class StrategyModeRequest(BaseModel):
    mode: str = Field(..., pattern=r"^(independent|ensemble)$")
    enabled_strategies: list[str] = []


class StrategyModeResponse(BaseModel):
    status: str
    mode: str
    enabled_strategies: list[str]


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

class RLModelInfo(BaseModel):
    id: int
    model_name: str = ""
    episode: int = 0
    epsilon: float = 1.0
    total_reward: float = 0.0
    win_rate: float = 0.0
    profile: str | None = None
    timeframe: str | None = None
    created_at: str | None = None


class RLModelListResponse(BaseModel):
    models: list[RLModelInfo] = []


class RLModelActivateRequest(BaseModel):
    model_id: int


class RLBacktestRequest(BaseModel):
    timeframe: str = Field(default="H1", pattern=r"^(M1|M5|M15|H1|H4|D1)$")
    count: int = Field(default=2000, ge=100, le=10000)
    epochs: int = Field(default=3, ge=1, le=20)
    train_every: int = Field(default=5, ge=1, le=50)
    profile: str = Field(default="max_profit", pattern=r"^(max_profit|aggressive|medium|conservative)$")
    model_id: int | None = None
    symbols: list[str] = Field(default=["GC=F"])
    augmentation_factor: int = Field(default=0, ge=0, le=5)
    agent_type: str | None = Field(default=None, pattern=r"^(sac|ppo|ddpg|dqn|ensemble)$")


class PipelineStep(BaseModel):
    timeframe: str = Field(default="H1", pattern=r"^(M1|M5|M15|H1|H4|D1)$")
    count: int = Field(default=2000, ge=100, le=10000)
    epochs: int = Field(default=3, ge=1, le=20)
    symbols: list[str] = Field(default=["GC=F"])
    augmentation_factor: int = Field(default=0, ge=0, le=5)


class PipelineRequest(BaseModel):
    steps: list[PipelineStep] = []
    profile: str = Field(default="max_profit", pattern=r"^(max_profit|aggressive|medium|conservative)$")
    train_every: int = Field(default=5, ge=1, le=50)
    model_id: int | None = None
    preset: str | None = Field(default=None, pattern=r"^(scalping|quick|standard|thorough)$")
    agent_type: str | None = Field(default=None, pattern=r"^(sac|ppo|ddpg|dqn|ensemble)$")


class RLBacktestTradeResponse(BaseModel):
    epoch: int
    action: str
    entry_price: float
    exit_price: float
    pnl: float
    reward: float
    hold_bars: int
    bar_index: int
    exit_reason: str = "signal"
    sl_price: float = 0.0
    tp_price: float = 0.0


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
    profile: str = "medium"
    equity_curve: list[EquityCurvePoint] = []
    trades: list[RLBacktestTradeResponse] = []
    epoch_stats: list[RLBacktestEpochStats] = []
    claude_review: ClaudeInsight | None = None


# ---------------------------------------------------------------------------
# Agent configuration schemas
# ---------------------------------------------------------------------------

class AgentConfigRequest(BaseModel):
    agent_type: str = Field(..., pattern=r"^(sac|ppo|ddpg|dqn|ensemble)$")
    hidden_dim: int | None = None
    actor_lr: float | None = None
    critic_lr: float | None = None
    alpha_lr: float | None = None
    gamma: float | None = None
    tau: float | None = None
    use_quantile: bool | None = None
    n_quantiles: int | None = None
    risk_sensitivity: float | None = None
    lr: float | None = None  # PPO
    clip_epsilon: float | None = None
    gae_lambda: float | None = None
    noise_sigma: float | None = None  # DDPG
    agents: list[str] | None = None  # Ensemble
    strategy: str | None = None  # Ensemble
    transformer_enabled: bool | None = None
    use_per: bool | None = None


class AgentConfigResponse(BaseModel):
    agent_type: str
    config: dict = {}
    status: str = "ok"


class PerAgentStats(BaseModel):
    name: str
    sharpe: float = 0.0
    cumulative_reward: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    trade_count: int = 0
    weight: float = 0.0
    is_active: bool = False


class EnsembleStatsResponse(BaseModel):
    strategy: str = "none"
    agents: list[PerAgentStats] = []
    active_agent: str = ""


class RiskMetricsResponse(BaseModel):
    cvar_5: float = 0.0
    var_5: float = 0.0
    q_mean: float = 0.0
    q_std: float = 0.0
    upside: float = 0.0
    available: bool = False


class TrainingMetricsResponse(BaseModel):
    agent_type: str = "dqn"
    episode: int = 0
    epsilon: float = 1.0
    total_reward: float = 0.0
    buffer_size: int = 0
    buffer_capacity: int = 0
    buffer_fill_pct: float = 0.0
    training: bool = False
    win_rate: float = 0.0
    torch_available: bool = False
    alpha: float | None = None
    use_quantile: bool = False
    use_per: bool = False
    mean_priority: float | None = None
    readiness: str = "untrained"
    active_model_id: int | None = None
    active_model_name: str | None = None
    active_model_profile: str | None = None
    active_model_timeframe: str | None = None
