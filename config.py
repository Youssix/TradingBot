from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

@dataclass(frozen=True)
class MT5Config:
    login: int = int(os.getenv("MT5_LOGIN", "0"))
    password: str = os.getenv("MT5_PASSWORD", "")
    server: str = os.getenv("MT5_SERVER", "")
    path: str = os.getenv("MT5_PATH", "")
    symbol: str = "XAUUSD"
    timeframe: str = "M1"
    use_mock: bool = os.getenv("USE_MOCK", "true").lower() == "true"

@dataclass(frozen=True)
class EMAStrategyConfig:
    fast_ema: int = 9
    slow_ema: int = 21
    rsi_period: int = 14
    rsi_overbought: float = 70.0
    rsi_oversold: float = 30.0
    atr_period: int = 14
    sl_atr_multiplier: float = 0.5
    tp_sl_ratio: float = 1.5
    trailing_atr_trigger: float = 0.3

@dataclass(frozen=True)
class BreakoutStrategyConfig:
    asian_start_hour: int = 0
    asian_end_hour: int = 8
    active_start_hour: int = 8
    active_end_hour: int = 20
    atr_buffer_multiplier: float = 0.1
    min_range_pips: float = 1.0
    max_range_pips: float = 15.0
    tp_sl_ratio: float = 1.5

@dataclass(frozen=True)
class RiskConfig:
    risk_pct: float = 1.0
    max_open_trades: int = 3
    max_daily_trades: int = 30
    max_daily_drawdown_pct: float = 3.0
    max_total_drawdown_pct: float = 10.0
    friday_cutoff_hour: int = 20
    news_hours: list[tuple[int, int]] = field(default_factory=list)

# ---------------------------------------------------------------------------
# Learning system configs
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LearningConfig:
    """Master toggle and general learning settings."""
    enabled: bool = os.getenv("LEARNING_ENABLED", "true").lower() == "true"
    confidence_threshold: float = 0.4

@dataclass(frozen=True)
class RLConfig:
    """Reinforcement learning agent parameters."""
    state_dim: int = 23
    action_dim: int = 3
    lr: float = 1e-3
    gamma: float = 0.95
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.995
    buffer_capacity: int = 10_000
    batch_size: int = 64
    target_update_freq: int = 50
    train_every_n_steps: int = 5
    reward_scale: float = 1.0
    penalty_scale: float = 1.5

@dataclass(frozen=True)
class ClaudeConfig:
    """Claude AI strategy and reviewer settings."""
    api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    model: str = "claude-sonnet-4-20250514"
    strategy_interval: float = 30.0      # seconds between strategy calls
    review_interval_hours: float = 24.0  # hours between full reviews
    max_tokens_strategy: int = 256
    max_tokens_review: int = 1024

@dataclass(frozen=True)
class EnsembleConfig:
    """Ensemble weighting and thresholds."""
    ema_weight: float = 0.25
    breakout_weight: float = 0.15
    rl_weight: float = 0.35
    claude_weight: float = 0.25
    agreement_bonus: float = 0.15
    performance_window: int = 50

# Timeframe-adaptive cycle intervals (seconds)
TIMEFRAME_CYCLE_SECONDS: dict[str, int] = {
    "M1": 60,
    "M5": 300,
    "M15": 900,
    "H1": 3600,
    "H4": 14400,
    "D1": 86400,
}

@dataclass(frozen=True)
class AppConfig:
    mode: str = os.getenv("MODE", "dry-run")
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    db_path: str = os.getenv("DB_PATH", "trades.db")
    mt5: MT5Config = field(default_factory=MT5Config)
    ema_strategy: EMAStrategyConfig = field(default_factory=EMAStrategyConfig)
    breakout_strategy: BreakoutStrategyConfig = field(default_factory=BreakoutStrategyConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    learning: LearningConfig = field(default_factory=LearningConfig)
    rl: RLConfig = field(default_factory=RLConfig)
    claude: ClaudeConfig = field(default_factory=ClaudeConfig)
    ensemble: EnsembleConfig = field(default_factory=EnsembleConfig)
