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
    tp_sl_ratio: float = 2.5
    trailing_atr_trigger: float = 0.3
    volume_filter_multiplier: float = 0.0
    require_htf_confirmation: bool = False
    breakeven_trigger_atr: float = 0.3

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
class BOSStrategyConfig:
    swing_lookback: int = 20
    swing_strength: int = 3
    atr_period: int = 14
    sl_buffer_atr: float = 0.1
    tp_sl_ratio: float = 2.0

@dataclass(frozen=True)
class CandlePatternConfig:
    min_wick_body_ratio: float = 2.0
    max_opposite_wick_ratio: float = 0.3
    trend_lookback: int = 5
    atr_period: int = 14
    sl_buffer_atr: float = 0.2
    tp_sl_ratio: float = 2.0
    require_confirmation: bool = True

@dataclass(frozen=True)
class RiskConfig:
    risk_pct: float = 3.0
    max_open_trades: int = 5
    max_daily_trades: int = 100
    max_daily_drawdown_pct: float = 3.0
    max_total_drawdown_pct: float = 10.0
    friday_cutoff_hour: int = 20
    news_hours: list[tuple[int, int]] = field(default_factory=list)
    max_spread_pips: float = 2.0

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
    agent_type: str = "sac"  # "sac", "ppo", "ddpg", "dqn", "ensemble"
    state_dim: int = 23
    lr: float = 1e-3
    gamma: float = 0.99
    epsilon_start: float = 0.3
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.995
    buffer_capacity: int = 10_000
    batch_size: int = 64
    target_update_freq: int = 50
    train_every_n_steps: int = 5
    reward_scale: float = 1.0
    penalty_scale: float = 1.5

@dataclass(frozen=True)
class PERConfig:
    """Prioritized Experience Replay parameters."""
    enabled: bool = True
    alpha: float = 0.6
    beta_start: float = 0.4
    beta_frames: int = 100_000

@dataclass(frozen=True)
class SACConfig:
    """SAC-specific parameters."""
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    tau: float = 0.005
    target_entropy: float = -1.0
    hidden_dim: int = 256
    buffer_capacity: int = 1_000_000
    batch_size: int = 256
    use_quantile: bool = False
    n_quantiles: int = 32
    risk_sensitivity: float = 0.0
    initial_random_steps: int = 10_000

@dataclass(frozen=True)
class PPOConfig:
    """PPO-specific parameters."""
    lr: float = 3e-4
    hidden_dim: int = 256
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    entropy_coeff: float = 0.01
    value_coeff: float = 0.5
    rollout_size: int = 2048
    n_epochs_per_update: int = 10
    batch_size: int = 64
    max_grad_norm: float = 0.5

@dataclass(frozen=True)
class DDPGConfig:
    """DDPG-specific parameters."""
    actor_lr: float = 1e-4
    critic_lr: float = 3e-4
    tau: float = 0.005
    hidden_dim: int = 256
    buffer_capacity: int = 1_000_000
    batch_size: int = 256
    ou_theta: float = 0.15
    ou_sigma: float = 0.2
    initial_random_steps: int = 10_000

@dataclass(frozen=True)
class TransformerConfig:
    """Transformer state encoder parameters."""
    enabled: bool = False
    seq_len: int = 64
    embed_dim: int = 128
    nhead: int = 4
    num_layers: int = 3
    dim_feedforward: int = 256
    dropout: float = 0.1

@dataclass(frozen=True)
class CompositeRewardConfig:
    """Composite reward wrapper parameters."""
    enabled: bool = True
    drawdown_weight: float = 0.5
    drawdown_threshold: float = 0.02
    sortino_weight: float = 0.3
    sortino_window: int = 20
    consistency_weight: float = 0.05
    transaction_weight: float = 0.1
    transaction_fee_rate: float = 0.001

@dataclass(frozen=True)
class EnsembleAgentConfig:
    """Agent ensemble parameters."""
    enabled: bool = False
    agents: tuple[str, ...] = ("sac", "ppo", "ddpg")
    strategy: str = "weighted_average"
    sharpe_window: int = 100
    eval_interval: int = 20

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

# ---------------------------------------------------------------------------
# Strategy profiles for RL backtesting
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StrategyProfile:
    """Pre-configured risk/reward profile for RL backtesting."""
    name: str
    risk_pct: float          # % of balance risked per trade
    sl_atr_mult: float       # SL = ATR x this
    tp_atr_mult: float       # TP = ATR x this
    trailing_atr_mult: float # trail trigger = ATR x this
    max_hold_bars: int       # force close after N bars
    max_daily_trades: int    # daily trade cap
    reward_scale: float      # reward multiplier
    penalty_scale: float     # loss penalty multiplier
    epsilon_start: float     # exploration start
    epsilon_decay: float     # exploration decay rate

STRATEGY_PROFILES: dict[str, StrategyProfile] = {
    "max_profit": StrategyProfile(
        name="max_profit", risk_pct=3.0,
        sl_atr_mult=1.2, tp_atr_mult=3.0,
        trailing_atr_mult=0.8, max_hold_bars=20,
        max_daily_trades=50, reward_scale=1.0,
        penalty_scale=1.0, epsilon_start=1.0, epsilon_decay=0.998,
    ),
    "aggressive": StrategyProfile(
        name="aggressive", risk_pct=2.0,
        sl_atr_mult=1.0, tp_atr_mult=2.0,
        trailing_atr_mult=0.5, max_hold_bars=12,
        max_daily_trades=50, reward_scale=1.0,
        penalty_scale=1.0, epsilon_start=1.0, epsilon_decay=0.995,
    ),
    "medium": StrategyProfile(
        name="medium", risk_pct=1.5,
        sl_atr_mult=1.5, tp_atr_mult=3.0,
        trailing_atr_mult=1.0, max_hold_bars=15,
        max_daily_trades=20, reward_scale=1.0,
        penalty_scale=1.0, epsilon_start=1.0, epsilon_decay=0.995,
    ),
    "conservative": StrategyProfile(
        name="conservative", risk_pct=0.5,
        sl_atr_mult=2.0, tp_atr_mult=4.0,
        trailing_atr_mult=1.5, max_hold_bars=25,
        max_daily_trades=10, reward_scale=1.0,
        penalty_scale=1.0, epsilon_start=0.8, epsilon_decay=0.993,
    ),
}


@dataclass(frozen=True)
class AppConfig:
    mode: str = os.getenv("MODE", "dry-run")
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    db_path: str = os.getenv("DB_PATH", "trades.db")
    mt5: MT5Config = field(default_factory=MT5Config)
    ema_strategy: EMAStrategyConfig = field(default_factory=EMAStrategyConfig)
    breakout_strategy: BreakoutStrategyConfig = field(default_factory=BreakoutStrategyConfig)
    bos_strategy: BOSStrategyConfig = field(default_factory=BOSStrategyConfig)
    candle_pattern: CandlePatternConfig = field(default_factory=CandlePatternConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    learning: LearningConfig = field(default_factory=LearningConfig)
    rl: RLConfig = field(default_factory=RLConfig)
    per: PERConfig = field(default_factory=PERConfig)
    sac: SACConfig = field(default_factory=SACConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    ddpg: DDPGConfig = field(default_factory=DDPGConfig)
    transformer: TransformerConfig = field(default_factory=TransformerConfig)
    composite_reward: CompositeRewardConfig = field(default_factory=CompositeRewardConfig)
    ensemble_agent: EnsembleAgentConfig = field(default_factory=EnsembleAgentConfig)
    claude: ClaudeConfig = field(default_factory=ClaudeConfig)
    ensemble: EnsembleConfig = field(default_factory=EnsembleConfig)
    slippage_pips: float = 0.2
