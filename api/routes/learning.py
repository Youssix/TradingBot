"""REST API endpoints for the learning system."""

from __future__ import annotations

import asyncio
import json
import traceback

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from loguru import logger

import dataclasses

from api.schemas import (
    AgentConfigRequest,
    AgentConfigResponse,
    ClaudeInsight,
    EnsembleStatsResponse,
    EquityCurvePoint,
    LearningStatus,
    PerAgentStats,
    PipelineRequest,
    PipelineStep,
    RegimeEntry,
    RiskMetricsResponse,
    RLBacktestEpochStats,
    RLBacktestRequest,
    RLBacktestResponse,
    RLBacktestTradeResponse,
    RLModelActivateRequest,
    RLModelInfo,
    RLModelListResponse,
    RLStats,
    StrategyPerformance,
    StrategyWeight,
    ToggleLearningRequest,
    TrainingMetricsResponse,
)

router = APIRouter(prefix="/api/learning", tags=["learning"])


def _get_learning_state(request: Request) -> dict:
    """Get learning components from app state, or return defaults."""
    state = request.app.state
    return {
        "enabled": getattr(state, "learning_enabled", False),
        "feature_engine": getattr(state, "feature_engine", None),
        "regime_detector": getattr(state, "regime_detector", None),
        "rl_agent": getattr(state, "rl_agent", None),
        "rl_env": getattr(state, "rl_env", None),
        "performance_tracker": getattr(state, "performance_tracker", None),
        "ensemble": getattr(state, "ensemble", None),
        "claude_strategy": getattr(state, "claude_strategy", None),
        "claude_reviewer": getattr(state, "claude_reviewer", None),
        "config": getattr(state, "config", None),
        "db": getattr(state, "db", None),
    }


@router.get("/status", response_model=LearningStatus)
async def get_learning_status(request: Request) -> LearningStatus:
    """Get overall learning system status."""
    ls = _get_learning_state(request)
    config = ls["config"]

    weights: list[StrategyWeight] = []
    if ls["ensemble"]:
        ens_weights = ls["ensemble"].weights
        tracker = ls["performance_tracker"]
        for name, w in ens_weights.items():
            stats_obj = tracker.get_stats(name) if tracker else None
            weights.append(StrategyWeight(
                name=name,
                weight=round(w, 4),
                win_rate=round(stats_obj.win_rate, 4) if stats_obj else 0.0,
                trades=stats_obj.trades if stats_obj else 0,
            ))

    rl_stats = RLStats()
    if ls["rl_agent"]:
        s = ls["rl_agent"].get_stats()
        rl_env = ls["rl_env"]
        rl_stats = RLStats(
            episode=s["episode"],
            epsilon=s["epsilon"],
            total_reward=s["total_reward"],
            buffer_size=s["buffer_size"],
            training=s["training"],
            win_rate=round(rl_env.win_rate, 4) if rl_env else 0.0,
            torch_available=s["torch_available"],
        )

    regime = "unknown"
    regime_conf = 0.0
    if ls["regime_detector"]:
        rd = ls["regime_detector"].current
        regime = rd.regime.value
        regime_conf = rd.confidence

    context_tfs: list[str] = []
    if ls["feature_engine"]:
        context_tfs = ls["feature_engine"].context_timeframes

    # Get session from latest regime history entry
    session = "unknown"
    if ls["regime_detector"] and ls["regime_detector"].history:
        # session isn't in regime history, but we can derive from features
        pass
    # Try getting it from the last stored market features
    if ls["db"]:
        try:
            cursor = ls["db"]._conn.execute(
                "SELECT session FROM market_features ORDER BY id DESC LIMIT 1"
            )
            row = cursor.fetchone()
            if row and row[0]:
                session = row[0]
        except Exception:
            pass

    return LearningStatus(
        enabled=ls["enabled"],
        regime=regime,
        regime_confidence=round(regime_conf, 3),
        session=session,
        timeframe=config.mt5.timeframe if config else "M1",
        context_timeframes=context_tfs,
        confidence_threshold=config.learning.confidence_threshold if config else 0.4,
        weights=weights,
        rl_stats=rl_stats,
    )


@router.get("/performance", response_model=list[StrategyPerformance])
async def get_performance(request: Request) -> list[StrategyPerformance]:
    """Get per-strategy rolling performance stats."""
    ls = _get_learning_state(request)
    tracker = ls["performance_tracker"]
    if not tracker:
        return []

    all_stats = tracker.get_all_stats()
    return [
        StrategyPerformance(
            name=s.name,
            trades=s.trades,
            wins=s.wins,
            losses=s.losses,
            total_pnl=round(s.total_pnl, 2),
            win_rate=round(s.win_rate, 4),
            profit_factor=round(s.profit_factor, 2),
        )
        for s in all_stats
    ]


@router.get("/regime-history", response_model=list[RegimeEntry])
async def get_regime_history(request: Request) -> list[RegimeEntry]:
    """Get recent regime classifications."""
    ls = _get_learning_state(request)
    detector = ls["regime_detector"]
    if not detector:
        return []

    return [
        RegimeEntry(
            regime=h["regime"],
            confidence=h["confidence"],
            adx=h.get("adx", 0.0),
            atr_ratio=h.get("atr_ratio", 0.0),
            timestamp=str(h.get("timestamp", "")),
        )
        for h in detector.history[-50:]
    ]


@router.get("/insights", response_model=list[ClaudeInsight])
async def get_insights(request: Request) -> list[ClaudeInsight]:
    """Get Claude review insights."""
    ls = _get_learning_state(request)

    # Try DB first
    db = ls["db"]
    if db:
        try:
            rows = db.get_recent_insights(20)
            return [
                ClaudeInsight(
                    review_type=r.get("review_type", ""),
                    timestamp=r.get("created_at", ""),
                    analysis=str(r.get("analysis_json", "")),
                    recommendations=r.get("recommendations_json", []) if isinstance(r.get("recommendations_json"), list) else [],
                    market_brief="",
                    score=r.get("backtest_score", 0.5) or 0.5,
                )
                for r in rows
            ]
        except Exception:
            pass

    # Fallback to in-memory insights
    reviewer = ls["claude_reviewer"]
    if reviewer:
        return [
            ClaudeInsight(
                review_type=i.get("review_type", ""),
                timestamp=i.get("timestamp", ""),
                analysis=i.get("analysis", ""),
                recommendations=i.get("recommendations", []),
                market_brief=i.get("market_brief", ""),
                score=i.get("score", 0.5),
            )
            for i in reviewer.insights[-20:]
        ]

    return []


@router.get("/rl-stats", response_model=RLStats)
async def get_rl_stats(request: Request) -> RLStats:
    """Get detailed RL agent stats."""
    ls = _get_learning_state(request)
    agent = ls["rl_agent"]
    env = ls["rl_env"]
    if not agent:
        return RLStats()

    s = agent.get_stats()
    return RLStats(
        episode=s["episode"],
        epsilon=s["epsilon"],
        total_reward=s["total_reward"],
        buffer_size=s["buffer_size"],
        training=s["training"],
        win_rate=round(env.win_rate, 4) if env else 0.0,
        torch_available=s["torch_available"],
    )


@router.post("/review")
async def trigger_review(request: Request) -> dict:
    """Trigger a Claude review now."""
    ls = _get_learning_state(request)
    reviewer = ls["claude_reviewer"]
    db = ls["db"]

    if not reviewer:
        raise HTTPException(status_code=503, detail="Claude reviewer not initialized")

    # Get recent trades for review
    trades = db.get_recent_closed_trades(50) if db else []
    rl_stats = ls["rl_agent"].get_stats() if ls["rl_agent"] else None
    regime = ls["regime_detector"].current.regime.value if ls["regime_detector"] else "unknown"

    result = reviewer.review_trades(trades, rl_stats, regime)

    # Store in DB
    if db:
        try:
            db.insert_claude_insight(
                review_type="manual_review",
                analysis=result.get("analysis", ""),
                recommendations=result.get("recommendations", []),
                strategy_rules=result.get("strategy_rules", []),
                backtest_score=result.get("score", 0.5),
            )
        except Exception:
            pass

    return {"status": "ok", "result": result}


@router.post("/optimize")
async def trigger_optimization(request: Request) -> dict:
    """Trigger parameter optimization."""
    ls = _get_learning_state(request)
    ensemble = ls["ensemble"]

    if not ensemble:
        raise HTTPException(status_code=503, detail="Ensemble not initialized")

    new_weights = ensemble.update_weights()
    return {"status": "ok", "weights": new_weights}


@router.post("/toggle")
async def toggle_learning(request: Request, body: ToggleLearningRequest) -> dict:
    """Enable or disable the learning system."""
    state = request.app.state
    state.learning_enabled = body.enabled

    if hasattr(state, "rl_agent") and state.rl_agent:
        state.rl_agent.training = body.enabled

    return {"status": "ok", "enabled": body.enabled}


# ---------------------------------------------------------------------------
# Agent configuration & advanced stats endpoints
# ---------------------------------------------------------------------------


@router.get("/agent-config", response_model=AgentConfigResponse)
async def get_agent_config(request: Request) -> AgentConfigResponse:
    """Get current agent type and configuration."""
    ls = _get_learning_state(request)
    config = ls["config"]
    if not config:
        return AgentConfigResponse(agent_type="dqn", config={}, status="no_config")

    agent_type = config.rl.agent_type
    cfg_dict: dict = {}

    if agent_type == "sac":
        cfg_dict = dataclasses.asdict(config.sac)
    elif agent_type == "ppo":
        cfg_dict = dataclasses.asdict(config.ppo)
    elif agent_type == "ddpg":
        cfg_dict = dataclasses.asdict(config.ddpg)
    elif agent_type == "ensemble":
        cfg_dict = dataclasses.asdict(config.ensemble_agent)
    elif agent_type == "dqn":
        cfg_dict = dataclasses.asdict(config.rl)

    # Add cross-cutting config
    cfg_dict["transformer_enabled"] = config.transformer.enabled
    cfg_dict["use_per"] = config.per.enabled

    return AgentConfigResponse(agent_type=agent_type, config=cfg_dict, status="ok")


@router.post("/agent-config", response_model=AgentConfigResponse)
async def set_agent_config(request: Request, body: AgentConfigRequest) -> AgentConfigResponse:
    """Set agent configuration and reinitialize the agent."""
    from core.learning.agent_factory import create_agent

    state = request.app.state
    config = getattr(state, "config", None)
    if not config:
        raise HTTPException(status_code=503, detail="Config not initialized")

    # Build updated sub-configs using dataclasses.replace on frozen dataclasses
    new_rl = dataclasses.replace(config.rl, agent_type=body.agent_type)

    new_sac = config.sac
    new_ppo = config.ppo
    new_ddpg = config.ddpg
    new_ensemble_agent = config.ensemble_agent
    new_transformer = config.transformer
    new_per = config.per

    if body.agent_type == "sac":
        replacements = {}
        if body.hidden_dim is not None:
            replacements["hidden_dim"] = body.hidden_dim
        if body.actor_lr is not None:
            replacements["actor_lr"] = body.actor_lr
        if body.critic_lr is not None:
            replacements["critic_lr"] = body.critic_lr
        if body.alpha_lr is not None:
            replacements["alpha_lr"] = body.alpha_lr
        if body.tau is not None:
            replacements["tau"] = body.tau
        if body.use_quantile is not None:
            replacements["use_quantile"] = body.use_quantile
        if body.n_quantiles is not None:
            replacements["n_quantiles"] = body.n_quantiles
        if body.risk_sensitivity is not None:
            replacements["risk_sensitivity"] = body.risk_sensitivity
        if replacements:
            new_sac = dataclasses.replace(config.sac, **replacements)

    elif body.agent_type == "ppo":
        replacements = {}
        if body.hidden_dim is not None:
            replacements["hidden_dim"] = body.hidden_dim
        if body.lr is not None:
            replacements["lr"] = body.lr
        if body.clip_epsilon is not None:
            replacements["clip_epsilon"] = body.clip_epsilon
        if body.gae_lambda is not None:
            replacements["gae_lambda"] = body.gae_lambda
        if replacements:
            new_ppo = dataclasses.replace(config.ppo, **replacements)

    elif body.agent_type == "ddpg":
        replacements = {}
        if body.hidden_dim is not None:
            replacements["hidden_dim"] = body.hidden_dim
        if body.actor_lr is not None:
            replacements["actor_lr"] = body.actor_lr
        if body.critic_lr is not None:
            replacements["critic_lr"] = body.critic_lr
        if body.tau is not None:
            replacements["tau"] = body.tau
        if body.noise_sigma is not None:
            replacements["ou_sigma"] = body.noise_sigma
        if replacements:
            new_ddpg = dataclasses.replace(config.ddpg, **replacements)

    elif body.agent_type == "ensemble":
        replacements = {}
        if body.agents is not None:
            replacements["agents"] = tuple(body.agents)
            replacements["enabled"] = True
        if body.strategy is not None:
            replacements["strategy"] = body.strategy
        if replacements:
            new_ensemble_agent = dataclasses.replace(config.ensemble_agent, **replacements)

    # Cross-cutting params
    if body.gamma is not None:
        new_rl = dataclasses.replace(new_rl, gamma=body.gamma)
    if body.transformer_enabled is not None:
        new_transformer = dataclasses.replace(config.transformer, enabled=body.transformer_enabled)
    if body.use_per is not None:
        new_per = dataclasses.replace(config.per, enabled=body.use_per)

    # Build new frozen config
    new_config = dataclasses.replace(
        config,
        rl=new_rl,
        sac=new_sac,
        ppo=new_ppo,
        ddpg=new_ddpg,
        ensemble_agent=new_ensemble_agent,
        transformer=new_transformer,
        per=new_per,
    )

    # Reinitialize agent
    new_agent = create_agent(new_config)

    # Update app state
    state.config = new_config
    state.rl_agent = new_agent

    # Return current config
    cfg_dict: dict = {}
    if body.agent_type == "sac":
        cfg_dict = dataclasses.asdict(new_sac)
    elif body.agent_type == "ppo":
        cfg_dict = dataclasses.asdict(new_ppo)
    elif body.agent_type == "ddpg":
        cfg_dict = dataclasses.asdict(new_ddpg)
    elif body.agent_type == "ensemble":
        cfg_dict = dataclasses.asdict(new_ensemble_agent)
    else:
        cfg_dict = dataclasses.asdict(new_rl)

    cfg_dict["transformer_enabled"] = new_transformer.enabled
    cfg_dict["use_per"] = new_per.enabled

    return AgentConfigResponse(agent_type=body.agent_type, config=cfg_dict, status="ok")


@router.get("/ensemble-stats", response_model=EnsembleStatsResponse)
async def get_ensemble_stats(request: Request) -> EnsembleStatsResponse:
    """Get per-agent stats when running ensemble."""
    from core.learning.agent_ensemble import AgentEnsemble

    ls = _get_learning_state(request)
    agent = ls["rl_agent"]

    if not agent or not isinstance(agent, AgentEnsemble):
        return EnsembleStatsResponse()

    stats = agent.get_agent_stats()
    active = agent.get_stats().get("active_agent", "")
    strategy = agent.get_stats().get("strategy", "weighted_average")

    # Compute weights from Sharpe ratios (softmax)
    import math
    sharpes = {name: s.get("sharpe", 0.0) for name, s in stats.items()}
    max_s = max(sharpes.values()) if sharpes else 0.0
    exp_vals = {name: math.exp(s - max_s) for name, s in sharpes.items()}
    total_exp = sum(exp_vals.values()) or 1.0
    weights = {name: v / total_exp for name, v in exp_vals.items()}

    agents = [
        PerAgentStats(
            name=name,
            sharpe=round(s.get("sharpe", 0.0), 4),
            cumulative_reward=round(s.get("cumulative_reward", 0.0), 4),
            max_drawdown=round(s.get("max_drawdown", 0.0), 4),
            win_rate=round(s.get("win_rate", 0.0), 4),
            trade_count=s.get("trade_count", 0),
            weight=round(weights.get(name, 0.0), 4),
            is_active=(name == active),
        )
        for name, s in stats.items()
    ]

    return EnsembleStatsResponse(
        strategy=strategy,
        agents=agents,
        active_agent=active,
    )


@router.get("/risk-metrics", response_model=RiskMetricsResponse)
async def get_risk_metrics(request: Request) -> RiskMetricsResponse:
    """Get risk metrics from quantile SAC agent."""
    ls = _get_learning_state(request)
    agent = ls["rl_agent"]
    env = ls["rl_env"]

    if not agent or not hasattr(agent, "get_risk_metrics"):
        return RiskMetricsResponse(available=False)

    try:
        # Get the last state from the environment
        state = None
        if env and hasattr(env, "last_state"):
            state = env.last_state
        if state is None:
            # Use a zero state as fallback
            state_dim = getattr(agent, "_state_dim", 23)
            state = [0.0] * state_dim

        action, _ = agent.select_action(state)
        metrics = agent.get_risk_metrics(state, action)

        return RiskMetricsResponse(
            cvar_5=round(metrics.get("cvar_5", 0.0), 6),
            var_5=round(metrics.get("var_5", 0.0), 6),
            q_mean=round(metrics.get("q_mean", 0.0), 6),
            q_std=round(metrics.get("q_std", 0.0), 6),
            upside=round(metrics.get("upside", 0.0), 6),
            available=True,
        )
    except Exception as e:
        logger.warning(f"Failed to get risk metrics: {e}")
        return RiskMetricsResponse(available=False)


@router.get("/training-metrics", response_model=TrainingMetricsResponse)
async def get_training_metrics(request: Request) -> TrainingMetricsResponse:
    """Get enhanced training metrics including buffer fill, alpha, PER info."""
    ls = _get_learning_state(request)
    agent = ls["rl_agent"]
    env = ls["rl_env"]
    config = ls["config"]

    if not agent:
        return TrainingMetricsResponse()

    s = agent.get_stats()
    agent_type = config.rl.agent_type if config else "dqn"

    buffer_size = s.get("buffer_size", 0)
    # Determine buffer capacity based on agent type
    buffer_capacity = 0
    if config:
        if agent_type == "sac":
            buffer_capacity = config.sac.buffer_capacity
        elif agent_type == "ddpg":
            buffer_capacity = config.ddpg.buffer_capacity
        elif agent_type == "dqn":
            buffer_capacity = config.rl.buffer_capacity
        elif agent_type == "ensemble":
            # Use SAC capacity as representative
            buffer_capacity = config.sac.buffer_capacity

    buffer_fill_pct = round(buffer_size / max(buffer_capacity, 1) * 100, 1)

    episode = s.get("episode", 0)
    epsilon = s.get("epsilon", 1.0)
    win_rate = round(env.win_rate, 4) if env else 0.0

    # Compute readiness
    if epsilon < 0.3 and win_rate >= 0.5 and episode >= 1000:
        readiness = "ready"
    elif episode < 100 and epsilon > 0.8:
        readiness = "untrained"
    else:
        readiness = "learning"

    app_state = request.app.state

    return TrainingMetricsResponse(
        agent_type=agent_type,
        episode=episode,
        epsilon=epsilon,
        total_reward=s.get("total_reward", 0.0),
        buffer_size=buffer_size,
        buffer_capacity=buffer_capacity,
        buffer_fill_pct=buffer_fill_pct,
        training=s.get("training", False),
        win_rate=win_rate,
        torch_available=s.get("torch_available", False),
        alpha=s.get("alpha"),
        use_quantile=s.get("use_quantile", False),
        use_per=s.get("use_per", False),
        mean_priority=s.get("mean_priority"),
        readiness=readiness,
        active_model_id=getattr(app_state, "active_model_id", None),
        active_model_name=getattr(app_state, "active_model_name", None),
        active_model_profile=getattr(app_state, "active_model_profile", None),
        active_model_timeframe=getattr(app_state, "active_model_timeframe", None),
    )


def _fetch_yfinance_data(symbol: str, timeframe: str, count: int) -> "pd.DataFrame":
    """Fetch OHLCV data from YFinance for a single symbol/timeframe.

    Handles column normalization, timezone stripping, H4 resampling,
    and trimming to the requested bar count.
    """
    import pandas as pd
    import yfinance as yf

    ticker = yf.Ticker(symbol)

    tf_map = {
        "M1": ("1m", "5d"),
        "M5": ("5m", "60d"),
        "M15": ("15m", "60d"),
        "H1": ("1h", "730d"),
        "H4": ("1h", "730d"),
        "D1": ("1d", "max"),
    }
    interval, period = tf_map.get(timeframe, ("1h", "730d"))

    df = ticker.history(period=period, interval=interval)
    if df.empty:
        raise HTTPException(status_code=502, detail=f"YFinance returned no data for {symbol}")

    # Normalize columns
    df = df.reset_index()
    col_map = {
        "Datetime": "datetime", "Date": "datetime",
        "Open": "open", "High": "high", "Low": "low",
        "Close": "close", "Volume": "volume",
    }
    df = df.rename(columns=col_map)
    df = df[["datetime", "open", "high", "low", "close", "volume"]].copy()

    if hasattr(df["datetime"].dtype, "tz") and df["datetime"].dtype.tz is not None:
        df["datetime"] = df["datetime"].dt.tz_localize(None)

    # Resample to H4 if requested (YFinance gives 1h, we aggregate)
    if timeframe == "H4":
        df = df.set_index("datetime")
        df = df.resample("4h").agg({
            "open": "first", "high": "max", "low": "min",
            "close": "last", "volume": "sum",
        }).dropna().reset_index()

    # Trim to requested count
    df = df.tail(count).reset_index(drop=True)
    return df


async def _fetch_multi_symbol_data(
    symbols: list[str], timeframe: str, count: int
) -> list["pd.DataFrame"]:
    """Fetch data for multiple symbols with 0.5s delay between requests."""
    frames: list["pd.DataFrame"] = []
    for i, symbol in enumerate(symbols):
        if i > 0:
            await asyncio.sleep(0.5)
        df = _fetch_yfinance_data(symbol, timeframe, count)
        frames.append(df)
        logger.info(f"Fetched {len(df)} {timeframe} candles for {symbol}")
    return frames


def _build_training_datasets(
    frames: list["pd.DataFrame"],
    augmentation_factor: int,
) -> list["pd.DataFrame"]:
    """Build full list of DataFrames: originals + augmented copies."""
    from core.learning.data_augmentation import augment_dataframe

    datasets: list["pd.DataFrame"] = list(frames)
    if augmentation_factor > 0:
        for i, df in enumerate(frames):
            augmented = augment_dataframe(df, factor=augmentation_factor, seed=42 + i)
            datasets.extend(augmented)
    return datasets


@router.post("/backtest", response_model=RLBacktestResponse)
async def run_rl_backtest(
    request: Request, body: RLBacktestRequest
) -> RLBacktestResponse:
    """Train RL agent on historical candles and return results."""
    from core.learning.feature_engine import FeatureEngine
    from core.learning.regime_detector import RegimeDetector
    from core.learning.agent_factory import create_agent
    from core.learning.rl_backtester import RLBacktestConfig, RLBacktester
    from core.learning.rl_environment import TradingEnvironment

    ls = _get_learning_state(request)
    config = ls["config"]

    # --- Fetch historical data ---
    try:
        frames = await _fetch_multi_symbol_data(body.symbols, body.timeframe, body.count)
        datasets = _build_training_datasets(frames, body.augmentation_factor)
        logger.info(
            f"RL backtest: {len(datasets)} datasets "
            f"({len(body.symbols)} symbols x {1 + body.augmentation_factor} copies)"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed to fetch data: {e}")

    # --- Set up RL components ---
    rl_cfg = config.rl if config else None
    agent = create_agent(config)
    env = TradingEnvironment(
        reward_scale=rl_cfg.reward_scale if rl_cfg else 1.0,
        penalty_scale=rl_cfg.penalty_scale if rl_cfg else 1.5,
    )

    # --- Load checkpoint if continuing from saved model ---
    db = ls["db"]
    if body.model_id is not None and db:
        row = db.load_rl_model_by_id(body.model_id)
        if row and row.get("model_blob"):
            agent.load_checkpoint(row["model_blob"])
            logger.info(f"Loaded checkpoint model_id={body.model_id} (ep={row.get('episode',0)}, eps={row.get('epsilon',1.0):.4f})")

    feature_engine = FeatureEngine(primary_tf=body.timeframe)
    regime_detector = RegimeDetector(primary_tf=body.timeframe)

    bt_config = RLBacktestConfig(
        timeframe=body.timeframe,
        count=body.count,
        epochs=body.epochs,
        train_every=body.train_every,
        profile=body.profile,
    )

    backtester = RLBacktester(
        config=bt_config,
        agent=agent,
        env=env,
        feature_engine=feature_engine,
        regime_detector=regime_detector,
    )

    # --- Run backtest on each dataset sequentially ---
    result = None
    for df in datasets:
        result = backtester.run(df)

    if result is None:
        raise HTTPException(status_code=500, detail="No data to train on")

    # --- Optional: Claude review of backtest trades ---
    claude_insight: ClaudeInsight | None = None
    reviewer = ls["claude_reviewer"]
    if reviewer and result.trades:
        trade_dicts = [
            {
                "direction": t.action,
                "pnl": t.pnl,
                "reward": t.reward,
                "hold_bars": t.hold_bars,
                "regime": "backtest",
                "session": "backtest",
            }
            for t in result.trades[-50:]
        ]
        review = reviewer.review_trades(
            trade_dicts,
            agent.get_stats(),
            "backtest",
        )
        claude_insight = ClaudeInsight(
            review_type="rl_backtest_review",
            timestamp=None,
            analysis=review.get("analysis", ""),
            recommendations=review.get("recommendations", []),
            market_brief=review.get("market_brief", ""),
            score=review.get("score", 0.5),
        )

        if db:
            try:
                db.insert_claude_insight(
                    review_type="rl_backtest_review",
                    analysis=review.get("analysis", ""),
                    recommendations=review.get("recommendations", []),
                    strategy_rules=review.get("strategy_rules", []),
                    backtest_score=review.get("score", 0.5),
                )
            except Exception:
                pass

    # --- Save RL checkpoint ---
    if db:
        try:
            blob = agent.save_checkpoint()
            if blob:
                db.save_rl_model(
                    model_name=body.profile,
                    model_blob=blob,
                    episode=agent.episode,
                    epsilon=agent.epsilon,
                    total_reward=agent.total_reward,
                    win_rate=result.win_rate,
                    profile=body.profile,
                    timeframe=body.timeframe,
                )
                logger.info(f"RL backtest model saved to DB (profile={body.profile})")
        except Exception as e:
            logger.warning(f"Failed to save RL checkpoint: {e}")

    # --- Build response ---
    return RLBacktestResponse(
        total_trades=result.total_trades,
        win_rate=result.win_rate,
        total_pnl=result.total_pnl,
        total_reward=result.total_reward,
        avg_reward_per_trade=result.avg_reward_per_trade,
        max_drawdown=result.max_drawdown,
        final_epsilon=result.final_epsilon,
        episodes=result.episodes,
        epochs_completed=result.epochs_completed,
        profile=result.profile,
        equity_curve=[
            EquityCurvePoint(
                trade_index=pt["trade_index"],
                cumulative_pnl=pt["cumulative_pnl"],
            )
            for pt in result.equity_curve
        ],
        trades=[
            RLBacktestTradeResponse(
                epoch=t.epoch,
                action=t.action,
                entry_price=t.entry_price,
                exit_price=t.exit_price,
                pnl=t.pnl,
                reward=t.reward,
                hold_bars=t.hold_bars,
                bar_index=t.bar_index,
                exit_reason=t.exit_reason,
                sl_price=t.sl_price,
                tp_price=t.tp_price,
            )
            for t in result.trades
        ],
        epoch_stats=[
            RLBacktestEpochStats(**s) for s in result.epoch_stats
        ],
        claude_review=claude_insight,
    )


@router.post("/backtest/stream")
async def run_rl_backtest_stream(request: Request, body: RLBacktestRequest):
    """Train RL agent on historical candles with SSE streaming progress."""
    from core.learning.feature_engine import FeatureEngine
    from core.learning.regime_detector import RegimeDetector
    from core.learning.agent_factory import create_agent
    from core.learning.rl_backtester import RLBacktestConfig, RLBacktester
    from core.learning.rl_environment import TradingEnvironment

    ls = _get_learning_state(request)
    config = ls["config"]

    # --- Fetch historical data ---
    try:
        frames = await _fetch_multi_symbol_data(body.symbols, body.timeframe, body.count)
        datasets = _build_training_datasets(frames, body.augmentation_factor)
        logger.info(
            f"RL backtest stream: {len(datasets)} datasets "
            f"({len(body.symbols)} symbols x {1 + body.augmentation_factor} copies)"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed to fetch data: {e}")

    # --- Set up RL components ---
    rl_cfg = config.rl if config else None
    agent = create_agent(config)
    env = TradingEnvironment(
        reward_scale=rl_cfg.reward_scale if rl_cfg else 1.0,
        penalty_scale=rl_cfg.penalty_scale if rl_cfg else 1.5,
    )

    # --- Load checkpoint if continuing from saved model ---
    db = ls["db"]
    if body.model_id is not None and db:
        row = db.load_rl_model_by_id(body.model_id)
        if row and row.get("model_blob"):
            agent.load_checkpoint(row["model_blob"])
            logger.info(f"Loaded checkpoint model_id={body.model_id}")

    feature_engine = FeatureEngine(primary_tf=body.timeframe)
    regime_detector = RegimeDetector(primary_tf=body.timeframe)

    bt_config = RLBacktestConfig(
        timeframe=body.timeframe,
        count=body.count,
        epochs=body.epochs,
        train_every=body.train_every,
        profile=body.profile,
    )
    backtester = RLBacktester(
        config=bt_config, agent=agent, env=env,
        feature_engine=feature_engine, regime_detector=regime_detector,
    )

    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_event_loop()

    def _on_epoch(stats: dict):
        loop.call_soon_threadsafe(queue.put_nowait, ("epoch", stats))

    def _on_progress(epoch: int, bar: int, total_bars: int):
        pct = round(bar / max(total_bars, 1) * 100, 1)
        loop.call_soon_threadsafe(
            queue.put_nowait,
            ("progress", {"epoch": epoch, "bar": bar, "total_bars": total_bars, "pct": pct}),
        )

    async def _run_backtest():
        try:
            result = None
            for ds_idx, df in enumerate(datasets):
                result = await asyncio.to_thread(
                    backtester.run, df, None, _on_epoch, _on_progress,
                )

            if result is None:
                await queue.put(("error", "No data to train on"))
                return

            # Save checkpoint
            if db:
                try:
                    blob = agent.save_checkpoint()
                    if blob:
                        db.save_rl_model(
                            model_name=body.profile, model_blob=blob,
                            episode=agent.episode, epsilon=agent.epsilon,
                            total_reward=agent.total_reward, win_rate=result.win_rate,
                            profile=body.profile, timeframe=body.timeframe,
                        )
                except Exception as e:
                    logger.warning(f"Failed to save RL checkpoint: {e}")

            # Build done payload (same shape as RLBacktestResponse)
            done_data = {
                "total_trades": result.total_trades,
                "win_rate": result.win_rate,
                "total_pnl": result.total_pnl,
                "total_reward": result.total_reward,
                "avg_reward_per_trade": result.avg_reward_per_trade,
                "max_drawdown": result.max_drawdown,
                "final_epsilon": result.final_epsilon,
                "episodes": result.episodes,
                "epochs_completed": result.epochs_completed,
                "profile": result.profile,
                "equity_curve": [
                    {"trade_index": pt["trade_index"], "cumulative_pnl": pt["cumulative_pnl"]}
                    for pt in result.equity_curve
                ],
                "trades": [
                    {
                        "epoch": t.epoch, "action": t.action,
                        "entry_price": t.entry_price, "exit_price": t.exit_price,
                        "pnl": t.pnl, "reward": t.reward,
                        "hold_bars": t.hold_bars, "bar_index": t.bar_index,
                        "exit_reason": t.exit_reason,
                        "sl_price": t.sl_price, "tp_price": t.tp_price,
                    }
                    for t in result.trades
                ],
                "epoch_stats": result.epoch_stats,
                "claude_review": None,
            }
            await queue.put(("done", done_data))
        except Exception as e:
            logger.error(f"RL backtest stream error: {traceback.format_exc()}")
            await queue.put(("error", str(e)))

    async def sse_generator():
        task = asyncio.create_task(_run_backtest())
        try:
            while True:
                event_type, data = await queue.get()
                yield f"event: {event_type}\ndata: {json.dumps(data)}\n\n"
                if event_type in ("done", "error"):
                    break
        finally:
            if not task.done():
                task.cancel()

    return StreamingResponse(sse_generator(), media_type="text/event-stream")


PIPELINE_PRESETS: dict[str, list[dict]] = {
    "scalping": [
        {"timeframe": "M1", "count": 5000, "epochs": 15, "symbols": ["GC=F"], "augmentation_factor": 3},
    ],
    "quick": [
        {"timeframe": "H1", "count": 5000, "epochs": 5, "symbols": ["GC=F"], "augmentation_factor": 2},
    ],
    "standard": [
        {"timeframe": "D1", "count": 5000, "epochs": 3, "symbols": ["GC=F", "SI=F"], "augmentation_factor": 2},
        {"timeframe": "H1", "count": 5000, "epochs": 5, "symbols": ["GC=F"], "augmentation_factor": 3},
    ],
    "thorough": [
        {"timeframe": "D1", "count": 5000, "epochs": 3, "symbols": ["GC=F", "SI=F", "CL=F", "ES=F"], "augmentation_factor": 3},
        {"timeframe": "H4", "count": 5000, "epochs": 3, "symbols": ["GC=F", "SI=F"], "augmentation_factor": 2},
        {"timeframe": "H1", "count": 5000, "epochs": 5, "symbols": ["GC=F"], "augmentation_factor": 3},
        {"timeframe": "M15", "count": 5000, "epochs": 3, "symbols": ["GC=F"], "augmentation_factor": 2},
    ],
}


@router.post("/train-pipeline/stream")
async def run_pipeline_stream(request: Request, body: PipelineRequest):
    """Multi-step training pipeline with SSE streaming.

    Runs through multiple timeframe/symbol steps sequentially,
    persisting the same agent across all steps for progressive learning.
    """
    from core.learning.feature_engine import FeatureEngine
    from core.learning.regime_detector import RegimeDetector
    from core.learning.agent_factory import create_agent
    from core.learning.rl_backtester import RLBacktestConfig, RLBacktester
    from core.learning.rl_environment import TradingEnvironment

    ls = _get_learning_state(request)
    config = ls["config"]

    # Reject if a pipeline is already running
    existing_task = getattr(request.app.state, "pipeline_task", None)
    if existing_task and not existing_task.done():
        raise HTTPException(status_code=409, detail="A pipeline is already running")

    # Resolve preset or use explicit steps
    if body.preset and body.preset in PIPELINE_PRESETS:
        steps = [PipelineStep(**s) for s in PIPELINE_PRESETS[body.preset]]
    elif body.steps:
        steps = body.steps
    else:
        raise HTTPException(status_code=400, detail="Provide steps or a valid preset")

    # --- Set up persistent RL agent ---
    rl_cfg = config.rl if config else None
    agent = create_agent(config)
    env = TradingEnvironment(
        reward_scale=rl_cfg.reward_scale if rl_cfg else 1.0,
        penalty_scale=rl_cfg.penalty_scale if rl_cfg else 1.5,
    )

    # Load checkpoint if continuing
    db = ls["db"]
    if body.model_id is not None and db:
        row = db.load_rl_model_by_id(body.model_id)
        if row and row.get("model_blob"):
            agent.load_checkpoint(row["model_blob"])
            logger.info(f"Pipeline: loaded checkpoint model_id={body.model_id}")

    # Use app.state to persist pipeline across page refreshes
    app_state = request.app.state
    cancelled = {"value": False}
    app_state.pipeline_cancelled = cancelled
    app_state.pipeline_logs = []
    app_state.pipeline_progress = {
        "running": True,
        "step": 0,
        "total_steps": len(steps),
        "step_timeframe": "",
        "pct": 0,
    }

    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_event_loop()

    async def _run_pipeline():
        try:
            last_result = None
            for step_idx, step in enumerate(steps):
                if cancelled["value"]:
                    break

                app_state.pipeline_progress.update({
                    "step": step_idx,
                    "step_timeframe": step.timeframe,
                    "pct": round(step_idx / len(steps) * 100, 1),
                })

                # Notify step start
                step_start_data = {
                    "step": step_idx,
                    "total_steps": len(steps),
                    "timeframe": step.timeframe,
                    "symbols": step.symbols,
                    "epochs": step.epochs,
                    "augmentation_factor": step.augmentation_factor,
                }
                app_state.pipeline_logs.append(
                    f"--- Step {step_idx + 1}/{len(steps)}: {step.timeframe} | "
                    f"{', '.join(step.symbols)} | {step.epochs} epochs | aug={step.augmentation_factor}x ---"
                )
                try:
                    queue.put_nowait(("step_start", step_start_data))
                except asyncio.QueueFull:
                    pass

                # Fetch data for this step
                try:
                    frames = await _fetch_multi_symbol_data(
                        step.symbols, step.timeframe, step.count,
                    )
                    datasets = _build_training_datasets(frames, step.augmentation_factor)
                except Exception as e:
                    app_state.pipeline_progress["running"] = False
                    app_state.pipeline_logs.append(f"ERROR: Step {step_idx} data fetch failed: {e}")
                    try:
                        queue.put_nowait(("error", f"Step {step_idx} data fetch failed: {e}"))
                    except asyncio.QueueFull:
                        pass
                    return

                # Build backtester for this step (same agent/env persists)
                feature_engine = FeatureEngine(primary_tf=step.timeframe)
                regime_detector = RegimeDetector(primary_tf=step.timeframe)

                bt_config = RLBacktestConfig(
                    timeframe=step.timeframe,
                    count=step.count,
                    epochs=step.epochs,
                    train_every=body.train_every,
                    profile=body.profile,
                )
                backtester = RLBacktester(
                    config=bt_config, agent=agent, env=env,
                    feature_engine=feature_engine, regime_detector=regime_detector,
                )

                def _on_epoch(stats: dict, _si=step_idx):
                    stats["step"] = _si
                    wr = (stats.get("win_rate", 0) * 100)
                    app_state.pipeline_logs.append(
                        f"  Epoch {stats.get('epoch', '?')}: {stats.get('trades', 0)} trades | "
                        f"WR {wr:.0f}% | reward {stats.get('total_reward', 0):.2f} | "
                        f"eps {stats.get('epsilon', 1.0):.4f}"
                    )
                    try:
                        loop.call_soon_threadsafe(queue.put_nowait, ("epoch", stats))
                    except Exception:
                        pass

                def _on_progress(epoch: int, bar: int, total_bars: int, _si=step_idx):
                    pct = round(bar / max(total_bars, 1) * 100, 1)
                    overall = round((_si + pct / 100) / len(steps) * 100, 1)
                    app_state.pipeline_progress.update({
                        "pct": overall,
                        "step": _si,
                    })
                    try:
                        loop.call_soon_threadsafe(
                            queue.put_nowait,
                            ("progress", {
                                "step": _si, "epoch": epoch,
                                "bar": bar, "total_bars": total_bars, "pct": pct,
                            }),
                        )
                    except Exception:
                        pass

                # Train on each dataset in this step
                for df in datasets:
                    if cancelled["value"]:
                        break
                    last_result = await asyncio.to_thread(
                        backtester.run, df, None, _on_epoch, _on_progress,
                    )

                # Save checkpoint after each step
                if db:
                    try:
                        blob = agent.save_checkpoint()
                        if blob:
                            db.save_rl_model(
                                model_name=f"pipeline_{body.profile}_{step.timeframe}",
                                model_blob=blob,
                                episode=agent.episode,
                                epsilon=agent.epsilon,
                                total_reward=agent.total_reward,
                                win_rate=last_result.win_rate if last_result else 0.0,
                                profile=body.profile,
                                timeframe=step.timeframe,
                            )
                    except Exception as e:
                        logger.warning(f"Pipeline: failed to save step checkpoint: {e}")

                step_done_data = {
                    "step": step_idx,
                    "timeframe": step.timeframe,
                    "trades": last_result.total_trades if last_result else 0,
                    "win_rate": last_result.win_rate if last_result else 0.0,
                    "epsilon": round(agent.epsilon, 4),
                }
                app_state.pipeline_logs.append(
                    f"  Step done: {step.timeframe} | "
                    f"{step_done_data['trades']} trades | WR {step_done_data['win_rate']*100:.1f}% | saved"
                )
                try:
                    queue.put_nowait(("step_done", step_done_data))
                except asyncio.QueueFull:
                    pass

            # Pipeline complete
            summary = {
                "steps_completed": len(steps) if not cancelled["value"] else step_idx,
                "total_episodes": agent.episode,
                "final_epsilon": round(agent.epsilon, 4),
                "cancelled": cancelled["value"],
            }
            if last_result:
                summary["final_win_rate"] = last_result.win_rate
                summary["final_total_reward"] = round(agent.total_reward, 4)

            done_msg = (
                f"--- Pipeline cancelled after {summary['steps_completed']} steps ---"
                if cancelled["value"]
                else f"--- Pipeline complete: {summary['steps_completed']} steps | "
                     f"{summary['total_episodes']} episodes | eps {summary['final_epsilon']:.4f} ---"
            )
            app_state.pipeline_logs.append(done_msg)
            app_state.pipeline_progress["running"] = False
            try:
                queue.put_nowait(("pipeline_done", summary))
            except asyncio.QueueFull:
                pass

        except Exception as e:
            logger.error(f"Pipeline error: {traceback.format_exc()}")
            # Save partial checkpoint on error
            if db:
                try:
                    blob = agent.save_checkpoint()
                    if blob:
                        db.save_rl_model(
                            model_name=f"pipeline_{body.profile}_partial",
                            model_blob=blob,
                            episode=agent.episode,
                            epsilon=agent.epsilon,
                            total_reward=agent.total_reward,
                            win_rate=0.0,
                            profile=body.profile,
                            timeframe="partial",
                        )
                except Exception:
                    pass
            app_state.pipeline_progress["running"] = False
            app_state.pipeline_logs.append(f"ERROR: {e}")
            try:
                queue.put_nowait(("error", str(e)))
            except asyncio.QueueFull:
                pass

    task = asyncio.create_task(_run_pipeline())
    app_state.pipeline_task = task

    async def sse_generator():
        try:
            while True:
                try:
                    event_type, data = await asyncio.wait_for(queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    # Check if task is done but we missed the event
                    if task.done():
                        break
                    continue
                yield f"event: {event_type}\ndata: {json.dumps(data)}\n\n"
                if event_type in ("pipeline_done", "error"):
                    break
        finally:
            # Do NOT cancel the task — let it run in background
            pass

    return StreamingResponse(sse_generator(), media_type="text/event-stream")


@router.get("/pipeline-status")
async def get_pipeline_status(request: Request) -> dict:
    """Check if a pipeline training is currently running."""
    state = request.app.state
    task = getattr(state, "pipeline_task", None)
    is_running = task is not None and not task.done()
    progress = getattr(state, "pipeline_progress", None)
    logs = getattr(state, "pipeline_logs", [])

    return {
        "running": is_running,
        "progress": progress if is_running else None,
        "logs": logs[-50:],
    }


@router.post("/cancel-pipeline")
async def cancel_pipeline(request: Request) -> dict:
    """Cancel a running pipeline training."""
    state = request.app.state
    cancelled = getattr(state, "pipeline_cancelled", None)
    if cancelled is not None:
        cancelled["value"] = True
    return {"status": "ok"}


@router.get("/models", response_model=RLModelListResponse)
async def list_rl_models(request: Request, profile: str = "") -> RLModelListResponse:
    """List saved RL models, optionally filtered by profile."""
    ls = _get_learning_state(request)
    db = ls["db"]
    if not db:
        return RLModelListResponse()

    rows = db.list_rl_models(profile=profile)
    return RLModelListResponse(
        models=[
            RLModelInfo(
                id=r["id"],
                model_name=r.get("model_name", ""),
                episode=r.get("episode", 0),
                epsilon=r.get("epsilon", 1.0),
                total_reward=r.get("total_reward", 0.0),
                win_rate=r.get("win_rate", 0.0),
                profile=r.get("profile"),
                timeframe=r.get("timeframe"),
                created_at=r.get("created_at"),
            )
            for r in rows
        ]
    )


@router.delete("/models/{model_id}")
async def delete_rl_model(request: Request, model_id: int) -> dict:
    """Delete a saved RL model by ID."""
    ls = _get_learning_state(request)
    db = ls["db"]
    if not db:
        raise HTTPException(status_code=503, detail="Database not available")

    deleted = db.delete_rl_model(model_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Model not found")
    return {"status": "ok", "deleted_id": model_id}


@router.post("/models/activate")
async def activate_rl_model(request: Request, body: RLModelActivateRequest) -> dict:
    """Load a specific RL model into the live agent."""
    ls = _get_learning_state(request)
    db = ls["db"]
    agent = ls["rl_agent"]

    if not db:
        raise HTTPException(status_code=503, detail="Database not available")
    if not agent:
        raise HTTPException(status_code=503, detail="RL agent not initialized")

    row = db.load_rl_model_by_id(body.model_id)
    if not row:
        raise HTTPException(status_code=404, detail="Model not found")
    if not row.get("model_blob"):
        raise HTTPException(status_code=400, detail="Model has no checkpoint data")

    agent.load_checkpoint(row["model_blob"])

    # Update active model tracking on app state
    request.app.state.active_model_id = row["id"]
    request.app.state.active_model_name = row.get("model_name")
    request.app.state.active_model_profile = row.get("profile")
    request.app.state.active_model_timeframe = row.get("timeframe")

    logger.info(
        f"Activated RL model id={row['id']} "
        f"(profile={row.get('profile', '?')}, episode={row.get('episode', '?')})"
    )
    return {
        "status": "ok",
        "model_id": row["id"],
        "profile": row.get("profile", ""),
        "episode": row.get("episode", 0),
    }
