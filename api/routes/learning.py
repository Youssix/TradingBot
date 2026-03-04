"""REST API endpoints for the learning system."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from loguru import logger

from api.schemas import (
    ClaudeInsight,
    EquityCurvePoint,
    LearningStatus,
    RegimeEntry,
    RLBacktestEpochStats,
    RLBacktestRequest,
    RLBacktestResponse,
    RLBacktestTradeResponse,
    RLStats,
    StrategyPerformance,
    StrategyWeight,
    ToggleLearningRequest,
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


@router.post("/backtest", response_model=RLBacktestResponse)
async def run_rl_backtest(
    request: Request, body: RLBacktestRequest
) -> RLBacktestResponse:
    """Train RL agent on historical candles and return results."""
    import pandas as pd

    from core.learning.feature_engine import FeatureEngine
    from core.learning.regime_detector import RegimeDetector
    from core.learning.rl_agent import RLAgent
    from core.learning.rl_backtester import RLBacktestConfig, RLBacktester
    from core.learning.rl_environment import TradingEnvironment

    ls = _get_learning_state(request)
    config = ls["config"]

    # --- Fetch historical data via YFinance ---
    try:
        import yfinance as yf

        ticker = yf.Ticker("GC=F")

        tf_map = {
            "M1": ("1m", "5d"),
            "M5": ("5m", "60d"),
            "M15": ("15m", "60d"),
            "H1": ("1h", "730d"),
            "H4": ("1h", "730d"),
            "D1": ("1d", "max"),
        }
        interval, period = tf_map.get(body.timeframe, ("1h", "730d"))

        df = ticker.history(period=period, interval=interval)
        if df.empty:
            raise HTTPException(status_code=502, detail="YFinance returned no data")

        # Normalize columns
        df = df.reset_index()
        col_map = {
            "Datetime": "datetime",
            "Date": "datetime",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
        df = df.rename(columns=col_map)
        df = df[["datetime", "open", "high", "low", "close", "volume"]].copy()

        if hasattr(df["datetime"].dtype, "tz") and df["datetime"].dtype.tz is not None:
            df["datetime"] = df["datetime"].dt.tz_localize(None)

        # Trim to requested count
        df = df.tail(body.count).reset_index(drop=True)
        logger.info(
            f"RL backtest: fetched {len(df)} {body.timeframe} candles"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=502, detail=f"Failed to fetch data: {e}"
        )

    # --- Set up fresh RL components for the backtest ---
    rl_cfg = config.rl if config else None
    agent = RLAgent(
        state_dim=rl_cfg.state_dim if rl_cfg else 23,
        action_dim=rl_cfg.action_dim if rl_cfg else 3,
        lr=rl_cfg.lr if rl_cfg else 1e-3,
        gamma=rl_cfg.gamma if rl_cfg else 0.95,
        epsilon_start=rl_cfg.epsilon_start if rl_cfg else 1.0,
        epsilon_end=rl_cfg.epsilon_end if rl_cfg else 0.05,
        epsilon_decay=rl_cfg.epsilon_decay if rl_cfg else 0.995,
        buffer_capacity=rl_cfg.buffer_capacity if rl_cfg else 10_000,
        batch_size=rl_cfg.batch_size if rl_cfg else 64,
        target_update_freq=rl_cfg.target_update_freq if rl_cfg else 50,
    )
    env = TradingEnvironment(
        reward_scale=rl_cfg.reward_scale if rl_cfg else 1.0,
        penalty_scale=rl_cfg.penalty_scale if rl_cfg else 1.5,
    )
    feature_engine = FeatureEngine(primary_tf=body.timeframe)
    regime_detector = RegimeDetector(primary_tf=body.timeframe)

    bt_config = RLBacktestConfig(
        timeframe=body.timeframe,
        count=body.count,
        epochs=body.epochs,
        train_every=body.train_every,
    )

    backtester = RLBacktester(
        config=bt_config,
        agent=agent,
        env=env,
        feature_engine=feature_engine,
        regime_detector=regime_detector,
    )

    # --- Run backtest ---
    result = backtester.run(df)

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

        # Store in DB
        db = ls["db"]
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
    db = ls["db"]
    if db:
        try:
            blob = agent.save_checkpoint()
            if blob:
                db.save_rl_checkpoint(
                    blob,
                    episode=agent.episode,
                    epsilon=agent.epsilon,
                    total_reward=agent.total_reward,
                )
                logger.info("RL backtest checkpoint saved to DB")
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
            )
            for t in result.trades
        ],
        epoch_stats=[
            RLBacktestEpochStats(**s) for s in result.epoch_stats
        ],
        claude_review=claude_insight,
    )
