"""FastAPI application entry point for the TradingBot API."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from api.routes import account, backtest, bot, candles, trades, learning, logs
from config import AppConfig, TIMEFRAME_CYCLE_SECONDS
from core.mt5_client import create_mt5_client
from utils.db import TradeDB

# Ensure API logs go to the same bot log file so the dashboard can display them
from pathlib import Path
_log_dir = Path("logs")
_log_dir.mkdir(exist_ok=True)
logger.add(
    str(_log_dir / "bot_{time:YYYY-MM-DD}.log"),
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    rotation="1 day",
    retention="7 days",
    level="INFO",
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage shared resources across the application lifetime."""
    config = AppConfig()

    # Database
    db = TradeDB(db_path=config.db_path)
    db.connect()
    logger.info("TradeDB connected")

    # MT5 client (mock or real based on config)
    mt5_client = create_mt5_client(config)
    mt5_client.connect()
    logger.info("MT5 client connected")

    # Store on app.state for route access
    app.state.config = config
    app.state.db = db
    app.state.mt5_client = mt5_client

    # Strategy mode defaults
    app.state.strategy_mode = "independent"
    app.state.enabled_strategies = ["ema_crossover", "asian_breakout", "bos", "candle_pattern"]

    # Pipeline training state (persists across SSE disconnections)
    app.state.pipeline_task = None
    app.state.pipeline_cancelled = {"value": False}
    app.state.pipeline_progress = None
    app.state.pipeline_logs = []

    # Initialize learning system components
    app.state.learning_enabled = config.learning.enabled
    _init_learning_state(app, config)

    # Start the background learning loop
    learning_task = None
    if app.state.learning_enabled:
        learning_task = asyncio.create_task(_learning_loop(app))
        logger.info("Background learning loop started")

    yield

    # Shutdown
    if learning_task:
        learning_task.cancel()
        try:
            await learning_task
        except asyncio.CancelledError:
            pass

    # Save RL checkpoint before exit
    _save_rl_checkpoint(app)

    mt5_client.disconnect()
    db.close()
    logger.info("Resources released")


def _save_rl_checkpoint(app: FastAPI, profile: str = "medium") -> None:
    """Save RL model checkpoint to DB."""
    agent = getattr(app.state, "rl_agent", None)
    env = getattr(app.state, "rl_env", None)
    db = getattr(app.state, "db", None)
    if agent and db:
        try:
            blob = agent.save_checkpoint()
            if blob:
                db.save_rl_model(
                    model_name=profile,
                    model_blob=blob,
                    episode=agent.episode,
                    epsilon=agent.epsilon,
                    total_reward=agent.total_reward,
                    win_rate=env.win_rate if env else 0.0,
                    profile=profile,
                    timeframe=getattr(app.state.config, "mt5", None) and app.state.config.mt5.timeframe or "",
                )
                logger.info(f"RL checkpoint saved (profile={profile}, episode={agent.episode})")
        except Exception as e:
            logger.error(f"Failed to save RL checkpoint: {e}")


def _apply_claude_rules(
    action: float, features: dict, regime: str, session: str, db,
) -> tuple[float, str]:
    """Check active Claude rules and override action if needed.

    Args:
        action: continuous action float in [-1, +1]

    Returns (possibly overridden action, reason or empty string).
    """
    if not db:
        return action, ""
    try:
        rules = db.get_active_rules()
    except Exception:
        return action, ""

    for rule_set in rules:
        for rule in rule_set.get("rules_json", []):
            cond = rule.get("condition", "")
            rule_action = rule.get("action", "").upper()
            reason = rule.get("reason", "")

            triggered = False
            try:
                if "ADX" in cond:
                    adx = features.get("adx", 25)
                    if "<" in cond:
                        threshold = float(cond.split("<")[-1].strip())
                        triggered = adx < threshold
                    elif ">" in cond:
                        threshold = float(cond.split(">")[-1].strip())
                        triggered = adx > threshold
                elif "confidence" in cond:
                    conf = features.get("confidence", 0.5)
                    if "<" in cond:
                        threshold = float(cond.split("<")[-1].strip())
                        triggered = conf < threshold
                elif "session" in cond:
                    parts = cond.lower().replace("and", "&&").split("&&")
                    all_match = True
                    for part in parts:
                        part = part.strip()
                        if "session" in part and "==" in part:
                            val = part.split("==")[-1].strip()
                            if session.lower() != val:
                                all_match = False
                        elif "regime" in part and "==" in part:
                            val = part.split("==")[-1].strip()
                            if regime.lower() != val:
                                all_match = False
                    triggered = all_match
                elif "RSI" in cond or "rsi" in cond:
                    rsi = features.get("rsi", 50)
                    if ">" in cond:
                        threshold = float(cond.split(">")[-1].strip())
                        triggered = rsi > threshold
                    elif "<" in cond:
                        threshold = float(cond.split("<")[-1].strip())
                        triggered = rsi < threshold
            except (ValueError, IndexError):
                continue

            if triggered and rule_action == "HOLD":
                return 0.0, reason  # 0.0 = hold in continuous space

    return action, ""


async def _learning_loop(app: FastAPI) -> None:
    """Background loop: fetch data, extract features, step RL, train.

    Runs every cycle_interval seconds (60s for M1, 300s for M5, etc.).
    This is what makes the Strategy Builder tab show live data.

    FEEDBACK LOOP:
    1. RL agent learns from market data → makes BUY/SELL/HOLD decisions
    2. Claude rules can override RL actions (filter bad trades)
    3. Completed trades are saved to DB (visible in dashboard)
    4. Every N trades, Claude auto-reviews and generates new rules
    5. New rules feed back to step 2 → loop closes
    """
    config: AppConfig = app.state.config
    candle_seconds = TIMEFRAME_CYCLE_SECONDS.get(config.mt5.timeframe, 60)
    cycle_seconds = min(candle_seconds, 5)
    step_count = 0
    rl_trade_count = 0  # Trades since last Claude review
    last_state: list[float] | None = None
    REVIEW_EVERY_N_TRADES = 10  # Auto-trigger Claude review after this many trades

    logger.info(
        f"Learning loop: cycle={cycle_seconds}s (candle TF={candle_seconds}s), "
        f"TF={config.mt5.timeframe}, "
        f"context={app.state.feature_engine.context_timeframes if app.state.feature_engine else '?'}"
    )

    while True:
        try:
            if not getattr(app.state, "learning_enabled", False):
                await asyncio.sleep(cycle_seconds)
                continue

            mt5 = app.state.mt5_client
            fe = app.state.feature_engine
            rd = app.state.regime_detector
            agent = app.state.rl_agent
            env = app.state.rl_env
            tracker = app.state.performance_tracker
            db = app.state.db

            if not all([mt5, fe, rd, agent, env]):
                await asyncio.sleep(cycle_seconds)
                continue

            # --- 1. Fetch market data ---
            rates = mt5.get_rates(config.mt5.symbol, config.mt5.timeframe, 200)

            htf_data = {}
            for tf in fe.context_timeframes:
                try:
                    htf_data[tf] = mt5.get_rates(config.mt5.symbol, tf, 100)
                except Exception:
                    pass

            # --- 2. Extract features ---
            features = fe.extract(rates, htf_data)
            state = features["vector"]
            regime = rd.current.regime.value if rd.current else "unknown"
            session = features.get("session", "unknown")

            # --- 3. Update regime ---
            rd.update(features)
            regime = rd.current.regime.value

            # --- 4. RL agent decides (continuous action) ---
            action, info = agent.select_action(state)

            # --- 4b. Apply Claude rules as filters ---
            original_action = action
            action, override_reason = _apply_claude_rules(
                action, features, regime, session, db,
            )
            if override_reason:
                logger.debug(
                    f"Claude rule overrode RL action "
                    f"{original_action:.2f}→{action:.2f}: {override_reason}"
                )

            # --- 5. Execute action in environment ---
            bar_atr = float(features.get("atr", 0.0))
            bar_high = float(rates.iloc[-1]["high"])
            bar_low = float(rates.iloc[-1]["low"])
            reward, trade_result = env.step(
                action, features["close"], state,
                atr=bar_atr, high=bar_high, low=bar_low,
            )

            # --- 6. Store transition in replay buffer ---
            from core.learning.replay_buffer import Transition
            agent.store_transition(Transition(
                state=last_state if last_state else state,
                action=action,
                reward=reward,
                next_state=state,
                done=trade_result is not None,
            ))
            last_state = state

            if tracker:
                tracker.record_rl_reward(reward)

            # --- 7. If a trade completed → save to DB + trigger review ---
            if trade_result:
                agent.end_episode(trade_result.reward)

                # Convert raw point P&L to dollar P&L using real position sizing
                point_value = 10.0  # $10 per point per lot for gold
                sl_distance = abs(trade_result.entry_price - trade_result.sl_price) if trade_result.sl_price > 0 else 0.0
                account_info = mt5.get_account_info()
                balance = account_info.get("balance", 10_000.0)
                risk_pct = config.risk.risk_pct

                if sl_distance > 0 and point_value > 0:
                    risk_amount = balance * (risk_pct / 100.0)
                    lot_size = round(max(risk_amount / (sl_distance * point_value), 0.01), 2)
                else:
                    lot_size = 0.01

                dollar_pnl = round(trade_result.pnl * lot_size * point_value, 2)

                mt5.realize_pnl(dollar_pnl)
                if tracker:
                    tracker.record_trade("rl_agent", dollar_pnl)

                # Save RL trade to trades DB table (visible in dashboard)
                if db:
                    try:
                        from datetime import datetime, timezone
                        now = datetime.now(timezone.utc).isoformat()
                        trade_id = db.insert_trade({
                            "strategy": "rl_agent",
                            "symbol": config.mt5.symbol,
                            "direction": trade_result.action.upper(),
                            "entry_price": trade_result.entry_price,
                            "exit_price": trade_result.exit_price,
                            "sl": trade_result.sl_price,
                            "tp": trade_result.tp_price,
                            "lot_size": lot_size,
                            "pnl": dollar_pnl,
                            "opened_at": now,
                            "closed_at": now,
                            "status": "closed",
                        })
                        # Save trade features for Claude analysis
                        q_val = info.get("q_value", 0.0) if info else 0.0
                        db.insert_trade_features(
                            trade_id=trade_id,
                            features={"vector": features["vector"]},
                            regime=regime,
                            confidence=abs(q_val),
                            rl_action=action,
                            rl_q_values=[action],
                            claude_reasoning=override_reason,
                        )
                    except Exception as e:
                        logger.error(f"Failed to save RL trade to DB: {e}")

                rl_trade_count += 1
                logger.info(
                    f"RL trade #{rl_trade_count} closed: {trade_result.action} | "
                    f"P&L=${dollar_pnl:.2f} ({lot_size} lots) | "
                    f"reward={trade_result.reward:.4f} | "
                    f"held {trade_result.hold_bars} bars | regime={regime}"
                )

                # --- 8. Auto-trigger Claude review every N trades ---
                if rl_trade_count % REVIEW_EVERY_N_TRADES == 0:
                    await _auto_claude_review(app, regime)

            # --- 9. Train from replay buffer ---
            loss = agent.train_step()

            step_count += 1

            if step_count % 10 == 0:
                logger.info(
                    f"Learning step {step_count} | "
                    f"regime={regime} | "
                    f"epsilon={agent.epsilon:.4f} | "
                    f"buffer={agent.buffer_size} | "
                    f"episodes={agent.episode} | "
                    f"rl_trades={rl_trade_count} | "
                    f"reward={agent.total_reward:.2f}"
                    + (f" | loss={loss:.4f}" if loss else "")
                )

            if step_count % 50 == 0:
                ensemble = getattr(app.state, "ensemble", None)
                if ensemble:
                    ensemble.update_weights()

            if step_count % 5 == 0 and db:
                try:
                    db.insert_market_features(
                        timestamp=str(features.get("timestamp", "")),
                        features={"vector": features["vector"]},
                        regime=regime,
                        session=session,
                    )
                except Exception:
                    pass

            if step_count % 100 == 0:
                _save_rl_checkpoint(app)

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"Learning loop error: {e}")

        await asyncio.sleep(cycle_seconds)


async def _auto_claude_review(app: FastAPI, regime: str) -> None:
    """Auto-trigger Claude review of recent RL trades.

    Called every N trades. Claude analyzes patterns, generates rules,
    and stores them in DB so they feed back into the RL decision filter.
    """
    reviewer = getattr(app.state, "claude_reviewer", None)
    db = getattr(app.state, "db", None)
    agent = getattr(app.state, "rl_agent", None)

    if not db:
        return

    try:
        # Get recent closed trades with features
        trades = db.get_recent_closed_trades(30)
        if not trades:
            return

        rl_stats = agent.get_stats() if agent else None

        # Run review (Claude API or fallback rule-based)
        if reviewer:
            result = reviewer.review_trades(trades, rl_stats, regime)
        else:
            return

        # Store insight in DB
        db.insert_claude_insight(
            review_type="auto_review",
            analysis=result.get("analysis", ""),
            recommendations=result.get("recommendations", []),
            strategy_rules=result.get("strategy_rules", []),
            backtest_score=result.get("score", 0.5),
        )

        # Store generated rules so they feed back into action filtering
        rules = result.get("strategy_rules", [])
        if rules:
            db.insert_strategy_rule(
                name=f"claude_auto_{int(asyncio.get_event_loop().time())}",
                rules=rules,
                source="claude_auto_review",
                backtest_win_rate=result.get("score", 0.0),
            )
            logger.info(
                f"Claude auto-review: {len(rules)} new rules generated | "
                f"Score: {result.get('score', 0):.2f} | "
                f"Analysis: {result.get('analysis', '')[:80]}"
            )
        else:
            logger.info(f"Claude auto-review complete (no new rules): {result.get('analysis', '')[:80]}")

    except Exception as e:
        logger.error(f"Auto Claude review failed: {e}")


def _init_learning_state(app: FastAPI, config: AppConfig) -> None:
    """Initialize learning components and attach to app.state."""
    if not config.learning.enabled:
        app.state.feature_engine = None
        app.state.regime_detector = None
        app.state.rl_agent = None
        app.state.rl_env = None
        app.state.performance_tracker = None
        app.state.ensemble = None
        app.state.claude_strategy = None
        app.state.claude_reviewer = None
        app.state.confidence_scorer = None
        app.state.active_model_id = None
        app.state.active_model_name = None
        app.state.active_model_profile = None
        app.state.active_model_timeframe = None
        return

    try:
        from core.learning.feature_engine import FeatureEngine
        from core.learning.regime_detector import RegimeDetector
        from core.learning.agent_factory import create_agent
        from core.learning.rl_environment import TradingEnvironment
        from core.learning.performance_tracker import PerformanceTracker
        from core.learning.confidence_scorer import ConfidenceScorer
        from core.learning.ensemble import StrategyEnsemble
        from core.learning.claude_strategy import ClaudeStrategy
        from core.learning.claude_reviewer import ClaudeReviewer

        tf = config.mt5.timeframe

        app.state.feature_engine = FeatureEngine(primary_tf=tf)
        app.state.regime_detector = RegimeDetector(primary_tf=tf)
        app.state.rl_agent = create_agent(config)
        app.state.rl_env = TradingEnvironment(
            reward_scale=config.rl.reward_scale,
            penalty_scale=config.rl.penalty_scale,
        )

        # Wrap env with composite reward if enabled
        if config.composite_reward.enabled:
            from core.learning.composite_reward import CompositeRewardWrapper
            app.state.rl_env = CompositeRewardWrapper(
                env=app.state.rl_env,
                drawdown_weight=config.composite_reward.drawdown_weight,
                drawdown_threshold=config.composite_reward.drawdown_threshold,
                sortino_weight=config.composite_reward.sortino_weight,
                sortino_window=config.composite_reward.sortino_window,
                consistency_weight=config.composite_reward.consistency_weight,
            )
        app.state.performance_tracker = PerformanceTracker(
            window_size=config.ensemble.performance_window,
        )
        app.state.confidence_scorer = ConfidenceScorer()
        app.state.ensemble = StrategyEnsemble(
            performance_tracker=app.state.performance_tracker,
            base_weights={
                "ema_crossover": config.ensemble.ema_weight,
                "asian_breakout": config.ensemble.breakout_weight,
                "rl_agent": config.ensemble.rl_weight,
                "claude_ai": config.ensemble.claude_weight,
            },
            agreement_bonus=config.ensemble.agreement_bonus,
            confidence_threshold=config.learning.confidence_threshold,
        )
        app.state.claude_strategy = ClaudeStrategy(
            api_key=config.claude.api_key,
            model=config.claude.model,
            min_interval=config.claude.strategy_interval,
            max_tokens=config.claude.max_tokens_strategy,
        )
        app.state.claude_reviewer = ClaudeReviewer(
            api_key=config.claude.api_key,
            model=config.claude.model,
            max_tokens=config.claude.max_tokens_review,
        )

        # Try loading latest RL checkpoint from DB (try each profile, fall back to "dqn")
        checkpoint = None
        for profile_name in ("medium", "conservative", "aggressive", "max_profit", "dqn"):
            checkpoint = app.state.db.load_latest_rl_model(profile_name)
            if checkpoint and checkpoint.get("model_blob"):
                try:
                    app.state.rl_agent.load_checkpoint(checkpoint["model_blob"])
                    app.state.active_model_id = checkpoint.get("id")
                    app.state.active_model_name = checkpoint.get("model_name")
                    app.state.active_model_profile = checkpoint.get("profile")
                    app.state.active_model_timeframe = checkpoint.get("timeframe")
                    logger.info(f"Loaded RL model: profile={profile_name}, id={checkpoint.get('id')}")
                    break
                except Exception as ckpt_err:
                    logger.warning(
                        f"Skipping incompatible checkpoint profile={profile_name}: {ckpt_err}"
                    )
                    continue
        else:
            app.state.active_model_id = None
            app.state.active_model_name = None
            app.state.active_model_profile = None
            app.state.active_model_timeframe = None
            logger.info("No compatible RL model found, starting fresh")

        logger.info("Learning system initialized for API")
    except Exception as e:
        logger.error(f"Failed to init learning for API: {e}")
        app.state.learning_enabled = False
        app.state.feature_engine = None
        app.state.regime_detector = None
        app.state.rl_agent = None
        app.state.rl_env = None
        app.state.performance_tracker = None
        app.state.ensemble = None
        app.state.claude_strategy = None
        app.state.claude_reviewer = None
        app.state.confidence_scorer = None
        app.state.active_model_id = None
        app.state.active_model_name = None
        app.state.active_model_profile = None
        app.state.active_model_timeframe = None


app = FastAPI(
    title="TradingBot API",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS – allow the Vite dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(account.router)
app.include_router(candles.router)
app.include_router(trades.router)
app.include_router(backtest.router)
app.include_router(bot.router)
app.include_router(learning.router)
app.include_router(logs.router)
