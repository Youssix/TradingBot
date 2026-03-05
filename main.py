from __future__ import annotations

import argparse
import signal
import sys
import threading
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from apscheduler.schedulers.background import BackgroundScheduler
from loguru import logger

from config import AppConfig, TIMEFRAME_CYCLE_SECONDS
from core.mt5_client import create_mt5_client
from core.risk_manager import RiskConfig, RiskManager
from core.strategy import (
    AsianRangeBreakoutStrategy,
    BOSStrategy,
    CandlePatternStrategy,
    EMACrossoverStrategy,
    Strategy,
)
from core.trade_executor import TradeExecutor
from utils.db import TradeDB
from utils.logger import setup_logger


class TradingBot:
    """Main trading bot orchestrator — scalping mode (1-second tick loop)."""

    def __init__(self, config: AppConfig, strategies: list[Strategy]) -> None:
        self._config = config
        self._strategies = strategies
        self._client = create_mt5_client(config)
        self._risk = RiskManager(RiskConfig(
            risk_pct=config.risk.risk_pct,
            max_open_trades=config.risk.max_open_trades,
            max_daily_trades=config.risk.max_daily_trades,
            max_daily_drawdown_pct=config.risk.max_daily_drawdown_pct,
            max_total_drawdown_pct=config.risk.max_total_drawdown_pct,
            friday_cutoff_hour=config.risk.friday_cutoff_hour,
            news_hours=list(config.risk.news_hours) if config.risk.news_hours else None,
            max_spread_pips=config.risk.max_spread_pips,
        ))
        self._db = TradeDB(config.db_path)
        self._executor = TradeExecutor(
            mt5_client=self._client,
            risk_manager=self._risk,
            db=self._db,
            dry_run=(config.mode == "dry-run"),
            slippage_pips=config.slippage_pips,
            breakeven_trigger_atr=config.ema_strategy.breakeven_trigger_atr,
        )
        self._scheduler: BackgroundScheduler | None = None
        self._initial_balance: float | None = None
        self._running = True
        self._lock = threading.Lock()
        # Throttle signal checks — only look for new entries every N seconds
        self._signal_interval = 5  # check strategies every 5s
        self._last_signal_check = 0.0

        # Learning system components (lazy init)
        self._learning_enabled = config.learning.enabled
        self._feature_engine = None
        self._regime_detector = None
        self._rl_agent = None
        self._rl_env = None
        self._performance_tracker = None
        self._confidence_scorer = None
        self._ensemble = None
        self._claude_strategy = None
        self._claude_reviewer = None
        self._learning_step_count = 0
        self._last_features: dict[str, Any] | None = None
        self._last_state: list[float] | None = None

        # Strategy mode: "independent" or "ensemble"
        self.strategy_mode: str = "independent"
        self.enabled_strategies: list[str] = [s.name for s in strategies]

    def _init_learning(self) -> None:
        """Initialize learning system components."""
        if not self._learning_enabled:
            logger.info("Learning system disabled")
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

            tf = self._config.mt5.timeframe
            cfg = self._config

            self._feature_engine = FeatureEngine(primary_tf=tf)
            self._regime_detector = RegimeDetector(primary_tf=tf)
            self._rl_agent = create_agent(cfg)
            self._rl_env = TradingEnvironment(
                reward_scale=cfg.rl.reward_scale,
                penalty_scale=cfg.rl.penalty_scale,
            )

            # Wrap env with composite reward if enabled
            if cfg.composite_reward.enabled:
                from core.learning.composite_reward import CompositeRewardWrapper
                self._rl_env = CompositeRewardWrapper(
                    env=self._rl_env,
                    drawdown_weight=cfg.composite_reward.drawdown_weight,
                    drawdown_threshold=cfg.composite_reward.drawdown_threshold,
                    sortino_weight=cfg.composite_reward.sortino_weight,
                    sortino_window=cfg.composite_reward.sortino_window,
                    consistency_weight=cfg.composite_reward.consistency_weight,
                )
            self._performance_tracker = PerformanceTracker(
                window_size=cfg.ensemble.performance_window,
            )
            self._confidence_scorer = ConfidenceScorer()
            self._ensemble = StrategyEnsemble(
                performance_tracker=self._performance_tracker,
                base_weights={
                    "ema_crossover": cfg.ensemble.ema_weight,
                    "asian_breakout": cfg.ensemble.breakout_weight,
                    "rl_agent": cfg.ensemble.rl_weight,
                    "claude_ai": cfg.ensemble.claude_weight,
                },
                agreement_bonus=cfg.ensemble.agreement_bonus,
                confidence_threshold=cfg.learning.confidence_threshold,
            )
            self._claude_strategy = ClaudeStrategy(
                api_key=cfg.claude.api_key,
                model=cfg.claude.model,
                min_interval=cfg.claude.strategy_interval,
                max_tokens=cfg.claude.max_tokens_strategy,
            )
            self._claude_reviewer = ClaudeReviewer(
                api_key=cfg.claude.api_key,
                model=cfg.claude.model,
                max_tokens=cfg.claude.max_tokens_review,
            )

            # Try loading latest RL checkpoint
            model_name = cfg.rl.agent_type
            checkpoint = self._db.load_latest_rl_model(model_name)
            if checkpoint and checkpoint.get("model_blob"):
                self._rl_agent.load_checkpoint(checkpoint["model_blob"])

            logger.info("Learning system initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize learning system: {e}")
            self._learning_enabled = False

    def start(self) -> None:
        """Start the trading bot."""
        setup_logger(self._config.log_level)
        logger.info(f"Starting trading bot in {self._config.mode} mode")

        self._db.connect()
        self._client.connect()

        account = self._client.get_account_info()
        self._initial_balance = account["balance"]
        logger.info(f"Account balance: {self._initial_balance:.2f}")

        # Initialize learning system
        self._init_learning()

        self._scheduler = BackgroundScheduler()

        # Fast tick loop — manage open trades every 1 second
        self._scheduler.add_job(
            self._tick_loop,
            "interval",
            seconds=1,
            id="tick_loop",
            max_instances=1,
        )

        # Signal scan — look for new entries every 5 seconds
        self._scheduler.add_job(
            self._signal_scan,
            "interval",
            seconds=self._signal_interval,
            id="signal_scan",
            next_run_time=datetime.now(),
            max_instances=1,
        )

        # Learning cycle — run at timeframe-adaptive interval
        if self._learning_enabled:
            cycle_seconds = TIMEFRAME_CYCLE_SECONDS.get(self._config.mt5.timeframe, 60)
            self._scheduler.add_job(
                self._learning_cycle,
                "interval",
                seconds=cycle_seconds,
                id="learning_cycle",
                max_instances=1,
            )
            # RL training — every N steps
            self._scheduler.add_job(
                self._rl_train_step,
                "interval",
                seconds=max(cycle_seconds // 2, 5),
                id="rl_train",
                max_instances=1,
            )
            # Weight update — every 10 cycles
            self._scheduler.add_job(
                self._update_weights,
                "interval",
                seconds=cycle_seconds * 10,
                id="weight_update",
                max_instances=1,
            )
            logger.info(
                f"Learning cycle: every {cycle_seconds}s | "
                f"TF: {self._config.mt5.timeframe} | "
                f"Context: {self._feature_engine.context_timeframes if self._feature_engine else '?'}"
            )

        # Graceful shutdown
        signal.signal(signal.SIGINT, self._shutdown)
        signal.signal(signal.SIGTERM, self._shutdown)

        try:
            logger.info("Bot started — tick loop: 1s | signal scan: 5s | timeframe: M1")
            self._scheduler.start()
            # Keep main thread alive
            import time
            while self._running:
                time.sleep(0.5)
        except (KeyboardInterrupt, SystemExit):
            self._shutdown()

    def _tick_loop(self) -> None:
        """Fast loop: check open trades against current price, close at TP/SL."""
        try:
            with self._lock:
                self._manage_open_trades()
        except Exception as e:
            logger.error(f"Tick loop error: {e}")

    def _signal_scan(self) -> None:
        """Scan strategies for new entry signals."""
        try:
            with self._lock:
                self._check_signals()
        except Exception as e:
            logger.error(f"Signal scan error: {e}")

    def _manage_open_trades(self) -> None:
        """Check all open trades against current price, close at TP/SL."""
        # Get open trades from DB (status = 'open' or 'dry-run')
        open_trades = self._db._conn.execute(
            "SELECT * FROM trades WHERE status IN ('open', 'dry-run') AND closed_at IS NULL"
        ).fetchall()

        if not open_trades:
            return

        tick = self._client.get_tick(self._config.mt5.symbol)
        bid = tick["bid"]
        ask = tick["ask"]

        # Compute ATR for breakeven checks (cached per tick loop)
        atr_value = 0.0
        if self._executor._breakeven_trigger_atr > 0:
            try:
                import pandas_ta as _ta
                rates = self._client.get_rates(self._config.mt5.symbol, self._config.mt5.timeframe, 50)
                atr_series = _ta.atr(rates["high"], rates["low"], rates["close"], length=14)
                if atr_series is not None and len(atr_series) > 0:
                    last_atr = atr_series.iloc[-1]
                    if not pd.isna(last_atr):
                        atr_value = float(last_atr)
            except Exception:
                pass

        for trade in open_trades:
            trade = dict(trade)
            trade_id = trade["id"]
            direction = trade["direction"]
            entry = trade["entry_price"]
            sl = trade["sl"]
            tp = trade["tp"]
            lot_size = trade["lot_size"]

            # Breakeven check
            if atr_value > 0:
                current_be_price = bid if direction in ("BUY", "Direction.BUY") else ask
                new_sl = self._executor.check_breakeven(trade, current_be_price, atr_value)
                if new_sl is not None:
                    sl = new_sl
                    trade["sl"] = sl
                    self._db.update_trade(trade_id, {"sl": sl})

            # Current price depends on direction
            if direction in ("BUY", "Direction.BUY"):
                current_price = bid  # close a BUY at bid
                hit_tp = current_price >= tp
                hit_sl = current_price <= sl
            else:
                current_price = ask  # close a SELL at ask
                hit_tp = current_price <= tp
                hit_sl = current_price >= sl

            if hit_tp or hit_sl:
                exit_reason = "TP" if hit_tp else "SL"
                exit_price = tp if hit_tp else sl

                # Calculate PnL (points * lot_size * point_value)
                point_value = 10.0  # $10 per point for gold
                if direction in ("BUY", "Direction.BUY"):
                    pnl = (exit_price - entry) * lot_size * point_value
                else:
                    pnl = (entry - exit_price) * lot_size * point_value

                pnl = round(pnl, 2)
                now = datetime.now(timezone.utc).isoformat()

                self._db.update_trade(trade_id, {
                    "exit_price": round(exit_price, 5),
                    "pnl": pnl,
                    "closed_at": now,
                    "status": "closed",
                })

                # Update client balance with realized P&L
                self._client.realize_pnl(pnl)

                # Record in performance tracker
                if self._performance_tracker:
                    strategy_name = trade.get("strategy", "unknown")
                    self._performance_tracker.record_trade(strategy_name, pnl)

                # RL reward feedback
                if self._rl_env and self._rl_env.in_position:
                    reward, _ = self._rl_env.step(
                        0,  # HOLD to trigger close evaluation
                        exit_price,
                        self._last_features.get("vector", []) if self._last_features else [],
                    )
                    if self._performance_tracker:
                        self._performance_tracker.record_rl_reward(reward)

                emoji = "+" if pnl >= 0 else ""
                logger.info(
                    f"Trade #{trade_id} closed at {exit_reason} | "
                    f"{direction} {entry:.2f} -> {exit_price:.2f} | "
                    f"P&L: {emoji}${pnl:.2f}"
                )

    def _check_signals(self) -> None:
        """Check strategies for new entry signals."""
        account = self._client.get_account_info()

        # In dry-run mode, count open trades from DB (not from mock client)
        open_in_db = self._db._conn.execute(
            "SELECT COUNT(*) FROM trades WHERE status IN ('open', 'dry-run') AND closed_at IS NULL"
        ).fetchone()[0]

        if open_in_db >= self._risk._config.max_open_trades:
            return  # already at max capacity

        positions = self._client.get_positions(self._config.mt5.symbol)
        # Combine real positions + DB open trades for risk check
        fake_positions = [{"symbol": "XAUUSD"}] * open_in_db
        today = datetime.now(timezone.utc).date()
        daily_trades = self._db.get_trades_by_date(today)
        daily_pnl = self._db.get_daily_pnl(today)

        rates = self._client.get_rates(
            self._config.mt5.symbol,
            self._config.mt5.timeframe,
            200,
        )

        # Fetch HTF data for trend confirmation
        htf_data: dict[str, Any] = {}
        if self._feature_engine:
            for tf in self._feature_engine.context_timeframes:
                try:
                    htf_data[tf] = self._client.get_rates(
                        self._config.mt5.symbol, tf, 100,
                    )
                except Exception:
                    pass
        elif self._config.ema_strategy.require_htf_confirmation:
            # Fallback: fetch H1 data even without learning system
            try:
                htf_data["H1"] = self._client.get_rates(
                    self._config.mt5.symbol, "H1", 100,
                )
            except Exception:
                pass

        # Get current spread for risk check
        tick = self._client.get_tick(self._config.mt5.symbol)
        current_spread = tick.get("spread", tick["ask"] - tick["bid"])

        # Collect signals from enabled strategies
        active_strategies = [
            s for s in self._strategies if s.name in self.enabled_strategies
        ]

        def _enrich(sig, strat_name):
            if self._learning_enabled and self._last_features:
                regime = self._regime_detector.current if self._regime_detector else None
                sig.regime = regime.regime.value if regime else ""
                sig.features = self._last_features
                if self._confidence_scorer and regime:
                    stats = self._performance_tracker.get_stats(strat_name) if self._performance_tracker else None
                    rl_info = None
                    if self._rl_agent and self._last_state:
                        _, rl_info = self._rl_agent.select_action(self._last_state)
                    sig.confidence = self._confidence_scorer.score(
                        direction=sig.direction.value,
                        features=self._last_features,
                        regime=regime,
                        strategy_win_rate=stats.win_rate if stats else 0.5,
                        rl_info=rl_info,
                    )

        if self.strategy_mode == "ensemble":
            # Ensemble: only trade when 2+ strategies agree on direction
            signals = []
            for strategy in active_strategies:
                sig = strategy.analyze(rates, htf_data=htf_data)
                if sig:
                    signals.append(sig)

            if len(signals) >= 2:
                from collections import Counter
                direction_counts = Counter(s.direction for s in signals)
                best_dir, count = direction_counts.most_common(1)[0]
                if count >= 2:
                    # Pick the first signal with the consensus direction
                    chosen = next(s for s in signals if s.direction == best_dir)
                    _enrich(chosen, chosen.strategy_name)
                    logger.info(
                        f"Ensemble signal: {chosen.direction} "
                        f"({count}/{len(signals)} agree)"
                    )
                    self._executor.execute_signal(
                        signal=chosen,
                        account_info=account,
                        open_positions=fake_positions if self._config.mode == "dry-run" else positions,
                        daily_trades_count=len(daily_trades),
                        daily_pnl=daily_pnl,
                        initial_balance=self._initial_balance,
                        current_spread=current_spread,
                    )
        else:
            # Independent: each strategy trades on its own signal
            for strategy in active_strategies:
                signal_result = strategy.analyze(rates, htf_data=htf_data)
                if signal_result:
                    _enrich(signal_result, strategy.name)
                    logger.info(f"Signal from {strategy.name}: {signal_result.direction}")
                    self._executor.execute_signal(
                        signal=signal_result,
                        account_info=account,
                        open_positions=fake_positions if self._config.mode == "dry-run" else positions,
                        daily_trades_count=len(daily_trades),
                        daily_pnl=daily_pnl,
                        initial_balance=self._initial_balance,
                        current_spread=current_spread,
                    )
                    break  # one signal per scan cycle

        # Manage trailing stops
        if positions:
            self._executor.manage_trailing_stops(positions)

    def _learning_cycle(self) -> None:
        """Run one learning cycle: extract features, update regime, step RL."""
        if not self._learning_enabled:
            return
        try:
            with self._lock:
                rates = self._client.get_rates(
                    self._config.mt5.symbol,
                    self._config.mt5.timeframe,
                    200,
                )

                # Fetch HTF data
                htf_data = {}
                if self._feature_engine:
                    for tf in self._feature_engine.context_timeframes:
                        try:
                            htf_data[tf] = self._client.get_rates(
                                self._config.mt5.symbol, tf, 100,
                            )
                        except Exception:
                            pass

                # Extract features
                features = self._feature_engine.extract(rates, htf_data)
                self._last_features = features
                previous_state = self._last_state
                state = features["vector"]

                # Update regime
                if self._regime_detector:
                    self._regime_detector.update(features)

                # RL agent step (continuous action)
                if self._rl_agent and self._rl_env:
                    action, info = self._rl_agent.select_action(state)
                    bar_atr = float(features.get("atr", 0.0))
                    bar_high = float(rates.iloc[-1]["high"])
                    bar_low = float(rates.iloc[-1]["low"])
                    reward, trade_result = self._rl_env.step(
                        action, features["close"], state,
                        atr=bar_atr, high=bar_high, low=bar_low,
                    )

                    # Store transition: previous_state -> action -> state
                    if previous_state is not None:
                        from core.learning.replay_buffer import Transition
                        self._rl_agent.store_transition(Transition(
                            state=previous_state,
                            action=action,
                            reward=reward,
                            next_state=state,
                            done=False,
                        ))

                    if self._performance_tracker:
                        self._performance_tracker.record_rl_reward(reward)

                    if trade_result:
                        self._rl_agent.end_episode(trade_result.reward)

                # Update last_state after transition is stored
                self._last_state = state

                # Store features in DB periodically
                self._learning_step_count += 1
                if self._learning_step_count % 10 == 0:
                    self._db.insert_market_features(
                        timestamp=str(features.get("timestamp", "")),
                        features={"vector": features["vector"]},
                        regime=self._regime_detector.current.regime.value if self._regime_detector else "",
                        session=features.get("session", ""),
                    )

        except Exception as e:
            logger.error(f"Learning cycle error: {e}")

    def _rl_train_step(self) -> None:
        """Perform RL training from replay buffer."""
        if not self._learning_enabled or not self._rl_agent:
            return
        try:
            loss = self._rl_agent.train_step()
            if loss is not None and self._learning_step_count % 50 == 0:
                logger.debug(f"RL train loss: {loss:.4f} | epsilon: {self._rl_agent.epsilon:.4f}")
        except Exception as e:
            logger.error(f"RL training error: {e}")

    def _update_weights(self) -> None:
        """Update ensemble weights from performance data."""
        if not self._learning_enabled or not self._ensemble:
            return
        try:
            new_weights = self._ensemble.update_weights()
            logger.info(f"Updated ensemble weights: {new_weights}")
        except Exception as e:
            logger.error(f"Weight update error: {e}")

    def _shutdown(self, *args) -> None:
        """Graceful shutdown."""
        logger.info("Shutting down trading bot...")
        self._running = False

        # Save RL checkpoint
        if self._rl_agent and self._rl_env:
            try:
                blob = self._rl_agent.save_checkpoint()
                if blob:
                    model_name = self._config.rl.agent_type
                    self._db.save_rl_model(
                        model_name, blob, self._rl_agent.episode,
                        self._rl_agent.epsilon, self._rl_agent.total_reward,
                        self._rl_env.win_rate,
                    )
                    logger.info("RL checkpoint saved")
            except Exception as e:
                logger.error(f"Failed to save RL checkpoint: {e}")

        if self._scheduler and self._scheduler.running:
            self._scheduler.shutdown(wait=False)
        self._client.disconnect()
        self._db.close()
        logger.info("Bot stopped")
        sys.exit(0)


def build_strategies(config: AppConfig, strategy_filter: str = "all") -> list[Strategy]:
    """Build strategy instances from config."""
    strategies: list[Strategy] = []

    if strategy_filter in ("all", "ema"):
        strategies.append(EMACrossoverStrategy(
            fast_ema=config.ema_strategy.fast_ema,
            slow_ema=config.ema_strategy.slow_ema,
            rsi_period=config.ema_strategy.rsi_period,
            rsi_overbought=config.ema_strategy.rsi_overbought,
            rsi_oversold=config.ema_strategy.rsi_oversold,
            atr_period=config.ema_strategy.atr_period,
            sl_atr_multiplier=config.ema_strategy.sl_atr_multiplier,
            tp_sl_ratio=config.ema_strategy.tp_sl_ratio,
            trailing_atr_trigger=config.ema_strategy.trailing_atr_trigger,
            volume_filter_multiplier=config.ema_strategy.volume_filter_multiplier,
            require_htf_confirmation=config.ema_strategy.require_htf_confirmation,
        ))

    if strategy_filter in ("all", "breakout"):
        strategies.append(AsianRangeBreakoutStrategy(
            asian_start_hour=config.breakout_strategy.asian_start_hour,
            asian_end_hour=config.breakout_strategy.asian_end_hour,
            active_start_hour=config.breakout_strategy.active_start_hour,
            active_end_hour=config.breakout_strategy.active_end_hour,
            atr_buffer_multiplier=config.breakout_strategy.atr_buffer_multiplier,
            min_range_pips=config.breakout_strategy.min_range_pips,
            max_range_pips=config.breakout_strategy.max_range_pips,
            tp_sl_ratio=config.breakout_strategy.tp_sl_ratio,
        ))

    if strategy_filter in ("all", "bos"):
        strategies.append(BOSStrategy(
            swing_lookback=config.bos_strategy.swing_lookback,
            swing_strength=config.bos_strategy.swing_strength,
            atr_period=config.bos_strategy.atr_period,
            sl_buffer_atr=config.bos_strategy.sl_buffer_atr,
            tp_sl_ratio=config.bos_strategy.tp_sl_ratio,
        ))

    if strategy_filter in ("all", "candle"):
        strategies.append(CandlePatternStrategy(
            min_wick_body_ratio=config.candle_pattern.min_wick_body_ratio,
            max_opposite_wick_ratio=config.candle_pattern.max_opposite_wick_ratio,
            trend_lookback=config.candle_pattern.trend_lookback,
            atr_period=config.candle_pattern.atr_period,
            sl_buffer_atr=config.candle_pattern.sl_buffer_atr,
            tp_sl_ratio=config.candle_pattern.tp_sl_ratio,
            require_confirmation=config.candle_pattern.require_confirmation,
        ))

    return strategies


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="XAU/USD Scalping Bot")
    parser.add_argument(
        "--mode", choices=["live", "dry-run", "backtest"],
        default=None, help="Trading mode",
    )
    parser.add_argument(
        "--strategy", choices=["ema", "breakout", "bos", "candle", "all"],
        default="all", help="Strategy to use",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point."""
    args = parse_args()
    config = AppConfig()

    if args.mode:
        # Override config mode with CLI arg
        config = AppConfig(
            mode=args.mode,
            log_level=config.log_level,
            db_path=config.db_path,
            mt5=config.mt5,
            ema_strategy=config.ema_strategy,
            breakout_strategy=config.breakout_strategy,
            bos_strategy=config.bos_strategy,
            candle_pattern=config.candle_pattern,
            risk=config.risk,
            learning=config.learning,
            rl=config.rl,
            claude=config.claude,
            ensemble=config.ensemble,
        )

    if config.mode == "backtest":
        from backtesting.backtester import Backtester
        from backtesting.report import BacktestReport
        setup_logger(config.log_level)
        client = create_mt5_client(config)
        client.connect()
        strategies = build_strategies(config, args.strategy)
        backtester = Backtester(client)
        all_trades = []
        for strategy in strategies:
            data = client.get_rates(config.mt5.symbol, config.mt5.timeframe, 5000)
            trades = backtester.run(strategy, data)
            all_trades.extend(trades)
        report = BacktestReport(all_trades)
        report.print_summary()
        client.disconnect()
    else:
        strategies = build_strategies(config, args.strategy)
        bot = TradingBot(config, strategies)
        bot.start()


if __name__ == "__main__":
    main()
