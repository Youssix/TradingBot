from __future__ import annotations

import time
from abc import ABC, abstractmethod
from datetime import datetime
from functools import wraps
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger


def retry(max_attempts: int = 3, base_delay: float = 1.0):
    """Decorator for retrying MT5 calls with exponential backoff."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_attempts:
                        delay = base_delay * (2 ** (attempt - 1))
                        logger.warning(
                            f"{func.__name__} attempt {attempt}/{max_attempts} failed: {e}. "
                            f"Retrying in {delay:.1f}s..."
                        )
                        time.sleep(delay)
                    else:
                        logger.error(f"{func.__name__} failed after {max_attempts} attempts: {e}")
            raise last_exception  # type: ignore[misc]
        return wrapper
    return decorator


class MT5Client(ABC):
    """Abstract base class for MetaTrader 5 client operations."""

    @abstractmethod
    def connect(self) -> bool:
        """Initialize connection to MT5 terminal."""
        ...

    @abstractmethod
    def disconnect(self) -> None:
        """Shutdown MT5 connection."""
        ...

    @abstractmethod
    def get_rates(self, symbol: str, timeframe: str, count: int) -> pd.DataFrame:
        """Get OHLCV rates for a symbol."""
        ...

    @abstractmethod
    def get_tick(self, symbol: str) -> dict[str, Any]:
        """Get latest tick data for a symbol."""
        ...

    @abstractmethod
    def send_order(
        self,
        symbol: str,
        order_type: str,
        volume: float,
        price: float,
        sl: float,
        tp: float,
        comment: str = "",
    ) -> dict[str, Any]:
        """Send a trade order. Returns order result dict."""
        ...

    @abstractmethod
    def close_position(self, ticket: int) -> bool:
        """Close an open position by ticket number."""
        ...

    @abstractmethod
    def get_positions(self, symbol: str | None = None) -> list[dict[str, Any]]:
        """Get open positions, optionally filtered by symbol."""
        ...

    @abstractmethod
    def get_account_info(self) -> dict[str, Any]:
        """Get account information (balance, equity, margin, etc.)."""
        ...

    def realize_pnl(self, pnl: float) -> None:
        """Add realized P&L to account balance. No-op for real MT5 (broker tracks it)."""
        pass


class RealMT5Client(MT5Client):
    """Real MetaTrader 5 client implementation."""

    TIMEFRAME_MAP: dict[str, int] = {}  # Populated on import if mt5 available

    def __init__(self, login: int, password: str, server: str, path: str = "") -> None:
        self._login = login
        self._password = password
        self._server = server
        self._path = path
        self._mt5: Any = None

    @retry(max_attempts=3)
    def connect(self) -> bool:
        """Initialize MT5 connection."""
        try:
            import MetaTrader5 as mt5
            self._mt5 = mt5
            self.TIMEFRAME_MAP = {
                "M1": mt5.TIMEFRAME_M1, "M5": mt5.TIMEFRAME_M5,
                "M15": mt5.TIMEFRAME_M15, "M30": mt5.TIMEFRAME_M30,
                "H1": mt5.TIMEFRAME_H1, "H4": mt5.TIMEFRAME_H4,
                "D1": mt5.TIMEFRAME_D1,
            }
        except ImportError:
            raise RuntimeError("MetaTrader5 package not available. Use MockMT5Client on non-Windows platforms.")

        init_kwargs: dict[str, Any] = {"login": self._login, "password": self._password, "server": self._server}
        if self._path:
            init_kwargs["path"] = self._path

        if not mt5.initialize(**init_kwargs):
            error = mt5.last_error()
            raise ConnectionError(f"MT5 initialization failed: {error}")
        logger.info(f"Connected to MT5 server: {self._server}")
        return True

    def disconnect(self) -> None:
        """Shutdown MT5 connection."""
        if self._mt5:
            self._mt5.shutdown()
            logger.info("MT5 connection closed")

    @retry(max_attempts=3)
    def get_rates(self, symbol: str, timeframe: str, count: int) -> pd.DataFrame:
        """Get OHLCV rates from MT5."""
        tf = self.TIMEFRAME_MAP.get(timeframe)
        if tf is None:
            raise ValueError(f"Unknown timeframe: {timeframe}")
        rates = self._mt5.copy_rates_from_pos(symbol, tf, 0, count)
        if rates is None:
            raise RuntimeError(f"Failed to get rates: {self._mt5.last_error()}")
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df.rename(columns={"time": "datetime", "tick_volume": "volume"}, inplace=True)
        return df

    @retry(max_attempts=3)
    def get_tick(self, symbol: str) -> dict[str, Any]:
        """Get latest tick from MT5."""
        tick = self._mt5.symbol_info_tick(symbol)
        if tick is None:
            raise RuntimeError(f"Failed to get tick: {self._mt5.last_error()}")
        return {"bid": tick.bid, "ask": tick.ask, "time": datetime.fromtimestamp(tick.time)}

    @retry(max_attempts=3)
    def send_order(
        self, symbol: str, order_type: str, volume: float, price: float,
        sl: float, tp: float, comment: str = "",
    ) -> dict[str, Any]:
        """Send order to MT5."""
        action = self._mt5.TRADE_ACTION_DEAL
        type_map = {"BUY": self._mt5.ORDER_TYPE_BUY, "SELL": self._mt5.ORDER_TYPE_SELL}
        request = {
            "action": action, "symbol": symbol, "volume": volume,
            "type": type_map[order_type], "price": price, "sl": sl, "tp": tp,
            "deviation": 20, "magic": 234000, "comment": comment,
            "type_time": self._mt5.ORDER_TIME_GTC,
            "type_filling": self._mt5.ORDER_FILLING_IOC,
        }
        result = self._mt5.order_send(request)
        if result is None or result.retcode != self._mt5.TRADE_RETCODE_DONE:
            error = result.comment if result else self._mt5.last_error()
            raise RuntimeError(f"Order failed: {error}")
        return {"ticket": result.order, "price": result.price, "volume": result.volume, "comment": result.comment}

    @retry(max_attempts=3)
    def close_position(self, ticket: int) -> bool:
        """Close position by ticket."""
        positions = self._mt5.positions_get(ticket=ticket)
        if not positions:
            raise RuntimeError(f"Position {ticket} not found")
        pos = positions[0]
        close_type = self._mt5.ORDER_TYPE_SELL if pos.type == 0 else self._mt5.ORDER_TYPE_BUY
        request = {
            "action": self._mt5.TRADE_ACTION_DEAL, "position": ticket,
            "symbol": pos.symbol, "volume": pos.volume,
            "type": close_type, "price": self._mt5.symbol_info_tick(pos.symbol).bid if pos.type == 0 else self._mt5.symbol_info_tick(pos.symbol).ask,
            "deviation": 20, "magic": 234000, "comment": "close",
            "type_time": self._mt5.ORDER_TIME_GTC,
            "type_filling": self._mt5.ORDER_FILLING_IOC,
        }
        result = self._mt5.order_send(request)
        if result is None or result.retcode != self._mt5.TRADE_RETCODE_DONE:
            raise RuntimeError(f"Close position failed: {result}")
        return True

    @retry(max_attempts=3)
    def get_positions(self, symbol: str | None = None) -> list[dict[str, Any]]:
        """Get open positions."""
        if symbol:
            positions = self._mt5.positions_get(symbol=symbol)
        else:
            positions = self._mt5.positions_get()
        if positions is None:
            return []
        return [
            {"ticket": p.ticket, "symbol": p.symbol, "type": "BUY" if p.type == 0 else "SELL",
             "volume": p.volume, "price_open": p.price_open, "sl": p.sl, "tp": p.tp,
             "profit": p.profit, "comment": p.comment}
            for p in positions
        ]

    @retry(max_attempts=3)
    def get_account_info(self) -> dict[str, Any]:
        """Get account info."""
        info = self._mt5.account_info()
        if info is None:
            raise RuntimeError(f"Failed to get account info: {self._mt5.last_error()}")
        return {"balance": info.balance, "equity": info.equity, "margin": info.margin,
                "free_margin": info.margin_free, "profit": info.profit, "leverage": info.leverage}


class MockMT5Client(MT5Client):
    """Mock MT5 client for testing and macOS development."""

    def __init__(self, initial_balance: float = 10000.0) -> None:
        self._connected = False
        self._balance = initial_balance
        self._positions: list[dict[str, Any]] = []
        self._next_ticket = 1000
        self._custom_rates: pd.DataFrame | None = None
        self._custom_tick: dict[str, Any] | None = None

    def set_rates(self, rates: pd.DataFrame) -> None:
        """Set custom rates data for testing."""
        self._custom_rates = rates

    def set_tick(self, tick: dict[str, Any]) -> None:
        """Set custom tick data for testing."""
        self._custom_tick = tick

    def connect(self) -> bool:
        self._connected = True
        logger.info("Mock MT5 client connected")
        return True

    def disconnect(self) -> None:
        self._connected = False
        logger.info("Mock MT5 client disconnected")

    def get_rates(self, symbol: str, timeframe: str, count: int) -> pd.DataFrame:
        if self._custom_rates is not None:
            return self._custom_rates.tail(count).reset_index(drop=True)
        # Generate synthetic gold data with a guaranteed EMA crossover near the
        # last bar and RSI in the 35-65 range.  Alternates bullish/bearish each
        # 5-minute cycle.  The trick is: flat → trend → sine-wave recovery whose
        # oscillation keeps RSI moderate while EMAs cross.
        now = datetime.now()
        cycle = int(now.timestamp()) // 60  # changes every 1 min
        rng = np.random.RandomState(cycle)
        dates = pd.date_range(end=now, periods=count, freq="1min")
        base_price = 1950.0
        bullish = cycle % 2 == 0

        flat_len = max(count - 80, 50)
        ramp_len = 30
        wave_len = count - flat_len - ramp_len

        x_wave = np.arange(wave_len, dtype=float)
        if bullish:
            # Tuned: freq=1.5*pi, amp=2.0, slope_end=0
            ramp = np.linspace(0, -6, ramp_len)
            wave = np.linspace(-6, 0, wave_len) + np.sin(x_wave / wave_len * 1.5 * np.pi) * 2.0
        else:
            # Tuned: freq=2.6*pi, amp=1.5, slope_end=1.0
            ramp = np.linspace(0, 6, ramp_len)
            wave = np.linspace(6, 1.0, wave_len) + np.sin(x_wave / wave_len * 2.6 * np.pi) * 1.5

        trend = np.concatenate([np.zeros(flat_len), ramp, wave])
        noise = rng.randn(count) * 0.15  # tiny noise to not disrupt the crossover
        closes = base_price + trend + noise

        opens = closes + rng.randn(count) * 0.2
        highs = np.maximum(opens, closes) + np.abs(rng.randn(count)) * 1.0
        lows = np.minimum(opens, closes) - np.abs(rng.randn(count)) * 1.0
        volumes = rng.randint(100, 5000, count)
        return pd.DataFrame({
            "datetime": dates, "open": opens, "high": highs,
            "low": lows, "close": closes, "volume": volumes,
        })

    def get_tick(self, symbol: str) -> dict[str, Any]:
        if self._custom_tick:
            return self._custom_tick
        # Derive live tick from the latest candle data so price actually moves
        rates = self.get_rates(symbol, "M1", 200)
        last_close = float(rates["close"].iloc[-1])
        # Adaptive spread by session (Asian wider, London/NY tighter)
        hour = datetime.now().hour
        if 0 <= hour < 8:
            spread = 0.50  # Asian session
        else:
            spread = 0.25  # London/NY session
        jitter = float(np.random.randn() * 0.15)
        bid = last_close + jitter
        return {"bid": round(bid, 2), "ask": round(bid + spread, 2), "time": datetime.now(), "spread": spread}

    def send_order(
        self, symbol: str, order_type: str, volume: float, price: float,
        sl: float, tp: float, comment: str = "",
    ) -> dict[str, Any]:
        ticket = self._next_ticket
        self._next_ticket += 1
        position = {
            "ticket": ticket, "symbol": symbol, "type": order_type,
            "volume": volume, "price_open": price, "sl": sl, "tp": tp,
            "profit": 0.0, "comment": comment,
        }
        self._positions.append(position)
        logger.info(f"Mock order filled: #{ticket} {order_type} {volume} {symbol} @ {price}")
        return {"ticket": ticket, "price": price, "volume": volume, "comment": comment}

    def close_position(self, ticket: int) -> bool:
        self._positions = [p for p in self._positions if p["ticket"] != ticket]
        logger.info(f"Mock position #{ticket} closed")
        return True

    def get_positions(self, symbol: str | None = None) -> list[dict[str, Any]]:
        if symbol:
            return [p for p in self._positions if p["symbol"] == symbol]
        return list(self._positions)

    def get_account_info(self) -> dict[str, Any]:
        total_profit = sum(p["profit"] for p in self._positions)
        return {
            "balance": self._balance, "equity": self._balance + total_profit,
            "margin": 0.0, "free_margin": self._balance + total_profit,
            "profit": total_profit, "leverage": 100,
        }

    def realize_pnl(self, pnl: float) -> None:
        self._balance += pnl


class YFinanceClient(MT5Client):
    """MT5-compatible client using Yahoo Finance for real gold market data.

    Uses GC=F (gold futures) as proxy for XAUUSD. Provides real historical
    candles and live-ish tick prices for backtesting and dry-run trading.
    """

    TIMEFRAME_MAP = {
        "M1": ("1m", "5d"),    # 1-min candles, max 7 days back
        "M5": ("5m", "60d"),
        "M15": ("15m", "60d"),
        "H1": ("1h", "730d"),
        "H4": ("1h", "730d"),  # no native 4h, use 1h
        "D1": ("1d", "max"),
    }

    def __init__(self, initial_balance: float = 10_000.0) -> None:
        self._connected = False
        self._balance = initial_balance
        self._positions: list[dict[str, Any]] = []
        self._next_ticket = 1000
        self._ticker = None
        self._data_cache: dict[str, tuple[pd.DataFrame, float]] = {}  # key -> (df, timestamp)
        self._cache_ttl = 10.0  # refresh real data every 10s for scalping

    def connect(self) -> bool:
        try:
            import yfinance as yf
            self._ticker = yf.Ticker("GC=F")
            # Test fetch
            test = self._ticker.history(period="1d", interval="1m")
            if test.empty:
                logger.warning("YFinance returned empty data, falling back to 5m")
            self._connected = True
            logger.info(f"YFinance client connected — real gold data (GC=F)")
            return True
        except Exception as e:
            logger.error(f"YFinance connection failed: {e}")
            return False

    def disconnect(self) -> None:
        self._connected = False
        self._data_cache.clear()
        logger.info("YFinance client disconnected")

    def _fetch_data(self, timeframe: str, count: int) -> pd.DataFrame:
        """Fetch real data from Yahoo Finance with caching."""
        cache_key = f"{timeframe}_{count}"
        now = time.time()

        # Return cached data if fresh enough
        if cache_key in self._data_cache:
            cached_df, cached_time = self._data_cache[cache_key]
            if now - cached_time < self._cache_ttl:
                return cached_df

        interval, period = self.TIMEFRAME_MAP.get(timeframe, ("1m", "5d"))

        try:
            df = self._ticker.history(period=period, interval=interval)
            if df.empty:
                raise ValueError(f"No data returned for interval={interval}")

            # Normalize columns to match MT5 format
            df = df.reset_index()
            col_map = {
                "Datetime": "datetime", "Date": "datetime",
                "Open": "open", "High": "high", "Low": "low",
                "Close": "close", "Volume": "volume",
            }
            df = df.rename(columns=col_map)

            # Keep only needed columns
            df = df[["datetime", "open", "high", "low", "close", "volume"]].copy()

            # Remove timezone info for consistency
            if hasattr(df["datetime"].dtype, "tz") and df["datetime"].dtype.tz is not None:
                df["datetime"] = df["datetime"].dt.tz_localize(None)

            # Return last `count` rows
            df = df.tail(count).reset_index(drop=True)

            self._data_cache[cache_key] = (df, now)
            return df

        except Exception as e:
            logger.error(f"YFinance data fetch error: {e}")
            # Return last cached data if available
            if cache_key in self._data_cache:
                return self._data_cache[cache_key][0]
            raise

    def get_rates(self, symbol: str, timeframe: str, count: int) -> pd.DataFrame:
        return self._fetch_data(timeframe, count)

    def get_tick(self, symbol: str) -> dict[str, Any]:
        """Get latest real price from the most recent candle."""
        df = self._fetch_data("M1", 5)
        last_close = float(df["close"].iloc[-1])
        # Adaptive spread by session
        hour = datetime.now().hour
        if 0 <= hour < 8:
            spread = 0.50  # Asian session
        else:
            spread = 0.25  # London/NY session
        return {
            "bid": round(last_close, 2),
            "ask": round(last_close + spread, 2),
            "time": datetime.now(),
            "spread": spread,
        }

    def send_order(
        self, symbol: str, order_type: str, volume: float, price: float,
        sl: float, tp: float, comment: str = "",
    ) -> dict[str, Any]:
        ticket = self._next_ticket
        self._next_ticket += 1
        position = {
            "ticket": ticket, "symbol": symbol, "type": order_type,
            "volume": volume, "price_open": price, "sl": sl, "tp": tp,
            "profit": 0.0, "comment": comment,
        }
        self._positions.append(position)
        logger.info(f"Paper order filled: #{ticket} {order_type} {volume} {symbol} @ {price}")
        return {"ticket": ticket, "price": price, "volume": volume, "comment": comment}

    def close_position(self, ticket: int) -> bool:
        self._positions = [p for p in self._positions if p["ticket"] != ticket]
        logger.info(f"Paper position #{ticket} closed")
        return True

    def get_positions(self, symbol: str | None = None) -> list[dict[str, Any]]:
        if symbol:
            return [p for p in self._positions if p["symbol"] == symbol]
        return list(self._positions)

    def get_account_info(self) -> dict[str, Any]:
        total_profit = sum(p["profit"] for p in self._positions)
        return {
            "balance": self._balance, "equity": self._balance + total_profit,
            "margin": 0.0, "free_margin": self._balance + total_profit,
            "profit": total_profit, "leverage": 100,
        }

    def realize_pnl(self, pnl: float) -> None:
        self._balance += pnl


def create_mt5_client(config: Any) -> MT5Client:
    """Factory function to create appropriate MT5 client based on config."""
    if config.mt5.use_mock:
        logger.info("Using YFinance client (real gold data)")
        return YFinanceClient()
    logger.info(f"Using Real MT5 client for server: {config.mt5.server}")
    return RealMT5Client(
        login=config.mt5.login, password=config.mt5.password,
        server=config.mt5.server, path=config.mt5.path,
    )
