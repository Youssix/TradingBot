from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from core.strategy import (
    AsianRangeBreakoutStrategy,
    BOSStrategy,
    CandlePatternStrategy,
    Direction,
    EMACrossoverStrategy,
    Signal,
)


def _make_ohlcv(prices: list[float], start: datetime | None = None) -> pd.DataFrame:
    """Create synthetic OHLCV DataFrame from close prices."""
    n = len(prices)
    start = start or datetime(2024, 1, 1)
    dates = [start + timedelta(minutes=5 * i) for i in range(n)]
    return pd.DataFrame({
        "datetime": dates,
        "open": [p - 0.5 for p in prices],
        "high": [p + 1.0 for p in prices],
        "low": [p - 1.0 for p in prices],
        "close": prices,
        "volume": [1000] * n,
    })


def _make_crossover_data(direction: str = "bullish", length: int = 50) -> pd.DataFrame:
    """Generate data that produces an EMA crossover at the last candle.

    Generates a long price series with a trend reversal, computes EMAs to find
    the exact crossover bar, then truncates so the crossover is at the end.
    The price series uses a gradual reversal to keep RSI in a moderate range.
    """
    import pandas_ta as ta

    extra = 100
    total = length + extra
    if direction == "bullish":
        # Long downtrend then gradual recovery (gentle slope keeps RSI moderate)
        down = list(np.linspace(2000, 1920, total - 40))
        up = list(np.linspace(1921, 1980, 40))
        prices = down + up
    else:
        up = list(np.linspace(1900, 1980, total - 40))
        down = list(np.linspace(1979, 1920, 40))
        prices = up + down

    df = _make_ohlcv(prices)
    df["ema_9"] = ta.ema(df["close"], length=9)
    df["ema_21"] = ta.ema(df["close"], length=21)
    df["rsi"] = ta.rsi(df["close"], length=14)

    # Find the crossover bar where RSI is also in the valid range
    diff = df["ema_9"] - df["ema_21"]
    crossover_idx = None
    if direction == "bullish":
        for i in range(1, len(diff)):
            if (not np.isnan(diff.iloc[i - 1]) and diff.iloc[i - 1] <= 0
                    and diff.iloc[i] > 0 and df["rsi"].iloc[i] < 70):
                crossover_idx = i
                break
    else:
        for i in range(1, len(diff)):
            if (not np.isnan(diff.iloc[i - 1]) and diff.iloc[i - 1] >= 0
                    and diff.iloc[i] < 0 and df["rsi"].iloc[i] > 30):
                crossover_idx = i
                break

    assert crossover_idx is not None, f"No valid {direction} crossover found in synthetic data"

    # Truncate so crossover is at the last bar, keeping at least `length` bars
    start_idx = max(0, crossover_idx - length + 1)
    end_idx = crossover_idx + 1
    result = df.iloc[start_idx:end_idx].drop(
        columns=["ema_9", "ema_21", "rsi"],
    ).reset_index(drop=True)
    return result


@pytest.fixture
def strategy() -> EMACrossoverStrategy:
    return EMACrossoverStrategy()


class TestEMACrossoverBuySignal:
    def test_generates_buy_signal(self, strategy: EMACrossoverStrategy) -> None:
        df = _make_crossover_data("bullish", length=60)
        signal = strategy.analyze(df)
        assert signal is not None
        assert signal.direction == Direction.BUY
        assert signal.strategy_name == "ema_crossover"

    def test_buy_signal_sl_below_entry(self, strategy: EMACrossoverStrategy) -> None:
        df = _make_crossover_data("bullish", length=60)
        signal = strategy.analyze(df)
        assert signal is not None
        assert signal.sl < signal.entry_price

    def test_buy_signal_tp_above_entry(self, strategy: EMACrossoverStrategy) -> None:
        df = _make_crossover_data("bullish", length=60)
        signal = strategy.analyze(df)
        assert signal is not None
        assert signal.tp > signal.entry_price

    def test_buy_signal_has_trailing_stop(self, strategy: EMACrossoverStrategy) -> None:
        df = _make_crossover_data("bullish", length=60)
        signal = strategy.analyze(df)
        assert signal is not None
        assert signal.trailing_stop_trigger is not None
        assert signal.trailing_stop_distance is not None


class TestEMACrossoverSellSignal:
    def test_generates_sell_signal(self, strategy: EMACrossoverStrategy) -> None:
        df = _make_crossover_data("bearish", length=60)
        signal = strategy.analyze(df)
        assert signal is not None
        assert signal.direction == Direction.SELL

    def test_sell_signal_sl_above_entry(self, strategy: EMACrossoverStrategy) -> None:
        df = _make_crossover_data("bearish", length=60)
        signal = strategy.analyze(df)
        assert signal is not None
        assert signal.sl > signal.entry_price

    def test_sell_signal_tp_below_entry(self, strategy: EMACrossoverStrategy) -> None:
        df = _make_crossover_data("bearish", length=60)
        signal = strategy.analyze(df)
        assert signal is not None
        assert signal.tp < signal.entry_price


class TestRSIFilter:
    def test_blocks_buy_when_overbought(self) -> None:
        strategy = EMACrossoverStrategy(rsi_overbought=30.0)  # Very low threshold
        df = _make_crossover_data("bullish", length=60)
        signal = strategy.analyze(df)
        # Should be filtered because RSI would likely be above 30
        assert signal is None

    def test_blocks_sell_when_oversold(self) -> None:
        strategy = EMACrossoverStrategy(rsi_oversold=70.0)  # Very high threshold
        df = _make_crossover_data("bearish", length=60)
        signal = strategy.analyze(df)
        assert signal is None


class TestNoSignal:
    def test_no_crossover_flat_market(self, strategy: EMACrossoverStrategy) -> None:
        # Flat prices - no crossover
        prices = [1950.0] * 60
        df = _make_ohlcv(prices)
        signal = strategy.analyze(df)
        assert signal is None

    def test_insufficient_data(self, strategy: EMACrossoverStrategy) -> None:
        prices = [1950.0] * 10  # Less than slow_ema + 2
        df = _make_ohlcv(prices)
        signal = strategy.analyze(df)
        assert signal is None


class TestSLTPCalculation:
    def test_tp_is_2x_sl_for_buy(self, strategy: EMACrossoverStrategy) -> None:
        df = _make_crossover_data("bullish", length=60)
        signal = strategy.analyze(df)
        assert signal is not None
        sl_dist = signal.entry_price - signal.sl
        tp_dist = signal.tp - signal.entry_price
        assert abs(tp_dist / sl_dist - 2.0) < 0.01

    def test_tp_is_2x_sl_for_sell(self, strategy: EMACrossoverStrategy) -> None:
        df = _make_crossover_data("bearish", length=60)
        signal = strategy.analyze(df)
        assert signal is not None
        sl_dist = signal.sl - signal.entry_price
        tp_dist = signal.entry_price - signal.tp
        assert abs(tp_dist / sl_dist - 2.0) < 0.01


# ---------------------------------------------------------------------------
# Asian Range Breakout Strategy Tests
# ---------------------------------------------------------------------------


def _make_asian_breakout_data(
    breakout: str = "above",
    asian_high: float = 1960.0,
    asian_low: float = 1950.0,
    breakout_hour: int = 10,
) -> pd.DataFrame:
    """Build OHLCV data with an Asian range then a breakout candle.

    Creates 5-minute candles: Asian session (00:00-08:00) with a defined range,
    then candles during active hours with a breakout price at the end.
    Enough bars are generated (>= 16) so ATR(14) is available.
    """
    rows: list[dict] = []
    base_date = datetime(2024, 6, 10)  # A Monday
    mid = (asian_high + asian_low) / 2.0

    # Asian session candles: 00:00 to 07:55 UTC (96 five-minute bars)
    for i in range(96):
        dt = base_date + timedelta(minutes=5 * i)
        # Oscillate between high and low to establish the range
        if i % 2 == 0:
            c = asian_high - 1.0
            h = asian_high
            l = mid
        else:
            c = asian_low + 1.0
            h = mid
            l = asian_low
        rows.append({
            "datetime": dt,
            "open": c - 0.5,
            "high": h,
            "low": l,
            "close": c,
            "volume": 1000,
        })

    # Active session candles leading up to the breakout hour
    active_start_min = 8 * 60  # 08:00
    breakout_min = breakout_hour * 60
    for m in range(active_start_min, breakout_min, 5):
        dt = base_date + timedelta(minutes=m)
        rows.append({
            "datetime": dt,
            "open": mid - 0.5,
            "high": mid + 1.0,
            "low": mid - 1.0,
            "close": mid,
            "volume": 1000,
        })

    # Breakout candle
    dt = base_date + timedelta(minutes=breakout_min)
    if breakout == "above":
        breakout_price = asian_high + 5.0  # Well above range + buffer
        rows.append({
            "datetime": dt,
            "open": asian_high + 1.0,
            "high": breakout_price + 1.0,
            "low": asian_high,
            "close": breakout_price,
            "volume": 2000,
        })
    else:
        breakout_price = asian_low - 5.0  # Well below range - buffer
        rows.append({
            "datetime": dt,
            "open": asian_low - 1.0,
            "high": asian_low,
            "low": breakout_price - 1.0,
            "close": breakout_price,
            "volume": 2000,
        })

    return pd.DataFrame(rows)


@pytest.fixture
def asian_strategy() -> AsianRangeBreakoutStrategy:
    return AsianRangeBreakoutStrategy()


class TestAsianBreakoutBuySignal:
    def test_breakout_above_range(self, asian_strategy: AsianRangeBreakoutStrategy) -> None:
        df = _make_asian_breakout_data(breakout="above")
        signal = asian_strategy.analyze(df)
        assert signal is not None
        assert signal.direction == Direction.BUY
        assert signal.strategy_name == "asian_breakout"
        assert signal.sl < signal.entry_price
        assert signal.tp > signal.entry_price


class TestAsianBreakoutSellSignal:
    def test_breakout_below_range(self, asian_strategy: AsianRangeBreakoutStrategy) -> None:
        df = _make_asian_breakout_data(breakout="below")
        signal = asian_strategy.analyze(df)
        assert signal is not None
        assert signal.direction == Direction.SELL
        assert signal.strategy_name == "asian_breakout"
        assert signal.sl > signal.entry_price
        assert signal.tp < signal.entry_price


class TestAsianBreakoutFilters:
    def test_range_too_narrow(self) -> None:
        # Range of 2.0 (1951-1949) is below default min_range_pips=3.0
        strategy = AsianRangeBreakoutStrategy(min_range_pips=3.0)
        df = _make_asian_breakout_data(
            breakout="above", asian_high=1951.0, asian_low=1949.0,
        )
        signal = strategy.analyze(df)
        assert signal is None

    def test_range_too_wide(self) -> None:
        # Range of 40.0 is above default max_range_pips=30.0
        strategy = AsianRangeBreakoutStrategy(max_range_pips=30.0)
        df = _make_asian_breakout_data(
            breakout="above", asian_high=1970.0, asian_low=1930.0,
        )
        signal = strategy.analyze(df)
        assert signal is None

    def test_outside_active_hours(self) -> None:
        # Breakout at 22:00 is outside 08:00-20:00
        strategy = AsianRangeBreakoutStrategy()
        df = _make_asian_breakout_data(breakout="above", breakout_hour=22)
        signal = strategy.analyze(df)
        assert signal is None


# ---------------------------------------------------------------------------
# BOS Strategy Tests
# ---------------------------------------------------------------------------


def _make_bos_data(
    direction: str = "bullish",
    swing_high: float = 1970.0,
    swing_low: float = 1940.0,
    n_bars: int = 60,
) -> pd.DataFrame:
    """Build data with clear swing points then a breakout candle.

    Creates a range-bound series with swing highs/lows, then ends with
    a candle that breaks the structure.
    """
    start = datetime(2024, 1, 1)
    rows: list[dict] = []
    mid = (swing_high + swing_low) / 2.0

    # Build range-bound data with swing points
    for i in range(n_bars - 1):
        dt = start + timedelta(minutes=5 * i)
        # Oscillate around mid to create swing points
        cycle = np.sin(2 * np.pi * i / 12) * (swing_high - swing_low) / 2.5
        c = mid + cycle
        rows.append({
            "datetime": dt,
            "open": c - 0.5,
            "high": c + 2.0,
            "low": c - 2.0,
            "close": c,
            "volume": 1000,
        })

    # Ensure we have definite swing high and low within the data
    # Place a clear swing high about 15 bars before end
    sh_idx = len(rows) - 15
    rows[sh_idx]["high"] = swing_high
    rows[sh_idx]["close"] = swing_high - 1.0
    # Ensure neighbors are lower
    for j in range(1, 5):
        if sh_idx - j >= 0:
            rows[sh_idx - j]["high"] = min(rows[sh_idx - j]["high"], swing_high - 3.0)
        if sh_idx + j < len(rows):
            rows[sh_idx + j]["high"] = min(rows[sh_idx + j]["high"], swing_high - 3.0)

    # Place a clear swing low about 10 bars before end
    sl_idx = len(rows) - 10
    rows[sl_idx]["low"] = swing_low
    rows[sl_idx]["close"] = swing_low + 1.0
    for j in range(1, 5):
        if sl_idx - j >= 0:
            rows[sl_idx - j]["low"] = max(rows[sl_idx - j]["low"], swing_low + 3.0)
        if sl_idx + j < len(rows):
            rows[sl_idx + j]["low"] = max(rows[sl_idx + j]["low"], swing_low + 3.0)

    # Breakout candle
    dt = start + timedelta(minutes=5 * (n_bars - 1))
    if direction == "bullish":
        bp = swing_high + 5.0
        rows.append({
            "datetime": dt,
            "open": swing_high,
            "high": bp + 1.0,
            "low": swing_high - 1.0,
            "close": bp,
            "volume": 2000,
        })
    else:
        bp = swing_low - 5.0
        rows.append({
            "datetime": dt,
            "open": swing_low,
            "high": swing_low + 1.0,
            "low": bp - 1.0,
            "close": bp,
            "volume": 2000,
        })

    return pd.DataFrame(rows)


@pytest.fixture
def bos_strategy() -> BOSStrategy:
    return BOSStrategy()


class TestBOSBuySignal:
    def test_bos_buy_on_swing_high_break(self, bos_strategy: BOSStrategy) -> None:
        df = _make_bos_data("bullish")
        signal = bos_strategy.analyze(df)
        assert signal is not None
        assert signal.direction == Direction.BUY
        assert signal.strategy_name == "bos"

    def test_bos_sl_at_opposite_swing_buy(self, bos_strategy: BOSStrategy) -> None:
        df = _make_bos_data("bullish", swing_low=1940.0)
        signal = bos_strategy.analyze(df)
        assert signal is not None
        # SL should be near (at or below) the swing low
        assert signal.sl < signal.entry_price
        assert signal.sl <= 1940.0


class TestBOSSellSignal:
    def test_bos_sell_on_swing_low_break(self, bos_strategy: BOSStrategy) -> None:
        df = _make_bos_data("bearish")
        signal = bos_strategy.analyze(df)
        assert signal is not None
        assert signal.direction == Direction.SELL
        assert signal.strategy_name == "bos"


class TestBOSNoSignal:
    def test_bos_no_signal_in_range(self, bos_strategy: BOSStrategy) -> None:
        """Price stays between swing high and low → no signal."""
        df = _make_bos_data("bullish", swing_high=1970.0, swing_low=1940.0)
        # Override the last candle close to be in the middle of the range
        df.loc[df.index[-1], "close"] = 1955.0
        df.loc[df.index[-1], "high"] = 1956.0
        signal = bos_strategy.analyze(df)
        assert signal is None


# ---------------------------------------------------------------------------
# CandlePattern Strategy Tests
# ---------------------------------------------------------------------------


def _make_candle_pattern_data(
    pattern: str = "hammer",
    with_confirmation: bool = True,
    with_trend: bool = True,
    n_bars: int = 40,
) -> pd.DataFrame:
    """Build data ending with a candlestick pattern + optional confirmation.

    Args:
        pattern: "hammer" or "shooting_star"
        with_confirmation: whether the current (last) candle confirms
        with_trend: whether there's a prior trend
    """
    start = datetime(2024, 1, 1)
    rows: list[dict] = []

    # Build prior trend
    if pattern == "hammer":
        # Need prior downtrend for hammer
        if with_trend:
            prices = list(np.linspace(1980, 1950, n_bars - 2))
        else:
            prices = [1951.0] * (n_bars - 2)  # flat at pattern level, no trend
    else:
        # Need prior uptrend for shooting star
        if with_trend:
            prices = list(np.linspace(1920, 1960, n_bars - 2))
        else:
            prices = [1959.0] * (n_bars - 2)  # flat at pattern level

    for i, p in enumerate(prices):
        dt = start + timedelta(minutes=5 * i)
        rows.append({
            "datetime": dt,
            "open": p - 0.3,
            "high": p + 1.0,
            "low": p - 1.0,
            "close": p,
            "volume": 1000,
        })

    # Pattern candle (second to last)
    dt = start + timedelta(minutes=5 * (n_bars - 2))
    if pattern == "hammer":
        # Hammer: body at top, long lower wick
        o, c = 1950.0, 1951.0  # small bullish body
        h = 1951.5  # tiny upper wick
        l = 1944.0  # long lower wick (body ~1, lower wick ~6 → ratio=6)
    else:
        # Shooting star: body at bottom, long upper wick
        o, c = 1960.0, 1959.0  # small bearish body
        l = 1958.5  # tiny lower wick
        h = 1966.0  # long upper wick

    rows.append({
        "datetime": dt,
        "open": o,
        "high": h,
        "low": l,
        "close": c,
        "volume": 1500,
    })

    # Confirmation candle (last)
    dt = start + timedelta(minutes=5 * (n_bars - 1))
    if pattern == "hammer":
        if with_confirmation:
            # Bullish confirmation
            rows.append({"datetime": dt, "open": 1951.0, "high": 1955.0, "low": 1950.0, "close": 1954.0, "volume": 1200})
        else:
            # Bearish — no confirmation
            rows.append({"datetime": dt, "open": 1951.0, "high": 1952.0, "low": 1948.0, "close": 1949.0, "volume": 1200})
    else:
        if with_confirmation:
            # Bearish confirmation
            rows.append({"datetime": dt, "open": 1959.0, "high": 1960.0, "low": 1955.0, "close": 1956.0, "volume": 1200})
        else:
            # Bullish — no confirmation
            rows.append({"datetime": dt, "open": 1959.0, "high": 1963.0, "low": 1958.0, "close": 1962.0, "volume": 1200})

    return pd.DataFrame(rows)


@pytest.fixture
def candle_strategy() -> CandlePatternStrategy:
    return CandlePatternStrategy()


class TestHammerBuy:
    def test_hammer_buy_after_downtrend(self, candle_strategy: CandlePatternStrategy) -> None:
        df = _make_candle_pattern_data("hammer", with_confirmation=True, with_trend=True)
        signal = candle_strategy.analyze(df)
        assert signal is not None
        assert signal.direction == Direction.BUY
        assert signal.strategy_name == "candle_pattern"
        assert signal.sl < signal.entry_price
        assert signal.tp > signal.entry_price


class TestShootingStarSell:
    def test_shooting_star_sell_after_uptrend(self, candle_strategy: CandlePatternStrategy) -> None:
        df = _make_candle_pattern_data("shooting_star", with_confirmation=True, with_trend=True)
        signal = candle_strategy.analyze(df)
        assert signal is not None
        assert signal.direction == Direction.SELL
        assert signal.strategy_name == "candle_pattern"
        assert signal.sl > signal.entry_price
        assert signal.tp < signal.entry_price


class TestCandlePatternNoSignal:
    def test_no_signal_without_confirmation(self, candle_strategy: CandlePatternStrategy) -> None:
        df = _make_candle_pattern_data("hammer", with_confirmation=False, with_trend=True)
        signal = candle_strategy.analyze(df)
        assert signal is None

    def test_no_signal_without_trend(self, candle_strategy: CandlePatternStrategy) -> None:
        df = _make_candle_pattern_data("hammer", with_confirmation=True, with_trend=False)
        signal = candle_strategy.analyze(df)
        assert signal is None
