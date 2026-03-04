from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from core.strategy import AsianRangeBreakoutStrategy, Direction, EMACrossoverStrategy, Signal


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
