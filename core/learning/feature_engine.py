"""Extract normalized market features from primary + context timeframes."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import pandas_ta as ta
from loguru import logger

# Maps primary TF to its context (higher) timeframes
CONTEXT_TF_MAP: dict[str, list[str]] = {
    "M1": ["M5", "M15"],
    "M5": ["H1", "H4"],
    "M15": ["H1", "H4"],
    "H1": ["H4", "D1"],
    "H4": ["D1", "W1"],
    "D1": ["W1", "MN"],
}


def _safe(val: Any) -> float:
    """Convert to float, replacing NaN with 0."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return 0.0
    return float(val)


def _normalize(val: float, lo: float, hi: float) -> float:
    """Min-max normalize to [0, 1]."""
    if hi == lo:
        return 0.5
    return max(0.0, min(1.0, (val - lo) / (hi - lo)))


def _detect_session(dt: datetime) -> str:
    """Classify trading session from UTC hour."""
    h = dt.hour
    if 0 <= h < 8:
        return "asian"
    if 8 <= h < 16:
        return "london"
    return "newyork"


def _session_one_hot(session: str) -> list[float]:
    """One-hot encode session: [asian, london, newyork]."""
    return [
        1.0 if session == "asian" else 0.0,
        1.0 if session == "london" else 0.0,
        1.0 if session == "newyork" else 0.0,
    ]


class FeatureEngine:
    """Extract 20+ normalized features from OHLCV data.

    Features:
      0  rsi_norm        RSI / 100
      1  atr_norm        ATR / close (volatility ratio)
      2  adx_norm        ADX / 100
      3  ema_fast_dist   (close - EMA9) / ATR
      4  ema_slow_dist   (close - EMA21) / ATR
      5  ema_cross       1 if fast > slow, 0 otherwise
      6  macd_norm       MACD / ATR
      7  macd_signal_d   (MACD - signal) / ATR
      8  bb_pos          Bollinger %B (0-1)
      9  bb_width        BB width / close
     10  stoch_k         Stochastic %K / 100
     11  stoch_d         Stochastic %D / 100
     12  volume_ratio    current vol / SMA(vol, 20)
     13  close_change    (close - prev_close) / ATR
     14  high_low_range  (high - low) / ATR
     15  candle_body     (close - open) / ATR
     16  session_asian   1 if Asian session
     17  session_london  1 if London session
     18  session_ny      1 if New York session
     19  htf1_trend      Higher-TF EMA trend (+1/0/-1) normalized
     20  htf2_trend      Second higher-TF EMA trend
     21  hour_sin        sin(2*pi*hour/24)
     22  hour_cos        cos(2*pi*hour/24)
    """

    FEATURE_DIM = 23

    def __init__(
        self,
        primary_tf: str = "M1",
        ema_fast: int = 9,
        ema_slow: int = 21,
        rsi_period: int = 14,
        atr_period: int = 14,
        adx_period: int = 14,
        bb_period: int = 20,
    ) -> None:
        self._primary_tf = primary_tf
        self._context_tfs = CONTEXT_TF_MAP.get(primary_tf, ["H1", "H4"])
        self._ema_fast = ema_fast
        self._ema_slow = ema_slow
        self._rsi_period = rsi_period
        self._atr_period = atr_period
        self._adx_period = adx_period
        self._bb_period = bb_period

    @property
    def context_timeframes(self) -> list[str]:
        return list(self._context_tfs)

    def extract(
        self,
        df: pd.DataFrame,
        htf_data: dict[str, pd.DataFrame] | None = None,
    ) -> dict[str, Any]:
        """Extract feature vector from primary OHLCV DataFrame.

        Args:
            df: Primary timeframe OHLCV with columns datetime,open,high,low,close,volume
            htf_data: Optional dict mapping context TF names to their OHLCV DataFrames

        Returns:
            Dict with 'vector' (list[float]), 'names' (list[str]),
            'session' (str), 'timestamp' (datetime)
        """
        if len(df) < max(self._ema_slow, self._bb_period, self._adx_period) + 5:
            return self._empty_result(df)

        df = df.copy()
        c = df.iloc[-1]
        close = float(c["close"])
        dt = c.get("datetime", datetime.now())

        # --- Technical indicators ---
        rsi_s = ta.rsi(df["close"], length=self._rsi_period)
        atr_s = ta.atr(df["high"], df["low"], df["close"], length=self._atr_period)
        adx_df = ta.adx(df["high"], df["low"], df["close"], length=self._adx_period)
        ema_fast_s = ta.ema(df["close"], length=self._ema_fast)
        ema_slow_s = ta.ema(df["close"], length=self._ema_slow)
        macd_df = ta.macd(df["close"])
        bb_df = ta.bbands(df["close"], length=self._bb_period)
        stoch_df = ta.stoch(df["high"], df["low"], df["close"])

        rsi = _safe(rsi_s.iloc[-1]) if rsi_s is not None else 50.0
        atr = _safe(atr_s.iloc[-1]) if atr_s is not None else 1.0
        atr = max(atr, 0.01)  # avoid div-by-zero

        adx = 25.0
        if adx_df is not None:
            adx_col = [col for col in adx_df.columns if "ADX" in col]
            if adx_col:
                adx = _safe(adx_df[adx_col[0]].iloc[-1])

        ema_fast_val = _safe(ema_fast_s.iloc[-1]) if ema_fast_s is not None else close
        ema_slow_val = _safe(ema_slow_s.iloc[-1]) if ema_slow_s is not None else close

        macd_val = 0.0
        macd_signal = 0.0
        if macd_df is not None:
            macd_cols = macd_df.columns.tolist()
            if len(macd_cols) >= 2:
                macd_val = _safe(macd_df[macd_cols[0]].iloc[-1])
                macd_signal = _safe(macd_df[macd_cols[1]].iloc[-1])

        bb_lower, bb_mid, bb_upper = close - atr, close, close + atr
        bb_pctb = 0.5
        bb_width = 0.0
        if bb_df is not None:
            bb_cols = bb_df.columns.tolist()
            if len(bb_cols) >= 3:
                bb_lower = _safe(bb_df[bb_cols[0]].iloc[-1])
                bb_mid = _safe(bb_df[bb_cols[1]].iloc[-1])
                bb_upper = _safe(bb_df[bb_cols[2]].iloc[-1])
                if bb_upper != bb_lower:
                    bb_pctb = (close - bb_lower) / (bb_upper - bb_lower)
                bb_width = (bb_upper - bb_lower) / close if close > 0 else 0.0

        stoch_k, stoch_d = 50.0, 50.0
        if stoch_df is not None:
            stoch_cols = stoch_df.columns.tolist()
            if len(stoch_cols) >= 2:
                stoch_k = _safe(stoch_df[stoch_cols[0]].iloc[-1])
                stoch_d = _safe(stoch_df[stoch_cols[1]].iloc[-1])

        # Volume ratio
        vol_sma = df["volume"].rolling(20).mean().iloc[-1]
        vol_ratio = float(c["volume"]) / max(float(vol_sma), 1.0) if not pd.isna(vol_sma) else 1.0

        # Price features
        prev_close = float(df.iloc[-2]["close"]) if len(df) > 1 else close
        close_change = (close - prev_close) / atr
        high_low_range = (float(c["high"]) - float(c["low"])) / atr
        candle_body = (close - float(c["open"])) / atr

        # Session
        session = _detect_session(dt)
        session_oh = _session_one_hot(session)

        # HTF trends
        htf1_trend = 0.0
        htf2_trend = 0.0
        if htf_data:
            tfs = self._context_tfs
            if tfs[0] in htf_data and len(htf_data[tfs[0]]) > self._ema_slow:
                htf_ema_f = ta.ema(htf_data[tfs[0]]["close"], length=self._ema_fast)
                htf_ema_s = ta.ema(htf_data[tfs[0]]["close"], length=self._ema_slow)
                if htf_ema_f is not None and htf_ema_s is not None:
                    htf1_trend = 1.0 if _safe(htf_ema_f.iloc[-1]) > _safe(htf_ema_s.iloc[-1]) else -1.0
            if len(tfs) > 1 and tfs[1] in htf_data and len(htf_data[tfs[1]]) > self._ema_slow:
                htf_ema_f = ta.ema(htf_data[tfs[1]]["close"], length=self._ema_fast)
                htf_ema_s = ta.ema(htf_data[tfs[1]]["close"], length=self._ema_slow)
                if htf_ema_f is not None and htf_ema_s is not None:
                    htf2_trend = 1.0 if _safe(htf_ema_f.iloc[-1]) > _safe(htf_ema_s.iloc[-1]) else -1.0

        # Time features (cyclical encoding)
        hour = dt.hour if hasattr(dt, "hour") else 12
        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)

        # Build feature vector
        vector = [
            rsi / 100.0,                             # 0  rsi_norm
            atr / close if close > 0 else 0.0,       # 1  atr_norm
            adx / 100.0,                              # 2  adx_norm
            np.clip((close - ema_fast_val) / atr, -3, 3) / 3,  # 3  ema_fast_dist
            np.clip((close - ema_slow_val) / atr, -3, 3) / 3,  # 4  ema_slow_dist
            1.0 if ema_fast_val > ema_slow_val else 0.0,        # 5  ema_cross
            np.clip(macd_val / atr, -3, 3) / 3,      # 6  macd_norm
            np.clip((macd_val - macd_signal) / atr, -3, 3) / 3,  # 7  macd_signal_d
            np.clip(bb_pctb, 0, 1),                   # 8  bb_pos
            bb_width,                                  # 9  bb_width
            stoch_k / 100.0,                          # 10 stoch_k
            stoch_d / 100.0,                          # 11 stoch_d
            min(vol_ratio, 5.0) / 5.0,               # 12 volume_ratio
            np.clip(close_change, -3, 3) / 3,         # 13 close_change
            min(high_low_range, 5.0) / 5.0,           # 14 high_low_range
            np.clip(candle_body, -3, 3) / 3,           # 15 candle_body
            session_oh[0],                             # 16 session_asian
            session_oh[1],                             # 17 session_london
            session_oh[2],                             # 18 session_ny
            (htf1_trend + 1) / 2,                      # 19 htf1_trend (0-1)
            (htf2_trend + 1) / 2,                      # 20 htf2_trend (0-1)
            (hour_sin + 1) / 2,                        # 21 hour_sin (0-1)
            (hour_cos + 1) / 2,                        # 22 hour_cos (0-1)
        ]

        names = [
            "rsi_norm", "atr_norm", "adx_norm", "ema_fast_dist", "ema_slow_dist",
            "ema_cross", "macd_norm", "macd_signal_d", "bb_pos", "bb_width",
            "stoch_k", "stoch_d", "volume_ratio", "close_change", "high_low_range",
            "candle_body", "session_asian", "session_london", "session_ny",
            "htf1_trend", "htf2_trend", "hour_sin", "hour_cos",
        ]

        return {
            "vector": [float(v) for v in vector],
            "names": names,
            "session": session,
            "timestamp": dt,
            "close": close,
            "atr": atr,
            "rsi": rsi,
            "adx": adx,
        }

    def _empty_result(self, df: pd.DataFrame) -> dict[str, Any]:
        """Return zero-vector when not enough data."""
        dt = df.iloc[-1].get("datetime", datetime.now()) if len(df) > 0 else datetime.now()
        return {
            "vector": [0.0] * self.FEATURE_DIM,
            "names": [""] * self.FEATURE_DIM,
            "session": "unknown",
            "timestamp": dt,
            "close": 0.0,
            "atr": 0.0,
            "rsi": 50.0,
            "adx": 25.0,
        }
