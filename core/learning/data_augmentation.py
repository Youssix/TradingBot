"""Data augmentation for RL training: multiply historical data via transformations."""

from __future__ import annotations

import numpy as np
import pandas as pd


def augment_dataframe(
    df: pd.DataFrame,
    factor: int = 3,
    seed: int = 42,
) -> list[pd.DataFrame]:
    """Generate augmented copies of an OHLCV DataFrame.

    Each copy applies 2-3 randomly selected techniques:
      - Gaussian noise (0.05-0.15% of price)
      - Price level scaling (0.8x-1.2x)
      - Volatility scaling (0.7-1.3x range around close)
      - Time reversal (reverse + swap open/close)

    Args:
        df: DataFrame with columns [datetime, open, high, low, close, volume].
        factor: Number of augmented copies to generate (0-5, capped at 5).
        seed: Random seed for reproducibility.

    Returns:
        List of augmented DataFrames (does NOT include the original).
    """
    factor = min(max(factor, 0), 5)
    if factor == 0 or df.empty:
        return []

    rng = np.random.default_rng(seed)
    results: list[pd.DataFrame] = []

    techniques = [_gaussian_noise, _price_scaling, _volatility_scaling, _time_reversal]

    for i in range(factor):
        copy = df.copy()

        # Pick 2-3 random techniques for this copy
        n_techniques = rng.integers(2, 4)  # 2 or 3
        chosen = rng.choice(len(techniques), size=n_techniques, replace=False)

        for idx in chosen:
            copy = techniques[idx](copy, rng)

        # Ensure data integrity
        copy = _ensure_integrity(copy)
        results.append(copy.reset_index(drop=True))

    return results


def _gaussian_noise(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Add Gaussian noise (0.05-0.15% of price) to OHLC columns."""
    noise_pct = rng.uniform(0.0005, 0.0015)
    for col in ["open", "high", "low", "close"]:
        noise = rng.normal(0, noise_pct * df[col].values)
        df[col] = df[col].values + noise
    return df


def _price_scaling(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Scale all prices by a random factor (0.8x-1.2x)."""
    scale = rng.uniform(0.8, 1.2)
    for col in ["open", "high", "low", "close"]:
        df[col] = df[col].values * scale
    return df


def _volatility_scaling(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Scale the range around close by 0.7-1.3x."""
    vol_scale = rng.uniform(0.7, 1.3)
    close = df["close"].values
    for col in ["open", "high", "low"]:
        diff = df[col].values - close
        df[col] = close + diff * vol_scale
    return df


def _time_reversal(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Reverse the time series and swap open/close."""
    df = df.iloc[::-1].reset_index(drop=True)
    # Swap open and close (a reversed candle)
    df["open"], df["close"] = df["close"].copy(), df["open"].copy()
    # Keep original datetime sequence (don't reverse timestamps)
    return df


def _ensure_integrity(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure high >= low and no negative prices."""
    for col in ["open", "high", "low", "close"]:
        df[col] = df[col].clip(lower=1e-6)

    # Fix high/low to contain open and close
    ohlc_max = df[["open", "close"]].max(axis=1)
    ohlc_min = df[["open", "close"]].min(axis=1)
    df["high"] = df["high"].clip(lower=ohlc_max)
    df["low"] = df["low"].clip(upper=ohlc_min)

    return df
