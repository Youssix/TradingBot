"""Tests for the data augmentation module."""

import numpy as np
import pandas as pd
import pytest

from core.learning.data_augmentation import augment_dataframe


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Create a simple OHLCV DataFrame for testing."""
    n = 100
    rng = np.random.default_rng(0)
    close = 2000.0 + np.cumsum(rng.normal(0, 2, n))
    return pd.DataFrame({
        "datetime": pd.date_range("2024-01-01", periods=n, freq="h"),
        "open": close + rng.normal(0, 1, n),
        "high": close + abs(rng.normal(2, 1, n)),
        "low": close - abs(rng.normal(2, 1, n)),
        "close": close,
        "volume": rng.integers(100, 10000, n),
    })


def test_correct_count(sample_df: pd.DataFrame):
    """augment_dataframe returns exactly `factor` copies."""
    result = augment_dataframe(sample_df, factor=3, seed=42)
    assert len(result) == 3

    result = augment_dataframe(sample_df, factor=1, seed=42)
    assert len(result) == 1


def test_shape_preserved(sample_df: pd.DataFrame):
    """Each augmented copy has the same shape as the original."""
    for aug in augment_dataframe(sample_df, factor=3, seed=42):
        assert aug.shape == sample_df.shape


def test_columns_preserved(sample_df: pd.DataFrame):
    """Each augmented copy keeps the same columns."""
    for aug in augment_dataframe(sample_df, factor=3, seed=42):
        assert list(aug.columns) == list(sample_df.columns)


def test_no_negative_prices(sample_df: pd.DataFrame):
    """All price columns remain positive."""
    for aug in augment_dataframe(sample_df, factor=5, seed=42):
        for col in ["open", "high", "low", "close"]:
            assert (aug[col] > 0).all(), f"Negative price in {col}"


def test_high_gte_low(sample_df: pd.DataFrame):
    """high >= low in every row of every augmented copy."""
    for aug in augment_dataframe(sample_df, factor=5, seed=42):
        assert (aug["high"] >= aug["low"]).all()


def test_factor_zero_returns_empty(sample_df: pd.DataFrame):
    """factor=0 returns an empty list."""
    assert augment_dataframe(sample_df, factor=0) == []


def test_factor_capped_at_five(sample_df: pd.DataFrame):
    """factor above 5 is capped to 5."""
    result = augment_dataframe(sample_df, factor=10, seed=42)
    assert len(result) == 5


def test_deterministic_with_seed(sample_df: pd.DataFrame):
    """Same seed produces identical results."""
    a = augment_dataframe(sample_df, factor=3, seed=123)
    b = augment_dataframe(sample_df, factor=3, seed=123)
    for df_a, df_b in zip(a, b):
        pd.testing.assert_frame_equal(df_a, df_b)


def test_different_from_original(sample_df: pd.DataFrame):
    """Augmented data should differ from the original."""
    for aug in augment_dataframe(sample_df, factor=3, seed=42):
        # At least one OHLC column should differ
        different = False
        for col in ["open", "high", "low", "close"]:
            if not np.allclose(aug[col].values, sample_df[col].values, atol=1e-3):
                different = True
                break
        assert different, "Augmented copy is identical to original"


def test_time_reversal_direction(sample_df: pd.DataFrame):
    """At least one augmented copy should have reversed order (close values)."""
    # Use a seed and factor that will include time reversal
    augmented = augment_dataframe(sample_df, factor=5, seed=42)
    original_close = sample_df["close"].values

    found_reversed = False
    for aug in augmented:
        aug_close = aug["close"].values
        # Check if close values are approximately reversed
        reversed_orig = original_close[::-1]
        if np.corrcoef(aug_close[:20], reversed_orig[:20])[0, 1] > 0.5:
            found_reversed = True
            break

    # With 5 copies and 4 techniques, time reversal should appear at least once
    assert found_reversed, "No augmented copy shows reversed time direction"
