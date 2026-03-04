from __future__ import annotations

from datetime import datetime, timezone

import pytest

from core.risk_manager import RiskConfig, RiskManager


@pytest.fixture
def config() -> RiskConfig:
    return RiskConfig(
        risk_pct=1.0,
        max_open_trades=2,
        max_daily_trades=6,
        max_daily_drawdown_pct=3.0,
        max_total_drawdown_pct=10.0,
        friday_cutoff_hour=20,
        news_hours=[(13, 15)],
    )


@pytest.fixture
def rm(config: RiskConfig) -> RiskManager:
    return RiskManager(config)


class TestPositionSizing:
    def test_standard_calculation(self, rm: RiskManager) -> None:
        size = rm.calculate_position_size(
            account_balance=10000.0, sl_points=50.0, point_value=10.0
        )
        # 1% of 10000 = 100, 100 / (50 * 10) = 0.20
        assert size == 0.20

    def test_custom_risk_pct(self, rm: RiskManager) -> None:
        size = rm.calculate_position_size(
            account_balance=10000.0, sl_points=50.0, point_value=10.0, risk_pct=2.0
        )
        assert size == 0.40

    def test_minimum_lot_size(self, rm: RiskManager) -> None:
        size = rm.calculate_position_size(
            account_balance=100.0, sl_points=500.0, point_value=10.0
        )
        assert size == 0.01

    def test_zero_sl_returns_zero(self, rm: RiskManager) -> None:
        assert rm.calculate_position_size(10000.0, 0.0, 10.0) == 0.0

    def test_zero_point_value_returns_zero(self, rm: RiskManager) -> None:
        assert rm.calculate_position_size(10000.0, 50.0, 0.0) == 0.0

    def test_negative_sl_returns_zero(self, rm: RiskManager) -> None:
        assert rm.calculate_position_size(10000.0, -10.0, 10.0) == 0.0


class TestMaxOpenTrades:
    def test_below_limit(self, rm: RiskManager) -> None:
        assert rm.check_max_open_trades(1) is True

    def test_at_limit(self, rm: RiskManager) -> None:
        assert rm.check_max_open_trades(2) is False

    def test_above_limit(self, rm: RiskManager) -> None:
        assert rm.check_max_open_trades(5) is False

    def test_zero_open(self, rm: RiskManager) -> None:
        assert rm.check_max_open_trades(0) is True


class TestMaxDailyTrades:
    def test_below_limit(self, rm: RiskManager) -> None:
        assert rm.check_max_daily_trades(3) is True

    def test_at_limit(self, rm: RiskManager) -> None:
        assert rm.check_max_daily_trades(6) is False

    def test_above_limit(self, rm: RiskManager) -> None:
        assert rm.check_max_daily_trades(10) is False


class TestDailyDrawdown:
    def test_no_drawdown(self, rm: RiskManager) -> None:
        assert rm.check_daily_drawdown(100.0, 10000.0) is True

    def test_within_limit(self, rm: RiskManager) -> None:
        assert rm.check_daily_drawdown(-200.0, 10000.0) is True  # 2% < 3%

    def test_at_limit(self, rm: RiskManager) -> None:
        assert rm.check_daily_drawdown(-300.0, 10000.0) is False  # 3% >= 3%

    def test_beyond_limit(self, rm: RiskManager) -> None:
        assert rm.check_daily_drawdown(-500.0, 10000.0) is False

    def test_zero_balance(self, rm: RiskManager) -> None:
        assert rm.check_daily_drawdown(-100.0, 0.0) is False


class TestTotalDrawdown:
    def test_no_drawdown(self, rm: RiskManager) -> None:
        assert rm.check_total_drawdown(500.0, 10000.0) is True

    def test_within_limit(self, rm: RiskManager) -> None:
        assert rm.check_total_drawdown(-800.0, 10000.0) is True  # 8% < 10%

    def test_at_limit(self, rm: RiskManager) -> None:
        assert rm.check_total_drawdown(-1000.0, 10000.0) is False  # 10% >= 10%

    def test_beyond_limit(self, rm: RiskManager) -> None:
        assert rm.check_total_drawdown(-1500.0, 10000.0) is False


class TestFridayCutoff:
    def test_friday_before_cutoff(self, rm: RiskManager) -> None:
        # Friday at 18:00 UTC
        t = datetime(2024, 1, 5, 18, 0, tzinfo=timezone.utc)
        assert rm.is_friday_cutoff(t) is False

    def test_friday_at_cutoff(self, rm: RiskManager) -> None:
        # Friday at 20:00 UTC
        t = datetime(2024, 1, 5, 20, 0, tzinfo=timezone.utc)
        assert rm.is_friday_cutoff(t) is True

    def test_friday_after_cutoff(self, rm: RiskManager) -> None:
        t = datetime(2024, 1, 5, 22, 0, tzinfo=timezone.utc)
        assert rm.is_friday_cutoff(t) is True

    def test_monday(self, rm: RiskManager) -> None:
        t = datetime(2024, 1, 1, 22, 0, tzinfo=timezone.utc)
        assert rm.is_friday_cutoff(t) is False

    def test_naive_datetime(self, rm: RiskManager) -> None:
        # Friday naive datetime
        t = datetime(2024, 1, 5, 21, 0)
        assert rm.is_friday_cutoff(t) is True


class TestNewsHour:
    def test_during_news(self, rm: RiskManager) -> None:
        t = datetime(2024, 1, 3, 14, 0, tzinfo=timezone.utc)
        assert rm.is_news_hour(t) is True

    def test_outside_news(self, rm: RiskManager) -> None:
        t = datetime(2024, 1, 3, 10, 0, tzinfo=timezone.utc)
        assert rm.is_news_hour(t) is False

    def test_no_news_hours_configured(self) -> None:
        rm = RiskManager(RiskConfig(news_hours=None))
        t = datetime(2024, 1, 3, 14, 0, tzinfo=timezone.utc)
        assert rm.is_news_hour(t) is False


class TestCanTrade:
    def test_all_checks_pass(self, rm: RiskManager) -> None:
        account = {"balance": 10000.0, "profit": 0.0}
        # Wednesday at 10:00 UTC
        t = datetime(2024, 1, 3, 10, 0, tzinfo=timezone.utc)
        allowed, reason = rm.can_trade(account, [], 0, 0.0, t)
        assert allowed is True
        assert reason == "OK"

    def test_blocked_by_friday_cutoff(self, rm: RiskManager) -> None:
        account = {"balance": 10000.0, "profit": 0.0}
        t = datetime(2024, 1, 5, 21, 0, tzinfo=timezone.utc)
        allowed, reason = rm.can_trade(account, [], 0, 0.0, t)
        assert allowed is False
        assert "Friday" in reason

    def test_blocked_by_news_hour(self, rm: RiskManager) -> None:
        account = {"balance": 10000.0, "profit": 0.0}
        t = datetime(2024, 1, 3, 14, 0, tzinfo=timezone.utc)
        allowed, reason = rm.can_trade(account, [], 0, 0.0, t)
        assert allowed is False
        assert "News" in reason

    def test_blocked_by_max_open_trades(self, rm: RiskManager) -> None:
        account = {"balance": 10000.0, "profit": 0.0}
        positions = [{"ticket": 1}, {"ticket": 2}]
        t = datetime(2024, 1, 3, 10, 0, tzinfo=timezone.utc)
        allowed, reason = rm.can_trade(account, positions, 0, 0.0, t)
        assert allowed is False
        assert "open trades" in reason.lower()

    def test_blocked_by_daily_drawdown(self, rm: RiskManager) -> None:
        account = {"balance": 10000.0, "profit": 0.0}
        t = datetime(2024, 1, 3, 10, 0, tzinfo=timezone.utc)
        allowed, reason = rm.can_trade(account, [], 0, -400.0, t)
        assert allowed is False
        assert "drawdown" in reason.lower()
