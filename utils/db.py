from __future__ import annotations
import json
import sqlite3
from datetime import date, datetime
from typing import Any
from loguru import logger


class TradeDB:
    """SQLite database helper for trade history storage."""

    def __init__(self, db_path: str = "trades.db") -> None:
        self.db_path = db_path
        self._conn: sqlite3.Connection | None = None

    def connect(self) -> None:
        """Open database connection and create tables."""
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False, timeout=10)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._create_tables()
        logger.debug(f"Connected to database: {self.db_path}")

    def _create_tables(self) -> None:
        """Create trades table if it doesn't exist."""
        assert self._conn is not None
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy TEXT NOT NULL,
                symbol TEXT NOT NULL,
                direction TEXT NOT NULL,
                entry_price REAL NOT NULL,
                exit_price REAL,
                sl REAL NOT NULL,
                tp REAL NOT NULL,
                lot_size REAL NOT NULL,
                pnl REAL DEFAULT 0.0,
                opened_at TEXT NOT NULL,
                closed_at TEXT,
                status TEXT NOT NULL DEFAULT 'open'
            )
        """)
        # --- Learning system tables ---
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS market_features (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                features_json TEXT,
                regime TEXT,
                session TEXT,
                created_at TEXT DEFAULT (datetime('now'))
            )
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS rl_transitions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                state_json TEXT,
                action INTEGER,
                reward REAL,
                next_state_json TEXT,
                trade_id INTEGER,
                timestamp TEXT,
                created_at TEXT DEFAULT (datetime('now'))
            )
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS rl_models (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT,
                model_blob BLOB,
                episode INTEGER,
                epsilon REAL,
                total_reward REAL,
                win_rate REAL,
                created_at TEXT DEFAULT (datetime('now'))
            )
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS claude_insights (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                review_type TEXT,
                analysis_json TEXT,
                recommendations_json TEXT,
                strategy_rules_json TEXT,
                backtest_score REAL,
                approved INTEGER DEFAULT 0,
                created_at TEXT DEFAULT (datetime('now'))
            )
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS strategy_rules (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                version INTEGER DEFAULT 1,
                rules_json TEXT,
                source TEXT,
                backtest_win_rate REAL,
                backtest_pf REAL,
                active INTEGER DEFAULT 1,
                created_at TEXT DEFAULT (datetime('now'))
            )
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS trade_features (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trade_id INTEGER REFERENCES trades(id),
                features_json TEXT,
                regime TEXT,
                confidence REAL,
                rl_action INTEGER,
                rl_q_values_json TEXT,
                claude_reasoning TEXT,
                created_at TEXT DEFAULT (datetime('now'))
            )
        """)
        self._conn.commit()

    # ------------------------------------------------------------------
    # Original trade methods
    # ------------------------------------------------------------------

    def insert_trade(self, trade: dict[str, Any]) -> int:
        """Insert a new trade record. Returns the trade ID."""
        assert self._conn is not None
        cursor = self._conn.execute(
            """INSERT INTO trades
               (strategy, symbol, direction, entry_price, exit_price, sl, tp, lot_size, pnl, opened_at, closed_at, status)
               VALUES (:strategy, :symbol, :direction, :entry_price, :exit_price, :sl, :tp, :lot_size, :pnl, :opened_at, :closed_at, :status)""",
            trade,
        )
        self._conn.commit()
        logger.info(f"Inserted trade #{cursor.lastrowid}: {trade['direction']} {trade['symbol']}")
        return cursor.lastrowid  # type: ignore[return-value]

    def update_trade(self, trade_id: int, updates: dict[str, Any]) -> None:
        """Update a trade record by ID."""
        assert self._conn is not None
        set_clause = ", ".join(f"{k} = :{k}" for k in updates)
        updates["id"] = trade_id
        self._conn.execute(f"UPDATE trades SET {set_clause} WHERE id = :id", updates)
        self._conn.commit()

    def get_trades_by_date(self, trade_date: date) -> list[dict[str, Any]]:
        """Get all trades opened on a specific date."""
        assert self._conn is not None
        date_str = trade_date.isoformat()
        cursor = self._conn.execute(
            "SELECT * FROM trades WHERE opened_at LIKE ?", (f"{date_str}%",)
        )
        return [dict(row) for row in cursor.fetchall()]

    def get_trades_by_strategy(self, strategy_name: str) -> list[dict[str, Any]]:
        """Get all trades for a specific strategy."""
        assert self._conn is not None
        cursor = self._conn.execute(
            "SELECT * FROM trades WHERE strategy = ?", (strategy_name,)
        )
        return [dict(row) for row in cursor.fetchall()]

    def get_daily_pnl(self, trade_date: date) -> float:
        """Calculate total PnL for a specific date."""
        assert self._conn is not None
        date_str = trade_date.isoformat()
        cursor = self._conn.execute(
            "SELECT COALESCE(SUM(pnl), 0.0) FROM trades WHERE opened_at LIKE ?",
            (f"{date_str}%",),
        )
        result = cursor.fetchone()
        return float(result[0])

    def get_open_trades(self) -> list[dict[str, Any]]:
        """Get all currently open trades."""
        assert self._conn is not None
        cursor = self._conn.execute("SELECT * FROM trades WHERE status = 'open'")
        return [dict(row) for row in cursor.fetchall()]

    # ------------------------------------------------------------------
    # Learning system methods
    # ------------------------------------------------------------------

    def insert_market_features(self, timestamp: str, features: dict, regime: str, session: str) -> int:
        """Store market features snapshot."""
        assert self._conn is not None
        cursor = self._conn.execute(
            "INSERT INTO market_features (timestamp, features_json, regime, session) VALUES (?, ?, ?, ?)",
            (timestamp, json.dumps(features), regime, session),
        )
        self._conn.commit()
        return cursor.lastrowid  # type: ignore

    def insert_rl_transition(
        self, state: list, action: int, reward: float, next_state: list,
        trade_id: int | None = None, timestamp: str = "",
    ) -> int:
        """Store an RL transition."""
        assert self._conn is not None
        cursor = self._conn.execute(
            """INSERT INTO rl_transitions (state_json, action, reward, next_state_json, trade_id, timestamp)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (json.dumps(state), action, reward, json.dumps(next_state), trade_id, timestamp),
        )
        self._conn.commit()
        return cursor.lastrowid  # type: ignore

    def save_rl_model(
        self, model_name: str, model_blob: bytes, episode: int,
        epsilon: float, total_reward: float, win_rate: float,
    ) -> int:
        """Save RL model checkpoint."""
        assert self._conn is not None
        cursor = self._conn.execute(
            """INSERT INTO rl_models (model_name, model_blob, episode, epsilon, total_reward, win_rate)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (model_name, model_blob, episode, epsilon, total_reward, win_rate),
        )
        self._conn.commit()
        return cursor.lastrowid  # type: ignore

    def load_latest_rl_model(self, model_name: str = "dqn") -> dict[str, Any] | None:
        """Load the latest RL model checkpoint."""
        assert self._conn is not None
        cursor = self._conn.execute(
            "SELECT * FROM rl_models WHERE model_name = ? ORDER BY id DESC LIMIT 1",
            (model_name,),
        )
        row = cursor.fetchone()
        return dict(row) if row else None

    def insert_claude_insight(
        self, review_type: str, analysis: str, recommendations: list,
        strategy_rules: list, backtest_score: float, approved: bool = False,
    ) -> int:
        """Store a Claude review insight."""
        assert self._conn is not None
        cursor = self._conn.execute(
            """INSERT INTO claude_insights
               (review_type, analysis_json, recommendations_json, strategy_rules_json, backtest_score, approved)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                review_type,
                json.dumps(analysis),
                json.dumps(recommendations),
                json.dumps(strategy_rules),
                backtest_score,
                int(approved),
            ),
        )
        self._conn.commit()
        return cursor.lastrowid  # type: ignore

    def get_recent_insights(self, limit: int = 20) -> list[dict[str, Any]]:
        """Get recent Claude insights."""
        assert self._conn is not None
        cursor = self._conn.execute(
            "SELECT * FROM claude_insights ORDER BY id DESC LIMIT ?", (limit,)
        )
        rows = [dict(row) for row in cursor.fetchall()]
        for row in rows:
            for key in ("analysis_json", "recommendations_json", "strategy_rules_json"):
                if row.get(key):
                    try:
                        row[key] = json.loads(row[key])
                    except (json.JSONDecodeError, TypeError):
                        pass
        return rows

    def insert_strategy_rule(
        self, name: str, rules: list[dict], source: str,
        backtest_win_rate: float = 0.0, backtest_pf: float = 0.0,
    ) -> int:
        """Store a strategy rule set."""
        assert self._conn is not None
        cursor = self._conn.execute(
            """INSERT INTO strategy_rules (name, rules_json, source, backtest_win_rate, backtest_pf)
               VALUES (?, ?, ?, ?, ?)""",
            (name, json.dumps(rules), source, backtest_win_rate, backtest_pf),
        )
        self._conn.commit()
        return cursor.lastrowid  # type: ignore

    def get_active_rules(self) -> list[dict[str, Any]]:
        """Get all active strategy rules."""
        assert self._conn is not None
        cursor = self._conn.execute(
            "SELECT * FROM strategy_rules WHERE active = 1 ORDER BY id DESC"
        )
        rows = [dict(row) for row in cursor.fetchall()]
        for row in rows:
            if row.get("rules_json"):
                try:
                    row["rules_json"] = json.loads(row["rules_json"])
                except (json.JSONDecodeError, TypeError):
                    pass
        return rows

    def insert_trade_features(
        self, trade_id: int, features: dict, regime: str, confidence: float,
        rl_action: int | None = None, rl_q_values: list | None = None,
        claude_reasoning: str = "",
    ) -> int:
        """Store per-trade feature context."""
        assert self._conn is not None
        cursor = self._conn.execute(
            """INSERT INTO trade_features
               (trade_id, features_json, regime, confidence, rl_action, rl_q_values_json, claude_reasoning)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                trade_id,
                json.dumps(features),
                regime,
                confidence,
                rl_action,
                json.dumps(rl_q_values) if rl_q_values else None,
                claude_reasoning,
            ),
        )
        self._conn.commit()
        return cursor.lastrowid  # type: ignore

    def get_recent_closed_trades(self, limit: int = 50) -> list[dict[str, Any]]:
        """Get recent closed trades with optional features."""
        assert self._conn is not None
        cursor = self._conn.execute(
            """SELECT t.*, tf.regime as trade_regime, tf.confidence as trade_confidence,
                      tf.claude_reasoning
               FROM trades t
               LEFT JOIN trade_features tf ON t.id = tf.trade_id
               WHERE t.status = 'closed'
               ORDER BY t.id DESC LIMIT ?""",
            (limit,),
        )
        return [dict(row) for row in cursor.fetchall()]

    def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
            logger.debug("Database connection closed")
