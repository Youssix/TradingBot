# TradingBot MCP Server

Custom MCP (Model Context Protocol) server that gives Claude direct introspection into the trading system.

## What it does

Instead of Claude guessing what's wrong, it can now query your bot in real-time:

- **Live status** — Is the bot running? What's the account balance? What model is active?
- **Trade analysis** — All trades with filtering, stats, strategy comparison
- **Model ranking** — Which RL model performs best? Compare by win rate, episodes, profile
- **Diagnostics** — Auto-detect overtrading, poor win rate, high drawdown, rapid cycling
- **Equity curve** — Track cumulative P&L over time

## 16 Tools Available

### Live Status
| Tool | What it does |
|------|-------------|
| `bot_status` | Running mode, account info, active model, open position count |
| `open_positions` | All open positions with entry, SL, TP, duration |
| `account_info` | Balance, equity, margin, profit, leverage |

### Trade Analysis
| Tool | What it does |
|------|-------------|
| `get_trades` | Query trades with filters (status, strategy, direction, date, count) |
| `trade_stats` | Win rate, profit factor, Sharpe, avg win/loss, max drawdown |
| `strategy_performance` | Side-by-side comparison of all strategies |
| `daily_summary` | P&L, trade count, win rate per day |
| `hourly_trade_distribution` | Trades per hour — spot overtrading patterns |
| `trade_pnl_distribution` | P&L histogram — see the distribution of outcomes |
| `equity_curve_data` | Cumulative P&L with drawdown tracking |
| `recent_signals` | Trades with confidence scores, regime, RL action context |

### Model Analysis
| Tool | What it does |
|------|-------------|
| `list_models` | All saved RL models with metrics |
| `model_comparison` | Rank models by win rate, best per profile/timeframe |
| `training_status` | Readiness, epsilon, episode, pipeline progress |

### Diagnostics
| Tool | What it does |
|------|-------------|
| `diagnose` | Auto-detect issues: overtrading, low WR, drawdown, rapid cycling, direction bias |
| `get_config` | Current bot configuration |

## Setup

The MCP server is already registered in `~/.claude.json`. It runs automatically when Claude Code starts.

To test manually:
```bash
python3 mcp_server.py
```

## How Claude uses it

When you ask Claude about trading performance, it will automatically call these tools:

- "What's wrong with my bot?" → `diagnose`
- "How are my trades doing?" → `trade_stats` + `strategy_performance`
- "Which model is best?" → `model_comparison`
- "Is the bot running?" → `bot_status`
- "Show me recent trades" → `get_trades`

## Requirements

- `mcp` Python package (installed via `pip3 install "mcp[cli]"`)
- `httpx` for API calls
- The FastAPI backend must be running on `http://localhost:8000`
- SQLite database at `trades.db`
