// --- Types ---

export interface AccountInfo {
  balance: number;
  equity: number;
  margin: number;
  free_margin: number;
  profit: number;
  leverage: number;
}

export interface Candle {
  datetime: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface Trade {
  id: number;
  strategy: string;
  symbol: string;
  direction: "buy" | "sell";
  entry_price: number;
  exit_price: number | null;
  sl: number;
  tp: number;
  lot_size: number;
  pnl: number | null;
  opened_at: string;
  closed_at: string | null;
  status: "open" | "closed";
}

export interface BacktestMetrics {
  total_trades: number;
  win_rate: number;
  profit_factor: number;
  total_pnl: number;
  avg_win: number;
  avg_loss: number;
  max_drawdown: number;
  sharpe_ratio: number;
}

export interface EquityCurvePoint {
  time: string;
  value: number;
}

export interface BacktestResult {
  metrics: BacktestMetrics;
  equity_curve: EquityCurvePoint[];
  trades: Trade[];
}

export interface BotStatus {
  mode: string;
  symbol: string;
  timeframe: string;
  use_mock: boolean;
  strategies: string[];
}

// --- Learning system types ---

export interface StrategyWeight {
  name: string;
  weight: number;
  win_rate: number;
  trades: number;
}

export interface RLStats {
  episode: number;
  epsilon: number;
  total_reward: number;
  buffer_size: number;
  training: boolean;
  win_rate: number;
  torch_available: boolean;
}

export interface LearningStatus {
  enabled: boolean;
  regime: string;
  regime_confidence: number;
  session: string;
  timeframe: string;
  context_timeframes: string[];
  confidence_threshold: number;
  weights: StrategyWeight[];
  rl_stats: RLStats;
}

export interface StrategyPerformance {
  name: string;
  trades: number;
  wins: number;
  losses: number;
  total_pnl: number;
  win_rate: number;
  profit_factor: number;
}

export interface RegimeEntry {
  regime: string;
  confidence: number;
  adx: number;
  atr_ratio: number;
  timestamp: string | null;
}

export interface ClaudeInsight {
  review_type: string;
  timestamp: string | null;
  analysis: string;
  recommendations: string[];
  market_brief: string;
  score: number;
}

// --- RL Backtest types ---

export interface RLBacktestRequest {
  timeframe: string;
  count: number;
  epochs: number;
  train_every: number;
}

export interface RLBacktestTrade {
  epoch: number;
  action: string;
  entry_price: number;
  exit_price: number;
  pnl: number;
  reward: number;
  hold_bars: number;
  bar_index: number;
}

export interface RLBacktestEpochStats {
  epoch: number;
  trades: number;
  wins: number;
  win_rate: number;
  total_reward: number;
  avg_train_loss: number;
  epsilon: number;
}

export interface RLBacktestResult {
  total_trades: number;
  win_rate: number;
  total_pnl: number;
  total_reward: number;
  avg_reward_per_trade: number;
  max_drawdown: number;
  final_epsilon: number;
  episodes: number;
  epochs_completed: number;
  equity_curve: { trade_index: number; cumulative_pnl: number }[];
  trades: RLBacktestTrade[];
  epoch_stats: RLBacktestEpochStats[];
  claude_review: ClaudeInsight | null;
}

// --- API functions ---

async function fetchJSON<T>(url: string, options?: RequestInit): Promise<T> {
  const res = await fetch(url, options);
  if (!res.ok) {
    throw new Error(`API error: ${res.status} ${res.statusText}`);
  }
  return res.json();
}

export function getAccount(): Promise<AccountInfo> {
  return fetchJSON("/api/account");
}

export function getCandles(
  count = 200,
  timeframe = "H1"
): Promise<Candle[]> {
  return fetchJSON(`/api/candles?count=${count}&timeframe=${timeframe}`);
}

export function getClosedTrades(): Promise<Trade[]> {
  return fetchJSON("/api/trades?status=closed");
}

export function getOpenTrades(): Promise<Trade[]> {
  return fetchJSON("/api/trades/open");
}

export function runBacktest(
  strategy: "ema" | "breakout",
  count = 500
): Promise<BacktestResult> {
  return fetchJSON("/api/backtest", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ strategy, count }),
  });
}

export function getBotStatus(): Promise<BotStatus> {
  return fetchJSON("/api/bot/status");
}

// --- Learning API functions ---

export function getLearningStatus(): Promise<LearningStatus> {
  return fetchJSON("/api/learning/status");
}

export function getLearningPerformance(): Promise<StrategyPerformance[]> {
  return fetchJSON("/api/learning/performance");
}

export function getRegimeHistory(): Promise<RegimeEntry[]> {
  return fetchJSON("/api/learning/regime-history");
}

export function getClaudeInsights(): Promise<ClaudeInsight[]> {
  return fetchJSON("/api/learning/insights");
}

export function getRLStats(): Promise<RLStats> {
  return fetchJSON("/api/learning/rl-stats");
}

export function triggerReview(): Promise<{ status: string }> {
  return fetchJSON("/api/learning/review", { method: "POST" });
}

export function triggerOptimization(): Promise<{ status: string }> {
  return fetchJSON("/api/learning/optimize", { method: "POST" });
}

export function toggleLearning(enabled: boolean): Promise<{ status: string }> {
  return fetchJSON("/api/learning/toggle", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ enabled }),
  });
}

export function runRLBacktest(
  params: RLBacktestRequest
): Promise<RLBacktestResult> {
  return fetchJSON("/api/learning/backtest", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params),
  });
}

// --- Logs ---

export function getLogs(lines = 200): Promise<string[]> {
  return fetchJSON(`/api/logs?lines=${lines}`);
}

export function clearLogs(): Promise<{ status: string }> {
  return fetchJSON("/api/logs", { method: "DELETE" });
}
