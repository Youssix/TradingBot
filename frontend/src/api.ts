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
  strategy_mode: string;
  enabled_strategies: string[];
}

export interface StrategyModeRequest {
  mode: "independent" | "ensemble";
  enabled_strategies: string[];
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
  profile: string;
  model_id?: number | null;
  symbols?: string[];
  augmentation_factor?: number;
  agent_type?: string;
}

export interface PipelineStep {
  timeframe: string;
  count: number;
  epochs: number;
  symbols: string[];
  augmentation_factor: number;
}

export interface PipelineRequest {
  steps?: PipelineStep[];
  profile: string;
  train_every: number;
  model_id?: number | null;
  preset?: string | null;
}

export interface PipelineStreamCallbacks {
  onStepStart?: (data: { step: number; total_steps: number; timeframe: string; symbols: string[]; epochs: number; augmentation_factor: number }) => void;
  onProgress?: (data: { step: number; epoch: number; bar: number; total_bars: number; pct: number }) => void;
  onEpoch?: (data: { step: number; epoch: number; trades: number; wins: number; win_rate: number; total_reward: number; avg_train_loss: number; epsilon: number; cumulative_pnl: number }) => void;
  onStepDone?: (data: { step: number; timeframe: string; trades: number; win_rate: number; epsilon: number }) => void;
  onPipelineDone?: (data: { steps_completed: number; total_episodes: number; final_epsilon: number; cancelled: boolean; final_win_rate?: number; final_total_reward?: number }) => void;
  onError?: (message: string) => void;
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
  exit_reason: string;
  sl_price: number;
  tp_price: number;
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
  profile: string;
  equity_curve: { trade_index: number; cumulative_pnl: number }[];
  trades: RLBacktestTrade[];
  epoch_stats: RLBacktestEpochStats[];
  claude_review: ClaudeInsight | null;
}

// --- RL Model Management types ---

export interface RLModelInfo {
  id: number;
  model_name: string;
  episode: number;
  epsilon: number;
  total_reward: number;
  win_rate: number;
  profile: string | null;
  timeframe: string | null;
  created_at: string | null;
}

export interface RLModelListResponse {
  models: RLModelInfo[];
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

export function setStrategyMode(
  params: StrategyModeRequest
): Promise<{ status: string; mode: string; enabled_strategies: string[] }> {
  return fetchJSON("/api/bot/strategy-mode", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params),
  });
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

// --- RL Backtest Streaming (SSE) ---

export interface BacktestStreamCallbacks {
  onProgress?: (data: { epoch: number; bar: number; total_bars: number; pct: number }) => void;
  onEpoch?: (data: RLBacktestEpochStats & { cumulative_pnl: number }) => void;
  onDone?: (data: RLBacktestResult) => void;
  onError?: (message: string) => void;
}

export function runRLBacktestStream(
  params: RLBacktestRequest,
  callbacks: BacktestStreamCallbacks,
): AbortController {
  const controller = new AbortController();

  (async () => {
    try {
      const res = await fetch("/api/learning/backtest/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(params),
        signal: controller.signal,
      });

      if (!res.ok) {
        callbacks.onError?.(`API error: ${res.status} ${res.statusText}`);
        return;
      }

      const reader = res.body!.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";

        let currentEvent = "";
        for (const line of lines) {
          if (line.startsWith("event: ")) {
            currentEvent = line.slice(7).trim();
          } else if (line.startsWith("data: ") && currentEvent) {
            const data = JSON.parse(line.slice(6));
            switch (currentEvent) {
              case "progress":
                callbacks.onProgress?.(data);
                break;
              case "epoch":
                callbacks.onEpoch?.(data);
                break;
              case "done":
                callbacks.onDone?.(data);
                break;
              case "error":
                callbacks.onError?.(data);
                break;
            }
            currentEvent = "";
          }
        }
      }
    } catch (err: unknown) {
      if (err instanceof Error && err.name !== "AbortError") {
        callbacks.onError?.(err.message);
      }
    }
  })();

  return controller;
}

// --- Pipeline Training (SSE) ---

export function runPipelineStream(
  params: PipelineRequest,
  callbacks: PipelineStreamCallbacks,
): AbortController {
  const controller = new AbortController();

  (async () => {
    try {
      const res = await fetch("/api/learning/train-pipeline/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(params),
        signal: controller.signal,
      });

      if (!res.ok) {
        callbacks.onError?.(`API error: ${res.status} ${res.statusText}`);
        return;
      }

      const reader = res.body!.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";

        let currentEvent = "";
        for (const line of lines) {
          if (line.startsWith("event: ")) {
            currentEvent = line.slice(7).trim();
          } else if (line.startsWith("data: ") && currentEvent) {
            const data = JSON.parse(line.slice(6));
            switch (currentEvent) {
              case "step_start":
                callbacks.onStepStart?.(data);
                break;
              case "progress":
                callbacks.onProgress?.(data);
                break;
              case "epoch":
                callbacks.onEpoch?.(data);
                break;
              case "step_done":
                callbacks.onStepDone?.(data);
                break;
              case "pipeline_done":
                callbacks.onPipelineDone?.(data);
                break;
              case "error":
                callbacks.onError?.(data);
                break;
            }
            currentEvent = "";
          }
        }
      }
    } catch (err: unknown) {
      if (err instanceof Error && err.name !== "AbortError") {
        callbacks.onError?.(err.message);
      }
    }
  })();

  return controller;
}

// --- Pipeline Status ---

export interface PipelineStatus {
  running: boolean;
  progress: {
    running: boolean;
    step: number;
    total_steps: number;
    step_timeframe: string;
    pct: number;
  } | null;
  logs: string[];
}

export function getPipelineStatus(): Promise<PipelineStatus> {
  return fetchJSON("/api/learning/pipeline-status");
}

export function cancelPipeline(): Promise<{ status: string }> {
  return fetchJSON("/api/learning/cancel-pipeline", { method: "POST" });
}

// --- RL Model Management ---

export function listRLModels(profile = ""): Promise<RLModelListResponse> {
  const q = profile ? `?profile=${encodeURIComponent(profile)}` : "";
  return fetchJSON(`/api/learning/models${q}`);
}

export function deleteRLModel(modelId: number): Promise<{ status: string }> {
  return fetchJSON(`/api/learning/models/${modelId}`, { method: "DELETE" });
}

export function activateRLModel(modelId: number): Promise<{ status: string; model_id: number; profile: string; episode: number }> {
  return fetchJSON("/api/learning/models/activate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model_id: modelId }),
  });
}

// --- Logs ---

export function getLogs(lines = 200): Promise<string[]> {
  return fetchJSON(`/api/logs?lines=${lines}`);
}

export function clearLogs(): Promise<{ status: string }> {
  return fetchJSON("/api/logs", { method: "DELETE" });
}

// --- Agent Config & Advanced RL types ---

export interface AgentConfigRequest {
  agent_type: string;
  hidden_dim?: number;
  actor_lr?: number;
  critic_lr?: number;
  alpha_lr?: number;
  gamma?: number;
  tau?: number;
  use_quantile?: boolean;
  n_quantiles?: number;
  risk_sensitivity?: number;
  lr?: number;
  clip_epsilon?: number;
  gae_lambda?: number;
  noise_sigma?: number;
  agents?: string[];
  strategy?: string;
  transformer_enabled?: boolean;
  use_per?: boolean;
}

export interface AgentConfigResponse {
  agent_type: string;
  config: Record<string, unknown>;
  status: string;
}

export interface PerAgentStats {
  name: string;
  sharpe: number;
  cumulative_reward: number;
  max_drawdown: number;
  win_rate: number;
  trade_count: number;
  weight: number;
  is_active: boolean;
}

export interface EnsembleStatsResponse {
  strategy: string;
  agents: PerAgentStats[];
  active_agent: string;
}

export interface RiskMetricsResponse {
  cvar_5: number;
  var_5: number;
  q_mean: number;
  q_std: number;
  upside: number;
  available: boolean;
}

export interface TrainingMetricsResponse {
  agent_type: string;
  episode: number;
  epsilon: number;
  total_reward: number;
  buffer_size: number;
  buffer_capacity: number;
  buffer_fill_pct: number;
  training: boolean;
  win_rate: number;
  torch_available: boolean;
  alpha: number | null;
  use_quantile: boolean;
  use_per: boolean;
  mean_priority: number | null;
  readiness: "untrained" | "learning" | "ready";
  active_model_id: number | null;
  active_model_name: string | null;
  active_model_profile: string | null;
  active_model_timeframe: string | null;
}

// --- Agent Config & Advanced RL API functions ---

export async function getAgentConfig(): Promise<AgentConfigResponse> {
  return fetchJSON<AgentConfigResponse>("/api/learning/agent-config");
}

export async function setAgentConfig(config: AgentConfigRequest): Promise<AgentConfigResponse> {
  return fetchJSON<AgentConfigResponse>("/api/learning/agent-config", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(config),
  });
}

export async function getEnsembleStats(): Promise<EnsembleStatsResponse> {
  return fetchJSON<EnsembleStatsResponse>("/api/learning/ensemble-stats");
}

export async function getRiskMetrics(): Promise<RiskMetricsResponse> {
  return fetchJSON<RiskMetricsResponse>("/api/learning/risk-metrics");
}

export async function getTrainingMetrics(): Promise<TrainingMetricsResponse> {
  return fetchJSON<TrainingMetricsResponse>("/api/learning/training-metrics");
}

// --- Hyperparameter Metadata ---

export interface HyperparamMeta {
  key: string;
  label: string;
  tooltip: string;
  type: "slider" | "toggle" | "select";
  min?: number;
  max?: number;
  step?: number;
  options?: { value: string; label: string }[];
  default: number | boolean | string;
  category: "learning" | "architecture" | "exploration" | "risk";
  agents: string[];
}

export const HYPERPARAMS: HyperparamMeta[] = [
  // Architecture
  { key: "hidden_dim", label: "Hidden Size", tooltip: "Brain size — how many neurons the agent uses to think. Bigger = smarter but slower to train.", type: "slider", min: 64, max: 512, step: 64, default: 256, category: "architecture", agents: ["sac", "ppo", "ddpg"] },

  // Learning
  { key: "actor_lr", label: "Actor Learning Rate", tooltip: "How fast the decision-maker learns. Too high = unstable, too low = slow progress.", type: "slider", min: 0.00001, max: 0.01, step: 0.00001, default: 0.0003, category: "learning", agents: ["sac", "ddpg"] },
  { key: "critic_lr", label: "Critic Learning Rate", tooltip: "How fast the evaluator learns. The critic judges whether decisions are good or bad.", type: "slider", min: 0.00001, max: 0.01, step: 0.00001, default: 0.0003, category: "learning", agents: ["sac", "ddpg"] },
  { key: "alpha_lr", label: "Alpha Learning Rate", tooltip: "How fast the exploration balance adjusts. SAC automatically tunes how much to explore.", type: "slider", min: 0.00001, max: 0.01, step: 0.00001, default: 0.0003, category: "learning", agents: ["sac"] },
  { key: "lr", label: "Learning Rate", tooltip: "How fast the agent learns from experience. Like adjusting how big each study step is.", type: "slider", min: 0.00001, max: 0.01, step: 0.00001, default: 0.0003, category: "learning", agents: ["ppo"] },
  { key: "gamma", label: "Discount Factor", tooltip: "How far ahead the agent plans. Higher = thinks long-term, lower = focuses on quick wins.", type: "slider", min: 0.9, max: 0.999, step: 0.001, default: 0.99, category: "learning", agents: ["sac", "ppo", "ddpg"] },
  { key: "tau", label: "Soft Update Rate", tooltip: "How gradually the agent updates its knowledge. Lower = more stable but slower adaptation.", type: "slider", min: 0.001, max: 0.05, step: 0.001, default: 0.005, category: "learning", agents: ["sac", "ddpg"] },
  { key: "clip_epsilon", label: "Clip Range", tooltip: "Limits how much the agent can change per update. Prevents wild swings in behavior.", type: "slider", min: 0.05, max: 0.5, step: 0.05, default: 0.2, category: "learning", agents: ["ppo"] },
  { key: "gae_lambda", label: "GAE Lambda", tooltip: "Balances bias vs variance in advantage estimation. Higher = less bias, more variance.", type: "slider", min: 0.8, max: 1.0, step: 0.01, default: 0.95, category: "learning", agents: ["ppo"] },

  // Exploration
  { key: "noise_sigma", label: "Noise Level", tooltip: "How much randomness in exploration. Higher = tries more diverse trades, lower = sticks to what works.", type: "slider", min: 0.05, max: 0.5, step: 0.05, default: 0.2, category: "exploration", agents: ["ddpg"] },
  { key: "use_per", label: "Prioritized Replay", tooltip: "Learn more from surprising experiences. The agent replays important trades more often.", type: "toggle", default: true, category: "exploration", agents: ["sac", "ddpg"] },
  { key: "transformer_enabled", label: "Transformer Encoder", tooltip: "Adds a powerful pattern-recognition layer. Helps the agent spot complex market sequences.", type: "toggle", default: false, category: "architecture", agents: ["sac", "ppo", "ddpg"] },

  // Risk
  { key: "use_quantile", label: "Quantile Critics", tooltip: "Enables uncertainty-aware decisions. The agent estimates a range of outcomes instead of just one.", type: "toggle", default: false, category: "risk", agents: ["sac"] },
  { key: "n_quantiles", label: "Number of Quantiles", tooltip: "How many outcome scenarios to consider. More = finer-grained risk picture.", type: "slider", min: 8, max: 64, step: 8, default: 32, category: "risk", agents: ["sac"] },
  { key: "risk_sensitivity", label: "Risk Sensitivity", tooltip: "Negative = cautious (avoids losses), zero = neutral, positive = aggressive (chases gains).", type: "slider", min: -1.0, max: 1.0, step: 0.1, default: 0.0, category: "risk", agents: ["sac"] },

  // Ensemble
  { key: "strategy", label: "Ensemble Strategy", tooltip: "How the team of agents decides together. Weighted average blends all opinions, best Sharpe picks the top performer.", type: "select", options: [{ value: "weighted_average", label: "Weighted Average" }, { value: "best_sharpe", label: "Best Sharpe" }, { value: "majority_vote", label: "Majority Vote" }], default: "weighted_average", category: "learning", agents: ["ensemble"] },
];
