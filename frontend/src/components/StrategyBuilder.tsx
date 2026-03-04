import { useState, useEffect, useCallback } from "react";
import {
  getLearningStatus,
  getClaudeInsights,
  triggerReview,
  triggerOptimization,
  toggleLearning,
  runRLBacktest,
  type LearningStatus,
  type ClaudeInsight,
  type RLBacktestResult,
} from "../api";

// --- Sub-components ---

function RLStatusCard({ status }: { status: LearningStatus }) {
  const rl = status.rl_stats;
  return (
    <div className="rounded-xl border border-gray-700 bg-gray-800 p-5">
      <div className="mb-3 flex items-center justify-between">
        <h3 className="text-sm font-semibold text-gray-300">RL Agent Status</h3>
        <span
          className={`rounded-full px-2.5 py-0.5 text-xs font-medium ${
            rl.training
              ? "bg-green-500/20 text-green-400"
              : "bg-gray-600/30 text-gray-400"
          }`}
        >
          {rl.training ? "Training" : "Idle"}
        </span>
      </div>
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        <Stat label="Episode" value={rl.episode.toLocaleString()} />
        <Stat label="Epsilon" value={rl.epsilon.toFixed(4)} />
        <Stat label="Total Reward" value={rl.total_reward.toFixed(1)} highlight />
        <Stat
          label="Win Rate"
          value={`${(rl.win_rate * 100).toFixed(1)}%`}
          highlight
        />
      </div>
      <div className="mt-3 grid grid-cols-2 gap-3">
        <Stat label="Buffer Size" value={rl.buffer_size.toLocaleString()} />
        <Stat
          label="PyTorch"
          value={rl.torch_available ? "Available" : "Missing"}
        />
      </div>
    </div>
  );
}

function StrategyWeights({ status }: { status: LearningStatus }) {
  return (
    <div className="rounded-xl border border-gray-700 bg-gray-800 p-5">
      <h3 className="mb-3 text-sm font-semibold text-gray-300">
        Strategy Weights
      </h3>
      <div className="space-y-2.5">
        {status.weights.map((w) => (
          <div key={w.name} className="flex items-center gap-3">
            <span className="w-28 truncate text-xs text-gray-400">
              {w.name}
            </span>
            <div className="relative flex-1 h-4 rounded-full bg-gray-700">
              <div
                className="h-4 rounded-full bg-blue-500 transition-all"
                style={{ width: `${Math.round(w.weight * 100)}%` }}
              />
              <span className="absolute inset-0 flex items-center justify-center text-[10px] font-bold text-white">
                {Math.round(w.weight * 100)}%
              </span>
            </div>
            <span className="w-20 text-right text-xs text-gray-500">
              WR: {(w.win_rate * 100).toFixed(0)}% ({w.trades})
            </span>
          </div>
        ))}
        {status.weights.length === 0 && (
          <p className="text-xs text-gray-500">No strategies active</p>
        )}
      </div>
    </div>
  );
}

function MarketContext({ status }: { status: LearningStatus }) {
  const regimeBadge: Record<string, string> = {
    trending: "bg-green-500/20 text-green-400",
    ranging: "bg-yellow-500/20 text-yellow-400",
    volatile: "bg-red-500/20 text-red-400",
    unknown: "bg-gray-600/30 text-gray-400",
  };

  return (
    <div className="rounded-xl border border-gray-700 bg-gray-800 p-5">
      <h3 className="mb-3 text-sm font-semibold text-gray-300">
        Market Context
      </h3>
      <div className="flex flex-wrap items-center gap-3">
        <div className="flex items-center gap-2">
          <span className="text-xs text-gray-500">Timeframe:</span>
          <span className="rounded bg-gray-700 px-2 py-0.5 text-xs font-medium text-white">
            {status.timeframe}
          </span>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-xs text-gray-500">Context:</span>
          <span className="text-xs text-gray-300">
            {status.context_timeframes.join(", ") || "—"}
          </span>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-xs text-gray-500">Regime:</span>
          <span
            className={`rounded-full px-2.5 py-0.5 text-xs font-medium ${
              regimeBadge[status.regime] || regimeBadge.unknown
            }`}
          >
            {status.regime.toUpperCase()}
          </span>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-xs text-gray-500">Session:</span>
          <span className="text-xs capitalize text-gray-300">
            {status.session}
          </span>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-xs text-gray-500">Conf. Threshold:</span>
          <span className="text-xs font-medium text-white">
            {status.confidence_threshold.toFixed(2)}
          </span>
        </div>
      </div>
    </div>
  );
}

function ClaudeReviewLog({ insights }: { insights: ClaudeInsight[] }) {
  return (
    <div className="rounded-xl border border-gray-700 bg-gray-800 p-5">
      <h3 className="mb-3 text-sm font-semibold text-gray-300">
        Claude Review Log
      </h3>
      <div className="max-h-64 space-y-2 overflow-y-auto pr-1">
        {insights.map((ins, i) => (
          <div
            key={i}
            className="rounded-lg border border-gray-700/50 bg-gray-900/50 p-3"
          >
            <div className="mb-1 flex items-center gap-2">
              <span
                className={`rounded px-1.5 py-0.5 text-[10px] font-medium ${
                  ins.review_type === "market_brief"
                    ? "bg-blue-500/20 text-blue-400"
                    : "bg-purple-500/20 text-purple-400"
                }`}
              >
                {ins.review_type}
              </span>
              <span className="text-[10px] text-gray-600">
                {ins.timestamp
                  ? new Date(ins.timestamp).toLocaleString()
                  : "—"}
              </span>
            </div>
            {ins.analysis && (
              <p className="text-xs text-gray-300">{ins.analysis}</p>
            )}
            {ins.market_brief && (
              <p className="mt-1 text-xs italic text-gray-400">
                {ins.market_brief}
              </p>
            )}
            {ins.recommendations.length > 0 && (
              <ul className="mt-1 space-y-0.5">
                {ins.recommendations.map((r, j) => (
                  <li key={j} className="text-[11px] text-gray-500">
                    &bull; {r}
                  </li>
                ))}
              </ul>
            )}
          </div>
        ))}
        {insights.length === 0 && (
          <p className="text-xs text-gray-500">No insights yet</p>
        )}
      </div>
    </div>
  );
}

function RLTrainingPanel({
  onResult,
  loading,
  setLoading,
}: {
  onResult: (r: RLBacktestResult) => void;
  loading: string;
  setLoading: (s: string) => void;
}) {
  const [timeframe, setTimeframe] = useState("H1");
  const [count, setCount] = useState(2000);
  const [epochs, setEpochs] = useState(3);

  const handleTrain = async () => {
    setLoading("backtest");
    try {
      const result = await runRLBacktest({
        timeframe,
        count,
        epochs,
        train_every: 5,
      });
      onResult(result);
    } catch (e) {
      console.error("RL backtest failed:", e);
    } finally {
      setLoading("");
    }
  };

  return (
    <div className="rounded-xl border border-gray-700 bg-gray-800 p-5">
      <h3 className="mb-3 text-sm font-semibold text-gray-300">
        Train on Historical Data
      </h3>
      <div className="flex flex-wrap items-end gap-3">
        <div>
          <label className="mb-1 block text-[10px] uppercase tracking-wider text-gray-500">
            Timeframe
          </label>
          <select
            value={timeframe}
            onChange={(e) => setTimeframe(e.target.value)}
            className="rounded-lg border border-gray-600 bg-gray-700 px-3 py-1.5 text-xs text-white"
          >
            <option value="M5">M5 (60 days)</option>
            <option value="M15">M15 (60 days)</option>
            <option value="H1">H1 (2 years)</option>
            <option value="D1">D1 (all)</option>
          </select>
        </div>
        <div>
          <label className="mb-1 block text-[10px] uppercase tracking-wider text-gray-500">
            Candles
          </label>
          <input
            type="number"
            value={count}
            onChange={(e) => setCount(Number(e.target.value))}
            min={100}
            max={10000}
            className="w-24 rounded-lg border border-gray-600 bg-gray-700 px-3 py-1.5 text-xs text-white"
          />
        </div>
        <div>
          <label className="mb-1 block text-[10px] uppercase tracking-wider text-gray-500">
            Epochs
          </label>
          <input
            type="number"
            value={epochs}
            onChange={(e) => setEpochs(Number(e.target.value))}
            min={1}
            max={20}
            className="w-16 rounded-lg border border-gray-600 bg-gray-700 px-3 py-1.5 text-xs text-white"
          />
        </div>
        <button
          onClick={handleTrain}
          disabled={loading === "backtest"}
          className="rounded-lg border border-blue-600 bg-blue-700 px-4 py-1.5 text-xs font-medium text-white transition hover:bg-blue-600 disabled:opacity-50"
        >
          {loading === "backtest" ? "Training..." : "Train on Historical Data"}
        </button>
      </div>
    </div>
  );
}

function RLBacktestResults({ result }: { result: RLBacktestResult }) {
  const maxPnl = Math.max(
    ...result.equity_curve.map((p) => Math.abs(p.cumulative_pnl)),
    0.01
  );

  return (
    <div className="rounded-xl border border-blue-700/50 bg-gray-800 p-5">
      <h3 className="mb-3 text-sm font-semibold text-blue-400">
        Training Results
      </h3>

      {/* Metrics */}
      <div className="mb-4 grid grid-cols-2 gap-3 sm:grid-cols-4">
        <Stat label="Total Trades" value={result.total_trades.toLocaleString()} />
        <Stat
          label="Win Rate"
          value={`${(result.win_rate * 100).toFixed(1)}%`}
          highlight
        />
        <Stat label="Total PnL" value={result.total_pnl.toFixed(2)} highlight />
        <Stat label="Max Drawdown" value={result.max_drawdown.toFixed(2)} />
        <Stat label="Total Reward" value={result.total_reward.toFixed(1)} />
        <Stat
          label="Avg Reward/Trade"
          value={result.avg_reward_per_trade.toFixed(3)}
        />
        <Stat label="Final Epsilon" value={result.final_epsilon.toFixed(4)} />
        <Stat label="Episodes" value={result.episodes.toLocaleString()} />
      </div>

      {/* Epoch Stats */}
      {result.epoch_stats.length > 0 && (
        <div className="mb-4">
          <h4 className="mb-2 text-xs font-medium text-gray-400">
            Epoch Progress
          </h4>
          <div className="space-y-1">
            {result.epoch_stats.map((es) => (
              <div
                key={es.epoch}
                className="flex items-center gap-3 text-[11px] text-gray-400"
              >
                <span className="w-16">Epoch {es.epoch}</span>
                <span className="w-20">{es.trades} trades</span>
                <span className="w-16">
                  WR: {(es.win_rate * 100).toFixed(0)}%
                </span>
                <span className="w-24">reward: {es.total_reward.toFixed(1)}</span>
                <span className="w-20">eps: {es.epsilon.toFixed(4)}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Equity Curve (simple bar chart) */}
      {result.equity_curve.length > 0 && (
        <div className="mb-4">
          <h4 className="mb-2 text-xs font-medium text-gray-400">
            Equity Curve
          </h4>
          <div className="flex h-24 items-end gap-px">
            {result.equity_curve.map((pt, i) => {
              const h = Math.abs(pt.cumulative_pnl) / maxPnl;
              const positive = pt.cumulative_pnl >= 0;
              return (
                <div
                  key={i}
                  className={`flex-1 rounded-t ${
                    positive ? "bg-green-500/60" : "bg-red-500/60"
                  }`}
                  style={{ height: `${Math.max(h * 100, 2)}%` }}
                  title={`Trade ${pt.trade_index}: ${pt.cumulative_pnl.toFixed(2)}`}
                />
              );
            })}
          </div>
        </div>
      )}

      {/* Claude Review */}
      {result.claude_review && (
        <div className="rounded-lg border border-purple-700/50 bg-gray-900/50 p-3">
          <h4 className="mb-1 text-xs font-medium text-purple-400">
            Claude Analysis
          </h4>
          {result.claude_review.analysis && (
            <p className="text-xs text-gray-300">
              {result.claude_review.analysis}
            </p>
          )}
          {result.claude_review.recommendations.length > 0 && (
            <ul className="mt-1 space-y-0.5">
              {result.claude_review.recommendations.map((r, j) => (
                <li key={j} className="text-[11px] text-gray-500">
                  &bull; {r}
                </li>
              ))}
            </ul>
          )}
        </div>
      )}
    </div>
  );
}

function Stat({
  label,
  value,
  highlight,
}: {
  label: string;
  value: string;
  highlight?: boolean;
}) {
  return (
    <div>
      <p className="text-[10px] uppercase tracking-wider text-gray-500">
        {label}
      </p>
      <p
        className={`text-sm font-semibold ${
          highlight ? "text-white" : "text-gray-300"
        }`}
      >
        {value}
      </p>
    </div>
  );
}

// --- Main component ---

export default function StrategyBuilder() {
  const [status, setStatus] = useState<LearningStatus | null>(null);
  const [insights, setInsights] = useState<ClaudeInsight[]>([]);
  const [loading, setLoading] = useState("");
  const [backtestResult, setBacktestResult] =
    useState<RLBacktestResult | null>(null);

  const fetchData = useCallback(async () => {
    const results = await Promise.allSettled([
      getLearningStatus(),
      getClaudeInsights(),
    ]);
    if (results[0].status === "fulfilled") setStatus(results[0].value);
    if (results[1].status === "fulfilled") setInsights(results[1].value);
  }, []);

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 10_000);
    return () => clearInterval(interval);
  }, [fetchData]);

  const handleReview = async () => {
    setLoading("review");
    try {
      await triggerReview();
      await fetchData();
    } finally {
      setLoading("");
    }
  };

  const handleOptimize = async () => {
    setLoading("optimize");
    try {
      await triggerOptimization();
      await fetchData();
    } finally {
      setLoading("");
    }
  };

  const handleToggle = async () => {
    if (!status) return;
    setLoading("toggle");
    try {
      await toggleLearning(!status.enabled);
      await fetchData();
    } finally {
      setLoading("");
    }
  };

  if (!status) {
    return (
      <div className="flex items-center justify-center py-20">
        <p className="text-sm text-gray-500">Loading learning system...</p>
      </div>
    );
  }

  return (
    <div className="space-y-5">
      {/* Enabled banner */}
      {!status.enabled && (
        <div className="rounded-lg border border-yellow-700/50 bg-yellow-900/20 px-4 py-2 text-xs text-yellow-400">
          Learning system is disabled. Click "Toggle Learning" to enable.
        </div>
      )}

      {/* RL Status */}
      <RLStatusCard status={status} />

      {/* Train on History */}
      <RLTrainingPanel
        onResult={setBacktestResult}
        loading={loading}
        setLoading={setLoading}
      />

      {/* Backtest Results */}
      {backtestResult && <RLBacktestResults result={backtestResult} />}

      {/* Weights + Context row */}
      <div className="grid gap-5 lg:grid-cols-2">
        <StrategyWeights status={status} />
        <MarketContext status={status} />
      </div>

      {/* Claude Review Log */}
      <ClaudeReviewLog insights={insights} />

      {/* Controls */}
      <div className="flex flex-wrap gap-3">
        <button
          onClick={handleReview}
          disabled={loading === "review"}
          className="rounded-lg border border-gray-600 bg-gray-700 px-4 py-2 text-xs font-medium text-gray-200 transition hover:bg-gray-600 disabled:opacity-50"
        >
          {loading === "review" ? "Running..." : "Run Review Now"}
        </button>
        <button
          onClick={handleOptimize}
          disabled={loading === "optimize"}
          className="rounded-lg border border-gray-600 bg-gray-700 px-4 py-2 text-xs font-medium text-gray-200 transition hover:bg-gray-600 disabled:opacity-50"
        >
          {loading === "optimize" ? "Optimizing..." : "Trigger Optimization"}
        </button>
        <button
          onClick={handleToggle}
          disabled={loading === "toggle"}
          className={`rounded-lg border px-4 py-2 text-xs font-medium transition disabled:opacity-50 ${
            status.enabled
              ? "border-red-700/50 bg-red-900/30 text-red-400 hover:bg-red-900/50"
              : "border-green-700/50 bg-green-900/30 text-green-400 hover:bg-green-900/50"
          }`}
        >
          {loading === "toggle"
            ? "..."
            : status.enabled
            ? "Disable Learning"
            : "Enable Learning"}
        </button>
      </div>
    </div>
  );
}
