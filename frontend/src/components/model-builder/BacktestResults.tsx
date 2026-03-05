import type { RLBacktestResult } from "../../api";

interface BacktestResultsProps {
  result: RLBacktestResult | null;
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

function ExitReasonBreakdown({ result }: { result: RLBacktestResult }) {
  const counts: Record<string, number> = {};
  for (const t of result.trades) {
    counts[t.exit_reason] = (counts[t.exit_reason] || 0) + 1;
  }
  const total = result.trades.length || 1;

  const reasonColors: Record<string, string> = {
    sl: "bg-red-500/60",
    tp: "bg-green-500/60",
    trailing: "bg-yellow-500/60",
    signal: "bg-blue-500/60",
    max_hold: "bg-gray-500/60",
  };

  const reasonLabels: Record<string, string> = {
    sl: "Stop Loss",
    tp: "Take Profit",
    trailing: "Trailing Stop",
    signal: "Signal",
    max_hold: "Max Hold",
  };

  return (
    <div className="mb-4">
      <h4 className="mb-2 text-xs font-medium text-gray-400">Exit Reasons</h4>
      <div className="mb-2 flex h-4 overflow-hidden rounded-full">
        {Object.entries(counts).map(([reason, count]) => (
          <div
            key={reason}
            className={`${reasonColors[reason] || "bg-gray-500/60"}`}
            style={{ width: `${(count / total) * 100}%` }}
            title={`${reasonLabels[reason] || reason}: ${count} (${((count / total) * 100).toFixed(0)}%)`}
          />
        ))}
      </div>
      <div className="flex flex-wrap gap-3">
        {Object.entries(counts).map(([reason, count]) => (
          <div key={reason} className="flex items-center gap-1.5">
            <div
              className={`h-2.5 w-2.5 rounded-sm ${reasonColors[reason] || "bg-gray-500/60"}`}
            />
            <span className="text-[10px] text-gray-400">
              {reasonLabels[reason] || reason}: {count} ({((count / total) * 100).toFixed(0)}%)
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

export default function BacktestResults({ result }: BacktestResultsProps) {
  if (!result) return null;

  const MAX_BARS = 200;
  const curve = result.equity_curve;
  const sampledCurve =
    curve.length <= MAX_BARS
      ? curve
      : Array.from({ length: MAX_BARS }, (_, i) => {
          const idx = Math.round((i / (MAX_BARS - 1)) * (curve.length - 1));
          return curve[idx];
        });

  const maxPnl = Math.max(
    ...sampledCurve.map((p) => Math.abs(p.cumulative_pnl)),
    0.01
  );

  const profileBadge: Record<string, string> = {
    max_profit: "bg-blue-500/20 text-blue-400",
    aggressive: "bg-red-500/20 text-red-400",
    medium: "bg-yellow-500/20 text-yellow-400",
    conservative: "bg-green-500/20 text-green-400",
  };

  return (
    <div className="rounded-xl border border-blue-700/50 bg-gray-800 p-5">
      <div className="mb-3 flex items-center gap-3">
        <h3 className="text-sm font-semibold text-blue-400">
          Training Results
        </h3>
        <span
          className={`rounded-full px-2.5 py-0.5 text-[10px] font-medium uppercase tracking-wider ${
            profileBadge[result.profile] || "bg-gray-600/30 text-gray-400"
          }`}
        >
          {result.profile}
        </span>
      </div>

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

      {/* Exit Reason Breakdown */}
      {result.trades.length > 0 && <ExitReasonBreakdown result={result} />}

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
            {sampledCurve.map((pt, i) => {
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
