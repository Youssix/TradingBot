import type { AccountInfo, Trade, TrainingMetricsResponse, LearningStatus } from "../api";

interface KPICardsProps {
  account: AccountInfo | null;
  closedTrades: Trade[];
  trainingMetrics?: TrainingMetricsResponse | null;
  learningStatus?: LearningStatus | null;
}

function formatCurrency(value: number): string {
  return value.toLocaleString("en-US", {
    style: "currency",
    currency: "USD",
    minimumFractionDigits: 2,
  });
}

function computeSharpe(trades: Trade[]): number {
  const pnls = trades.map((t) => t.pnl ?? 0);
  if (pnls.length < 2) return 0;
  const mean = pnls.reduce((s, v) => s + v, 0) / pnls.length;
  const variance = pnls.reduce((s, v) => s + (v - mean) ** 2, 0) / (pnls.length - 1);
  const std = Math.sqrt(variance);
  return std === 0 ? 0 : mean / std;
}

export default function KPICards({
  account,
  closedTrades,
  trainingMetrics,
  learningStatus,
}: KPICardsProps) {
  const totalPnl = closedTrades.reduce((sum, t) => sum + (t.pnl ?? 0), 0);
  const wins = closedTrades.filter((t) => (t.pnl ?? 0) > 0).length;
  const winRate = closedTrades.length > 0 ? (wins / closedTrades.length) * 100 : 0;

  let peak = 0;
  let maxDrawdown = 0;
  let cumulative = 0;
  for (const trade of closedTrades) {
    cumulative += trade.pnl ?? 0;
    if (cumulative > peak) peak = cumulative;
    const dd = peak - cumulative;
    if (dd > maxDrawdown) maxDrawdown = dd;
  }

  const sharpe = computeSharpe(closedTrades);
  const balance = account?.balance ?? 0;
  const dailyReturn = balance > 0 ? (totalPnl / balance) * 100 : 0;
  const confidence = learningStatus ? learningStatus.regime_confidence * 100 : null;

  return (
    <div className="space-y-4">
      {/* Hero: Balance + P&L */}
      <div className="grid grid-cols-2 gap-4">
        <div className="rounded-2xl p-5 card-glow" style={{ background: "var(--bg-card)", border: "1px solid var(--border-subtle)" }}>
          <p className="text-xs font-medium uppercase tracking-widest" style={{ color: "var(--text-muted)" }}>
            Balance
          </p>
          <p className="mt-1 font-num text-3xl font-bold" style={{ color: "var(--text-primary)" }}>
            {account ? formatCurrency(account.balance) : "--"}
          </p>
        </div>
        <div className="rounded-2xl p-5 card-glow" style={{ background: "var(--bg-card)", border: "1px solid var(--border-subtle)" }}>
          <p className="text-xs font-medium uppercase tracking-widest" style={{ color: "var(--text-muted)" }}>
            Today's P&L
          </p>
          <div className="mt-1 flex items-baseline gap-3">
            <span
              className="font-num text-3xl font-bold"
              style={{ color: totalPnl >= 0 ? "var(--accent-teal)" : "var(--accent-rose)" }}
            >
              {totalPnl >= 0 ? "+" : ""}{formatCurrency(totalPnl)}
            </span>
            <span
              className="font-num text-sm font-semibold"
              style={{ color: dailyReturn >= 0 ? "var(--accent-teal)" : "var(--accent-rose)", opacity: 0.7 }}
            >
              {dailyReturn >= 0 ? "+" : ""}{dailyReturn.toFixed(2)}%
            </span>
          </div>
        </div>
      </div>

      {/* Secondary metrics row */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <MetricCard
          label="Win Rate"
          value={`${winRate.toFixed(1)}%`}
          color={winRate >= 50 ? "var(--accent-teal)" : "var(--accent-rose)"}
          sub={`${wins}W / ${closedTrades.length - wins}L`}
        />
        <MetricCard
          label="Max Drawdown"
          value={formatCurrency(maxDrawdown)}
          color="var(--accent-amber)"
        />
        <MetricCard
          label="Sharpe Ratio"
          value={sharpe.toFixed(2)}
          color={sharpe > 1 ? "var(--accent-teal)" : sharpe >= 0 ? "var(--accent-amber)" : "var(--accent-rose)"}
        />
        <MetricCard
          label="Trades"
          value={closedTrades.length.toString()}
          color="var(--accent-blue)"
          sub={confidence !== null ? `${confidence.toFixed(0)}% confidence` : undefined}
        />
      </div>
    </div>
  );
}

function MetricCard({
  label,
  value,
  color,
  sub,
}: {
  label: string;
  value: string;
  color: string;
  sub?: string;
}) {
  return (
    <div
      className="rounded-xl px-4 py-3 card-glow"
      style={{ background: "var(--bg-card)", border: "1px solid var(--border-subtle)" }}
    >
      <p className="text-[10px] font-medium uppercase tracking-widest" style={{ color: "var(--text-muted)" }}>
        {label}
      </p>
      <p className="mt-0.5 font-num text-xl font-bold" style={{ color }}>
        {value}
      </p>
      {sub && (
        <p className="mt-0.5 text-[11px]" style={{ color: "var(--text-muted)" }}>
          {sub}
        </p>
      )}
    </div>
  );
}
