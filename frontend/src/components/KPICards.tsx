import type { AccountInfo, Trade } from "../api";

interface KPICardsProps {
  account: AccountInfo | null;
  closedTrades: Trade[];
}

function formatCurrency(value: number): string {
  return value.toLocaleString("en-US", {
    style: "currency",
    currency: "USD",
    minimumFractionDigits: 2,
  });
}

function formatPercent(value: number): string {
  return `${value.toFixed(1)}%`;
}

export default function KPICards({ account, closedTrades }: KPICardsProps) {
  const totalPnl = closedTrades.reduce((sum, t) => sum + (t.pnl ?? 0), 0);
  const wins = closedTrades.filter((t) => (t.pnl ?? 0) > 0).length;
  const winRate =
    closedTrades.length > 0 ? (wins / closedTrades.length) * 100 : 0;

  // Calculate max drawdown from cumulative P&L
  let peak = 0;
  let maxDrawdown = 0;
  let cumulative = 0;
  for (const trade of closedTrades) {
    cumulative += trade.pnl ?? 0;
    if (cumulative > peak) peak = cumulative;
    const dd = peak - cumulative;
    if (dd > maxDrawdown) maxDrawdown = dd;
  }

  const cards = [
    {
      label: "Balance",
      value: account ? formatCurrency(account.balance) : "--",
      color: "text-white",
    },
    {
      label: "Total P&L",
      value: formatCurrency(totalPnl),
      color: totalPnl >= 0 ? "text-emerald-400" : "text-red-400",
    },
    {
      label: "Max Drawdown",
      value: formatCurrency(maxDrawdown),
      color: "text-amber-400",
    },
    {
      label: "Win Rate",
      value: formatPercent(winRate),
      color: winRate >= 50 ? "text-emerald-400" : "text-red-400",
    },
    {
      label: "Trade Count",
      value: closedTrades.length.toString(),
      color: "text-blue-400",
    },
  ];

  return (
    <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 lg:grid-cols-5">
      {cards.map((card) => (
        <div
          key={card.label}
          className="rounded-xl border border-gray-700 bg-gray-800 p-5"
        >
          <p className="mb-1 text-xs font-medium tracking-wider text-gray-400 uppercase">
            {card.label}
          </p>
          <p className={`text-2xl font-bold ${card.color}`}>{card.value}</p>
        </div>
      ))}
    </div>
  );
}
