import { useState, useMemo } from "react";
import type { Trade } from "../api";

interface TradesTableProps {
  closedTrades: Trade[];
  openTrades: Trade[];
}

type SortKey = "opened_at" | "pnl" | "strategy" | "direction";
type SortDir = "asc" | "desc";

function formatPnl(val: number | null): string {
  if (val === null || val === undefined) return "--";
  return val >= 0 ? `+$${val.toFixed(2)}` : `-$${Math.abs(val).toFixed(2)}`;
}

function formatTime(val: string | null): string {
  if (!val) return "--";
  const d = new Date(val);
  return d.toLocaleString("en-US", {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

export default function TradesTable({ closedTrades, openTrades }: TradesTableProps) {
  const [tab, setTab] = useState<"closed" | "open">("closed");
  const [sortKey, setSortKey] = useState<SortKey>("opened_at");
  const [sortDir, setSortDir] = useState<SortDir>("desc");
  const [strategyFilter, setStrategyFilter] = useState<string>("All");

  const trades = tab === "closed" ? closedTrades : openTrades;

  const strategies = useMemo(() => {
    const set = new Set<string>();
    trades.forEach((t) => set.add(t.strategy));
    return Array.from(set).sort();
  }, [trades]);

  const filtered = useMemo(() => {
    return trades.filter((t) => {
      if (strategyFilter !== "All" && t.strategy !== strategyFilter) return false;
      return true;
    });
  }, [trades, strategyFilter]);

  const sorted = useMemo(() => {
    return [...filtered].sort((a, b) => {
      let aVal: any = a[sortKey];
      let bVal: any = b[sortKey];
      if (sortKey === "opened_at") {
        aVal = new Date(aVal || 0).getTime();
        bVal = new Date(bVal || 0).getTime();
      }
      if (sortKey === "pnl") {
        aVal = aVal ?? 0;
        bVal = bVal ?? 0;
      }
      if (aVal < bVal) return sortDir === "asc" ? -1 : 1;
      if (aVal > bVal) return sortDir === "asc" ? 1 : -1;
      return 0;
    });
  }, [filtered, sortKey, sortDir]);

  function handleSort(key: SortKey) {
    if (sortKey === key) {
      setSortDir((d) => (d === "asc" ? "desc" : "asc"));
    } else {
      setSortKey(key);
      setSortDir("desc");
    }
  }

  function sortArrow(key: SortKey) {
    if (sortKey !== key) return "";
    return sortDir === "asc" ? " \u2191" : " \u2193";
  }

  // Summary stats for current filter
  const totalPnl = filtered.reduce((s, t) => s + (t.pnl ?? 0), 0);
  const winCount = filtered.filter((t) => (t.pnl ?? 0) > 0).length;

  return (
    <div className="rounded-2xl overflow-hidden" style={{ background: "var(--bg-card)", border: "1px solid var(--border-subtle)" }}>
      {/* Header bar */}
      <div className="flex items-center justify-between px-5 py-3" style={{ borderBottom: "1px solid var(--border-subtle)" }}>
        <div className="flex items-center gap-4">
          <h3 className="text-sm font-semibold" style={{ color: "var(--text-primary)" }}>Trades</h3>
          <div className="flex rounded-md overflow-hidden" style={{ border: "1px solid var(--border-medium)" }}>
            <button
              onClick={() => setTab("closed")}
              className="px-3 py-1 text-[11px] font-medium transition-colors"
              style={{
                background: tab === "closed" ? "var(--bg-elevated)" : "transparent",
                color: tab === "closed" ? "var(--text-primary)" : "var(--text-muted)",
              }}
            >
              Closed ({closedTrades.length})
            </button>
            <button
              onClick={() => setTab("open")}
              className="px-3 py-1 text-[11px] font-medium transition-colors"
              style={{
                background: tab === "open" ? "var(--bg-elevated)" : "transparent",
                color: tab === "open" ? "var(--text-primary)" : "var(--text-muted)",
              }}
            >
              Open ({openTrades.length})
            </button>
          </div>
        </div>

        <div className="flex items-center gap-3">
          {/* Summary */}
          <span className="font-num text-[11px] font-semibold" style={{ color: totalPnl >= 0 ? "var(--accent-teal)" : "var(--accent-rose)" }}>
            {totalPnl >= 0 ? "+" : ""}{totalPnl.toFixed(2)}
          </span>
          <span className="text-[10px]" style={{ color: "var(--text-muted)" }}>
            {winCount}W / {filtered.length - winCount}L
          </span>
          <span className="h-3 w-px" style={{ background: "var(--border-medium)" }} />
          <select
            value={strategyFilter}
            onChange={(e) => setStrategyFilter(e.target.value)}
            className="rounded-md px-2 py-1 text-[11px] outline-none"
            style={{ background: "var(--bg-elevated)", color: "var(--text-secondary)", border: "1px solid var(--border-medium)" }}
          >
            <option value="All">All Strategies</option>
            {strategies.map((s) => (
              <option key={s} value={s}>{s}</option>
            ))}
          </select>
        </div>
      </div>

      {/* Table */}
      <div className="overflow-x-auto">
        <table className="w-full">
          <thead>
            <tr style={{ borderBottom: "1px solid var(--border-subtle)" }}>
              {[
                { key: "direction" as SortKey, label: "Side" },
                { key: "strategy" as SortKey, label: "Strategy" },
              ].map(({ key, label }) => (
                <th
                  key={key}
                  onClick={() => handleSort(key)}
                  className="cursor-pointer px-4 py-2.5 text-left text-[10px] font-semibold uppercase tracking-wider transition-colors hover:text-white"
                  style={{ color: "var(--text-muted)" }}
                >
                  {label}{sortArrow(key)}
                </th>
              ))}
              <th className="px-4 py-2.5 text-left text-[10px] font-semibold uppercase tracking-wider" style={{ color: "var(--text-muted)" }}>Symbol</th>
              <th className="px-4 py-2.5 text-right text-[10px] font-semibold uppercase tracking-wider" style={{ color: "var(--text-muted)" }}>Entry</th>
              <th className="px-4 py-2.5 text-right text-[10px] font-semibold uppercase tracking-wider" style={{ color: "var(--text-muted)" }}>Exit</th>
              <th
                onClick={() => handleSort("pnl")}
                className="cursor-pointer px-4 py-2.5 text-right text-[10px] font-semibold uppercase tracking-wider transition-colors hover:text-white"
                style={{ color: "var(--text-muted)" }}
              >
                P&L{sortArrow("pnl")}
              </th>
              <th
                onClick={() => handleSort("opened_at")}
                className="cursor-pointer px-4 py-2.5 text-right text-[10px] font-semibold uppercase tracking-wider transition-colors hover:text-white"
                style={{ color: "var(--text-muted)" }}
              >
                Time{sortArrow("opened_at")}
              </th>
            </tr>
          </thead>
          <tbody>
            {sorted.length === 0 ? (
              <tr>
                <td colSpan={7} className="px-4 py-10 text-center text-sm" style={{ color: "var(--text-muted)" }}>
                  No trades to display
                </td>
              </tr>
            ) : (
              sorted.map((t) => {
                const pnl = t.pnl ?? 0;
                const isWin = pnl > 0;
                return (
                  <tr
                    key={t.id}
                    className="transition-colors hover:bg-white/[0.02]"
                    style={{ borderBottom: "1px solid var(--border-subtle)" }}
                  >
                    <td className="px-4 py-2.5">
                      <span
                        className="inline-flex items-center gap-1 font-num text-[11px] font-bold uppercase"
                        style={{ color: t.direction === "buy" ? "var(--accent-teal)" : "var(--accent-rose)" }}
                      >
                        <span className="text-[9px]">{t.direction === "buy" ? "\u25B2" : "\u25BC"}</span>
                        {t.direction}
                      </span>
                    </td>
                    <td className="px-4 py-2.5 text-[11px]" style={{ color: "var(--text-secondary)" }}>
                      {t.strategy}
                    </td>
                    <td className="px-4 py-2.5 text-[11px] font-medium" style={{ color: "var(--text-primary)" }}>
                      {t.symbol}
                    </td>
                    <td className="px-4 py-2.5 text-right font-num text-[11px]" style={{ color: "var(--text-secondary)" }}>
                      {t.entry_price.toFixed(2)}
                    </td>
                    <td className="px-4 py-2.5 text-right font-num text-[11px]" style={{ color: "var(--text-secondary)" }}>
                      {t.exit_price?.toFixed(2) ?? "--"}
                    </td>
                    <td className="px-4 py-2.5 text-right">
                      <span
                        className="font-num text-[11px] font-bold"
                        style={{ color: isWin ? "var(--accent-teal)" : pnl < 0 ? "var(--accent-rose)" : "var(--text-muted)" }}
                      >
                        {formatPnl(pnl)}
                      </span>
                    </td>
                    <td className="px-4 py-2.5 text-right text-[11px]" style={{ color: "var(--text-muted)" }}>
                      {formatTime(t.opened_at)}
                    </td>
                  </tr>
                );
              })
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
