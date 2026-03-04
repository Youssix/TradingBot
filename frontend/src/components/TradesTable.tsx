import { useState, useMemo } from "react";
import type { Trade } from "../api";

interface TradesTableProps {
  closedTrades: Trade[];
  openTrades: Trade[];
}

type SortKey = "opened_at" | "pnl" | "strategy" | "symbol" | "direction";
type SortDir = "asc" | "desc";

function formatPrice(val: number | null): string {
  if (val === null || val === undefined) return "--";
  return val.toFixed(5);
}

function formatPnl(val: number | null): string {
  if (val === null || val === undefined) return "--";
  return val >= 0 ? `+$${val.toFixed(2)}` : `-$${Math.abs(val).toFixed(2)}`;
}

function formatDate(val: string | null): string {
  if (!val) return "--";
  const d = new Date(val);
  return d.toLocaleString("en-US", {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

export default function TradesTable({
  closedTrades,
  openTrades,
}: TradesTableProps) {
  const [tab, setTab] = useState<"closed" | "open">("closed");
  const [sortKey, setSortKey] = useState<SortKey>("opened_at");
  const [sortDir, setSortDir] = useState<SortDir>("desc");

  const trades = tab === "closed" ? closedTrades : openTrades;

  const sorted = useMemo(() => {
    return [...trades].sort((a, b) => {
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
  }, [trades, sortKey, sortDir]);

  function handleSort(key: SortKey) {
    if (sortKey === key) {
      setSortDir((d) => (d === "asc" ? "desc" : "asc"));
    } else {
      setSortKey(key);
      setSortDir("desc");
    }
  }

  function sortIcon(key: SortKey) {
    if (sortKey !== key) return "";
    return sortDir === "asc" ? " \u25B2" : " \u25BC";
  }

  return (
    <div className="rounded-xl border border-gray-700 bg-gray-800 p-4">
      <div className="mb-4 flex items-center gap-4">
        <h3 className="text-sm font-semibold tracking-wider text-gray-400 uppercase">
          Trades
        </h3>
        <div className="flex rounded-lg bg-gray-900 p-0.5">
          <button
            onClick={() => setTab("closed")}
            className={`rounded-md px-4 py-1.5 text-xs font-medium transition ${
              tab === "closed"
                ? "bg-gray-700 text-white"
                : "text-gray-400 hover:text-white"
            }`}
          >
            Closed ({closedTrades.length})
          </button>
          <button
            onClick={() => setTab("open")}
            className={`rounded-md px-4 py-1.5 text-xs font-medium transition ${
              tab === "open"
                ? "bg-gray-700 text-white"
                : "text-gray-400 hover:text-white"
            }`}
          >
            Open ({openTrades.length})
          </button>
        </div>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-left text-sm">
          <thead>
            <tr className="border-b border-gray-700 text-xs text-gray-500 uppercase">
              <th
                className="cursor-pointer px-3 py-2 hover:text-gray-300"
                onClick={() => handleSort("direction")}
              >
                Dir{sortIcon("direction")}
              </th>
              <th
                className="cursor-pointer px-3 py-2 hover:text-gray-300"
                onClick={() => handleSort("strategy")}
              >
                Strategy{sortIcon("strategy")}
              </th>
              <th
                className="cursor-pointer px-3 py-2 hover:text-gray-300"
                onClick={() => handleSort("symbol")}
              >
                Symbol{sortIcon("symbol")}
              </th>
              <th className="px-3 py-2">Entry</th>
              <th className="px-3 py-2">Exit</th>
              <th className="px-3 py-2">SL</th>
              <th className="px-3 py-2">TP</th>
              <th className="px-3 py-2">Lots</th>
              <th
                className="cursor-pointer px-3 py-2 hover:text-gray-300"
                onClick={() => handleSort("pnl")}
              >
                P&L{sortIcon("pnl")}
              </th>
              <th
                className="cursor-pointer px-3 py-2 hover:text-gray-300"
                onClick={() => handleSort("opened_at")}
              >
                Time{sortIcon("opened_at")}
              </th>
            </tr>
          </thead>
          <tbody>
            {sorted.length === 0 ? (
              <tr>
                <td colSpan={10} className="px-3 py-8 text-center text-gray-500">
                  No trades to display
                </td>
              </tr>
            ) : (
              sorted.map((t) => (
                <tr
                  key={t.id}
                  className="border-b border-gray-700/50 transition hover:bg-gray-700/30"
                >
                  <td className="px-3 py-2.5">
                    <span
                      className={`inline-flex items-center gap-1 text-xs font-semibold ${
                        t.direction === "buy"
                          ? "text-emerald-400"
                          : "text-red-400"
                      }`}
                    >
                      {t.direction === "buy" ? "\u25B2" : "\u25BC"}{" "}
                      {t.direction.toUpperCase()}
                    </span>
                  </td>
                  <td className="px-3 py-2.5 text-gray-300">{t.strategy}</td>
                  <td className="px-3 py-2.5 font-medium text-white">
                    {t.symbol}
                  </td>
                  <td className="px-3 py-2.5 font-mono text-gray-300">
                    {formatPrice(t.entry_price)}
                  </td>
                  <td className="px-3 py-2.5 font-mono text-gray-300">
                    {formatPrice(t.exit_price)}
                  </td>
                  <td className="px-3 py-2.5 font-mono text-gray-500">
                    {formatPrice(t.sl)}
                  </td>
                  <td className="px-3 py-2.5 font-mono text-gray-500">
                    {formatPrice(t.tp)}
                  </td>
                  <td className="px-3 py-2.5 text-gray-300">
                    {t.lot_size}
                  </td>
                  <td className="px-3 py-2.5">
                    <span
                      className={`font-semibold ${
                        (t.pnl ?? 0) >= 0
                          ? "text-emerald-400"
                          : "text-red-400"
                      }`}
                    >
                      {formatPnl(t.pnl)}
                    </span>
                  </td>
                  <td className="px-3 py-2.5 text-gray-400">
                    {formatDate(t.opened_at)}
                    {t.closed_at && (
                      <span className="ml-1 text-gray-600">
                        - {formatDate(t.closed_at)}
                      </span>
                    )}
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
